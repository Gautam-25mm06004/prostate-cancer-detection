"""Contributor-specific heads with current timm backbones and native PyTorch AMP."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import timm


class SqueezeExcite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Conv2d(channels, max(1, channels // 4), 1)
        self.expand = nn.Conv2d(max(1, channels // 4), channels, 1)

    def forward(self, x):
        gate = self.expand(F.relu(self.reduce(x.mean((-2, -1), keepdim=True)))).sigmoid()
        return x * gate


class GeM(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        p = self.p.clamp(0.1, 10)
        return x.clamp(min=1e-6).pow(p).mean((-2, -1)).pow(1 / p)


def mosaic(tiles):
    b, n, c, h, w = tiles.shape
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError('Mosaic requires square tile count')
    return tiles.reshape(b, side, side, c, h, w).permute(0, 3, 1, 4, 2, 5).reshape(b, c, side*h, side*w)


class PandaModel(nn.Module):
    def __init__(self, config, pretrained=None):
        super().__init__()
        self.config = config
        use_pretrained = config.pretrained if pretrained is None else pretrained
        if config.backbone == 'tiny':  # Fast offline integration tests only.
            self.encoder = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU(),
                                         nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU())
            channels = 16
        elif config.branch == 'cateek':
            from .cateek_backbone import ResNeXt
            base = ResNeXt(4, 32, [3, 4, 6, 3], 1000)
            self.encoder = nn.Sequential(*list(base.children())[:-2])
            channels = 2048
            if use_pretrained:
                raise ValueError('CatEek GN/WS ImageNet weights are absent upstream. Set pretrained=false; use --init for a modern checkpoint.')
        else:
            self.encoder = timm.create_model(config.backbone, pretrained=use_pretrained,
                                             num_classes=0, global_pool='')
            channels = self.encoder.num_features
        self.attention_branch = config.branch.startswith('rguo')
        if self.attention_branch:
            self.attention = nn.Sequential(nn.Linear(channels, 512), nn.Tanh(), nn.Dropout(0.25), nn.Linear(512, 1))
            self.patch = nn.Linear(channels, 1)
            self.regression = nn.Sequential(nn.Dropout(0.25), nn.Linear(channels * 2, 1))
            self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(channels * 2, 6))
        elif config.branch == 'xie29':
            self.gem = GeM()
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 1))
        else:
            self.se = SqueezeExcite(channels)
            outputs = 7 if config.branch == 'drhb' else 1
            self.head = nn.Sequential(nn.Linear(2*channels, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, outputs))

    def train(self, mode=True):
        super().train(mode)
        # Small slide batches and chunked encoding need stable BN statistics.
        # Also prevents activation recomputation from updating BN twice.
        if self.config.freeze_batchnorm:
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def encode(self, flat_tiles):
        outputs = []
        for chunk in flat_tiles.split(self.config.tile_batch_size):
            if self.training and self.config.checkpoint_encoder:
                outputs.append(checkpoint(self.encoder, chunk, use_reentrant=False))
            else:
                outputs.append(self.encoder(chunk))
        return torch.cat(outputs)

    def tile_attention(self, tiles):
        if not self.attention_branch:
            raise ValueError('Tile selection requires an RGuo attention checkpoint')
        features = self.encode(tiles).mean((-2, -1))
        return self.attention(features).squeeze(-1)

    def forward(self, tiles):
        b, n, c, h, w = tiles.shape
        if self.config.branch == 'xie29':
            features = self.encode(mosaic(tiles))
            return {'regression': self.head(self.gem(features)).squeeze(-1)}
        features = self.encode(tiles.reshape(b*n, c, h, w))
        if self.attention_branch:
            features = features.mean((-2, -1)).reshape(b, n, -1)
            attention = self.attention(features).squeeze(-1)
            pooled = torch.cat(((features * attention.softmax(1)[..., None]).sum(1), features.max(1).values), 1)
            return {'regression': 7*self.regression(pooled).squeeze(-1).sigmoid()-1,
                    'logits': self.classifier(pooled), 'patch': self.patch(features).squeeze(-1),
                    'attention': attention}
        _, channels, fh, fw = features.shape
        # Match the original feature reshaping before squeeze-and-excitation.
        features = features.reshape(b, n, channels, fh, fw).permute(0, 2, 1, 3, 4).contiguous()
        side = math.isqrt(n)
        features = features.reshape(b, channels, fh*n//side, fw*side)
        features = self.se(features)
        pooled = torch.cat((features.amax((-2, -1)), features.mean((-2, -1))), 1)
        out = self.head(pooled)
        result = {'regression': 7*out[:, -1].sigmoid()-1}
        if self.config.branch == 'drhb':
            result['logits'] = out[:, :6]
        return result


def compute_loss(output, batch, config, training=True):
    target = batch['label'].float()
    prediction = output['regression'].float()
    if config.branch == 'cateek':
        return F.smooth_l1_loss(prediction, target)
    if config.branch in {'rguo_mid', 'rguo_high'} and training:
        target = config.pseudo_weight*target + (1-config.pseudo_weight)*batch['pseudo'].float()
    if config.branch in {'xie29', 'rguo_mid_eff'}:
        mse = (prediction-target).square()
        huber = F.smooth_l1_loss(prediction, target, reduction='none')
        if config.branch.startswith('rguo'):
            huber = 2*huber
        loss = torch.where(batch['provider'].bool(), huber, mse).mean()
    else:
        loss = F.mse_loss(prediction, target)
    if 'logits' in output:
        if config.branch.startswith('rguo') and training:
            probs = config.pseudo_weight*F.one_hot(batch['label'], 6) + (1-config.pseudo_weight)*batch['pseudo_probs']
            loss = loss - (probs * output['logits'].float().log_softmax(1)).sum(1).mean()
        else:
            loss = loss + F.cross_entropy(output['logits'].float(), batch['label'])
    if 'patch' in output:
        loss = loss + F.binary_cross_entropy_with_logits(output['patch'].float(), batch['patch_labels'].float(),
                                                         weight=batch['patch_valid'].float())
    return loss
