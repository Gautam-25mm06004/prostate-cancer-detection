"""Shared device, checkpoint, validation, and test-time augmentation helpers."""
from dataclasses import asdict
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from .config import Config
from .models import PandaModel, compute_loss


def choose_device(requested='auto'):
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; select a GPU runtime or use --device cpu')
    return torch.device('cuda' if requested == 'auto' and torch.cuda.is_available() else ('cpu' if requested == 'auto' else requested))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_checkpoint(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.pt')
    torch.save(value, temporary)
    temporary.replace(path)


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('format_version') != 1:
        raise ValueError('Expected a checkpoint trained by this project; legacy weights need a separate conversion')
    config = Config(**checkpoint['config']).validate()
    model = PandaModel(config, pretrained=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval()
    return model, checkpoint


def tta_output(model, tiles, views=1):
    # Sequential TTA avoids multiplying accelerator memory by eight.
    result = {}
    for view in range(views):
        transformed = torch.rot90(tiles, view % 4, (-2, -1))
        if view >= 4:
            transformed = transformed.flip(-1)
        output = model(transformed)
        for key, tensor in output.items():
            value = tensor.float().softmax(1) if key == 'logits' else tensor.float()
            result[key] = result.get(key, 0) + value / views
    return result


@torch.inference_mode()
def infer(model, loader, device, views=1):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc='Predict', leave=False):
        gpu = to_device(batch, device)
        with torch.amp.autocast(device.type, enabled=model.config.amp and device.type == 'cuda'):
            output = tta_output(model, gpu['tiles'], views)
        regression = output['regression'].cpu().numpy()
        probs = output['logits'].cpu().numpy() if 'logits' in output else None
        for i, image_id in enumerate(batch['image_id']):
            row = {'image_id': image_id, 'score': float(regression[i]), 'prediction_reg': float(regression[i])}
            if probs is not None:
                row.update({f'prob_{j}': float(probs[i, j]) for j in range(6)})
                row['classification'] = int(probs[i].argmax())
            rows.append(row)
    return pd.DataFrame(rows)


@torch.inference_mode()
def validate_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.amp.autocast(device.type, enabled=model.config.amp and device.type == 'cuda'):
            loss = compute_loss(model(batch['tiles']), batch, model.config, training=False)
        size = len(batch['label'])
        total += float(loss) * size
        count += size
    return total / count


@torch.inference_mode()
def validate(model, loader, device):
    """Compute validation loss and predictions together in one pass."""
    model.eval()
    total, count, rows = 0.0, 0, []
    for batch in tqdm(loader, desc='Validate', leave=False):
        batch = to_device(batch, device)
        with torch.amp.autocast(device.type, enabled=model.config.amp and device.type == 'cuda'):
            output = model(batch['tiles'])
            loss = compute_loss(output, batch, model.config, training=False)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite validation loss')
        total += float(loss) * len(batch['label'])
        count += len(batch['label'])
        regression = output['regression'].float().cpu().numpy()
        probabilities = output['logits'].float().softmax(1).cpu().numpy() if 'logits' in output else None
        for i, image_id in enumerate(batch['image_id']):
            row = {'image_id': image_id, 'score': float(regression[i]), 'prediction_reg': float(regression[i])}
            if probabilities is not None:
                row.update({f'prob_{j}': float(probabilities[i, j]) for j in range(6)})
                row['classification'] = int(probabilities[i].argmax())
            rows.append(row)
    if count == 0:
        raise ValueError('Validation loader is empty')
    return total/count, pd.DataFrame(rows)
