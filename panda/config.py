"""Small, explicit JSON configurations; paths are relative to the project cwd."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Config:
    branch: str = 'drhb'
    backbone: str = 'resnet34'
    csv: str = 'data/train.csv'
    image_dir: str = 'data/train_images'
    mask_dir: str | None = None
    cache_dir: str = 'data/tiles_drhb'
    output_dir: str = 'runs/drhb'
    level: int = 1
    tile_size: int = 224
    num_tiles: int = 49
    cache_tiles: int = 100
    cache_format: str = 'npy'
    jpeg_quality: int = 95
    offset: float = 0.0
    batch_size: int = 1
    tile_batch_size: int = 8
    accumulation_steps: int = 4
    num_workers: int = 0
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0002
    folds: int = 5
    fold: int = 0
    seed: int = 42
    pretrained: bool = True
    amp: bool = True
    checkpoint_encoder: bool = True
    freeze_batchnorm: bool = True
    pseudo_weight: float = 0.7
    require_patch_labels: bool = False
    invert: bool = True
    normalize: bool = True
    scheduler: str = 'cosine'
    optimizer: str = 'adamw'

    def validate(self):
        if self.cache_format not in {'npy', 'jpeg'} or not 1 <= self.jpeg_quality <= 100:
            raise ValueError('Use cache_format npy/jpeg and JPEG quality 1 through 100')
        if self.branch not in {'drhb', 'cateek', 'rguo_mid', 'rguo_mid_eff', 'rguo_high', 'xie29'}:
            raise ValueError(f'Unknown branch: {self.branch}')
        for key in ('tile_size', 'num_tiles', 'cache_tiles', 'batch_size', 'tile_batch_size',
                    'accumulation_steps', 'epochs'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if self.num_tiles > self.cache_tiles:
            raise ValueError('cache_tiles must be at least num_tiles')
        if self.num_workers < 0 or not 0 <= self.offset < 1:
            raise ValueError('num_workers must be >= 0 and offset in [0, 1)')
        if self.folds < 2 or not 0 <= self.fold < self.folds:
            raise ValueError('Use at least two folds and a valid fold index')
        if self.scheduler not in {'cosine', 'onecycle'}:
            raise ValueError('scheduler must be cosine or onecycle')
        if self.optimizer not in {'adamw', 'radam'}:
            raise ValueError('optimizer must be adamw or radam')
        if self.branch in {'drhb', 'cateek', 'xie29'} and int(self.num_tiles**0.5)**2 != self.num_tiles:
            raise ValueError('This branch requires a square number of tiles')
        return self

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n')


def load_config(path):
    return Config(**json.loads(Path(path).read_text())).validate()
