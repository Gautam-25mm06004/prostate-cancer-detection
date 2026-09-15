"""Load small metadata tables and lazily decode one slide's cached tiles."""
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from .cache import read_tiles


def read_metadata(path, labeled=True):
    frame = pd.read_csv(path, dtype={'image_id': str})
    if 'isup_grade' not in frame and 'isup_grade_x' in frame:
        frame = frame.rename(columns={'isup_grade_x': 'isup_grade'})
    required = {'image_id', 'data_provider'} | ({'isup_grade'} if labeled else set())
    if missing := required - set(frame.columns):
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    if frame.empty or frame.image_id.isna().any() or frame.image_id.duplicated().any():
        raise ValueError('Metadata must have nonempty, unique image_id values')
    if not frame.data_provider.isin(['radboud', 'karolinska']).all():
        raise ValueError('data_provider must be radboud or karolinska')
    if labeled and not frame.isup_grade.isin(range(6)).all():
        raise ValueError('isup_grade must be an integer from 0 through 5')
    return frame


def assign_folds(frame, folds, seed):
    frame = frame.copy()
    if 'fold' not in frame:
        if 'split' in frame:
            frame['fold'] = frame['split']
        else:
            strata = frame.data_provider + '_' + frame.isup_grade.astype(str)
            if strata.value_counts().min() < folds:
                strata = frame.isup_grade
            if strata.value_counts().min() < folds:
                raise ValueError('Too few samples per grade for these folds; reduce folds or use a larger subset')
            frame['fold'] = -1
            if 'patient_id' in frame:
                splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
                splits = splitter.split(frame, strata, frame.patient_id)
            else:
                splits = StratifiedKFold(folds, shuffle=True, random_state=seed).split(frame, strata)
            for fold, (_, valid) in enumerate(splits):
                frame.iloc[valid, frame.columns.get_loc('fold')] = fold
    if not frame.fold.isin(range(folds)).all():
        raise ValueError('Invalid fold values; expected integers in [0, folds)')
    if 'patient_id' in frame and frame.groupby('patient_id').fold.nunique().max() > 1:
        raise ValueError('Patient leakage: the same patient occurs in multiple folds')
    return frame


def normalize_tiles(tiles, config):
    tensor = torch.from_numpy(np.array(tiles, copy=True)).permute(0, 3, 1, 2).float() / 255
    if config.invert:
        tensor = 1 - tensor
    if config.normalize:
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        tensor = (tensor - mean) / std
    return tensor


class TileDataset(Dataset):
    def __init__(self, frame, config, training=False):
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.training = training

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        folder = Path(self.config.cache_dir) / row.image_id
        meta = json.loads((folder / 'metadata.json').read_text())
        c = self.config
        for key in ('level', 'tile_size', 'offset'):
            if meta[key] != getattr(c, key):
                raise ValueError(f'{row.image_id}: cache {key} differs from config; prepare a separate cache')
        if meta.get('cache_format', 'npy') != c.cache_format:
            raise ValueError('Cache format differs from the model configuration')
        if c.cache_format == 'jpeg' and meta.get('jpeg_quality') != c.jpeg_quality:
            raise ValueError('Cache JPEG quality differs from the model configuration')
        tiles = read_tiles(folder, meta, c.num_tiles)
        x = normalize_tiles(tiles[:c.num_tiles], c)
        if self.training:
            x = torch.rot90(x, random.randrange(4), (-2, -1))
            if random.random() < 0.5:
                x = x.flip(-1)
            if random.random() < 0.5:
                x = x.flip(-2)
        labels = np.array(meta['patch_labels'][:c.num_tiles], dtype=np.float32)
        valid = np.array(meta['patch_valid'][:c.num_tiles], dtype=np.float32)
        if c.require_patch_labels and not meta['has_mask']:
            raise ValueError(f'{row.image_id}: this profile requires a label mask')
        grade = int(row.get('isup_grade', 0))
        probs = np.array([row.get(f'prob_{i}', float(i == grade)) for i in range(6)], dtype=np.float32)
        if not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(), 1, atol=0.02):
            raise ValueError(f'{row.image_id}: invalid pseudo-label probabilities')
        pseudo = float(row.get('prediction_reg', grade))
        if not np.isfinite(pseudo):
            raise ValueError(f'{row.image_id}: invalid pseudo regression label')
        return {'tiles': x, 'label': grade, 'provider': int(row.data_provider == 'radboud'),
                'patch_labels': labels, 'patch_valid': valid, 'pseudo': pseudo,
                'pseudo_probs': probs / probs.sum(), 'image_id': row.image_id}


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(frame, config, training=False, ordered=False):
    generator = torch.Generator().manual_seed(config.seed)
    options = {'prefetch_factor': 1} if config.num_workers else {}
    return DataLoader(TileDataset(frame, config, training), batch_size=config.batch_size,
                      shuffle=training and not ordered, num_workers=config.num_workers, drop_last=False,
                      pin_memory=torch.cuda.is_available(), generator=generator,
                      worker_init_fn=seed_worker, **options)
