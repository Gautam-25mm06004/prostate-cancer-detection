"""Prepare all eligible PANDA slides with one downloaded slide at a time."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import pandas as pd
from download_subset import download_file
from prepare_data import prepare_slide
from panda.cache import verify_cache
from panda.config import load_config
from panda.data import read_metadata
from panda.engine import save_json


def prepare_stream(config, scratch_dir, cli='kaggle', max_slides=None, reserve_gb=5, fetch=download_file):
    """Resume verified caches; publish the training manifest only after every slide is ready."""
    if config.branch == 'rguo_high':
        raise ValueError('Use prepare_data.py with a fold-matched attention selector for rguo_high')
    if max_slides is not None and max_slides < 1:
        raise ValueError('max_slides must be positive')
    if reserve_gb < 0:
        raise ValueError('reserve_gb cannot be negative')
    frame = read_metadata(config.csv)
    excluded = {'0da0915a236f2fc98b299d6fdefe7b8b', '3790f55cad63053e956fb73027179707'}
    frame = frame[~frame.image_id.isin(excluded)].reset_index(drop=True)
    if frame.empty:
        raise ValueError('No eligible slides')
    if 'fold' not in frame:
        raise ValueError('Run make_folds.py first to fix the full-data split before preparation')
    cache = Path(config.cache_dir).resolve()
    scratch = Path(scratch_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    signature = {'competition': 'prostate-cancer-grade-assessment',
                 'metadata_sha256': hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
                 'preparation': {k: asdict(config)[k] for k in
                     ('branch', 'level', 'tile_size', 'cache_tiles', 'offset', 'cache_format', 'jpeg_quality', 'require_patch_labels')}}
    manifest = cache/'stream_manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise ValueError('Stream inputs changed; use a separate cache directory')
    else:
        if any(cache.iterdir()):
            raise ValueError('Start streaming in an empty cache directory')
        save_json(signature, manifest)
    ready, created, total_bytes = [], 0, 0
    for row in frame.to_dict('records'):
        image_id = row['image_id']
        if Path(image_id).name != image_id or '\\' in image_id or image_id in {'.', '..'}:
            raise ValueError('Invalid image_id')
        folder = cache/image_id
        if (folder/'metadata.json').exists():
            verify_cache(folder, config)
        else:
            if max_slides is not None and created >= max_slides:
                break
            if shutil.disk_usage(scratch).free < reserve_gb * 10**9:
                raise OSError('Scratch disk is below the requested free-space reserve')
            # TemporaryDirectory owns this new folder only, never the user's original slides.
            with tempfile.TemporaryDirectory(prefix='panda-slide-', dir=scratch) as name:
                temporary = Path(name).resolve()
                if not temporary.is_relative_to(scratch) or temporary == scratch:
                    raise ValueError('Temporary download folder escaped scratch directory')
                fetch(cli, image_id, 'train_images', temporary)
                if config.require_patch_labels:
                    fetch(cli, image_id, 'train_label_masks', temporary)
                local = replace(config, image_dir=str(temporary/'train_images'),
                                mask_dir=str(temporary/'train_label_masks') if config.require_patch_labels else '',
                                cache_dir=str(cache))
                prepare_slide(row, local)
                verify_cache(folder, config)
            created += 1
        ready.append(row)
        total_bytes += sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        print(f'Prepared {len(ready)}/{len(frame)} slides; cache bytes {total_bytes}', flush=True)
    partial = cache/'prepared.partial.csv'
    pd.DataFrame(ready, columns=frame.columns).to_csv(partial, index=False)
    complete = len(ready) == len(frame)
    save_json({'complete': complete, 'eligible_slides': len(frame), 'verified_slides': len(ready),
               'new_slides_this_run': created, 'verified_slide_cache_bytes': total_bytes}, cache/'cache_stats.json')
    if complete:
        partial.replace(cache/'prepared.csv')
        replace(config, csv=str(cache/'prepared.csv'), cache_dir=str(cache)).save(cache/'config.json')
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rguo_mid_eff_colab.json')
    parser.add_argument('--csv')
    parser.add_argument('--cache-dir')
    parser.add_argument('--scratch-dir', default='/content/panda-scratch')
    parser.add_argument('--max-slides', type=int, help='Stop after this many new slides; rerun to continue the full dataset')
    parser.add_argument('--reserve-gb', type=float, default=5, help='Minimum free scratch space before starting another slide')
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ('csv', 'cache_dir'):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    cli = shutil.which('kaggle')
    if cli is None:
        raise RuntimeError('Install and authenticate the Kaggle CLI first')
    complete = prepare_stream(config, args.scratch_dir, cli, args.max_slides, args.reserve_gb)
    print('Full cache ready. Train with its config.json.' if complete else 'Preparation paused. Rerun the same command to continue.')


if __name__ == '__main__':
    main()
