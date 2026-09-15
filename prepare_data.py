"""Prepare slide caches sequentially, with optional learned attention selection."""
import argparse
from contextlib import ExitStack
import heapq
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from panda.config import load_config
from panda.cache import TileWriter, verify_cache
from panda.slides import Slide, grid, darkest_coordinates, find_slide, patch_target


def attention_coordinates(slide, config, selector, device, count, source_level, source_size):
    import torch
    from panda.data import normalize_tiles
    heap, pending, coords = [], [], []
    serial = 0
    def consume():
        nonlocal serial
        with torch.inference_mode():
            x = normalize_tiles(np.stack(pending), selector.config).to(device)
            scores = selector.tile_attention(x).float().cpu().tolist()
        for score, (x, y) in zip(scores, coords):
            entry = (score, -serial, x, y)
            if len(heap) < count:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
            serial += 1
        pending.clear()
        coords.clear()
    for x, y in grid(slide, source_level, source_size, config.offset):
        tile = slide.read(x, y, source_level, source_size)
        if (tile.mean(2) > 240).mean() > 0.95:
            continue
        pending.append(tile)
        coords.append((x, y))
        if len(pending) == config.tile_batch_size:
            consume()
    if pending:
        consume()
    return [(x, y) for _, _, x, y in sorted(heap, reverse=True)]


def prepare_slide(row, config, selector=None, device='cpu', selector_path=None):
    source = find_slide(config.image_dir, row['image_id'])
    destination = Path(config.cache_dir) / row['image_id']
    destination.mkdir(parents=True, exist_ok=True)
    mask_path = find_slide(config.mask_dir, row['image_id'], mask=True) if config.mask_dir else None
    if config.require_patch_labels and mask_path is None:
        raise ValueError('This training profile requires masks; use --inference when preparing prediction images')
    signature = {key: getattr(config, key) for key in ('level', 'tile_size', 'cache_tiles', 'offset', 'cache_format', 'jpeg_quality')}
    signature.update(source=str(source.resolve()), source_size=source.stat().st_size,
                     source_mtime=source.stat().st_mtime_ns, mask=str(mask_path.resolve()) if mask_path else None,
                     mask_mtime=mask_path.stat().st_mtime_ns if mask_path else None,
                     selector=str(Path(selector_path).resolve()) if selector_path else None,
                     selector_mtime=Path(selector_path).stat().st_mtime_ns if selector_path else None)
    meta_path = destination / 'metadata.json'
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get('signature') != signature:
            raise ValueError(f'{destination}: existing cache differs; choose a new cache_dir')
        verify_cache(destination, config)
        return
    with ExitStack() as stack:
        slide = stack.enter_context(Slide(source))
        level = slide.level(config.level)
        mask = stack.enter_context(Slide(mask_path)) if mask_path else None
        if mask and mask.level_dimensions[0] != slide.level_dimensions[0]:
            raise ValueError('Mask/image level-0 dimensions differ')
        if selector:
            low_level = slide.level(selector.config.level)
            low_size = selector.config.tile_size
            coordinates = attention_coordinates(slide, config, selector, device, config.cache_tiles, low_level, low_size)
            # Same physical field at the finer level, then resized to the target tile size.
            read_size = round(low_size * slide.level_downsamples[low_level] / slide.level_downsamples[level])
        else:
            coordinates = darkest_coordinates(slide, level, config.tile_size, config.cache_tiles, config.offset)
            read_size = config.tile_size
        if read_size > 4096:
            raise ValueError('Requested tile region exceeds 4096 pixels; reduce source tile size')
        writer = stack.enter_context(TileWriter(destination, config))
        patch_labels, patch_valid = [], []
        for index in range(config.cache_tiles):
            label, valid = 0.0, 0.0
            if index < len(coordinates):
                x, y = coordinates[index]
                tile = slide.read(x, y, level, read_size)
                if read_size != config.tile_size:
                    tile = np.asarray(Image.fromarray(tile).resize((config.tile_size, config.tile_size), Image.Resampling.LANCZOS))
                if mask:
                    mask_level = mask.level(config.level)
                    if not np.isclose(mask.level_downsamples[mask_level], slide.level_downsamples[level]):
                        raise ValueError('Mask/image pyramid scales differ; use aligned PANDA masks')
                    pixels = mask.read(x, y, mask_level, read_size, background=0)
                    if read_size != config.tile_size:
                        pixels = np.asarray(Image.fromarray(pixels).resize((config.tile_size, config.tile_size), Image.Resampling.NEAREST))
                    label, valid = patch_target(pixels, row['data_provider'])
            else:
                tile = np.full((config.tile_size, config.tile_size, 3), 255, np.uint8)
            writer.write(index, tile)
            patch_labels.append(label)
            patch_valid.append(valid)
        # Close and publish tiles before publishing the completion metadata.
        stack.close()
        metadata = dict(signature, signature=signature, coordinates=coordinates, has_mask=bool(mask),
                        patch_labels=patch_labels, patch_valid=patch_valid)
        partial_meta = destination / 'metadata.partial.json'
        partial_meta.write_text(json.dumps(metadata))
        partial_meta.replace(meta_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/drhb.json')
    parser.add_argument('--csv')
    parser.add_argument('--image-dir')
    parser.add_argument('--mask-dir')
    parser.add_argument('--inference', action='store_true', help='Prepare prediction images without training label masks')
    parser.add_argument('--cache-dir')
    parser.add_argument('--exclude', help='CSV with image_id values to exclude')
    parser.add_argument('--limit', type=int, help='Process only the first N eligible slides for a trial')
    parser.add_argument('--attention-checkpoint', help='Modern RGuo mid-resolution checkpoint')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ('csv', 'image_dir', 'mask_dir', 'cache_dir'):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    if args.inference:
        config.mask_dir = None
        config.require_patch_labels = False
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    excluded = set(pd.read_csv(args.exclude).image_id) if args.exclude else set()
    # These two unreadable slides are explicitly excluded by upstream DrHB.
    excluded |= {'0da0915a236f2fc98b299d6fdefe7b8b', '3790f55cad63053e956fb73027179707'}
    selector, selector_checkpoint, device = None, None, 'cpu'
    if args.attention_checkpoint:
        from panda.engine import load_model, choose_device
        device = choose_device(args.device)
        selector, selector_checkpoint = load_model(args.attention_checkpoint, device)
        if not selector.attention_branch:
            parser.error('--attention-checkpoint must be an RGuo model')
    if config.branch == 'rguo_high' and selector is None:
        parser.error('rguo_high requires --attention-checkpoint to select high-resolution regions')
    destination = Path(config.cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if selector_checkpoint:
        provenance = {'checkpoint': str(Path(args.attention_checkpoint).resolve()),
                      'train_ids': selector_checkpoint['train_ids'], 'valid_ids': selector_checkpoint['valid_ids']}
        provenance_path = destination / 'selector.json'
        if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
            raise ValueError('Cache already belongs to another selector; choose a separate cache_dir')
        provenance_path.write_text(json.dumps(provenance))
    count, seen = 0, set()
    # CSV rows stream too. Only the set of IDs is retained for duplicate detection.
    output_csv = destination / 'prepared.partial.csv'
    with output_csv.open('w', newline='') as output:
        for chunk in pd.read_csv(config.csv, chunksize=256, dtype={'image_id': str}):
            required = {'image_id', 'data_provider'}
            if not required.issubset(chunk):
                raise ValueError(f'CSV requires {required}')
            rows = []
            for row in tqdm(chunk.to_dict('records'), desc=f'Prepared {count}', leave=False):
                if row['image_id'] in excluded:
                    continue
                if args.limit is not None and count >= args.limit:
                    break
                if row['image_id'] in seen:
                    raise ValueError(f'Duplicate image_id: {row["image_id"]}')
                seen.add(row['image_id'])
                prepare_slide(row, config, selector, device, args.attention_checkpoint)
                rows.append(row)
                count += 1
            if rows:
                pd.DataFrame(rows).to_csv(output, index=False, header=output.tell() == 0)
            if args.limit is not None and count >= args.limit:
                break
    if not count:
        raise ValueError('No eligible slides')
    output_csv.replace(destination / 'prepared.csv')
    config.csv = str(destination / 'prepared.csv')
    config.save(destination / 'config.json')
    print(f'Prepared {count} slides. Train with --config {destination / "config.json"}')


if __name__ == '__main__':
    main()
