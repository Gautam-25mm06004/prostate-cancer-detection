"""Predict raw scores and ISUP grades from one or several modern checkpoints."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from panda.data import read_metadata, make_loader
from panda.engine import choose_device, load_model, infer
from panda.metrics import grades, DEFAULT_THRESHOLDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output', default='predictions/predictions.csv')
    parser.add_argument('--thresholds', help='JSON saved by evaluate.py --calibrate')
    parser.add_argument('--tta', type=int, choices=[1, 4, 8], default=8)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    frame = read_metadata(args.csv, labeled=False)
    device = choose_device(args.device)
    scores = np.zeros(len(frame), dtype=np.float64)
    for path in args.checkpoints:
        model, _ = load_model(path, device)
        model.config.cache_dir = args.cache_dir
        model.config.batch_size = args.batch_size
        model.config.require_patch_labels = False
        # Inference must not depend on old training labels or pseudo labels.
        inputs = frame[['image_id', 'data_provider']]
        predictions = infer(model, make_loader(inputs, model.config), device, args.tta)
        scores += predictions.set_index('image_id').loc[frame.image_id, 'score'].to_numpy() / len(args.checkpoints)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    thresholds = json.loads(Path(args.thresholds).read_text())['thresholds'] if args.thresholds else DEFAULT_THRESHOLDS
    result = frame[['image_id']].copy()
    result['score'] = scores
    result['isup_grade'] = grades(scores, thresholds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    result[['image_id', 'isup_grade']].to_csv(output.with_name(output.stem + '_submission.csv'), index=False)
    print(f'Saved {len(result)} predictions to {output}')


if __name__ == '__main__':
    main()
