"""Evaluate/select checkpoints with TTA; calibrate only using held-out validation IDs."""
import argparse
from pathlib import Path
from panda.data import read_metadata, make_loader
from panda.engine import choose_device, load_model, infer, save_json
from panda.metrics import report, fit_thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output-dir', default='results/evaluation')
    parser.add_argument('--tta', type=int, default=8, choices=[1, 4, 8])
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--external-test', action='store_true', help='Evaluate all CSV rows as an independent labeled test set')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    if args.external_test and args.calibrate:
        parser.error('Never fit thresholds on the test set; calibrate on validation separately')
    if args.external_test and len(args.checkpoints) != 1:
        parser.error('Select one checkpoint on validation before evaluating the external test set')
    frame = read_metadata(args.csv)
    device = choose_device(args.device)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty evaluation output directory to keep each result separate')
    output.mkdir(parents=True, exist_ok=True)
    summaries, expected_ids = [], None
    for i, path in enumerate(args.checkpoints):
        model, ckpt = load_model(path, device)
        if args.external_test:
            subset = frame.copy()
            if set(subset.image_id) & (set(ckpt['train_ids']) | set(ckpt['valid_ids'])):
                raise ValueError('External test IDs overlap training or validation data')
        else:
            subset = frame[frame.image_id.isin(ckpt['valid_ids'])].copy()
            if set(subset.image_id) != set(ckpt['valid_ids']):
                raise ValueError('CSV is missing checkpoint validation IDs')
        if subset.empty:
            raise ValueError('No evaluation samples')
        ids = set(subset.image_id)
        if expected_ids is not None and expected_ids != ids:
            raise ValueError('Checkpoint selection requires the same validation samples')
        expected_ids = ids
        model.config.cache_dir = args.cache_dir
        model.config.require_patch_labels = False
        inputs = subset[['image_id', 'data_provider']]
        predictions = infer(model, make_loader(inputs, model.config), device, args.tta)
        subset = subset.drop(columns=['score', 'prediction_reg', 'classification'] + [f'prob_{j}' for j in range(6)], errors='ignore')
        result = subset.merge(predictions, on='image_id', validate='one_to_one')
        metrics = report(result)
        prefix = output / f'checkpoint_{i}'
        result.to_csv(prefix.with_suffix('.csv'), index=False)
        if args.calibrate:
            thresholds = fit_thresholds(result.score.to_numpy(), result.isup_grade.to_numpy())
            metrics['calibrated_on_this_validation_set'] = report(result, thresholds)
            save_json({'thresholds': thresholds, 'fitted_on': 'validation', 'checkpoint': str(path),
                       'image_ids': result.image_id.tolist()}, output / f'thresholds_{i}.json')
        save_json(metrics, prefix.with_suffix('.json'))
        summaries.append({'checkpoint': str(path), 'qwk': metrics['qwk'], 'samples': len(result)})
        del model
    if args.external_test:
        save_json({'split': 'external_test', 'evaluation': summaries[0]}, output / 'evaluation.json')
        print(summaries[0])
    else:
        selected = max(summaries, key=lambda row: float('-inf') if row['qwk'] is None else row['qwk'])
        save_json({'checkpoints': summaries, 'selected_by_uncalibrated_qwk': selected}, output / 'selection.json')
        print(selected)


if __name__ == '__main__':
    main()
