"""Small synthetic tests verify behavior, not diagnostic accuracy."""
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from PIL import Image
import torch
from panda.config import Config
from panda.data import assign_folds, TileDataset
from panda.models import PandaModel, compute_loss
from panda.metrics import grades
from panda.slides import Slide, darkest_coordinates, patch_target
from prepare_data import prepare_slide
from merge_prediction import merge_predictions

torch.set_num_threads(1)


def test_bounded_top_tiles():
    class FakeSlide:
        level_dimensions = [(10000, 128)]
        level_downsamples = [1.0]
        calls = 0
        def read(self, x, y, level, size):
            assert size == 32
            self.calls += 1
            return np.full((size, size, 3), (max(x, 0)//32) % 255, np.uint8)
    slide = FakeSlide()
    coordinates = darkest_coordinates(slide, 0, 32, 4)
    assert len(coordinates) == 4
    assert slide.calls > 1000


def test_padding_and_mask_mapping(tmp_path):
    path = tmp_path / 'small.png'
    Image.fromarray(np.zeros((10, 10, 3), np.uint8)).save(path)
    with Slide(path) as slide:
        image = slide.read(-4, -4, 0, 16)
        assert image[0, 0, 0] == 255
        assert image[5, 5, 0] == 0
    mask = np.full((16, 16, 3), 3, np.uint8)
    assert patch_target(mask, 'radboud') == (1.0, 1.0)
    assert patch_target(mask, 'karolinska') == (0.0, 1.0)


def test_ensemble_aligns_ids_and_boundaries(tmp_path):
    a, b = tmp_path/'a.csv', tmp_path/'b.csv'
    pd.DataFrame({'image_id': ['x', 'y'], 'score': [0, 3]}).to_csv(a, index=False)
    pd.DataFrame({'image_id': ['y', 'x'], 'score': [5, 2]}).to_csv(b, index=False)
    out = merge_predictions([a, b], [1, 1])
    assert out.score.tolist() == [1, 4]
    assert grades([0.5, 1.5, 4.5, 4.5001]).tolist() == [0, 1, 4, 5]
    pd.DataFrame({'image_id': ['z'], 'score': [0]}).to_csv(b, index=False)
    with pytest.raises(ValueError, match='IDs differ'):
        merge_predictions([a, b], [1, 1])


def test_patient_leakage_rejected():
    frame = pd.DataFrame({'image_id': ['a', 'b'], 'isup_grade': [1, 2],
                          'data_provider': ['radboud']*2, 'patient_id': ['patient']*2, 'fold': [0, 1]})
    with pytest.raises(ValueError, match='Patient leakage'):
        assign_folds(frame, 2, 1)


@pytest.mark.parametrize('branch', ['drhb', 'cateek', 'rguo_mid', 'rguo_mid_eff', 'rguo_high', 'xie29'])
def test_all_heads_backward(branch):
    config = Config(branch=branch, backbone='tiny', pretrained=False, num_tiles=4,
                    cache_tiles=4, tile_size=32, tile_batch_size=2)
    model = PandaModel(config).train()
    batch = {'tiles': torch.rand(2, 4, 3, 32, 32), 'label': torch.tensor([1, 4]),
             'provider': torch.tensor([0, 1]), 'pseudo': torch.tensor([1., 4.]),
             'pseudo_probs': torch.nn.functional.one_hot(torch.tensor([1, 4]), 6).float(),
             'patch_labels': torch.ones(2, 4), 'patch_valid': torch.ones(2, 4)}
    output = model(batch['tiles'])
    loss = compute_loss(output, batch, config)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


@pytest.mark.parametrize('branch,backbone', [('drhb', 'resnet34'), ('cateek', 'resnext50_gn_ws'),
                                         ('rguo_mid', 'seresnext50_32x4d'), ('rguo_mid_eff', 'tf_efficientnet_b0'),
                                         ('xie29', 'tf_efficientnet_b3.ns_jft_in1k')])
def test_real_backbone_forward(branch, backbone):
    config = Config(branch=branch, backbone=backbone, pretrained=False, num_tiles=4,
                    cache_tiles=4, tile_size=64, tile_batch_size=2)
    model = PandaModel(config).eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 4, 3, 64, 64))
    assert output['regression'].shape == (1,)
    assert output['regression'].isfinite().all()


def test_training_resume_evaluation_prediction(tmp_path, monkeypatch):
    import train
    import evaluate
    import predict
    images = tmp_path/'images'
    images.mkdir()
    config = Config(backbone='tiny', pretrained=False, csv=str(tmp_path/'train.csv'),
                    image_dir=str(images), cache_dir=str(tmp_path/'cache'), output_dir=str(tmp_path/'run'),
                    level=0, tile_size=32, num_tiles=4, cache_tiles=4, tile_batch_size=2,
                    batch_size=2, accumulation_steps=2, epochs=2, folds=2, normalize=False, invert=False)
    rows = []
    rng = np.random.default_rng(42)
    for i in range(14):
        row = {'image_id': f'slide_{i}', 'data_provider': 'radboud' if i%2 else 'karolinska',
               'isup_grade': i%6, 'fold': i%2}
        Image.fromarray(rng.integers(50, 255, (64, 64, 3), dtype=np.uint8)).save(images/f'slide_{i}.png')
        prepare_slide(row, config)
        rows.append(row)
    pd.DataFrame(rows).to_csv(config.csv, index=False)
    cfg = tmp_path/'config.json'
    config.save(cfg)
    dataset = TileDataset(pd.DataFrame(rows), config)
    assert dataset[0]['tiles'].shape == (4, 3, 32, 32)
    # Simulate an interruption after the first epoch checkpoint is saved.
    actual_save = train.save_checkpoint
    def interrupt_after_best(value, path):
        actual_save(value, path)
        if Path(path).name == 'best.pt' and value['epoch'] == 0:
            raise KeyboardInterrupt
    monkeypatch.setattr(train, 'save_checkpoint', interrupt_after_best)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(cfg), '--device', 'cpu'])
    with pytest.raises(KeyboardInterrupt):
        train.main()
    monkeypatch.setattr(train, 'save_checkpoint', actual_save)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(cfg), '--device', 'cpu', '--resume', str(tmp_path/'run/last.pt')])
    train.main()
    ckpt = torch.load(tmp_path/'run/last.pt', weights_only=True)
    assert ckpt['epoch'] == 1 and len(ckpt['history']) == 2
    assert not set(ckpt['train_ids']) & set(ckpt['valid_ids'])
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', '--checkpoints', str(tmp_path/'run/best.pt'), '--csv', config.csv,
                                    '--cache-dir', config.cache_dir, '--output-dir', str(tmp_path/'evaluation'), '--tta', '4', '--calibrate'])
    evaluate.main()
    assert (tmp_path/'evaluation/thresholds_0.json').exists()
    monkeypatch.setattr(sys, 'argv', ['predict.py', '--checkpoints', str(tmp_path/'run/best.pt'), '--csv', config.csv,
                                    '--cache-dir', config.cache_dir, '--output', str(tmp_path/'predictions.csv'), '--tta', '8'])
    predict.main()
    result = pd.read_csv(tmp_path/'predictions.csv')
    assert len(result) == 14 and result.isup_grade.between(0, 5).all()


def test_attention_cache_and_cache_mismatch(tmp_path):
    from prepare_data import attention_coordinates
    low = Config(branch='rguo_mid', backbone='tiny', pretrained=False, tile_size=32, level=0,
                 num_tiles=4, cache_tiles=4, tile_batch_size=2)
    model = PandaModel(low).eval()
    path = tmp_path/'tile.png'
    Image.fromarray(np.full((64, 64, 3), 100, np.uint8)).save(path)
    with Slide(path) as slide:
        coords = attention_coordinates(slide, low, model, torch.device('cpu'), 2, 0, 32)
    assert len(coords) == 2
    config = replace(low, image_dir=str(tmp_path), cache_dir=str(tmp_path/'cache'))
    row = {'image_id': 'tile', 'data_provider': 'radboud'}
    prepare_slide(row, config)
    prepare_slide(row, config)  # Resume reuses a matching complete cache.
    with pytest.raises(ValueError, match='cache differs'):
        prepare_slide(row, replace(config, offset=0.5))


def test_real_tiled_tiff_and_high_resolution(tmp_path, monkeypatch):
    tifffile = pytest.importorskip('tifffile')
    data = np.full((256, 256, 3), 150, np.uint8)
    data[64:128, 64:128] = 50
    path = tmp_path/'pyramid.tiff'
    # Generic tiled TIFF pyramid: explicit reduced-image IFD at half resolution.
    with tifffile.TiffWriter(path) as writer:
        writer.write(data, tile=(64, 64), photometric='rgb', metadata=None)
        writer.write(data[::2, ::2], tile=(64, 64), photometric='rgb', subfiletype=1, metadata=None)
    with Slide(path) as slide:
        assert len(slide.level_dimensions) == 2
        assert slide.level_downsamples[1] == 2
        assert slide.read(64, 64, 1, 32).shape == (32, 32, 3)
    low = Config(branch='rguo_mid', backbone='tiny', pretrained=False, tile_size=32, level=1,
                 num_tiles=4, cache_tiles=4, tile_batch_size=2)
    selector = PandaModel(low).eval()
    # A controlled scorer chooses the darkest region; an untrained model need not.
    monkeypatch.setattr(selector, 'tile_attention', lambda tiles: tiles.mean((1, 2, 3)))
    high = replace(low, branch='rguo_high', tile_size=64, level=0, image_dir=str(tmp_path), cache_dir=str(tmp_path/'high'))
    prepare_slide({'image_id': 'pyramid', 'data_provider': 'radboud'}, high, selector, torch.device('cpu'))
    array = np.load(tmp_path/'high/pyramid/tiles.npy', mmap_mode='r')
    assert array.shape == (4, 64, 64, 3)
    assert array.min() < 150


def test_high_resolution_initialization(tmp_path, monkeypatch):
    import train
    config = Config(branch='rguo_high', backbone='tiny', pretrained=False,
                    csv=str(tmp_path/'labels.csv'), image_dir=str(tmp_path),
                    cache_dir=str(tmp_path/'cache'), output_dir=str(tmp_path/'run'),
                    level=0, tile_size=32, num_tiles=4, cache_tiles=4, tile_batch_size=2,
                    batch_size=2, epochs=1, folds=2, require_patch_labels=False)
    rows = []
    for i in range(4):
        row = {'image_id': f'slide_{i}', 'isup_grade': i, 'data_provider': 'radboud', 'fold': i%2}
        Image.fromarray(np.full((64, 64, 3), 80+i*20, np.uint8)).save(tmp_path/f'slide_{i}.png')
        # Cache content can be prepared without a learned selector for this weight-transfer test.
        prepare_slide(row, replace(config, branch='rguo_mid_eff'))
        rows.append(row)
    pd.DataFrame(rows).to_csv(config.csv, index=False)
    train_ids = ['slide_1', 'slide_3']
    (Path(config.cache_dir)/'selector.json').write_text(json.dumps({'train_ids': train_ids}))
    low = replace(config, branch='rguo_mid_eff')
    checkpoint = {'format_version': 1, 'config': asdict(low), 'model': PandaModel(low).state_dict(), 'train_ids': train_ids}
    initial = tmp_path/'initial.pt'
    torch.save(checkpoint, initial)
    cfg = tmp_path/'config.json'
    config.save(cfg)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(cfg), '--device', 'cpu', '--init', str(initial)])
    train.main()
    trained = torch.load(Path(config.output_dir)/'last.pt', weights_only=True)
    assert trained['config']['branch'] == 'rguo_high' and trained['epoch'] == 0
    # Cross-resolution transfer is initialization only, never resumption of the old optimizer/run.
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(cfg), '--device', 'cpu', '--resume', str(initial)])
    with pytest.raises(ValueError, match='branch/backbone'):
        train.main()


def test_report_alignment_and_validation(tmp_path):
    from report_results import load_evaluation
    predictions, labels = tmp_path/'scores.csv', tmp_path/'labels.csv'
    pd.DataFrame({'image_id': ['b', 'a'], 'score': [2., 0.], 'isup_grade': [5, 5]}).to_csv(predictions, index=False)
    pd.DataFrame({'image_id': ['a', 'b'], 'isup_grade': [0, 2], 'data_provider': ['radboud']*2}).to_csv(labels, index=False)
    assert load_evaluation(predictions, labels).isup_grade.tolist() == [2, 0]
    pd.DataFrame({'image_id': ['unknown'], 'score': [0.]}).to_csv(predictions, index=False)
    with pytest.raises(ValueError, match='missing'):
        load_evaluation(predictions, labels)
    pd.DataFrame({'image_id': ['a', 'a'], 'score': [0., 1.]}).to_csv(predictions, index=False)
    with pytest.raises(ValueError, match='unique'):
        load_evaluation(predictions, labels)
    pd.DataFrame({'image_id': ['a'], 'score': [np.nan]}).to_csv(predictions, index=False)
    with pytest.raises(ValueError, match='finite'):
        load_evaluation(predictions, labels)


def test_measured_report_and_calibration_guard(tmp_path):
    pytest.importorskip('matplotlib')
    from report_results import create_report, file_hash
    predictions, labels = tmp_path/'scores.csv', tmp_path/'labels.csv'
    ids = [f'slide_{i}' for i in range(6)]
    pd.DataFrame({'image_id': ids, 'score': [0., 1., 2., 3., 4., 4.]}).to_csv(predictions, index=False)
    pd.DataFrame({'image_id': ids, 'isup_grade': list(range(6)),
                  'data_provider': ['radboud', 'karolinska']*3}).to_csv(labels, index=False)
    history = tmp_path/'history.json'
    history.write_text(json.dumps([{'epoch': i+1, 'train_loss': 3/(i+1), 'valid_loss': 4/(i+1), 'qwk': i/4}
                                   for i in range(3)]))
    folder = tmp_path/'report'
    metrics = create_report(predictions, labels, folder, 'validation', history)
    assert metrics['accuracy'] == pytest.approx(5/6)
    # Observed squared error = 1; expected squared error = 31 for these marginals.
    assert metrics['qwk'] == pytest.approx(1-1/31)
    assert np.asarray(metrics['confusion_matrix']).sum() == 6
    assert pd.read_csv(folder/'error_cases.csv').image_id.tolist() == ['slide_5']
    assert json.loads((folder/'provenance.json').read_text())['predictions_sha256'] == file_hash(predictions)
    for name in ['confusion_matrix', 'confusion_matrix_normalized', 'grade_distribution', 'loss_curve', 'qwk_curve']:
        with Image.open(folder/f'{name}.png') as figure:
            assert figure.width > 800 and figure.height > 500
    assert '_____' not in (folder/'REPORT.md').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        create_report(predictions, labels, folder, 'validation')
    threshold_file = tmp_path/'thresholds.json'
    threshold_file.write_text(json.dumps({'fitted_on': 'validation', 'image_ids': ids, 'thresholds': [.5, 1.5, 2.5, 3.5, 4.5]}))
    with pytest.raises(ValueError, match='overlaps'):
        create_report(predictions, labels, tmp_path/'test_report', 'test', thresholds_path=threshold_file)
    calibrated = create_report(predictions, labels, tmp_path/'calibrated', 'validation', thresholds_path=threshold_file)
    assert 'calibration-set score' in calibrated['calibration_note']


def test_compressed_cache_and_incomplete_archive(tmp_path):
    from panda.cache import read_tiles, verify_cache, TileWriter
    config = Config(image_dir=str(tmp_path), cache_dir=str(tmp_path/'cache'),
                    tile_size=32, num_tiles=4, cache_tiles=4, level=0, cache_format='jpeg')
    Image.fromarray(np.full((64, 64, 3), 100, np.uint8)).save(tmp_path/'slide.png')
    prepare_slide({'image_id': 'slide', 'data_provider': 'radboud'}, config)
    folder = tmp_path/'cache/slide'
    meta = verify_cache(folder, config)
    tiles = read_tiles(folder, meta, 2)
    assert tiles.shape == (2, 32, 32, 3) and tiles.dtype == np.uint8
    assert np.all(tiles == 100)
    assert (folder/'tiles.zip').stat().st_size < 4*32*32*3
    with pytest.raises(ValueError, match='quality'):
        verify_cache(folder, replace(config, jpeg_quality=90))
    incomplete = tmp_path/'incomplete'
    incomplete.mkdir()
    with pytest.raises(ValueError, match='Incomplete'):
        with TileWriter(incomplete, config) as writer:
            writer.write(0, tiles[0])
    assert not (incomplete/'tiles.zip').exists()


def test_streaming_resume_and_scratch_cleanup(tmp_path):
    from prepare_stream import prepare_stream
    config = Config(csv=str(tmp_path/'folds.csv'), cache_dir=str(tmp_path/'cache'),
                    level=0, tile_size=32, num_tiles=4, cache_tiles=4, cache_format='jpeg', require_patch_labels=True)
    rows = [{'image_id': f'slide_{i}', 'data_provider': 'radboud', 'isup_grade': i, 'fold': i%2} for i in range(3)]
    pd.DataFrame(rows).to_csv(config.csv, index=False)
    scratch = tmp_path/'scratch'
    scratch.mkdir()
    original = scratch/'user-file.txt'
    original.write_text('preserve')
    fetched = []
    def fetch(cli, image_id, folder, destination):
        fetched.append((image_id, folder))
        assert len(list(scratch.glob('panda-slide-*'))) == 1
        target = destination/folder
        target.mkdir(parents=True)
        suffix = '_mask.png' if folder == 'train_label_masks' else '.png'
        value = 3 if folder == 'train_label_masks' else 100
        Image.fromarray(np.full((64, 64, 3), value, np.uint8)).save(target/(image_id+suffix))
    assert not prepare_stream(config, scratch, max_slides=1, reserve_gb=0, fetch=fetch)
    assert not (Path(config.cache_dir)/'prepared.csv').exists()
    assert len(fetched) == 2 and list(scratch.iterdir()) == [original]
    assert prepare_stream(config, scratch, reserve_gb=0, fetch=fetch)
    assert len(fetched) == 6 and original.read_text() == 'preserve'
    assert pd.read_csv(Path(config.cache_dir)/'prepared.csv').image_id.tolist() == [r['image_id'] for r in rows]
    assert prepare_stream(config, scratch, reserve_gb=0, fetch=fetch)
    assert len(fetched) == 6
    with pytest.raises(ValueError, match='inputs changed'):
        prepare_stream(replace(config, jpeg_quality=90), scratch, reserve_gb=0, fetch=fetch)
    def failed_fetch(cli, image_id, folder, destination):
        fetch(cli, image_id, folder, destination)
        raise OSError('Simulated download failure')
    with pytest.raises(OSError, match='Simulated'):
        prepare_stream(replace(config, cache_dir=str(tmp_path/'failed-cache')), scratch, reserve_gb=0, fetch=failed_fetch)
    assert list(scratch.iterdir()) == [original]


def test_mid_epoch_resume_matches_uninterrupted_weights(tmp_path, monkeypatch):
    import train
    config = Config(backbone='tiny', pretrained=False, csv=str(tmp_path/'labels.csv'),
                    image_dir=str(tmp_path), cache_dir=str(tmp_path/'cache'), output_dir=str(tmp_path/'full'),
                    level=0, tile_size=32, num_tiles=4, cache_tiles=4, tile_batch_size=2,
                    batch_size=2, accumulation_steps=2, epochs=2, folds=2, cache_format='jpeg')
    rows = []
    for i in range(18):
        row = {'image_id': f'slide_{i:02d}', 'data_provider': 'radboud', 'isup_grade': i%6, 'fold': i%2}
        Image.fromarray(np.random.default_rng(i).integers(30, 220, (64, 64, 3), dtype=np.uint8)).save(tmp_path/f'slide_{i:02d}.png')
        prepare_slide(row, config)
        rows.append(row)
    pd.DataFrame(rows).to_csv(config.csv, index=False)
    cfg = tmp_path/'config.json'
    config.save(cfg)
    arguments = ['train.py', '--config', str(cfg), '--device', 'cpu', '--checkpoint-every-updates', '1']
    monkeypatch.setattr(sys, 'argv', arguments)
    train.main()
    actual_save = train.save_checkpoint
    def interrupt(value, path):
        actual_save(value, path)
        if not value['epoch_complete'] and value['next_sample'] == 4:
            raise KeyboardInterrupt
    monkeypatch.setattr(train, 'save_checkpoint', interrupt)
    monkeypatch.setattr(sys, 'argv', arguments+['--output-dir', str(tmp_path/'resumed')])
    with pytest.raises(KeyboardInterrupt):
        train.main()
    partial = torch.load(tmp_path/'resumed/last.pt', weights_only=True)
    assert partial['next_sample'] == 4 and len(partial['history']) == 0
    monkeypatch.setattr(train, 'save_checkpoint', actual_save)
    monkeypatch.setattr(sys, 'argv', arguments+['--output-dir', str(tmp_path/'resumed'), '--resume', str(tmp_path/'resumed/last.pt')])
    train.main()
    full = torch.load(tmp_path/'full/last.pt', weights_only=True)
    resumed = torch.load(tmp_path/'resumed/last.pt', weights_only=True)
    assert resumed['history'] == full['history']
    for key, tensor in full['model'].items():
        assert torch.equal(tensor, resumed['model'][key]), key
    # A user-selected session budget also saves at an update boundary and exits cleanly.
    monkeypatch.setattr(sys, 'argv', arguments+['--output-dir', str(tmp_path/'budget'), '--session-minutes', '1e-9'])
    train.main()
    budget = torch.load(tmp_path/'budget/last.pt', weights_only=True)
    assert not budget['epoch_complete'] and budget['next_sample'] == 4
    old_config = (tmp_path/'budget/config.json').read_bytes()
    old_splits = (tmp_path/'budget/splits.csv').read_bytes()
    monkeypatch.setattr(sys, 'argv', arguments+['--output-dir', str(tmp_path/'budget'), '--epochs', '3',
                                              '--resume', str(tmp_path/'budget/last.pt')])
    with pytest.raises(ValueError, match='Cannot change epochs'):
        train.main()
    assert (tmp_path/'budget/config.json').read_bytes() == old_config
    assert (tmp_path/'budget/splits.csv').read_bytes() == old_splits


def test_validation_uses_one_forward_per_batch():
    from panda.engine import validate
    config = Config(backbone='tiny', pretrained=False, tile_size=32, num_tiles=4, cache_tiles=4)
    model = PandaModel(config)
    batch = {'tiles': torch.rand(2, 4, 3, 32, 32), 'label': torch.tensor([1, 4]),
             'provider': torch.tensor([0, 1]), 'image_id': ['a', 'b']}
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(1))
    loss, predictions = validate(model, [batch], torch.device('cpu'))
    handle.remove()
    assert len(calls) == 1 and len(predictions) == 2 and np.isfinite(loss)


def test_mask_free_inference_preparation_is_resumable(tmp_path, monkeypatch):
    import prepare_data
    config = Config(branch='rguo_mid_eff', backbone='tiny', pretrained=False,
                    csv=str(tmp_path/'test.csv'), image_dir=str(tmp_path), mask_dir=str(tmp_path/'missing-masks'),
                    cache_dir=str(tmp_path/'cache'), level=0, tile_size=32, num_tiles=4, cache_tiles=4,
                    require_patch_labels=True, cache_format='jpeg')
    row = {'image_id': 'inference', 'data_provider': 'radboud'}
    Image.fromarray(np.full((64, 64, 3), 100, np.uint8)).save(tmp_path/'inference.png')
    pd.DataFrame([row]).to_csv(config.csv, index=False)
    config.save(tmp_path/'config.json')
    with pytest.raises(ValueError, match='requires masks'):
        prepare_slide(row, replace(config, mask_dir=None))
    monkeypatch.setattr(sys, 'argv', ['prepare_data.py', '--config', str(tmp_path/'config.json'), '--inference'])
    prepare_data.main()
    prepare_data.main()
    generated = Config(**json.loads((tmp_path/'cache/config.json').read_text()))
    assert generated.require_patch_labels is False and generated.mask_dir is None
    assert TileDataset(pd.DataFrame([row]), generated)[0]['tiles'].shape == (4, 3, 32, 32)


def test_external_test_evaluation_does_not_select_checkpoints(tmp_path, monkeypatch):
    import evaluate
    config = Config(backbone='tiny', pretrained=False, level=0, tile_size=32, num_tiles=4,
                    cache_tiles=4, image_dir=str(tmp_path), cache_dir=str(tmp_path/'cache'))
    rows = []
    for i in range(2):
        row = {'image_id': f'test_{i}', 'data_provider': 'radboud', 'isup_grade': i,
               'prediction_reg': float('nan'), 'prob_0': float('nan')}
        Image.fromarray(np.full((64, 64, 3), 80+i*30, np.uint8)).save(tmp_path/f'test_{i}.png')
        prepare_slide(row, config)
        rows.append(row)
    labels = tmp_path/'test.csv'
    pd.DataFrame(rows).to_csv(labels, index=False)
    checkpoint = tmp_path/'selected.pt'
    torch.save({'format_version': 1, 'config': asdict(config), 'model': PandaModel(config).state_dict(),
                'train_ids': ['training'], 'valid_ids': ['validation']}, checkpoint)
    base = ['evaluate.py', '--csv', str(labels), '--cache-dir', config.cache_dir, '--external-test',
            '--output-dir', str(tmp_path/'evaluation'), '--device', 'cpu', '--tta', '1', '--checkpoints']
    monkeypatch.setattr(sys, 'argv', base+[str(checkpoint), str(checkpoint)])
    with pytest.raises(SystemExit):
        evaluate.main()
    monkeypatch.setattr(sys, 'argv', base+[str(checkpoint)])
    evaluate.main()
    assert (tmp_path/'evaluation/evaluation.json').exists()
    assert not (tmp_path/'evaluation/selection.json').exists()
    assert json.loads((tmp_path/'evaluation/checkpoint_0.json').read_text())['samples'] == 2
    with pytest.raises(FileExistsError, match='empty evaluation'):
        evaluate.main()
