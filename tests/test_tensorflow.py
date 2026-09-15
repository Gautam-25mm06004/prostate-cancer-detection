"""Optional TFRecord/Keras integration test: pytest tests/test_tensorflow.py."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
import pytest

tf = pytest.importorskip('tensorflow')


def test_real_keras_efficientnet_forward():
    from panda.tensorflow_model import build_model
    model = build_model(64, pretrained=False)
    result = model(np.full((1, 64, 64, 3), 128, np.float32), training=False).numpy()
    assert result.shape == (1, 1)
    assert np.isfinite(result).all()
    tf.keras.backend.clear_session()


def test_tensorflow_records_training_resume_prediction(tmp_path, monkeypatch):
    import prepare_tfrecords
    import train_tensorflow
    import predict_tensorflow
    from prepare_data import prepare_slide
    from panda.config import Config
    images = tmp_path/'images'
    images.mkdir()
    config = Config(branch='xie29', backbone='tiny', pretrained=False, csv=str(tmp_path/'train.csv'),
                    image_dir=str(images), cache_dir=str(tmp_path/'cache'), level=0, num_tiles=4,
                    cache_tiles=4, tile_size=32, folds=2)
    rows = []
    for i in range(8):
        row = {'image_id': f'slide_{i}', 'data_provider': 'radboud' if i%3 else 'karolinska',
               'isup_grade': i%6, 'fold': i%2}
        Image.fromarray(np.full((64, 64, 3), 40+i*20, np.uint8)).save(images/f'slide_{i}.png')
        prepare_slide(row, config)
        rows.append(row)
    pd.DataFrame(rows).to_csv(config.csv, index=False)
    config_path = tmp_path/'config.json'
    config.save(config_path)
    records = tmp_path/'records'
    monkeypatch.setattr(sys, 'argv', ['prepare_tfrecords.py', '--config', str(config_path), '--output-dir', str(records), '--shard-size', '2'])
    prepare_tfrecords.main()
    assert len(list(records.glob('fold_*/*.tfrecord'))) == 4
    run = tmp_path/'run'
    argv = ['train_tensorflow.py', '--records', str(records), '--output-dir', str(run), '--epochs', '1', '--tiny', '--no-pretrained', '--batch-size', '2']
    monkeypatch.setattr(sys, 'argv', argv)
    train_tensorflow.main()
    argv[argv.index('--epochs')+1] = '2'
    monkeypatch.setattr(sys, 'argv', argv+['--resume', str(run/'last.keras')])
    train_tensorflow.main()
    assert json.loads((run/'last.state.json').read_text())['epoch'] == 2
    validation = tmp_path/'valid.csv'
    pd.DataFrame(rows)[lambda frame: frame.fold == 0].to_csv(validation, index=False)
    monkeypatch.setattr(sys, 'argv', ['predict_tensorflow.py', '--checkpoints', str(run/'best.keras'), '--config', str(config_path),
                                    '--csv', str(validation), '--cache-dir', config.cache_dir, '--output', str(tmp_path/'pred.csv'), '--evaluate'])
    predict_tensorflow.main()
    assert len(pd.read_csv(tmp_path/'pred.csv')) == 4
