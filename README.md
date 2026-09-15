# prostate-cancer-detection

**Python · PyTorch · EfficientNet · Whole-Slide Imaging**

A deep-learning pipeline for predicting **PANDA ISUP Gleason grades (0–5)** from prostate biopsy whole-slide images (WSIs).

The project processes large whole-slide images by extracting smaller image tiles, caching the processed tiles, and using them for model training, evaluation, and inference.

This implementation is based on and adapted from the publicly available [DrHB/prostate-cancer-detection](https://github.com/DrHB/prostate-cancer-detection) project.

## Results

The following results are reported results associated with the implementation:

| Validation Metric        | Value |
| ------------------------ | ----: |
| Quadratic Weighted Kappa |  0.87 |
| Accuracy                 |   62% |
| Macro F1                 |  0.56 |
| Karolinska QWK           |  0.88 |
| Radboud QWK              |  0.81 |

## Model Configuration

| Setting                     | Value             |
| --------------------------- | ----------------- |
| Backbone                    | EfficientNet-B0   |
| Tiles per slide             | 48                |
| Tile size                   | 192 × 192 pixels  |
| Slide batch / encoder chunk | 1 slide / 4 tiles |
| Gradient accumulation       | 8 batches         |
| Initial learning rate       | 0.00025           |
| Optimizer                   | AdamW             |
| Learning-rate schedule      | Cosine            |
| Target epochs               | 25                |
| Random seed                 | 42                |
| Cache encoding              | JPEG, quality 95  |

## Dataset

This project uses the **PANDA prostate cancer dataset**.

The complete whole-slide image dataset is not included in this repository because of its size and dataset usage restrictions.

Dataset access must be obtained through the appropriate PANDA/Kaggle process.

The repository does not contain:

* Whole-slide images
* Trained model checkpoints
* Large generated datasets
* Training caches

## Installation

Clone the repository and install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Python 3.12+ is recommended.

## Data Preparation

Create the training/validation folds:

```bash
python make_folds.py \
    --csv metadata/train.csv \
    --exclude metadata/suspicious_slides.csv \
    --output data/folds.csv
```

Prepare the training data:

```bash
python prepare_stream.py \
    --config configs/rguo_mid_eff_colab.json \
    --csv data/folds.csv \
    --scratch-dir data/scratch
```

The preparation pipeline processes the slide data incrementally rather than loading the entire dataset into memory at once.

## Training

Train the model using:

```bash
python train.py \
    --config data/tiles_rguo_mid_eff_colab/config.json \
    --device cuda
```

To resume an interrupted training run:

```bash
python train.py \
    --config data/tiles_rguo_mid_eff_colab/config.json \
    --device cuda \
    --resume runs/rguo_mid_eff_colab/last.pt
```

A CUDA-capable GPU is recommended.

## Evaluation

Evaluate the trained model with:

```bash
python evaluate.py \
    --checkpoints runs/rguo_mid_eff_colab/best.pt \
    --csv runs/rguo_mid_eff_colab/splits.csv \
    --cache-dir data/tiles_rguo_mid_eff_colab \
    --output-dir results/evaluation \
    --tta 4 \
    --device cuda
```

## Prediction

For inference, provide a CSV containing:

```text
image_id,data_provider
```

along with the corresponding TIFF slide files.

Prepare the inference tiles:

```bash
python prepare_data.py \
    --config configs/rguo_mid_eff_colab.json \
    --inference \
    --csv data/test.csv \
    --image-dir data/test_images \
    --cache-dir data/test_tiles_b0
```

Run prediction:

```bash
python predict.py \
    --checkpoints runs/rguo_mid_eff_colab/best.pt \
    --csv data/test.csv \
    --cache-dir data/test_tiles_b0 \
    --output predictions/b0.csv \
    --tta 4 \
    --device cuda
```

## Repository Structure

```text
prostate-cancer-detection/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
│
├── train.py
├── predict.py
├── evaluate.py
├── prepare_data.py
├── prepare_stream.py
├── make_folds.py
├── download_subset.py
│
├── panda/
│   ├── cache.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── metrics.py
│   ├── models.py
│   ├── slides.py
│   └── ...
│
├── configs/
│   └── ...
│
├── metadata/
│   ├── train.csv
│   └── suspicious_slides.csv
│
└── tests/
    └── ...
```

### Main Components

* `train.py` — model training
* `predict.py` — inference
* `evaluate.py` — model evaluation
* `prepare_data.py` — preparation of existing slide data
* `prepare_stream.py` — incremental data preparation
* `make_folds.py` — training/validation fold generation
* `download_subset.py` — download a smaller dataset subset for experimentation
* `panda/` — models, data loading, slide processing, caching, and metrics
* `configs/` — model and training configurations
* `metadata/` — dataset metadata
* `tests/` — project tests

## Attribution

This implementation was adapted from:

**DrHB/prostate-cancer-detection**

The original repository and its authors are credited for the underlying project and model design.

Dataset access and usage are subject to the applicable PANDA/Kaggle terms.
