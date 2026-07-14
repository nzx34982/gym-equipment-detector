# Gym Equipment Detector

A small PyTorch object-detection prototype for six categories of gym equipment.

> Status: undergraduate learning and research prototype. The repository demonstrates
> the data-loading, training, checkpoint, and inference pipeline. It does **not** yet
> provide a reproducible benchmark or claim production-level accuracy.

中文简介：基于 PyTorch Faster R-CNN 的六类健身器械目标检测原型，包含数据处理、训练和推理流程。

## What is implemented

- A VOC-style `Dataset` for bounding-box annotations.
- Faster R-CNN with a MobileNetV3-320-FPN backbone.
- Deterministic train/validation split generation.
- Training with checkpoint and JSON loss-history output.
- Single-image and validation-set inference scripts.
- Lightweight tests for dataset parsing and data-preparation utilities.

## Categories

| ID | Class |
|---:|---|
| 1 | `treadmill` |
| 2 | `elliptical` |
| 3 | `stair_climber` |
| 4 | `big_scissor_machine` |
| 5 | `single_arm_row_machine` |
| 6 | `tbar_row_machine` |

## Repository structure

```text
.
├── src/                    # Dataset, model, training, and inference code
├── tests/                  # Automated unit tests
├── splits/                 # Current train/validation identifiers (31/8)
├── images_jpg/             # Nested source images (currently tracked)
├── images_flat/            # Local flattened images (ignored)
├── annotations_voc/        # Local VOC XML annotations (ignored)
├── checkpoints/            # Local model checkpoints (ignored)
└── outputs/                # Local metrics and prediction images (ignored)
```

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Prepare the local image directory:

```bash
python flatten_images.py
python check_files.py --split-file splits/train.txt
```

VOC XML annotations must be placed in `annotations_voc/`. Each XML filename must
match the corresponding flattened image filename. Generate new deterministic splits
when the annotations change:

```bash
python make_splits.py --seed 42 --train-ratio 0.8
```

Train and run inference:

```bash
python -m src.train_detector --epochs 10
python -m src.predict_one --image images_flat/example.jpg
python -m src.predict_val_all
```

Run tests:

```bash
python -m pytest
```

## Data status and limitations

- The current repository contains 39 JPG images and split files for 31 training and
  8 validation samples.
- VOC XML annotations, trained checkpoints, and generated outputs are not currently
  published. A fresh clone therefore cannot reproduce training without preparing
  these local assets.
- Dataset provenance and redistribution permission have not yet been documented.
  Do not reuse the images until a dataset statement and license are added.
- The dataset is too small for broad claims about real-world performance.
- No mAP, per-class AP, or held-out test-set result is currently published. This
  README intentionally does not invent or infer missing metrics.

## Reproducibility checklist

- [x] Relative paths and command-line arguments
- [x] Dependency specification
- [x] Deterministic split seed
- [x] Automated utility and dataset tests
- [x] Checkpoint metadata and loss-history output
- [ ] Public annotation instructions or annotation release
- [ ] Dataset provenance and redistribution statement
- [ ] Fixed experiment configuration and environment lockfile
- [ ] Validation mAP and per-class AP
- [ ] Representative prediction images

## Safety and privacy

Do not commit `.env` files, virtual environments, checkpoints, training logs, or
images containing identifiable people or private location information. Review EXIF
metadata and dataset redistribution rights before publishing new images.

## License

No license has been granted yet. Code and data reuse terms must be defined after the
dataset provenance review; the code license and dataset license may need to differ.
