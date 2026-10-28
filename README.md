# STND-VAD

## Short–Long SDF for Weakly Supervised Video Anomaly Detection

This repository contains the core CLIP 10-crop implementation of the Short/Long SDF method on the XD-Violence dataset. It includes source code and usage instructions only; it does not include datasets, extracted features, model checkpoints, logs, or paper PDFs.

## Contents

- `code/main_short_long_sdf_10crop_try.py`: training entry point.
- `code/context_score_short_long_sdf_10crop_try.py` and its model and temporal mixing dependencies: implement the short and long branches, SDF evidence calculation, and score fusion.
- `code/train_context_score_try.py` and `code/train_semantic_mag_try.py`: training loss functions.
- `code/dataset.py`, `code/utils.py`, and `code/test_component_auc_10crop_try.py`: data loading, preprocessing, and evaluation.
- `code/setup_xd_lists.py`, `code/merge_xd_10crop_features.py`, and `code/check_xd_data.py`: utilities for generating data lists, preparing features, and checking data.

## Requirements

Use Python 3.10 or later. First, install a PyTorch version appropriate for your CPU/CUDA environment by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). Then install the remaining Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data Preparation

The training code requires CLIP 10-crop features for XD-Violence, with shape `[T, 10, 512]`. Obtain the dataset and features from an authorized source. Do not commit feature arrays or license-restricted data to this repository.

From the repository root, run the following command to generate the XD train/test lists and ground-truth annotation files:

```bash
python code/setup_xd_lists.py
```

The generated data lists refer to feature files in the following directories:

```text
features/xd_clip_10crop/train/
features/xd_clip_10crop/test/
```

Place the corresponding feature files in these directories and ensure their filenames match the entries in the lists. By default, the training dataset implementation assumes that the first 1,905 entries in the training list are anomalous and the rest are normal. Verify that your list ordering satisfies this assumption before training. Run the following command to check that the data aligns with the ground-truth (GT) annotations:

```bash
python code/check_xd_data.py
```

If your features are stored separately by crop, use `code/merge_xd_10crop_features.py` to merge them. Use the `--src-train` and `--src-test` arguments to specify paths matching your local feature directory structure.

## Training the XD Baseline

From the repository root, run:

```bash
bash run_xd.sh
```

This script uses the reproducible XD experiment configuration from the original project:

```bash
python code/main_short_long_sdf_10crop_try.py \
  --feature_extractor clip \
  --dataset xd \
  --gpus 0 \
  --max-epoch 45 \
  --early-stop-drop 0.015 \
  --early-stop-min-epoch 25 \
  --seed 237033 \
  --context-mode sdf \
  --component-lam 0.05 \
  --context-gap-lam 0.05 \
  --center-lam 0.0 \
  --sparse-lam 0.01 \
  --scale-gate-mode learned \
  --sdf-prior-mode mean \
  --num-normal-centers 4 \
  --sdf-topk 3 \
  --short-sdf-weight 0.2 \
  --long-sdf-weight 0.2 \
  --short-long-fusion learned \
  --long-head mlp \
  --long-temporal-kernel 3 \
  --train-score-smooth-mode max \
  --train-score-smooth-kernel 5 \
  --test-smooth-mode max \
  --test-smooth-kernel 5 \
  --deterministic
```

Set `--gpus` to the GPU IDs visible on your machine. Model checkpoints and run-configuration JSON files are saved in the `result/` directory.

## License and Citation

No license is currently specified for this repository. Before publicly distributing it, confirm that you have the right to redistribute this implementation and add an appropriate software license and paper citation. Dataset, feature, and annotation licenses are separate from the source code license.
