#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 code/main_short_long_sdf_10crop_try.py \
  --feature_extractor clip \
  --dataset xd \
  --gpus "${GPU:-0}" \
  --max-epoch 45 \
  --early-stop-drop 0.015 \
  --early-stop-min-epoch 25 \
  --seed "${SEED:-237033}" \
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
