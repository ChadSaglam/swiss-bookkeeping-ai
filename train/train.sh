#!/usr/bin/env bash
set -e
hf download ChadSaglam/swiss-sme-bookkeeping-dataset swiss_sme_bookkeeping_synthetic.csv --repo-type=dataset --local-dir .
python3 data/prepare_data_hard.py swiss_sme_bookkeeping_synthetic.csv
uv run --with mlx-lm mlx_lm lora --model Qwen/Qwen2.5-0.5B-Instruct --train \
  --data ./data_hard --iters 600 --batch-size 4 --adapter-path ./adapters_hard
uv run --with mlx-lm mlx_lm fuse --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-path ./adapters_hard --save-path ./fused
uv run --with mlx-lm python eval/eval_model.py adapters_hard
