#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

[ ! -d "dataset" ] && git submodule update --init --recursive

# 默认参数
EPOCHS=${1:-5}
BATCH_SIZE=${2:-8}
LEARNING_RATE=${3:-1e-4}
DEVICE=${4:-cuda}
SAVE_INTERVAL=${5:-1}
RESUME=${6:-0}
GPU_IDS=${7:-"0,1"}

python3 train.py \
    --epochs "$EPOCHS"  \
    --batch_size "$BATCH_SIZE"  \
    --learning_rate "$LEARNING_RATE"  \
    --device "$DEVICE"  \
    --save_interval "$SAVE_INTERVAL"  \
    --use_jsonl  \
    --use_scheduler  \
    --gpu_ids "$GPU_IDS"  \
    --resume_from_checkpoint

echo "✅ 完成！输出: ./output/"

python3 chat.py