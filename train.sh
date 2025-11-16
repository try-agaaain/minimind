#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

git submodule update --init --recursive

torchrun --nproc_per_node=2 train.py --use_jsonl --data_path ./dataset/pretrain.jsonl

# 默认参数
EPOCHS=${1:-5}
BATCH_SIZE=${2:-8}
LEARNING_RATE=${3:-1e-4}
NUM_GPUS=${4:-1}
SAVE_INTERVAL=${5:-1}
RESUME=${6:-0}

# 构建 torchrun 命令
TORCHRUN_CMD="uv run torchrun --nproc_per_node=$NUM_GPUS train.py"

# 根据 RESUME 参数决定是否添加 --resume_from_checkpoint
if [ "$RESUME" -eq 1 ]; then
    RESUME_FLAG="--resume_from_checkpoint"
else
    RESUME_FLAG=""
fi

$TORCHRUN_CMD \
    --epochs "$EPOCHS"  \
    --batch_size "$BATCH_SIZE"  \
    --learning_rate "$LEARNING_RATE"  \
    --save_interval "$SAVE_INTERVAL"  \
    --use_jsonl  \
    --use_scheduler  \
    $RESUME_FLAG

echo "✅ 完成！输出: ./output/"

python3 chat.py