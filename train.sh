#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

# 默认参数
EPOCHS=${1:-5}
BATCH_SIZE=${2:-8}
LEARNING_RATE=${3:-1e-4}
DEVICE=${4:-cuda}

echo "📚 创建数据集..."
python3 dataset.py

echo "🚀 训练模型..."
python3 train.py \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --device "$DEVICE" \
    --use_jsonl \
    --use_scheduler

echo "✅ 完成！输出: ./output/"

python3 chat.py