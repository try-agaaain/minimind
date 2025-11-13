# MiniMind Fine-Tuning 分支

这个分支专门用于 MiniMind 模型的微调训练。

## 文件说明

- `minimind.py`: MiniMind 模型定义文件 (从 master 分支的 model/model_minimind.py 复用)
- `train.py`: 简洁的模型训练脚本

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers
```

### 2. 运行训练

最简单的训练方式:
```bash
python train.py
```

### 3. 自定义训练参数

```bash
python train.py \
    --hidden_size 512 \
    --num_layers 8 \
    --num_heads 8 \
    --epochs 5 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --output_dir ./output
```

### 4. 使用预训练权重

```bash
python train.py --pretrained_path /path/to/pretrained/model.pth
```

## 参数说明

### 模型配置
- `--hidden_size`: 隐藏层维度 (默认: 512)
- `--num_layers`: 隐藏层数量 (默认: 8)
- `--num_heads`: 注意力头数量 (默认: 8)
- `--vocab_size`: 词表大小 (默认: 6400)
- `--max_seq_len`: 最大序列长度 (默认: 512)
- `--dropout`: Dropout率 (默认: 0.1)
- `--use_moe`: 是否使用MoE架构 (默认: 0)

### 训练配置
- `--epochs`: 训练轮数 (默认: 3)
- `--batch_size`: 批次大小 (默认: 4)
- `--learning_rate`: 学习率 (默认: 1e-4)
- `--weight_decay`: 权重衰减 (默认: 0.01)
- `--grad_clip`: 梯度裁剪 (默认: 1.0)

### 其他配置
- `--device`: 训练设备 (默认: cuda:0)
- `--output_dir`: 模型保存目录 (默认: ./output)
- `--pretrained_path`: 预训练权重路径 (默认: "")
- `--log_interval`: 日志打印间隔 (默认: 10)
- `--save_interval`: 模型保存间隔 (默认: 1 epoch)

## 注意事项

1. 当前的训练脚本使用简单的字符级tokenization作为示例
2. 实际使用时，建议使用专业的tokenizer
3. 可以根据需要扩展数据加载部分，支持从文件读取数据
4. 训练的模型会保存在 `output_dir` 指定的目录中

## 示例输出

训练过程中会看到类似如下的输出:

```
==================================================
MiniMind 训练配置:
==================================================
hidden_size: 512
num_layers: 8
epochs: 3
batch_size: 4
...
==================================================
使用设备: cuda:0
初始化模型...
开始训练 3 个epoch...
Epoch [1/3], Step [10/100], Loss: 8.7634, Avg Loss: 8.8521
Epoch [1/3], Step [20/100], Loss: 8.5234, Avg Loss: 8.6234
...
模型已保存到: ./output/model_epoch_1.pth
训练完成! 最终模型已保存到: ./output/model_final.pth
```

## 许可证

继承自原 MiniMind 项目
