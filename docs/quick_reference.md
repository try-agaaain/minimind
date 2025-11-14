# MiniMind 快速参考手册

本文档提供 MiniMind 关键概念和代码的快速查找参考。

## 目录

- [模型配置速查](#模型配置速查)
- [核心组件速查](#核心组件速查)
- [训练参数速查](#训练参数速查)
- [常用代码片段](#常用代码片段)

---

## 模型配置速查

### 预设配置

#### 小型模型 (~50M 参数)
```python
config = MiniMindConfig(
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=2,
    vocab_size=6400,
    max_position_embeddings=2048
)
```

#### 中型模型 (~200M 参数)
```python
config = MiniMindConfig(
    hidden_size=1024,
    num_hidden_layers=16,
    num_attention_heads=16,
    num_key_value_heads=4,
    vocab_size=32000,
    max_position_embeddings=4096
)
```

#### 大型模型 (~1B 参数)
```python
config = MiniMindConfig(
    hidden_size=2048,
    num_hidden_layers=24,
    num_attention_heads=32,
    num_key_value_heads=8,
    vocab_size=32000,
    max_position_embeddings=8192
)
```

### MoE 配置
```python
config = MiniMindConfig(
    # ... 基础配置 ...
    use_moe=True,
    num_experts_per_tok=2,      # 每个 token 激活的专家数
    n_routed_experts=8,          # 路由专家总数
    n_shared_experts=1,          # 共享专家数
    aux_loss_alpha=0.1           # 辅助损失权重
)
```

### 长序列配置（使用 YaRN）
```python
config = MiniMindConfig(
    # ... 基础配置 ...
    max_position_embeddings=32768,  # 32K 上下文
    inference_rope_scaling=True,     # 启用 YaRN
    rope_theta=1000000.0            # RoPE 基数
)
```

---

## 核心组件速查

### RMSNorm
**公式**: `output = weight * (x / sqrt(mean(x^2) + eps))`

**用途**: 
- 输入归一化（注意力前）
- FFN 归一化（FFN 前）
- 最终归一化（输出前）

**参数**: 
- `dim`: 特征维度
- `eps`: 数值稳定性（默认 1e-5）

### RoPE (旋转位置编码)
**核心思想**: 通过旋转操作注入位置信息

**频率**: `θ_i = base^(-2i/d)`，默认 `base = 1e6`

**应用位置**: 只应用于 Q 和 K，不应用于 V

**YaRN 外推**: 当序列长度超过训练长度时自动缩放

### GQA (分组查询注意力)
**配置关系**: 
```
n_heads = 8, n_kv_heads = 2
→ 每个 KV 头对应 4 个 Q 头
→ KV cache 减少 75%
```

**Flash Attention**: PyTorch 2.0+ 自动启用（训练更快）

**KV Cache**: 推理时缓存历史 K, V，避免重复计算

### SwiGLU FFN
**公式**: `output = (silu(W_gate(x)) * W_up(x)) @ W_down`

**中间维度**: 约为 `hidden_size * 2.67`（向上取整到 64 的倍数）

**激活函数**: SiLU (Swish) = `x * sigmoid(x)`

### MoE
**路由**: Top-K 选择，每个 token 激活 K 个专家

**负载均衡**: 辅助损失 `aux_loss = α * Σ(P_i * f_i)`

**共享专家**: 总是激活，学习通用特征

---

## 训练参数速查

### 推荐的训练配置

#### 小模型（GPU 显存 < 8GB）
```bash
python train.py \
    --hidden_size 512 \
    --num_layers 8 \
    --num_heads 8 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --grad_clip 1.0
```

#### 中等模型（GPU 显存 16-24GB）
```bash
python train.py \
    --hidden_size 1024 \
    --num_layers 16 \
    --num_heads 16 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --grad_clip 1.0
```

#### 使用 MoE
```bash
python train.py \
    --use_moe 1 \
    --hidden_size 1024 \
    --num_layers 16 \
    --batch_size 4 \
    --learning_rate 5e-5
```

### 优化器参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| learning_rate | 1e-4 ~ 5e-4 | 从零训练 |
| learning_rate | 1e-5 ~ 5e-5 | 微调预训练模型 |
| weight_decay | 0.01 ~ 0.1 | L2 正则化 |
| grad_clip | 1.0 | 梯度裁剪阈值 |
| betas | (0.9, 0.999) | AdamW 默认值 |

### 学习率调度

**预热步数**: 总步数的 1-10%
```python
warmup_steps = total_steps * 0.1
```

**余弦衰减**: 预热后使用余弦函数衰减
```python
lr = base_lr * 0.5 * (1 + cos(π * progress))
```

---

## 常用代码片段

### 初始化模型
```python
from minimind import MiniMindConfig, MiniMindForCausalLM

# 创建配置
config = MiniMindConfig(
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=2
)

# 初始化模型
model = MiniMindForCausalLM(config)
model = model.to(device)

# 加载预训练权重
state_dict = torch.load('model.pth', map_location=device)
model.load_state_dict(state_dict, strict=False)
```

### 训练循环（标准）
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(epochs):
    for input_ids, labels in dataloader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(input_ids)
        loss = criterion(outputs.logits.view(-1, vocab_size), labels.view(-1))
        
        # MoE 辅助损失
        if hasattr(outputs, 'aux_loss') and outputs.aux_loss:
            loss = loss + outputs.aux_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

### 训练循环（混合精度）
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(epochs):
    for input_ids, labels in dataloader:
        optimizer.zero_grad()
        
        # 混合精度前向
        with autocast():
            outputs = model(input_ids)
            loss = criterion(outputs.logits.view(-1, vocab_size), labels.view(-1))
            if hasattr(outputs, 'aux_loss') and outputs.aux_loss:
                loss = loss + outputs.aux_loss
        
        # 缩放反向传播
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
```

### 推理生成（基础）
```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens=50, temperature=1.0):
    model.eval()
    
    for _ in range(max_new_tokens):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if next_token.item() == eos_token_id:
            break
    
    return input_ids
```

### 推理生成（使用 KV Cache）
```python
@torch.no_grad()
def generate_with_cache(model, input_ids, max_new_tokens=50):
    model.eval()
    past_key_values = None
    
    for _ in range(max_new_tokens):
        # 第一次处理完整输入，之后只处理新 token
        model_inputs = input_ids if past_key_values is None else input_ids[:, -1:]
        
        outputs = model(model_inputs, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if next_token.item() == eos_token_id:
            break
    
    return input_ids
```

### Top-p (Nucleus) Sampling
```python
def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 移除累积概率超过 top_p 的 token
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
```

### 数据集（使用真实 Tokenizer）
```python
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()
        
        # 因果语言建模：输入和标签错位一位
        return input_ids[:-1], labels[1:]
```

### 保存和加载检查点
```python
# 保存
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'config': config.__dict__
}
torch.save(checkpoint, 'checkpoint.pt')

# 加载
checkpoint = torch.load('checkpoint.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### 计算模型参数量
```python
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"参数量: {total_params / 1e6:.2f}M")
    
    return total_params, trainable_params

count_parameters(model)
```

### 查看模型结构
```python
# 打印模型架构
print(model)

# 统计每层参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, {param.numel():,} params")

# 查看配置
print(model.config)
```

---

## 故障排除速查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 损失变为 NaN | 梯度爆炸、学习率过大 | 降低学习率、启用梯度裁剪 |
| 损失不下降 | 学习率过小、数据问题 | 增加学习率、检查数据 |
| 显存不足 | batch size 太大 | 减小 batch size、使用梯度累积 |
| 训练太慢 | 未使用优化 | 启用混合精度、Flash Attention |
| 过拟合 | 模型太大、数据太少 | 增加 dropout、权重衰减 |
| 生成重复 | Temperature 太低 | 增加 temperature、使用 top-p |
| 生成混乱 | Temperature 太高 | 降低 temperature |

---

## 性能优化清单

- [ ] 使用混合精度训练（`autocast` + `GradScaler`）
- [ ] 启用 Flash Attention（PyTorch 2.0+）
- [ ] 使用 KV Cache 加速推理
- [ ] 使用梯度累积模拟大 batch size
- [ ] 使用 DataLoader 的 `num_workers > 0`
- [ ] 固定序列长度（避免动态填充）
- [ ] 使用编译模式（PyTorch 2.0+ `torch.compile()`）
- [ ] 使用 GQA 减少 KV cache 大小
- [ ] 考虑使用 MoE 提高参数效率

---

**💡 提示**: 详细的原理和实现细节请参考完整文档：
- [模型架构详解](./minimind_architecture.md)
- [训练指南](./training_guide.md)
