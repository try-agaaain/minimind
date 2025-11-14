# MiniMind 训练指南

## 目录

1. [训练流程概述](#训练流程概述)
2. [数据处理](#数据处理)
3. [优化器与学习率](#优化器与学习率)
4. [训练技巧](#训练技巧)
5. [损失函数](#损失函数)
6. [推理与生成](#推理与生成)

---

## 训练流程概述

### 整体架构

`train.py` 实现了一个简洁但完整的训练流程，包含以下核心组件：

```
训练流程
├── 数据准备
│   ├── 数据集加载
│   ├── Tokenization
│   └── DataLoader
├── 模型初始化
│   ├── 配置创建
│   ├── 模型实例化
│   └── 预训练权重加载（可选）
├── 训练设置
│   ├── 优化器配置
│   ├── 损失函数
│   └── 设备设置
└── 训练循环
    ├── 前向传播
    ├── 损失计算
    ├── 反向传播
    ├── 梯度裁剪
    ├── 参数更新
    └── 模型保存
```

### 主函数流程

```python
def train(args):
    # 1. 设备设置
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 2. 模型配置和初始化
    config = MiniMindConfig(...)
    model = MiniMindForCausalLM(config).to(device)
    
    # 3. 加载预训练权重（可选）
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        state_dict = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    
    # 4. 准备数据
    dataset = SimpleTextDataset(texts, max_length=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 5. 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 6. 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 7. 训练循环
    for epoch in range(args.epochs):
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            # 前向传播
            outputs = model(input_ids)
            loss = criterion(outputs.logits.view(-1, vocab_size), 
                           labels.view(-1))
            
            # MoE 辅助损失
            if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                loss = loss + outputs.aux_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), save_path)
```

---

## 数据处理

### 数据集设计

训练脚本提供了一个简单的文本数据集示例：

```python
class SimpleTextDataset(Dataset):
    """简单的文本数据集"""
    def __init__(self, data, max_length=512):
        self.data = data  # 文本列表
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # 简单的 tokenization (示例)
        tokens = [ord(c) % 6400 for c in text[:self.max_length]]
        
        # 填充到固定长度
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        # 创建输入和标签
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, labels
```

### Tokenization 详解

#### 为什么需要 Tokenization？

神经网络不能直接处理文本，需要将文本转换为数字表示。Tokenization 是将文本分割成更小单元（tokens）的过程。

**Token 的类型**：
1. **字符级**（Character-level）
   - 每个字符是一个 token
   - 词表小（通常 < 256）
   - 序列长，计算量大
   - 示例代码使用的方法

2. **词级**（Word-level）
   - 每个单词是一个 token
   - 词表大（通常 > 50k）
   - 未登录词问题

3. **子词级**（Subword-level）
   - BPE、WordPiece、SentencePiece
   - 平衡词表大小和序列长度
   - 现代 LLM 的标准选择

#### 示例代码中的简单 Tokenization

```python
tokens = [ord(c) % 6400 for c in text[:self.max_length]]
```

**步骤解析**：
1. `ord(c)`: 获取字符的 Unicode 码点
   - 'A' → 65
   - '中' → 20013

2. `% 6400`: 取模映射到词表范围
   - 确保 token ID 在 [0, 6399] 范围内
   - 简单但会有冲突（不同字符映射到相同 ID）

**实际使用建议**：
```python
from transformers import AutoTokenizer

# 使用专业的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

def tokenize_function(text):
    return tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
```

### 因果语言建模的数据格式

因果语言模型（Causal Language Model）的训练任务是**下一个 token 预测**。

**输入和标签的关系**：
```
文本: "Hello world"
Token IDs: [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]

输入 (input_ids):  [72, 101, 108, 108, 111, 32, 119, 111, 114, 108]
标签 (labels):     [101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
```

**解释**：
- 模型看到 `72` (H)，预测 `101` (e)
- 模型看到 `72, 101` (He)，预测 `108` (l)
- 模型看到 `72, 101, 108` (Hel)，预测 `108` (l)
- 以此类推...

**代码实现**：
```python
input_ids = torch.tensor(tokens[:-1], dtype=torch.long)  # 去掉最后一个
labels = torch.tensor(tokens[1:], dtype=torch.long)      # 去掉第一个
```

### DataLoader 配置

```python
dataloader = DataLoader(
    dataset, 
    batch_size=args.batch_size,  # 批次大小
    shuffle=True,                # 随机打乱
    num_workers=args.num_workers # 并行加载
)
```

**参数说明**：

1. **batch_size**：
   - 每次训练使用的样本数
   - 影响梯度估计的准确性和内存占用
   - 常见值：4, 8, 16, 32
   - 显存限制：batch_size × seq_length × hidden_size

2. **shuffle**：
   - 每个 epoch 重新打乱数据
   - 防止模型记住数据顺序
   - 提高泛化能力

3. **num_workers**：
   - 数据加载的并行进程数
   - 0 = 主进程加载（简单但可能慢）
   - > 0 = 多进程并行加载
   - 建议值：CPU 核心数的 1/4 到 1/2

---

## 优化器与学习率

### AdamW 优化器

#### 背景知识

**优化器的演进**：
1. **SGD**（Stochastic Gradient Descent）
   - 最基础的优化器
   - 只使用当前梯度
   - 需要仔细调整学习率

2. **Momentum**
   - 加入动量，加速收敛
   - 减少震荡

3. **Adam**（Adaptive Moment Estimation）
   - 自适应学习率
   - 结合动量和 RMSprop
   - 广泛使用但有权重衰减问题

4. **AdamW**（Adam with decoupled Weight decay）
   - 修正 Adam 的权重衰减实现
   - 现代 Transformer 的标准选择

#### AdamW 原理

**核心思想**：将权重衰减从梯度更新中解耦。

**Adam 的权重衰减**（不正确）：
```
gradient = gradient + λ * weight  # 将权重衰减加到梯度上
weight = weight - lr * adam_update(gradient)
```

**AdamW 的权重衰减**（正确）：
```
weight = weight - lr * adam_update(gradient) - lr * λ * weight
```

**为什么 AdamW 更好？**
- 权重衰减不受自适应学习率影响
- 更稳定的正则化效果
- 在 Transformer 训练中表现更好

#### 配置参数

```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=args.learning_rate,      # 学习率
    weight_decay=args.weight_decay,  # 权重衰减
    betas=(0.9, 0.999),         # Adam 的 beta 参数（默认）
    eps=1e-8                    # 数值稳定性（默认）
)
```

**参数说明**：

1. **learning_rate**（学习率）
   - 控制参数更新的步长
   - **太大**：训练不稳定，损失震荡
   - **太小**：收敛太慢
   - 建议值：
     - 从零训练：1e-4 到 5e-4
     - 微调预训练模型：1e-5 到 5e-5

2. **weight_decay**（权重衰减）
   - L2 正则化，防止过拟合
   - 惩罚大的权重值
   - 建议值：0.01 到 0.1
   - 较小模型可以用更大的值

3. **betas**
   - `beta1`：一阶矩（梯度）的衰减率
   - `beta2`：二阶矩（梯度平方）的衰减率
   - 通常使用默认值 `(0.9, 0.999)`

### 学习率调度

虽然示例代码使用固定学习率，但实际训练中通常使用学习率调度器。

**常见的学习率策略**：

1. **线性预热 + 余弦衰减**（推荐）
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

# 预热阶段
warmup_steps = 1000
total_steps = len(dataloader) * epochs

def get_lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
```

2. **阶梯衰减**
```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=1000,  # 每 1000 步
    gamma=0.9        # 学习率乘以 0.9
)
```

3. **指数衰减**
```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, 
    gamma=0.95
)
```

**使用方法**：
```python
for epoch in range(epochs):
    for batch in dataloader:
        # 训练步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        scheduler.step()
```

---

## 训练技巧

### 梯度裁剪

#### 为什么需要梯度裁剪？

在训练深度神经网络时，可能遇到**梯度爆炸**问题：
- 梯度值变得非常大
- 参数更新幅度过大
- 训练不稳定，损失变为 NaN

**原因**：
- 深层网络中梯度的连乘效应
- 某些特殊的输入导致异常大的梯度
- 学习率设置不当

#### 梯度裁剪原理

**梯度范数裁剪**（Gradient Norm Clipping）：
```
如果 ||gradient|| > threshold:
    gradient = gradient * (threshold / ||gradient||)
```

**效果**：
- 保持梯度方向不变
- 限制梯度的幅度
- 防止参数更新过大

#### 实现

```python
# 反向传播
loss.backward()

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(
    model.parameters(),  # 要裁剪的参数
    args.grad_clip       # 阈值（通常是 1.0）
)

# 参数更新
optimizer.step()
```

**参数选择**：
- **1.0**：常用的默认值
- **0.5**：更保守，适合不稳定的训练
- **5.0**：更宽松，适合稳定的训练

### 混合精度训练

虽然示例代码没有使用，但混合精度训练可以显著加速训练。

#### 原理

**传统训练**：所有计算都使用 FP32（32位浮点数）

**混合精度训练**：
- 前向传播：FP16（16位浮点数）
- 权重存储：FP32
- 梯度累积：FP32
- 参数更新：FP32

**优势**：
- 2x 加速（理论）
- 减少显存占用（约 50%）
- 在现代 GPU（V100, A100）上效果显著

#### 实现（PyTorch）

```python
from torch.cuda.amp import autocast, GradScaler

# 创建梯度缩放器
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # 混合精度前向传播
    with autocast():
        outputs = model(input_ids)
        loss = criterion(outputs.logits.view(-1, vocab_size), 
                        labels.view(-1))
    
    # 缩放损失，反向传播
    scaler.scale(loss).backward()
    
    # 梯度裁剪（注意要unscale）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # 更新参数
    scaler.step(optimizer)
    scaler.update()
```

### 梯度累积

当显存不足以支持大的 batch size 时，可以使用梯度累积。

#### 原理

**目标**：模拟大的 batch size
**方法**：累积多个小 batch 的梯度，再更新参数

```python
accumulation_steps = 4  # 累积 4 个 batch

optimizer.zero_grad()
for i, (input_ids, labels) in enumerate(dataloader):
    # 前向传播
    outputs = model(input_ids)
    loss = criterion(outputs.logits.view(-1, vocab_size), 
                    labels.view(-1))
    
    # 归一化损失
    loss = loss / accumulation_steps
    
    # 反向传播（累积梯度）
    loss.backward()
    
    # 每 accumulation_steps 步更新一次
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

**等效关系**：
```
梯度累积 4 步，batch_size=2 
≈ 
不累积，batch_size=8
```

### 检查点保存

#### 保存策略

```python
# 定期保存（每 N 个 epoch）
if (epoch + 1) % args.save_interval == 0:
    save_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)

# 保存最终模型
final_path = os.path.join(args.output_dir, "model_final.pth")
torch.save(model.state_dict(), final_path)
```

#### 保存完整训练状态

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'config': config.__dict__
}
torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")
```

#### 加载检查点

```python
# 只加载模型权重
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict, strict=False)

# 加载完整训练状态
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 损失函数

### 交叉熵损失

语言模型训练使用**交叉熵损失**（Cross-Entropy Loss）。

#### 原理

**任务**：给定上下文，预测下一个 token

**公式**：
```
Loss = -Σ log P(target_token | context)
```

对于词表大小为 V 的模型：
```
P(token_i | context) = softmax(logits)_i 
                     = exp(logits_i) / Σ exp(logits_j)
```

**示例**：
```
真实 token: "cat" (ID = 10)
模型输出 logits: [0.1, 0.3, ..., 2.5, ..., 0.2]  # 第 10 个是 2.5
                                         ↑
                                     ID=10
                                     
概率: P_10 = exp(2.5) / Σ exp(logits_j) = 0.6
损失: -log(0.6) = 0.51
```

**目标**：最大化正确 token 的概率 = 最小化交叉熵

#### 实现

```python
criterion = nn.CrossEntropyLoss()

# 前向传播
outputs = model(input_ids)
logits = outputs.logits  # [batch, seq_len, vocab_size]

# 计算损失
loss = criterion(
    logits.view(-1, vocab_size),  # [batch * seq_len, vocab_size]
    labels.view(-1)               # [batch * seq_len]
)
```

**关键点**：
1. **重塑张量**：
   - `logits.view(-1, vocab_size)` 将 3D 张量变为 2D
   - `labels.view(-1)` 将 2D 张量变为 1D
   - CrossEntropyLoss 需要 2D logits 和 1D targets

2. **忽略填充**：
```python
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
```
- 不计算填充位置的损失
- 只关注实际的 token

### MoE 辅助损失

当使用 MoE 架构时，需要添加辅助损失。

#### 为什么需要辅助损失？

**问题**：负载不均衡
- 模型可能只使用少数几个专家
- 其他专家得不到训练
- 降低了 MoE 的优势

**解决方案**：辅助损失
- 鼓励均匀使用所有专家
- 平衡负载

#### 计算

```python
# 主损失
main_loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

# MoE 辅助损失
aux_loss = outputs.aux_loss  # 模型自动计算

# 总损失
total_loss = main_loss + aux_loss
```

**辅助损失公式**（在 MoEGate 中实现）：
```
aux_loss = α * Σ(P_i * f_i)

其中：
- P_i: 专家 i 被选中的平均概率
- f_i: 专家 i 实际被使用的次数（归一化）
- α: 辅助损失权重（通常 0.01 - 0.1）
```

**理想情况**：
- 每个专家被使用的概率和次数都相等
- `P_i ≈ 1/N` 且 `f_i ≈ 1/N`（N 是专家总数）
- 辅助损失接近 0

---

## 推理与生成

虽然训练脚本主要关注训练，但了解推理过程有助于理解模型。

### 自回归生成

语言模型使用**自回归生成**（Autoregressive Generation）。

#### 过程

1. **输入提示**：`"Once upon a time"`
2. **编码**：`[15, 234, 567, 12]`
3. **生成循环**：
   ```
   输入: [15, 234, 567, 12]
   → 预测下一个 token: 789 ("there")
   
   输入: [15, 234, 567, 12, 789]
   → 预测下一个 token: 456 ("was")
   
   输入: [15, 234, 567, 12, 789, 456]
   → 预测下一个 token: ...
   ```
4. **停止条件**：
   - 生成 EOS token
   - 达到最大长度

#### 实现

```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens=50, temperature=1.0):
    model.eval()
    
    for _ in range(max_new_tokens):
        # 前向传播
        outputs = model(input_ids)
        logits = outputs.logits
        
        # 只取最后一个位置的 logits
        next_token_logits = logits[:, -1, :] / temperature
        
        # 采样
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 拼接
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # 检查是否生成了 EOS
        if next_token.item() == eos_token_id:
            break
    
    return input_ids
```

### KV Cache 加速

在自回归生成中，每次只生成一个新 token，但需要重新计算所有之前 token 的注意力。

#### 问题

```
第 1 步: 输入 [A] → 输出 B
第 2 步: 输入 [A, B] → 计算 A 和 B 的 K, V → 输出 C
第 3 步: 输入 [A, B, C] → 重新计算 A, B, C 的 K, V → 输出 D
```

**浪费**：A 和 B 的 K, V 在第 3 步被重新计算，但其实可以复用第 2 步的结果。

#### 解决方案：KV Cache

**思路**：缓存已计算的 K 和 V

```
第 1 步: 输入 [A] → 计算 K_A, V_A → 缓存 → 输出 B
第 2 步: 输入 [B] → 计算 K_B, V_B → 拼接缓存 [K_A, K_B], [V_A, V_B] → 输出 C
第 3 步: 输入 [C] → 计算 K_C, V_C → 拼接缓存 [K_A, K_B, K_C], [V_A, V_B, V_C] → 输出 D
```

**优势**：
- 只需计算新 token 的 K, V
- 大幅减少计算量（~10x 加速）

#### 实现

```python
@torch.no_grad()
def generate_with_cache(model, input_ids, max_new_tokens=50):
    past_key_values = None
    
    for _ in range(max_new_tokens):
        # 第一次：处理完整输入
        # 之后：只处理新 token
        if past_key_values is None:
            model_inputs = input_ids
        else:
            model_inputs = input_ids[:, -1:]
        
        # 前向传播，使用和更新 cache
        outputs = model(
            model_inputs,
            past_key_values=past_key_values,
            use_cache=True
        )
        
        past_key_values = outputs.past_key_values
        
        # 采样下一个 token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return input_ids
```

### 采样策略

生成 token 时有多种采样策略。

#### 1. Greedy Decoding（贪婪解码）

**方法**：总是选择概率最高的 token

```python
next_token = torch.argmax(logits, dim=-1)
```

**优点**：确定性，可复现
**缺点**：可能重复，缺乏多样性

#### 2. Temperature Sampling（温度采样）

**方法**：调整概率分布的"平滑度"

```python
logits = logits / temperature
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

**Temperature 的作用**：
- **temperature < 1**：概率分布更"尖锐"，更确定
  ```
  原始: [0.5, 0.3, 0.2]
  T=0.5: [0.7, 0.2, 0.1]  # 高概率的更高
  ```

- **temperature = 1**：不改变分布

- **temperature > 1**：概率分布更"平滑"，更随机
  ```
  原始: [0.5, 0.3, 0.2]
  T=2.0: [0.4, 0.35, 0.25]  # 更平均
  ```

#### 3. Top-k Sampling

**方法**：只考虑概率最高的 k 个 token

```python
top_k = 50
top_k_logits, top_k_indices = torch.topk(logits, top_k)
probs = F.softmax(top_k_logits, dim=-1)
next_token_idx = torch.multinomial(probs, num_samples=1)
next_token = top_k_indices.gather(-1, next_token_idx)
```

**效果**：过滤低概率的噪声 token

#### 4. Top-p (Nucleus) Sampling

**方法**：选择累积概率达到 p 的最小 token 集合

```python
top_p = 0.9
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

# 找到累积概率超过 top_p 的位置
sorted_indices_to_remove = cumulative_probs > top_p
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
sorted_indices_to_remove[..., 0] = 0

indices_to_remove = sorted_indices[sorted_indices_to_remove]
logits[indices_to_remove] = -float('Inf')

probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

**优势**：动态调整候选集大小，比 top-k 更灵活

---

## 训练监控与调试

### 日志记录

```python
# 定期打印训练信息
if (batch_idx + 1) % args.log_interval == 0:
    avg_loss = total_loss / (batch_idx + 1)
    print(f"Epoch [{epoch+1}/{args.epochs}], "
          f"Step [{batch_idx+1}/{len(dataloader)}], "
          f"Loss: {loss.item():.4f}, "
          f"Avg Loss: {avg_loss:.4f}")
```

### 使用 TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./runs')

for epoch in range(epochs):
    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        # ... 训练步骤 ...
        
        global_step = epoch * len(dataloader) + batch_idx
        
        # 记录损失
        writer.add_scalar('Loss/train', loss.item(), global_step)
        
        # 记录学习率
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)
        
        # 记录梯度范数
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar('Gradient/norm', total_norm, global_step)

writer.close()
```

### 常见问题与解决方案

1. **损失变为 NaN**
   - **原因**：梯度爆炸、学习率过大
   - **解决**：降低学习率、使用梯度裁剪、检查数据

2. **损失不下降**
   - **原因**：学习率过小、数据问题、模型容量不足
   - **解决**：增加学习率、检查数据、增加模型大小

3. **过拟合**
   - **原因**：模型太大、数据太少
   - **解决**：增加 dropout、权重衰减、数据增强

4. **训练太慢**
   - **解决**：使用混合精度、增加 batch size、使用梯度累积

5. **显存不足**
   - **解决**：减小 batch size、减小模型、使用梯度累积、使用梯度检查点

---

## 总结

MiniMind 的训练脚本提供了一个简洁但完整的训练流程，涵盖了：

1. **数据处理**：Tokenization 和 DataLoader
2. **模型初始化**：配置和预训练权重加载
3. **优化器**：AdamW 和学习率调度
4. **训练循环**：前向、反向、优化
5. **训练技巧**：梯度裁剪、混合精度、梯度累积
6. **损失函数**：交叉熵和 MoE 辅助损失
7. **推理生成**：自回归生成和 KV Cache

这些组件结合在一起，构成了现代语言模型训练的标准流程。理解这些技术细节对于成功训练自己的语言模型至关重要。
