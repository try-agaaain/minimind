# SFT 训练技术深度解析：从 LoRA 到全参数微调

当你拿到一个预训练语言模型，想让它学会回答问题、遵循指令或者掌握特定领域知识时，监督微调（Supervised Fine-Tuning，SFT）是你最可能采用的技术路线。然而，"微调"这个词看似简单，背后却隐藏着参数效率、训练稳定性、泛化能力等多重考量。

本文将以 MiniMind 项目中的 `my_train_lora.py` 为切入点，深入剖析现代 SFT 训练的核心技术。我们不仅要理解"如何实现"，更要探讨"为什么这样设计"——在参数规模与计算资源的博弈中，工程师们做出了哪些精妙的权衡？

## 目录

1. [SFT 训练的本质与演进](#sft-训练的本质与演进)
2. [LoRA：参数效率的优雅解法](#lora参数效率的优雅解法)
3. [训练流水线的工程实践](#训练流水线的工程实践)
4. [优化器与学习率策略](#优化器与学习率策略)
5. [混合精度与梯度累积](#混合精度与梯度累积)
6. [分布式训练：从单卡到多卡](#分布式训练从单卡到多卡)
7. [实践中的关键细节](#实践中的关键细节)
8. [总结与展望](#总结与展望)

---

## SFT 训练的本质与演进

### 从预训练到微调：范式的转变

大语言模型的训练遵循一个优雅的范式：**先在海量数据上预训练，再在特定任务上微调**。这个范式的成功源于一个深刻的洞察——语言的统计规律是可迁移的。

预训练阶段，模型在万亿 token 的语料上学习语言的通用表示：词汇的语义、句法的结构、上下文的依赖。这个过程需要消耗巨大的计算资源——GPT-3 的训练据估计花费了数百万美元。

微调阶段，模型在相对小规模的标注数据上学习特定任务的模式：如何回答问题、如何遵循指令、如何生成代码。关键在于，预训练获得的通用知识成为了微调的起点，使得模型能够快速适应新任务，而无需从零学起。

这种"预训练-微调"范式的效率令人惊叹。一个在通用语料上训练的模型，只需几千条对话数据的微调，就能学会以特定风格回复用户。但效率的另一面是挑战：如何在保留预训练知识的同时，有效学习新任务？如何在有限的计算资源下完成微调？

### 全参数微调的困境

最直接的微调方式是**全参数微调**（Full Fine-Tuning）：更新模型的所有参数，让损失函数逐步下降。MiniMind 的 `my_train_sft.py` 正是这种方式的实现：

```python
# 全参数微调：优化所有参数
self.optimizer = optim.AdamW(
    model.parameters(),  # 所有参数都参与优化
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)
```

全参数微调的优势显而易见：
- **表达能力强**：所有参数都可以调整，理论上能学到任意复杂的模式
- **实现简单**：不需要额外的模块设计
- **效果上限高**：在数据充足时通常能达到最佳效果

然而，当模型规模增大时，全参数微调面临严峻的挑战：

**挑战一：显存瓶颈**

以 AdamW 优化器为例，它需要为每个参数维护两个状态（一阶矩和二阶矩）。在 FP32 精度下，优化器状态的显存占用是参数本身的 2 倍：

```
显存占用 ≈ 模型参数 (4B) + 梯度 (4B) + 优化器状态 (8B) = 16 × 参数量
```

对于一个 7B 参数的模型，这意味着约 112GB 的显存——远超单卡 GPU 的容量。

**挑战二：灾难性遗忘**

全参数微调可能导致模型"遗忘"预训练阶段学到的知识。当微调数据与预训练数据分布差异较大时，模型可能过度拟合微调数据，丧失通用能力。

**挑战三：存储成本**

每个微调后的模型都是完整的参数拷贝。如果你需要为不同任务训练多个版本，存储成本将线性增长。

这些挑战催生了**参数高效微调**（Parameter-Efficient Fine-Tuning，PEFT）的研究方向。其中，LoRA 凭借其简洁优雅的设计，成为了当前最流行的 PEFT 方法之一。

---

## LoRA：参数效率的优雅解法

### 低秩假设：简洁背后的深刻洞察

LoRA（Low-Rank Adaptation）的核心思想源于一个大胆的假设：**模型在微调过程中的参数更新是低秩的**。

这是什么意思？让我们从线性代数的角度理解。一个 $d \times d$ 的权重矩阵 $W$，在微调后变为 $W' = W + \Delta W$。如果 $\Delta W$ 是低秩的，意味着它可以分解为两个小矩阵的乘积：

$$\Delta W = B \cdot A$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$，$r \ll d$。

这个分解的参数量从 $d^2$ 降低到 $2dr$。当 $r = 8$，$d = 4096$ 时，参数量减少了 **256 倍**！

为什么这个假设是合理的？研究者们发现，预训练模型已经学到了丰富的语言表示，微调只需要在特定方向上做"微调整"。这些调整往往集中在少数几个主方向上，因此用低秩矩阵来近似是可行的。

从数学上看，任何矩阵都可以通过 SVD（奇异值分解）分解为若干秩-1 矩阵的和。如果 $\Delta W$ 的奇异值快速衰减，保留前 $r$ 个最大奇异值对应的成分就能很好地近似原矩阵。

### MiniMind 中的 LoRA 实现

让我们深入 `my_lora.py` 中的实现，理解 LoRA 的工程细节：

```python
class LoRA(nn.Module):
    """
    LoRA 低秩适配器模块
    
    通过两个低秩矩阵的乘积来近似权重更新:
    W' = W + BA，其中 B ∈ R^(out_features × rank)，A ∈ R^(rank × in_features)
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 关键：缩放因子
        
        # 低秩矩阵 A：高斯初始化
        self.A = nn.Linear(in_features, rank, bias=False)
        # 低秩矩阵 B：零初始化（确保训练开始时 LoRA 输出为 0）
        self.B = nn.Linear(rank, out_features, bias=False)
        
        # 初始化策略
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x)) * self.scaling
```

**初始化策略的精妙设计**

注意 LoRA 的初始化：$A$ 用高斯分布初始化，而 $B$ 初始化为零。这意味着训练开始时：

$$\text{LoRA}(x) = B \cdot A \cdot x = 0$$

这个设计有两个重要意义：

1. **无损启动**：训练开始时，带 LoRA 的模型行为与原始模型完全一致，不会因为随机初始化破坏预训练知识
2. **渐进适应**：随着训练进行，LoRA 逐渐学习需要的调整，平滑地从原始模型过渡到微调模型

如果 $A$ 和 $B$ 都随机初始化，训练开始时模型输出会被随机扰动，可能导致训练不稳定。

**缩放因子的作用**

代码中的 `scaling = alpha / rank` 是另一个关键设计。当增大 rank 时，LoRA 的输出幅度会增加（因为更多的参数贡献了输出）。为了保持不同 rank 下的训练动态一致，引入 $\alpha$ 作为缩放超参数。

经验上，设置 $\alpha = 2 \times \text{rank}$ 是一个常用的选择，相当于 `scaling = 2`。

### 将 LoRA 应用到模型

`apply_lora` 函数展示了如何将 LoRA 模块注入到现有模型中：

```python
def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 1.0, target_modules: list = None):
    """
    对模型应用 LoRA
    
    默认只对方形线性层（如 attention 的 q, k, v, o 投影）应用 LoRA，
    可以通过 target_modules 指定目标模块名称。
    """
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_apply = False
            
            if target_modules is not None:
                # 检查模块名称是否匹配
                for target in target_modules:
                    if target in name:
                        should_apply = True
                        break
            else:
                # 默认只对方形权重应用（通常是 attention 层）
                if module.weight.shape[0] == module.weight.shape[1]:
                    should_apply = True
            
            if should_apply:
                lora = LoRA(
                    in_features=module.weight.shape[1],
                    out_features=module.weight.shape[0],
                    rank=rank,
                    alpha=alpha
                ).to(device)
                
                setattr(module, "lora", lora)
                original_forward = module.forward
                
                # 使用闭包工厂函数来正确捕获变量
                def make_forward_with_lora(orig_forward, lora_module):
                    def forward_with_lora(x):
                        return orig_forward(x) + lora_module(x)
                    return forward_with_lora
                
                module.forward = make_forward_with_lora(original_forward, lora)
```

**为什么选择 Attention 层？**

代码默认只对"方形"线性层应用 LoRA。在 Transformer 中，这对应于 Attention 层的 Q、K、V、O 投影。这个选择基于经验观察：

1. **Attention 是知识的关键载体**：Attention 机制决定了模型"关注什么"，对任务适应至关重要
2. **参数效率**：只修改 Attention 层，而保留 FFN 层不变，可以进一步减少可训练参数
3. **稳定性**：Attention 层的权重通常更加结构化，低秩假设更容易成立

当然，你也可以通过 `target_modules` 参数指定其他层。有研究表明，同时在 FFN 层应用 LoRA 可以进一步提升效果，但参数量也会相应增加。

**闭包工厂函数的技巧**

代码中使用 `make_forward_with_lora` 工厂函数来创建新的 forward 方法，而不是直接在循环中定义。这是 Python 闭包的一个经典陷阱：

```python
# 错误示例：所有 module.forward 都会引用最后一个 lora
for module in modules:
    lora = create_lora()
    module.forward = lambda x: original_forward(x) + lora(x)  # lora 是循环变量！

# 正确做法：通过工厂函数捕获当前的 lora
def make_forward(orig, l):
    return lambda x: orig(x) + l(x)  # l 是参数，在函数定义时绑定
```

### 冻结与解冻：控制训练的精细粒度

LoRA 的另一个核心操作是冻结非 LoRA 参数：

```python
def freeze_non_lora_params(model: nn.Module):
    """冻结非 LoRA 参数"""
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
```

这确保了只有 LoRA 的 $A$ 和 $B$ 矩阵参与梯度更新，原始模型权重保持不变。

在 `my_train_lora.py` 中，训练流程明确体现了这一设计：

```python
# 应用 LoRA
apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)

# 冻结非 LoRA 参数
freeze_non_lora_params(model)
lora_params = get_lora_params(model)

# 只优化 LoRA 参数
self.optimizer = optim.AdamW(
    lora_params,  # 注意：只传入 LoRA 参数
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)
```

**参数统计**

让我们看一下 LoRA 的参数效率：

```python
param_stats = count_lora_params(model)
Logger(f"总参数量: {param_stats['total_params'] / 1e6:.3f} M")
Logger(f"LoRA 参数量: {param_stats['lora_params'] / 1e6:.3f} M")
Logger(f"LoRA 参数占比: {param_stats['lora_ratio'] * 100:.2f}%")
```

对于一个 512 hidden size、8 层的 MiniMind 模型，rank=8 的 LoRA 可训练参数通常只占总参数的 0.5%~2%。这种极致的参数效率使得在消费级 GPU 上微调大模型成为可能。

### LoRA 权重的保存与合并

LoRA 的一个实用优势是权重可以独立保存和加载：

```python
def save_lora(model: nn.Module, path: str):
    """保存 LoRA 权重到文件"""
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {
                f'{name}.lora.{k}': v 
                for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

这意味着你可以：
1. **保持基座模型不变**，为不同任务训练不同的 LoRA 权重
2. **快速切换任务**，只需加载对应的 LoRA 权重
3. **节省存储空间**，LoRA 权重通常只有几十 MB

更进一步，训练完成后可以将 LoRA 权重**合并**到原始权重中：

```python
def merge_lora(model: nn.Module):
    """将 LoRA 权重合并到原始权重中"""
    for name, module in model.named_modules():
        if hasattr(module, 'lora') and isinstance(module, nn.Linear):
            lora = module.lora
            # 计算合并后的权重: W' = W + scaling * B @ A
            delta_weight = lora.scaling * lora.B.weight @ lora.A.weight
            module.weight.data += delta_weight
            
            # 移除 LoRA
            delattr(module, 'lora')
            module.forward = nn.Linear.forward.__get__(module, nn.Linear)
```

合并后的模型与全参数微调的模型在数学上等价，但推理时不再有额外的计算开销。

---

## 训练流水线的工程实践

理解了 LoRA 的原理，让我们深入训练流水线的工程细节。`my_train_lora.py` 中的 `LoRATrainer` 类封装了完整的训练逻辑，其设计体现了工业级代码的最佳实践。

### 数据流：从原始文本到模型输入

训练数据的处理是 SFT 的起点。在 MiniMind 中，`MinimindDataset` 负责将 JSONL 格式的数据转换为模型可接受的张量：

```python
def __getitem__(self, idx):
    token_ids = self.tokenizer.encode(self.data[idx]["text"])
    
    # 截断或填充到 max_seq_len
    if len(token_ids) > self.max_seq_len:
        token_ids = token_ids[:self.max_seq_len]
    else:
        token_ids = token_ids + [0] * (self.max_seq_len - len(token_ids))
    
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    
    # 构造输入、标签和掩码
    input_ids = torch.concat([torch.tensor([0]), token_ids_tensor[:-1]])
    labels = token_ids_tensor
    loss_mask = (labels != 0).long()
    
    return input_ids, labels, loss_mask
```

这段代码体现了因果语言模型的核心设计：

**自回归建模**：输入和标签错位一个位置。模型根据 `input_ids[i]` 预测 `labels[i]`，即根据前 $i$ 个 token 预测第 $i+1$ 个 token。

**掩码机制**：`loss_mask` 标记了有效 token 的位置。填充位置（值为 0）的损失不参与反向传播，确保模型只学习真实的语言模式。

### 训练循环：精细控制每一步

`LoRATrainer.train_epoch` 方法实现了训练循环的核心逻辑：

```python
def train_epoch(self, epoch, total_epochs, wandb=None):
    self.model.train()
    iters = len(self.dataloader)
    total_iters = total_epochs * iters
    
    for step, (X, Y, loss_mask) in enumerate(data_iterator, start=1):
        X = X.to(self.device)
        Y = Y.to(self.device)
        loss_mask = loss_mask.to(self.device)
        
        # 动态调整学习率
        current_step = epoch * iters + step
        lr = get_lr(current_step, total_iters, self.args.learning_rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        with self.autocast_ctx:
            outputs = self.model(X)
            loss = self.loss_fct(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            
            # 应用掩码：只计算有效 token 的损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # 处理 MoE 辅助损失
            if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                loss = loss + outputs.aux_loss
            
            loss = loss / self.args.accumulation_steps
        
        self.scaler.scale(loss).backward()
        
        if (step + 1) % self.args.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.lora_params, self.args.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
```

让我们逐一剖析这段代码的关键设计。

**损失计算与掩码**

```python
loss = self.loss_fct(
    outputs.logits.view(-1, outputs.logits.size(-1)),
    Y.view(-1)
).view(Y.size())
loss = (loss * loss_mask).sum() / loss_mask.sum()
```

这里使用 `reduction='none'` 的交叉熵损失，得到每个位置的损失值，再通过掩码加权平均。相比直接使用 `ignore_index`，这种方式提供了更大的灵活性——你可以实现自定义的损失加权策略，比如对关键词给予更高权重。

**为什么除以 `loss_mask.sum()` 而非 `batch_size * seq_len`？**

这是一个微妙但重要的设计决策。考虑两个 batch：
- Batch A：1024 个 token，50 个 padding，有效 token 974 个
- Batch B：1024 个 token，500 个 padding，有效 token 524 个

如果除以总长度，Batch B 的平均损失会被"稀释"——同样的总损失，分母更大意味着平均更小。但这与有效 token 的数量无关，会导致训练动态不稳定。

除以有效 token 数，确保损失反映的是"每个有效 token 的平均损失"，使得不同 batch 的损失具有可比性。

---

## 优化器与学习率策略

### AdamW：解耦的权重衰减

MiniMind 使用 AdamW 作为优化器，这是现代深度学习的标准选择：

```python
self.optimizer = optim.AdamW(
    lora_params,
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)
```

AdamW 与原始 Adam 的区别在于权重衰减（weight decay）的实现方式。原始 Adam 将权重衰减作为 L2 正则化加入损失函数：

$$\mathcal{L}_{total} = \mathcal{L} + \lambda \|W\|^2$$

这导致权重衰减项也被 Adam 的自适应学习率缩放，与原本的设计意图不符。

AdamW 将权重衰减与梯度更新解耦：

$$W_{t+1} = W_t - \eta \cdot \text{Adam}(\nabla \mathcal{L}) - \eta \lambda W_t$$

权重衰减直接作用于参数，不受自适应学习率的影响。在实践中，AdamW 通常比 Adam 更稳定，尤其是在 Transformer 训练中。

### 余弦退火：优雅的学习率调度

学习率的动态调整是训练成功的关键。`get_lr` 函数实现了带有预热的余弦退火策略：

```python
def get_lr(current_step, total_steps, learning_rate, warmup_iters=100, min_lr=0.0):
    """Cosine learning rate schedule with warmup"""
    # 预热阶段：线性增长
    if current_step < warmup_iters:
        return learning_rate * current_step / warmup_iters
    
    # 超过总步数：使用最小学习率
    if current_step > total_steps:
        return min_lr
    
    # 余弦退火阶段
    decay_ratio = (current_step - warmup_iters) / (total_steps - warmup_iters)
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * 3.14159)))
    return min_lr + coeff * (learning_rate - min_lr)
```

**预热的必要性**

在训练初期，模型参数处于随机状态（或 LoRA 的零初始化状态），梯度的方向和幅度都不稳定。如果直接使用大学习率，可能导致参数更新过于激进，模型"跑飞"。

预热阶段让学习率从 0 逐渐增长到目标值，给模型一个"热身"的机会：

```
Step 0:   lr = 0
Step 50:  lr = 0.5 * target_lr
Step 100: lr = target_lr（预热完成）
```

**余弦退火的优势**

预热结束后，学习率按照余弦曲线衰减：

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$$

相比线性或阶梯衰减，余弦退火的特点是：
- **前期衰减慢**：在学习率较高时保持更长时间，充分探索参数空间
- **后期衰减快**：在接近收敛时快速降低学习率，做精细调整
- **平滑过渡**：没有突变点，训练曲线更加稳定

这个策略已经成为 LLM 训练的事实标准。

### 梯度裁剪：稳定训练的安全阀

深度神经网络容易出现梯度爆炸，尤其是在 Transformer 这种深层架构中。梯度裁剪是一个简单有效的防护措施：

```python
self.scaler.unscale_(self.optimizer)
torch.nn.utils.clip_grad_norm_(self.lora_params, self.args.grad_clip)
```

`clip_grad_norm_` 计算所有参数梯度的总范数，如果超过阈值则按比例缩放：

$$g' = g \cdot \frac{\text{max\_norm}}{\|g\|}$$

这保持了梯度的方向，只限制其幅度。常用的阈值是 1.0，意味着如果梯度范数超过 1，就将其缩放到 1。

**注意**：在混合精度训练中，必须先调用 `scaler.unscale_()` 将梯度恢复到原始尺度，再进行裁剪。否则你裁剪的是缩放后的梯度，阈值的含义就变了。

---

## 混合精度与梯度累积

### 混合精度：速度与精度的平衡

现代 GPU 对低精度计算有硬件加速。在 NVIDIA A100 上，FP16 运算的吞吐量是 FP32 的 8 倍。混合精度训练利用这一特性，在保持数值稳定的前提下大幅加速训练。

MiniMind 使用 PyTorch 原生的自动混合精度（AMP）：

```python
device_type = "cuda" if "cuda" in args.device else "cpu"
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

self.autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
    device_type=device_type, dtype=dtype
)
self.scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
```

**为什么需要 GradScaler？**

FP16 的表示范围有限（约 $6 \times 10^{-8}$ 到 $6.5 \times 10^4$）。当梯度很小时（如深层网络的低层），可能会下溢变成 0，导致训练停滞。

`GradScaler` 通过损失缩放（loss scaling）解决这个问题：
1. **前向传播**：在 FP16 下计算
2. **反向传播前**：将损失乘以一个大的缩放因子（如 2^16）
3. **反向传播**：梯度也被相应放大，避免下溢
4. **参数更新前**：将梯度除以缩放因子，恢复原始尺度

```python
# 缩放损失并反向传播
self.scaler.scale(loss).backward()

# 恢复梯度尺度
self.scaler.unscale_(self.optimizer)

# 检查是否有梯度溢出，决定是否更新参数
self.scaler.step(self.optimizer)

# 动态调整缩放因子
self.scaler.update()
```

`GradScaler` 还会自动调整缩放因子：如果检测到梯度溢出（出现 inf 或 nan），会跳过这次更新并降低缩放因子；如果连续多次没有溢出，会逐渐增大缩放因子。

**BFloat16 vs Float16**

代码支持两种低精度格式：
- **FP16**：更高的精度，但表示范围小，需要 loss scaling
- **BF16**：表示范围与 FP32 相同，但尾数精度低。优势是不需要 loss scaling

在较新的硬件（如 A100、H100）上，BF16 通常是更好的选择，因为它简化了训练流程且数值更稳定。

### 梯度累积：突破显存限制

当 batch size 受限于显存时，梯度累积提供了一种"虚拟增大 batch"的方法：

```python
loss = loss / self.args.accumulation_steps  # 归一化损失

self.scaler.scale(loss).backward()  # 累积梯度

if (step + 1) % self.args.accumulation_steps == 0:
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.lora_params, self.args.grad_clip)
    
    self.scaler.step(self.optimizer)
    self.scaler.update()
    
    self.optimizer.zero_grad(set_to_none=True)  # 清零梯度
```

**原理**

梯度累积的核心观察是：多个 mini-batch 的梯度求和，等价于一个大 batch 的梯度（在期望意义上）。

假设 `accumulation_steps=4`，`batch_size=8`：
- 每次前向传播处理 8 个样本
- 反向传播累积梯度到参数的 `.grad` 属性
- 每 4 步调用一次优化器更新

效果上，这与 `batch_size=32` 的单步更新等价。

**损失归一化的重要性**

注意代码中的 `loss = loss / accumulation_steps`。这是必须的，因为 PyTorch 的反向传播默认是累加梯度而非平均。如果不归一化，累积 4 步后的梯度是单步的 4 倍，相当于隐式增大了学习率。

**`set_to_none=True` 的优化**

`optimizer.zero_grad(set_to_none=True)` 将梯度设为 `None` 而非零张量。这有两个好处：
1. **节省显存**：不需要分配零张量
2. **避免复制**：下次反向传播时直接创建新的梯度张量

这是一个微小但值得注意的优化。

---

## 分布式训练：从单卡到多卡

当模型或数据规模增大时，单卡训练可能无法满足需求。MiniMind 支持使用 PyTorch 的 DistributedDataParallel（DDP）进行多卡训练。

### 初始化分布式环境

```python
def init_distributed_mode():
    """Initialize distributed training mode"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = -1
        world_size = -1
        local_rank = -1
        
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
    return local_rank
```

使用 `torchrun` 启动分布式训练时，它会自动设置环境变量：
- `RANK`：全局进程编号
- `LOCAL_RANK`：本机内的进程编号
- `WORLD_SIZE`：总进程数

### DDP 模型包装

```python
if dist.is_initialized():
    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
    model = DDP(model, device_ids=[local_rank])
```

DDP 的工作原理是：
1. 每个进程持有完整模型的副本
2. 数据被分片，每个进程处理不同的子集
3. 反向传播后，所有进程的梯度通过 AllReduce 同步
4. 每个进程独立更新参数（由于梯度相同，更新后的参数也相同）

**忽略特定参数**

`freqs_cos` 和 `freqs_sin` 是 RoPE（Rotary Position Embedding）的预计算缓冲区，不参与训练。将它们加入 `_ddp_params_and_buffers_to_ignore` 可以避免不必要的同步开销。

### 分布式数据采样

```python
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

dataloader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=(train_sampler is None),  # 使用 Sampler 时不能 shuffle
    sampler=train_sampler,
    num_workers=args.num_workers,
    pin_memory=True
)
```

`DistributedSampler` 确保每个进程处理不同的数据分片。比如 4 卡训练时：
- 进程 0 处理索引 [0, 4, 8, ...]
- 进程 1 处理索引 [1, 5, 9, ...]
- ...

需要注意的是，每个 epoch 开始时需要调用 `sampler.set_epoch(epoch)` 更新随机种子，否则每个 epoch 的数据顺序都相同。

### 主进程控制

某些操作只需在主进程执行，如日志打印和模型保存：

```python
def is_main_process():
    """Check if this is the main process in distributed training"""
    return not dist.is_initialized() or dist.get_rank() == 0

# 只在主进程打印日志
if is_main_process():
    print(f"Training loss: {loss.item()}")

# 只在主进程保存模型
if (step % self.args.save_interval == 0) and is_main_process():
    self._save_checkpoint(epoch, step)
```

这避免了多进程同时写文件造成的冲突，也减少了日志的重复输出。

---

## 实践中的关键细节

### 检查点的保存与恢复

训练中断是不可避免的——硬件故障、资源抢占、手动停止都可能发生。完善的检查点机制让你能够从中断处继续训练：

```python
def _save_checkpoint(self, epoch, step):
    """保存 LoRA 检查点"""
    output_dir = Path(self.args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 LoRA 权重
    lora_path = output_dir / f'{self.args.lora_name}_{self.base_model.config.hidden_size}.pth'
    save_lora(self.base_model, str(lora_path))
    
    # 保存完整检查点用于恢复训练
    checkpoint = {
        "optimizer_state": self.optimizer.state_dict(),
        "epoch": epoch,
        "step": step
    }
    torch.save(checkpoint, output_dir / "lora_checkpoint.pt")
```

检查点包含两部分：
1. **LoRA 权重**：模型的可训练参数
2. **训练状态**：优化器状态、当前 epoch、当前 step

恢复训练时，两者都需要加载：

```python
if args.resume_from_checkpoint:
    # 加载 LoRA 权重
    lora_path = Path(args.output_dir) / f'{args.lora_name}_{lm_config.hidden_size}.pth'
    if lora_path.exists():
        load_lora(trainer.base_model, str(lora_path))
    
    # 加载训练状态
    checkpoint_path = Path(args.output_dir) / "lora_checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
```

**为什么要恢复优化器状态？**

AdamW 优化器维护着每个参数的一阶矩和二阶矩估计。这些估计是随训练逐步积累的，直接影响参数更新的方向和幅度。

如果只加载模型权重而不恢复优化器状态，优化器的"记忆"被重置，训练曲线可能出现不连续——表现为损失突然上升或收敛速度变慢。

### 访问基础模型

在 DDP 包装后，模型被嵌套在 `DDP` 对象内部。访问原始模型需要通过 `.module` 属性：

```python
@property
def base_model(self):
    """返回基础模型实例"""
    if isinstance(self.model, DDP):
        return self.model.module
    return self.model
```

这个属性在保存模型时特别重要——你需要保存的是原始模型的参数，而非 DDP 包装后的版本。

### 随机性的控制

训练的可复现性对于调试和研究至关重要：

```python
def setup_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
```

在分布式训练中，每个进程需要不同但确定的随机种子：

```python
setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
```

这确保了：
1. **跨进程差异**：不同进程有不同的 dropout mask 和数据增强
2. **可复现性**：相同的 rank 在不同运行中产生相同的随机序列

### 显存管理

训练大模型时，显存管理是一个永恒的主题。MiniMind 采用了几个实用技巧：

**及时清理缓存**

```python
self.optimizer.zero_grad(set_to_none=True)
torch.cuda.empty_cache()
```

`empty_cache()` 释放 PyTorch 缓存的 CUDA 内存，但不影响已分配的张量。这可以减少显存碎片，让更多内存可用于下一个 batch。

**Pin Memory**

```python
dataloader = DataLoader(..., pin_memory=True)
```

`pin_memory=True` 将数据加载到锁页内存（pinned memory），可以加速 CPU 到 GPU 的数据传输。代价是消耗更多的 CPU 内存。

---

## 总结与展望

通过对 MiniMind 项目中 LoRA 训练代码的深入剖析，我们理解了现代 SFT 训练的核心技术：

### 技术要点回顾

**LoRA 的精妙设计**
- 低秩假设：用两个小矩阵的乘积近似参数更新
- 零初始化 B 矩阵：确保无损启动和渐进适应
- 缩放因子：保持不同 rank 下的训练动态一致
- 权重合并：训练完成后可无缝集成到原模型

**训练流水线**
- 掩码机制：精确控制哪些 token 参与损失计算
- 损失归一化：除以有效 token 数，保证损失可比性
- 动态学习率：预热 + 余弦退火的黄金组合

**优化技巧**
- AdamW：解耦的权重衰减
- 梯度裁剪：防止梯度爆炸
- 混合精度：速度与精度的平衡
- 梯度累积：突破显存限制

**分布式训练**
- DDP：高效的数据并行
- 分布式采样：确保数据不重复
- 主进程控制：避免冲突和冗余

### 未来方向

SFT 技术仍在快速发展，以下几个方向值得关注：

**更高效的 PEFT 方法**
- QLoRA：量化 + LoRA，进一步降低显存需求
- DoRA：分解权重更新的方向和幅度
- LoRA+：自适应调整不同参数组的学习率

**更好的训练策略**
- 课程学习：从简单样本到复杂样本
- 自适应 batch size：根据损失动态调整
- 多任务联合训练：共享基座，任务特定 LoRA

**与 RLHF 的结合**
- PPO/DPO 微调：从人类反馈中学习
- 奖励模型训练：评估生成质量
- Constitutional AI：自我约束与对齐

SFT 是让预训练模型"学会"特定任务的关键步骤。理解其技术细节，不仅有助于有效利用现有工具，更能为未来的创新奠定基础。希望本文的分析能够帮助读者在 LLM 微调的道路上走得更远、更稳。

---

**参考资源**

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
3. [PyTorch Distributed Training Guide](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
4. [MiniMind Project](https://github.com/try-agaaain/minimind)
