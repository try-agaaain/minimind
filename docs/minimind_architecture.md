# MiniMind 模型架构详解

## 目录

1. [整体架构概述](#整体架构概述)
2. [核心技术组件](#核心技术组件)
   - [RMSNorm 归一化](#rmsnorm-归一化)
   - [旋转位置编码 (RoPE)](#旋转位置编码-rope)
   - [分组查询注意力 (GQA)](#分组查询注意力-gqa)
   - [前馈神经网络 (FFN)](#前馈神经网络-ffn)
   - [混合专家模型 (MoE)](#混合专家模型-moe)
3. [完整模型结构](#完整模型结构)

---

## 整体架构概述

MiniMind 是一个轻量级的因果语言模型（Causal Language Model），采用了类似 GPT 的 Transformer Decoder-Only 架构。该模型的设计参考了现代大语言模型的最佳实践，包括：

- **Transformer Decoder 架构**：只使用解码器，没有编码器部分
- **因果注意力机制**：确保模型只能看到当前位置之前的信息
- **预归一化**：在每个子层之前进行归一化（Pre-normalization）
- **残差连接**：促进梯度流动和训练稳定性

### 模型层次结构

```
MiniMindForCausalLM
├── MiniMindModel
│   ├── Embedding Layer (词嵌入层)
│   ├── MiniMindBlock × N (Transformer层)
│   │   ├── RMSNorm (输入归一化)
│   │   ├── Attention (注意力机制)
│   │   ├── RMSNorm (FFN归一化)
│   │   └── FeedForward/MOEFeedForward (前馈网络)
│   └── RMSNorm (最终归一化)
└── LM Head (语言模型头)
```

---

## 核心技术组件

### RMSNorm 归一化

#### 技术背景

在深度神经网络中，归一化技术对于训练稳定性至关重要。传统的 LayerNorm 计算均值和方差，而 RMSNorm（Root Mean Square Normalization）是一种更简单高效的替代方案。

**发展历程**：
- **BatchNorm (2015)**：按批次归一化，在 CNN 中表现优异
- **LayerNorm (2016)**：按特征维度归一化，适用于序列模型
- **RMSNorm (2019)**：去除均值中心化，只保留缩放操作

#### 核心思想

RMSNorm 的关键观察是：LayerNorm 中的**重新中心化**（re-centering）操作可能不是必需的。只需要对特征进行**重新缩放**（re-scaling）即可获得类似的效果。

**LayerNorm 的计算**：
```
x_norm = (x - mean(x)) / sqrt(var(x) + eps)
output = gamma * x_norm + beta
```

**RMSNorm 的计算**：
```
rms = sqrt(mean(x^2) + eps)
output = gamma * (x / rms)
```

#### 实现细节

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数
    
    def _norm(self, x):
        # 计算 RMS: sqrt(mean(x^2))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # 先转换为 float32 进行计算，再转回原类型
        return self.weight * self._norm(x.float()).type_as(x)
```

**关键步骤解析**：

1. **计算平方的均值**：`x.pow(2).mean(-1, keepdim=True)`
   - 对最后一个维度（特征维度）计算平方的平均值
   - `keepdim=True` 保持维度用于广播

2. **计算倒数平方根**：`torch.rsqrt(...)`
   - `rsqrt(x) = 1/sqrt(x)`，比先开方再求倒数更高效
   - 加上 `eps` 防止除零

3. **归一化**：`x * torch.rsqrt(...)`
   - 将输入除以 RMS 值

4. **可学习缩放**：`self.weight * ...`
   - 允许模型学习每个特征的最佳缩放比例

**优势**：
- 计算量更小（无需计算均值）
- 内存占用更少
- 在 LLM 训练中效果与 LayerNorm 相当

---

### 旋转位置编码 (RoPE)

#### 技术背景

位置编码是 Transformer 模型的关键组件，因为自注意力机制本身是位置不变的（permutation-invariant）。

**位置编码的演进**：
1. **绝对位置编码**（Transformer, 2017）
   - 固定的正弦/余弦函数
   - 无法外推到更长序列

2. **可学习位置编码**（BERT, GPT）
   - 为每个位置学习独立的嵌入
   - 同样无法外推

3. **相对位置编码**（Transformer-XL, 2019）
   - 编码相对位置信息
   - 更好的外推能力

4. **旋转位置编码 RoPE**（RoFormer, 2021）
   - 通过旋转操作注入位置信息
   - 优秀的外推能力和相对位置建模

#### 核心思想

RoPE 的核心思想是将位置信息编码为**复平面上的旋转**。对于位置 `m` 的向量，通过旋转角度 `mθ` 来注入位置信息。

**数学原理**：

在复平面上，位置 `m` 对应的旋转矩阵：
```
R(m) = [cos(mθ)  -sin(mθ)]
       [sin(mθ)   cos(mθ)]
```

对于 d 维向量，我们将其分为 d/2 对，每对使用不同的频率 `θ_i`：
```
θ_i = 10000^(-2i/d), i = 0, 1, ..., d/2-1
```

**关键性质**：
- **相对位置**：位置 m 和 n 的点积只依赖于相对距离 (m-n)
- **长度不变**：旋转不改变向量长度
- **外推能力**：可以推广到训练时未见过的序列长度

#### 实现细节

**1. 预计算旋转频率**

```python
def precompute_freqs_cis(dim: int, end: int, rope_base: float = 1e6,
                        rope_scaling: Optional[dict] = None):
    # 计算每个维度对的频率: θ_i = base^(-2i/d)
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # YaRN 缩放（用于序列外推）
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 4)
        beta_fast = rope_scaling.get("beta_fast", 4.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        
        if end / orig_max > 1.0:
            # 找到临界维度
            corr_dim = next((i for i in range(dim // 2) 
                           if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            
            # 计算插值参数 β
            power = torch.arange(0, dim // 2).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            
            # YaRN 标准公式: λ = (β·α - β + 1)/(β·α)
            scale = torch.where(
                torch.arange(dim // 2) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1.0 / factor
            )
            freqs = freqs * scale
    
    # 生成位置索引
    t = torch.arange(end, device=freqs.device)
    
    # 计算 m·θ_i
    freqs = torch.outer(t, freqs).float()
    
    # 预计算 cos 和 sin，并复制以匹配维度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    
    return freqs_cos, freqs_sin
```

**关键点解析**：

1. **频率计算**：
   - `rope_base^(-2i/d)` 生成从高频到低频的序列
   - 低维度对应高频旋转，高维度对应低频旋转
   - 类似 Transformer 原始的位置编码

2. **YaRN 外推**：
   - 当序列长度超过训练长度时启用
   - 对不同维度使用不同的缩放策略
   - 高频部分缩放更多，低频部分缩放更少

3. **预计算优化**：
   - 预先计算所有位置的 cos 和 sin
   - 避免在前向传播中重复计算

**2. 应用旋转位置编码**

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        # 将向量的前半部分和后半部分交换并取负
        # [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
        return torch.cat((-x[..., x.shape[-1] // 2:], 
                         x[..., : x.shape[-1] // 2]), dim=-1)
    
    # 应用旋转公式:
    # q' = q * cos + rotate_half(q) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + \
              (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + \
              (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    
    return q_embed, k_embed
```

**旋转机制详解**：

对于向量 `[x1, x2, x3, x4]`，旋转操作相当于：
```
[x1, x2] 旋转 θ1 -> [x1*cos(θ1) - x2*sin(θ1), x1*sin(θ1) + x2*cos(θ1)]
[x3, x4] 旋转 θ2 -> [x3*cos(θ2) - x4*sin(θ2), x3*sin(θ2) + x4*cos(θ2)]
```

通过 `rotate_half` 实现这种配对旋转：
```
原始:        [x1, x2, x3, x4]
* cos:       [x1*cos, x2*cos, x3*cos, x4*cos]
rotate_half: [-x3, -x4, x1, x2]
* sin:       [-x3*sin, -x4*sin, x1*sin, x2*sin]
相加:        [x1*cos - x3*sin, x2*cos - x4*sin, 
             x3*cos + x1*sin, x4*cos + x2*sin]
```

---

### 分组查询注意力 (GQA)

#### 技术背景

注意力机制是 Transformer 的核心，但标准的多头注意力（MHA）在推理时有较高的内存开销。

**注意力机制的演进**：
1. **多头注意力 MHA**（Transformer, 2017）
   - 每个头都有独立的 Q, K, V 投影
   - 参数量大，KV cache 内存开销高

2. **多查询注意力 MQA**（Fast Transformer Decoding, 2019）
   - 所有查询头共享一组 K, V
   - 大幅减少 KV cache，但可能损失表达能力

3. **分组查询注意力 GQA**（LLaMA-2, 2023）
   - MHA 和 MQA 的折中方案
   - 将查询头分组，每组共享 K, V
   - 在性能和效率之间取得平衡

#### 核心思想

GQA 将多个查询头（Query heads）分组，每组共享一对键值头（Key-Value heads）。

**配置示例**：
```
num_attention_heads = 8      # 查询头数量
num_key_value_heads = 2      # 键值头数量
n_rep = 8 / 2 = 4           # 每个 KV 头对应 4 个 Q 头
```

**内存优势**：
- MHA: KV cache = batch × seq_len × num_heads × head_dim × 2
- GQA: KV cache = batch × seq_len × num_kv_heads × head_dim × 2
- 节省比例 = num_heads / num_kv_heads

#### 实现细节

```python
class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # 配置参数
        self.num_key_value_heads = args.num_attention_heads \
            if args.num_key_value_heads is None else args.num_key_value_heads
        
        # 确保可以整除
        assert args.num_attention_heads % self.num_key_value_heads == 0
        
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 重复次数
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        # 投影层
        self.q_proj = nn.Linear(args.hidden_size, 
                               args.num_attention_heads * self.head_dim, 
                               bias=False)
        self.k_proj = nn.Linear(args.hidden_size, 
                               self.num_key_value_heads * self.head_dim, 
                               bias=False)
        self.v_proj = nn.Linear(args.hidden_size, 
                               self.num_key_value_heads * self.head_dim, 
                               bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, 
                               args.hidden_size, 
                               bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        
        # Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') \
                    and args.flash_attn
```

**前向传播**：

```python
def forward(self, x, position_embeddings, past_key_value=None, 
           use_cache=False, attention_mask=None):
    bsz, seq_len, _ = x.shape
    
    # 1. 投影到 Q, K, V
    xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
    xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    
    # 2. 应用 RoPE
    cos, sin = position_embeddings
    xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
    
    # 3. KV Cache（用于推理加速）
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=1)
        xv = torch.cat([past_key_value[1], xv], dim=1)
    past_kv = (xk, xv) if use_cache else None
    
    # 4. 重复 KV 以匹配 Q 的头数
    xq = xq.transpose(1, 2)  # [bsz, n_heads, seq_len, head_dim]
    xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
    xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
    
    # 5. 计算注意力
    if self.flash:
        # 使用 Flash Attention（PyTorch 2.0+）
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
    else:
        # 标准注意力计算
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 因果掩码
        scores = scores + torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
            diagonal=1
        )
        
        # 注意力掩码
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            scores = scores + extended_attention_mask
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = scores @ xv
    
    # 6. 输出投影
    output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
    output = self.resid_dropout(self.o_proj(output))
    
    return output, past_kv
```

**KV 重复机制**：

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV 头重复以匹配 Q 头数量"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]  # 添加一个维度
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)  # 扩展
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)  # 重塑
    )
```

**关键步骤解析**：

1. **投影维度**：
   - Q: `[batch, seq, hidden]` → `[batch, seq, n_heads, head_dim]`
   - K/V: `[batch, seq, hidden]` → `[batch, seq, n_kv_heads, head_dim]`

2. **位置编码**：
   - 对 Q 和 K 应用 RoPE
   - V 不需要位置信息

3. **KV Cache**：
   - 推理时缓存历史的 K 和 V
   - 新 token 的 K, V 拼接到缓存中
   - 避免重复计算

4. **重复 KV**：
   - 将每个 KV 头复制 `n_rep` 次
   - 使其数量匹配 Q 头数量

5. **注意力计算**：
   - 标准的 scaled dot-product attention
   - 或使用优化的 Flash Attention

---

### 前馈神经网络 (FFN)

#### 技术背景

前馈网络是 Transformer 块的第二个主要组件，负责对每个位置独立地进行非线性转换。

**FFN 的演进**：
1. **标准 FFN**（Transformer, 2017）
   - 两层全连接，中间使用 ReLU
   - `FFN(x) = max(0, xW1 + b1)W2 + b2`

2. **GELU 激活**（BERT, GPT）
   - 更平滑的激活函数
   - 在预训练模型中表现更好

3. **SwiGLU**（GLU Variants, 2020; LLaMA, 2023)
   - 门控线性单元的变体
   - 使用 SiLU (Swish) 作为门控函数
   - 当前大多数 LLM 的标准选择

#### 核心思想

SwiGLU 结合了两个思想：
1. **GLU（Gated Linear Units）**：使用门控机制控制信息流
2. **SiLU（Sigmoid Linear Unit）**：平滑的激活函数

**公式**：
```
SwiGLU(x) = (xW_gate * SiLU(xW_gate)) ⊙ (xW_up)
其中 SiLU(x) = x * sigmoid(x)
```

**三个线性层**：
- `gate_proj`: 门控投影
- `up_proj`: 上投影
- `down_proj`: 下投影

#### 实现细节

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        
        # 计算中间层维度
        if config.intermediate_size is None:
            # LLaMA 风格: 8/3 倍隐藏层大小
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 向上取整到 64 的倍数（硬件友好）
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        # 三个投影层
        self.gate_proj = nn.Linear(config.hidden_size, 
                                   config.intermediate_size, 
                                   bias=False)
        self.up_proj = nn.Linear(config.hidden_size, 
                                config.intermediate_size, 
                                bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, 
                                   config.hidden_size, 
                                   bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  # 默认是 'silu'
    
    def forward(self, x):
        # SwiGLU: (silu(gate) * up) @ down
        gate = self.act_fn(self.gate_proj(x))  # 应用 SiLU 激活
        up = self.up_proj(x)                   # 上投影
        return self.dropout(self.down_proj(gate * up))  # 门控 * 上投影，再下投影
```

**计算流程**：

```
输入 x: [batch, seq, hidden_size]
    ↓
gate_proj: [batch, seq, intermediate_size]
    ↓ SiLU
gate: [batch, seq, intermediate_size]
    
输入 x: [batch, seq, hidden_size]
    ↓
up_proj: [batch, seq, intermediate_size]
    ↓
up: [batch, seq, intermediate_size]
    
gate * up: [batch, seq, intermediate_size] (逐元素乘法)
    ↓
down_proj: [batch, seq, hidden_size]
    ↓ Dropout
输出: [batch, seq, hidden_size]
```

**关键设计选择**：

1. **中间维度**：
   - 通常是隐藏维度的 2.7-4 倍
   - LLaMA 使用 8/3 ≈ 2.67
   - 向上取整到硬件友好的数字（如 64 的倍数）

2. **无偏置**：
   - 移除偏置项可以减少参数
   - 对最终性能影响很小
   - 简化模型结构

3. **SiLU 激活**：
   - `SiLU(x) = x * sigmoid(x)`
   - 平滑、可微、无上界
   - 比 ReLU 表达能力更强

---

### 混合专家模型 (MoE)

#### 技术背景

混合专家模型（Mixture of Experts）是一种条件计算技术，可以在保持参数量的同时减少每个 token 的计算量。

**MoE 的发展**：
1. **早期 MoE**（1991）
   - 最初用于传统机器学习
   - 多个"专家"模型，门控网络选择

2. **稀疏 MoE**（Outrageously Large Neural Networks, 2017）
   - 每个 token 只激活部分专家
   - 实现大规模扩展

3. **现代 LLM 中的 MoE**（Switch Transformer, 2021; Mixtral, 2023）
   - 替换 Transformer 中的 FFN
   - 路由算法和负载均衡
   - Mixtral: 每层 8 个专家，每次激活 2 个

#### 核心思想

MoE 的核心思想是**稀疏激活**：
- 有 N 个专家（FFN）
- 每个 token 只路由到 top-K 个专家
- 不同 token 可能使用不同的专家组合

**优势**：
- **参数效率**：总参数多，但激活参数少
- **计算效率**：每个 token 的 FLOP 与标准 FFN 相当
- **专业化**：不同专家可以学习不同的模式

**挑战**：
- **负载均衡**：确保专家使用均匀
- **训练稳定性**：需要辅助损失
- **推理优化**：动态路由增加复杂度

#### 实现细节

**1. MoE 门控网络**

```python
class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个 token 选择的专家数
        self.n_routed_experts = config.n_routed_experts  # 总专家数
        
        # 门控参数
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha  # 辅助损失权重
        self.seq_aux = config.seq_aux  # 序列级辅助损失
        self.norm_topk_prob = config.norm_topk_prob  # 归一化 top-k 概率
        
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, 
                                               self.gating_dim)))
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        
        # 1. 计算门控分数
        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1)
        
        # 2. 选择 top-k 专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, 
                                           dim=-1, sorted=False)
        
        # 3. 归一化权重（可选）
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        
        # 4. 计算辅助损失（训练时）
        if self.training and self.alpha > 0.0:
            aux_loss = self._compute_aux_loss(scores, topk_idx, bsz, seq_len)
        else:
            aux_loss = 0
        
        return topk_idx, topk_weight, aux_loss
    
    def _compute_aux_loss(self, scores, topk_idx, bsz, seq_len):
        """计算负载均衡辅助损失"""
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        
        if self.seq_aux:
            # 序列级损失
            scores_for_seq_aux = scores.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.n_routed_experts, 
                           device=scores.device)
            ce.scatter_add_(
                1, topk_idx_for_aux_loss,
                torch.ones(bsz, seq_len * self.top_k, 
                          device=scores.device)
            ).div_(seq_len * self.top_k / self.n_routed_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean()
            aux_loss = aux_loss * self.alpha
        else:
            # 全局级损失
            mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), 
                               num_classes=self.n_routed_experts)
            ce = mask_ce.float().mean(0)  # 实际使用比例
            Pi = scores.mean(0)           # 预测概率
            fi = ce * self.n_routed_experts  # 归一化使用次数
            aux_loss = (Pi * fi).sum() * self.alpha
        
        return aux_loss
```

**门控机制解析**：

1. **评分函数**：
   ```
   logits = hidden_states @ weight^T
   scores = softmax(logits)  # [batch*seq, n_experts]
   ```
   - 为每个 token 对每个专家计算亲和度分数

2. **Top-K 选择**：
   ```
   topk_weight, topk_idx = torch.topk(scores, k=top_k)
   ```
   - 选择分数最高的 K 个专家
   - 返回权重和索引

3. **辅助损失**：
   - 目标：鼓励专家使用均匀
   - 方法：最小化实际使用与理想分布的差异
   - 公式：`aux_loss = Σ(P_i * f_i)`
     - `P_i`: 专家 i 被选中的平均概率
     - `f_i`: 专家 i 实际被使用的次数（归一化）

**2. MoE 前馈网络**

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        
        # 路由专家
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        
        # 门控网络
        self.gate = MoEGate(config)
        
        # 共享专家（可选）
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])
    
    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # 1. 门控路由
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # 2. 重塑输入
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        # 3. 专家计算
        if self.training:
            # 训练模式：每个 token 计算 K 次
            y = self._training_forward(x, flat_topk_idx, topk_weight, orig_shape)
        else:
            # 推理模式：优化的批处理
            y = self.moe_infer(x, flat_topk_idx, 
                             topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 4. 共享专家
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        
        self.aux_loss = aux_loss
        return y
    
    def _training_forward(self, x, flat_topk_idx, topk_weight, orig_shape):
        """训练时的前向传播"""
        # 将每个 token 重复 K 次
        x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
        y = torch.empty_like(x, dtype=torch.float16)
        
        # 每个专家处理分配给它的 token
        for i, expert in enumerate(self.experts):
            mask = (flat_topk_idx == i)
            if mask.any():
                y[mask] = expert(x[mask]).to(y.dtype)
        
        # 加权聚合
        y = y.view(*topk_weight.shape, -1)  # [batch*seq, K, hidden]
        y = (y * topk_weight.unsqueeze(-1)).sum(dim=1)  # 加权求和
        y = y.view(*orig_shape)
        
        return y
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """推理时的优化前向传播"""
        expert_cache = torch.zeros_like(x)
        
        # 按专家索引排序
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # 批处理：每个专家一次性处理所有分配给它的 token
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            
            # 专家计算
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            
            # 应用权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            
            # 累加到输出
            expert_cache.scatter_add_(
                0, 
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), 
                expert_out
            )
        
        return expert_cache
```

**MoE 计算流程**：

**训练模式**：
```
输入 x: [batch*seq, hidden]
    ↓
门控: 选择 top-K 专家
    ↓
重复 K 次: [batch*seq*K, hidden]
    ↓
并行计算: 每个专家处理对应的 token
    ↓
加权聚合: Σ(weight_i * expert_i(x))
    ↓
输出: [batch*seq, hidden]
```

**推理模式**（优化）：
```
输入 x: [batch*seq, hidden]
    ↓
门控: 选择 top-K 专家，得到 indices 和 weights
    ↓
排序: 按专家索引排序
    ↓
批处理: 每个专家一次性处理所有分配的 token
    ↓
scatter_add: 将结果累加到对应位置
    ↓
输出: [batch*seq, hidden]
```

**关键优化**：

1. **推理批处理**：
   - 不是为每个 token 调用 K 次专家
   - 而是为每个专家调用一次，处理所有分配的 token
   - 更好的硬件利用率

2. **共享专家**：
   - 总是激活的专家，不参与路由
   - 可以学习通用特征
   - 提高模型容量和稳定性

3. **辅助损失**：
   - 鼓励负载均衡
   - 防止专家崩溃（所有 token 只用一个专家）
   - 提高训练稳定性

---

## 完整模型结构

### MiniMindBlock

Transformer 的基本构建块，组合了注意力和前馈网络。

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.layer_id = layer_id
        
        # 归一化层
        self.input_layernorm = RMSNorm(config.hidden_size, 
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, 
                                               eps=config.rms_norm_eps)
        
        # 注意力
        self.self_attn = Attention(config)
        
        # 前馈网络（标准或 MoE）
        self.mlp = FeedForward(config) if not config.use_moe \
                  else MOEFeedForward(config)
    
    def forward(self, hidden_states, position_embeddings, 
               past_key_value=None, use_cache=False, attention_mask=None):
        # 1. 自注意力子层（带残差）
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # Pre-normalization
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states = hidden_states + residual  # 残差连接
        
        # 2. 前馈子层（带残差）
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)  # Pre-normalization
        )
        
        return hidden_states, present_key_value
```

**Block 结构**：
```
输入
  ↓
LayerNorm → Attention → 残差 +
  ↓                      ↓
  └──────────────────────┘
  ↓
LayerNorm → FFN/MoE → 残差 +
  ↓                    ↓
  └────────────────────┘
  ↓
输出
```

### MiniMindModel

完整的 Transformer 模型。

```python
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, 
                                        config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            MiniMindBlock(l, config) 
            for l in range(config.num_hidden_layers)
        ])
        
        # Final Norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 预计算 RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    
    def forward(self, input_ids, attention_mask=None, 
               past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        
        # 处理 KV cache
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] \
                   if past_key_values[0] is not None else 0
        
        # 1. Embedding
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        
        # 2. 获取位置编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )
        
        # 3. Transformer Layers
        presents = []
        for layer_idx, (layer, past_key_value) in \
            enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        
        # 4. Final Normalization
        hidden_states = self.norm(hidden_states)
        
        # 5. 收集 MoE 辅助损失
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )
        
        return hidden_states, presents, aux_loss
```

### MiniMindForCausalLM

因果语言模型，添加了语言模型头。

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        
        # Transformer
        self.model = MiniMindModel(self.config)
        
        # LM Head
        self.lm_head = nn.Linear(self.config.hidden_size, 
                                self.config.vocab_size, 
                                bias=False)
        
        # 权重共享
        self.model.embed_tokens.weight = self.lm_head.weight
        
        self.OUT = CausalLMOutputWithPast()
    
    def forward(self, input_ids, attention_mask=None, 
               past_key_values=None, use_cache=False, 
               logits_to_keep=0, **args):
        # 1. Transformer forward
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        # 2. LM Head
        slice_indices = slice(-logits_to_keep, None) \
                       if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        
        # 3. 构造输出
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        
        return self.OUT
```

**完整前向传播流程**：

```
输入 token IDs: [batch, seq]
    ↓
Embedding: [batch, seq, hidden]
    ↓
Dropout
    ↓
┌─────────────────────────────┐
│  MiniMindBlock × N          │
│  ┌─────────────────────┐    │
│  │ RMSNorm             │    │
│  │ Attention (GQA)     │    │
│  │ + RoPE              │    │
│  │ Residual            │    │
│  ├─────────────────────┤    │
│  │ RMSNorm             │    │
│  │ FFN / MoE           │    │
│  │ Residual            │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
    ↓
Final RMSNorm
    ↓
LM Head: [batch, seq, vocab_size]
    ↓
输出 logits
```

---

## 配置说明

```python
class MiniMindConfig(PretrainedConfig):
    def __init__(
        self,
        # 基础配置
        vocab_size: int = 6400,              # 词表大小
        hidden_size: int = 512,              # 隐藏层维度
        num_hidden_layers: int = 8,          # Transformer 层数
        num_attention_heads: int = 8,        # 注意力头数
        num_key_value_heads: int = 2,        # KV 头数（GQA）
        
        # FFN 配置
        intermediate_size: int = None,       # FFN 中间层维度
        hidden_act: str = 'silu',           # 激活函数
        
        # 位置编码配置
        max_position_embeddings: int = 32768,  # 最大序列长度
        rope_theta: int = 1000000.0,          # RoPE 基数
        inference_rope_scaling: bool = False,  # 是否使用 YaRN 外推
        
        # 归一化配置
        rms_norm_eps: float = 1e-05,         # RMSNorm epsilon
        
        # Dropout 和其他
        dropout: float = 0.0,                # Dropout 概率
        bos_token_id: int = 1,              # 开始 token ID
        eos_token_id: int = 2,              # 结束 token ID
        flash_attn: bool = True,            # 是否使用 Flash Attention
        
        # MoE 配置
        use_moe: bool = False,              # 是否使用 MoE
        num_experts_per_tok: int = 2,       # 每个 token 激活的专家数
        n_routed_experts: int = 4,          # 路由专家总数
        n_shared_experts: int = 1,          # 共享专家数
        scoring_func: str = 'softmax',      # 门控评分函数
        aux_loss_alpha: float = 0.1,        # 辅助损失权重
        seq_aux: bool = True,               # 序列级辅助损失
        norm_topk_prob: bool = True,        # 归一化 top-k 概率
        **kwargs
    ):
        ...
```

**配置建议**：

1. **小模型**（~50M 参数）：
   - hidden_size: 512
   - num_hidden_layers: 8
   - num_attention_heads: 8
   - num_key_value_heads: 2

2. **中等模型**（~200M 参数）：
   - hidden_size: 1024
   - num_hidden_layers: 16
   - num_attention_heads: 16
   - num_key_value_heads: 4

3. **大模型**（~1B 参数）：
   - hidden_size: 2048
   - num_hidden_layers: 24
   - num_attention_heads: 32
   - num_key_value_heads: 8

---

## 总结

MiniMind 是一个现代化的 Transformer 语言模型实现，集成了多项最新技术：

1. **RMSNorm**：更简单高效的归一化
2. **RoPE**：优秀的位置编码，支持外推
3. **GQA**：平衡性能和效率的注意力机制
4. **SwiGLU FFN**：更强的非线性转换能力
5. **MoE**：可选的稀疏激活，提高参数效率

这些技术的组合使得 MiniMind 在保持轻量级的同时，具备了与大型语言模型相似的架构优势。
