# 分组查询注意力（Grouped Query Attention, GQA）

注意力机制是 Transformer 架构的核心，它让模型能够动态地关注输入序列中的不同部分。但随着模型规模的增长和序列长度的扩展，注意力机制面临着严峻的内存和计算挑战。分组查询注意力（GQA）是一种精巧的设计，它在保持模型性能的同时，显著降低了推理时的内存开销。

## 注意力机制的演进：从 MHA 到 GQA

要理解 GQA 的价值，我们需要先回顾注意力机制的演进历程。这个故事始于多头注意力（Multi-Head Attention, MHA），它是原始 Transformer 论文提出的标准设计。

### 多头注意力（MHA）：并行的多视角

在多头注意力中，输入被投影到多个"头"（heads），每个头独立地计算注意力。具体来说，对于一个输入向量 $x \in \mathbb{R}^{d}$，我们有：

$$
\begin{aligned}
Q_i &= x W_i^Q, \quad K_i = x W_i^K, \quad V_i = x W_i^V \\
\text{head}_i &= \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i \\
\text{MHA}(x) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
\end{aligned}
$$

**为什么要分成多个头？** 单头注意力的一个根本问题是：一个注意力机制只能学习一种"关注模式"。想象你在阅读一个句子"小明在北京的清华大学学习人工智能"。理解这个句子需要同时处理多种关系：语法结构（主谓宾关系）、地理层次（北京→清华）、动作和对象（学习→人工智能）。如果只有一个注意力头，模型必须在这些不同类型的关系之间做权衡——要么关注语法，要么关注语义，很难同时兼顾。

多头设计解决了这个问题。通过使用不同的投影矩阵 $W_i^Q, W_i^K, W_i^V$，每个头将输入投影到不同的表示子空间。在这些不同的子空间中，注意力可以专注于捕获不同类型的依赖关系。实验观察表明，训练后的模型中，不同的头确实会自发地学习不同的模式：有些头关注局部的词法关系（如形容词修饰名词），有些关注长距离的句法依赖（如主语和谓语），还有些专注于语义角色（如施事和受事）。

**为什么可以这样分头计算？** 关键在于线性投影的独立性。我们不是简单地将 $d$ 维向量分成 $h$ 段，而是通过 $h$ 组不同的权重矩阵将同一个输入映射到 $h$ 个不同的表示空间。每个头的维度是 $d_k = d/h$（比如总维度 4096 分成 32 个头，每头 128 维）。这 $h$ 个子空间是互相独立的——一个头在它的 128 维空间中学到的模式，不会干扰另一个头在它的 128 维空间中学到的模式。数学上，这是因为不同头的参数 $W_i$ 是独立优化的，它们可以学习到输入的不同投影方向。

**多个头如何合并？** 在计算完 $h$ 个头的输出后，我们将它们拼接（concatenate）起来，然后通过一个输出投影 $W^O$ 混合：

$$
\text{output} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

这个拼接操作是简单的维度拼接：如果每个 head 输出是 $[\text{seq\_len}, d_k]$，拼接后是 $[\text{seq\_len}, h \times d_k] = [\text{seq\_len}, d]$，恢复到原始维度。**为什么还需要输出投影 $W^O$？** 因为直接拼接只是将不同头的信息放在一起，但它们还没有交互。输出投影 $W^O$ 的作用是让这 $h$ 个头"对话"——它学习如何将来自不同视角的信息融合成一个统一的表示。比如，如果头1发现了语法关系，头2发现了语义关系，$W^O$ 需要学会如何权衡和组合这两种信息，形成一个对下游任务最有用的表示。

这种"分而治之再融合"的策略是多头注意力强大的根源。它让模型可以并行地从多个角度理解输入，然后智能地整合这些理解。这就是为什么多头注意力成为了 Transformer 的标准配置。

但这种设计也带来了代价。假设我们有 32 个注意力头，隐藏维度是 4096，那么每个头的维度是 128。在推理时，我们需要缓存所有层的所有头的 K 和 V 向量——这就是著名的 KV Cache。对于一个 40 层的模型，处理长度为 2048 的序列，KV Cache 的大小是：

$$
\text{Memory} = 2 \times \text{layers} \times \text{seq\_len} \times \text{num\_heads} \times \text{head\_dim} \times \text{bytes\_per\_param}
$$

$$
= 2 \times 40 \times 2048 \times 32 \times 128 \times 2 = 2.1 \text{ GB}
$$

这只是一个batch的开销！在服务大量并发请求时，内存迅速成为瓶颈。

### 多查询注意力（MQA）：激进的共享

2019年，Google的研究者提出了多查询注意力（Multi-Query Attention, MQA），这是一个激进的想法：为什么不让所有的头共享同一组 K 和 V？

在 MQA 中，我们只有一组 K 和 V 投影，但保留多个 Q 投影：

$$
\begin{aligned}
Q_i &= x W_i^Q \quad (\text{每个头独立}) \\
K &= x W^K, \quad V = x W^V \quad (\text{所有头共享})
\end{aligned}
$$

这个设计的内存节省是惊人的。在上面的例子中，KV Cache 从 2.1 GB 降到了仅仅 67 MB——节省了 32 倍！原因很简单：我们只需要缓存一组 K 和 V，而不是 32 组。

但这种激进的共享是有代价的。实验表明，MQA 在某些任务上会损失一定的性能，特别是在需要精细建模复杂依赖关系的场景中。问题的根源在于：当所有的 Q 头都必须查询同一组 K 和 V 时，模型失去了从不同视角捕获信息的能力。就像用同一副眼镜看世界，无论你换多少个观察角度，你看到的东西的"本质"是受限的。

### 分组查询注意力（GQA）：平衡之道

2023年，Google再次提出了一个更优雅的方案——分组查询注意力（GQA）。GQA 是 MHA 和 MQA 之间的中间地带：它将注意力头分成若干组，每组共享一对 K 和 V。

假设我们有 32 个 Q 头，但只有 8 组 KV 头。那么每 4 个 Q 头共享一对 K 和 V：

$$
\begin{aligned}
Q_{4i}, Q_{4i+1}, Q_{4i+2}, Q_{4i+3} &\rightarrow \text{共享} \quad K_i, V_i
\end{aligned}
$$

这个设计兼具两者的优点：
- **内存效率**：KV Cache 减少了 4 倍（从 32 组降到 8 组）
- **表达能力**：保留了 8 个不同的"视角"，远好于 MQA 的单一视角
- **计算效率**：Q 头仍然是并行的，没有引入额外的计算复杂度

实验表明，GQA 在内存和性能之间取得了极好的平衡。在许多基准测试中，GQA 的性能与 MHA 相当，但内存开销却接近 MQA。

## KV Cache：推理加速的关键

要深入理解 GQA 的价值，我们需要理解 KV Cache 在自回归生成中的作用。

### 为什么需要 KV Cache？

在语言模型的自回归生成中，我们一次生成一个 token。在第 $t$ 步，模型需要计算第 $t$ 个 token 对前面所有 token 的注意力：

$$
\text{Attention}_t = \text{softmax}\left(\frac{q_t [k_1, k_2, \ldots, k_t]^T}{\sqrt{d_k}}\right) [v_1, v_2, \ldots, v_t]
$$

这里的 $k_1, \ldots, k_{t-1}$ 和 $v_1, \ldots, v_{t-1}$ 在前面的步骤中已经计算过了！如果我们每次都重新计算它们，会造成大量的冗余计算。KV Cache 的想法很简单：把已经计算过的 K 和 V 缓存起来，每一步只需要计算新 token 的 K 和 V，然后拼接到缓存中：

```python
# 第一次：prompt 阶段，计算所有 token
k_cache = compute_k(prompt)  # shape: [batch, seq_len, num_kv_heads, head_dim]
v_cache = compute_v(prompt)

# 之后每一步：只计算新 token
k_new = compute_k(new_token)  # shape: [batch, 1, num_kv_heads, head_dim]
v_new = compute_v(new_token)
k_cache = torch.cat([k_cache, k_new], dim=1)  # 拼接到缓存
v_cache = torch.cat([v_cache, v_new], dim=1)
```

这个优化将每步的计算复杂度从 $O(t^2)$ 降低到 $O(t)$——这是自回归生成实用性的基础。但代价是内存：我们必须保存所有历史的 K 和 V。

### GQA 如何降低 KV Cache 开销？

这就是 GQA 的价值所在。在 MHA 中，如果我们有 32 个头，我们需要缓存 32 组 K 和 V。而在 GQA 中，如果我们使用 8 个 KV 组，我们只需要缓存 8 组——内存立即减少了 4 倍。

更重要的是，这种节省是"零成本"的——我们不需要改变模型的训练过程，不需要重新训练模型。我们只需要在推理时让多个 Q 头查询同一组 K 和 V。这种设计的优雅之处在于：它利用了注意力机制本身的灵活性。

让我们看看代码中是如何实现的：

```python
# 在 Attention 类的初始化中
self.num_key_value_heads = args.num_key_value_heads  # GQA 的 KV 头数，例如 8
self.n_local_heads = args.num_attention_heads  # Q 头数，例如 32
self.n_rep = self.n_local_heads // self.num_key_value_heads  # 重复次数，32/8 = 4

# K 和 V 投影的输出维度是 KV 头数
self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
# Q 投影的输出维度是 Q 头数
self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
```

关键在于 `repeat_kv` 函数，它将 KV 头"扩展"到与 Q 头相同的数量：

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV 头重复 n_rep 次以匹配 Q 头的数量"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:  # MHA 的情况，无需重复
        return x
    return (
        x[:, :, :, None, :]  # 在第4维插入新维度
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)  # 扩展
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)  # 重塑
    )
```

这个函数的巧妙之处在于：它不是真的复制数据，而是通过 `expand` 操作创建一个视图。在底层，数据仍然只有一份，但在逻辑上每个 Q 头都能访问它。这样既节省了内存，又保持了计算的高效性。

在前向传播中：

```python
# 计算 Q, K, V
xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

# KV Cache：拼接历史
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)

# 重复 KV 以匹配 Q 的头数
xk = repeat_kv(xk, self.n_rep)
xv = repeat_kv(xv, self.n_rep)
```

注意这里的细节：KV Cache 中存储的是**未重复**的 KV（只有 8 组），而只在计算注意力时才重复它们。这确保了内存的节省。

## Flash Attention：内存和速度的双重优化

GQA 优化了 KV Cache 的大小，但注意力计算本身仍然有内存和速度的问题。Flash Attention 是另一个重要的优化，它从算法层面重新设计了注意力的计算方式。

### 标准注意力的内存问题

标准的注意力计算需要显式地构造注意力矩阵：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

这个过程的问题在于 $QK^T$：对于序列长度 $n$，这是一个 $n \times n$ 的矩阵。当 $n = 2048$ 时，这个矩阵有 400 万个元素。对于 32 个头，我们需要存储 1.28 亿个浮点数——这是一个巨大的内存开销。

更糟糕的是，这个矩阵只是中间结果！我们先计算 $QK^T$，然后应用 softmax，最后乘以 $V$。在反向传播时，我们还需要再次访问这个矩阵。这意味着 GPU 的高带宽内存（HBM）和片上内存（SRAM）之间会有大量的数据传输，而 HBM 的访问速度远慢于计算速度。

### Flash Attention 的核心思想：分块计算

Flash Attention 的关键洞察是：我们可以将注意力计算分成小块，在 SRAM 中完成每一块的计算，避免将整个注意力矩阵写入 HBM。

具体来说，Flash Attention 将 Q、K、V 分成若干块（blocks），然后逐块计算注意力。对于每一块，它：
1. 从 HBM 加载 Q、K、V 的当前块到 SRAM
2. 在 SRAM 中计算这一块的注意力
3. 使用一个巧妙的技巧在线更新 softmax 的统计量
4. 将结果写回 HBM

这个算法的美妙之处在于：它在数学上与标准注意力完全等价，但内存访问模式更优化。它减少了 HBM 的读写次数，将大部分计算保持在快速的 SRAM 中。

在 PyTorch 2.0+ 中，Flash Attention 被集成为 `F.scaled_dot_product_attention`。代码中的实现是：

```python
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

if self.flash and seq_len > 1:
    # 使用 Flash Attention
    output = F.scaled_dot_product_attention(
        xq, xk, xv, 
        attn_mask=attn_mask, 
        dropout_p=self.dropout if self.training else 0.0, 
        is_causal=True
    )
else:
    # 标准注意力：显式计算注意力矩阵
    scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores = scores + torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
        diagonal=1
    )
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = scores @ xv
```

注意这里的 `is_causal=True` 参数：它告诉 Flash Attention 这是因果注意力（即每个位置只能看到之前的位置）。Flash Attention 可以利用这个信息进一步优化计算，避免计算和存储上三角部分的注意力分数。

### 为什么只在 seq_len > 1 时使用 Flash Attention？

这是一个有趣的细节。当 `seq_len = 1` 时（即推理的自回归生成阶段，每次只生成一个 token），注意力矩阵退化为一个向量——新 token 对所有历史 token 的注意力。在这种情况下：
- 内存开销已经很小（只有 $n$ 个元素，而不是 $n^2$）
- Flash Attention 的分块策略没有优势
- 标准实现可能更简单高效

所以代码选择在短序列时使用标准实现，这是一个务实的工程选择。

## 注意力掩码：控制信息流

在因果语言模型中，我们需要确保每个位置只能看到它之前的位置，而不能"偷看"未来。这是通过注意力掩码（attention mask）实现的。

### 因果掩码：禁止看到未来

标准的因果掩码是一个上三角矩阵，上三角部分（对应未来位置）填充 $-\infty$：

$$
\text{mask} = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

当我们将这个掩码加到注意力分数上时，未来位置的分数变成 $-\infty$，经过 softmax 后变成 0，这样就完全屏蔽了未来的信息。

代码中的实现：

```python
scores = scores + torch.triu(
    torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
    diagonal=1  # 主对角线上方的部分填充 -inf
)
```

`torch.triu` 创建上三角矩阵，`diagonal=1` 表示从主对角线上方一格开始填充，这样主对角线本身（即位置对自己）保持为 0，可以看到。

### 填充掩码：处理变长序列

在批处理中，不同的序列可能有不同的长度。为了高效处理，我们通常将它们填充到相同的长度，但我们不希望模型关注填充的部分。这就需要填充掩码（padding mask）。

填充掩码是一个二值矩阵，其中填充位置标记为 0，真实位置标记为 1：

```python
if attention_mask is not None:
    # attention_mask: [batch, seq_len], 1 for real tokens, 0 for padding
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    # 扩展到 [batch, 1, 1, seq_len]，以便广播
    extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
    # 将 0（填充）变成 -1e9，1（真实）变成 0
    scores = scores + extended_attention_mask
```

这里使用 $-10^9$ 而不是 $-\infty$ 是为了数值稳定性。在 float16 精度下，$-\infty$ 可能导致 NaN。

## 实现细节：让理论落地

理解了原理，让我们看看代码中的实现细节，以及为什么要这样写。

### 头维度的选择

```python
self.head_dim = args.hidden_size // args.num_attention_heads
```

头维度通常选择为 64、128 或 256。为什么不选择更大或更小的值？

- **太小**（如 32）：每个头的表达能力有限，可能无法捕获复杂的模式
- **太大**（如 512）：计算 $QK^T$ 时需要除以 $\sqrt{d_k}$，太大的 $d_k$ 会让梯度变得很小，训练困难
- **64-128 是平衡点**：既有足够的表达能力，又保持良好的数值性质

### Q、K、V 投影的维度

```python
self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
```

注意 Q 投影和 KV 投影的维度不同：
- Q 投影：`num_attention_heads * head_dim`（例如 32 * 128 = 4096）
- KV 投影：`num_key_value_heads * head_dim`（例如 8 * 128 = 1024）

这正是 GQA 的体现：我们用更少的 KV 头，节省参数和内存。

### 为什么不使用 bias？

所有的投影都设置了 `bias=False`。这是为什么？

在 Transformer 中，bias 的作用通常可以被后续的 LayerNorm 或 RMSNorm 吸收。移除 bias 可以：
- 减少参数量（虽然比例不大）
- 简化初始化（不用担心 bias 的初始值）
- 在某些情况下略微提高数值稳定性

这是现代 LLM 的一个常见实践。

### reshape 和 transpose：为什么要重排维度？

```python
xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
xq = xq.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
```

这个操作看起来很繁琐，但它是必要的。注意力计算需要：
- 对每个头独立计算
- 对序列维度进行矩阵乘法

所以我们需要将头维度提到前面，序列维度在后面。这样在计算 $QK^T$ 时，PyTorch 会自动对每个头并行计算。

### 输出投影

```python
self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
```

注意力的输出是所有头的拼接，然后通过一个线性层混合：

```python
output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
output = self.resid_dropout(self.o_proj(output))
```

这个投影的作用是让不同的头"交流"——它们各自提取的信息在这里融合成一个统一的表示。

### Dropout 的位置

代码中有两个 dropout：
- `self.attn_dropout`：应用在注意力权重上（在标准实现中）
- `self.resid_dropout`：应用在输出投影之后

为什么需要两个？它们的作用不同：
- 注意力 dropout 增加了注意力模式的随机性，防止模型过度依赖某些位置
- 残差 dropout 应用在残差连接之前，这是标准的 Transformer 设计

## 性能分析：GQA 的权衡

最后，让我们量化分析 GQA 的收益和代价。

### 内存节省

对于一个 40 层、隐藏维度 4096、32 个 Q 头、8 个 KV 头的模型，处理长度 2048 的序列：

**MHA (32 个 KV 头)**:
$$
\text{KV Cache} = 2 \times 40 \times 2048 \times 32 \times 128 \times 2 = 2.1 \text{ GB}
$$

**GQA (8 个 KV 头)**:
$$
\text{KV Cache} = 2 \times 40 \times 2048 \times 8 \times 128 \times 2 = 524 \text{ MB}
$$

节省：**75%** 的 KV Cache 内存！

### 计算开销

GQA 的计算量与 MHA 几乎相同。唯一的额外开销是 `repeat_kv` 操作，但这只是一个视图操作，不涉及实际的数据复制。在现代 GPU 上，这个开销可以忽略不计。

### 质量损失

实验表明，在大多数任务上，GQA（8-16 个 KV 组）与 MHA 的性能差距在 1-2% 以内，这通常是可以接受的。对于某些特定任务（如需要非常精细的长距离建模），可能需要更多的 KV 组。

但在实际部署中，内存节省带来的吞吐量提升往往远超过这点性能损失的影响。这就是为什么 GQA 成为了现代 LLM（如 Llama 2、Mistral）的标准配置。

## 总结

分组查询注意力是注意力机制演进中的一个优雅平衡点。它继承了 MHA 的多视角能力，又借鉴了 MQA 的内存效率，在两者之间找到了一个实用的中间地带。通过让多个 Q 头共享一组 KV，GQA 在几乎不损失性能的情况下，显著降低了推理时的内存开销。

结合 KV Cache 和 Flash Attention 等技术，现代的注意力实现已经变得高度优化。理解这些技术不仅能帮助我们更好地使用现有模型，也为未来的改进提供了方向——也许下一个突破性的想法，就藏在这些细节之中。
