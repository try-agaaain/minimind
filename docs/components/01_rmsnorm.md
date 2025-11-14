# RMSNorm：根均方归一化详解

## 目录

1. [引言](#引言)
2. [归一化技术的历史与动机](#归一化技术的历史与动机)
3. [LayerNorm 的数学原理](#layernorm-的数学原理)
4. [RMSNorm 的提出与创新](#rmsnorm-的提出与创新)
5. [数学推导与理论基础](#数学推导与理论基础)
6. [实现细节与代码解析](#实现细节与代码解析)
7. [性能分析](#性能分析)
8. [实际应用中的考量](#实际应用中的考量)
9. [总结](#总结)

---

## 引言

归一化（Normalization）是深度学习中的核心技术之一，它能够稳定训练过程、加速收敛、提高模型性能。在 Transformer 架构中，归一化层扮演着至关重要的角色。本文将深入探讨 RMSNorm（Root Mean Square Layer Normalization），这是一种简化但高效的归一化方法，被现代大语言模型（如 LLaMA、PaLM）广泛采用。

### 为什么需要归一化？

在深度神经网络训练过程中，我们面临以下挑战：

1. **内部协变量偏移**（Internal Covariate Shift）
   - 随着网络层数加深，每层的输入分布会不断变化
   - 这使得后续层需要不断适应新的输入分布
   - 导致训练不稳定、收敛缓慢

2. **梯度问题**
   - **梯度消失**：在深层网络中，梯度可能变得极小，导致参数更新缓慢
   - **梯度爆炸**：梯度可能变得极大，导致参数更新过度，训练不稳定

3. **学习率敏感性**
   - 不同层可能需要不同的学习率
   - 归一化可以减少对学习率的敏感性，使训练更加稳定

**归一化的核心思想**：将每层的输入或激活值调整到一个标准的分布（通常是均值为 0、方差为 1），从而稳定训练过程。

---

## 归一化技术的历史与动机

### 2.1 Batch Normalization (2015)

**论文**：Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"

**核心思想**：对每个 mini-batch 进行归一化

**数学公式**：
```
对于输入 x ∈ ℝ^(B×D)，其中 B 是 batch size，D 是特征维度

μ_B = (1/B) Σ_{i=1}^B x_i          # 批次均值
σ²_B = (1/B) Σ_{i=1}^B (x_i - μ_B)²  # 批次方差

x̂_i = (x_i - μ_B) / √(σ²_B + ε)   # 归一化
y_i = γ ⊙ x̂_i + β                  # 缩放和平移
```

**优点**：
- 显著加速训练
- 允许使用更大的学习率
- 减少对初始化的依赖
- 具有正则化效果

**缺点**：
- 依赖 batch size：小 batch 时性能下降
- 训练和推理行为不一致（需要维护移动平均）
- 不适合序列模型（RNN、Transformer）

**为什么 BatchNorm 不适合 Transformer？**

在 Transformer 中：
- 序列长度可变，batch 中不同样本的序列长度不同
- 注意力机制使得每个位置的统计特性差异很大
- NLP 任务中 batch size 通常较小（显存限制）

### 2.2 Layer Normalization (2016)

**论文**：Ba, Kiros & Hinton, "Layer Normalization"

**核心思想**：对每个样本的所有特征进行归一化，不依赖 batch

**数学公式**：
```
对于输入 x ∈ ℝ^D（单个样本的特征向量）

μ = (1/D) Σ_{i=1}^D x_i             # 特征均值
σ² = (1/D) Σ_{i=1}^D (x_i - μ)²    # 特征方差

x̂_i = (x_i - μ) / √(σ² + ε)        # 归一化
y_i = γ_i · x̂_i + β_i              # 可学习的缩放和平移
```

**优点**：
- 不依赖 batch size，适用于任意 batch
- 训练和推理行为一致
- 适合 RNN 和 Transformer
- 在 NLP 任务中表现优异

**缺点**：
- 需要计算均值和方差，计算量较大
- 对每个特征维度都有可学习参数（γ 和 β）

**LayerNorm 在 Transformer 中的成功**：
- Transformer 的原始论文就采用了 LayerNorm
- 位于 Multi-Head Attention 和 Feed-Forward Network 之前（Pre-LN）或之后（Post-LN）
- 成为了 Transformer 架构的标准组件

### 2.3 从 LayerNorm 到 RMSNorm 的动机

尽管 LayerNorm 在 Transformer 中表现出色，研究者们仍在思考：
1. **计算效率**：能否进一步减少计算量？
2. **简化设计**：LayerNorm 的所有组件都是必需的吗？
3. **理论理解**：为什么 LayerNorm 有效？哪些是关键因素？

Zhang & Sennrich (2019) 在论文 "Root Mean Square Layer Normalization" 中提出了 RMSNorm，通过实验和分析发现：

**关键洞察**：LayerNorm 的成功主要来自于**重新缩放**（re-scaling），而**重新中心化**（re-centering，即减去均值）的作用相对较小。

---

## LayerNorm 的数学原理

在深入理解 RMSNorm 之前，我们需要完全理解 LayerNorm 的工作原理。

### 3.1 完整的数学推导

**输入**：特征向量 $\mathbf{x} = [x_1, x_2, ..., x_D]^T \in \mathbb{R}^D$

**步骤 1：计算统计量**
```
均值：μ = (1/D) Σ_{i=1}^D x_i

方差：σ² = (1/D) Σ_{i=1}^D (x_i - μ)²
```

**步骤 2：标准化**
```
x̂_i = (x_i - μ) / √(σ² + ε)
```

其中 $\epsilon$ 是一个很小的常数（如 $10^{-5}$），用于数值稳定性。

**步骤 3：仿射变换**
```
y_i = γ_i · x̂_i + β_i
```

其中 $\gamma$ 和 $\beta$ 是可学习参数。

### 3.2 LayerNorm 的作用分解

我们可以将 LayerNorm 的作用分解为两个部分：

**1. 重新中心化（Re-centering）**：减去均值 $\mu$
- 使得归一化后的数据均值为 0
- 消除了输入的偏移

**2. 重新缩放（Re-scaling）**：除以标准差 $\sqrt{\sigma^2 + \epsilon}$
- 使得归一化后的数据方差为 1
- 控制了激活值的尺度

### 3.3 重新中心化真的必要吗？

Zhang & Sennrich (2019) 进行了消融实验，比较了三种变体：

1. **完整 LayerNorm**：包含重新中心化和重新缩放
2. **只重新缩放**：省略均值，只除以 RMS
3. **只重新中心化**：减去均值，但不除以标准差

**实验结果**（在机器翻译任务上）：
- 完整 LayerNorm：BLEU = 27.2
- 只重新缩放（RMSNorm）：BLEU = 27.1
- 只重新中心化：BLEU = 20.5

**结论**：
- 重新缩放是 LayerNorm 成功的**关键因素**
- 重新中心化的贡献**很小**，可以省略
- 这为 RMSNorm 提供了理论依据

### 3.4 为什么重新缩放更重要？

**直觉解释**：

1. **尺度的重要性**
   - 神经网络对输入的尺度非常敏感
   - 过大的激活值可能导致梯度爆炸
   - 过小的激活值可能导致梯度消失
   - 重新缩放确保了激活值在一个合适的范围内

2. **均值的影响较小**
   - 现代激活函数（如 ReLU、GELU、SiLU）通常不以 0 为中心
   - 偏置项（bias）可以调整输出的均值
   - 后续的非线性层可以适应不同的均值

3. **计算效率**
   - 计算均值需要额外的计算和内存
   - 省略均值可以减少约 7-15% 的计算时间

---

## RMSNorm 的提出与创新

### 4.1 RMSNorm 的定义

RMSNorm（Root Mean Square Layer Normalization）的核心思想是：**只使用 RMS（Root Mean Square）进行归一化，省略均值的计算**。

**数学定义**：
```
对于输入 x ∈ ℝ^D

RMS(x) = √[(1/D) Σ_{i=1}^D x_i²]    # 根均方值

x̂_i = x_i / RMS(x)                  # 归一化

y_i = γ_i · x̂_i                     # 缩放（通常省略偏置 β）
```

**与 LayerNorm 的对比**：

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算均值 | ✓ | ✗ |
| 计算方差/RMS | ✓ | ✓ |
| 重新中心化 | ✓ | ✗ |
| 重新缩放 | ✓ | ✓ |
| 可学习参数 | γ, β | γ |
| 计算复杂度 | O(2D) | O(D) |

### 4.2 为什么叫"Root Mean Square"？

让我们逐步理解这个名字：

**1. Square（平方）**
```
x² = [x_1², x_2², ..., x_D²]
```

**2. Mean（均值）**
```
mean(x²) = (1/D) Σ_{i=1}^D x_i²
```

**3. Root（平方根）**
```
RMS = √[mean(x²)] = √[(1/D) Σ_{i=1}^D x_i²]
```

**RMS 与标准差的关系**：

回忆标准差的定义：
```
σ = √[(1/D) Σ_{i=1}^D (x_i - μ)²]
```

展开：
```
σ² = (1/D) Σ_{i=1}^D (x_i - μ)²
   = (1/D) Σ_{i=1}^D (x_i² - 2x_iμ + μ²)
   = (1/D) Σ_{i=1}^D x_i² - 2μ(1/D)Σ_{i=1}^D x_i + μ²
   = (1/D) Σ_{i=1}^D x_i² - 2μ² + μ²
   = (1/D) Σ_{i=1}^D x_i² - μ²
```

因此：
```
RMS² = (1/D) Σ_{i=1}^D x_i²
σ² = RMS² - μ²

当 μ ≈ 0 时，RMS ≈ σ
```

**关键洞察**：当数据均值接近 0 时，RMS 近似等于标准差。在深度网络中，经过多层变换后，激活值的均值往往接近 0，因此 RMSNorm 和 LayerNorm 的效果相近。

### 4.3 RMSNorm 的优势

**1. 计算效率提升**

LayerNorm 的计算：
```
# 第一次遍历：计算均值
μ = (1/D) Σ x_i

# 第二次遍历：计算方差
σ² = (1/D) Σ (x_i - μ)²

# 第三次遍历：归一化
x̂_i = (x_i - μ) / √(σ² + ε)
```

RMSNorm 的计算：
```
# 第一次遍历：计算 RMS
RMS = √[(1/D) Σ x_i²]

# 第二次遍历：归一化
x̂_i = x_i / RMS
```

**理论加速**：
- 减少了一次遍历（不需要计算均值）
- 减少了 D 次减法操作
- 实际加速：7-15%（取决于硬件和实现）

**2. 内存占用减少**

- LayerNorm 需要存储均值 μ 用于反向传播
- RMSNorm 只需要存储 RMS
- 在大规模模型中，内存节省显著

**3. 数值稳定性**

RMSNorm 的计算更加稳定：
```
RMS = √[(1/D) Σ x_i²]
```

相比于：
```
σ = √[(1/D) Σ (x_i - μ)²]
```

RMSNorm 避免了减法操作，减少了潜在的数值误差。

**4. 实现简洁**

代码更加简洁，易于优化和并行化。

---

## 数学推导与理论基础

### 5.1 RMSNorm 的完整数学形式

给定输入向量 $\mathbf{x} = [x_1, x_2, ..., x_D]^T \in \mathbb{R}^D$

**定义根均方值**：
$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{D} \sum_{i=1}^{D} x_i^2}
$$

**归一化**：
$$
\hat{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x})}
$$

**加上数值稳定项**：
$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{D} \sum_{i=1}^{D} x_i^2 + \epsilon}
$$

其中 $\epsilon$ 是一个很小的常数（通常为 $10^{-5}$ 或 $10^{-6}$）。

**可学习的缩放**：
$$
y_i = \gamma_i \cdot \hat{x}_i = \gamma_i \cdot \frac{x_i}{\text{RMS}(\mathbf{x})}
$$

其中 $\gamma = [\gamma_1, \gamma_2, ..., \gamma_D]^T$ 是可学习参数，初始化为全 1 向量。

### 5.2 向量形式

使用向量记法，可以更简洁地表示：

$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{\|\mathbf{x}\|_2^2}{D} + \epsilon}
$$

$$
\hat{\mathbf{x}} = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}
$$

$$
\mathbf{y} = \boldsymbol{\gamma} \odot \hat{\mathbf{x}}
$$

其中 $\odot$ 表示逐元素乘法（element-wise multiplication），$\|\mathbf{x}\|_2$ 是 L2 范数。

### 5.3 反向传播推导

为了在深度学习框架中实现 RMSNorm，我们需要推导其梯度。

**前向传播**：
$$
\begin{align}
r &= \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{D} \sum_{i=1}^{D} x_i^2 + \epsilon} \\
\hat{x}_i &= \frac{x_i}{r} \\
y_i &= \gamma_i \hat{x}_i
\end{align}
$$

**反向传播**：给定损失 $L$，我们需要计算 $\frac{\partial L}{\partial x_i}$ 和 $\frac{\partial L}{\partial \gamma_i}$。

**1. 对 $\gamma$ 的梯度**（简单）：
$$
\frac{\partial L}{\partial \gamma_i} = \frac{\partial L}{\partial y_i} \cdot \hat{x}_i
$$

**2. 对 $\hat{x}_i$ 的梯度**：
$$
\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma_i
$$

**3. 对 $r$ 的梯度**：
$$
\frac{\partial L}{\partial r} = \sum_{j=1}^{D} \frac{\partial L}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial r}
$$

其中：
$$
\frac{\partial \hat{x}_j}{\partial r} = \frac{\partial}{\partial r}\left(\frac{x_j}{r}\right) = -\frac{x_j}{r^2}
$$

因此：
$$
\frac{\partial L}{\partial r} = \sum_{j=1}^{D} \frac{\partial L}{\partial \hat{x}_j} \cdot \left(-\frac{x_j}{r^2}\right) = -\frac{1}{r^2} \sum_{j=1}^{D} \frac{\partial L}{\partial \hat{x}_j} \cdot x_j
$$

**4. 对 $x_i$ 的梯度**：

使用链式法则：
$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial L}{\partial r} \cdot \frac{\partial r}{\partial x_i}
$$

计算各项：
$$
\frac{\partial \hat{x}_i}{\partial x_i} = \frac{1}{r}
$$

$$
\frac{\partial r}{\partial x_i} = \frac{\partial}{\partial x_i} \sqrt{\frac{1}{D} \sum_{j=1}^{D} x_j^2 + \epsilon} = \frac{1}{2r} \cdot \frac{2x_i}{D} = \frac{x_i}{Dr}
$$

代入：
$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{r} + \frac{\partial L}{\partial r} \cdot \frac{x_i}{Dr}
$$

$$
= \frac{1}{r} \left[ \frac{\partial L}{\partial \hat{x}_i} + \frac{x_i}{D} \cdot \frac{\partial L}{\partial r} \right]
$$

代入 $\frac{\partial L}{\partial r}$ 的表达式：
$$
\frac{\partial L}{\partial x_i} = \frac{1}{r} \left[ \frac{\partial L}{\partial \hat{x}_i} - \frac{x_i}{Dr^2} \sum_{j=1}^{D} \frac{\partial L}{\partial \hat{x}_j} \cdot x_j \right]
$$

**最终梯度公式**：
$$
\frac{\partial L}{\partial x_i} = \frac{\gamma_i}{r} \left[ \frac{\partial L}{\partial y_i} - \frac{x_i}{Dr^2} \sum_{j=1}^{D} \gamma_j \frac{\partial L}{\partial y_j} x_j \right]
$$

### 5.4 为什么 RMSNorm 有效？

**理论分析**：

1. **尺度不变性**
   - RMSNorm 使得输出的 RMS 值固定为 1（忽略 $\gamma$）
   - 这确保了激活值的尺度稳定
   - 减少了对学习率和初始化的敏感性

2. **梯度流动**
   - 归一化后的梯度更加稳定
   - 减少了梯度消失和爆炸的风险
   - 加速了收敛

3. **表达能力**
   - 可学习参数 $\gamma$ 允许模型学习最优的缩放
   - 每个特征维度可以有不同的重要性

**实验验证**：

在多个任务上的实验表明：
- 语言建模：RMSNorm 与 LayerNorm 性能相当
- 机器翻译：BLEU 分数相差 < 0.1
- 图像分类：准确率相差 < 0.2%

同时获得：
- 训练速度提升 7-15%
- 内存占用减少
- 实现更简洁

---

## 实现细节与代码解析

### 6.1 基础 PyTorch 实现

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        RMSNorm 初始化
        
        参数:
            dim: 特征维度
            eps: 数值稳定性常数，防止除零
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数，初始化为 1
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        """
        计算 RMS 归一化
        
        参数:
            x: 输入张量，形状为 [..., dim]
        
        返回:
            归一化后的张量
        """
        # 计算 RMS: sqrt(mean(x^2) + eps)
        # x.pow(2): 逐元素平方
        # .mean(-1, keepdim=True): 对最后一个维度求均值，保持维度用于广播
        # torch.rsqrt: 计算平方根的倒数，即 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 [..., dim]
        
        返回:
            归一化并缩放后的张量
        """
        # 转换为 float32 进行计算（提高数值稳定性）
        # 然后转回原始类型（可能是 float16 或 bfloat16）
        output = self._norm(x.float()).type_as(x)
        # 应用可学习的缩放参数
        return self.weight * output
```

### 6.2 代码详解

让我们逐行分析关键部分：

**1. `torch.rsqrt` 的使用**

```python
return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

为什么使用 `rsqrt` 而不是 `1 / torch.sqrt(...)`？

- `rsqrt(x) = 1 / sqrt(x)` 是一个单一的操作
- 比先计算 `sqrt` 再计算倒数更高效
- 某些硬件（如 GPU）对 `rsqrt` 有专门的优化指令
- 数值稳定性更好

**等价形式对比**：
```python
# 方式 1：使用 rsqrt（推荐）
norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

# 方式 2：使用 sqrt 和除法（不推荐）
rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
norm = x / rms

# 方式 3：更直观但更慢
rms = torch.sqrt((x ** 2).mean(-1, keepdim=True) + self.eps)
norm = x / rms
```

**2. `keepdim=True` 的重要性**

```python
x.pow(2).mean(-1, keepdim=True)
```

假设输入 `x` 的形状是 `[batch, seq_len, hidden_dim]`：

- 不使用 `keepdim`：结果形状为 `[batch, seq_len]`
- 使用 `keepdim=True`：结果形状为 `[batch, seq_len, 1]`

使用 `keepdim=True` 的好处：
- 保持维度，可以直接与 `x` 进行广播
- 避免了显式的维度扩展操作

**示例**：
```python
import torch

x = torch.randn(2, 3, 4)  # [batch=2, seq=3, dim=4]

# 不使用 keepdim
mean1 = x.pow(2).mean(-1)
print(mean1.shape)  # torch.Size([2, 3])

# 使用 keepdim
mean2 = x.pow(2).mean(-1, keepdim=True)
print(mean2.shape)  # torch.Size([2, 3, 1])

# 广播
result = x / mean2  # 可以直接广播
```

**3. 类型转换的必要性**

```python
output = self._norm(x.float()).type_as(x)
```

为什么要先转换为 `float32` 再转回原类型？

- **混合精度训练**：在训练大模型时，通常使用 `float16` 或 `bfloat16` 来节省内存
- **数值稳定性**：归一化涉及平方、求和、开方等操作，`float16` 容易溢出
- **最佳实践**：在关键的数值计算中使用 `float32`，然后转回原类型

**数值范围对比**：
- `float16`: 范围 ≈ ±65,504，精度约 3-4 位十进制
- `float32`: 范围 ≈ ±3.4×10³⁸，精度约 7 位十进制
- `bfloat16`: 范围与 `float32` 相同，但精度较低

**示例问题**：
```python
import torch

# float16 容易溢出
x_fp16 = torch.randn(1000, device='cuda', dtype=torch.float16) * 100
x_squared = x_fp16.pow(2)  # 可能溢出为 inf

# 转换为 float32 计算
x_squared_safe = x_fp16.float().pow(2)  # 安全
```

### 6.3 完整使用示例

```python
# 在 Transformer 中使用 RMSNorm
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.ffn = FeedForward(dim)
        
        # Pre-normalization 风格
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
    
    def forward(self, x):
        # 注意力子层
        x = x + self.attention(self.norm1(x))
        
        # 前馈子层
        x = x + self.ffn(self.norm2(x))
        
        return x

# 批量处理示例
batch_size, seq_len, hidden_dim = 32, 128, 512
x = torch.randn(batch_size, seq_len, hidden_dim)

# 初始化 RMSNorm
norm = RMSNorm(hidden_dim)

# 前向传播
output = norm(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"输出 RMS: {output.pow(2).mean(-1).sqrt().mean():.4f}")
# 预期：接近 1.0（因为 weight 初始化为 1）
```

---

## 性能分析

### 7.1 计算复杂度

**LayerNorm 的复杂度**：

```
1. 计算均值: O(D)
2. 计算方差: O(D)
3. 归一化:   O(D)
4. 仿射变换: O(D)
总计: O(4D) ≈ O(D)
```

**RMSNorm 的复杂度**：

```
1. 计算平方和: O(D)
2. 计算 RMS:   O(1)  # 开方是标量操作
3. 归一化:     O(D)
4. 缩放:       O(D)
总计: O(3D) ≈ O(D)
```

**理论加速比**：
```
加速比 = 4D / 3D = 1.33x
```

实际加速比通常在 1.1-1.15x 之间，因为：
- 内存访问成为瓶颈
- 现代 GPU 对这些操作有高度优化
- 其他操作（如矩阵乘法）占据大部分时间

### 7.2 内存使用

**前向传播的内存**：

LayerNorm 需要存储：
- 输入 x: O(BD)
- 均值 μ: O(B)
- 方差 σ²: O(B)
- 输出 y: O(BD)

RMSNorm 需要存储：
- 输入 x: O(BD)
- RMS: O(B)
- 输出 y: O(BD)

**内存节省**：
- 不需要存储均值
- 在大规模模型中，累积节省显著

---

## 实际应用中的考量

### 8.1 何时使用 RMSNorm

**推荐使用 RMSNorm 的场景**：

1. **大规模语言模型**
   - 参数量 > 1B
   - 序列长度 > 1024
   - 训练成本高，需要提高效率

2. **资源受限环境**
   - 显存有限
   - 计算预算有限
   - 需要快速推理

3. **新项目**
   - 没有历史包袱
   - 可以直接采用最新技术

**谨慎使用的场景**：

1. **需要迁移学习**
   - 预训练模型使用 LayerNorm
   - 转换可能导致性能下降

2. **小规模模型**
   - 加速效果不明显
   - LayerNorm 已经足够快

### 8.2 超参数选择

**epsilon (ε) 的选择**：

- 常用值：`1e-5` 或 `1e-6`
- 太大：影响归一化效果
- 太小：可能导致数值不稳定

**权重初始化**：

```python
# 标准初始化：全 1
self.weight = nn.Parameter(torch.ones(dim))

# 也可以尝试其他初始化
# 全 0（不推荐，会导致初始输出为 0）
# self.weight = nn.Parameter(torch.zeros(dim))

# 小的随机值
# self.weight = nn.Parameter(torch.randn(dim) * 0.02)
```

---

## 总结

### 9.1 核心要点回顾

1. **RMSNorm 的本质**
   - 简化的 LayerNorm，省略了重新中心化
   - 只保留重新缩放，使用 RMS 而非标准差

2. **理论基础**
   - 实验表明重新缩放是关键，重新中心化贡献较小
   - 当激活值均值接近 0 时，RMS ≈ 标准差

3. **优势**
   - 计算效率提升 10-15%
   - 内存占用减少
   - 实现更简洁
   - 性能与 LayerNorm 相当

4. **数学形式**
   ```
   RMS(x) = √[(1/D) Σ x_i²]
   y = γ ⊙ (x / RMS(x))
   ```

5. **实现关键点**
   - 使用 `torch.rsqrt` 提高效率
   - `keepdim=True` 便于广播
   - 转换为 float32 保证数值稳定性

### 9.2 延伸阅读

**论文**：
1. Zhang & Sennrich (2019). "Root Mean Square Layer Normalization"
2. Ba et al. (2016). "Layer Normalization"
3. Ioffe & Szegedy (2015). "Batch Normalization"

**实现参考**：
1. LLaMA 模型：使用 RMSNorm
2. PaLM 模型：使用 RMSNorm
3. Hugging Face Transformers：支持 RMSNorm
