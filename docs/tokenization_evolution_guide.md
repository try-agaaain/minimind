# LLM 分词技术完全指南：从原理到实践

## 引言

分词（Tokenization）是大语言模型（LLM）训练和推理的第一步，也是最关键的一步。选择合适的分词方法不仅影响模型的性能，还直接关系到训练效率和多语言支持能力。本文将系统性地介绍主流分词技术的演进历程、核心原理以及实际应用，帮助你深入理解并掌握 LLM 分词的完整知识体系。

## 一、子词分词算法的演进

### 1.1 BPE：基于频率的合并策略

**BPE (Byte Pair Encoding)** 是最早被广泛应用的子词分词算法，其核心思想极其简单直接：

#### 算法原理

BPE 采用**贪婪策略**，通过迭代合并最频繁出现的相邻字符对来构建词汇表：

$$
\text{Score}(A, B) = \text{Freq}(AB)
$$

**训练过程：**
1. 初始化：将所有单词拆分为单字符
2. 统计：计算所有相邻字符对的出现频率
3. 合并：选择频率最高的字符对进行合并
4. 迭代：重复步骤 2-3，直到达到目标词汇表大小

**优势：**
- 实现简单，计算高效
- 能够有效处理未登录词（OOV）

**局限性：**
- 仅基于频率统计，可能合并语义价值低的高频随机组合
- 不考虑子词之间的语义关联性

**典型应用：** GPT-2/3, RoBERTa

### 1.2 WordPiece：基于似然度增益的优化

**WordPiece** 是 BPE 的改进版本，由 Google 提出并应用于 BERT 等模型。它在选择合并对时引入了统计学和语言模型的考量。

#### 算法原理

WordPiece 使用**似然度增益**作为合并标准：

$$
\text{Score}(A, B) = \frac{\text{Freq}(AB)}{\text{Freq}(A) \cdot \text{Freq}(B)}
$$

这个评分公式的核心思想是：
- **分子** $\text{Freq}(AB)$：$A$ 和 $B$ 实际共现的频率
- **分母** $\text{Freq}(A) \cdot \text{Freq}(B)$：假设 $A$ 和 $B$ 独立出现时的期望频率

#### 语义解释

- **高得分**：说明 $A$ 和 $B$ 的共现频率远高于独立期望，表明它们具有强烈的**绑定关系**（如 `un` 和 `happy`）
- **低得分**：即使 $\text{Freq}(AB)$ 很高，但如果 $A$ 和 $B$ 独立出现频率更高，说明可能是随机共现，合并价值较小

**核心优势：**
- 考虑了子词的**统计相关性**，而非单纯的频率
- 最大化整个训练语料在 Unigram 语言模型下的对数似然度
- 生成的词汇表语义质量更高

**前缀标记：**
- 使用 `##` 标记词的内部子词（如 `play` → `##ing`）

**典型应用：** BERT, DistilBERT, MobileBERT

### 1.3 BPE vs WordPiece 对比总结

| 特性 | WordPiece | BPE |
|------|-----------|-----|
| **合并标准** | 基于似然度增益 | 基于频率 |
| **评分公式** | $\frac{\text{Freq}(AB)}{\text{Freq}(A) \cdot \text{Freq}(B)}$ | $\text{Freq}(AB)$ |
| **优化目标** | 最大化语料库对数似然度 | 贪婪合并最高频对 |
| **语义质量** | 更倾向合并强关联子词 | 可能合并随机高频组合 |
| **前缀标记** | `##` (内部子词标记) | `_` (词首标记) |

## 二、SentencePiece 与 Unigram Language Model

### 2.1 为什么需要 SentencePiece？

传统的 BPE 和 WordPiece 都依赖于**预分词**（pre-tokenization），即在应用子词分割前，需要先用空格或标点符号将文本分割。这在处理以下场景时存在问题：

- **无空格语言**：中文、日文、泰文等
- **多语言统一处理**：不同语言需要不同的预分词规则
- **可逆性**：预分词可能丢失信息

**SentencePiece** 应运而生，提供了一种**语言无关、无需预分词**的统一解决方案。

### 2.2 SentencePiece 的核心特性

#### 1. 统一处理（Pre-tokenization-free）
- 将所有输入文本视为**原始字符序列**或 Unicode 字符流
- 空格被视为普通字符，通常用特殊符号 `_` 表示
- 示例：`"hello world"` → `["▁hello", "▁world"]`

#### 2. 语言无关性
- 相同的训练和应用流程适用于任何语言
- 对构建**多语言模型**（如 mBERT、T5、Llama）至关重要

#### 3. 可逆性（Lossless）
- 分词结果可以精确重构回原始文本
- 不会丢失任何信息（包括空格位置）

### 2.3 Unigram Language Model (ULM) 算法

ULM 是 SentencePiece 最推荐的训练算法，它与 BPE/WordPiece 的最大区别在于**训练方向**和**概率切分能力**。

#### 训练方向对比

- **BPE/WordPiece**：从单字符开始，**迭代合并**（从小到大）
- **ULM**：从大词汇表开始，**迭代修剪**（从大到小）

#### ULM 训练过程

**步骤 1：初始词汇表构建**

构建一个包含所有单字符和常见 N-gram 的大型初始词汇表 $V_{\text{initial}}$。

**步骤 2：计算子词概率**

使用 EM 算法为每个子词 $x$ 估计概率 $P(x)$，基于 Unigram 模型假设。

**步骤 3：迭代修剪（Pruning）**

这是 ULM 的核心步骤，迭代执行直到词汇表达到目标大小 $K$：

1. **计算损失**：对每个子词 $x$，计算移除它后整体语料库的对数似然度损失 $\text{Loss}(x)$
2. **选择修剪对象**：找出损失最小的子词（即最冗余的）
3. **修剪**：移除排名最低的 $\eta$ 比例（如 10%-20%）的子词
4. **重新计算**：用新词汇表重新计算剩余子词的概率

通过修剪，ULM 保留那些**对编码贡献最大**的子词单元。

#### ULM 推理：概率切分

这是 ULM 最独特的特性。当给定一个词 $W$ 时，ULM 不使用简单的贪婪匹配，而是使用 **Viterbi 算法**找出概率最大的切分序列：

$$
S^* = \arg\max_{S} \prod_{i=1}^{k} P(s_i)
$$

**具体示例：**

假设词汇表和概率如下：

| 子词 $x$ | 概率 $P(x)$ | $\log P(x)$ |
|---------|------------|-------------|
| `_play` | 0.20 | -1.61 |
| `_ing` | 0.05 | -3.00 |
| `_p` | 0.15 | -1.90 |
| `_lay` | 0.10 | -2.30 |
| `ing` | 0.08 | -2.53 |

对于单词 `"_playing"`，可能的切分：

**路径 A**：`[_play, ing]`
- 总概率：$P = 0.20 \times 0.08 = 0.016$
- 对数概率：$\log P = -1.61 + (-2.53) = -4.14$

**路径 B**：`[_p, lay, ing]`
- 总概率：$P = 0.15 \times 0.10 \times 0.08 = 0.0012$
- 对数概率：$\log P = -1.90 + (-2.30) + (-2.53) = -6.73$

**最优切分**：路径 A（概率更高）

#### ULM 的正则化采样

在训练时，ULM 可以启用**采样模式**，不总是选择最优切分，而是基于概率分布随机采样：

$$
P(S | W) = \frac{P(S)}{\sum_{S'} P(S')}
$$

**优势：**
1. **数据增强**：为模型引入分词边界的轻微变化
2. **鲁棒性**：使模型不过度依赖某一种固定切分方式

### 2.4 ULM 的优势总结

1. **概率切分**：基于统计模型而非贪婪匹配
2. **更好拟合**：基于似然度最大化的优化
3. **多语言统一**：结合 SentencePiece，是现代多语言 LLM 的基石
4. **训练正则化**：采样切分提供数据增强效果

**典型应用：** T5, XLNet, ALBERT, Llama

## 三、实践指南：分词器的使用与训练

### 3.1 两个核心库的分工

在 Hugging Face 生态系统中，`transformers` 和 `tokenizers` 两个库各司其职：

| 库名 | 定位 | 核心功能 |
|------|------|---------|
| **`tokenizers`** | 高性能分词引擎 | 训练、保存、加载分词器（Rust 后端） |
| **`transformers`** | 模型生态集成 | 加载预训练分词器，集成到模型管线 |

### 3.2 使用场景一：加载预训练分词器（最常用）

当你使用 BERT、GPT-2、Llama 等预训练模型时，使用 `transformers` 库：

```python
from transformers import AutoTokenizer

# 加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 编码文本
text = "The quick brown fox jumps over the lazy dog."
encoding = tokenizer(text)

print("Tokens:", tokenizer.convert_ids_to_tokens(encoding['input_ids']))
print("IDs:", encoding['input_ids'])

# 批量编码（自动填充和截断）
texts = ["This is a short sentence.", 
         "This is a much longer sentence that needs to be handled."]
batch_encoding = tokenizer(
    texts, 
    padding=True,          # 填充到最长序列
    truncation=True,       # 截断超长序列
    return_tensors="pt"    # 返回 PyTorch Tensor
)

# 解码（Token IDs → 文本）
decoded_text = tokenizer.decode(batch_encoding['input_ids'][0])
print("Decoded:", decoded_text)
```

**优势：**
- 自动识别并加载正确的分词算法
- 自动处理特殊 Token（`[CLS]`, `[SEP]`, `[PAD]`）
- 支持批处理和多种 Tensor 格式

### 3.3 使用场景二：训练自定义分词器

当你需要为特定领域（如医学、法律、小说）训练新分词器时，使用 `tokenizers` 库：

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. 初始化分词器（选择算法）
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# 2. 设置 Pre-Tokenizer（如何预处理文本）
tokenizer.pre_tokenizer = Whitespace()

# 3. 配置训练器
trainer = WordPieceTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=32768,  # 常用大小：16k, 32k, 64k
    continuing_subword_prefix="##"  # WordPiece 的子词前缀
)

# 4. 训练分词器
files = ["./corpus_1.txt", "./corpus_2.txt"]
tokenizer.train(files, trainer=trainer)

# 5. 保存分词器
tokenizer.save("my_custom_tokenizer.json")
```

**优势：**
- 完全控制分词流程（Normalizer → Pre-Tokenizer → Model → Post-Processor）
- 高性能 Rust 后端
- 训练好的分词器可集成回 `transformers`

### 3.4 库选择建议

| 场景 | 推荐库 | 典型操作 |
|------|--------|---------|
| **使用预训练模型** | `transformers` | `AutoTokenizer.from_pretrained()` |
| **训练新分词器** | `tokenizers` | `Tokenizer.train()` |
| **极速编码/解码** | `tokenizers` | 直接调用 Rust 后端 |
| **自定义分词规则** | `tokenizers` | 定制 Pre-Tokenizer/Normalizer |

**推荐工作流：**
1. **应用阶段**：用 `transformers` 加载和使用分词器
2. **定制阶段**：用 `tokenizers` 训练新分词器，然后导入 `transformers`

## 四、构建预训练数据集

### 4.1 数据准备两阶段

#### 阶段一：分词器训练数据

**目标**：训练高质量的分词器

**格式要求**：原始文本文件（TXT）

**数据量**：数百万至数十亿字符的代表性子集（无需全部数据）

**操作建议**：
- 从所有 TXT 文件中随机抽样
- 合并为一个或几个大文件
- 确保覆盖数据的词汇分布

#### 阶段二：预训练数据构建

**目标**：构建 Token ID 序列用于模型训练

**格式要求**：固定长度的 Token ID 块

**操作步骤**：
1. 使用训练好的分词器对所有原始数据分词
2. 将所有 Token ID 连接成连续序列
3. 切分成固定长度的块（如 512, 1024, 2048）

### 4.2 实际操作代码

```python
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from itertools import chain
import glob
import os

# 1. 加载自定义分词器
custom_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="my_custom_tokenizer.json"
)

# 2. 加载原始数据
raw_datasets = load_dataset(
    "text", 
    data_files={"train": glob.glob("./novels/*.txt")}
)

# 3. 分词函数
def tokenize_function(examples):
    return custom_tokenizer(examples["text"])

# 4. 批量分词
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=["text"],
)

# 5. 分块（Chunking）- 核心步骤
block_size = 1024

def group_texts(examples):
    # 连接所有文本
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # 截断到 block_size 的倍数
    total_length = (total_length // block_size) * block_size
    
    # 切分成等长块
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=os.cpu_count(),
)

# 6. 保存数据集
lm_datasets["train"].save_to_disk("./preprocessed_dataset")
```

### 4.3 因果语言模型 (CLM) vs 掩码语言模型 (MLM)

虽然两种模型的底层数据格式相同（分块的 Token IDs），但训练时的数据处理方式不同：

#### 因果语言模型（如 GPT, Llama）

**训练目标**：预测下一个词元 $t_{i+1}$，给定前面所有词元 $(t_1, \ldots, t_i)$

**数据格式**：
- **输入 (input_ids)**：$[t_1, t_2, t_3, \ldots, t_L]$
- **标签 (labels)**：$[t_2, t_3, t_4, \ldots, t_L, -100]$（左移一位）

**关键点**：
- 标签是输入的错位版本
- 使用 `DataCollatorForLanguageModeling(mlm=False)` **自动生成**

#### 掩码语言模型（如 BERT, RoBERTa）

**训练目标**：预测被掩盖的词元 $\hat{t}_i$，给定其他所有词元

**数据格式**：
- **输入 (input_ids)**：$[t_1, t_2, \text{[MASK]}, \ldots, t_L]$
- **标签 (labels)**：$[-100, -100, t_{\text{original}}, \ldots, -100]$

**关键点**：
- 只有被掩盖位置有标签
- 使用 `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)` **动态掩码**

### 4.4 统一数据格式 + 动态处理（推荐方案）

**核心思想**：只准备一种通用格式（分块的 Token IDs），在训练时根据模型类型动态转换。

**操作流程**：

1. **数据准备**：构建统一的分块数据集（仅包含 `input_ids` 和 `attention_mask`）

2. **CLM 训练配置**：
```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 因果语言模型
)
```

3. **MLM 训练配置**：
```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # 掩码语言模型
    mlm_probability=0.15  # 掩码概率
)
```

**优势**：
- 数据预处理只需执行一次
- 灵活切换训练任务
- 符合主流 LLM 训练实践

## 五、特殊 Token 的设计与应用

### 5.1 常见特殊 Token 分类

| Token 类型 | 示例 | 作用 | 必需性 |
|-----------|------|------|--------|
| **未登录词** | `[UNK]` | 替换词汇表外的词 | 必须 |
| **填充** | `[PAD]` | 批处理时对齐序列长度 | 必须 |
| **分类/起始** | `[CLS]`, `<s>` | 标记序列开始 | 视架构 |
| **分隔** | `[SEP]`, `</s>` | 分隔文本段或标记结束 | 视架构 |
| **掩码** | `[MASK]` | MLM 任务专用 | 仅 MLM |

### 5.2 现代 LLM 的控制 Token

现代大模型（如 GPT-3, Llama, Qwen）引入了更多**控制 Token**用于结构化输入：

| Token | 用途 | 典型场景 |
|-------|------|---------|
| `<|endoftext|>` | 文档边界标记 | 防止跨文档信息泄漏 |
| `<|im_start|>`, `<|im_end|>` | 消息边界 | 多轮对话、指令微调 |
| `<|system|>`, `<|user|>`, `<|assistant|>` | 角色标识 | 对话系统 |

### 5.3 对话模板示例

```
<|im_start|>system
你是一个乐于助人的 AI 助手。<|im_end|>
<|im_start|>user
请解释黑洞的形成原理。<|im_end|>
<|im_start|>assistant
黑洞是由大质量恒星坍缩形成的...<|im_end|>
```

**作用**：
- 明确区分系统指令、用户输入和模型回复
- 使模型学习对话结构
- 提高指令跟随能力

### 5.4 设置特殊 Token 的建议

**针对因果语言模型（如 GPT 风格）：**
```python
special_tokens = ["[UNK]", "[PAD]", "<s>", "</s>", "<|endoftext|>"]
```

**针对掩码语言模型（如 BERT 风格）：**
```python
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
```

**针对对话模型（如 ChatGPT 风格）：**
```python
special_tokens = [
    "[UNK]", "[PAD]", 
    "<|im_start|>", "<|im_end|>",
    "<|system|>", "<|user|>", "<|assistant|>"
]
```

**关键原则**：
1. 在分词器训练时添加特殊 Token
2. 确保它们被视为不可分割的单元
3. 与基础模型规范保持一致

## 六、指令微调（Instruction Tuning）

### 6.1 LLM 训练的三阶段

| 阶段 | 名称 | 目的 | 数据类型 |
|------|------|------|---------|
| **阶段 1** | 预训练 | 学习语言的语法、语义和世界知识 | 原始文本 |
| **阶段 2** | 指令微调 | 转化为遵循指令的助手 | (指令, 回复) 对 |
| **阶段 3** | RLHF/DPO | 对齐人类偏好 | 偏好数据 |

### 6.2 指令微调的核心作用

#### 1. 提升指令跟随能力

**问题**：预训练模型只会"续写"，不会"回答"

**解决**：通过 (指令, 期望回复) 对训练，模型学会：
- 识别指令意图
- 遵循格式限制
- 生成针对性的回答而非续写

#### 2. 提升泛化能力

**作用**：显著提高零样本/少样本学习能力

**原理**：从大量不同任务中学习共性结构

#### 3. 结构化对齐

**作用**：使模型理解对话模板和角色标记

**示例**：学会何时处理请求（`user`）、何时生成回复（`assistant`）

### 6.3 指令微调数据集要求

#### 格式要求

每个样本必须是**结构化的交互历史**：

```
<|im_start|>user
请将以下句子翻译成英文：这是一本好书。<|im_end|>
<|im_start|>assistant
This is a good book.<|im_end|>
```

#### 内容要求（多样性是关键）

1. **任务多样性**：
   - 问答、摘要、翻译、代码生成、推理等
   - 目标：学习任务的一般性模式

2. **风格多样性**：
   - 简洁、详细、正式、非正式等不同表达
   - 目标：提高对不同沟通风格的理解

3. **角色多样性**：
   - 单轮、多轮、带系统指令等
   - 目标：处理复杂对话历史

4. **质量要求**：
   - 回复必须准确、安全、有用
   - 避免错误信息和有害内容

#### 数据集格式示例

**JSON 格式（Alpaca 风格）：**
```json
{
  "instruction": "将下列句子翻译成英文",
  "input": "这是一本好书。",
  "output": "This is a good book."
}
```

**对话格式（ShareGPT 风格）：**
```json
{
  "conversations": [
    {"from": "system", "value": "你是一个翻译助手。"},
    {"from": "user", "value": "翻译：这是一本好书。"},
    {"from": "assistant", "value": "This is a good book."}
  ]
}
```

### 6.4 指令标签定义原则

**约定优先于自由**：

1. **遵循基础模型规范**：
   - 使用开源模型的官方标签（如 Llama 的 `<|start_header_id|>`）
   - 确保与基础模型训练时的格式一致

2. **保证唯一性**：
   - 标签必须是词汇表中唯一的 Token
   - 使用特殊符号封装（如 `<|...|>`）

3. **Tokenization 兼容性**：
   - 在分词器训练时添加为 Special Tokens
   - 确保被视为不可分割的单元

## 七、最佳实践与总结

### 7.1 分词器选择决策树

```
是否有现成的预训练模型可用？
├── 是 → 使用 transformers.AutoTokenizer
└── 否 → 是否需要多语言/无空格语言支持？
    ├── 是 → 使用 SentencePiece (ULM)
    └── 否 → 领域特定语料
        ├── 英文为主 → WordPiece 或 BPE
        └── 效率优先 → BPE
```

### 7.2 数据准备最佳实践

1. **分词器训练**：
   - 使用代表性数据子集（数百 MB 至数 GB）
   - 词汇表大小：16k-64k（平衡效率和细粒度）
   - 添加必要的特殊 Token

2. **预训练数据**：
   - 统一格式：分块的 Token ID 序列
   - 使用 DataCollator 动态生成任务特定格式
   - 确保序列长度一致（如 1024, 2048）

3. **指令微调数据**：
   - 确保任务、风格、角色的多样性
   - 高质量回复（准确、安全、有用）
   - 使用标准对话模板

### 7.3 常见陷阱与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 词汇表过大/过小 | 参数设置不当 | 16k-32k 是通用起点，根据数据调整 |
| OOV 率高 | 训练数据与应用数据分布不一致 | 使用目标领域数据训练分词器 |
| 多语言性能差 | 使用了依赖预分词的算法 | 改用 SentencePiece (ULM) |
| 特殊 Token 被拆分 | 未在训练时添加 | 在 `special_tokens` 参数中明确添加 |
| 指令跟随能力差 | 指令微调数据质量低 | 提高数据多样性和回复质量 |

### 7.4 核心要点回顾

1. **算法演进**：BPE（频率）→ WordPiece（似然度）→ ULM（概率+修剪）
2. **库分工**：`tokenizers`（训练）+ `transformers`（应用）
3. **数据统一**：一种格式 + DataCollator 动态处理
4. **特殊 Token**：必须在训练时添加，确保唯一性和完整性
5. **指令微调**：预训练后的关键步骤，多样性决定泛化能力

## 八、参考资源

### 核心论文

- **BPE**: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- **WordPiece**: [Japanese and Korean Voice Search](https://research.google/pubs/pub37842/)
- **SentencePiece**: [SentencePiece: A simple and language independent approach to subword tokenization](https://arxiv.org/abs/1808.06226)
- **Unigram LM**: [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959)

### 工具文档

- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [SentencePiece GitHub](https://github.com/google/sentencepiece)

### 推荐阅读

- [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)
- [Byte Pair Encoding is Suboptimal for Language Model Pretraining](https://arxiv.org/abs/2004.03720)
- [FLAN: Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)

---

**本文档持续更新中。如有疑问或建议，欢迎提交 Issue 或 PR。**
