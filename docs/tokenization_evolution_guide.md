# 深入理解大语言模型的分词：从信息论到概率模型

## 分词的本质：在压缩率与泛化性之间的权衡

分词不只是技术细节，它是 LLM 的信息瓶颈。神经网络只能处理数字，我们必须将文本映射为 token 序列。这个映射面临根本性矛盾：

**字符级编码**简单但低效。一个 1000 词的文章变成 5000+ 字符，Transformer 的 O(n²) 复杂度让计算量暴涨 25 倍。更致命的是信息密度过低——模型必须从 'c','a','t' 三个独立符号学习"猫"的概念。

**词级编码**信息密度高但词汇表爆炸。英语常用词就数万个，词嵌入矩阵 `vocab_size × dim` 会膨胀到数亿参数。新词、罕见词、错别字全是 OOV（未登录词）。

分词算法的核心使命：**用 3-5 万的词汇表，通过子词组合表示任意文本**。这本质上是数据压缩问题——如何用固定大小的"码本"高效编码无限可能的"消息"。

## BPE：贪婪压缩的得与失

BPE (Byte Pair Encoding) 源自 1994 年的数据压缩算法，2016 年被引入 NLP。核心思想极简：**迭代合并最高频的相邻符号对**。

### 算法直觉

初始状态：所有词拆分为字符。
迭代过程：统计相邻符号对频率 → 合并最高频对 → 更新词汇表。

例如：`"aaabdaaabac"` → 合并 `aa`(4次) 得 `Z` → `"ZabdZabac"` → 合并 `ab`(2次) 得 `Y` → `"ZYdZYac"`。

高频词根（"ing", "tion"）自然浮现为独立 token。

### 数学本质：最小化编码长度

BPE 隐式求解：

$$\min_{V, |V|=k} \mathbb{E}_{w \sim \text{corpus}}[|T(w)|]$$

即用 k 大小的词汇表 V 编码语料，最小化平均 token 数。这是 NP-hard 问题，BPE 用贪婪策略逼近：每次选择能最大减少总 token 数的合并。

合并符号对 $(a, b)$ 后，token 总数减少 $f(a,b)$（从 2 个变 1 个）。BPE 选 $\arg\max f(a,b)$，是最优解的一阶近似。

### 成功之处

**1. 数据驱动，语言无关**：无需语言学知识，英语/中文/代码通用。高频模式自动捕获。

**2. 优雅处理 OOV**：任何文本可编码，最坏退化到字符级。新词 "unbelievableness" 分解为 `["un", "believ", "able", "ness"]`。

**3. 自适应粒度**：高频词粗粒度（整词），低频词细粒度（子词），符合 Zipf 定律。

### 致命缺陷：局部最优的陷阱

BPE 只看**当前**频率，忽略**未来**复用价值。

反例：假设语料中
- "un" + "related" 出现 1000 次
- "unre" 在其他词（unreasonable, unreal）中各出现 500 次

BPE 优先合并 "unrelated"（频率 1000），但 "unre" 虽频率低（500），却能在多个词中复用。**全局看，"unre" 更有价值**。

结果：BPE 词汇表充斥"过拟合" token——训练集高频但泛化性差的偶然组合。

## WordPiece：从频率到关联性的跃迁

### 核心洞察：什么才是"好" token？

WordPiece (Google 2012，BERT 采用) 只改了一个评分函数，却体现深刻统计学思想。

**问题**：频率高 = 应该合并？

对比两种情况：
- 情况 A：$(a, b)$ 出现 1000 次，但 $a$ 出现 10000 次，$b$ 出现 10000 次
- 情况 B：$(c, d)$ 出现 80 次，但 $c$ 出现 100 次，$d$ 出现 100 次

BPE 选 A（频率 1000 > 80）。但仔细想：A 中 $a,b$ 多数时候独立出现，共现可能是随机的；B 中 $c,d$ 几乎总一起出现，有强绑定关系。**语言学上，B 更应合并**。

**WordPiece 关注相对频率**：共现频率相对于独立期望有多高。

### 评分函数：点互信息的指数形式

$$\text{Score}(a, b) = \frac{P(ab)}{P(a)P(b)} = \frac{f(ab) \cdot N}{f(a) \cdot f(b)}$$

比值含义：
- **> 1**：共现超出独立期望，正相关
- **= 1**：符合独立假设，无特殊关系  
- **< 1**：共现低于期望，负相关

这正是点互信息 (PMI) 的指数：$\text{PMI}(a,b) = \log \frac{P(ab)}{P(a)P(b)}$。

WordPiece 最大化符号对的统计关联性，而非绝对频率。

### 似然度视角：最大化语料概率

在 Unigram 模型下，语料对数似然：$\log P = \sum_i \log P(t_i)$。

合并 $(a,b)$ 为 $ab$ 后，似然变化：

$$\Delta \log P = k \cdot \log \frac{P(ab)}{P(a)P(b)}$$

其中 $k = f(ab)$ 是合并次数。WordPiece **贪婪最大化似然增益**，每次合并都让语料在 Unigram 模型下更"可能"。

### 数值例子：WordPiece 的智慧

设语料总量 $N = 100000$：
- "a" 10000 次，"b" 10000 次，"ab" 1000 次
- "c" 100 次，"d" 100 次，"cd" 80 次

**BPE**：Score(a,b) = 1000 > Score(c,d) = 80 → 选 (a,b)

**WordPiece**：
- Score(a,b) = $1000 \times 100000 / (10000 \times 10000) = 1.0$
- Score(c,d) = $80 \times 100000 / (100 \times 100) = 800.0$ → 选 (c,d)

**分析**：
- (a,b) 独立期望共现 $10000 \times 10000 / 100000 = 1000$，实际恰好 1000，无关联。
- (c,d) 独立期望仅 $100 \times 100 / 100000 = 0.1$，实际 80，是期望 800 倍！

WordPiece 识别出 (c,d) 的强绑定，BPE 被绝对频率误导。

### BERT 的选择：语义一致性

WordPiece 生成的词汇表质量更高：

**1. 捕获语义单元**：前缀 "un-"、后缀 "-ing"/"-tion" 在不同词中反复出现，PMI 高，被识别为独立 token。模型更易学习语义（"un-" = 否定）。

**2. 鲁棒性强**：对语料采样变化不敏感。BPE 可能因采样差异选择不同的高频对，WordPiece 关注关联性更稳定。

**3. 前缀标记**：用 `##` 标记词内子词（`play` + `##ing`），显式保留词边界信息，帮助模型理解词法结构。

## Unigram Language Model：从合并到修剪的范式转换

BPE/WordPiece 都是**自底向上**：从字符开始迭代合并。Unigram LM (Kudo 2018) 反其道而行：**自顶向下**，从大词汇表开始修剪。

### 训练流程：EM 算法 + 修剪

**步骤 1**：初始化大词汇表 $V_{\text{init}}$（所有字符 + 常见 N-gram）。

**步骤 2**：EM 算法估计每个子词概率 $P(x)$。
- E-step：给定当前 $P(x)$，计算每个词的最优切分
- M-step：基于切分结果，更新 $P(x)$

**步骤 3**：修剪循环（至词汇表达目标大小 $K$）：
1. 对每个子词 $x$，计算移除它后的似然损失 $\mathcal{L}(x)$
2. 删除损失最小的 $\eta\%$（如 20%）
3. 重新运行 EM 更新概率

**核心思想**：保留对编码贡献大的子词，删除冗余的。

### 推理：Viterbi 解码的概率切分

给定词 $W$，Unigram 不用贪婪匹配，而是找**概率最大**的切分：

$$S^* = \arg\max_{S} \prod_{i} P(s_i)$$

实践中用 Viterbi 算法（动态规划）高效求解。

**例子**：词汇表 `{_play: 0.2, _p: 0.15, _lay: 0.1, ing: 0.08}`，对 "_playing"：
- 切分 1：`[_play, ing]`，概率 $0.2 \times 0.08 = 0.016$
- 切分 2：`[_p, lay, ing]`，概率 $0.15 \times 0.1 \times 0.08 = 0.0012$

Viterbi 选概率更高的切分 1。

### 独特优势：正则化与鲁棒性

**1. 概率采样**：训练时不总选最优切分，而是按概率分布采样。如对 "_playing"，有 1.6% 概率选切分 1，0.12% 概率选切分 2。

这是**数据增强**：同一词的不同切分方式让模型见到更多变化，提高对分词边界的鲁棒性。

**2. 全局优化**：EM 算法考虑所有可能切分，而非贪婪选择。词汇表更"自洽"——每个子词的概率与其在最优切分中的使用频率一致。

**3. 理论保证**：最大化似然度有明确目标函数，修剪过程是有原则的近似，而非启发式。

## SentencePiece：语言无关的工程实现

### 为什么需要 SentencePiece？

BPE/WordPiece 依赖**预分词**（按空格/标点切分），这在英语等语言可行，但在**无空格语言**（中文、日文、泰文）失效。不同语言需要不同的预分词规则，导致：

1. **复杂性**：多语言模型需要维护多套规则
2. **信息丢失**：预分词可能引入错误，且不可逆
3. **不一致性**：训练和推理可能用不同预分词器

### SentencePiece 的解决方案

**核心**：将文本视为**原始字符流**，包括空格。空格用特殊符号 `▁` 表示。

例：`"Hello world"` → `["▁Hello", "▁world"]`

这实现了：
- **语言无关**：相同流程处理任何语言
- **可逆性**：token 序列可精确还原原文（包括空格位置）
- **统一性**：训练/推理行为一致

SentencePiece 支持 BPE/Unigram 两种算法，但**推荐 Unigram**（T5、XLNet、ALBERT、Llama 都用它）。

## 实战：训练自定义分词器

### 明确需求：特殊 Token 的设计

不同任务需要不同的特殊 token：

**因果语言模型** (GPT 风格)：
```python
special_tokens = ["<unk>", "<pad>", "<s>", "</s>"]
```
- `<s>`/`</s>`：序列边界，防止跨文档信息泄漏
- `<unk>`：未知词，`<pad>`：填充

**掩码语言模型** (BERT 风格)：
```python
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
```
- `[MASK]`：掩码位置
- `[CLS]`：分类 token，`[SEP]`：分隔符

**对话模型** (ChatGPT 风格)：
```python
special_tokens = [
    "<unk>", "<pad>",
    "<|im_start|>", "<|im_end|>",  # 消息边界
    "<|system|>", "<|user|>", "<|assistant|>"  # 角色标识
]
```

**关键原则**：
1. 在分词器训练时添加为 Special Tokens（确保不可分割）
2. 与基础模型规范一致（如使用 Llama 预训练模型，必须用其官方 token）
3. 保证唯一性（用 `<|...|>` 等特殊符号封装）

### 代码示例：训练 Unigram 分词器

```python
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Metaspace  # 处理空格

# 1. 初始化（Unigram 模型）
tokenizer = Tokenizer(Unigram())

# 2. 设置预处理器（保留空格信息）
tokenizer.pre_tokenizer = Metaspace(replacement="▁", add_prefix_space=True)

# 3. 配置训练器
trainer = UnigramTrainer(
    vocab_size=32000,  # 词汇表大小
    special_tokens=["<unk>", "<s>", "</s>"],
    unk_token="<unk>",
)

# 4. 训练（可用语料抽样，无需全部数据）
files = ["sample_corpus_1.txt", "sample_corpus_2.txt"]
tokenizer.train(files, trainer=trainer)

# 5. 保存
tokenizer.save("unigram_tokenizer.json")
```

**要点**：
- **vocab_size**：通常 16k-64k。太小导致序列长，太大导致参数膨胀。
- **抽样训练**：分词器训练只需捕获词汇分布，数百 MB 代表性数据即可，无需全量语料。
- **Metaspace**：SentencePiece 风格的空格处理，确保可逆性。

### 集成到 Transformers

```python
from transformers import PreTrainedTokenizerFast

# 加载自定义分词器
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="unigram_tokenizer.json",
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
)

# 现在可以像使用预训练分词器一样使用
text = "大语言模型的分词技术"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)
```

## 数据集构建：统一格式的智慧

### 核心理念：一种数据，多种用途

**问题**：因果语言模型 (CLM) 和掩码语言模型 (MLM) 的训练格式看似不同，是否需要两套数据？

**答案**：只需一种格式——**分块的 Token ID 序列**。训练时用 `DataCollator` 动态转换。

### 数据准备流程

```python
from datasets import load_dataset
from itertools import chain

# 1. 加载原始文本
raw_datasets = load_dataset("text", data_files={"train": "corpus/*.txt"})

# 2. 分词
def tokenize(examples):
    return tokenizer(examples["text"])

tokenized = raw_datasets.map(tokenize, batched=True, remove_columns=["text"])

# 3. 分块（关键步骤）
block_size = 2048  # 序列长度

def group_texts(examples):
    # 连接所有 token
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    
    # 截断到 block_size 倍数
    total_length = (total_length // block_size) * block_size
    
    # 切分成等长块
    result = {
        k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    return result

lm_datasets = tokenized.map(group_texts, batched=True)
```

**关键**：连续文本被切分成固定长度的块（如 2048），每个块是一个训练样本。

### 动态转换：DataCollator 的魔法

**CLM 训练**（GPT 风格）：
```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 不做掩码
)
```

DataCollator 自动生成 `labels`：将 `input_ids` 左移一位。
- Input: `[t1, t2, t3, ..., tn]`
- Labels: `[t2, t3, t4, ..., tn, -100]`

模型学习预测下一个 token。

**MLM 训练**（BERT 风格）：
```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15  # 15% token 被掩码
)
```

DataCollator 动态掩码：随机选 15% 位置替换为 `[MASK]`，生成对应 labels。
- Input: `[t1, [MASK], t3, t4, [MASK]]`
- Labels: `[-100, t2, -100, -100, t5]`（只有掩码位置有标签）

模型学习预测被掩盖的 token。

**优势**：
- 数据预处理一次，训练时灵活切换任务
- 动态掩码让每个 epoch 看到不同的掩码模式（正则化效果）
- 符合主流实践（Hugging Face 官方推荐）

## 深入理解：控制 Token 与指令微调

### 控制 Token 的作用

现代 LLM 不只预测下一个词，还要理解任务结构。**控制 token** 帮助模型区分不同角色和边界：

| Token | 用途 | 例子 |
|-------|------|------|
| `<|endoftext|>` | 文档边界 | 防止模型从上一篇文章的结尾预测下一篇文章的开头 |
| `<|im_start|>`, `<|im_end|>` | 消息边界 | 标记每轮对话的起止 |
| `<|system|>`, `<|user|>`, `<|assistant|>` | 角色标识 | 区分系统指令、用户输入、模型回复 |

**示例**：
```
<|im_start|>system
你是一个专业的 AI 助手。<|im_end|>
<|im_start|>user
解释量子纠缠。<|im_end|>
<|im_start|>assistant
量子纠缠是指两个粒子的状态相互关联...<|im_end|>
```

### 指令微调：从续写器到助手的质变

**预训练**：模型学习语言的统计规律（"the cat sat on the ___" → "mat"）。

**指令微调**：在 (指令, 回复) 对上训练，让模型学会：
1. **理解指令意图**（翻译、摘要、问答）
2. **遵循格式约束**（"用三句话回答"、"以列表形式"）
3. **生成针对性回复**（而非续写）

**数据要求**：
- **任务多样性**：覆盖问答、翻译、代码、推理等多种任务
- **风格多样性**：简洁/详细、正式/口语等不同表达
- **高质量回复**：准确、安全、有用（错误数据会被模型学习）

**数据格式**：
```json
{
  "conversations": [
    {"from": "system", "value": "你是翻译助手"},
    {"from": "user", "value": "翻译：这是一本好书"},
    {"from": "assistant", "value": "This is a good book."}
  ]
}
```

**关键**：指令标签必须与分词器的 special tokens 一致，且在训练时正确应用（用控制 token 封装每段对话）。

## 总结：选择与权衡

### 分词器选择决策

```
需要多语言/无空格语言支持？
├─ 是 → SentencePiece + Unigram
└─ 否 → 已有预训练模型可用？
    ├─ 是 → 使用其分词器（transformers.AutoTokenizer）
    └─ 否 → 领域特定语料？
        ├─ 追求语义质量 → WordPiece
        └─ 追求简单高效 → BPE
```

### 核心要点回顾

1. **BPE**：贪婪合并高频对，简单高效，但可能过拟合训练集。
2. **WordPiece**：最大化似然增益，关注符号关联性，语义质量更高。
3. **Unigram**：自顶向下修剪，EM 算法全局优化，支持概率采样正则化。
4. **SentencePiece**：语言无关实现，处理原始字符流，推荐 Unigram 算法。
5. **数据准备**：统一格式（分块 Token IDs）+ DataCollator 动态转换。
6. **控制 Token**：必须在分词器训练时添加，保证与模型规范一致。

### 实践建议

- **词汇表大小**：16k-32k 通用，长上下文模型可用 64k+。
- **训练数据**：分词器只需数百 MB 代表性抽样，无需全量语料。
- **评估指标**：关注平均 token 长度（compression ratio）和 OOV 率。
- **迭代优化**：先用默认配置快速验证，再根据任务特点调整（如代码任务可能需要更大词汇表）。

**分词是 LLM 的第一步，也是最容易被忽视的一步。但正如本文所示，选对分词算法并深入理解其原理，能为模型性能和训练效率带来实质提升。**

## 参考文献

- Sennrich et al. (2016). "Neural Machine Translation of Rare Words with Subword Units" - BPE 原始论文
- Schuster & Nakajima (2012). "Japanese and Korean Voice Search" - WordPiece 首次提出
- Kudo (2018). "Subword Regularization: Improving Neural Network Translation Models" - Unigram LM
- Kudo & Richardson (2018). "SentencePiece: A simple and language independent approach" - SentencePiece 实现
- Hugging Face Tokenizers 文档：https://huggingface.co/docs/tokenizers/
