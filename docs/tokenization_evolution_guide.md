# 深入理解大语言模型的分词：从 BPE 到 Unigram 的演进之路

## 分词的本质：压缩与泛化的根本矛盾

神经网络只能处理数字，这是所有深度学习的起点。当我们想让模型理解文本时，必须将文本转换为数字序列。最直接的方案是**字符级编码**——为每个字符分配一个 ID，'a' → 1, 'b' → 2。这种方案简单优雅，词汇表极小（英文仅需 26 个字母加标点），永远不会遇到未登录词（OOV）。

但字符级编码有致命缺陷。一篇 1000 词的文章，平均每词 5 个字符，就是 5000 个 token。Transformer 的自注意力机制复杂度是 O(n²)，序列长度增加 5 倍，计算量暴涨 25 倍。更严重的问题是**信息密度过低**——模型必须从 'c','a','t' 三个独立符号中学习"猫"这个概念，而不是直接从一个完整的符号学习。这就像让你通过观察每个字母的笔画来理解文章，效率极其低下。

另一个极端是**词级编码**。将每个完整单词映射为一个 token，"cat" 就是一个 token，信息密度极高。但英语常用词汇数万，加上专业术语、人名地名，词汇表可能膨胀到百万级。这带来两个灾难性后果：首先是**参数爆炸**——词嵌入矩阵大小是 `vocab_size × embedding_dim`，百万词汇表配合 768 维嵌入，仅词嵌入层就需要 7.68 亿参数；其次是**OOV 问题**——任何新词、罕见词、错别字都无法处理。"COVID-19" 在 2019 年前的模型中就是 OOV。

**分词算法要在这两个极端之间找到平衡点**。我们需要一个适中的词汇表（通常 3-5 万），它应该能通过子词组合表示任意文本。从信息论角度看，这是一个数据压缩问题：如何用固定大小的"码本"（词汇表）高效编码无限可能的"消息"（文本），同时保证编码后的 token 序列既不会太长（计算效率），也不会丢失太多信息（表达能力）。

## BPE：简单而强大的贪婪算法

### 从数据压缩到自然语言处理

BPE (Byte Pair Encoding) 最初是 1994 年提出的数据压缩算法，核心思想极其简单：反复找出数据中最常出现的相邻字节对，用一个新字节替换它们。2016 年，Sennrich 等人将其引入 NLP，开启了子词分词的时代。

让我们通过一个具体例子理解这个过程。假设训练语料中有词 "low"、"lower"、"newest"、"widest"，初始状态下每个词被拆分为字符序列，并在词尾加上特殊标记 `</w>`（标记词边界）：

```
low</w>: 5次
lower</w>: 2次  
newest</w>: 6次
widest</w>: 3次
```

**第一轮迭代**：统计所有相邻符号对的频率。假设 'e' 和 's' 相邻出现最频繁（在 "newest" 和 "widest" 中），我们将它们合并为新符号 'es'：

```
low</w>: 5次
lower</w>: 2次
newest</w>: 6次  → new es t</w>: 6次
widest</w>: 3次  → wid es t</w>: 3次
```

**第二轮迭代**：继续统计，发现 'es' 和 't' 最频繁（9次），合并为 'est'：

```
low</w>: 5次
lower</w>: 2次
new est</w>: 6次
wid est</w>: 3次
```

这个过程持续进行，直到词汇表达到预定大小（如 32000）。最终，常见的词根、词缀（如 "un-", "-ing", "-tion"）会自然地成为独立的 token。

### 数学本质：最小化平均编码长度

BPE 在求解一个优化问题。设训练语料为 C，词汇表为 V，词汇表大小为 k。用 V 编码语料中的词 w，得到 token 序列 T(w)。BPE 的目标是：

$$\min_{V, |V|=k} \mathbb{E}_{w \sim C}[|T(w)|]$$

即：在词汇表大小固定的约束下，最小化平均每个词需要的 token 数。这是一个组合优化问题，寻找全局最优解是 NP-hard 的。BPE 采用贪婪策略：每次选择当前能最大程度减少总 token 数的合并操作。

具体来说，合并符号对 (a, b) 后，语料中所有 "a 后接 b" 的位置都从 2 个 token 变成 1 个 token（新符号 ab）。如果这种情况出现 f(a,b) 次，合并后 token 总数减少 f(a,b)。因此 BPE 选择：

$$\arg\max_{(a,b)} f(a,b)$$

这是对全局最优解的一阶贪心近似。每一步都选择当前收益最大的合并，期望通过多步迭代逼近最优解。

### BPE 的成功与局限

BPE 在 GPT-2、GPT-3、RoBERTa 等主流模型中取得巨大成功，原因在于：

**数据驱动的自适应性**。BPE 完全基于统计，不依赖语言学规则，对任何语言都通用。高频模式自动被捕获——英语中的 "ing"、"tion"，中文中的"的"、"了"，代码中的常见函数名，都会自动成为独立 token。这种自适应性符合 Zipf 定律：少数词高频出现（用粗粒度整词表示），大量词低频出现（用细粒度子词组合）。

**优雅的 OOV 处理**。BPE 保证任何文本都能编码，最坏情况下退化到字符级。新词 "unbelievableness" 即使从未见过，也能分解为 `["un", "##believ", "##able", "##ness"]`，每个子词都可能在词汇表中。

然而，BPE 的贪婪本质带来一个根本性问题：**只考虑局部最优，忽略全局影响**。考虑这个例子：

- 语料中 "unrelated" 出现 1000 次
- "unreasonable" 出现 300 次
- "unreal" 出现 200 次

BPE 会优先合并 "un" + "related"（频率 1000），因为它当前频率最高。但从全局看，合并 "un" 和 "re" 形成 "unre" 可能更优——虽然 "unre" 的绝对频率只有 500（300 + 200），但它能在多个词中复用。"unrelated" 作为整体只能用于这一个词，泛化价值低。

**这个问题的深层原因是：BPE 没有考虑合并决策的"可复用性"**。一个合并的价值不仅在于它当前减少了多少 token，还在于它产生的新 token 在未来能被复用多少次。BPE 的评分函数 f(a,b) 只看当前频率，缺乏预见性。结果是词汇表中充斥着"过拟合"的 token——它们在训练集中高频，但只是偶然共现，在新数据中很少出现，降低了词汇表效率和模型泛化能力。

## WordPiece：从频率到关联性的本质跃迁

### 重新定义"好" token 的标准

BPE 的问题启发我们思考：**什么样的符号对才真正值得合并**？WordPiece（Google 2012 年提出，BERT 采用）给出了一个深刻的答案：不是绝对频率高的，而是**相对频率高**的——即共现频率相对于独立期望有多高。

让我们通过对比来理解这个思想。假设语料总量 N = 100000：

- 情况 A：符号 a 出现 10000 次，b 出现 10000 次，ab 连续出现 1000 次
- 情况 B：符号 c 出现 100 次，d 出现 100 次，cd 连续出现 80 次

**BPE 的选择**：Score(a,b) = 1000 > Score(c,d) = 80，选择合并 (a,b)。

**WordPiece 的评分**：

$$\text{Score}(a, b) = \frac{f(ab) \cdot N}{f(a) \cdot f(b)} = \frac{P(ab)}{P(a)P(b)}$$

- Score(a,b) = $\frac{1000 \times 100000}{10000 \times 10000} = 1.0$
- Score(c,d) = $\frac{80 \times 100000}{100 \times 100} = 800.0$

WordPiece 选择合并 (c,d)！

**为什么？** 在情况 A 中，如果 a 和 b 是独立出现的，期望共现次数是 $\frac{10000 \times 10000}{100000} = 1000$。实际共现恰好是 1000，说明 a 和 b 的共现完全可以用随机性解释，它们没有特殊的"绑定关系"。

在情况 B 中，c 和 d 独立时期望共现仅 $\frac{100 \times 100}{100000} = 0.1$ 次。实际共现 80 次，是期望的 800 倍！这表明 c 和 d 几乎总是一起出现，有极强的关联性。**从语义角度，(c,d) 才是真正应该成为不可分割单元的组合**。

### 点互信息与似然度：两个视角的统一

WordPiece 的评分函数实际上是**点互信息 (PMI)** 的指数形式。PMI 是信息论中衡量两个事件相关性的经典指标：

$$\text{PMI}(a,b) = \log \frac{P(ab)}{P(a)P(b)}$$

PMI > 0 表示正相关（共现超出期望），PMI = 0 表示独立，PMI < 0 表示负相关（互斥）。WordPiece 使用 $e^{\text{PMI}} = \frac{P(ab)}{P(a)P(b)}$，本质上在最大化符号对之间的统计关联性。

从另一个角度看，WordPiece 在最大化训练语料的似然度。假设我们用 Unigram 语言模型（token 独立出现），语料的对数似然是：

$$\log P(\text{corpus}) = \sum_{i=1}^{m} \log P(t_i)$$

合并符号对 (a,b) 为新 token ab，语料中有 k 个 "ab" 序列从 2 个 token 变成 1 个。似然度的变化是：

$$\Delta \log P = k \cdot [\log P(ab) - \log P(a) - \log P(b)] = k \cdot \log \frac{P(ab)}{P(a)P(b)}$$

由于 $k \propto f(ab)$，最大化似然增益等价于最大化 $f(ab) \cdot \frac{P(ab)}{P(a)P(b)}$。在实践中简化为：

$$\arg\max \frac{f(ab)}{f(a) \cdot f(b)}$$

**WordPiece 的每次合并都在贪婪地增加整个语料库在 Unigram 模型下的对数似然度**。这不仅仅是启发式，而是有明确的概率论基础。

### 词汇表质量的提升

WordPiece 相比 BPE 的优势在实践中体现为：

**更强的语义一致性**。前缀 "un-"、后缀 "-ing"/"-tion" 在不同词中反复出现（unhappy, unfair, unreal; running, jumping, walking），它们的 PMI 很高，会被 WordPiece 优先识别为独立 token。模型因此更容易学习这些子词的语义（"un-" 表否定，"-ing" 表进行时），泛化能力提升。相比之下，BPE 可能因为某个包含这些词缀的完整词恰好高频，而将整词作为 token，错失了可复用的语义单元。

**对采样的鲁棒性**。BPE 对训练数据的采样敏感——如果某个词在采样中恰好高频，可能影响合并决策。WordPiece 关注的是关联性而非绝对频率，对采样波动更稳定。这使得用相对小的代表性数据就能训练出高质量分词器。

**显式的词边界信息**。WordPiece 使用 `##` 标记词内部的子词（`play` + `##ing`），而 BPE 通常用 `_` 标记词首（`_play` + `ing`）。虽然都能工作，但 `##` 标记更直观地表达了"这是某个词的一部分"，帮助模型理解词法结构。BERT 的成功部分归功于 WordPiece——在相同词汇表大小下，它生成的 token 语义质量更高，让模型能更高效地学习语言规律。

## Unigram Language Model：范式转换与全局优化

### 从合并到修剪：算法设计的重新思考

BPE 和 WordPiece 都采用**自底向上**的策略：从字符开始，逐步合并。Unigram LM（Kudo 2018）完全反其道而行：**自顶向下**——从一个非常大的初始词汇表开始，逐步删除不重要的子词，直到达到目标大小。

这个范式转换背后有深刻的数学动机。自底向上的合并是离散的、不可逆的决策——一旦合并了 (a,b)，就无法回退。而自顶向下的修剪可以在每一步重新评估所有子词的价值，通过概率模型进行全局优化。

### EM 算法：概率建模的力量

Unigram LM 的核心是一个概率生成模型。给定词汇表 V，每个子词 x 有一个概率 P(x)。一个词 w 可能有多种切分方式（如 "playing" 可以是 `[play, ing]` 或 `[p, lay, ing]`），每种切分 S = [s₁, s₂, ..., sₖ] 的概率是：

$$P(S) = \prod_{i=1}^{k} P(s_i)$$

词 w 的概率是所有可能切分的概率之和：

$$P(w) = \sum_{S \in \text{Segmentations}(w)} P(S)$$

训练目标是最大化训练语料的对数似然：

$$\mathcal{L} = \sum_{w \in \text{corpus}} \log P(w)$$

这是一个典型的 EM (Expectation-Maximization) 问题：

**E-step**：给定当前的子词概率 P(x)，计算每个词的最优切分（用 Viterbi 算法）。

**M-step**：基于所有词的最优切分，重新估计每个子词的概率——子词 x 的新概率正比于它在所有最优切分中出现的频率。

**修剪步骤**：在 EM 迭代过程中，周期性地评估每个子词的重要性。对于子词 x，计算如果将其从词汇表中移除，似然度 $\mathcal{L}$ 会下降多少。下降最小的子词是最"冗余"的——其他子词的组合可以很好地替代它。每次修剪删除损失最小的 10-20% 子词，然后重新运行 EM 更新概率。

这个过程持续到词汇表达到目标大小（如 32000）。最终保留的是那些**对编码语料贡献最大**的子词。

### Viterbi 解码：概率切分的优雅

推理时，给定词 w，Unigram 不使用贪婪匹配（总是选最长的匹配子词），而是用 Viterbi 算法找概率最大的切分：

$$S^* = \arg\max_{S} P(S) = \arg\max_{S} \prod_{i} P(s_i)$$

实践中通过动态规划求解，时间复杂度 O(n²)，n 是词长。

**具体例子**：假设词汇表和概率为：

```
_play: 0.20    _p: 0.15    _lay: 0.10    ing: 0.08
```

对词 "_playing"，可能的切分有：
- `[_play, ing]`：概率 = 0.20 × 0.08 = 0.016
- `[_p, lay, ing]`：概率 = 0.15 × 0.10 × 0.08 = 0.0012

Viterbi 选择概率更高的 `[_play, ing]`。

更有趣的是，Unigram 支持**概率采样**：训练时不总是选最优切分，而是按概率分布随机采样。对 "_playing"，有 $\frac{0.016}{0.016+0.0012} \approx 93\%$ 概率选切分 1，7% 概率选切分 2。这种随机性是强大的**正则化机制**——模型在训练中见到同一个词的不同切分方式，提高了对分词边界变化的鲁棒性。这类似于 Dropout 的思想：通过引入受控的随机性，防止过拟合。

### 全局优化的优势

相比 BPE 和 WordPiece 的贪婪策略，Unigram 的优势在于：

**理论保证**。EM 算法有明确的目标函数（最大化似然），修剪过程是有原则的近似。每一步都在优化全局目标，而不是局部启发式。

**词汇表的自洽性**。在收敛的 Unigram 模型中，每个子词的概率与它在语料最优切分中的使用频率一致。这种自洽性意味着词汇表是"平衡"的——没有过度重要或过度冗余的子词。

**灵活性**。概率模型允许我们引入额外的约束或正则项。例如，可以惩罚过长的子词（鼓励更细粒度的切分），或者偏向某些特定模式的子词（如鼓励保留完整的词根）。这些在 BPE/WordPiece 的离散框架下很难实现。

正因为这些优势，Unigram 被现代多语言模型广泛采用——T5、XLNet、ALBERT、Llama 等都使用 Unigram 分词器。

## SentencePiece：统一多语言的工程实现

### 预分词的困境

BPE 和 WordPiece 在引入 NLP 时，都假设文本已经被分割成词（按空格或标点）。这在英语等有明确词边界的语言中可行，但在**无空格语言**（中文、日文、泰文）中失效。我们可以为每种语言开发特定的分词器（如中文的 jieba），但这带来新问题：

1. **维护负担**：多语言模型需要集成多种分词器，每种都有自己的规则和依赖。
2. **不一致性**：不同语言的预分词质量不同，可能导致模型对某些语言的效果特别差。
3. **信息丢失**：预分词器可能犯错（如将"iPhone"分成"i Phone"），且这些错误不可逆。

### 统一的解决方案

SentencePiece（Google 2018，Kudo 和 Richardson）提出了一个优雅的解决方案：**将文本视为原始字符流，包括空格**。

核心思想是：空格不是特殊的分隔符，而是普通字符，用特殊符号 `▁` 表示。例如：

```
"Hello world" → ["▁Hello", "▁world"]
"你好世界"   → ["▁你好", "▁世界"]
```

这实现了三个关键性质：

**语言无关性**。无论英语、中文还是代码，都用相同的方式处理——扫描字符流，应用 BPE/Unigram 算法。不需要语言特定的预分词规则。

**可逆性**。由于空格被显式编码为 `▁`，从 token 序列可以精确还原原始文本，包括所有空格位置。这对于某些应用（如代码生成）至关重要。

**训练-推理一致性**。没有预分词意味着训练和推理的行为完全一致，不会因为预分词器版本不同或配置差异导致结果不同。

SentencePiece 支持 BPE 和 Unigram 两种算法，但**官方推荐 Unigram**——因为 Unigram 的概率框架更适合处理无分隔符的连续文本，且正则化采样在多语言场景下特别有效。T5、Llama、Qwen 等现代多语言模型都采用 SentencePiece + Unigram 的组合，这已成为事实标准。

## 从理论到实践：训练自定义分词器

### 理解特殊 Token 的作用

在训练分词器之前，需要明确模型类型和对应的特殊 token。不同架构需要不同的控制 token：

**因果语言模型**（GPT 系列）需要序列边界标记：
- `<s>` (BOS, Beginning of Sequence)：标记序列开始
- `</s>` (EOS, End of Sequence)：标记序列结束，也是生成停止的信号
- `<unk>`：未知词，`<pad>`：填充

**掩码语言模型**（BERT 系列）需要额外的任务标记：
- `[MASK]`：掩码位置，MLM 训练的核心
- `[CLS]`：分类 token，放在序列开头，其表示用于下游分类任务
- `[SEP]`：分隔符，用于分隔句子对（如问答中的问题和上下文）

**对话模型**（ChatGPT、Llama-Chat）需要结构化标记：
- `<|im_start|>`, `<|im_end|>`：消息边界，标记每轮对话的起止
- `<|system|>`, `<|user|>`, `<|assistant|>`：角色标识，区分系统指令、用户输入和模型回复

这些 token 必须在分词器训练时作为 Special Tokens 添加，确保它们被视为不可分割的单元，不会被进一步拆分。

### 训练 Unigram 分词器

使用 Hugging Face 的 `tokenizers` 库训练 Unigram 分词器：

```python
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Metaspace

# 1. 初始化 Unigram 模型
tokenizer = Tokenizer(Unigram())

# 2. 设置预处理器（SentencePiece 风格）
tokenizer.pre_tokenizer = Metaspace(replacement="▁", add_prefix_space=True)

# 3. 配置训练器
trainer = UnigramTrainer(
    vocab_size=32000,
    special_tokens=["<unk>", "<s>", "</s>"],
    unk_token="<unk>",
)

# 4. 训练（使用代表性语料抽样）
files = ["corpus_sample_1.txt", "corpus_sample_2.txt"]
tokenizer.train(files, trainer=trainer)

# 5. 保存
tokenizer.save("custom_unigram.json")
```

**关键参数**：
- **vocab_size**：16k-32k 适合大多数应用，长上下文模型可用 64k。
- **抽样策略**：分词器训练不需要全部语料（那是模型预训练的事），数百 MB 的代表性数据即可。目标是捕获词汇分布，而不是学习语义。
- **Metaspace**：SentencePiece 的空格处理方式，保证可逆性。

训练完成后，可以集成到 Transformers：

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="custom_unigram.json",
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
)

# 使用
text = "大语言模型的分词技术"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)
```

### 数据集构建：统一格式的力量

构建预训练数据集时，一个常见疑问是：因果语言模型（CLM）和掩码语言模型（MLM）的训练数据格式不同，是否需要两套数据？

答案是：**只需一套数据**——分块的 Token ID 序列。训练时用 `DataCollator` 动态转换为特定格式。

**数据准备流程**：

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
block_size = 2048

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

**CLM 训练**（GPT 风格）：

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
```

DataCollator 自动将 `input_ids` 左移一位生成 `labels`：
- Input: `[t₁, t₂, t₃, ..., tₙ]`
- Labels: `[t₂, t₃, t₄, ..., tₙ, -100]`

**MLM 训练**（BERT 风格）：

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)
```

DataCollator 动态随机掩码 15% 的 token：
- Input: `[t₁, [MASK], t₃, [MASK], t₅]`
- Labels: `[-100, t₂, -100, t₄, -100]`

这种**统一数据 + 动态转换**的方案是目前的最佳实践。数据预处理只需做一次，训练时灵活切换任务。动态掩码还带来正则化效果——每个 epoch 看到不同的掩码模式，防止过拟合。

## 指令微调：从预训练到对话的桥梁

现代 LLM 的训练分为三个阶段：**预训练** → **指令微调** → **人类反馈对齐**。分词器在前两个阶段都起关键作用。

### 控制 Token 的结构化作用

预训练让模型学习语言的统计规律，但它只会"续写"，不会"回答"。给它 "What is quantum entanglement?"，预训练模型可能续写成另一个问题或随机文本，而不是给出答案。

指令微调通过在 (指令, 回复) 对上训练，教会模型理解任务结构。这需要**控制 token** 来标记不同的角色和边界：

```
<|im_start|>system
你是一个专业的物理学家。<|im_end|>
<|im_start|>user
解释量子纠缠。<|im_end|>
<|im_start|>assistant
量子纠缠是指两个或多个粒子的量子态相互关联...<|im_end|>
```

这些控制 token（`<|im_start|>`, `<|im_end|>`, `<|system|>`, `<|user|>`, `<|assistant|>`）必须在分词器训练时添加。在指令微调时，模型学习：
- 看到 `<|user|>` 后应该理解这是一个请求
- 看到 `<|assistant|>` 后应该生成回复而非续写
- `<|im_end|>` 标记一轮对话结束

### 数据质量的重要性

指令微调的数据要求远高于预训练：

**任务多样性**：覆盖问答、翻译、摘要、代码、推理等，让模型学习任务的一般性模式。

**风格多样性**：包含简洁/详细、正式/口语、友好/中性等不同风格，提高鲁棒性。

**高质量回复**：每个回复必须准确、安全、有用。模型会直接学习这些回复的模式——如果训练数据中有错误信息或有害内容，模型会学习这些不良行为。

**格式一致性**：所有数据必须用统一的控制 token 格式。如果混用不同格式（有的用 `<user>`，有的用 `[User]`），模型会混淆。

## 总结：算法选择与最佳实践

### 分词器选择决策

选择分词器时，考虑三个因素：

**语言类型**：
- 有明确词边界（英语、德语）→ BPE/WordPiece 都可
- 无空格语言（中文、日文）→ 必须用 SentencePiece

**质量要求**：
- 追求词汇表语义质量 → WordPiece（PMI 评分捕获语义单元）
- 追求全局优化和正则化 → Unigram（EM 算法，概率采样）

**工程实践**：
- 使用现有预训练模型 → 用其配套分词器（`AutoTokenizer`）
- 训练新模型 → 推荐 SentencePiece + Unigram（T5/Llama 的选择）

### 核心要点

**BPE**：简单高效，贪婪最小化编码长度。局限在于局部优化，可能产生过拟合 token。适合快速原型和简单场景。

**WordPiece**：优化似然增益，评分基于 PMI。捕获语义关联性强于 BPE，词汇表质量更高。BERT 的成功验证了其有效性。

**Unigram**：范式转换（修剪而非合并），EM 全局优化，支持概率采样正则化。理论基础最强，适合大规模多语言模型。

**SentencePiece**：工程实现，语言无关，可逆性强。与 Unigram 结合是现代 LLM 的主流选择。

**实践建议**：
- 词汇表大小：通用场景 16k-32k，长上下文或多语言 64k
- 训练数据：抽样数百 MB 代表性语料即可
- 特殊 token：在训练时添加，确保与模型架构匹配
- 评估指标：关注压缩率（平均 token 长度）和下游任务性能

分词是 LLM 的第一步，也是容易被忽视的一步。但正如本文所示，从 BPE 的简单贪婪到 Unigram 的全局优化，算法设计的每一个细节都影响着最终模型的质量。深入理解分词原理，选择合适的算法，是构建高性能 LLM 的关键基础。

## 参考文献

- Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units". *ACL 2016*.
- Schuster, M., & Nakajima, K. (2012). "Japanese and Korean Voice Search". *ICASSP 2012*.
- Kudo, T. (2018). "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates". *ACL 2018*.
- Kudo, T., & Richardson, J. (2018). "SentencePiece: A simple and language independent approach to subword tokenization". *EMNLP 2018*.
- Hugging Face Tokenizers 文档：https://huggingface.co/docs/tokenizers/
