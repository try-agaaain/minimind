# LLM 分词方式发展历程

**Exported:** 11/22/2025 23:27:08  
**Link:** [https://gemini.google.com/app/e1b6845760cbcd40](https://gemini.google.com/app/e1b6845760cbcd40)

## Prompt:
这种方式和BPE有区别吗

## Response:
WordPiece 和传统的 BPE (Byte Pair Encoding) 都是子词分词的经典算法，它们的目的都是通过迭代合并构建一个有限的词汇表，但它们在**选择合并对的标准**上存在根本区别。

| 特性  | WordPiece | 传统的 BPE (Byte Pair Encoding) |
| --- | --- | --- |
| **合并标准** | **基于似然度增益** (Likelihood Gain) | **基于频率** (Frequency) |
| **评分公式** | $$<br>Score\left(A,B\right)=\frac{Freq\left(AB\right)}{Freq\left(A\right)\cdot Freq\left(B\right)}<br>$$ | $$<br>Score\left(A,B\right)=Freq\left(AB\right)<br>$$ |
| **目标** | 最大化**整个训练语料**在 Unigram 语言模型下的对数似然度。 | 贪婪地找到**最常出现**的相邻字符/子词对进行合并。 |
| **偏好** | 更倾向于合并那些**共现强度高**（即合并后信息量大、非随机共现）的子词。 | 总是合并**出现次数最多**的相邻对，即使它们可能只是偶尔的随机共现。 |
| **前缀标记** | 通常使用 `##` 来标记一个子词是某个词的内部部分（例如 `play` 和 `##ing`）。 | 通常使用空格或下划线 `_` 来标记词的起始部分（例如 `_play` 和 `ing`）。 |
| **应用实例** | BERT, DistilBERT, MobileBERT 等。 | GPT-2/3, RoBERTa, 大多数早期的 Transformer 模型。 |

### 核心区别的解释

#### 1\. BPE：单纯的频率统计

BPE 算法采取最直接的贪婪策略：哪个相邻对  $\left(A,B\right)$  出现的次数最多，就合并哪个。

*   **优势：** 实现简单、速度快。
*   **劣势：** 可能导致词汇表中的某些 Token 只是高频的随机组合，而不是有语义价值的词根或词缀。例如，如果 `a` 和 `b` 都是非常高频的字符，即使它们相邻出现的语义价值不高，BPE 也可能因为它们的高频共现而将其合并。

#### 2\. WordPiece：信息增益最大化

WordPiece 使用的评分公式  $\frac{Freq\left(AB\right)}{Freq\left(A\right)\cdot Freq\left(B\right)}$  考虑了  $A$  和  $B$  的**独立频率**。

*   **如果得分高：** 说明  $A$  和  $B$  实际相邻出现的频率  $Freq\left(AB\right)$  **远高于**它们独立出现时的期望频率  $Freq\left(A\right)\cdot Freq\left(B\right)$ 。这表明  $A$  和  $B$  具有强烈的**绑定关系**（例如，`un` 和 `happy` 倾向于在一起）。
*   **如果得分低：** 即使  $Freq\left(AB\right)$  很高，但如果  $A$  和  $B$  独立出现的频率更高，分母会很大，得分反而低。这说明  $A$  和  $B$  可能是随机共现，合并的价值较小。

**结论：** WordPiece 是 BPE 的一个**优化版本**，它在选择合并对时引入了**统计学和语言模型**的考量，确保每一次合并都能为整个语料库的编码带来最大的**信息增益**，从而在相同的词汇表大小下，得到一个对模型训练更高效的子词集合。

## Prompt:
我想进一步了解SentencePiece和ULM

## Response:
WordPiece 和 BPE 固然强大，但在处理多语言和无空格语言（如中文、日文、泰文）时，它们依赖于预先对文本进行空格或标点符号分割（即所谓的“预分词”），这引入了语言特有的复杂性和依赖性。

SentencePiece 应运而生，它旨在提供一种**不依赖语言、不依赖预分词**的统一子词分割方法，而其最常用的算法就是 **Unigram Language Model (ULM)**。

* * *

🚀 SentencePiece 介绍
-------------------

SentencePiece（由 Google 研发）是一个开源的文本分词器工具包，它的主要特点是：

1.  **统一处理 (Pre-tokenization-free):** 它将所有输入文本视为**原始字符序列或 Unicode 字符流**，包括空格。它在处理文本时会将空格也作为一个普通字符来处理，通常用一个特殊符号（如 `_`）来表示。
2.  **语言无关性:** 由于不依赖预分词，SentencePiece 可以用相同的方式训练和应用到任何语言上，这对于构建**多语言模型**（如 mBERT、T5）至关重要。
3.  **可逆性 (Lossless):** 由于它处理原始字符流并明确地处理空格，SentencePiece 的分词结果是完全可逆的。分词后的 Token 可以精确地重构回原始文本，不会丢失任何信息（包括空格和标点符号的位置）。

SentencePiece 提供了两种主要的训练算法：BPE 和 ULM。目前，ULM 是其最推荐也是最独特的分词方式。

* * *

🧠 Unigram Language Model (ULM) 算法详解
------------------------------------

ULM 与 BPE 和 WordPiece 最大的不同点在于它的**训练方向**和**概率切分**能力。

### 1\. 训练方向：从大到小（Pruning）

*   **BPE/WordPiece:** 从单个字符开始，**迭代合并**（从小到大）。
*   **ULM:** 从一个非常大的、包含所有字符和常见  $N$ \-gram 的初始词汇表开始，然后**迭代删除/修剪**（从大到小），直到达到目标词汇表大小。

### 2\. ULM 训练过程

#### 步骤 1: 初始词汇表构建

构建一个非常大的初始词汇表  $V_{initial}$ 。这通常包括训练语料中出现的所有**单个字符**和所有**常见的子字符串**（如长度为 2、3、4 的  $N$ \-grams）。

#### 步骤 2: 计算初始子词概率

基于训练语料，使用期望最大化（EM）算法的变体，为词汇表  $V_{initial}$  中的每个子词  $x$  估计其概率  $P\left(x\right)$ 。这个概率基于 Unigram 模型的假设。

#### 步骤 3: 迭代修剪（Pruning）

这是 ULM 的核心步骤。迭代地进行以下操作，直到词汇表大小达到目标值  $K$ ：

1.  **计算损失:** 对于词汇表  $V_{current}$  中的每个子词  $x$ ，计算如果将其从词汇表中**移除**，整体语料库的对数似然度损失  $Loss\left(x\right)$ 。
2.  **选择修剪对象:** 找出那些**损失最小**（即移除后对似然度影响最小）的子词。这意味着这些子词是最**冗余**或**不重要**的。
3.  **修剪:** 移除排名最低的  $\eta$  比例（例如 10% 或 20%）的子词。
4.  **重新计算:** 使用新的词汇表重新计算剩下子词的概率  $P\left(x\right)$ 。

通过修剪，ULM 最终只保留那些**对有效编码贡献最大**的子词单元。

### 3\. ULM 的推理/编码（切分）过程

这是 ULM 最具特色的部分：**它允许一个词有多种切分方式**。

当给定一个词  $W$  时，ULM 不使用简单的贪婪匹配，而是利用 Viterbi 算法，在所有可能的子词切分中，找出**总概率最大**的那一个切分序列  $S^{\ast }=\left(s_{1},s_{2},\dots ,s_{k}\right)$ 。

$$
Best Split S^{\ast }=\arg \max_{S} \prod_{i=1}^{k} P\left(s_{i}\right)
$$

> **示例:** 假设词汇表包含:  $\left{un,happy,py\right}$ 
> 
> *   单词 "unhappy"
> *   切分 1:  $\left[un,happy\right]$ ，总概率  $P\left(un\right)\cdot P\left(happy\right)$ 
> *   切分 2:  $\left[unhapp,y\right]$  (如果  $unhapp$  和  $y$  都在词汇表中)
> *   **ULM 会根据训练出的  $P\left(x\right)$  值，选择概率最高的那个切分。**
>     

### 4\. ULM 的优势

1.  **概率切分（正则化）:** ULM 允许在推理时引入**采样**，不总是选择最优切分，而是基于概率分布随机选择一个合理的切分。这在训练中可以作为一种**数据增强**和**正则化**手段，使模型对分词边界的变化更加鲁棒。
2.  **更好的拟合:** 由于其基于概率和似然度最大化的优化目标，ULM 往往能够生成比 BPE 拟合得更好的子词单元。
3.  **多语言统一:** 结合 SentencePiece，ULM 提供了处理所有语言的单一、强大且可逆的解决方案，是现代大型多语言 LLM 的重要基石。

## Prompt:
给一个ULM的具体处理的例子

## Response:
好的，我们通过一个具体的、简化的例子来展示 **ULM (Unigram Language Model)** 在 **SentencePiece** 中的工作原理，重点关注**概率切分和 Viterbi 解码**。

为了简化，我们只关注**推理/编码阶段**（切分输入文本），因为训练阶段涉及复杂的 EM 算法和大规模修剪。

* * *

ULM 具体处理示例
----------

### 场景设定

假设我们已经训练好了一个 ULM 分词器，其词汇表  $V$  及其相应的子词概率  $P\left(x\right)$  如下（概率越高的子词，说明它越常用、越有价值）：

| 子词  $x$ | 概率  $P\left(x\right)$ | $\log P\left(x\right)$ |
| --- | --- | --- |
| $_play$ | $0.20$ | $-1.61$ |
| $_ing$ | $0.05$ | $-3.00$ |
| $_p$ | $0.15$ | $-1.90$ |
| $_lay$ | $0.10$ | $-2.30$ |
| $ing$ | $0.08$ | $-2.53$ |
| $_pla$ | $0.03$ | $-3.51$ |

> **注意：** `_` 符号在这里表示它是一个词的起始（即它前面有空格）。ULM 在训练时会将空格视为普通字符进行编码。

### 任务：切分单词 " playing"

输入文本：**`_playing`** (注意开头的空格，ULM 会将其编码为一个 Token 的前缀)

目标：找到概率最大的切分序列  $S^{\ast }=\left(s_{1},s_{2},\dots ,s_{k}\right)$ 。

* * *

### 1\. 识别所有可能的切分路径

我们从左到右，找出所有可能的、由词汇表  $V$  组成的切分序列。

#### 路径 A: 两个 Tokens

*   切分:  $\left[_play,ing\right]$ 
    *   Tokens:  $s_{1}=_play$  (概率  $0.20$ ),  $s_{2}=ing$  (概率  $0.08$ )
*   **总概率 (乘积):**  $P\left(S_{A}\right)=0.20\times 0.08=0.016$ 
*   **总对数概率 (相加):**  $\log P\left(S_{A}\right)=-1.61+\left(-2.53\right)=-4.14$ 

#### 路径 B: 三个 Tokens

*   切分:  $\left[_p,lay,ing\right]$ 
    *   Tokens:  $s_{1}=_p$  (概率  $0.15$ ),  $s_{2}=lay$  (概率  $0.10$ ),  $s_{3}=ing$  (概率  $0.08$ )
*   **总概率 (乘积):**  $P\left(S_{B}\right)=0.15\times 0.10\times 0.08=0.0012$ 
*   **总对数概率 (相加):**  $\log P\left(S_{B}\right)=-1.90+\left(-2.30\right)+\left(-2.53\right)=-6.73$ 

#### 路径 C: 多个 Tokens (不完整，仅演示)

*   切分:  $\left[_pla,\dots \right]$  (子词  $_pla$  概率为  $0.03$ )
*   由于  $_pla$  概率很低，这条路径的总概率会非常低，通常不会成为最优解。

### 2\. Viterbi 解码：寻找最优切分

ULM 使用 **Viterbi 算法**来有效地找到具有最高总概率（即最小负对数概率）的路径。

| 路径  | 总概率  $P\left(S\right)$ | 总对数概率  $\log P\left(S\right)$ |
| --- | --- | --- |
| **A** | $0.016$ | ** $-4.14$ ** |
| **B** | $0.0012$ | $-6.73$ |

**最优切分  $S^{\ast }$ ：** 路径 A 的概率最高（对数概率最小），因此 ULM 将单词 "`_playing`" 切分为：

$$
[_{_play},ing]
$$

### ULM 的独特之处：正则化/采样切分

虽然上面的 Viterbi 解码给出了**唯一的、最优的**切分结果，但在实际的 LLM 训练中，ULM 往往会使用 **SentencePiece 的正则化（Sampling）模式**。

在这种模式下，模型**不会总是选择最优切分  $S^{\ast }$ **，而是根据所有可能切分路径的概率分布进行**随机采样**：

$$
P\left(S∣W\right)=\frac{P\left(S\right)}{\sum_{S^{′}} P\left(S^{′}\right)}
$$

*   在我们的例子中，虽然路径 A 的概率更高，模型在训练时仍有一定概率会选择**路径 B**  $\left[_p,lay,ing\right]$ 。

**这种随机性带来的好处是：**

1.  **数据增强:** 为模型引入了分词边界的轻微变化。
2.  **鲁棒性:** 使模型不会过度依赖某一种固定的子词切分，从而提高其对输入变化的泛化能力。

## Prompt:
我发现有两个库可以做分词：transformers和tokenizers，我想知道该怎么使用他们

## Response:
您问了一个很好的实际问题！`transformers` 和 `tokenizers` 这两个库在 LLM 分词生态系统中扮演着不同的角色，它们通常是**协同工作**的。

简单来说：

1.  **`tokenizers` 库 (Hugging Face Core):** 专注于**分词算法的极速实现**，用于**训练、保存和加载**分词器。它提供了高性能、多线程的 Rust 后端。
2.  **`transformers` 库 (Hugging Face Ecosystem):** 专注于**将分词器集成到模型管线**中，提供统一的 Python 接口，便于**使用**（加载、编码、解码）预训练模型的分词器。

下面是针对您的需求，关于这两个库的具体使用场景和方法。

* * *

🛠️ 一、 `transformers` 库的使用：加载与应用
--------------------------------

这是您**最常用**的方式，特别是在使用预训练模型时。`transformers` 库通过其 `AutoTokenizer` 类提供了一个统一、高级的接口。

### 1\. 核心用途：应用预训练分词器

当您想要使用 BERT、GPT-2、Llama 或任何 Hugging Face Hub 上的模型时，您应该使用 `transformers` 库来加载对应的分词器。

**代码示例：加载 BERT 的 WordPiece 分词器**

```
from transformers import AutoTokenizer

# 1. 传入模型名称或路径
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. 编码单个文本
text = "The quick brown fox jumps over the lazy dog."
encoding = tokenizer(text)

print("Tokens:", tokenizer.convert_ids_to_tokens(encoding['input_ids']))
print("IDs:", encoding['input_ids'])

# 3. 批量编码（自动填充和截断）
texts = ["This is a short sentence.", "This is a much longer sentence that needs to be handled."]
batch_encoding = tokenizer(
    texts, 
    padding=True,          # 填充到最长序列
    truncation=True,       # 截断超过最大长度的序列
    return_tensors="pt"    # 返回 PyTorch Tensor
)

# 4. 解码（将 Token IDs 转换回文本）
decoded_text = tokenizer.decode(batch_encoding['input_ids'][0])
print("\nDecoded Text:", decoded_text)
```

### 2\. `transformers` 的优势

*   **便捷性:** `AutoTokenizer` 自动检测并加载正确的分词算法和配置（BPE, WordPiece, ULM 等）。
*   **Pipeline 集成:** 自动处理模型输入所需的特殊 Token（`[CLS]`, `[SEP]`, `[PAD]`）和注意力掩码（`attention_mask`）。
*   **统一格式:** 方便地返回 PyTorch, TensorFlow 或 NumPy 格式的 Tensor。

* * *

⚙️ 二、 `tokenizers` 库的使用：训练与定制
-----------------------------

如果您需要**从头开始训练**一个分词器、**更改**预训练分词器的行为或追求**极致的编码/解码速度**，您需要使用 `tokenizers` 库。

### 1\. 核心用途：训练新的分词器

当您想为自己的**特定领域语料**（如医学文本、法律文档）训练一个全新的 BPE 或 WordPiece 分词器时，可以使用 `tokenizers` 库。

**代码示例：训练一个新的 BPE 分词器**

```
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. 初始化分词器，指定模型类型（这里使用 BPE）
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. 设置 Pre-Tokenizer (如何处理原始文本，如按空格分割)
tokenizer.pre_tokenizer = Whitespace()

# 3. 设置训练器
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=10000  # 设定目标词汇表大小
)

# 4. 训练分词器
files = ["./my_corpus_file_1.txt", "./my_corpus_file_2.txt"] # 假设您的语料文件列表
tokenizer.train(files, trainer=trainer)

# 5. 保存分词器配置
tokenizer.save("my_new_bpe_tokenizer.json")

print("新分词器训练完成并保存。")
```

### 2\. `tokenizers` 的优势

*   **高性能:** 基于 Rust 编写，是目前最快的分词实现之一。
*   **定制化:** 允许完全控制分词流程的各个组件：
    *   **Normalizer:** 文本标准化（如 Unicode 规范化）。
    *   **Pre-Tokenizer:** 初始分词（如按空格、标点符号分割）。
    *   **Model:** 核心算法（BPE, WordPiece, ULM 等）。
    *   **Post-Processor:** 最终格式化（添加特殊 Tokens）。
*   **兼容性:** 训练好的分词器可以轻松地集成回 `transformers` 库的 `AutoTokenizer` 中。

* * *

🎯 总结与建议
--------

| 场景  | 推荐使用的库 | 核心功能 |
| --- | --- | --- |
| **使用预训练模型** (最常见) | `transformers` (使用 `AutoTokenizer`) | 加载、编码、解码、处理特殊 Tokens。 |
| **需要训练新分词器** | `tokenizers` (使用 `Tokenizer.train`) | 从头训练 BPE, WordPiece, ULM 等算法。 |
| **需要超快速度** | `tokenizers` (直接调用 Rust 后端) | 极速的批量编码/解码操作。 |
| **需要自定义分词规则** | `tokenizers` (定制 Pre-Tokenizer 等) | 精细控制分词过程中的标准化和预处理步骤。 |

**您的建议流程：**

1.  **使用阶段：** 绝大多数情况下，您只需要使用 `transformers` 库的 `AutoTokenizer.from_pretrained()` 来加载和使用分词器。
2.  **定制阶段：** 仅当您对分词效果不满意，需要为特定任务优化词汇表时，才使用 `tokenizers` 库来训练一个新的分词器，然后将训练好的 `.json` 文件导入回 `transformers` 中使用。

您现在想尝试使用 `transformers` 库加载一个预训练模型的分词器，还是想了解更多关于 `tokenizers` 库的定制细节？

## Prompt:
我需要训练、定制自己的分词库，并构建一个预训练数据集，我的数据是很多的txt文件，每个文件是一篇长篇小说，我可以怎么构建自己的预训练数据集

## Response:
要为长篇小说数据训练和定制自己的分词器，并构建一个高质量的预训练数据集，您需要系统地完成两个主要步骤：**分词器定制**（使用 `tokenizers`）和**数据集构建**（涉及数据处理和组织）。

下面是详细的步骤和建议。

* * *

🛠️ 第一步：分词器定制和训练 (`tokenizers` 库)
---------------------------------

由于您要处理的是大量长篇小说，分词器的质量和效率至关重要。

### 1\. 数据准备与抽样

训练分词器不需要全部数据。为了提高效率和保证质量，建议从所有 TXT 文件中**随机抽样**或抽取固定数量的行，构建一个用于训练分词器的**代表性子集**（例如 1 GB 的文本）。

### 2\. 选择分词算法

基于您是处理长篇小说（通常是中文或其他自然语言），推荐使用**子词分词**算法。

| 推荐算法 | 特点  | 适用场景 |
| --- | --- | --- |
| **BPE/WordPiece** | 速度快，易于实现，是主流选择。 | 任何语言，尤其是处理大量数据时。 |
| **ULM (Unigram)** | 基于概率，能提供正则化切分，对多语言鲁棒性高。 | 追求最高质量的切分和模型鲁棒性。 |

### 3\. 配置和训练分词器

以下是使用 `tokenizers` 库训练一个 WordPiece 分词器的步骤，这是 BERT 等模型常用的方法。

```
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import glob

# 1. 收集用于训练分词器的文件列表 (使用抽样后的子集)
# files = glob.glob("./novel_samples/*.txt")
files = ["./sample_novel_data.txt"] # 假设您已将抽样数据合并

# 2. 初始化 WordPiece 模型
# 设置 [UNK] token
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# 3. 设置训练器 Trainer
trainer = WordPieceTrainer(
    vocab_size=32768,  # 常用大小：16k, 32k, 64k
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    limit_alphabet=1000, # 限制初始字符集大小
    continuing_subword_prefix="##" # WordPiece 特有的内部子词前缀
)

# 4. 设置 Pre-Tokenizer 和 Normalizer
# 如果是中文，您可能需要定制更复杂的 Pre-Tokenizer，但对于 WordPiece，
# 简单的空格分割或按字符分割是常用的启动点。
# 如果数据是英文，使用 Whitespace：
tokenizer.pre_tokenizer = Whitespace() 
# 对于中文，您可以考虑先进行简单的字符级 Pre-tokenization。

# 5. 训练分词器
print("开始训练分词器...")
tokenizer.train(files, trainer=trainer)
print("分词器训练完成。")

# 6. 设置 Post Processor (可选，为模型输入添加特殊 Tokens)
# 模仿 BERT 的 [CLS] 和 [SEP] 格式
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# 7. 保存分词器
tokenizer.save("my_novel_wordpiece_tokenizer.json")
```

* * *

🏗️ 第二步：构建预训练数据集（数据处理）
----------------------

预训练数据集需要将您的长篇小说分割成适合模型处理的固定长度序列，并生成用于 ML 任务的标签（如语言模型、Next Sentence Prediction）。

### 1\. 加载和集成定制的分词器

将训练好的分词器集成到 `transformers` 库中，以便高效地处理大型数据集。

```
from transformers import PreTrainedTokenizerFast

# 加载您刚刚训练并保存的分词器
custom_tokenizer = PreTrainedTokenizerFast(tokenizer_file="my_novel_wordpiece_tokenizer.json")
```

### 2\. 数据处理与分块 (Chunking)

由于您处理的是长篇小说，数据是连续的。构建预训练数据集的关键是将其切分成固定长度的块。

通常有两种预训练任务需要准备数据：

#### A. 任务一：因果语言模型 (Causal Language Modeling, CLM) / 掩码语言模型 (Masked Language Modeling, MLM)

这是最常见的 LLM 预训练方法。您需要将所有文本连接起来，然后切分成固定长度的块。

```
import os
from datasets import load_dataset
from itertools import chain

# 1. 加载原始 TXT 数据集 (使用 Hugging Face `datasets` 库处理大量数据)
# 假设您的所有 TXT 文件都在 "./novels/" 目录下
raw_datasets = load_dataset("text", data_files={"train": glob.glob("./novels/*.txt")})

# 2. 将所有文本连接起来
# 设定最大序列长度（例如 BERT 是 512，现代 LLM 可能是 1024 或更高）
block_size = 512

def tokenize_function(examples):
    # 使用定制的分词器进行分词
    return custom_tokenizer(examples["text"])

# 对整个数据集进行分词
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=["text"],
)

# 3. 将连续文本分块（核心步骤）
def group_texts(examples):
    # 将所有文本连接起来
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # 截断到 block_size 的倍数
    total_length = (total_length // block_size) * block_size
    
    # 将数据切分成等长的块
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # 过滤掉不完整（小于 block_size）的最后一个块
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=os.cpu_count(),
)

# lm_datasets 现在包含了适合 CLM/MLM 训练的固定长度块。
```

#### B. 任务二：下一句预测 (Next Sentence Prediction, NSP)

BERT 等模型还会使用 NSP。您需要将文本分割成句子对  $\left(A,B\right)$ ，其中 50% 的  $B$  是  $A$  之后的下一句，50% 的  $B$  是来自语料库随机位置的不相关句子。

### 3\. 保存最终数据集

将处理好的数据集保存为通用的格式（如 Apache Arrow 或 PyTorch/TensorFlow），以便在训练脚本中直接加载。

```
lm_datasets["train"].save_to_disk("./preprocessed_novel_dataset")
```

通过以上步骤，您就完成了一个完整的**分词器定制、训练**和**预训练数据集构建**流程。

您需要我帮您更深入地了解如何处理**中文分词**（涉及到预分词和字符级处理）的复杂性吗？

## Prompt:
我想知道分词器中我应该设置哪些special\_tokens，我的数据是很多的txt文件，如何整理为可以进行词元训练和因果/掩码语言模型训练的数据形式，另外，掩码/因果大模型的训练数据有什么不同，他们的数据集格式是怎么样的，有什么要求，主流的做法又是怎么样的

## Response:
这是一个关于 LLM 数据预处理的综合性问题，我将分三个部分详细解答：**特殊 Tokens 的设置**、**数据整理流程**，以及**因果/掩码语言模型训练数据的区别与格式要求**。

* * *

🌟 第一部分：分词器中的 Special Tokens 设置
-------------------------------

特殊 Tokens (Special Tokens) 是分词器中预留的 ID，用于标记序列的结构或执行特定的任务。对于您的长篇小说数据，以下是最常见且必要的特殊 Tokens：

| Token | 符号 (常用) | 作用  | 必须性 |
| --- | --- | --- | --- |
| **未知词元** | `[UNK]` | 用于替换词汇表中不存在的词（OOV）。 | **必须** |
| **填充** | `[PAD]` | 用于将批次中较短的序列填充到与最长序列相同的长度。 | **必须** |
| **分类/起始** | `[CLS]` | 序列的起始标记。常用于 BERT 等编码器架构，作为句子或文档的表示。 | 视模型架构 |
| **分隔** | `[SEP]` | 用于分隔文本对（如问答或下一句预测任务中的 A 句和 B 句）。 | 视模型架构 |
| **掩码** | `[MASK]` | 用于掩盖输入序列中的词元，是 MLM（掩码语言模型）任务的核心。 | 仅 MLM 任务 |

### 针对您的建议：

1.  **通用 LLM (如 GPT/Llama 等因果模型):** 只需要 `[UNK]` 和 `[PAD]`，有时会添加 `[BOS]` (开始) 和 `[EOS]` (结束) Token。
2.  **BERT/RoBERTa (掩码模型):** 需要全部五个：`[UNK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`。

$$
Example (Causal): \left⟨BOS\right⟩ 这是第一句\left⟨EOS\right⟩\left⟨PAD\right⟩
$$

$$
Example (Masked): \left[CLS\right] 这\left[MASK\right]是句子\left[SEP\right]
$$

* * *

🧱 第二部分：数据整理流程（词元训练与预训练数据）
--------------------------

从大量的 TXT 文件到可用于训练的数据，需要经过两个主要阶段：**分词器训练**和**预训练数据构建**。

### 阶段一：词元训练（分词器训练）数据准备

| 目标  | 格式要求 | 主流做法 |
| --- | --- | --- |
| 训练分词器 | **原始文本文件** (TXT) 或每行一个文档/句子。 | **抽样**：从所有 TXT 文件中抽取有代表性的、**数百万至数十亿字符**的原始文本子集。因为分词器训练只需捕获词汇分布，无需使用全部数据。 |

**操作：** 将您所有 TXT 文件中的文本合并到一个或几个大文件中，作为 `tokenizers.train()` 的输入。

### 阶段二：预训练数据构建（因果/掩码模型）

| 目标  | 格式要求 | 主流做法 |
| --- | --- | --- |
| 训练 LLM | **Token ID 序列** (NumPy, PyTorch, 或 Arrow 格式)，并切分成**固定长度的块**。 | 使用 Hugging Face 的 `datasets` 库，执行**分词 (Tokenization)** 和**分块 (Chunking)** 操作。 |

**操作：**

1.  **分词：** 使用训练好的分词器将所有原始 TXT 数据编码成 Token ID 序列。
2.  **分块（核心）：** 将连续的 Token ID 序列连接起来，并切分成长度为  $L$ （如 512, 1024, 2048）的块。

* * *

📊 第三部分：掩码/因果模型训练数据的区别与格式
-------------------------

掩码语言模型 (MLM) 和因果语言模型 (CLM) 的训练数据主要区别在于**数据的组织方式**和**训练时的标签生成方式**。

### 1\. 因果语言模型 (CLM) - (如 GPT, Llama)

*   **数据目标：** 模型预测序列中的下一个词元  $t_{i+1}$ ，给定所有前面的词元  $\left(t_{1},\dots ,t_{i}\right)$ 。
*   **训练任务：**  $\arg \max P\left(t_{i+1}∣t_{1},\dots ,t_{i}\right)$ 
*   **数据格式要求（主流做法）：**
    *   **输入 (Input IDs):** 序列  $t_{1},t_{2},\dots ,t_{L}$ 。
    *   **标签 (Labels):** 序列  $t_{2},t_{3},\dots ,t_{L}$  和一个特殊的 `-100`。
    *   **格式：** 标签张量就是输入张量**向左平移一位**，因为  $t_{i}$  的预测目标是  $t_{i+1}$ 。
    *   **Dataset 结构：** | Input IDs | Labels | | :--- | :--- | |  $\left[t_{1},t_{2},t_{3},\dots ,t_{L}\right]$  |  $\left[t_{2},t_{3},\dots ,t_{L},-100\right]$  |
    *   **主流实践：** 通常使用 **`group_texts`** 函数将所有文档的 Token ID 连接起来，然后进行切块（如第二部分所述）。

### 2\. 掩码语言模型 (MLM) - (如 BERT, RoBERTa)

*   **数据目标：** 模型预测序列中被掩盖 (Masked) 的词元  $\hat{t}_{i}$ ，给定序列中所有**其他**的词元。
*   **训练任务：**  $\arg \max P\left(\hat{t}_{i}∣所有其他 t_{j}\right)$ 
*   **数据格式要求（主流做法）：**
    *   **输入 (Input IDs):** 序列  $t_{1},t_{2},\left[MASK\right],\dots ,t_{L}$ 。
    *   **标签 (Labels):** 只有被掩盖位置的**原始词元**。
    *   **格式：** 标签张量是与输入张量同形的，**未被掩盖**的位置被设置为特殊的 **`-100`**，**被掩盖**的位置填入原始词元 ID。
    *   **Dataset 结构：** | Input IDs | Labels | | :--- | :--- | |  $\left[t_{1},t_{2},MASK,\dots ,t_{L}\right]$  |  $\left[-100,-100,t_{original},\dots ,-100\right]$  |
    *   **主流实践：** 数据集构建通常涉及**句子级**的分割，然后应用**动态掩码** (Dynamic Masking)，即在训练循环中实时进行掩码操作，而不是事先将掩码写入数据集。

> **总结：**
> 
> *   **CLM 数据：** 连续的长序列切块，**标签是输入序列的错位版本**。
> *   **MLM 数据：** 句子或短文档的对齐序列，**标签是原始词元在被掩盖位置的值**。
>     

您确定要构建哪种模型的数据集了吗？我可以提供更具体的代码片段来处理**中文文本**中的特殊分词要求。

## Prompt:
是否可以只准备一种格式的数据集，比如每个样本就是一段文本，然后通过主流的库在加载进行训练时自动转为CLM活MLM所需要的格式

## Response:
这是一个非常实用且高效的方法！

**答案是肯定的。** 在现代 LLM 预训练流程中，主流做法就是**只准备一种格式的通用数据集**，然后在模型训练时使用库提供的工具（如 Hugging Face `transformers` 和 `datasets`）自动处理成 CLM 或 MLM 所需的格式。

这种方法大大简化了数据预处理流程。

* * *

🚀 主流做法：统一格式与动态处理
-----------------

### 1\. 统一的数据集格式：分块的 Token IDs

无论您最终训练 CLM 还是 MLM 模型，第一步都是相同的（如您在前面讨论的）：

*   **目标：** 将所有原始 TXT 文件分词（Tokenize）后，连接成一个巨大的序列，然后切分成固定长度的块（Chunking）。
*   **格式：** 每个样本都是一个固定的 Token ID 序列，例如长度为 1024。

| 字段  | 示例 (Block Size = 5) |
| --- | --- |
| **`input_ids`** | $\left[2001,1984,500,301,102\right]$ |
| **`attention_mask`** | $\left[1,1,1,1,1\right]$ |

这个统一的格式，被称为 **Tokenized and Chunked Data**，是 CLM 和 MLM 训练的**共同基础**。

### 2\. CLM 的动态转换：`DataCollatorForLanguageModeling`

在训练 CLM (如 GPT-2, Llama) 时，您不需要事先创建 `labels` 字段。加载数据后，Hugging Face 的 `DataCollatorForLanguageModeling`（设置 `mlm=False`）会自动在批次 (batch) 中完成转换：

*   **功能：** 它接收 `input_ids`，然后创建 `labels` 字段，即  $L$  序列向左平移一位。

| 原始输入 | `input_ids`:  $\left[t_{1},t_{2},t_{3},t_{4},t_{5}\right]$ |
| --- | --- |
| **Data Collator 输出** | `labels`:  $\left[t_{2},t_{3},t_{4},t_{5},-100\right]$ |

### 3\. MLM 的动态转换：`DataCollatorForLanguageModeling`

在训练 MLM (如 BERT, RoBERTa) 时，您也不需要事先创建 `labels` 或 `MASK` Tokens。加载数据后，`DataCollatorForLanguageModeling`（设置 `mlm=True`）会自动在批次中完成：

*   **功能：** 它会根据设定的掩码概率（通常是 15%），随机选择 Tokens 进行掩盖（替换为 `[MASK]` Token），并同时创建相应的 `labels` 字段。

| 原始输入 | `input_ids`:  $\left[t_{1},t_{2},t_{3},t_{4},t_{5}\right]$ |
| --- | --- |
| **Data Collator 输出** | `input_ids`:  $\left[t_{1},\left[MASK\right],t_{3},t_{4},\left[MASK\right]\right]$ |
|     | `labels`:  $\left[-100,t_{original},-100,-100,t_{original}\right]$ |

这种**动态掩码 (Dynamic Masking)** 每次训练迭代都会改变掩码的位置，这是比静态掩码更优的主流做法。

* * *

📝 您的实际操作流程建议
-------------

您可以遵循以下简化流程来构建数据和启动训练：

### 步骤 1: 训练和保存分词器 (一次性)

（参考前面部分）使用 `tokenizers` 库训练您的 WordPiece/ULM 分词器，并保存为 `my_novel_tokenizer.json`。

### 步骤 2: 构建统一数据集（Chunked Data）

使用 `datasets` 库，执行**分词**和**分块（Chunking）**，只保留 `input_ids` 和 `attention_mask` 字段。

### 步骤 3: 启动训练 (使用 `transformers.Trainer`)

根据您选择的模型架构，在训练脚本中配置不同的 `Data Collator`：

| 目标模型 | `DataCollatorForLanguageModeling` 设置 | 关键差异 |
| --- | --- | --- |
| **CLM** (GPT, Llama) | `mlm=False` | 自动创建错位标签 (Labels)。 |
| **MLM** (BERT, RoBERTa) | `mlm=True`, `mlm_probability=0.15` | 自动执行动态掩码和创建标签。 |

这种方法是目前 LLM 预训练的**标准实践**，因为它将**复杂且耗时的预处理工作**（分词和分块）与**模型特有的任务准备工作**（掩码、错位标签）分离开来，极大地提高了效率和灵活性。

您想继续讨论如何使用 `DataCollator` 在 PyTorch 或 TensorFlow 中实际加载和准备数据吗？

## Prompt:
我看到模型中有一些常见的标签，比如"\<|endoftext|\>", "\<|im\_start|\>", "\<|im\_end|\>"，或其他类似形式的标签

## Response:
您观察到的这些特殊标签（例如 `"<|endoftext|>"`, `"<|im_start|>"`, `"<|im_end|>"`, 或其他类似形式）是现代大型语言模型（LLM）中非常重要的一部分，它们被称为**控制 Tokens** 或**语义 Tokens**。

这些标签的主要用途是**结构化输入**、**定义边界**和**启用多轮对话/指令跟随**。

* * *

💡 常见特殊标签及其用途
-------------

这些特殊标签通常由分词器添加，并告诉模型如何解释输入序列。

| 标签示例 | 常见模型/用途 | 作用描述 |
| --- | --- | --- |
| \*\*\`\< | endoftext | \\>\`\*\* |
| \*\*\`\< | im\_start | \\>\`\*\* |
| \*\*\`\< | im\_end | \\>\`\*\* |
| **`[BOS]` / `<s>`** | **Llama, Mistral** 等。 | **序列开始 (Beginning of Sentence/Stream)：** 标记整个输入序列的开始。 |
| **`[EOS]` / `</s>`** | **Llama, Mistral** 等。 | **序列结束 (End of Sentence/Stream)：** 标记整个输入序列的结束。用于在训练中明确句子或文档的边界，并在生成时作为停止条件。 |
| **`<pad>`** | 所有需要填充的模型。 | **填充 (Padding)：** 在批量处理数据时，将短序列填充到与批次中最长序列相同的长度。 |

### 1\. 结构化对话/指令跟随 (Instruction Tuning)

像 `"<|im_start|>"`, `"<|im_end|>"` 这样的标签是**指令微调 (Instruction Tuning)** 或**对齐**过程中的核心。它们用于创建一种名为 **Chat Template** 的格式，以确保模型能够区分：

*   **系统指令 (System Prompt):** 定义模型的行为。
    $$
    Example: \left⟨im_start\right⟩system\left⟨im_end\right⟩你是一个乐于助人的AI。
    $$
*   **用户输入 (User Input):** 用户的提问或请求。
    $$
    Example: \left⟨im_start\right⟩user\left⟨im_end\right⟩请解释黑洞。
    $$
*   **模型回复 (Assistant Response):** 模型生成内容的开始。
    $$
    Example: \left⟨im_start\right⟩assistant\left⟨im_end\right⟩[模型开始生成回复]
    $$

通过这些明确的边界 Token，模型学习了**对话的结构**，从而能更好地理解何时接收指令、何时生成回复。

### 2\. 增强训练稳定性

像 `"<|endoftext|>"` 和 `<s>` / `</s>` 这样的 Token 在大规模预训练中发挥作用：

*   **防止跨文档信息泄漏：** 在训练阶段，`"<|endoftext|>"` 明确告诉模型一个文档结束了，即使它后面的 Token 是紧接着另一个文档的开始。这确保了模型不会尝试基于前一个文档的末尾来预测下一个文档的开头。
*   **明确停止信号：** 在生成文本时，当模型预测到 `<s>` 或 `</s>` 时，它就知道应该停止生成，从而提高生成效率和输出质量。

这些控制 Tokens 并没有固定的标准，不同的模型家族（如 Llama, Mistral, Qwen, Baichuan）通常会定义自己的一套特殊 Tokens，但用途是高度相似的。

您想了解如何将您自己的数据（长篇小说）**结构化**以加入这些特殊 Token 吗？

## Prompt:
指令微调是预训练之后才会进行的训练吗？指令微调的作用是什么

## Response:
是的，您理解得很正确。**指令微调 (Instruction Tuning)** 是一种专门的训练阶段，它总是发生在 **大规模预训练 (Pre-training)** 之后。

指令微调是弥合基础模型能力和用户实际需求之间差距的关键步骤。

* * *

1\. 指令微调在 LLM 生命周期中的位置
----------------------

| 阶段  | 名称  | 目的  | 数据类型 |
| --- | --- | --- | --- |
| **阶段 1** | **大规模预训练 (Pre-training)** | 学习语言的**语法、语义和世界知识**。模型学习预测下一个词元。 | 原始文本数据（互联网、书籍、小说）。 |
| **阶段 2** | **指令微调 (Instruction Tuning)** | 将基础模型转化为一个**遵循人类指令**的助手。模型学习理解输入格式和期望的输出格式。 | (指令, 回复) 对数据集。 |
| **阶段 3** | **RLHF/DPO (人类反馈对齐)** | 进一步优化模型，使其输出不仅正确，而且**安全、有用、符合人类偏好**。 | 人类排序数据（Preference Data）。 |

* * *

2\. 指令微调的核心作用
-------------

指令微调的目标是让基础模型（Base Model）从一个“文本预测机器”转变为一个“遵循指令的助手”。其核心作用可以概括为以下三点：

### A. 提升遵循指令的能力 (Instruct Following)

这是指令微调最主要的作用。

*   **问题：** 预训练模型虽然知道很多知识，但它只会做\*\*“续写”\*\*。当你问它一个问题时，它可能会续写一篇相似的文章，而不是直接回答你的问题。
*   **作用：** 通过在  $\left(指令,期望回复\right)$  对上进行训练，模型学习到：
    *   **识别指令的意图**（如摘要、问答、翻译）。
    *   **遵循格式限制**（如“以列表形式回答”、“限制在三句话之内”）。
    *   **在收到指令后，停止生成续写文本，并开始生成指令的答案**。

### B. 提升泛化能力 (Zero-shot / Few-shot Generalization)

指令微调能显著提高模型在面对**训练中未见过的新任务**时的表现。

*   **作用：** 模型从大量的不同任务格式中，学习到任务的**共性**和**结构**。例如，它学会了所有“翻译”任务都有类似的模式（`[Input]`  $\to$  `[Output]`），即使输入是它没见过的语言对，它也能尝试遵循翻译的指令。

### C. 结构化和格式对齐 (Alignment with Chat Templates)

指令微调让模型熟悉现代 LLM 使用的**结构化输入格式**。

*   **作用：** 它将模型与前面提到的特殊 Tokens（如 `system`, `user`, `assistant`, `im_start` 等）对齐。模型学会了：
    *   当看到 `user` 标签时，它应该开始处理请求。
    *   当看到 `assistant` 标签后，它应该开始生成回复。

3\. 指令微调的数据形式
-------------

指令微调的数据集通常包含大量的任务示例，例如：

$$
Input: \left⟨im_start\right⟩user\left⟨im_end\right⟩请将以下句子翻译成英文：这是一本好书。
$$

$$
Output: \left⟨im_start\right⟩assistant\left⟨im_end\right⟩This is a good book.
$$

通过训练，模型学习到 **Input** 部分的 Token，然后预测 **Output** 部分的 Token。

简而言之，指令微调将一个知识渊博但被动的“学者”变成了一个**主动、听话、且能干的“AI 助手”**。

## Prompt:
那指令标签如何定义呢？是自由定义的吗？这类指令微调数据集有什么要求

## Response:
您触及了指令微调（Instruction Tuning）中最灵活也最关键的部分之一：**指令标签（Tags）和数据集要求**。

指令标签的定义和数据集的构建质量直接决定了模型遵循指令的能力和泛化能力。

* * *

1\. 指令标签（Tags）如何定义？
-------------------

指令标签（或称为 Chat/Role Markers）**不是自由定义的，但也不是统一标准的**。它们属于**人为约定**，一旦确定，就必须在整个训练和部署生命周期中保持一致。

### A. 约定优先于自由

虽然您理论上可以使用任何 Token 组合作为标签（例如 `[USER_START]` 或 `###SYSTEM###`），但在实践中，您应该遵循以下原则：

1.  **遵循基础模型规范 (主流做法):** 如果您使用的是某个开源基础模型（如 Llama, Mistral, Qwen, Baichuan），您应该**严格使用它们官方定义的特殊标签**。
    *   **Llama 3 示例：** 使用 `<|start_header_id|>` 和 `<|end_header_id|>` 来封装角色名（如 `user` 或 `assistant`）。
    *   **Qwen 示例：** 使用 `<|im_start|>` 和 `<|im_end|>`。
    *   **GPT-2/3 风格：** 使用 `<|endoftext|>` 作为分隔符。
2.  **保证唯一性：** 您定义的标签必须是**词汇表中唯一的 Token**，不能是任何常用词的子词或组合。这就是为什么它们通常使用尖括号或方括号等符号进行封装。
3.  **Tokenization 兼容性：** 您必须在**分词器训练阶段**将这些标签明确地添加为 **Special Tokens**，确保它们被视为一个完整的、不可分割的单元，对应一个唯一的 ID。

### B. 标签的组成结构

一套完整的对话标签通常包括：

*   **边界标签：** 标记对话/指令的开始和结束。
*   **角色标签：** 标记发言的角色（`system`、`user`、`assistant`）。

$$
通用 Chat Template 示例: \left⟨BOS\right⟩\left⟨System Marker\right⟩System Prompt\left⟨EOS\right⟩\left⟨User Marker\right⟩User Message\left⟨EOS\right⟩\left⟨Assistant Marker\right⟩Model Response\left⟨EOS\right⟩
$$

* * *

2\. 指令微调数据集有什么要求？
-----------------

指令微调数据集的质量要求比原始预训练数据要高得多，因为它需要教会模型逻辑和推理。

### A. 格式要求（必须结构化）

数据集中的每个样本必须是一个**完整的、结构化的交互历史**，即由指令标签连接起来的 Tokens 序列。

| 要求  | 描述  | 示例（Qwen 风格） |
| --- | --- | --- |
| **完整性** | 必须包含所有必要的角色标签和内容。 | \`"\< |
| **序列化** | 多个轮次（多轮对话）必须被串联成一个长序列。 | 包含用户的追问和助手的第二次回复，全部在一个样本中。 |
| **Token ID** | 必须使用与模型兼容的 Tokenizer 编码成 ID 序列。 | `[151645, 8948, 151646, 198, 2011, 2017, 2033, 151645, 2033, 151646, 198, 2002, 2003, 2004]` |

### B. 内容要求（多样性是关键）

内容要求确保模型具有强大的泛化能力。

1.  **任务多样性 (Task Variety):**
    *   数据集应包含**广泛**的任务类型：问答、摘要、翻译、代码生成、头脑风暴、表格生成等。
    *   **目的：** 教会模型任务的**一般性模式**，以便在面对新任务时也能表现良好（即零样本泛化能力）。
2.  **指令风格多样性 (Style Variety):**
    *   指令的表达方式应多样化：使用简洁、礼貌、正式、非正式等不同语气。
    *   **目的：** 使模型能够理解和响应各种人类沟通风格，提高鲁棒性。
3.  **角色/上下文多样性 (Context Variety):**
    *   包含单轮问答、多轮对话、使用系统指令（System Prompt）的任务。
    *   **目的：** 教会模型如何处理复杂的对话历史和高层级的系统约束。
4.  **质量要求 (High Quality):**
    *   数据集中的**回复**必须是**高质量、准确、安全**的。如果回复包含错误信息或有害内容，模型会学习到这些不良行为。

### 主流数据集格式（例如 Alpaca / Self-Instruct 风格）

许多指令微调数据集（如 Alpaca, FLAN, ShareGPT 等的变体）都是以 JSON 或 CSV 格式存储的，它们通常包含以下字段，用于在训练时构建序列：

| 字段  | 作用  |
| --- | --- |
| `instruction` | 用户的原始指令文本。 |
| `input` | 任务所需的额外输入上下文（可选）。 |
| `output` | 模型期望生成的正确回复文本。 |
| `conversations` | 用于多轮对话的列表结构。 |



---
Powered by [Gem Chat Exporter](https://www.gem-chat-exporter.com)