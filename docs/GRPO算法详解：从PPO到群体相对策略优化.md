# GRPO 算法详解：从强化学习对齐到群体相对策略优化

当一个语言模型完成了预训练和监督微调（SFT）后，它已经具备了基本的对话能力。但如果我们希望模型不仅能"说话"，还能"说好话"——符合人类价值观、偏好和期望——就需要进入模型训练的最后一个关键阶段：**强化学习对齐**（Reinforcement Learning Alignment）。

本文将基于 MiniMind 项目中的 `my_train_grpo.py` 实现，深入剖析 GRPO（Group Relative Policy Optimization）这一新兴的对齐算法。我们不仅会探讨"如何实现"，更会深入"为什么这样设计"——理解 GRPO 相对于 PPO 的改进思路，以及它在实际工程中的关键实现细节。

## 目录

1. [为什么需要强化学习对齐](#为什么需要强化学习对齐)
2. [从 PPO 到 GRPO：对齐算法的演进](#从-ppo-到-grpo对齐算法的演进)
3. [GRPO 核心原理：群体相对优势](#grpo-核心原理群体相对优势)
4. [MiniMind 中的 GRPO 实现剖析](#minimind-中的-grpo-实现剖析)
5. [奖励函数设计：对齐的灵魂](#奖励函数设计对齐的灵魂)
6. [KL 散度约束：防止遗忘的护栏](#kl-散度约束防止遗忘的护栏)
7. [工程实践中的关键细节](#工程实践中的关键细节)
8. [总结与展望](#总结与展望)

---

## 为什么需要强化学习对齐

在深入 GRPO 算法之前，让我们先理解一个根本性问题：为什么仅靠监督学习无法训练出"好用"的模型？

### 监督学习的困境：模仿的局限性

预训练和 SFT 本质上都是**模仿学习**——模型学习复制训练数据中的模式。这种学习方式存在几个根本性的局限：

**局限一：数据质量的天花板**

无论训练数据多么精心筛选，都难以覆盖所有场景下的"最佳回答"。考虑一个简单的例子：

```
用户：请解释相对论

数据集中的回答A：相对论是爱因斯坦提出的物理学理论...（学术风格）
数据集中的回答B：简单来说，相对论告诉我们时间和空间...（通俗风格）
数据集中的回答C：想象你坐在一辆高速行驶的火车上...（比喻风格）
```

这三个回答都是"正确"的，但 SFT 只能学会它们的平均分布，而无法根据用户的隐含偏好动态调整。

**局限二：反馈粒度的粗糙**

SFT 的损失函数是 token 级别的交叉熵，它把每个词的预测视为独立事件。但人类评估回答质量是从整体出发的：

- 回答是否有帮助？
- 逻辑是否连贯？
- 语气是否恰当？
- 是否有潜在的危害？

这些**序列级**的质量信号无法通过 token 级损失有效传递。

**局限三：分布匹配的误区**

SFT 的优化目标是最大化训练数据的似然：

$$\max_\theta \mathbb{E}_{x,y \sim \mathcal{D}} [\log P_\theta(y|x)]$$

这意味着模型在努力"成为训练数据的完美复制品"。但我们真正想要的不是复制，而是**超越**——生成比训练数据更好的回答。

### 强化学习：从模仿到优化

强化学习提供了一个不同的框架。它不再追问"训练数据是怎么回答的"，而是直接优化"什么样的回答能获得更高的奖励"：

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim P_\theta(\cdot|x)} [R(x, y)]$$

这个优化目标有几个关键特性：

1. **目标明确**：直接优化我们真正关心的质量指标（奖励 R）
2. **探索能力**：模型生成的 $y$ 来自自身分布，可以发现训练数据之外的好回答
3. **整体评估**：奖励函数可以对整个回答进行评估，不局限于 token 级别

但这个框架也带来了新的挑战：如何定义奖励？如何稳定优化过程？如何防止模型"钻空子"？这正是各种对齐算法需要解决的问题。

---

## 从 PPO 到 GRPO：对齐算法的演进

### PPO：开创性的对齐方案

PPO（Proximal Policy Optimization）是 InstructGPT/ChatGPT 背后的核心算法。它的核心思想是：在最大化奖励的同时，限制策略更新的幅度，避免训练崩溃。

PPO 的训练流程如下：

```
1. 用当前策略 π_θ 生成回答 y
2. 用奖励模型 R(x, y) 评估回答质量
3. 计算优势函数 A(x, y)
4. 更新策略，同时用 clip 限制更新幅度
```

PPO 的损失函数（简化版）：

$$L^{PPO}(\theta) = -\mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中 $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是概率比。

**PPO 的问题**

尽管 PPO 在实践中取得了巨大成功，但它存在几个显著问题：

1. **依赖单独的奖励模型**：需要先训练一个奖励模型，增加了复杂性和成本
2. **多次采样**：需要从旧策略和新策略分别采样，计算开销大
3. **训练不稳定**：价值函数估计的偏差会导致训练震荡
4. **内存密集**：需要同时保存策略模型和价值模型

### DPO：去掉奖励模型的尝试

DPO（Direct Preference Optimization）通过数学推导，将奖励模型隐式地嵌入到损失函数中：

$$L^{DPO}(\theta) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

其中 $y_w$ 是偏好的回答，$y_l$ 是不偏好的回答。

DPO 大大简化了训练流程——只需要偏好数据对，不需要单独的奖励模型。但它也有局限：

- **需要配对数据**：必须有明确的"好/坏"回答对
- **离线学习**：无法利用模型自己生成的新回答进行学习
- **探索受限**：只能在已有数据分布上优化

### GRPO：群体相对优势的创新

GRPO（Group Relative Policy Optimization）由 DeepSeek 团队在 DeepSeek-Math 论文中提出。它在保留 PPO 在线学习优势的同时，巧妙地解决了对奖励模型的依赖和训练稳定性问题。

GRPO 的核心创新：

1. **群体相对优势**：不依赖单独的价值函数，而是用同一批次内回答的相对排名来计算优势
2. **无需奖励模型**：可以直接使用规则奖励或简单的评估函数
3. **在线学习**：持续从模型自身的分布中采样，保持探索能力
4. **计算高效**：不需要维护价值网络，显存占用更低

让我们深入理解 GRPO 的工作原理。

---

## GRPO 核心原理：群体相对优势

GRPO 的名字中"Group Relative"揭示了它的核心思想：**不评估单个回答的绝对好坏，而是评估它在一组回答中的相对排名**。

### 从绝对到相对：思维方式的转变

传统的强化学习方法需要一个价值函数 $V(s)$ 来估计状态的期望回报，然后计算优势：

$$A(s,a) = Q(s,a) - V(s)$$

但价值函数的估计往往不准确，会引入偏差和方差。

GRPO 的思路是：**既然估计绝对值困难，为什么不直接比较相对值？**

对于每个提示 $x$，GRPO 生成一组回答 $\{y_1, y_2, ..., y_G\}$（通常 $G=4$ 到 $8$）。然后计算每个回答的奖励 $\{r_1, r_2, ..., r_G\}$，并通过组内标准化得到优势：

$$A_i = \frac{r_i - \mu_r}{\sigma_r}$$

其中 $\mu_r$ 和 $\sigma_r$ 是这一组奖励的均值和标准差。

这个简单的操作有几个深刻的含义：

1. **自动零均值**：每组的优势均值为 0，避免了整体偏移
2. **自动标准化**：不同 prompt 的奖励尺度不同不再是问题
3. **对比学习**：模型学会区分同一问题的好回答和坏回答
4. **无需价值函数**：完全避免了价值估计的问题

### GRPO 损失函数

GRPO 的完整损失函数如下：

$$L^{GRPO}(\theta) = -\mathbb{E}_{x,\{y_i\}}\left[\frac{1}{G}\sum_{i=1}^G \left(\min(r_i(\theta) A_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) A_i) - \beta D_{KL}[\pi_\theta || \pi_{ref}]\right)\right]$$

让我们逐一解析每个组成部分：

**1. 概率比 $r_i(\theta)$**

$$r_i(\theta) = \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}$$

衡量新策略相对于旧策略对某个回答的偏好程度。如果新策略更倾向于生成 $y_i$，则 $r > 1$；反之 $r < 1$。

**2. 裁剪操作 clip**

$$\text{clip}(r, 1-\epsilon, 1+\epsilon) = \max(\min(r, 1+\epsilon), 1-\epsilon)$$

这是 PPO 的经典设计，限制策略更新的幅度。典型的 $\epsilon = 0.2$，意味着概率比被限制在 $[0.8, 1.2]$ 范围内。

**3. 取 min 操作**

$$\min(r \cdot A, \text{clip}(r) \cdot A)$$

这确保了：
- 当 $A > 0$（好回答）时，如果 $r$ 太大，会被裁剪，防止过度强化
- 当 $A < 0$（坏回答）时，如果 $r$ 太小，也会被裁剪，防止过度惩罚

**4. KL 散度惩罚**

$$\beta D_{KL}[\pi_\theta || \pi_{ref}]$$

这一项确保新策略不会偏离参考策略（通常是 SFT 后的模型）太远。$\beta$ 通常取 0.01-0.05。

### 从公式到直觉

让我们用一个具体例子理解 GRPO 的工作方式：

```
提示：请解释什么是机器学习

模型生成 4 个回答：
y1: 机器学习是人工智能的一个分支，通过数据训练模型... (奖励: 0.8)
y2: 机器学习就是让计算机自己学习啦！(奖励: 0.3)
y3: ML 是 AI 的子领域，核心在于从数据中学习模式... (奖励: 0.7)
y4: 机器学习是... [生成中断] (奖励: 0.1)

组内标准化：
均值 μ = (0.8+0.3+0.7+0.1)/4 = 0.475
标准差 σ = 0.28

优势：
A1 = (0.8 - 0.475) / 0.28 = +1.16  → 正优势，强化
A2 = (0.3 - 0.475) / 0.28 = -0.63  → 负优势，弱化
A3 = (0.7 - 0.475) / 0.28 = +0.80  → 正优势，强化
A4 = (0.1 - 0.475) / 0.28 = -1.34  → 负优势，弱化
```

GRPO 会调整模型参数，使得：
- 更倾向于生成类似 $y_1$ 和 $y_3$ 的回答
- 更不倾向于生成类似 $y_2$ 和 $y_4$ 的回答

关键是，这种调整是**相对**的——即使 $y_2$ 的绝对奖励 0.3 可能在某些场景下不算差，但在这个组里它低于平均，就会被弱化。

---

## MiniMind 中的 GRPO 实现剖析

理论清晰后，让我们深入 MiniMind 的 `my_train_grpo.py` 实现，看看这些概念如何落地为代码。

### 整体架构

```python
class GRPOTrainer:
    """MiniMind GRPO 训练器"""
    
    def __init__(self, args, model, ref_model, tokenizer, dataloader, local_rank=-1):
        self.model = model          # 策略模型（可训练）
        self.ref_model = ref_model  # 参考模型（冻结）
        # ...
```

GRPO 需要两个模型：
1. **策略模型（model）**：我们要优化的模型
2. **参考模型（ref_model）**：SFT 后的原始模型，用于 KL 约束

参考模型在训练开始时复制策略模型的权重，然后保持冻结：

```python
# 初始化 Reference 模型
ref_model = MiniMindForCausalLM(lm_config).to(args.device)
ref_model.load_state_dict(model.state_dict())  # 复制权重
ref_model.eval()
ref_model.requires_grad_(False)  # 冻结参数
```

### 生成多个回答

GRPO 的第一步是对每个 prompt 生成多个回答：

```python
# 生成多个响应
with torch.no_grad():
    model_for_gen = self.base_model
    
    outputs = model_for_gen.generate(
        **prompt_inputs,
        max_new_tokens=self.args.max_gen_len,
        do_sample=True,
        temperature=0.8,
        num_return_sequences=self.args.num_generations,  # 通常 4-8
        pad_token_id=pad_token_id
    )
```

几个关键参数：

- **`do_sample=True`**：启用采样，让模型生成多样的回答
- **`temperature=0.8`**：适中的温度，既有多样性又不太离谱
- **`num_return_sequences`**：每个 prompt 生成的回答数，这就是"Group"的大小

### 计算对数概率

生成回答后，需要计算策略模型和参考模型对这些回答的对数概率：

```python
def get_per_token_logps(model, input_ids: torch.Tensor, n_keep: int) -> torch.Tensor:
    """
    计算每个 token 的对数概率
    """
    if model.training:
        ctx = torch.enable_grad()
    else:
        ctx = torch.no_grad()
    
    with ctx:
        # 获取 logits，只保留需要的部分
        logits = model(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
        
        per_token_logps = []
        target_ids = input_ids[:, -n_keep:]
        
        for logits_row, ids_row in zip(logits, target_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_logps = torch.gather(log_probs, 1, ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_logps)
        
        return torch.stack(per_token_logps)
```

这个函数的设计体现了几个工程考量：

1. **`logits_to_keep` 参数**：只计算生成部分的 logits，节省内存和计算
2. **分离训练/推理模式**：策略模型需要梯度，参考模型不需要
3. **`log_softmax` + `gather`**：高效地获取目标 token 的对数概率

在训练循环中的使用：

```python
# 提取生成的部分
prompt_len = prompt_inputs["input_ids"].size(1)
completion_ids = outputs[:, prompt_len:]  # [B*num_gen, R]

# 计算 policy 模型的 log probabilities
per_token_logps = get_per_token_logps(
    self.model, 
    outputs, 
    completion_ids.size(1)
)

# 计算参考模型的 log probabilities
with torch.no_grad():
    ref_per_token_logps = get_per_token_logps(
        self.ref_model,
        outputs,
        completion_ids.size(1)
    )
```

### 计算组相对优势

这是 GRPO 的核心。首先获取奖励，然后进行组内标准化：

```python
# 解码生成的文本
completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

# 计算奖励
rewards = calculate_simple_reward(completions, self.device)

# 计算组相对优势
grouped_rewards = rewards.view(-1, self.args.num_generations)  # [B, G]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(self.args.num_generations)  # [B*G]
std_r = grouped_rewards.std(dim=1).repeat_interleave(self.args.num_generations)    # [B*G]

# 标准化并裁剪
advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)

# 全局标准化（可选，增加稳定性）
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

这段代码有几个细节值得注意：

1. **`repeat_interleave`**：将每个组的均值/标准差扩展到组内每个样本
2. **`+ 1e-4`**：防止除零，特别是当组内所有奖励相同时
3. **`clamp(-10, 10)`**：限制优势的范围，避免极端值
4. **全局二次标准化**：进一步稳定训练

### 处理完成标记

语言模型生成会在 EOS token 处停止。在计算损失时，需要一个掩码来区分有效 token 和 padding：

```python
# 创建 completion mask (到 EOS 为止)
eos_token_id = self.tokenizer.eos_token_id
if eos_token_id is None:
    eos_token_id = 0

is_eos = completion_ids == eos_token_id
eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.device)
eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]

# mask[i,j] = 1 如果 j <= eos_idx[i]，否则 = 0
completion_mask = (torch.arange(is_eos.size(1), device=self.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()
```

这段代码处理了三种情况：

1. **正常完成**：找到 EOS 位置，mask 到该位置
2. **未完成（达到最大长度）**：整个序列都有效
3. **空回答**：如果第一个就是 EOS，mask 为空

### 计算 KL 散度

KL 散度惩罚确保策略不会偏离参考模型太远：

```python
# 计算 KL 散度
kl_div = ref_per_token_logps - per_token_logps  # log(p_ref / p_policy)
per_token_kl = torch.exp(kl_div) - kl_div - 1   # KL 散度的一种近似形式
```

这里使用的是 KL 散度的一种变体：

$$D_{KL} \approx e^{\log \frac{p_{ref}}{p_\theta}} - \log \frac{p_{ref}}{p_\theta} - 1 = \frac{p_{ref}}{p_\theta} - \log \frac{p_{ref}}{p_\theta} - 1$$

这个形式在 $p_{ref} \approx p_\theta$ 时近似等于标准 KL 散度，但在数值上更稳定。

### 计算最终损失

将所有组件结合起来：

```python
# 计算策略损失
per_token_loss = -(
    torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) 
    - self.args.beta * per_token_kl
)

# 计算最终损失（加权平均）
loss = ((per_token_loss * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-8)).mean()
loss = loss / self.args.accumulation_steps

loss.backward()
```

让我们理解关键的计算：

```python
torch.exp(per_token_logps - per_token_logps.detach())
```

这计算的是概率比 $r(\theta) = \frac{\pi_\theta}{\pi_{\theta_{old}}}$。使用 `detach()` 确保分母不参与梯度计算。

实际上，由于 $e^{\log \pi_\theta - \log \pi_{\theta_{old}}} = \frac{\pi_\theta}{\pi_{\theta_{old}}}$，这是概率比的另一种表达形式。

在 MiniMind 的简化实现中，省略了 PPO 的 clip 操作，直接使用指数形式。这种简化在实践中通常也能工作，特别是配合 KL 惩罚时。

---

## 奖励函数设计：对齐的灵魂

奖励函数是强化学习对齐的核心。一个好的奖励函数需要准确反映我们希望模型学到的行为。MiniMind 实现了一个简单但有效的奖励函数：

```python
def calculate_simple_reward(responses: list, device: torch.device) -> torch.Tensor:
    """
    简单的奖励函数，基于响应质量评估
    """
    rewards = []
    
    for response in responses:
        reward = 0.0
        
        # 长度奖励：适中长度给予正向奖励
        length = len(response)
        if 50 <= length <= 500:
            reward += 0.5
        elif length < 50:
            reward -= 0.3
        elif length > 1000:
            reward -= 0.2
        
        # 格式奖励：检查是否有完整的句子
        if response.strip().endswith(('。', '！', '？', '.', '!', '?')):
            reward += 0.3
        
        # 避免重复
        words = response.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            reward += unique_ratio * 0.2
        
        rewards.append(reward)
    
    return torch.tensor(rewards, device=device)
```

这个奖励函数虽然简单，但包含了几个重要的设计理念：

### 1. 长度约束

```python
if 50 <= length <= 500:
    reward += 0.5
elif length < 50:
    reward -= 0.3
elif length > 1000:
    reward -= 0.2
```

为什么需要长度约束？因为语言模型有两种常见的退化模式：

- **过短回答**：模型学会给出敷衍的简短回答来"钻空子"
- **过长回答**：模型生成冗长无关的内容来"填充"

适中的长度范围（50-500 字符）鼓励模型给出信息充分但不冗余的回答。

### 2. 格式完整性

```python
if response.strip().endswith(('。', '！', '？', '.', '!', '?')):
    reward += 0.3
```

一个完整的回答应该有结束标点。这个简单的检查可以：

- 惩罚中途截断的回答
- 鼓励模型学习完整的句子结构

### 3. 多样性奖励

```python
words = response.split()
if len(words) > 5:
    unique_ratio = len(set(words)) / len(words)
    reward += unique_ratio * 0.2
```

重复是语言模型的另一个常见问题。这个检查通过计算词汇的独特性比例来惩罚重复：

- 如果一个回答中 80% 的词是独特的，获得 +0.16 奖励
- 如果只有 30% 的词是独特的（大量重复），只获得 +0.06 奖励

### 实际应用中的奖励设计

在生产环境中，奖励函数通常更复杂：

**方案一：奖励模型**

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids):
        hidden = self.backbone(input_ids).last_hidden_state[:, -1, :]
        return self.reward_head(hidden)
```

奖励模型通过人类偏好数据训练，学习判断回答质量。

**方案二：规则组合**

```python
def composite_reward(response, prompt):
    reward = 0.0
    
    # 安全性检查
    if contains_harmful_content(response):
        reward -= 2.0
    
    # 相关性（用简单的 embedding 相似度）
    relevance = compute_relevance(prompt, response)
    reward += relevance * 0.5
    
    # 流畅度（用困惑度代理）
    fluency = -compute_perplexity(response) / 100
    reward += fluency
    
    # 事实准确性（可选，需要外部知识库）
    if verify_facts(response):
        reward += 0.3
    
    return reward
```

**方案三：多模型评判（Constitutional AI 风格）**

```python
def constitutional_reward(response, prompt):
    critics = [
        ("这个回答是否有帮助？", critic_model),
        ("这个回答是否安全？", safety_model),
        ("这个回答是否诚实？", honesty_model),
    ]
    
    total_reward = 0.0
    for question, model in critics:
        score = model.evaluate(prompt + response + question)
        total_reward += score
    
    return total_reward / len(critics)
```

### 奖励黑客的风险

一个重要的警示：如果奖励函数设计不当，模型可能学会"钻空子"：

```
问题：奖励长度在 50-500 之间的回答

模型可能学会的技巧：
- 无论问什么，都回答正好 100 个字符的无意义内容
- 添加大量无关的"因此"、"综上所述"来凑字数
```

这就是为什么：

1. **多维度奖励**：不依赖单一指标
2. **KL 约束**：限制模型偏离原始分布的程度
3. **人工审核**：定期检查模型行为

---

## KL 散度约束：防止遗忘的护栏

在 GRPO 的损失函数中，KL 散度惩罚项扮演着关键角色：

$$L = L_{policy} - \beta \cdot D_{KL}[\pi_\theta || \pi_{ref}]$$

让我们深入理解这个约束的意义和实现。

### 为什么需要 KL 约束

没有 KL 约束的强化学习对齐可能导致几个问题：

**1. 灾难性遗忘**

模型可能在追求高奖励的过程中，忘记了 SFT 阶段学到的有用能力：

```
SFT 后：模型能用多种风格回答问题
RL 后（无 KL）：模型只会用一种获得高奖励的固定风格

示例：
原始能力：可以正式回答，可以幽默回答，可以简洁回答
RL 后：只会正式回答（因为正式回答在奖励函数中得分最高）
```

**2. 模式崩溃**

模型可能收敛到狭窄的"安全区"，对所有问题给出相似的回答：

```
问：什么是机器学习？
答：机器学习是人工智能的一个重要分支...

问：今天天气如何？
答：天气是人工智能的一个重要分支...（崩溃！）
```

**3. 奖励黑客**

模型可能发现奖励函数的漏洞，学会生成高奖励但无意义的内容：

```
如果奖励函数偏好包含关键词 "AI" 的回答：
问：什么是苹果？
答：苹果是 AI 领域的重要概念，AI 技术可以识别 AI 苹果... AI AI AI
```

KL 约束通过惩罚偏离参考分布的行为，有效缓解这些问题。

### KL 散度的实现细节

在 MiniMind 的实现中，KL 散度是在 token 级别计算的：

```python
# 计算 KL 散度
kl_div = ref_per_token_logps - per_token_logps  # log(p_ref / p_policy)
per_token_kl = torch.exp(kl_div) - kl_div - 1   # KL 的一种形式
```

这里使用的形式是：

$$D_{approx}(p || q) = \frac{p}{q} - \log\frac{p}{q} - 1$$

当 $p \approx q$ 时，这接近标准 KL 散度 $D_{KL}(p || q) = p \log \frac{p}{q}$。

这种形式的优点：
1. **对称性更好**：当 $p = q$ 时，$D_{approx} = 0$
2. **数值稳定**：避免了 $\log 0$ 的问题
3. **计算简单**：只需要两个对数概率的差

### β 参数的选择

$\beta$ 控制 KL 惩罚的强度，典型值在 0.01-0.1 之间：

**β 过小（如 0.001）**：
- KL 约束太弱
- 模型可能剧烈偏离参考分布
- 容易出现模式崩溃

**β 过大（如 1.0）**：
- KL 约束太强
- 模型无法有效学习
- 相当于不做 RL，只保持原样

**经验法则**：

```python
# 监控训练过程中的 KL 散度
avg_kl = per_token_kl.mean().item()

# 如果 KL > 5，考虑增加 β
# 如果 KL < 0.5 且奖励不增长，考虑减小 β
```

在 MiniMind 的默认配置中，$\beta = 0.02$，这是一个相对保守的选择，优先保证训练稳定性。

---

## 工程实践中的关键细节

除了核心算法，还有许多工程细节对训练效果至关重要。

### 1. 内存管理

GRPO 需要同时处理多个生成序列，内存压力较大。MiniMind 采用了几种策略：

```python
# 训练步骤结束时清理
del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask
torch.cuda.empty_cache()
gc.collect()
```

**为什么需要显式删除和 gc.collect()？**

PyTorch 的张量即使不再被引用，也可能不会立即释放 GPU 内存。`torch.cuda.empty_cache()` 释放缓存的内存块，`gc.collect()` 确保 Python 对象被回收。

### 2. 生成与训练的分离

注意生成过程使用 `torch.no_grad()`：

```python
with torch.no_grad():
    model_for_gen = self.base_model
    outputs = model_for_gen.generate(...)
```

而计算策略模型的对数概率时需要梯度：

```python
per_token_logps = get_per_token_logps(
    self.model,  # 需要梯度
    outputs, 
    completion_ids.size(1)
)
```

这种分离确保：
1. 生成时不记录计算图，节省内存
2. 只对策略评估部分计算梯度

### 3. 梯度累积

当 batch size 受显存限制时，梯度累积可以模拟更大的 batch：

```python
loss = loss / self.args.accumulation_steps

loss.backward()

if (step + 1) % self.args.accumulation_steps == 0:
    if self.args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
    self.optimizer.step()
    self.scheduler.step()
    self.optimizer.zero_grad()
    torch.cuda.empty_cache()
```

**关键点**：
- 损失除以累积步数，保证梯度尺度正确
- 梯度裁剪在所有梯度累积完成后进行
- 清空优化器梯度后再清理缓存

### 4. 学习率调度

GRPO 训练通常使用较小的学习率和余弦衰减：

```python
# 学习率调度器
total_steps = len(dataloader) * args.epochs // args.accumulation_steps
self.scheduler = CosineAnnealingLR(
    self.optimizer, 
    T_max=total_steps, 
    eta_min=args.learning_rate / 10  # 衰减到初始的 1/10
)
```

**为什么 RL 阶段需要更小的学习率？**

1. 模型已经通过 SFT 训练好，只需微调
2. RL 的梯度信号噪声较大，需要谨慎更新
3. 防止破坏预训练和 SFT 学到的知识

MiniMind 的默认学习率是 `8e-8`，比 SFT 阶段的 `5e-7` 小约 6 倍。

### 5. 检查点保存

训练中断是常见的，保存完整的训练状态很重要：

```python
def _save_checkpoint(self, epoch, step):
    output_dir = Path(self.args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_state = self.base_model.state_dict()
    
    # 半精度保存（节省空间）
    moe_suffix = '_moe' if self.base_model.config.use_moe else ''
    ckp = output_dir / f'{self.args.save_weight}_{self.base_model.config.hidden_size}{moe_suffix}.pth'
    torch.save({k: v.half() for k, v in model_state.items()}, ckp)
    
    # 保存完整检查点（用于恢复训练）
    checkpoint = {
        "model_state": model_state,
        "optimizer_state": self.optimizer.state_dict(),
        "scheduler_state": self.scheduler.state_dict(),
        "epoch": epoch,
        "step": step
    }
    torch.save(checkpoint, output_dir / "grpo_checkpoint.pt")
```

保存两种格式：
1. **半精度权重**：用于推理，体积更小
2. **完整检查点**：包含优化器状态，用于恢复训练

### 6. 分布式训练支持

MiniMind 的 GRPO 实现支持多 GPU 分布式训练：

```python
# DDP 包装模型
if dist.is_initialized():
    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
    model = DDP(model, device_ids=[local_rank])
```

注意 `_ddp_params_and_buffers_to_ignore`——RoPE 的频率张量不需要同步，忽略它们可以避免不必要的通信。

---

## 总结与展望

通过对 MiniMind 中 GRPO 实现的深入剖析，我们理解了这一先进对齐算法的核心理念和工程实践。

### GRPO 的核心优势

1. **简化训练流程**：不需要单独的价值函数或奖励模型
2. **群体相对评估**：通过组内比较自动解决奖励尺度问题
3. **在线学习**：持续从模型自身采样，保持探索能力
4. **计算高效**：单模型架构，显存友好

### 关键实现要点

1. **多样本生成**：每个 prompt 生成多个回答，形成比较组
2. **组内标准化**：将绝对奖励转化为相对优势
3. **KL 约束**：防止策略偏离参考模型太远
4. **工程细节**：内存管理、学习率调度、检查点保存

### 未来方向

GRPO 虽然强大，但仍有提升空间：

**1. 更好的奖励信号**

```python
# 从规则奖励 → 学习奖励 → 过程奖励
# 过程奖励：评估推理过程，而非只评估最终答案
def process_reward(response, intermediate_steps):
    step_rewards = [evaluate_step(s) for s in intermediate_steps]
    return sum(step_rewards)  # 每一步都给反馈
```

**2. 在线人类反馈**

```python
# 实时收集人类偏好，动态更新训练
def collect_online_feedback(prompt, responses):
    # 展示给用户，收集排名
    ranking = human_interface.get_ranking(prompt, responses)
    return ranking_to_rewards(ranking)
```

**3. 多目标优化**

```python
# 同时优化多个目标：帮助性、安全性、诚实性
rewards = {
    'helpful': helpful_reward(response),
    'safe': safety_reward(response),
    'honest': honesty_reward(response),
}
# 帕累托优化，寻找平衡点
```

**4. 自我改进循环**

```python
# 模型评估自己的回答，自我迭代
def self_improve(model, prompt):
    responses = model.generate(prompt, n=10)
    self_eval = model.evaluate(responses)  # 模型自评
    best_response = responses[self_eval.argmax()]
    # 用 best_response 更新模型
```

GRPO 代表了强化学习对齐技术的一个重要方向：**用更简单的方法解决复杂的问题**。它证明了在精心设计的框架下，不需要复杂的价值网络或奖励模型，也能实现有效的模型对齐。

随着大语言模型的快速发展，对齐技术的重要性只会越来越高。理解 GRPO 的原理和实现，不仅有助于应用现有技术，更能为探索下一代对齐方法打下基础。
