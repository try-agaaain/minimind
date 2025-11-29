"""
MiniMind GRPO (Group Relative Policy Optimization) 训练脚本
用于通过强化学习进行模型对齐优化。

GRPO 是一种改进的策略优化方法，使用组相对优势来训练模型。
"""
import os
import re
import gc
import argparse
import time
import warnings
from pathlib import Path
from contextlib import nullcontext

import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore')


# ========== Helper Functions ==========
def is_main_process():
    """Check if this is the main process in distributed training"""
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(msg):
    """Simple logger that only prints on main process"""
    if is_main_process():
        print(msg)


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


def setup_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


class RLAIFDataset(Dataset):
    """RLAIF 数据集，用于 GRPO 训练"""
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data(jsonl_path)
    
    def _load_data(self, path: str):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    samples.append(data)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _create_chat_prompt(self, conversations):
        """构建对话提示"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            content = turn.get('content', turn.get('value', ''))
            messages.append({"role": role, "content": content})
        
        # 如果 tokenizer 有 chat_template，使用它
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True
                )
                answer = messages[-1]['content'] if messages else ''
                return prompt, answer
            except Exception:
                pass
        
        # 回退：简单拼接
        prompt = ""
        for i, msg in enumerate(messages[:-1]):
            role = msg['role']
            content = msg['content']
            prompt += f"<|{role}|>\n{content}\n"
        prompt += "<|assistant|>\n"
        answer = messages[-1]['content'] if messages else ''
        
        return prompt, answer
    
    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample.get('conversations', sample.get('messages', []))
        prompt, answer = self._create_chat_prompt(conversations)
        
        return {
            'prompt': prompt,
            'answer': answer
        }


def calculate_simple_reward(responses: list, device: torch.device) -> torch.Tensor:
    """
    简单的奖励函数，基于响应质量评估
    
    在实际应用中，应该使用专门的奖励模型
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


def get_per_token_logps(model, input_ids: torch.Tensor, n_keep: int) -> torch.Tensor:
    """
    计算每个 token 的对数概率
    
    Args:
        model: 语言模型
        input_ids: 输入 token ID [B, seq_len]
        n_keep: 保留的 token 数量
        
    Returns:
        每个 token 的对数概率 [B, n_keep]
    """
    # 根据模型的训练状态选择适当的上下文管理器
    if model.training:
        ctx = torch.enable_grad()
    else:
        ctx = torch.no_grad()
    
    with ctx:
        logits = model(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
        
        per_token_logps = []
        target_ids = input_ids[:, -n_keep:]
        
        for logits_row, ids_row in zip(logits, target_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_logps = torch.gather(log_probs, 1, ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_logps)
        
        return torch.stack(per_token_logps)


class GRPOTrainer:
    """MiniMind GRPO 训练器"""
    
    def __init__(self, args, model, ref_model, tokenizer, dataloader, local_rank=-1):
        self.args = args
        self.model = model
        self.ref_model = ref_model  # 参考模型（冻结）
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.local_rank = local_rank
        self.device = args.device
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate
        )
        
        # 学习率调度器
        total_steps = len(dataloader) * args.epochs // args.accumulation_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps, 
            eta_min=args.learning_rate / 10
        )
        
        # 混合精度
        device_type = "cuda" if "cuda" in args.device else "cpu"
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        self.autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    @property
    def base_model(self):
        """返回基础模型实例"""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
    
    @property
    def base_ref_model(self):
        """返回基础参考模型实例"""
        if isinstance(self.ref_model, DDP):
            return self.ref_model.module
        return self.ref_model
    
    def train_epoch(self, epoch, total_epochs, wandb=None):
        """训练一个 epoch"""
        self.model.train()
        iters = len(self.dataloader)
        
        data_iterator = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{total_epochs}") if is_main_process() else self.dataloader
        
        for step, batch in enumerate(data_iterator, start=1):
            prompts = batch['prompt']  # list[str], length B
            
            # Tokenize prompts
            prompt_inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=self.args.max_seq_len,
                return_token_type_ids=False
            ).to(self.device)
            
            # 生成多个响应
            with torch.no_grad():
                model_for_gen = self.base_model
                
                # 设置 pad_token_id
                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.tokenizer.eos_token_id
                
                outputs = model_for_gen.generate(
                    **prompt_inputs,
                    max_new_tokens=self.args.max_gen_len,
                    do_sample=True,
                    temperature=0.8,
                    num_return_sequences=self.args.num_generations,
                    pad_token_id=pad_token_id
                )
            
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
            
            # 解码生成的文本
            completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            
            # 计算奖励
            rewards = calculate_simple_reward(completions, self.device)
            
            # 计算组相对优势
            grouped_rewards = rewards.view(-1, self.args.num_generations)
            mean_r = grouped_rewards.mean(dim=1).repeat_interleave(self.args.num_generations)
            std_r = grouped_rewards.std(dim=1).repeat_interleave(self.args.num_generations)
            advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 创建 completion mask (到 EOS 为止)
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is None:
                eos_token_id = 0
            
            is_eos = completion_ids == eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            completion_mask = (torch.arange(is_eos.size(1), device=self.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()
            
            # 计算 KL 散度
            kl_div = ref_per_token_logps - per_token_logps
            per_token_kl = torch.exp(kl_div) - kl_div - 1
            
            # 计算策略损失
            per_token_loss = -(
                torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) 
                - self.args.beta * per_token_kl
            )
            
            # 计算最终损失
            loss = ((per_token_loss * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-8)).mean()
            loss = loss / self.args.accumulation_steps
            
            loss.backward()
            
            if (step + 1) % self.args.accumulation_steps == 0:
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
            
            # 日志
            if step % self.args.log_interval == 0 or step == iters:
                policy_loss_val = loss.item() * self.args.accumulation_steps
                avg_reward_val = rewards.mean().item()
                avg_len_val = completion_mask.sum(dim=1).float().mean().item()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                Logger(f'Epoch: {epoch+1}, Step: {step}/{iters}, '
                       f'Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, '
                       f'Avg Len: {avg_len_val:.2f}, LR: {current_lr:.2e}')
                
                if wandb and is_main_process():
                    wandb.log({
                        "policy_loss": policy_loss_val,
                        "reward": avg_reward_val,
                        "avg_response_len": avg_len_val,
                        "advantages_mean": advantages.mean().item(),
                        "learning_rate": current_lr
                    })
            
            # 保存检查点
            if (step % self.args.save_interval == 0 or step == iters) and is_main_process():
                self._save_checkpoint(epoch, step)
            
            # 清理内存
            del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
            del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask
            torch.cuda.empty_cache()
            gc.collect()
    
    def _save_checkpoint(self, epoch, step):
        """保存模型检查点"""
        Logger(f"\n保存 GRPO 模型 (epoch {epoch+1}, step {step})...")
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取模型状态
        model_state = self.base_model.state_dict()
        
        # 半精度保存
        moe_suffix = '_moe' if self.base_model.config.use_moe else ''
        ckp = output_dir / f'{self.args.save_weight}_{self.base_model.config.hidden_size}{moe_suffix}.pth'
        torch.save({k: v.half() for k, v in model_state.items()}, ckp)
        
        # 保存完整检查点
        checkpoint = {
            "model_state": model_state,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "epoch": epoch,
            "step": step
        }
        torch.save(checkpoint, output_dir / "grpo_checkpoint.pt")
        
        self.base_model.config.save_pretrained(str(output_dir))
        Logger(f"✅ 已保存到: {output_dir}")
    
    def train(self, start_epoch=0, wandb=None):
        """执行完整训练"""
        Logger("\n🚀 开始 GRPO 训练...\n")
        
        for epoch in range(start_epoch, self.args.epochs):
            if hasattr(self.dataloader.sampler, 'set_epoch'):
                self.dataloader.sampler.set_epoch(epoch)
            
            self.train_epoch(epoch, self.args.epochs, wandb)
        
        Logger("✅ GRPO 训练完成!")


def main():
    parser = argparse.ArgumentParser(description="MiniMind GRPO Training")
    
    # 保存配置
    parser.add_argument("--output_dir", type=str, default="./output/grpo", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    
    # GRPO 配置
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--max_gen_len", type=int, default=256, help="生成的最大长度")
    
    # 模型配置
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=128, type=int, help="Prompt最大长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    
    # 数据配置
    parser.add_argument("--data_path", type=str, default="./dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="./unigram_tokenizer.json", help="Tokenizer路径")
    
    # 预训练权重
    parser.add_argument('--from_weight', default='sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="从检查点恢复训练")
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    
    args = parser.parse_args()
    
    # ========== 1. 初始化环境 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录 ==========
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========== 3. 加载 Tokenizer ==========
    Logger(f"Loading tokenizer from {args.tokenizer_path}")
    if not os.path.exists(args.tokenizer_path):
        raise ValueError(f"Tokenizer path does not exist: {args.tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ========== 4. 初始化模型配置 ==========
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe)
    )
    
    # ========== 5. 初始化 Policy 模型 ==========
    Logger(f"Initializing policy model...")
    model = MiniMindForCausalLM(lm_config).to(args.device)
    
    # 加载预训练权重
    if args.from_weight != 'none' and os.path.exists(args.from_weight):
        Logger(f"Loading weights from {args.from_weight}")
        weights = torch.load(args.from_weight, map_location=args.device)
        if isinstance(weights, dict) and 'model_state' in weights:
            model.load_state_dict(weights['model_state'], strict=False)
        else:
            model.load_state_dict(weights, strict=False)
    
    # ========== 6. 初始化 Reference 模型 ==========
    Logger(f"Initializing reference model...")
    ref_model = MiniMindForCausalLM(lm_config).to(args.device)
    
    # 复制 policy 模型的权重
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    ref_model.requires_grad_(False)
    
    Logger(f"模型可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M")
    
    # ========== 7. 加载数据集 ==========
    Logger(f"Loading dataset from {args.data_path}")
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")
    
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # ========== 8. DDP 包装模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DDP(model, device_ids=[local_rank])
    
    # ========== 9. 创建 DataLoader ==========
    dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # ========== 10. 配置 Wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        try:
            import swanlab as wandb
            wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
            wandb.init(project=args.wandb_project, name=wandb_run_name)
        except ImportError:
            Logger("swanlab not installed, skipping wandb logging")
            wandb = None
    
    # ========== 11. 创建训练器并开始训练 ==========
    trainer = GRPOTrainer(args, model, ref_model, tokenizer, dataloader, local_rank)
    
    # 恢复训练
    start_epoch = 0
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.output_dir) / "grpo_checkpoint.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            trainer.base_model.load_state_dict(checkpoint['model_state'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint['epoch'] + 1
            Logger(f"从检查点恢复训练，起始 epoch: {start_epoch}")
    
    trainer.train(start_epoch=start_epoch, wandb=wandb)
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
