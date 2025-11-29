"""
MiniMind Knowledge Distillation 训练脚本
用于通过知识蒸馏将大模型的知识迁移到小模型。

知识蒸馏通过让学生模型学习教师模型的软标签（logits 分布）来实现。
"""
import os
import argparse
import time
import warnings
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from minimind import MiniMindConfig, MiniMindForCausalLM
from my_dataset import MinimindDataset

warnings.filterwarnings('ignore')


# ========== Helper Functions ==========
def get_lr(current_step, total_steps, learning_rate, warmup_iters=100, min_lr=0.0):
    """Cosine learning rate schedule with warmup"""
    if current_step < warmup_iters:
        return learning_rate * current_step / warmup_iters
    if current_step > total_steps:
        return min_lr
    decay_ratio = (current_step - warmup_iters) / (total_steps - warmup_iters)
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * 3.14159)))
    return min_lr + coeff * (learning_rate - min_lr)


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


def kl_divergence_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                        temperature: float = 2.0, reduction: str = 'batchmean') -> torch.Tensor:
    """
    计算 KL 散度损失（知识蒸馏核心损失）
    
    Args:
        student_logits: 学生模型的 logits [B, seq_len, vocab_size]
        teacher_logits: 教师模型的 logits [B, seq_len, vocab_size]
        temperature: 温度参数，较高的温度产生更软的分布
        reduction: 损失归约方式
        
    Returns:
        KL 散度损失
    """
    # 应用温度缩放
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    # 计算 KL 散度
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction=reduction)
    
    # 乘以 T^2 来保持梯度规模
    return kl_loss * (temperature ** 2)


class DistillationTrainer:
    """MiniMind 知识蒸馏训练器"""
    
    def __init__(self, args, student_model, teacher_model, tokenizer, dataloader, local_rank=-1):
        self.args = args
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.local_rank = local_rank
        self.device = args.device
        
        # 确保教师模型处于评估模式且不计算梯度
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # 损失函数
        self.ce_loss_fct = nn.CrossEntropyLoss(reduction='none')
        
        # 优化器
        self.optimizer = optim.AdamW(
            student_model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        
        # 混合精度
        device_type = "cuda" if "cuda" in args.device else "cpu"
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        self.autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
        self.scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
        
    @property
    def base_student_model(self):
        """返回基础学生模型实例"""
        if isinstance(self.student_model, DDP):
            return self.student_model.module
        return self.student_model
    
    @property
    def base_teacher_model(self):
        """返回基础教师模型实例"""
        if isinstance(self.teacher_model, DDP):
            return self.teacher_model.module
        return self.teacher_model
    
    def compute_distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                                   labels: torch.Tensor, loss_mask: torch.Tensor) -> dict:
        """
        计算蒸馏损失（软标签损失 + 硬标签损失）
        
        Args:
            student_logits: 学生模型的 logits
            teacher_logits: 教师模型的 logits
            labels: 真实标签
            loss_mask: 损失掩码
            
        Returns:
            包含各损失分量的字典
        """
        vocab_size = student_logits.size(-1)
        
        # 处理词表大小不匹配的情况
        min_vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
        student_logits_trimmed = student_logits[..., :min_vocab_size]
        teacher_logits_trimmed = teacher_logits[..., :min_vocab_size]
        
        # 1. 软标签损失（KL 散度）
        soft_loss = kl_divergence_loss(
            student_logits_trimmed, 
            teacher_logits_trimmed, 
            temperature=self.args.temperature
        )
        
        # 2. 硬标签损失（交叉熵）
        hard_loss = self.ce_loss_fct(
            student_logits.view(-1, vocab_size),
            labels.view(-1)
        ).view(labels.size())
        
        # 应用 loss_mask
        hard_loss = (hard_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        
        # 3. 组合损失
        alpha = self.args.alpha  # 软标签权重
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        
        return {
            'total_loss': total_loss,
            'soft_loss': soft_loss,
            'hard_loss': hard_loss
        }
    
    def train_epoch(self, epoch, total_epochs, wandb=None):
        """训练一个 epoch"""
        self.student_model.train()
        iters = len(self.dataloader)
        total_iters = total_epochs * iters
        start_time = time.time()
        
        # 仅在主进程上使用 tqdm
        data_iterator = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{total_epochs}") if is_main_process() else self.dataloader
        
        for step, (X, Y, loss_mask) in enumerate(data_iterator, start=1):
            X = X.to(self.device)
            Y = Y.to(self.device)
            loss_mask = loss_mask.to(self.device)
            
            # 更新学习率
            current_step = epoch * iters + step
            lr = get_lr(current_step, total_iters, self.args.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            with self.autocast_ctx:
                # 获取学生模型输出
                student_outputs = self.student_model(X)
                student_logits = student_outputs.logits
                
                # 获取教师模型输出（无梯度）
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(X)
                    teacher_logits = teacher_outputs.logits
                
                # 计算蒸馏损失
                losses = self.compute_distillation_loss(
                    student_logits, teacher_logits, Y, loss_mask
                )
                
                total_loss = losses['total_loss']
                
                # 处理 MoE 辅助损失
                if hasattr(student_outputs, 'aux_loss') and student_outputs.aux_loss is not None:
                    total_loss = total_loss + student_outputs.aux_loss
                
                total_loss = total_loss / self.args.accumulation_steps
            
            self.scaler.scale(total_loss).backward()
            
            if (step + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.args.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            
            # 日志
            if step % self.args.log_interval == 0 or step == iters:
                spend_time = time.time() - start_time
                current_loss = total_loss.item() * self.args.accumulation_steps
                soft_loss_val = losses['soft_loss'].item()
                hard_loss_val = losses['hard_loss'].item()
                current_lr = self.optimizer.param_groups[-1]['lr']
                eta_min = spend_time / step * iters // 60 - spend_time // 60
                
                Logger(f'Epoch:[{epoch+1}/{total_epochs}]({step}/{iters}) '
                       f'total_loss:{current_loss:.6f} soft:{soft_loss_val:.6f} hard:{hard_loss_val:.6f} '
                       f'lr:{current_lr:.12f} epoch_Time:{eta_min}min')
                
                if wandb and is_main_process():
                    wandb.log({
                        "total_loss": current_loss, 
                        "soft_loss": soft_loss_val,
                        "hard_loss": hard_loss_val,
                        "lr": current_lr, 
                        "epoch_Time": eta_min
                    })
            
            # 保存检查点
            if (step % self.args.save_interval == 0 or step == iters) and is_main_process():
                self._save_checkpoint(epoch, step)
    
    def _save_checkpoint(self, epoch, step):
        """保存模型检查点"""
        Logger(f"\n保存蒸馏模型 (epoch {epoch+1}, step {step})...")
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取模型状态
        model_state = self.base_student_model.state_dict()
        
        # 半精度保存
        moe_suffix = '_moe' if self.base_student_model.config.use_moe else ''
        ckp = output_dir / f'{self.args.save_weight}_{self.base_student_model.config.hidden_size}{moe_suffix}.pth'
        torch.save({k: v.half() for k, v in model_state.items()}, ckp)
        
        # 保存完整检查点
        checkpoint = {
            "model_state": model_state,
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": step
        }
        torch.save(checkpoint, output_dir / "distill_checkpoint.pt")
        
        self.base_student_model.config.save_pretrained(str(output_dir))
        Logger(f"✅ 已保存到: {output_dir}")
    
    def train(self, start_epoch=0, wandb=None):
        """执行完整训练"""
        Logger("\n🚀 开始知识蒸馏训练...\n")
        
        for epoch in range(start_epoch, self.args.epochs):
            if hasattr(self.dataloader.sampler, 'set_epoch'):
                self.dataloader.sampler.set_epoch(epoch)
            
            self.train_epoch(epoch, self.args.epochs, wandb)
        
        Logger("✅ 知识蒸馏训练完成!")


def main():
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation Training")
    
    # 保存配置
    parser.add_argument("--output_dir", type=str, default="./output/distill", help="模型保存目录")
    parser.add_argument('--save_weight', default='distill', type=str, help="保存权重的前缀名")
    
    # 蒸馏配置
    parser.add_argument("--temperature", type=float, default=2.0, help="蒸馏温度")
    parser.add_argument("--alpha", type=float, default=0.5, help="软标签损失权重 (1-alpha 为硬标签权重)")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    
    # 学生模型配置
    parser.add_argument('--student_hidden_size', default=256, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=4, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    
    # 教师模型配置
    parser.add_argument('--teacher_hidden_size', default=512, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师模型隐藏层数量")
    parser.add_argument("--teacher_weight", type=str, default="./output/sft/sft_512.pth", help="教师模型权重路径")
    
    # 数据配置
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain.jsonl", help="训练数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="./unigram_tokenizer.json", help="Tokenizer路径")
    
    # 学生模型预训练权重
    parser.add_argument('--student_weight', default='none', type=str, help="学生模型初始权重")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="从检查点恢复训练")
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distill", help="wandb项目名")
    
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
    
    # ========== 4. 初始化教师模型 ==========
    Logger(f"Initializing teacher model...")
    teacher_config = MiniMindConfig(
        hidden_size=args.teacher_hidden_size,
        num_hidden_layers=args.teacher_num_layers,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_seq_len,
        use_moe=bool(args.use_moe)
    )
    teacher_model = MiniMindForCausalLM(teacher_config).to(args.device)
    
    # 加载教师模型权重
    if os.path.exists(args.teacher_weight):
        Logger(f"Loading teacher weights from {args.teacher_weight}")
        weights = torch.load(args.teacher_weight, map_location=args.device)
        if isinstance(weights, dict) and 'model_state' in weights:
            teacher_model.load_state_dict(weights['model_state'], strict=False)
        else:
            teacher_model.load_state_dict(weights, strict=False)
    else:
        Logger(f"⚠️ 教师模型权重不存在: {args.teacher_weight}，使用随机初始化")
    
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    Logger(f"教师模型参数: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M")
    
    # ========== 5. 初始化学生模型 ==========
    Logger(f"Initializing student model...")
    student_config = MiniMindConfig(
        hidden_size=args.student_hidden_size,
        num_hidden_layers=args.student_num_layers,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_seq_len,
        use_moe=bool(args.use_moe)
    )
    student_model = MiniMindForCausalLM(student_config).to(args.device)
    
    # 加载学生模型初始权重（如果有）
    if args.student_weight != 'none' and os.path.exists(args.student_weight):
        Logger(f"Loading student weights from {args.student_weight}")
        weights = torch.load(args.student_weight, map_location=args.device)
        if isinstance(weights, dict) and 'model_state' in weights:
            student_model.load_state_dict(weights['model_state'], strict=False)
        else:
            student_model.load_state_dict(weights, strict=False)
    
    Logger(f"学生模型可训练参数: {sum(p.numel() for p in student_model.parameters() if p.requires_grad) / 1e6:.3f} M")
    Logger(f"压缩比: {sum(p.numel() for p in teacher_model.parameters()) / sum(p.numel() for p in student_model.parameters()):.2f}x")
    
    # ========== 6. 加载数据集 ==========
    Logger(f"Loading dataset from {args.data_path}")
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")
    
    train_ds = MinimindDataset(
        args.data_path,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len
    )
    
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # ========== 7. DDP 包装学生模型 ==========
    if dist.is_initialized():
        student_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        student_model = DDP(student_model, device_ids=[local_rank])
    
    # ========== 8. 创建 DataLoader ==========
    dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # ========== 9. 配置 Wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        try:
            import swanlab as wandb
            wandb_run_name = f"MiniMind-Distill-T{args.teacher_hidden_size}-S{args.student_hidden_size}"
            wandb.init(project=args.wandb_project, name=wandb_run_name)
        except ImportError:
            Logger("swanlab not installed, skipping wandb logging")
            wandb = None
    
    # ========== 10. 创建训练器并开始训练 ==========
    trainer = DistillationTrainer(
        args, student_model, teacher_model, tokenizer, dataloader, local_rank
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.output_dir) / "distill_checkpoint.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            trainer.base_student_model.load_state_dict(checkpoint['model_state'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            Logger(f"从检查点恢复训练，起始 epoch: {start_epoch}")
    
    trainer.train(start_epoch=start_epoch, wandb=wandb)
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
