"""
MiniMind SFT (Supervised Fine-Tuning) 训练脚本 - 使用 torchrun/DistributedDataParallel (DDP)
用于在预训练模型基础上进行监督微调训练。
"""
import os
import argparse
import time
import warnings
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
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


class SFTTrainer:
    """MiniMind SFT 训练器"""
    
    def __init__(self, args, model, tokenizer, dataloader, local_rank=-1):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.local_rank = local_rank
        self.device = args.device
        
        # 损失函数
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        
        # 混合精度
        device_type = "cuda" if "cuda" in args.device else "cpu"
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        self.autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
        self.scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
        
    @property
    def base_model(self):
        """返回基础模型实例"""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
    
    def train_epoch(self, epoch, total_epochs, wandb=None):
        """训练一个 epoch"""
        self.model.train()
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
                outputs = self.model(X)
                loss = self.loss_fct(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                
                # 应用 loss_mask
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                
                # 处理 MoE 辅助损失
                if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                    loss = loss + outputs.aux_loss
                
                loss = loss / self.args.accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            
            # 日志
            if step % self.args.log_interval == 0 or step == iters:
                spend_time = time.time() - start_time
                current_loss = loss.item() * self.args.accumulation_steps
                current_lr = self.optimizer.param_groups[-1]['lr']
                eta_min = spend_time / step * iters // 60 - spend_time // 60
                
                Logger(f'Epoch:[{epoch+1}/{total_epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min')
                
                if wandb and is_main_process():
                    wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})
            
            # 保存检查点
            if (step % self.args.save_interval == 0 or step == iters) and is_main_process():
                self._save_checkpoint(epoch, step)
    
    def _save_checkpoint(self, epoch, step):
        """保存模型检查点"""
        Logger(f"\n保存模型 (epoch {epoch+1}, step {step})...")
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取模型状态
        model_state = self.base_model.state_dict()
        
        # 半精度保存
        moe_suffix = '_moe' if self.base_model.config.use_moe else ''
        ckp = output_dir / f'{self.args.save_weight}_{self.base_model.config.hidden_size}{moe_suffix}.pth'
        torch.save({k: v.half() for k, v in model_state.items()}, ckp)
        
        # 保存完整检查点用于恢复训练
        checkpoint = {
            "model_state": model_state,
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": step
        }
        torch.save(checkpoint, output_dir / "sft_checkpoint.pt")
        
        # 保存配置
        self.base_model.config.save_pretrained(str(output_dir))
        Logger(f"✅ 已保存到: {output_dir}")
    
    def train(self, start_epoch=0, wandb=None):
        """执行完整训练"""
        Logger("\n🚀 开始 SFT 训练...\n")
        
        for epoch in range(start_epoch, self.args.epochs):
            if hasattr(self.dataloader.sampler, 'set_epoch'):
                self.dataloader.sampler.set_epoch(epoch)
            
            self.train_epoch(epoch, self.args.epochs, wandb)
        
        Logger("✅ SFT 训练完成!")


def main():
    parser = argparse.ArgumentParser(description="MiniMind SFT Training")
    
    # 保存配置
    parser.add_argument("--output_dir", type=str, default="./output/sft", help="模型保存目录")
    parser.add_argument('--save_weight', default='sft', type=str, help="保存权重的前缀名")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    
    # 模型配置
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    
    # 数据配置
    parser.add_argument("--data_path", type=str, default="./dataset/sft.jsonl", help="SFT训练数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="./unigram_tokenizer.json", help="Tokenizer路径")
    
    # 预训练权重
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="从检查点恢复训练")
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-SFT", help="wandb项目名")
    
    args = parser.parse_args()
    
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录和模型参数 ==========
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========== 3. 加载 Tokenizer ==========
    Logger(f"Loading tokenizer from {args.tokenizer_path}")
    if not os.path.exists(args.tokenizer_path):
        raise ValueError(f"Tokenizer path does not exist: {args.tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    
    # ========== 4. 初始化模型配置 ==========
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_seq_len,
        use_moe=bool(args.use_moe)
    )
    
    # ========== 5. 初始化模型 ==========
    Logger(f"Initializing model with config: hidden_size={lm_config.hidden_size}, layers={lm_config.num_hidden_layers}")
    model = MiniMindForCausalLM(lm_config).to(args.device)
    
    # 加载预训练权重
    if args.from_weight != 'none' and os.path.exists(args.from_weight):
        Logger(f"Loading weights from {args.from_weight}")
        weights = torch.load(args.from_weight, map_location=args.device)
        if isinstance(weights, dict) and 'model_state' in weights:
            model.load_state_dict(weights['model_state'], strict=False)
        else:
            model.load_state_dict(weights, strict=False)
    
    Logger(f"模型可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M")
    
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
    
    # ========== 7. DDP 包装模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DDP(model, device_ids=[local_rank])
    
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
            wandb_run_name = f"MiniMind-SFT-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
            wandb.init(project=args.wandb_project, name=wandb_run_name)
        except ImportError:
            Logger("swanlab not installed, skipping wandb logging")
            wandb = None
    
    # ========== 10. 创建训练器并开始训练 ==========
    trainer = SFTTrainer(args, model, tokenizer, dataloader, local_rank)
    
    # 恢复训练
    start_epoch = 0
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.output_dir) / "sft_checkpoint.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            trainer.base_model.load_state_dict(checkpoint['model_state'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            Logger(f"从检查点恢复训练，起始 epoch: {start_epoch}")
    
    trainer.train(start_epoch=start_epoch, wandb=wandb)
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
