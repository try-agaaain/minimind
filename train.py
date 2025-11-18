"""
MiniMind DDP 训练脚本 - 使用 torchrun/DistributedDataParallel (DDP)
相比 nn.DataParallel，DDP 具有更好的性能和内存效率。
"""
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from minimind import MiniMindConfig, MiniMindForCausalLM
from dataset import MiniMindTokenizerFast, NovelDatasetPreparator, MinimindDataset 


class Trainer:
    """MiniMind 分布式训练器 (DDP)"""
    
    def __init__(self, args, rank, world_size):
        self.args = args
        self.rank = rank
        self.world_size = world_size

        # 核心 DDP 逻辑：使用 local_rank 绑定到唯一的 GPU
        if args.local_rank == -1 or world_size == 0:
             # 在 main() 中已处理该错误，这里作为二次检查
             raise ValueError("DDP 环境初始化失败：local_rank 无效或 world_size 为 0。")
        
        self.device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(self.device)
        
        # 仅在主进程上打印信息
        if self.rank == 0:
            print(f"设备: {self.device} (进程 {self.rank}/{self.world_size})")
        
        # --- 初始化 Tokenizer 和 Config ---
        tokenizer_path = args.tokenizer_path
        # 只有 rank 0 检查并准备文件
        if not os.path.exists(tokenizer_path) and self.rank == 0:
            print(f"⚠️  Tokenizer 路径不存在: {tokenizer_path}，对数据集进行词元处理")
            preparator = NovelDatasetPreparator(
                dataset_dir=args.dataset_dir,
                pretrain_path=args.pretrain_path,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                tokenizer_path=args.tokenizer_path
            )
            preparator.prepare_dataset()
        
        # 确保 rank 0 完成文件准备后再继续（避免其他进程找不到文件）
        if self.world_size > 1:
            dist.barrier() 

        self.tokenizer = MiniMindTokenizerFast.from_pretrained(tokenizer_path)
        
        # 初始化模型配置
        config = MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=args.max_seq_len,
            dropout=args.dropout,
            use_moe=args.use_moe
        )
        # 将模型实例化并移动到指定设备
        model = MiniMindForCausalLM(config).to(self.device)
        
        # --- 检查点和预训练加载 ---
        self.epoch = 0
        
        if args.resume_from_checkpoint and os.path.exists(args.output_dir):
            # 注意：在 DDP 中，只有主进程进行文件 I/O
            checkpoint_path = Path(args.output_dir) / "minimind_model.pt"
            if checkpoint_path.exists():
                # 使用 map_location 确保加载到正确的设备
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    model.load_state_dict(checkpoint["model_state"], strict=False)
                    self.epoch = checkpoint.get("epoch", 0)
                    if self.rank == 0:
                        print(f"从断点继续训练，起始 epoch: {self.epoch}")
                else:
                    model.load_state_dict(checkpoint, strict=False)
            elif self.rank == 0:
                print(f"⚠️  未找到断点文件: {checkpoint_path}")
        elif args.pretrained_path and os.path.exists(args.pretrained_path):
            model.load_state_dict(torch.load(args.pretrained_path, map_location=self.device), strict=False)
        
        # --- DDP 包装模型 ---
        # 这一步将模型注册到分布式组中
        self.model = DDP(model, device_ids=[args.local_rank])
        
        # --- 优化器和调度器 ---
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr) if args.use_scheduler else None
        
        # --- 数据加载器和 Sampler ---
        if args.use_jsonl and os.path.exists(args.data_path):
            dataset = MinimindDataset(args.data_path, max_length=args.max_seq_len)
        else:
            raise FileNotFoundError(f"⚠️  数据路径不存在或未指定 JSONL 格式: {args.data_path}")
        
        # DDP: 使用 DistributedSampler，它根据 self.rank 和 self.world_size 划分数据
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                     num_workers=args.num_workers, pin_memory=True)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    @property
    def base_model(self):
        """返回基础模型实例，即 DDP 包装下的 .module。"""
        return self.model.module

    def train(self):
        """执行训练"""
        if self.rank == 0:
            print("\n🚀 开始 DDP 训练...\n")
        self.model.train()
        
        for epoch in range(self.epoch, self.args.epochs):
            # DDP: 每次 epoch 开始时设置 Sampler，确保不同 epoch 得到不同的数据顺序
            self.dataloader.sampler.set_epoch(epoch)
            
            if self.rank == 0:
                print(f"Epoch {epoch + 1}/{self.args.epochs}")
            
            total_loss = 0.0
            
            # 仅在主进程上使用 tqdm
            data_iterator = tqdm(self.dataloader, desc="训练") if self.rank == 0 else self.dataloader
            
            for input_ids, labels, loss_mask in data_iterator:
                # 每个进程将自己的数据加载到自己的 GPU 上
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                loss_mask = loss_mask.to(self.device)
                
                outputs = self.model(input_ids)
                
                # 使用 reduction='none' 计算每个位置的损失，然后通过 loss_mask 加权
                loss = self.criterion(
                    outputs.logits.view(-1, self.base_model.config.vocab_size),
                    labels.view(-1)
                ).view(labels.size())
                
                # 应用 loss_mask，忽略 padding 位置的损失
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                
                # 如果模型支持辅助损失（例如 MoE），添加辅助损失
                if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                    loss = loss + outputs.aux_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪在 DDP 模型上执行
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                self.optimizer.step()
                total_loss += loss.item()
                
                if self.rank == 0:
                    data_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if self.scheduler:
                self.scheduler.step()
            
            # 仅在主进程上报告平均损失
            if self.rank == 0:
                avg_loss = total_loss / len(self.dataloader)
                print(f"平均损失: {avg_loss:.4f}\n")
            
            # 按间隔保存模型，仅在主进程上执行
            if self.rank == 0 and (epoch + 1) % self.args.save_interval == 0:
                self._save_model(epoch)
        
        if self.rank == 0:
            self._save_model(self.args.epochs - 1)
        
        # 结束训练时销毁进程组
        dist.destroy_process_group()
    
    def _save_model(self, epoch):
        """保存模型 (仅在 rank 0 上执行)"""
        if self.rank != 0:
            return
            
        print(f"\n保存模型 (epoch {epoch})...")
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用 self.base_model 获取未包装的模型 (即 DDP.module)
        model_state = self.base_model.state_dict()
        checkpoint = {
            "model_state": model_state,
            "epoch": epoch
        }
        torch.save(checkpoint, output_dir / "minimind_model.pt")
        
        # 保存配置和 tokenizer
        self.base_model.config.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir / "tokenizer"))
        print(f"✅ 已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    
    # DDP/torchrun 自动设置 local_rank
    # 修复: 从环境变量获取 local_rank，如果不存在则默认为 -1
    local_rank_env = os.environ.get("LOCAL_RANK")
    parser.add_argument("--local_rank", type=int, default=int(local_rank_env) if local_rank_env is not None else -1, help="Local rank is set by torchrun")
    
    # ... 其他参数保持不变 ...
    # 数据准备配置
    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--chunk_overlap", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--tokenizer_path", type=str, default="./dataset/tokenizer")
    parser.add_argument("--pretrain_path", type=str, default="./dataset/pretrain.jsonl")
    
    # 模型配置
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_moe", action="store_true")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_scheduler", action="store_true")
    
    # 数据配置
    parser.add_argument("--use_jsonl", action="store_true")
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain.jsonl")
    
    # 检查点配置
    parser.add_argument("--save_interval", type=int, default=1, help="每多少轮保存一次模型")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="从上次保存的模型继续训练")
    
    # 其他配置
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output")
    # DDP 环境中 device 总是 "cuda"
    parser.add_argument("--device", type=str, default="cuda") 
    parser.add_argument("--gpu_ids", type=str, default=None, help="GPU设备编号，在 DDP 中通常不需要手动设置")
    parser.add_argument("--num_workers", type=int, default=2)
    
    args = parser.parse_args()
    
    # --- DDP 初始化检查 ---
    # 如果 local_rank 是 -1 且没有 WORLD_SIZE 环境变量，则脚本未通过 torchrun 正确启动
    if args.local_rank == -1 and 'WORLD_SIZE' not in os.environ:
        print("致命错误：未检测到 DDP 环境变量。请使用 'torchrun --nproc_per_node=N train_ddp.py ...' 启动脚本。")
        return # 退出程序，避免继续执行 DDP 初始化
    
    # torchrun 会自动设置这些环境变量，并由 dist.init_process_group() 读取
    dist.init_process_group(backend="nccl") 
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    trainer = Trainer(args, rank, world_size)
    trainer.train()

if __name__ == "__main__":
    main()