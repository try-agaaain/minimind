"""
MiniMind 训练脚本 - 精简高效实现
"""
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from minimind import MiniMindConfig, MiniMindForCausalLM
from dataset import MiniMindTokenizerFast, NovelDatasetPreparator
from dataset import MinimindDataset

class Trainer:
    """MiniMind 训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"设备: {self.device}")
        
        # 初始化tokenizer
        tokenizer_path = args.tokenizer_path
        if not os.path.exists(tokenizer_path):
            print(f"⚠️  Tokenizer 路径不存在: {tokenizer_path}")
            print(f"   使用示例数据进行演示训练")
            preparator = NovelDatasetPreparator(
                dataset_dir=args.dataset_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                tokenizer_path=args.tokenizer_path
            )
            preparator.prepare_dataset()
        self.tokenizer = MiniMindTokenizerFast.from_pretrained(tokenizer_path)
        
        # 初始化模型
        config = MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=args.max_seq_len,
            dropout=args.dropout,
            use_moe=args.use_moe
        )
        self.model = MiniMindForCausalLM(config).to(self.device)
        
        # 初始化 epoch
        self.epoch = 0
        
        # 加载预训练权重或断点续训
        if args.resume_from_checkpoint and os.path.exists(args.output_dir):
            checkpoint_path = Path(args.output_dir) / "minimind_model.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state"], strict=False)
                    self.epoch = checkpoint.get("epoch", 0)
                    print(f"从断点继续训练，起始 epoch: {self.epoch}")
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
            else:
                print(f"⚠️  未找到断点文件: {checkpoint_path}")
        elif args.pretrained_path and os.path.exists(args.pretrained_path):
            self.model.load_state_dict(torch.load(args.pretrained_path, map_location=self.device), strict=False)
        
        # 初始化优化器和调度器
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr) if args.use_scheduler else None
        
        # 初始化数据加载器
        if args.use_jsonl and os.path.exists(args.data_path):
            dataset = MinimindDataset(args.data_path, max_length=args.max_seq_len)
        else:
            assert False, f"⚠️  数据路径不存在或未指定 JSONL 格式: {args.data_path}"
        
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                                    num_workers=args.num_workers, pin_memory=self.device.type == 'cuda')
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def train(self):
        """执行训练"""
        print("\n🚀 开始训练...\n")
        self.model.train()
        
        for epoch in range(self.epoch, self.args.epochs):
            print(f"Epoch {epoch + 1}/{self.args.epochs}")
            total_loss = 0.0
            
            with tqdm(total=len(self.dataloader), desc="训练") as pbar:
                for input_ids, labels in self.dataloader:
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(input_ids)
                    loss = self.criterion(
                        outputs.logits.view(-1, self.model.config.vocab_size),
                        labels.view(-1)
                    )
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    if self.args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    
                    self.optimizer.step()
                    total_loss += loss.item()
                    
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if self.scheduler:
                self.scheduler.step()
            
            print(f"平均损失: {total_loss / len(self.dataloader):.4f}\n")
            
            # 按间隔保存模型
            if (epoch + 1) % self.args.save_interval == 0:
                self._save_model(epoch)
        
        self._save_model(self.args.epochs - 1)
    
    def _save_model(self, epoch):
        """保存模型"""
        print(f"\n保存模型 (epoch {epoch})...")
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state": self.model.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, output_dir / "minimind_model.pt")
        self.model.config.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir / "tokenizer"))
        print(f"✅ 已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    
    # 数据准备配置
    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--chunk_overlap", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--tokenizer_path", type=str, default="./dataset/tokenizer")
    
    # 模型配置
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    # parser.add_argument("--vocab_size", type=int, default=6400)
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    
    args = parser.parse_args()
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
