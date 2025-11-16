"""
MiniMind 训练脚本 - 精简高效实现
"""
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from minimind import MiniMindConfig, MiniMindForCausalLM
from tokenizer import SimpleCharTokenizer
from dataset import MinimindDataset, SimpleTextDataset


class Trainer:
    """MiniMind 训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"🖥️  设备: {self.device}")
        
        # 初始化tokenizer
        if args.tokenizer_path and os.path.exists(args.tokenizer_path):
            self.tokenizer = SimpleCharTokenizer.load(args.tokenizer_path)
        else:
            self.tokenizer = SimpleCharTokenizer(vocab_size=args.vocab_size)
        
        # 初始化模型
        config = MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            vocab_size=self.tokenizer.get_vocab_size(),
            max_position_embeddings=args.max_seq_len,
            dropout=args.dropout,
            use_moe=args.use_moe
        )
        
        self.model = MiniMindForCausalLM(config)
        self.model.to(self.device)
        
        # 加载预训练权重
        if args.pretrained_path and os.path.exists(args.pretrained_path):
            state_dict = torch.load(args.pretrained_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        
        # 初始化优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        ) if args.use_scheduler else None
        
        # 初始化数据加载器
        if args.use_jsonl and os.path.exists(args.data_path):
            dataset = MinimindDataset(args.data_path, max_length=args.max_seq_len)
        else:
            texts = [
                "这是一个简单的训练示例。",
                "MiniMind 是一个小语言模型。",
                "我们正在使用示例数据进行训练。",
                "语言模型通过预测下一个token来学习。",
            ] * (args.batch_size * 10)
            dataset = SimpleTextDataset(texts, self.tokenizer, max_length=args.max_seq_len)
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def train(self):
        """执行训练"""
        print("\n" + "="*60)
        print("🚀 开始训练...")
        print("="*60 + "\n")
        
        self.model.train()
        
        for epoch in range(self.args.epochs):
            print(f"📊 Epoch {epoch + 1}/{self.args.epochs}")
            
            total_loss = 0.0
            num_batches = 0
            
            with tqdm(total=len(self.dataloader), desc="训练进度") as pbar:
                for input_ids, labels in self.dataloader:
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(input_ids)
                    loss = self.criterion(
                        outputs.logits.view(-1, self.model.config.vocab_size),
                        labels.view(-1)
                    )
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    if self.args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    avg_loss = total_loss / num_batches
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            print(f"   ✅ Epoch 平均损失: {total_loss / num_batches:.4f}\n")
        
        # 保存模型
        self._save_model()
    
    def _save_model(self):
        """保存模型"""
        print("\n" + "="*60)
        print("💾 保存模型...")
        print("="*60)
        
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), output_dir / "minimind_model.pt")
        
        # 保存配置
        self.model.config.save_pretrained(str(output_dir))
        
        # 保存tokenizer
        self.tokenizer.save(str(output_dir / "tokenizer.pkl"))
        
        print(f"✅ 模型已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    
    # 模型配置
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=6400)
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
    parser.add_argument("--tokenizer_path", type=str, default=None)
    
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
