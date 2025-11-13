"""
简单的MiniMind模型训练脚本
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from minimind import MiniMindConfig, MiniMindForCausalLM
import argparse
import os


class SimpleTextDataset(Dataset):
    """简单的文本数据集"""
    def __init__(self, data, max_length=512):
        self.data = data
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        # 简单的tokenization (实际使用时应该使用tokenizer)
        tokens = [ord(c) % 6400 for c in text[:self.max_length]]
        # 填充到固定长度
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, labels


def train(args):
    """训练函数"""
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型配置
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_seq_len,
        dropout=args.dropout,
        use_moe=args.use_moe
    )
    
    # 初始化模型
    print("初始化模型...")
    model = MiniMindForCausalLM(config)
    model = model.to(device)
    
    # 如果有预训练权重，加载它
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"加载预训练权重: {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    
    # 准备示例数据（实际使用时应该从文件加载）
    sample_texts = [
        "Hello, this is a simple training example.",
        "MiniMind is a small language model.",
        "We are training the model with some sample data.",
        "This is just for demonstration purposes."
    ] * 100  # 重复数据以便训练
    
    # 创建数据集和数据加载器
    dataset = SimpleTextDataset(sample_texts, max_length=args.max_seq_len)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print(f"\n开始训练 {args.epochs} 个epoch...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(input_ids)
            logits = outputs.logits
            
            # 计算损失
            loss = criterion(
                logits.view(-1, config.vocab_size),
                labels.view(-1)
            )
            
            # 如果使用MoE，添加辅助损失
            if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                loss = loss + outputs.aux_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # 更新参数
            optimizer.step()
            
            total_loss += loss.item()
            
            # 打印训练信息
            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{args.epochs}], "
                      f"Step [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Avg Loss: {avg_loss:.4f}")
        
        # Epoch结束后保存模型
        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存到: {save_path}")
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n训练完成! 最终模型已保存到: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind简单训练脚本")
    
    # 模型配置
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=8, help="隐藏层数量")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数量")
    parser.add_argument("--vocab_size", type=int, default=6400, help="词表大小")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用MoE (0=否, 1=是)")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数")
    
    # 其他配置
    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备")
    parser.add_argument("--output_dir", type=str, default="./output", help="模型保存目录")
    parser.add_argument("--pretrained_path", type=str, default="", help="预训练权重路径")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1, help="模型保存间隔(epoch)")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("MiniMind 训练配置:")
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 50)
    
    train(args)
