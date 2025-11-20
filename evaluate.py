"""
MiniMind 模型评估脚本
支持多种评估指标：困惑度(Perplexity)、准确率、损失等
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from minimind import MiniMindConfig, MiniMindForCausalLM
from dataset import MiniMindTokenizerFast, MinimindDataset


class ModelEvaluator:
    """MiniMind 模型评估器"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "cuda",
        batch_size: int = 4,
        max_seq_len: int = 512
    ):
        """
        初始化评估器
        
        Args:
            model_path: 模型权重路径
            tokenizer_path: 分词器路径
            device: 计算设备
            batch_size: 批次大小
            max_seq_len: 最大序列长度
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        
        # 加载分词器
        print(f"📚 加载分词器: {tokenizer_path}")
        self.tokenizer = MiniMindTokenizerFast.from_pretrained(tokenizer_path)
        
        # 加载模型
        print(f"🤖 加载模型: {model_path}")
        self._load_model(model_path)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        print(f"✅ 评估器初始化完成 (设备: {self.device})")
    
    def _load_model(self, model_path: str):
        """加载模型权重"""
        # 尝试加载配置
        config_path = Path(model_path).parent / "config.json"
        if config_path.exists():
            self.config = MiniMindConfig.from_pretrained(str(config_path.parent))
            # 确保词表大小匹配
            self.config.vocab_size = self.tokenizer.vocab_size
        else:
            # 使用默认配置
            self.config = MiniMindConfig(vocab_size=self.tokenizer.vocab_size)
        
        # 初始化模型
        self.model = MiniMindForCausalLM(self.config)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"], strict=False)
            print(f"📊 已加载检查点 (epoch: {checkpoint.get('epoch', 'unknown')})")
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_dataset(
        self,
        data_path: str,
        num_workers: int = 2,
        show_samples: int = 5
    ) -> Dict[str, float]:
        """
        在数据集上评估模型
        
        Args:
            data_path: 数据集路径（JSONL格式）
            num_workers: 数据加载器工作进程数
            show_samples: 显示多少个生成样本
            
        Returns:
            包含各项指标的字典
        """
        print(f"\n{'='*60}")
        print(f"开始评估: {data_path}")
        print(f"{'='*60}\n")
        
        # 加载数据集
        dataset = MinimindDataset(data_path, max_length=self.max_seq_len)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"📊 数据集大小: {len(dataset)} 样本")
        print(f"📊 批次数量: {len(dataloader)}\n")
        
        # 评估指标
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        
        # 评估循环
        with torch.no_grad():
            for batch_idx, (input_ids, labels, loss_mask) in enumerate(tqdm(dataloader, desc="评估中")):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                loss_mask = loss_mask.to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids)
                logits = outputs.logits
                
                # 计算损失
                loss = self.criterion(
                    logits.reshape(-1, self.config.vocab_size),
                    labels.reshape(-1)
                ).reshape(labels.size())
                
                # 应用mask，只计算非padding位置的损失
                masked_loss = (loss * loss_mask).sum()
                num_tokens = loss_mask.sum()
                
                total_loss += masked_loss.item()
                total_tokens += num_tokens.item()
                
                # 计算准确率
                predictions = logits.argmax(dim=-1)
                correct = ((predictions == labels) * loss_mask).sum()
                correct_predictions += correct.item()
        
        # 计算指标
        metrics = self._calculate_metrics(total_loss, total_tokens, correct_predictions)
        
        # 打印结果
        self._print_metrics(metrics)
        
        # 生成样本
        if show_samples > 0:
            self._generate_samples(dataset, show_samples)
        
        return metrics
    
    def _calculate_metrics(
        self,
        total_loss: float,
        total_tokens: int,
        correct_predictions: int
    ) -> Dict[str, float]:
        """计算评估指标"""
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')  # 避免溢出
        accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "total_tokens": total_tokens
        }
    
    def _print_metrics(self, metrics: Dict[str, float]):
        """打印评估指标"""
        print(f"\n{'='*60}")
        print("评估结果:")
        print(f"{'='*60}")
        print(f"📊 平均损失 (Loss):        {metrics['loss']:.4f}")
        print(f"📊 困惑度 (Perplexity):     {metrics['perplexity']:.4f}")
        print(f"📊 Token准确率 (Accuracy):  {metrics['accuracy']*100:.2f}%")
        print(f"📊 评估Token总数:           {metrics['total_tokens']:,}")
        print(f"{'='*60}\n")
    
    def _generate_samples(self, dataset: MinimindDataset, num_samples: int):
        """生成一些文本样本展示模型生成能力"""
        print(f"\n{'='*60}")
        print(f"生成样本 (共 {num_samples} 个):")
        print(f"{'='*60}\n")
        
        for i in range(min(num_samples, len(dataset))):
            # 获取数据集中的一个样本
            sample_data = dataset.data[i * (len(dataset) // num_samples)]
            prompt_text = sample_data.get("text", "")
            
            # 截取前面一部分作为提示
            if len(prompt_text) > 50:
                prompt = prompt_text[:50]
                expected = prompt_text[50:150]
            else:
                prompt = prompt_text[:len(prompt_text)//2]
                expected = prompt_text[len(prompt_text)//2:]
            
            # 生成文本
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(output_ids[0].tolist())
            generated_continuation = generated[len(prompt):]
            
            print(f"样本 {i+1}:")
            print(f"  提示: {prompt}")
            print(f"  期望: {expected[:100]}...")
            print(f"  生成: {generated_continuation[:100]}...")
            print()
    
    def evaluate_single_text(self, text: str) -> Dict[str, float]:
        """
        评估单个文本样本
        
        Args:
            text: 输入文本
            
        Returns:
            包含各项指标的字典
        """
        # 编码文本
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_seq_len, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        
        # 准备标签（右移一位）
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            
            # 计算损失
            loss = self.criterion(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1)
            )
            avg_loss = loss.mean().item()
            
            # 计算准确率
            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels).sum().item()
            accuracy = correct / labels.numel()
        
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "text_length": len(text)
        }


def main():
    parser = argparse.ArgumentParser(description="MiniMind 模型评估脚本")
    
    # 模型配置
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/minimind_model.pt",
        help="模型权重路径"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./dataset/tokenizer",
        help="分词器路径"
    )
    
    # 数据配置
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/pretrain.jsonl",
        help="评估数据集路径（JSONL格式）"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="最大序列长度"
    )
    
    # 评估配置
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批次大小"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="数据加载器工作进程数"
    )
    parser.add_argument(
        "--show_samples",
        type=int,
        default=5,
        help="显示多少个生成样本"
    )
    
    # 输出配置
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="保存评估结果的JSON文件路径（可选）"
    )
    
    # 单文本评估
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="评估单个文本（如果提供，将不使用数据集）"
    )
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len
    )
    
    # 执行评估
    if args.text:
        # 评估单个文本
        print(f"\n评估单个文本:")
        print(f"文本: {args.text}\n")
        metrics = evaluator.evaluate_single_text(args.text)
        evaluator._print_metrics(metrics)
    else:
        # 评估数据集
        metrics = evaluator.evaluate_dataset(
            data_path=args.data_path,
            num_workers=args.num_workers,
            show_samples=args.show_samples
        )
    
    # 保存结果
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 评估结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
