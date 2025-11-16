"""
MiniMind 数据集处理 - 使用 langchain-text-splitters 和 torch Dataset
"""
import json
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tokenizer import SimpleCharTokenizer


class NovelDatasetPreparator:
    """小说数据集准备器 - 文本分割 -> tokenization -> JSONL"""
    
    def __init__(
        self,
        dataset_dir: str = "./dataset",
        output_path: str = "./dataset/pretrain.jsonl",
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        vocab_size: int = 6400,
        tokenizer_path: Optional[str] = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_path = Path(output_path)
        
        # 初始化 tokenizer
        tokenizer_file = Path(tokenizer_path) if tokenizer_path else None
        if tokenizer_file and tokenizer_file.exists():
            self.tokenizer = SimpleCharTokenizer.load(str(tokenizer_file))
        else:
            self.tokenizer = SimpleCharTokenizer(vocab_size=vocab_size)
        
        # 文本分割器
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "，", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    def get_novel_files(self) -> list:
        """获取所有小说文件"""
        return sorted(self.dataset_dir.rglob("*.txt"))
    
    def load_novel_text(self, file_path: Path) -> Optional[str]:
        """加载小说文本，支持多种编码"""
        for enc in ["utf-8", "gbk", "gb2312"]:
            try:
                return file_path.read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue
        print(f"⚠️  无法读取文件: {file_path}")
        return None
    
    def prepare_dataset(self) -> None:
        """准备数据集并保存为 JSONL"""
        novels = self.get_novel_files()
        print(f"📚 找到 {len(novels)} 个小说文件\n")
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        total_chunks = 0
        
        with open(self.output_path, 'w', encoding='utf-8') as out_f:
            for novel_path in tqdm(novels, desc="📖 处理小说"):
                text = self.load_novel_text(novel_path)
                if text is None:
                    continue
                
                # 清理文本
                text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
                
                # 分割并保存
                chunks = self.splitter.split_text(text)
                for chunk in chunks:
                    record = {
                        "text": chunk,
                        "token_ids": self.tokenizer.encode(chunk),
                        "source": novel_path.name,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    total_chunks += 1
        
        print(f"✅ 数据集已保存: {self.output_path} ({total_chunks} 个块)")
        
        # 保存 tokenizer
        tokenizer_path = self.output_path.parent / "tokenizer.pkl"
        self.tokenizer.save(str(tokenizer_path))


class MinimindDataset(Dataset):
    """从 JSONL 文件加载已 tokenized 的数据"""
    
    def __init__(self, jsonl_path: str, max_length: int = 512):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        token_ids = self.data[idx]["token_ids"]
        
        # 截断或填充
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))
        
        token_ids = torch.tensor(token_ids[:self.max_length], dtype=torch.long)
        return token_ids[:-1], token_ids[1:]


class SimpleTextDataset(Dataset):
    """简单文本数据集 - 动态 tokenization"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        token_ids = self.tokenizer.encode(self.texts[idx])
        
        # 截断或填充
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))
        
        token_ids = torch.tensor(token_ids[:self.max_length], dtype=torch.long)
        return token_ids[:-1], token_ids[1:]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--output", default="./dataset/pretrain.jsonl")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--chunk_overlap", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--tokenizer_path", default=None)
    
    args = parser.parse_args()
    
    preparator = NovelDatasetPreparator(
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vocab_size=args.vocab_size,
        tokenizer_path=args.tokenizer_path
    )
    preparator.prepare_dataset()
