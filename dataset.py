"""
MiniMind 数据集处理 - 使用 langchain-text-splitters 和 torch Dataset
"""
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Iterator

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

class MiniMindTokenizerFast(PreTrainedTokenizerFast):
    """
    MiniMind 分词器 - 继承 PreTrainedTokenizerFast
    """
    
    # 告诉父类，底层 tokenizers 库的文件叫什么名字
    tokenizer_file = "tokenizer.json" 
    
    # 模型输入名称
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    
    def __init__(
        self, 
        tokenizer_object: Optional[Tokenizer] = None, 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        cls_token="[CLS]", 
        sep_token="[SEP]", 
        mask_token="[MASK]", 
        **kwargs
    ):
        """
        初始化方法配置特殊标记并调用父类的 __init__。
        
        Args:
            tokenizer_object: 底层的 tokenizers.Tokenizer 对象
            unk_token: 未知标记，默认 "[UNK]"
            pad_token: 填充标记，默认 "[PAD]"
            cls_token: 分类标记，默认 "[CLS]"
            sep_token: 分隔符，默认 "[SEP]"
            mask_token: 掩码标记，默认 "[MASK]"
            **kwargs: 其他参数传递给父类
        """
        super().__init__(
            tokenizer_object=tokenizer_object,
            unk_token=unk_token,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, model_id_or_path: str, **kwargs) -> "MiniMindTokenizerFast":
        """
        从预训练模型加载 tokenizer
        
        Args:
            model_id_or_path: 模型 ID 或本地路径
            **kwargs: 其他参数传递给父类
        
        Returns:
            MiniMindTokenizerFast: 加载的 tokenizer 实例
        """
        # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        tokenizer = super().from_pretrained(model_id_or_path, **kwargs)
        print(f"✅ 已加载 (词表大小: {tokenizer.vocab_size})")
        return tokenizer


# ============================================================
# 步骤 3: 便利函数 - 用于训练和保存流程
# ============================================================

def train_and_save_tokenizer(
    files: List[str],
    save_path: str,
    vocab_size: int = 6400,
) -> MiniMindTokenizerFast:
    """
    使用 BPE 算法训练分词器并保存（包括 tokenizer.json 和 tokenizer_config.json）
    
    Args:
        files: 训练文本文件列表
        save_path: 保存路径
        vocab_size: 词表大小
        min_frequency: 最小频率
    
    Returns:
        MiniMindTokenizerFast: 训练后的 tokenizer
    """
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise ValueError("没有找到有效的训练文件")

    # 1. 初始化 BPE Tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    print(f"📚 训练分词器 ({len(files)} 个文件)...")
    
    # 2. 配置 BPE 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # 3. 从文件列表训练
    tokenizer.train(
        files=files,
        trainer=trainer
    )
    
    # 4. 设置解码器
    tokenizer.decoder = decoders.ByteLevel()
    
    print(f"✅ 训练完成 (词表大小: {tokenizer.get_vocab_size()})")

    # 5. 保存底层文件 (tokenizer.json)
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(Path(save_path) / "tokenizer.json"), pretty=True)
        print(f"💾 已保存到: {save_path}/tokenizer.json")

    # 6. 创建 MiniMindTokenizerFast 实例
    fast_tokenizer = MiniMindTokenizerFast(tokenizer_object=tokenizer)
    
    # 7. 保存配置文件（生成 tokenizer_config.json）
    fast_tokenizer.save_pretrained(save_path)
    print(f"✅ 分词器已保存到: {save_path}")
    
    return fast_tokenizer

class NovelDatasetPreparator:
    """小说数据集准备器 - 文本分割 -> tokenization -> JSONL"""
    
    def __init__(
        self,
        dataset_dir: str = "./dataset",
        pretrain_path: str = "./dataset/pretrain.jsonl",
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        tokenizer_path: Optional[str] = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.pretrain_path = Path(pretrain_path)
        
        # 初始化 tokenizer
        self.tokenizer = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            print(f"✅ 已加载 Tokenizer (词表大小: {self.tokenizer.vocab_size})")
        
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
        print(f"📚 找到 {len(novels)} 个小说文件")
        
        self.pretrain_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果tokenizer未初始化，则先训练
        if self.tokenizer is None:
            temp_files = []
            for novel_path in novels:
                text = self.load_novel_text(novel_path)
                if text is None:
                    continue
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                    f.write('\n'.join(line.strip() for line in text.split('\n') if line.strip()))
                    temp_files.append(f.name)
            
            if temp_files:
                # 使用新的函数进行训练和保存
                self.tokenizer = train_and_save_tokenizer(
                    files=temp_files,
                    save_path=str(self.pretrain_path.parent / "tokenizer"),
                    vocab_size=6400
                )
            
            for f in temp_files:
                import os
                os.unlink(f)
        
        # 编码并保存
        with open(self.pretrain_path, 'w', encoding='utf-8') as out_f:
            for novel_path in tqdm(novels, desc="处理"):
                text = self.load_novel_text(novel_path)
                if text is None:
                    continue
                text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
                
                for chunk in self.splitter.split_text(text):
                    out_f.write(json.dumps({
                        "text": chunk,
                        "token_ids": self.tokenizer.encode(chunk),
                        "source": novel_path.name,
                    }, ensure_ascii=False) + '\n')
        
        print(f"✅ 数据集已保存: {self.pretrain_path}")


class MinimindDataset(Dataset):
    """从 JSONL 文件加载已 tokenized 的数据，支持动态分词"""
    
    def __init__(self, jsonl_path: str, max_length: int = 512, tokenizer_path: Optional[str] = None):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
                            
        self.max_length = max_length
        
        # 初始化 tokenizer
        self.tokenizer = None
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            print(f"✅ 已加载 Tokenizer (词表大小: {self.tokenizer.vocab_size})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 检查是否存在 token_ids，如果不存在则使用 tokenizer 分词
        if "token_ids" not in self.data[idx]:
            if self.tokenizer is None:
                raise ValueError(f"缺少 token_ids 且未提供 tokenizer，无法处理样本 {idx}")
            
            text = self.data[idx].get("text", "")
            token_ids = self.tokenizer.encode(text)
            
            # 保存到 self.data 中，避免重复分词
            self.data[idx]["token_ids"] = token_ids
        else:
            token_ids = self.data[idx]["token_ids"]
        
        # 截断或填充
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))
        
        token_ids = torch.tensor(token_ids[:self.max_length], dtype=torch.long)
        
        # 创建 loss_mask：padding token (0) 位置的 mask 为 0，其他位置为 1
        loss_mask = (token_ids[1:] != 0).long()
        
        return token_ids[:-1], token_ids[1:], loss_mask

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--chunk_overlap", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--tokenizer_path", default=None)
    
    args = parser.parse_args()
    
    preparator = NovelDatasetPreparator(
        dataset_dir=args.dataset_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        tokenizer_path=args.tokenizer_path
    )
    preparator.prepare_dataset()
