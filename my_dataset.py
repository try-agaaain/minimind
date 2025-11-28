import os
import json
from pathlib import Path
from typing import Union, List
import torch
from torch.utils.data import Dataset

from langchain_text_splitters import RecursiveCharacterTextSplitter 
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import Unigram, BPE
from tokenizers.trainers import UnigramTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Metaspace, Whitespace
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class MinimindDataset(Dataset):

    def __init__(self, 
            dataset_path: str,
            tokenizer_path: str,
            max_seq_len: int = 256,  # LangChain 中的 chunk_size
            char_overlap: int = 64,     # LangChain 中的 chunk_overlap
            corpus_path_list: Union[str, List[str]] = "book_corpus/*.txt",
    ) -> None:
        """初始化处理器和 LangChain 切分器。"""
        self.corpus_path_list = corpus_path_list
        self.dataset_path = Path(dataset_path)
        
        tokenizer = None

        if os.path.exists(tokenizer_path):
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        else:
            
            tokenizer = train_tokenizer(vocab_size=6400, 
                                        file_list=corpus_path_list, 
                                        algorithm="unigram", output_filename=tokenizer_path)
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # 初始化 LangChain 切分器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_seq_len,
            chunk_overlap=char_overlap,
            # 默认分隔符通常是基于换行符和空格，适用于通用文本
            separators=["\n", " ", ""], 
            length_function=len,
        )
        self.data = []
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        else:
            self.data = self.text_to_dataset()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 检查是否存在 input_ids，如果不存在则生成并保存
        if "input_ids" not in self.data[idx]:
            text = self.data[idx].get("text", "")
            token_ids = self.tokenizer.encode(text)
            
            # 截断或填充
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
            else:
                token_ids = token_ids + [0] * (self.max_seq_len - len(token_ids))
            
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
            
            # 生成 input_ids、labels、loss_mask
            input_ids = torch.concat([torch.tensor([0]), token_ids_tensor[:-1]])
            labels = token_ids_tensor
            loss_mask = (labels != 0).long()
            
            # 保存到 self.data 中（存放 torch tensor）
            self.data[idx]["input_ids"] = input_ids
            self.data[idx]["labels"] = labels
            self.data[idx]["loss_mask"] = loss_mask
        else:
            # 从 self.data 中读取
            input_ids = self.data[idx]["input_ids"]
            labels = self.data[idx]["labels"]
            loss_mask = self.data[idx]["loss_mask"]
        
        return input_ids, labels, loss_mask
    
    def get_token(self, idx):
        return self.tokenizer.tokenize(self.data[idx]["text"])
        
    def text_to_dataset(self) -> List[dict]:
        """执行切块并将所有结果流式写入单个 JSONL 文件。"""
        
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        total_chunks_count = 0
        
        # 流式写入 JSONL 文件
        all_records = []
        for path in self.corpus_path_list:
            try:
                with open(path, 'r', encoding='utf-8') as input_f:
                    raw_text = input_f.read()
                
                # 简单清理（仅标点符号统一）
                text = raw_text.replace(',', '，').replace('!', '！').replace('?', '？')
                
                chunks = self.splitter.create_documents([text])
                for doc in chunks:
                    chunk = doc.page_content.strip()
                    if chunk:
                        # 为样本添加开始和结束标识
                        chunk_with_markers = f"<s>{chunk}</s>"
                        token_ids = self.tokenizer.encode(chunk_with_markers)
                        record = {"text": chunk_with_markers}
                        all_records.append(json.dumps(record, ensure_ascii=False))
                        total_chunks_count += 1

            except Exception as e:
                print(f"警告：跳过文件 {path}，处理失败: {e}")
                continue
        # 写入 JSONL 格式: {"text": "..."}
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_records))
       
        print(f"\n--- 处理完成 ---\n总共生成 {total_chunks_count} 个切块，保存到 {self.dataset_path}")
        data = [json.loads(line) for line in all_records]
        return data

def train_tokenizer(
    vocab_size: int = 6400,
    file_list: List[str] = None,
    algorithm: str = "unigram", # <--- 新增参数
    output_filename: str = "custom_tokenizer.json", # <--- 新增参数
    bos_token: str = "<s>",
    eos_token: str = "</s>",
    unk_token: str = "<unk>",
) -> PreTrainedTokenizer:
    """
    根据指定的算法训练分词器。

    :param vocab_size: 词汇表大小。
    :param file_list: 训练文件列表。
    :param algorithm: 分词算法，支持 "unigram" 和 "bpe"。
    :param output_filename: 保存分词器文件的名称。
    :return: 训练好的 tokenizers.Tokenizer 对象。
    """
    
    # 定义特殊 tokens
    special_tokens = [unk_token, bos_token, eos_token]
    
    if algorithm == "unigram":
        # 1. 初始化 Unigram 模型
        tokenizer = Tokenizer(Unigram())
        
        # 2. 设置预处理器（SentencePiece 风格）
        # Unigram 通常使用 Metaspace
        tokenizer.pre_tokenizer = Metaspace(replacement=" ", prepend_scheme="always")
        
        # 3. 配置训练器
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token=unk_token,
        )
        
    elif algorithm == "bpe":
        # 1. 初始化 BPE 模型
        # unk_token 需要在模型初始化时指定
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        
        # BPE 通常使用 Whitespace 或 ByteLevel pre_tokenizer
        # 为了兼容性，使用 Whitespace 进行基于单词的分词
        tokenizer.pre_tokenizer = Whitespace() 

        # 3. 配置训练器
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            # BPE 训练器可能还需要其他参数，例如 min_frequency=2
        )
    else:
        raise ValueError(f"不支持的分词算法: {algorithm}。请选择 'unigram' 或 'bpe'。")

    valid_files = []
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read()  # 尝试读取以检查编码
            valid_files.append(file_path)
        except UnicodeDecodeError:
            print(f"跳过非 UTF-8 文件: {file_path}")
            
    tokenizer.train(valid_files, trainer=trainer)

    # 5. 保存
    tokenizer.save(output_filename)
    lm_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, # 传入训练好的 tokenizers.Tokenizer 对象
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        # 其他可能需要的参数，例如 padding_side, truncation_side 等
    )
    return lm_tokenizer

if __name__ == "__main__":
    # --- 演示用例 ---
    # 假设待处理的文本文件位于当前目录下的 'dataset' 文件夹中
    dataset_dir = Path(__file__).parent / "dataset"
    file_list = list(map(str, dataset_dir.rglob("*.txt")))
    tokenizer_alg = "unigram"
    tokenizer_file = f"{tokenizer_alg}_tokenizer.json"
    if os.path.exists(tokenizer_file):
        lm_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    else:
        lm_tokenizer = train_tokenizer(vocab_size=6400, 
                                       file_list=file_list, 
                                       algorithm=tokenizer_alg, output_filename=tokenizer_file)
    # 实例化并运行处理器
    dataset = MinimindDataset(
        tokenizer = lm_tokenizer,
        dataset_path="all_processed_chunks.jsonl", 
        max_seq_len=1024,
        char_overlap=256,
        corpus_path_list=file_list,
    )
    
    print(dataset[0])
    print(dataset.get_token(0))
    print(len(dataset))
    