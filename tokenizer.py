"""
MiniMind Tokenizer 实现 - 字符级 Tokenizer
精简实现，专注于字符级编码
"""
import os
import pickle
from typing import List, Optional
from pathlib import Path


class SimpleCharTokenizer:
    """
    简化版的字符级 Tokenizer - 不继承 PreTrainedTokenizer
    用于快速原型和测试
    """
    
    def __init__(self, vocab_size: int = 6400):
        self.vocab_size = vocab_size
        self._token_to_id = {}
        self._id_to_token = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """构建词表"""
        # 特殊 tokens
        special = ["<pad>", "<unk>", "<bos>", "<eos>"]
        for token in special:
            idx = len(self._token_to_id)
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
        
        # ASCII 字符
        for i in range(128):
            if len(self._token_to_id) >= self.vocab_size:
                break
            char = chr(i)
            if char not in self._token_to_id:
                idx = len(self._token_to_id)
                self._token_to_id[char] = idx
                self._id_to_token[idx] = char
        
        # 中文字符
        common_chinese = "的一是在了不和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产样配对面后因为要下看天到能好站提新法月话高自二十三多久有年没是去两都除向知理让通过自己生活第一次把来说得上所产出发人中了个大年要以为主可成对从其他方或等等重要信息"
        
        for char in common_chinese:
            if len(self._token_to_id) >= self.vocab_size:
                break
            if char not in self._token_to_id:
                idx = len(self._token_to_id)
                self._token_to_id[char] = idx
                self._id_to_token[idx] = char
        
        # CJK 字符填充
        if len(self._token_to_id) < self.vocab_size:
            for codepoint in range(0x4E00, 0x9FFF):
                if len(self._token_to_id) >= self.vocab_size:
                    break
                char = chr(codepoint)
                if char not in self._token_to_id:
                    idx = len(self._token_to_id)
                    self._token_to_id[char] = idx
                    self._id_to_token[idx] = char
    
    def encode(self, text: str) -> List[int]:
        """编码文本为 token IDs"""
        return [self._token_to_id.get(char, self._token_to_id.get("<unk>", 0)) for char in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """解码 token IDs 为文本"""
        return "".join(
            self._id_to_token.get(tid, "<unk>") 
            for tid in token_ids 
            if not self._id_to_token.get(tid, "").startswith("<")
        )
    
    def get_vocab_size(self) -> int:
        """获取词表大小"""
        return len(self._token_to_id)
    
    def save(self, path: str):
        """保存 tokenizer"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "token_to_id": self._token_to_id,
                "id_to_token": self._id_to_token,
                "vocab_size": self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """加载 tokenizer"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer._token_to_id = data["token_to_id"]
        tokenizer._id_to_token = data["id_to_token"]
        return tokenizer


if __name__ == "__main__":
    print("=" * 50)
    print("SimpleCharTokenizer 演示")
    print("=" * 50)
    
    tokenizer = SimpleCharTokenizer(vocab_size=6400)
    text = "你好，这是一个测试文本。MiniMind 是一个小语言模型。"
    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)
    
    print(f"原文本: {text}")
    print(f"Token IDs: {token_ids[:20]}... (总共 {len(token_ids)} 个tokens)")
    print(f"解码后: {decoded_text}")
    print(f"词表大小: {tokenizer.get_vocab_size()}")
    
    tokenizer.save("./minimind_tokenizer.pkl")
    loaded = SimpleCharTokenizer.load("./minimind_tokenizer.pkl")
    print(f"\n✅ Tokenizer 已保存和加载，词表大小: {loaded.get_vocab_size()}")
