"""
MiniMind 模型导出脚本 - 将训练好的模型导出为 HuggingFace 格式以供 vLLM 使用

使用方法:
    python export_for_vllm.py \
        --model_path ./output/minimind_model.pt \
        --tokenizer_path ./unigram_tokenizer.json \
        --output_dir ./minimind_hf_model

导出后可以使用 vLLM 加载:
    from vllm import LLM
    llm = LLM(model="./minimind_hf_model")
"""
import os
import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import PreTrainedTokenizerFast

from minimind import MiniMindConfig, MiniMindForCausalLM


def export_model_for_vllm(
    model_path: str,
    tokenizer_path: str,
    output_dir: str,
    config_overrides: Optional[dict] = None,
) -> None:
    """
    将 MiniMind 模型导出为 HuggingFace 格式，以便 vLLM 加载。
    
    Args:
        model_path: 训练好的模型检查点路径 (.pt 文件)
        tokenizer_path: tokenizer 文件路径 (.json 文件)
        output_dir: 输出目录路径
        config_overrides: 可选的配置覆盖参数
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🚀 MiniMind 模型导出工具 (vLLM 格式)")
    print("=" * 60)
    
    # 1. 加载 tokenizer
    print("\n[1/4] 加载 Tokenizer...")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer 文件不存在: {tokenizer_path}")
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # 设置必要的特殊 token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<s>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.unk_token is None:
        tokenizer.unk_token = "<unk>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  ✅ Tokenizer 加载成功")
    print(f"     词汇表大小: {len(tokenizer)}")
    print(f"     BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    print(f"     EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    
    # 2. 加载模型配置和权重
    print("\n[2/4] 加载模型...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 尝试从模型目录加载配置
    model_dir = Path(model_path).parent
    config_path = model_dir / "config.json"
    
    if config_path.exists():
        config = MiniMindConfig.from_pretrained(str(model_dir))
        print(f"  ✅ 从 {config_path} 加载配置")
    else:
        # 使用默认配置
        config = MiniMindConfig(vocab_size=len(tokenizer))
        print("  ⚠️  未找到配置文件，使用默认配置")
    
    # 应用配置覆盖
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"  📝 配置覆盖: {key} = {value}")
    
    # 确保 vocab_size 与 tokenizer 一致
    config.vocab_size = len(tokenizer)
    
    # 设置必要的 token id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    
    # 创建模型并加载权重
    model = MiniMindForCausalLM(config)
    
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_state = checkpoint["model_state"]
        epoch = checkpoint.get("epoch", "unknown")
        print(f"  📦 检查点 epoch: {epoch}")
    else:
        model_state = checkpoint
    
    model.load_state_dict(model_state, strict=False)
    print(f"  ✅ 模型权重加载成功")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  📊 隐藏层维度: {config.hidden_size}")
    print(f"  📊 层数: {config.num_hidden_layers}")
    print(f"  📊 注意力头数: {config.num_attention_heads}")
    print(f"  📊 最大序列长度: {config.max_position_embeddings}")
    
    # 3. 保存为 HuggingFace 格式
    print("\n[3/4] 导出模型...")
    
    # 保存模型权重和配置
    model.save_pretrained(str(output_path))
    print(f"  ✅ 模型已保存到: {output_path}")
    
    # 4. 保存 tokenizer
    print("\n[4/4] 保存 Tokenizer...")
    tokenizer.save_pretrained(str(output_path))
    print(f"  ✅ Tokenizer 已保存到: {output_path}")
    
    # 创建模型卡片
    model_card = f"""---
tags:
- minimind
- text-generation
- causal-lm
language:
- zh
library_name: transformers
---

# MiniMind Model

这是使用 MiniMind 训练的语言模型，已导出为 HuggingFace 格式。

## 模型信息

- **模型类型**: MiniMind (Causal Language Model)
- **参数量**: {total_params:,} ({total_params/1e6:.2f}M)
- **隐藏层维度**: {config.hidden_size}
- **层数**: {config.num_hidden_layers}
- **注意力头数**: {config.num_attention_heads}
- **最大序列长度**: {config.max_position_embeddings}
- **词汇表大小**: {config.vocab_size}
- **使用 MoE**: {config.use_moe}

## 使用方法

### 使用 Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")

inputs = tokenizer("你好，", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 使用 vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="{output_dir}")
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(["你好，"], sampling_params)

# 注意: 确保 outputs 不为空再访问
if outputs and outputs[0].outputs:
    print(outputs[0].outputs[0].text)
```

### 启动 OpenAI 兼容 API 服务

```bash
python -m vllm.entrypoints.openai.api_server \\
    --model {output_dir} \\
    --host 0.0.0.0 \\
    --port 8000
```
"""
    
    with open(output_path / "README.md", "w", encoding="utf-8") as f:
        f.write(model_card)
    print(f"  ✅ 模型卡片已保存到: {output_path / 'README.md'}")
    
    print("\n" + "=" * 60)
    print("✨ 导出完成!")
    print("=" * 60)
    print(f"\n输出目录: {output_path}")
    print("\n导出的文件:")
    for file in output_path.iterdir():
        size = file.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.2f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"  - {file.name}: {size_str}")
    
    print("\n📖 接下来可以:")
    print("  1. 使用 vLLM 加载模型: python serve_vllm.py --model_path", output_dir)
    print("  2. 查看部署文档: docs/vllm_deployment.md")


def main():
    parser = argparse.ArgumentParser(
        description="将 MiniMind 模型导出为 HuggingFace 格式以供 vLLM 使用"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/minimind_model.pt",
        help="训练好的模型检查点路径 (default: ./output/minimind_model.pt)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./unigram_tokenizer.json",
        help="Tokenizer 文件路径 (default: ./unigram_tokenizer.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./minimind_hf_model",
        help="输出目录路径 (default: ./minimind_hf_model)"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="覆盖模型隐藏层维度 (可选)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="覆盖模型层数 (可选)"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="覆盖最大序列长度 (可选)"
    )
    
    args = parser.parse_args()
    
    # 构建配置覆盖
    config_overrides = {}
    if args.hidden_size is not None:
        config_overrides["hidden_size"] = args.hidden_size
    if args.num_layers is not None:
        config_overrides["num_hidden_layers"] = args.num_layers
    if args.max_seq_len is not None:
        config_overrides["max_position_embeddings"] = args.max_seq_len
    
    export_model_for_vllm(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        config_overrides=config_overrides if config_overrides else None,
    )


if __name__ == "__main__":
    main()
