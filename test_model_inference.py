"""
MiniMind 模型输出效果测试脚本
测试5个不同的用例，验证模型生成能力
"""
import os
import sys
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from transformers import PreTrainedTokenizerFast
from minimind import MiniMindConfig, MiniMindForCausalLM


class ModelInferenceTester:
    """模型推理测试器"""
    
    def __init__(self, 
                 model_path: str = "./output/minimind_model.pt",
                 tokenizer_path: str = "./unigram_tokenizer.json",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 max_new_tokens: int = 50):
        """
        初始化推理测试器
        
        Args:
            model_path: 模型检查点路径
            tokenizer_path: 分词器路径
            device: 运行设备 (cuda 或 cpu)
            max_new_tokens: 最大生成 token 数
        """
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        
        print(f"[INFO] 设备: {self.device}")
        print(f"[INFO] 加载分词器...")
        
        # 加载分词器
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"分词器文件不存在: {tokenizer_path}")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        print(f"✓ 分词器加载成功 (词汇表大小: {len(self.tokenizer)})")
        
        # 加载模型
        print(f"[INFO] 加载模型...")
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"✓ 模型加载成功")
    
    def _load_model(self, model_path: str) -> MiniMindForCausalLM:
        """加载模型"""
        if not os.path.exists(model_path):
            print(f"⚠️  模型文件不存在: {model_path}，将创建新模型进行演示")
            # 创建默认配置的模型用于演示
            config = MiniMindConfig(
                hidden_size=512,
                num_hidden_layers=8,
                num_attention_heads=8,
                vocab_size=6400,
                max_position_embeddings=1024,
                dropout=0.1,
                use_moe=False
            )
            model = MiniMindForCausalLM(config).to(self.device)
        else:
            # 从检查点加载
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                model_state = checkpoint["model_state"]
            else:
                model_state = checkpoint
            
            # 获取配置
            config_path = Path(model_path).parent / "config.json"
            if config_path.exists():
                config = MiniMindConfig.from_pretrained(str(config_path.parent))
            else:
                config = MiniMindConfig(vocab_size=6400)
            
            model = MiniMindForCausalLM(config).to(self.device)
            model.load_state_dict(model_state, strict=False)
        
        return model
    
    @torch.no_grad()
    def generate(self, prompt: str, temperature: float = 0.8, top_k: int = 50) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示文本
            temperature: 温度参数 (控制多样性)
            top_k: top-k 采样参数
            
        Returns:
            生成的文本
        """
        # 编码提示
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 生成
        generated_ids = input_ids.clone()
        
        for _ in range(self.max_new_tokens):
            # 前向传播
            outputs = self.model(generated_ids)
            logits = outputs.logits[:, -1, :]  # 获取最后一个token的logits
            
            # 应用温度和top-k采样
            logits = logits / temperature
            
            if top_k > 0:
                # Top-k采样
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
                logits = logits_filtered
            
            # softmax
            probs = torch.softmax(logits, dim=-1)
            
            # 采样
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # 检查是否生成了EOS token (通常是token 2)
            if next_token.item() == 2:
                break
        
        # 解码
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
    
    def run_test_cases(self) -> List[Dict]:
        """运行5个测试用例"""
        
        test_cases = [
            {
                "name": "测试用例1: 基础文本完成",
                "prompt": "今天天气",
                "description": "测试模型基础的文本续写能力"
            },
            {
                "name": "测试用例2: 中文问答",
                "prompt": "问：什么是机器学习？答：",
                "description": "测试模型的问答理解能力"
            },
            {
                "name": "测试用例3: 诗歌创作",
                "prompt": "春风吹过",
                "description": "测试模型的创意文本生成能力"
            },
            {
                "name": "测试用例4: 故事开头",
                "prompt": "从前有一个",
                "description": "测试模型的故事叙述能力"
            },
            {
                "name": "测试用例5: 常用表达",
                "prompt": "感谢您的",
                "description": "测试模型在日常表达中的表现"
            },
        ]
        
        results = []
        
        print("\n" + "="*80)
        print("🚀 开始模型输出效果测试 (5个用例)")
        print("="*80 + "\n")
        
        for idx, test_case in enumerate(test_cases, 1):
            print(f"\n{test_case['name']}")
            print(f"📝 描述: {test_case['description']}")
            print(f"🔤 输入提示: \"{test_case['prompt']}\"")
            print("-" * 80)
            
            try:
                # 生成文本
                generated = self.generate(test_case['prompt'], temperature=0.7, top_k=50)
                
                # 显示输出
                print(f"📄 生成文本:")
                print(f"  {generated}")
                
                # 统计信息
                input_tokens = len(self.tokenizer.encode(test_case['prompt']))
                output_tokens = len(self.tokenizer.encode(generated)) - input_tokens
                
                print(f"\n📊 统计信息:")
                print(f"  输入 tokens: {input_tokens}")
                print(f"  生成 tokens: {output_tokens}")
                print(f"  总 tokens: {output_tokens + input_tokens}")
                
                result = {
                    "test_number": idx,
                    "test_name": test_case['name'],
                    "prompt": test_case['prompt'],
                    "generated": generated,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "status": "✓ 成功"
                }
                
            except Exception as e:
                print(f"❌ 生成失败: {str(e)}")
                result = {
                    "test_number": idx,
                    "test_name": test_case['name'],
                    "prompt": test_case['prompt'],
                    "generated": f"ERROR: {str(e)}",
                    "status": "✗ 失败"
                }
            
            results.append(result)
            print()
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """打印测试总结"""
        print("\n" + "="*80)
        print("📋 测试总结")
        print("="*80 + "\n")
        
        successful = sum(1 for r in results if "成功" in r["status"])
        failed = len(results) - successful
        
        print(f"总测试数: {len(results)}")
        print(f"成功: {successful} ✓")
        print(f"失败: {failed} ✗")
        print(f"成功率: {(successful/len(results)*100):.1f}%\n")
        
        # 详细结果
        print("详细结果:")
        print("-" * 80)
        for result in results:
            print(f"\n[{result['test_number']}] {result['test_name']}")
            print(f"状态: {result['status']}")
            print(f"输入: \"{result['prompt']}\"")
            if "ERROR" not in result['generated']:
                print(f"输出: \"{result['generated'][:100]}{'...' if len(result['generated']) > 100 else ''}\"")
                if 'input_tokens' in result:
                    print(f"Token数: 输入={result['input_tokens']}, 生成={result['output_tokens']}")
            else:
                print(f"错误: {result['generated']}")


def main():
    parser = argparse.ArgumentParser(description="MiniMind 模型推理效果测试")
    parser.add_argument("--model_path", type=str, default="./output/minimind_model.pt",
                        help="模型检查点路径 (default: ./output/minimind_model.pt)")
    parser.add_argument("--tokenizer_path", type=str, default="./unigram_tokenizer.json",
                        help="分词器路径 (default: ./unigram_tokenizer.json)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备 (default: cuda if available else cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="最大生成 token 数 (default: 50)")
    parser.add_argument("--save_results", type=str, default=None,
                        help="保存结果到文件 (可选)")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ModelInferenceTester(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )
    
    # 运行测试
    results = tester.run_test_cases()
    
    # 打印总结
    tester.print_summary(results)
    
    # 保存结果（如果指定）
    if args.save_results:
        import json
        with open(args.save_results, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已保存到: {args.save_results}")


if __name__ == "__main__":
    main()
