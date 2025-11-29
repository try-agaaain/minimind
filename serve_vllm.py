"""
MiniMind vLLM 服务脚本 - 使用 vLLM 部署 MiniMind 模型

使用方法:
    # 基础启动
    python serve_vllm.py --model_path ./minimind_hf_model

    # 自定义配置
    python serve_vllm.py \
        --model_path ./minimind_hf_model \
        --host 0.0.0.0 \
        --port 8000 \
        --max_model_len 2048

启动后可以使用 OpenAI 兼容的 API:
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "minimind", "prompt": "你好", "max_tokens": 100}'
"""
import argparse
import os
import sys
from pathlib import Path


def check_vllm_installed():
    """检查 vLLM 是否已安装"""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def start_vllm_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_model_len: int = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    trust_remote_code: bool = True,
    served_model_name: str = "minimind",
):
    """
    启动 vLLM OpenAI 兼容 API 服务器

    Args:
        model_path: HuggingFace 格式的模型路径
        host: 服务器主机地址
        port: 服务器端口
        max_model_len: 最大模型序列长度
        gpu_memory_utilization: GPU 显存利用率
        tensor_parallel_size: 张量并行大小 (多 GPU)
        dtype: 数据类型 (auto, float16, bfloat16)
        trust_remote_code: 是否信任远程代码
        served_model_name: 服务的模型名称
    """
    if not check_vllm_installed():
        print("❌ 错误: vLLM 未安装")
        print("\n请先安装 vLLM:")
        print("  pip install vllm")
        print("\n或者使用 conda:")
        print("  conda install -c conda-forge vllm")
        sys.exit(1)

    # 验证模型路径
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ 错误: 模型路径不存在: {model_path}")
        print("\n请先使用 export_for_vllm.py 导出模型:")
        print("  python export_for_vllm.py --model_path ./output/minimind_model.pt --output_dir ./minimind_hf_model")
        sys.exit(1)

    # 检查必要的文件
    required_files = ["config.json", "model.safetensors"]
    missing_files = [f for f in required_files if not (model_path / f).exists()]
    
    # 如果没有 model.safetensors，检查是否有 pytorch_model.bin
    if "model.safetensors" in missing_files:
        if (model_path / "pytorch_model.bin").exists():
            missing_files.remove("model.safetensors")
    
    if missing_files:
        print(f"❌ 错误: 模型目录缺少必要文件: {missing_files}")
        print("\n请确保模型已正确导出为 HuggingFace 格式")
        sys.exit(1)

    print("=" * 60)
    print("🚀 MiniMind vLLM 服务器")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  模型路径: {model_path.absolute()}")
    print(f"  服务地址: http://{host}:{port}")
    print(f"  模型名称: {served_model_name}")
    print(f"  GPU 显存利用率: {gpu_memory_utilization}")
    print(f"  张量并行大小: {tensor_parallel_size}")
    print(f"  数据类型: {dtype}")
    if max_model_len:
        print(f"  最大序列长度: {max_model_len}")
    print()

    # 构建启动命令
    cmd_args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path.absolute()),
        "--host", host,
        "--port", str(port),
        "--served-model-name", served_model_name,
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--dtype", dtype,
    ]

    if max_model_len:
        cmd_args.extend(["--max-model-len", str(max_model_len)])

    if trust_remote_code:
        cmd_args.append("--trust-remote-code")

    print("📋 启动命令:")
    print("  " + " ".join(cmd_args))
    print()

    print("=" * 60)
    print("🌐 API 端点:")
    print("=" * 60)
    print(f"\n  文本生成 (Completions):")
    print(f"    POST http://{host}:{port}/v1/completions")
    print(f"\n  对话生成 (Chat):")
    print(f"    POST http://{host}:{port}/v1/chat/completions")
    print(f"\n  模型列表:")
    print(f"    GET http://{host}:{port}/v1/models")
    print(f"\n  健康检查:")
    print(f"    GET http://{host}:{port}/health")
    print()

    print("=" * 60)
    print("📝 使用示例:")
    print("=" * 60)
    print(f"""
# 使用 curl 测试文本生成
curl http://{host}:{port}/v1/completions \\
    -H "Content-Type: application/json" \\
    -d '{{
        "model": "{served_model_name}",
        "prompt": "从前有一个",
        "max_tokens": 100,
        "temperature": 0.8
    }}'

# 使用 Python OpenAI SDK
from openai import OpenAI

client = OpenAI(
    base_url="http://{host}:{port}/v1",
    api_key="not-needed"
)

response = client.completions.create(
    model="{served_model_name}",
    prompt="你好，",
    max_tokens=100
)
print(response.choices[0].text)
""")

    print("=" * 60)
    print("⏳ 正在启动服务器...")
    print("=" * 60)
    print()

    # 启动服务器
    os.execv(sys.executable, cmd_args)


def run_offline_inference(
    model_path: str,
    prompts: list,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
):
    """
    运行离线推理 (不启动服务器)

    Args:
        model_path: HuggingFace 格式的模型路径
        prompts: 输入提示列表
        max_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: Top-p 采样参数
    """
    if not check_vllm_installed():
        print("❌ 错误: vLLM 未安装")
        sys.exit(1)

    from vllm import LLM, SamplingParams

    print("=" * 60)
    print("🚀 MiniMind vLLM 离线推理")
    print("=" * 60)

    print(f"\n加载模型: {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    print(f"\n生成参数:")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  max_tokens: {max_tokens}")

    print(f"\n正在生成...")
    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "=" * 60)
    print("📝 生成结果:")
    print("=" * 60)

    for i, output in enumerate(outputs):
        print(f"\n[{i+1}] 输入: {output.prompt}")
        print(f"    输出: {output.outputs[0].text}")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="MiniMind vLLM 服务脚本 - 部署模型为 OpenAI 兼容 API"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # serve 命令
    serve_parser = subparsers.add_parser("serve", help="启动 API 服务器")
    serve_parser.add_argument(
        "--model_path",
        type=str,
        default="./minimind_hf_model",
        help="HuggingFace 格式的模型路径 (default: ./minimind_hf_model)"
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器主机地址 (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (default: 8000)"
    )
    serve_parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="最大模型序列长度 (可选，默认使用模型配置)"
    )
    serve_parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU 显存利用率 (default: 0.9)"
    )
    serve_parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="张量并行大小，用于多 GPU (default: 1)"
    )
    serve_parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="数据类型 (default: auto)"
    )
    serve_parser.add_argument(
        "--served_model_name",
        type=str,
        default="minimind",
        help="服务的模型名称 (default: minimind)"
    )

    # infer 命令
    infer_parser = subparsers.add_parser("infer", help="运行离线推理")
    infer_parser.add_argument(
        "--model_path",
        type=str,
        default="./minimind_hf_model",
        help="HuggingFace 格式的模型路径 (default: ./minimind_hf_model)"
    )
    infer_parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["你好，", "从前有一个"],
        help="输入提示列表"
    )
    infer_parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="最大生成 token 数 (default: 100)"
    )
    infer_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="温度参数 (default: 0.8)"
    )
    infer_parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p 采样参数 (default: 0.95)"
    )

    args = parser.parse_args()

    # 如果没有指定命令，默认为 serve 并使用默认参数
    if args.command is None:
        # 使用默认参数启动 serve
        start_vllm_server(
            model_path="./minimind_hf_model",
            host="0.0.0.0",
            port=8000,
            max_model_len=None,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
            dtype="auto",
            served_model_name="minimind",
        )
        return

    if args.command == "serve":
        start_vllm_server(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            served_model_name=args.served_model_name,
        )
    elif args.command == "infer":
        run_offline_inference(
            model_path=args.model_path,
            prompts=args.prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
