# MiniMind vLLM 部署指南

本指南详细介绍如何将使用 `my_dataset.py` 和 `my_train.py` 训练的 MiniMind 模型导出并通过 vLLM 部署为高性能推理服务。

## 📋 目录

1. [概述](#概述)
2. [前置条件](#前置条件)
3. [训练流程回顾](#训练流程回顾)
4. [模型导出](#模型导出)
5. [vLLM 服务部署](#vllm-服务部署)
6. [API 使用示例](#api-使用示例)
7. [最佳实践](#最佳实践)
8. [故障排除](#故障排除)

---

## 概述

[vLLM](https://github.com/vllm-project/vllm) 是一个高性能的大语言模型推理引擎，支持：

- **PagedAttention**: 高效的 KV Cache 管理
- **连续批处理**: 动态批处理请求以提高吞吐量
- **OpenAI 兼容 API**: 可直接替代 OpenAI API 使用
- **多 GPU 支持**: 张量并行和流水线并行

本指南将帮助你：
1. 将训练好的 MiniMind 模型导出为 HuggingFace 格式
2. 使用 vLLM 部署模型
3. 通过 REST API 对外提供服务

---

## 前置条件

### 硬件要求

- **GPU**: NVIDIA GPU，建议显存 >= 8GB
- **CUDA**: 11.8 或更高版本
- **内存**: >= 16GB RAM

### 软件依赖

```bash
# 基础依赖 (训练时已安装)
pip install torch transformers

# vLLM 安装
pip install vllm

# 可选: OpenAI Python SDK (用于 API 调用)
pip install openai
```

### 版本要求

| 软件 | 最低版本 | 推荐版本 |
|------|---------|---------|
| Python | 3.8 | 3.10+ |
| PyTorch | 2.0 | 2.1+ |
| vLLM | 0.4.0 | 最新版 |
| CUDA | 11.8 | 12.1 |

---

## 训练流程回顾

使用 `my_dataset.py` 和 `my_train.py` 完成模型训练：

### 1. 数据准备

```python
# my_dataset.py 会自动处理以下步骤:
# 1. 训练 tokenizer (如果不存在)
# 2. 将文本切分为固定长度的 chunks
# 3. 保存为 JSONL 格式
```

### 2. 模型训练

```bash
# 使用 DDP 进行分布式训练
torchrun --nproc_per_node=4 my_train.py \
    --hidden_size 512 \
    --num_layers 8 \
    --num_heads 8 \
    --epochs 10 \
    --batch_size 8 \
    --output_dir ./output
```

### 3. 训练完成后的文件结构

```
output/
├── minimind_model.pt       # 模型权重 (checkpoint)
├── config.json             # 模型配置
unigram_tokenizer.json      # Tokenizer 文件
```

---

## 模型导出

vLLM 需要 HuggingFace 格式的模型。使用我们提供的导出脚本：

### 基本用法

```bash
python export_for_vllm.py \
    --model_path ./output/minimind_model.pt \
    --tokenizer_path ./unigram_tokenizer.json \
    --output_dir ./minimind_hf_model
```

### 导出脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 训练好的模型检查点路径 | `./output/minimind_model.pt` |
| `--tokenizer_path` | Tokenizer 文件路径 | `./unigram_tokenizer.json` |
| `--output_dir` | 输出目录 | `./minimind_hf_model` |
| `--hidden_size` | 覆盖隐藏层维度 (可选) | - |
| `--num_layers` | 覆盖层数 (可选) | - |
| `--max_seq_len` | 覆盖最大序列长度 (可选) | - |

### 导出后的文件结构

```
minimind_hf_model/
├── config.json             # 模型配置
├── model.safetensors       # 模型权重 (SafeTensors 格式)
├── tokenizer.json          # Tokenizer
├── tokenizer_config.json   # Tokenizer 配置
├── special_tokens_map.json # 特殊 token 映射
└── README.md               # 模型卡片
```

---

## vLLM 服务部署

### 方式一：使用提供的脚本 (推荐)

```bash
# 启动服务器
python serve_vllm.py serve \
    --model_path ./minimind_hf_model \
    --host 0.0.0.0 \
    --port 8000

# 或使用简写
python serve_vllm.py --model_path ./minimind_hf_model
```

### 方式二：直接使用 vLLM 命令

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./minimind_hf_model \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name minimind \
    --trust-remote-code
```

### 服务配置选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 服务器地址 | `0.0.0.0` |
| `--port` | 服务端口 | `8000` |
| `--max_model_len` | 最大序列长度 | 模型配置值 |
| `--gpu_memory_utilization` | GPU 显存利用率 | `0.9` |
| `--tensor_parallel_size` | 张量并行大小 | `1` |
| `--dtype` | 数据类型 | `auto` |

### 多 GPU 部署

```bash
# 使用 2 个 GPU 进行张量并行
python serve_vllm.py serve \
    --model_path ./minimind_hf_model \
    --tensor_parallel_size 2
```

---

## API 使用示例

### OpenAI 兼容 API

服务启动后，提供以下 API 端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/completions` | POST | 文本补全 |
| `/v1/chat/completions` | POST | 对话补全 |
| `/v1/models` | GET | 获取模型列表 |
| `/health` | GET | 健康检查 |

### 使用 curl

```bash
# 文本补全
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "minimind",
        "prompt": "从前有一个",
        "max_tokens": 100,
        "temperature": 0.8
    }'

# 对话补全
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "minimind",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ],
        "max_tokens": 100
    }'
```

### 使用 Python OpenAI SDK

```python
from openai import OpenAI

# 创建客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM 不需要 API key
)

# 文本补全
response = client.completions.create(
    model="minimind",
    prompt="春风吹过",
    max_tokens=100,
    temperature=0.8
)
print(response.choices[0].text)

# 对话补全
response = client.chat.completions.create(
    model="minimind",
    messages=[
        {"role": "user", "content": "什么是机器学习？"}
    ],
    max_tokens=200
)
print(response.choices[0].message.content)
```

### 使用 vLLM Python API (离线推理)

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="./minimind_hf_model",
    trust_remote_code=True
)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# 批量生成
prompts = ["你好，", "从前有一个", "春风吹过"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 50)
```

---

## 最佳实践

### 1. 性能优化

#### GPU 显存管理

```bash
# 根据 GPU 显存调整利用率
# 较小模型: 可以降低以留出空间给其他任务
python serve_vllm.py serve \
    --gpu_memory_utilization 0.7

# 充分利用显存以最大化吞吐量
python serve_vllm.py serve \
    --gpu_memory_utilization 0.95
```

#### 数据类型选择

```bash
# 自动选择 (推荐)
--dtype auto

# 强制使用 float16 (节省显存)
--dtype float16

# 使用 bfloat16 (A100/H100 等新 GPU)
--dtype bfloat16
```

### 2. 生产部署建议

#### 使用 Docker

```dockerfile
FROM vllm/vllm-openai:latest

COPY ./minimind_hf_model /model

EXPOSE 8000

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/model", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--trust-remote-code"]
```

#### 使用 systemd 管理服务

```ini
# /etc/systemd/system/minimind-vllm.service
[Unit]
Description=MiniMind vLLM Service
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/minimind
ExecStart=/usr/bin/python serve_vllm.py serve --model_path ./minimind_hf_model
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 3. 负载均衡

对于高并发场景，可以部署多个 vLLM 实例并使用负载均衡：

```bash
# 实例 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python serve_vllm.py serve --port 8001

# 实例 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python serve_vllm.py serve --port 8002
```

使用 Nginx 进行负载均衡：

```nginx
upstream vllm_backend {
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 8000;
    location / {
        proxy_pass http://vllm_backend;
    }
}
```

---

## 故障排除

### 常见问题

#### 1. CUDA 内存不足

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案**:
- 降低 `--gpu_memory_utilization` 值
- 减小 `--max_model_len`
- 使用 `--dtype float16` 减少显存占用

#### 2. 模型加载失败

```
ValueError: Cannot find model architecture for MiniMindForCausalLM
```

**解决方案**:
- 确保使用 `--trust-remote-code` 参数
- 检查模型目录中是否包含 `minimind.py` 或配置正确

#### 3. Tokenizer 错误

```
ValueError: tokenizer_class is not defined
```

**解决方案**:
- 确保 `tokenizer_config.json` 中包含正确的配置
- 重新运行 `export_for_vllm.py` 导出模型

#### 4. 导入错误

```
ModuleNotFoundError: No module named 'minimind'
```

**解决方案**:
- 确保在正确的目录下运行命令
- 将项目路径添加到 `PYTHONPATH`:
  ```bash
  export PYTHONPATH=/path/to/minimind:$PYTHONPATH
  ```

### 日志调试

```bash
# 启用详细日志
VLLM_LOGGING_LEVEL=DEBUG python serve_vllm.py serve --model_path ./minimind_hf_model
```

### 获取帮助

- vLLM 官方文档: https://docs.vllm.ai/
- vLLM GitHub: https://github.com/vllm-project/vllm
- MiniMind 问题反馈: 在项目 Issues 中提交

---

## 完整工作流程

```bash
# 1. 准备数据和训练模型
torchrun --nproc_per_node=4 my_train.py \
    --epochs 10 \
    --output_dir ./output

# 2. 导出模型为 HuggingFace 格式
python export_for_vllm.py \
    --model_path ./output/minimind_model.pt \
    --tokenizer_path ./unigram_tokenizer.json \
    --output_dir ./minimind_hf_model

# 3. 启动 vLLM 服务
python serve_vllm.py serve \
    --model_path ./minimind_hf_model \
    --port 8000

# 4. 测试 API
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "minimind", "prompt": "你好", "max_tokens": 50}'
```

---

## 附录

### A. 配置参考

#### vLLM 服务器完整参数

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./minimind_hf_model \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name minimind \
    --trust-remote-code \
    --dtype auto \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --disable-log-requests
```

### B. 性能基准测试

```python
from vllm import LLM, SamplingParams
import time

llm = LLM(model="./minimind_hf_model")
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

# 预热
llm.generate(["测试"], sampling_params)

# 基准测试
prompts = ["你好"] * 100
start = time.time()
outputs = llm.generate(prompts, sampling_params)
end = time.time()

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f"Total time: {end - start:.2f}s")
print(f"Throughput: {total_tokens / (end - start):.2f} tokens/s")
```
