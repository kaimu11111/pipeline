#!/bin/bash

# 第一步：启动 vLLM server
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m vllm.entrypoints.openai.api_server \
  --model cognition-ai/Kevin-32B \
  --tensor-parallel-size 4 \
  --dtype half \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --port 8000 \
  --trust-remote-code &

# 保存 server 的进程号
SERVER_PID=$!
# 第一次启动可能要下载模型，时间较长
# 给 server 一点时间启动（可以根据实际情况调节）
sleep 30

# 第二步：运行 main_multi_task.py
python3 main_multi_task.py KernelBench/test2 \
  --first_n 20 \
  --gpu "Quadro RTX 6000" \
  --server_type vllm \
  --server_address localhost \
  --server_port 8000 \
  --model_name cognition-ai/Kevin-32B \
  --device 7 \  # 测试kernel使用的GPU
  --round 10 \
  --max_tokens 16384

# 脚本结束后自动杀掉 vLLM server
kill $SERVER_PID
