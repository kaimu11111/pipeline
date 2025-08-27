
# baseline
启动vllm example：
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server   --model Qwen/QwQ-32B   --tensor-parallel-size 4   --dtype half   --max-model-len 32768   --reasoning-parser qwen3   --port 30000   --trust-remote-code
```

运行程序 API：
```bash
python3 main_multi_task.py KernelBench/level2   --first_n 100   --gpu "Quadro RTX 6000"   --server_type openai   --model_name o1   --device 7   --round 1
```
# pipeline
