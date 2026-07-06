#!/usr/bin/env bash
# CHASM LoRA 微调 (Table 3)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-0}"
echo "Using GPU $GPU"

# 阶段1：纯文本 QLoRA（3090 24GB 可跑，约 2-4 小时）
env CUDA_VISIBLE_DEVICES="$GPU" python -m src.training.finetune_lora \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --max-images 0 \
  --output-dir outputs/finetune

# 阶段2：用微调权重在 test 上评估
env CUDA_VISIBLE_DEVICES="$GPU" python -m src.models.local_inference \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --lora-path outputs/finetune/lora_adapter \
  --split test --max-samples 1000 \
  --load-4bit --max-images 1 \
  --output-dir outputs/finetune/inference
