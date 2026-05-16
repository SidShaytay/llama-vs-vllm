#!/usr/bin/env bash
set -euo pipefail

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"

exec toolbox run -c vllm vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
  --enable-prefix-caching \
  --kv-cache-dtype "$VLLM_KV_CACHE_DTYPE" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --no-enable-log-requests
