#!/usr/bin/env bash
set -euo pipefail

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-ROCM_ATTN}"
VLLM_MM_ENCODER_ATTN_BACKEND="${VLLM_MM_ENCODER_ATTN_BACKEND:-TRITON_ATTN}"

exec toolbox run -c vllm env VLLM_USE_TRITON_AWQ=1 VLLM_ROCM_USE_AITER=1 VLLM_DISABLE_COMPILE_CACHE=1 \
  vllm serve cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
  --host 0.0.0.0 \
  --port 8001 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
  --enable-prefix-caching \
  --kv-cache-dtype "$VLLM_KV_CACHE_DTYPE" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --attention-backend "$VLLM_ATTENTION_BACKEND" \
  --mm-encoder-attn-backend "$VLLM_MM_ENCODER_ATTN_BACKEND" \
  --no-enable-log-requests
