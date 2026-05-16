#!/usr/bin/env bash
set -euo pipefail

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
LLAMACPP_TOOLBOX="${LLAMACPP_TOOLBOX:-llama-rocm-7.2.3}"
LLAMACPP_SERVER="${LLAMACPP_SERVER:-llama-server}"
LLAMACPP_MODEL="${LLAMACPP_MODEL:-}"
LLAMACPP_PARALLEL="${LLAMACPP_PARALLEL:-16}"
LLAMACPP_CTX_SIZE="${LLAMACPP_CTX_SIZE:-262144}"
LLAMACPP_BATCH_SIZE="${LLAMACPP_BATCH_SIZE:-8192}"
LLAMACPP_UBATCH_SIZE="${LLAMACPP_UBATCH_SIZE:-1024}"
LLAMACPP_CACHE_RAM="${LLAMACPP_CACHE_RAM:-8192}"

if [[ -z "$LLAMACPP_MODEL" ]]; then
  shopt -s nullglob
  candidates=("$HF_CACHE"/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/*/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf)
  shopt -u nullglob
  if (( ${#candidates[@]} == 0 )); then
    cat >&2 <<'EOF'
Could not find Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf in the Hugging Face cache.
Download it first:
  hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
Or set LLAMACPP_MODEL=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf.
EOF
    exit 1
  fi
  LLAMACPP_MODEL="${candidates[0]}"
elif [[ ! -f "$LLAMACPP_MODEL" ]]; then
  cat >&2 <<EOF
Could not find llama.cpp model at:
  $LLAMACPP_MODEL
Set LLAMACPP_MODEL=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf.
EOF
  exit 1
fi

exec toolbox run -c "$LLAMACPP_TOOLBOX" "$LLAMACPP_SERVER" \
  --host 0.0.0.0 \
  --port 18080 \
  --model "$LLAMACPP_MODEL" \
  --alias qwen3.6-35b-a3b \
  --parallel "$LLAMACPP_PARALLEL" \
  --ctx-size "$LLAMACPP_CTX_SIZE" \
  --batch-size "$LLAMACPP_BATCH_SIZE" \
  --ubatch-size "$LLAMACPP_UBATCH_SIZE" \
  --cont-batching \
  --kv-unified \
  --cache-idle-slots \
  --cache-ram "$LLAMACPP_CACHE_RAM" \
  --n-gpu-layers 999 \
  --flash-attn on \
  --mlock \
  --no-warmup \
  --metrics
