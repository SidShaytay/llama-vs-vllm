#!/usr/bin/env bash
set -euo pipefail

VLLM_TOOLBOX="${VLLM_TOOLBOX:-vllm}"
VLLM_IMAGE="${VLLM_IMAGE:-docker.io/kyuz0/vllm-therock-gfx1151:stable}"
LLAMACPP_TOOLBOX="${LLAMACPP_TOOLBOX:-llama-rocm-7.2.3}"
LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2.3}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

toolbox_exists() {
  toolbox run -c "$1" true >/dev/null 2>&1
}

create_vllm_toolbox() {
  if toolbox_exists "$VLLM_TOOLBOX"; then
    echo "Toolbox $VLLM_TOOLBOX already exists"
    return
  fi

  echo "Pulling $VLLM_IMAGE"
  podman image pull "$VLLM_IMAGE"

  echo "Creating toolbox $VLLM_TOOLBOX"
  toolbox create --image "$VLLM_IMAGE" "$VLLM_TOOLBOX"
}

create_llamacpp_toolbox() {
  if toolbox_exists "$LLAMACPP_TOOLBOX"; then
    echo "Toolbox $LLAMACPP_TOOLBOX already exists"
    return
  fi

  echo "Pulling $LLAMACPP_IMAGE"
  podman image pull "$LLAMACPP_IMAGE"

  echo "Creating toolbox $LLAMACPP_TOOLBOX"
  toolbox create --image "$LLAMACPP_IMAGE" "$LLAMACPP_TOOLBOX"
}

verify_toolboxes() {
  echo "Checking vLLM"
  toolbox run -c "$VLLM_TOOLBOX" vllm --version

  echo "Checking llama.cpp GPU access"
  toolbox run -c "$LLAMACPP_TOOLBOX" llama-cli --list-devices

  echo "Checking llama-server"
  toolbox run -c "$LLAMACPP_TOOLBOX" llama-server --version
}

download_hf() {
  local toolbox_name="$1"
  local repo="$2"
  local file="${3:-}"
  local env_args=("HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}")

  if [[ -n "${HF_HOME:-}" ]]; then
    env_args+=("HF_HOME=$HF_HOME")
  fi

  echo "Downloading $repo ${file:+$file }via toolbox $toolbox_name"
  toolbox run -c "$toolbox_name" \
    env "${env_args[@]}" \
    bash -lc '
      repo="$1"
      file="${2:-}"
      if ! command -v hf >/dev/null 2>&1; then
        echo "Missing required command in toolbox: hf" >&2
        exit 1
      fi

      if [[ -n "$file" ]]; then
        hf download "$repo" "$file"
      else
        hf download "$repo"
      fi
    ' setup-requirements "$repo" "$file"
}

download_models() {
  if [[ "$DOWNLOAD_MODELS" == "0" ]]; then
    echo "Skipping model downloads because DOWNLOAD_MODELS=0"
    return
  fi

  download_hf "$VLLM_TOOLBOX" "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
  download_hf "$LLAMACPP_TOOLBOX" "unsloth/Qwen3.6-35B-A3B-GGUF" "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
}

main() {
  need_command podman
  need_command toolbox

  if [[ -n "${HF_HOME:-}" ]]; then
    echo "Using HF_HOME=$HF_HOME"
  else
    echo "Using default Hugging Face cache under ~/.cache/huggingface"
  fi

  create_vllm_toolbox
  create_llamacpp_toolbox
  verify_toolboxes
  download_models

  echo "Setup complete. Run: python3 bench.py"
}

main "$@"
