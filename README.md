# What is this

This repo answers one practical question on an AMD Strix Halo 128 GB machine:

> How many concurrent local Qwen 3.6 35B A3B agent sessions can run before degradation ?

The default run compares vLLM vs Llama.cpp

- `vllm-awq`: vLLM serving `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`
- `llamacpp-q4kxl`: llama.cpp `llama-server` serving Unsloth `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`

Those aren't *exact* model quants but close enough

## Performance Results

These are from May 15, 2026 and for AMD Strix Halo 128 GB with the Qwen3.6 35 billion parameter, Mixture of Experts with 3B active params. They show how various performance metrics scale with additional agents. `N`: number of concurrent simulated agent workers.

> Llama.cpp actually does better than expected compared to vLLM

### Time to first token scaling

I was actually happy how llama.cpp scaled till 8 but something went wrong after that? `TTFT p95` is 95th percentile time to first streamed token. This is the main interactive latency metric.

![TTFT p95 vs N](docs/benchmarks/2026-05-15_0102-default-n1-2-4-8-16/ttft-p95.svg)

### Aggregate Throughput Scaling

How much total generation does the server produce as concurrency increases? `Agg gen tok/s` is generated output chunks/tokens divided by the measured wall-clock span for that concurrency level.

![Aggregate generation throughput vs N](docs/benchmarks/2026-05-15_0102-default-n1-2-4-8-16/aggregate-gen-throughput.svg)

### Per-Agent Throughput Scaling

As number of agents scale, how much does each agent get? `Gen tok/s` is generated output chunks/tokens divided by summed successful request duration. This is normalized by request duration, so it approximates per-stream throughput.

![Generation throughput vs N](docs/benchmarks/2026-05-15_0102-default-n1-2-4-8-16/gen-throughput.svg)

### Inter-token latency scaling

`ITL p95`: 95th percentile inter-token latency between streamed output chunks after the first token.

![ITL p95 vs N](docs/benchmarks/2026-05-15_0102-default-n1-2-4-8-16/itl-p95.svg)

### Memory Peak Scaling

`Memory peak`: host memory used observed around requests from `/proc/meminfo`. Inference-server logs contain additional GPU memory details.

![Memory peak vs N](docs/benchmarks/2026-05-15_0102-default-n1-2-4-8-16/memory-peak.svg)

### Result Table

| Inference stack | N | Requests | Success | TTFT p95 | ITL p95 | Gen tok/s/stream | Agg gen tok/s | Memory peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `vllm-awq` | 1 | 2 | 100% | 0.950s | 0.057s | 16.990 | 16.988 | 60.59 GiB |
| `vllm-awq` | 2 | 4 | 100% | 0.924s | 0.098s | 11.736 | 18.675 | 60.70 GiB |
| `vllm-awq` | 4 | 4 | 100% | 2.104s | 0.140s | 7.105 | 18.834 | 61.09 GiB |
| `vllm-awq` | 8 | 9 | 100% | 2.431s | 0.214s | 5.809 | 17.591 | 61.74 GiB |
| `vllm-awq` | 16 | 17 | 100% | 2.706s | 0.274s | 4.043 | 25.687 | 61.82 GiB |
| `llamacpp-q4kxl` | 1 | 5 | 100% | 1.294s | 0.021s | 31.602 | 31.586 | 45.48 GiB |
| `llamacpp-q4kxl` | 2 | 5 | 100% | 1.622s | 0.028s | 26.386 | 47.622 | 47.47 GiB |
| `llamacpp-q4kxl` | 4 | 7 | 100% | 2.558s | 0.041s | 17.150 | 44.650 | 50.10 GiB |
| `llamacpp-q4kxl` | 8 | 12 | 100% | 3.327s | 0.067s | 10.957 | 34.895 | 53.29 GiB |
| `llamacpp-q4kxl` | 16 | 19 | 100% | 8.247s | 0.165s | 5.535 | 35.831 | 53.47 GiB |


#### Setup

- OS/kernel: Fedora 44 / Silverblue, kernel `7.0.4-200.fc44.x86_64`
- GPU: AMD Radeon 8060S Graphics / gfx1151
- Model family: Qwen 3.6 35B A3B
- Concurrency ramp: `N = 1, 2, 4, 8, 16`
- Timing window per N: 3s warmup, then 15s measured; in-flight requests are allowed to finish. On the shorter side but sufficient for patterns to emerge.
- Workload: real streamed chat completions, 6-turn simulated coding-agent sessions, temperature `0`
- `vllm-awq`: vLLM `0.19.2rc1.dev113+g6aa057c9d.d20260422`, 16k max model length, 16 max sequences, fp8 KV cache, prefix caching, ROCm attention, Triton AWQ kernels
- `llamacpp-q4kxl`: llama.cpp build `b9146-de6562ffc`, 262k total context, 16 parallel slots, continuous batching, unified KV, flash attention, prompt cache, 8 GiB prompt-cache RAM

## Run it yourself

### 1. Clone Repo

```bash
git clone https://github.com/SidShaytay/llama-vs-vllm
cd llama-vs-vllm
```

### 2. Setup Requirements

Common path:

```bash
# Optional: choose a large model cache location.
export HF_HOME=/path/to/your/models

./scripts/setup-requirements.sh
```

This creates the `vllm` and `llama-rocm-7.2.3` toolboxes, verifies the server binaries, and downloads both benchmark models.

Manual equivalent:

```bash
podman image pull docker.io/kyuz0/vllm-therock-gfx1151:stable
toolbox create --image docker.io/kyuz0/vllm-therock-gfx1151:stable vllm

podman image pull docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2.3
toolbox create --image docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2.3 llama-rocm-7.2.3

toolbox run -c vllm vllm --version
toolbox run -c llama-rocm-7.2.3 llama-cli --list-devices
toolbox run -c llama-rocm-7.2.3 llama-server --version

toolbox run -c vllm hf download cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
toolbox run -c llama-rocm-7.2.3 hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

Before running, make sure nothing else is listening on `localhost:8001` or `localhost:18080`.

### 3. Run the Benchmark

This runs the default full comparison benchmark

```bash
python3 bench.py
```

By default, this uses:

- inference stacks: `vllm-awq`, `llamacpp-q4kxl`
- concurrency: `1,2,4,8,16`
- warmup: 3 seconds per N
- measured window: 15 seconds per N
- output directory: a timestamped child under `results/`

The runner launches each configured inference server, waits until it is actually usable for chat completions, runs the benchmark, then terminates the process it launched. If an endpoint is already healthy before launch, the runner refuses to use it unless that stack explicitly sets `allow_existing: true`; this prevents contamination from a system service such as `llama-swap`.

Useful commands:

```bash
# Default full run across the two working inference stacks.
python3 bench.py

# Single-stack run.
python3 bench.py --only-runtime llamacpp-q4kxl
python3 bench.py --only-runtime vllm-awq

# Short ramp for quick checks.
python3 bench.py --only-runtime vllm-awq --concurrency 1,2,4 --results-dir results/vllm-awq-short

# Smoke test output handling without a model server.
python3 bench.py --config configs/smoke-unreachable.json --results-dir results/smoke

# Optional overload/queueing stress points, excluded from N*.
python3 bench.py --overload-concurrency 20,40
```

> `vllm-fp8` exists as an optional inference-server script but is not in the default comparison. On this Strix Halo/gfx1151 setup it is currently blocked by unsupported FP8 MoE backend support in the tested vLLM/ROCm stack.

## Use A Different Model Or Inference Server

There are two common paths.

### Edit Or Add An Inference Server Script

Inference server scripts are plain shell scripts under `runtimes/`. They should start an OpenAI-compatible server and keep it in the foreground so `bench.py` owns the process lifecycle.

For llama.cpp, copy `runtimes/llamacpp-q4kxl.sh` and change:

- `MODEL`
- `--alias`
- `--port`
- context/parallel/batch settings

For vLLM, copy `runtimes/vllm-awq.sh` and change:

- model name passed to `vllm serve`
- `--port`
- `--max-model-len`
- `--max-num-seqs`
- quantization/cache/backend flags

### Add A Config Entry

Create a JSON config that adds or overrides `runtimes`. Minimal shape:

```json
{
  "primary_concurrency": [1, 2, 4, 8],
  "warmup_duration_s": 3,
  "measure_duration_s": 15,
  "runtimes": [
    {
      "name": "my-runtime",
      "endpoint": "http://localhost:18081/v1",
      "model": "my-served-model-name",
      "command": ["./runtimes/my-runtime.sh"],
      "description": "Short human-readable inference-stack description.",
      "context_description": "Context length, batching, cache, and backend settings.",
      "cleanup_patterns": ["unique process command substring for cleanup"]
    }
  ]
}
```

Then run:

```bash
python3 bench.py --config configs/my-runtime.json
```
