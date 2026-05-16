# TASKS

Marker legend: `[ ]` todo, `[x]` done, `[-]` in progress, `[!]` blocked/needs attention.

## Resume Context

This repository benchmarks local agentic Qwen 3.6 35B A3B serving performance on `vega` across repo-owned llama.cpp and vLLM services. The main benchmark runner is `bench.py`; design background is in `concurrency-test.md`; user-facing commands are in `README.md`; launch scripts are in `runtimes/`.

Current intent:

- `bench.py` should launch its own runtime process from each runtime's configured `command`, wait for `/v1/models`, benchmark `/v1/chat/completions`, then terminate only the process it spawned.
- By default, if an endpoint is already healthy before launch, the runner should refuse to use it. A runtime must explicitly set `allow_existing: true` to use a pre-existing server.
- Default comparison targets:
  - `llamacpp-q4kxl`: repo-owned llama.cpp via `toolbox run -c llama-rocm-7.2.3 llama-server`, endpoint `http://localhost:18080/v1`, script `runtimes/llamacpp-q4kxl.sh`. The toolbox is created from `docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2.3`.
  - `vllm-fp8`: vLLM toolbox via `toolbox run -c vllm`, endpoint `http://localhost:8000/v1`, script `runtimes/vllm-fp8.sh`.
  - `vllm-awq`: vLLM toolbox via `toolbox run -c vllm`, endpoint `http://localhost:8001/v1`, script `runtimes/vllm-awq.sh`.
- `scripts/setup-requirements.sh` installs the published toolbox images, creates missing toolboxes, verifies the installed CLIs, and downloads the default models.
- Generated `results/` are ignored by git and should not be committed unless the user explicitly asks.

Recommended resume procedure for agents:

1. Read `AGENTS.md`, then this file, then `README.md`, then inspect `bench.py` and `runtimes/*.sh`.
2. Run `git status --short` and avoid overwriting unrelated user changes.
3. Pick the highest-priority `[ ]` task below, change it to `[-]`, and commit that marker update with the implementation checkpoint or with the first meaningful code change.
4. Prefer small, verifiable fixes. Run `python3 -m py_compile bench.py` after Python changes.
5. Use short benchmark configs before full runs, because vLLM/model startup can be expensive.
6. Update task markers as work completes or becomes blocked; include blockers inline with `[!]`.

1. [!] Get vLLM FP8 working. Blocked for `vllm-fp8`: the Strix Halo toolbox documents that gfx1x patches disable AITER MoE and Linear FP8 paths, and `Qwen/Qwen3.6-35B-A3B-FP8` fails with no supported FP8 MoE backend on ROCm/gfx1151. `vllm-awq` passes readiness and benchmark runs.

## Completed

- [x] Fix `scripts/setup-requirements.sh` toolbox creation for Fedora 44 Toolbx by using the supported `toolbox create --image IMAGE NAME` syntax instead of passing unsupported Podman flags after `--`.
- [x] Add aggregate generation throughput reporting from measured request wall-clock span, alongside the existing per-stream request-duration throughput.
