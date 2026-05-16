#!/usr/bin/env python3
"""Agentic concurrency benchmark for OpenAI-compatible local LLM servers."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import html
import json
import math
import os
import platform
import queue
import random
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class Runtime:
    name: str
    endpoint: str
    model: str
    command: list[str] | str | None = None
    description: str | None = None
    context_description: str | None = None
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    allow_existing: bool = False
    launch_timeout_s: int = 900
    readiness_chat_timeout_s: int = 60
    cleanup_patterns: list[str] = dataclasses.field(default_factory=list)


DEFAULT_CONFIG: dict[str, Any] = {
    "primary_concurrency": [1, 2, 4, 8, 16],
    "overload_concurrency": [],
    "preflight": False,
    "use_preflight_results": True,
    "preflight_dir": "results/preflight",
    "preflight_concurrency": [1, 2, 4],
    "preflight_min_samples": 6,
    "preflight_session_turns": 3,
    "preflight_warmup_sessions": 0,
    "min_samples": 30,
    "max_samples": 60,
    "warmup_duration_s": 3,
    "measure_duration_s": 15,
    "warmup_sessions": 0,
    "session_turns": 6,
    "think_time_ms": 0,
    "stop_on_saturation": False,
    "request_timeout_s": 180,
    "temperature": 0,
    "seed": 42,
    "results_dir": "results",
    "traces": {
        "tool_call": {"weight": 0.7, "prompt": [1024, 2048], "output": [64, 192]},
        "code_edit": {"weight": 0.2, "prompt": [2048, 4096], "output": [128, 384]},
        "planning": {"weight": 0.1, "prompt": [1024, 2048], "output": [128, 384]},
    },
    "runtimes": [
        {
            "name": "vllm-awq",
            "endpoint": "http://localhost:8001/v1",
            "model": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
            "command": ["./runtimes/vllm-awq.sh"],
            "description": "vLLM OpenAI-compatible server running the AWQ 4-bit Qwen model.",
            "context_description": "16,384-token maximum model context, 16 active sequences, chunked prefill, prefix caching, fp8 KV cache, ROCm attention, Triton AWQ kernels.",
            "readiness_chat_timeout_s": 60,
            "cleanup_patterns": [
                "VLLM::EngineCore",
                "vllm serve cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
            ],
        },
        {
            "name": "llamacpp-q4kxl",
            "endpoint": "http://localhost:18080/v1",
            "model": "qwen3.6-35b-a3b",
            "command": ["./runtimes/llamacpp-q4kxl.sh"],
            "description": "llama.cpp OpenAI-compatible llama-server running the Unsloth Q4_K_XL GGUF.",
            "context_description": "262,144-token total context with 16 parallel slots, continuous batching, unified KV, flash attention, prompt cache, and 8 GiB prompt-cache RAM.",
            "readiness_chat_timeout_s": 60,
            "cleanup_patterns": ["llama-server --host 0.0.0.0 --port 18080"],
        },
    ],
    "preflight_candidates": {
        "llamacpp-q4kxl": [
            {"name": "default", "env": {}},
            {
                "name": "smaller-prefill-batch",
                "env": {"LLAMACPP_BATCH_SIZE": "4096", "LLAMACPP_UBATCH_SIZE": "512"},
            },
            {
                "name": "parallel-8",
                "env": {"LLAMACPP_PARALLEL": "8", "LLAMACPP_CTX_SIZE": "131072"},
            },
        ],
        "vllm-awq": [
            {"name": "default", "env": {}},
            {"name": "lower-memory", "env": {"VLLM_GPU_MEMORY_UTILIZATION": "0.65"}},
            {"name": "smaller-batched-tokens", "env": {"VLLM_MAX_NUM_BATCHED_TOKENS": "4096"}},
        ],
        "vllm-fp8": [
            {"name": "default", "env": {}},
            {"name": "lower-memory", "env": {"VLLM_GPU_MEMORY_UTILIZATION": "0.65"}},
            {"name": "shorter-context", "env": {"VLLM_MAX_MODEL_LEN": "16384"}},
        ],
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return DEFAULT_CONFIG

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return deep_merge(DEFAULT_CONFIG, json.loads(text))

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "YAML config requires PyYAML. Install it, use JSON, or run with defaults."
        ) from exc

    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a mapping: {config_path}")
    return deep_merge(DEFAULT_CONFIG, loaded)


def run_command(argv: list[str], timeout: int = 10) -> str | None:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout + completed.stderr).strip()
    return output or None


def meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, rest = line.split(":", 1)
            value = rest.strip().split()[0]
            info[key] = int(value) * 1024
    except (FileNotFoundError, ValueError):
        pass
    return info


def memory_snapshot() -> dict[str, Any]:
    info = meminfo()
    total = info.get("MemTotal")
    available = info.get("MemAvailable")
    swap_total = info.get("SwapTotal")
    swap_free = info.get("SwapFree")
    snapshot: dict[str, Any] = {
        "mem_total_bytes": total,
        "mem_available_bytes": available,
        "mem_used_bytes": total - available if total is not None and available is not None else None,
        "swap_total_bytes": swap_total,
        "swap_used_bytes": swap_total - swap_free if swap_total is not None and swap_free is not None else None,
    }
    rocm = run_command(["rocm-smi", "--showmeminfo", "vram", "--json"], timeout=10)
    if rocm:
        snapshot["rocm_smi_meminfo_vram"] = rocm
    return snapshot


def environment() -> dict[str, Any]:
    return {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "memory": memory_snapshot(),
        "vllm_version": run_command(["vllm", "--version"]),
        "llama_server_version": run_command(["llama-server", "--version"]),
        "benchmark_git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "benchmark_git_status": run_command(["git", "status", "--short"]),
        "rocm_smi_product": run_command(["rocm-smi", "--showproductname"]),
        "rocminfo_head": run_command(["rocminfo"], timeout=10),
    }


def approx_text_for_tokens(tokens: int, rng: random.Random) -> str:
    words = [
        "inspect",
        "patch",
        "latency",
        "prefix",
        "cache",
        "scheduler",
        "agent",
        "session",
        "context",
        "tokens",
        "runtime",
        "qwen",
        "tool",
        "diff",
        "result",
        "measure",
    ]
    # A short ASCII word plus a space averages near four chars/token for many tokenizers.
    count = max(1, int(tokens * 0.75))
    return " ".join(rng.choice(words) for _ in range(count))


def choose_trace(config: dict[str, Any], rng: random.Random) -> tuple[str, dict[str, Any]]:
    traces = config["traces"]
    total = sum(float(item["weight"]) for item in traces.values())
    pick = rng.random() * total
    seen = 0.0
    for name, trace in traces.items():
        seen += float(trace["weight"])
        if pick <= seen:
            return name, trace
    name = next(reversed(traces))
    return name, traces[name]


def request_chat(
    runtime: Runtime,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> dict[str, Any]:
    url = runtime.endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": runtime.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    start = time.perf_counter()
    first_token_at: float | None = None
    token_times: list[float] = []
    chunks = 0
    content_parts: list[str] = []

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or delta.get("reasoning_content")
                if not content:
                    continue
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                token_times.append(now)
                chunks += 1
                content_parts.append(content)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        end = time.perf_counter()
        return {
            "ok": False,
            "error": f"HTTP {exc.code}: {detail[:500]}",
            "duration_s": end - start,
            "request_start_s": start,
            "request_end_s": end,
        }
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        end = time.perf_counter()
        return {
            "ok": False,
            "error": str(exc),
            "duration_s": end - start,
            "request_start_s": start,
            "request_end_s": end,
        }

    end = time.perf_counter()
    output_text = "".join(content_parts)
    inter_token_latencies = [
        token_times[i] - token_times[i - 1] for i in range(1, len(token_times))
    ]
    return {
        "ok": True,
        "duration_s": end - start,
        "request_start_s": start,
        "request_end_s": end,
        "ttft_s": first_token_at - start if first_token_at else None,
        "chunks": chunks,
        "output_chars": len(output_text),
        "approx_output_tokens": max(chunks, len(output_text.split())),
        "itl_samples_s": inter_token_latencies,
    }


def endpoint_health(runtime: Runtime, timeout_s: int = 5) -> tuple[bool, str | None]:
    url = runtime.endpoint.rstrip("/") + "/models"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if 200 <= response.status < 300:
                return True, None
            return False, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        return False, str(exc)


def chat_health(runtime: Runtime, timeout_s: int) -> tuple[bool, str | None]:
    result = request_chat(
        runtime,
        "Health check. Reply with the single word ok.",
        max_tokens=4,
        temperature=0,
        timeout_s=timeout_s,
    )
    if not result.get("ok"):
        return False, str(result.get("error") or "chat completion failed")
    if result.get("ttft_s") is None or int(result.get("chunks") or 0) < 1:
        return False, "chat completion returned no streamed content"
    return True, None


def command_argv(command: list[str] | str | None) -> list[str] | None:
    if command is None:
        return None
    if isinstance(command, list):
        return [str(part) for part in command]
    return shlex.split(command)


def tail_text(path: Path, line_count: int = 20) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    return lines[-line_count:]


def process_exit_summary(
    process: subprocess.Popen[str] | None,
    log_path: Path,
    last_error: str | None = None,
) -> dict[str, Any]:
    return {
        "server_exit_code": process.returncode if process is not None else None,
        "server_log_path": str(log_path),
        "server_log_tail": tail_text(log_path),
        "last_health_error": last_error,
    }


def format_failure(message: str, details: dict[str, Any]) -> str:
    parts = [message]
    if details.get("server_exit_code") is not None:
        parts.append(f"exit_code={details['server_exit_code']}")
    if details.get("server_log_path"):
        parts.append(f"log={details['server_log_path']}")
    tail = details.get("server_log_tail") or []
    if tail:
        parts.append("last_log_lines=" + " | ".join(str(line) for line in tail[-5:]))
    return "; ".join(parts)


FATAL_STARTUP_LOG_PATTERNS = [
    "EngineCore failed to start",
    "Engine core initialization failed",
    "Free memory on device",
    "CUDA out of memory",
    "HIP out of memory",
    "OutOfMemoryError",
    "MemoryError:",
]


def fatal_startup_log_error(log_path: Path) -> str | None:
    lines = tail_text(log_path, line_count=200)
    for line in lines:
        if any(pattern in line for pattern in FATAL_STARTUP_LOG_PATTERNS):
            return line
    return None


def launch_runtime(runtime: Runtime, runtime_dir: Path) -> subprocess.Popen[str] | None:
    argv = command_argv(runtime.command)
    if not argv:
        return None

    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "server.log"
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in runtime.env.items()})
    env_prefix = ""
    if runtime.env:
        env_prefix = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in sorted(runtime.env.items()))
        env_prefix += " "
    print(f"Launching {runtime.name}: {env_prefix}{' '.join(shlex.quote(part) for part in argv)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        return subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )


def child_pids(parent_pid: int) -> list[int]:
    children: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            status = (proc / "status").read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        ppid: int | None = None
        for line in status.splitlines():
            if line.startswith("PPid:"):
                try:
                    ppid = int(line.split()[1])
                except (IndexError, ValueError):
                    ppid = None
                break
        if ppid == parent_pid:
            pid = int(proc.name)
            children.append(pid)
            children.extend(child_pids(pid))
    return sorted(set(children))


def signal_pids(pids: list[int], sig: signal.Signals) -> None:
    for pid in sorted(set(pids), reverse=True):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def terminate_runtime(process: subprocess.Popen[str] | None, timeout_s: int = 30) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        print("Terminating benchmark-owned runtime process", flush=True)
        descendants = child_pids(process.pid)
        os.killpg(process.pid, signal.SIGTERM)
        signal_pids(descendants, signal.SIGTERM)
        process.wait(timeout=timeout_s)
        print(f"Inference-server process terminated with exit_code={process.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        print("Inference-server process did not terminate after SIGTERM; sending SIGKILL", flush=True)
        descendants = child_pids(process.pid)
        os.killpg(process.pid, signal.SIGKILL)
        signal_pids(descendants, signal.SIGKILL)
        process.wait(timeout=timeout_s)
        print(f"Inference-server process killed with exit_code={process.returncode}", flush=True)
    except ProcessLookupError:
        return


def matching_pids(patterns: list[str]) -> list[int]:
    if not patterns:
        return []
    current_pid = os.getpid()
    pids: list[int] = []
    proc_dir = Path("/proc")
    for child in proc_dir.iterdir():
        if not child.name.isdigit():
            continue
        pid = int(child.name)
        if pid == current_pid:
            continue
        try:
            raw = (child / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        if cmdline and any(pattern in cmdline for pattern in patterns):
            pids.append(pid)
    return sorted(set(pids))


def cleanup_runtime_processes(runtime: Runtime, timeout_s: int = 20) -> None:
    pids = matching_pids(runtime.cleanup_patterns)
    if not pids:
        return
    print(f"{runtime.name}: terminating stale runtime process(es): {pids}", flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = matching_pids(runtime.cleanup_patterns)
        if not remaining:
            return
        time.sleep(1)
    remaining = matching_pids(runtime.cleanup_patterns)
    if remaining:
        print(f"{runtime.name}: force killing stale runtime process(es): {remaining}", flush=True)
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def wait_for_runtime(
    runtime: Runtime,
    process: subprocess.Popen[str] | None,
    log_path: Path,
) -> tuple[bool, str | None, dict[str, Any]]:
    deadline = time.monotonic() + int(runtime.launch_timeout_s)
    last_error: str | None = None
    last_status_at = 0.0
    print(f"{runtime.name}: waiting for readiness at {runtime.endpoint}", flush=True)
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            details = process_exit_summary(process, log_path, last_error)
            error_text = format_failure(
                f"server process exited before readiness; last health error: {last_error}",
                details,
            )
            return False, error_text, details
        fatal_log_error = fatal_startup_log_error(log_path)
        if fatal_log_error:
            details = process_exit_summary(process, log_path, last_error)
            error_text = format_failure(
                f"fatal startup error detected in server log: {fatal_log_error}",
                details,
            )
            return False, error_text, details

        healthy, error = endpoint_health(runtime)
        if healthy:
            chat_ready, chat_error = chat_health(runtime, int(runtime.readiness_chat_timeout_s))
            if chat_ready:
                print(f"{runtime.name}: ready", flush=True)
                return True, None, {}
            last_error = f"/v1/models healthy but chat completion probe failed: {chat_error}"
            if process is not None and process.poll() is not None:
                details = process_exit_summary(process, log_path, last_error)
                error_text = format_failure(
                    f"server process exited before readiness; last health error: {last_error}",
                    details,
                )
                return False, error_text, details
        else:
            last_error = error
        now = time.monotonic()
        if now - last_status_at >= 15:
            remaining = max(0, int(deadline - now))
            log_tail = tail_text(log_path, line_count=3)
            suffix = ""
            if log_tail:
                suffix = " last_log=" + " | ".join(log_tail)
            print(
                f"{runtime.name}: still waiting for readiness; "
                f"remaining={remaining}s last_health_error={last_error}{suffix}",
                flush=True,
            )
            last_status_at = now
        time.sleep(2)
    details = process_exit_summary(process, log_path, last_error)
    return False, f"timed out waiting for server readiness; last health error: {last_error}", details


def worker_session(
    runtime: Runtime,
    config: dict[str, Any],
    concurrency: int,
    worker_id: int,
    measured_target: int | None,
    warmup_sessions: int,
    progress_queue: queue.Queue[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(int(config["seed"]) + concurrency * 1000 + worker_id)
    session_turns = int(config["session_turns"])
    think_time_s = int(config["think_time_ms"]) / 1000
    temperature = float(config["temperature"])
    timeout_s = int(config["request_timeout_s"])
    warmup_duration_s = config.get("warmup_duration_s")
    measure_duration_s = config.get("measure_duration_s")
    use_duration = measure_duration_s is not None
    warmup_deadline = (
        time.monotonic() + float(warmup_duration_s or 0)
        if use_duration and warmup_duration_s is not None
        else None
    )
    measure_deadline: float | None = None
    rows: list[dict[str, Any]] = []
    measured_count = 0
    session_id = 0

    while True:
        if use_duration:
            now = time.monotonic()
            if measure_deadline is None and warmup_deadline is not None and now >= warmup_deadline:
                measure_deadline = now + float(measure_duration_s)
            if measure_deadline is None and warmup_deadline is None:
                measure_deadline = now + float(measure_duration_s)
            if measure_deadline is not None and now >= measure_deadline:
                break
        elif measured_target is not None and measured_count >= measured_target:
            break

        prefix = (
            "You are a local coding agent. Use concise reasoning, inspect files, "
            "apply patches, and report results.\n\n"
        )
        for turn in range(session_turns):
            if use_duration:
                now = time.monotonic()
                if measure_deadline is None and warmup_deadline is not None and now >= warmup_deadline:
                    measure_deadline = now + float(measure_duration_s)
                if measure_deadline is not None and now >= measure_deadline:
                    break

            trace_name, trace = choose_trace(config, rng)
            prompt_min, prompt_max = trace["prompt"]
            output_min, output_max = trace["output"]
            prompt_tokens = rng.randint(int(prompt_min), int(prompt_max))
            max_tokens = rng.randint(int(output_min), int(output_max))
            task_text = approx_text_for_tokens(prompt_tokens, rng)
            prompt = (
                f"{prefix}\n"
                f"Session {session_id}, worker {worker_id}, turn {turn + 1}.\n"
                f"Trace: {trace_name}.\n"
                f"Task material:\n{task_text}\n"
            )
            if use_duration:
                measured = measure_deadline is not None
            else:
                measured = session_id >= warmup_sessions
            before_mem = memory_snapshot()
            result = request_chat(runtime, prompt, max_tokens, temperature, timeout_s)
            after_mem = memory_snapshot()
            row = {
                "runtime": runtime.name,
                "endpoint": runtime.endpoint,
                "model": runtime.model,
                "concurrency": concurrency,
                "worker_id": worker_id,
                "session_id": session_id,
                "turn": turn + 1,
                "phase": "measure" if measured else "warmup",
                "trace": trace_name,
                "requested_prompt_tokens": prompt_tokens,
                "requested_output_tokens": max_tokens,
                "is_first_turn": turn == 0,
                "is_warm_prefix_turn": turn > 0,
                "ok": result.get("ok", False),
                "error": result.get("error"),
                "duration_s": result.get("duration_s"),
                "request_start_s": result.get("request_start_s"),
                "request_end_s": result.get("request_end_s"),
                "ttft_s": result.get("ttft_s"),
                "chunks": result.get("chunks", 0),
                "approx_output_tokens": result.get("approx_output_tokens", 0),
                "itl_samples_s": result.get("itl_samples_s", []),
                "memory_before": before_mem,
                "memory_after": after_mem,
                "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            }
            rows.append(row)
            if progress_queue is not None:
                progress_queue.put(row)
            if measured:
                measured_count += 1
            prefix += (
                f"\nUser turn {turn + 1} requested {trace_name}; assistant produced "
                f"{row['approx_output_tokens']} approximate tokens.\n"
            )
            if think_time_s:
                time.sleep(think_time_s)
            if not use_duration and measured_target is not None and measured_count >= measured_target:
                break
        session_id += 1
        if use_duration and measure_deadline is None and warmup_deadline is None:
            measure_deadline = time.monotonic() + float(measure_duration_s)
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return clean[lower]
    weight = rank - lower
    return clean[lower] * (1 - weight) + clean[upper] * weight


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if row["phase"] == "measure"]
    ok_rows = [row for row in measured if row["ok"]]
    ttfts = [row["ttft_s"] for row in ok_rows if row.get("ttft_s") is not None]
    first_ttfts = [
        row["ttft_s"]
        for row in ok_rows
        if row.get("is_first_turn") and row.get("ttft_s") is not None
    ]
    warm_ttfts = [
        row["ttft_s"]
        for row in ok_rows
        if row.get("is_warm_prefix_turn") and row.get("ttft_s") is not None
    ]
    itls: list[float] = []
    for row in ok_rows:
        itls.extend(float(v) for v in row.get("itl_samples_s") or [])
    durations = [float(row["duration_s"]) for row in ok_rows if row.get("duration_s")]
    output_tokens = sum(int(row.get("approx_output_tokens") or 0) for row in ok_rows)
    prompt_tokens = sum(int(row.get("requested_prompt_tokens") or 0) for row in ok_rows)
    elapsed = sum(durations)
    request_starts = [
        float(row["request_start_s"])
        for row in measured
        if row.get("request_start_s") is not None
    ]
    request_ends = [
        float(row["request_end_s"])
        for row in measured
        if row.get("request_end_s") is not None
    ]
    wall_elapsed = max(request_ends) - min(request_starts) if request_starts and request_ends else None
    mem_used = [
        row.get("memory_after", {}).get("mem_used_bytes")
        for row in rows
        if row.get("memory_after", {}).get("mem_used_bytes") is not None
    ]
    swap_used = [
        row.get("memory_after", {}).get("swap_used_bytes")
        for row in rows
        if row.get("memory_after", {}).get("swap_used_bytes") is not None
    ]
    ttft_mean = statistics.mean(ttfts) if ttfts else None
    ttft_cov = (
        statistics.stdev(ttfts) / ttft_mean
        if ttfts and len(ttfts) > 1 and ttft_mean and ttft_mean > 0
        else None
    )
    return {
        "runtime": measured[0]["runtime"] if measured else rows[0]["runtime"],
        "concurrency": measured[0]["concurrency"] if measured else rows[0]["concurrency"],
        "requests": len(measured),
        "successes": len(ok_rows),
        "success_rate": len(ok_rows) / len(measured) if measured else 0,
        "ttft_p50_s": percentile(ttfts, 0.50),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "first_turn_ttft_p50_s": percentile(first_ttfts, 0.50),
        "first_turn_ttft_p95_s": percentile(first_ttfts, 0.95),
        "warm_prefix_ttft_p50_s": percentile(warm_ttfts, 0.50),
        "warm_prefix_ttft_p95_s": percentile(warm_ttfts, 0.95),
        "itl_p50_s": percentile(itls, 0.50),
        "itl_p95_s": percentile(itls, 0.95),
        "prompt_tok_s": prompt_tokens / elapsed if elapsed else None,
        "gen_tok_s": output_tokens / elapsed if elapsed else None,
        "aggregate_gen_tok_s": output_tokens / wall_elapsed if wall_elapsed else None,
        "measure_wall_s": wall_elapsed,
        "memory_peak_bytes": max(mem_used) if mem_used else None,
        "swap_peak_bytes": max(swap_used) if swap_used else None,
        "ttft_cov": ttft_cov,
    }


def empty_summary(runtime_name: str, concurrency: int) -> dict[str, Any]:
    return {
        "runtime": runtime_name,
        "concurrency": concurrency,
        "requests": 0,
        "successes": 0,
        "success_rate": 0.0,
        "ttft_p50_s": None,
        "ttft_p95_s": None,
        "first_turn_ttft_p50_s": None,
        "first_turn_ttft_p95_s": None,
        "warm_prefix_ttft_p50_s": None,
        "warm_prefix_ttft_p95_s": None,
        "itl_p50_s": None,
        "itl_p95_s": None,
        "prompt_tok_s": None,
        "gen_tok_s": None,
        "aggregate_gen_tok_s": None,
        "measure_wall_s": None,
        "memory_peak_bytes": None,
        "swap_peak_bytes": None,
        "ttft_cov": None,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "runtime",
        "concurrency",
        "requests",
        "successes",
        "success_rate",
        "ttft_p50_s",
        "ttft_p95_s",
        "first_turn_ttft_p50_s",
        "first_turn_ttft_p95_s",
        "warm_prefix_ttft_p50_s",
        "warm_prefix_ttft_p95_s",
        "itl_p50_s",
        "itl_p95_s",
        "prompt_tok_s",
        "gen_tok_s",
        "aggregate_gen_tok_s",
        "measure_wall_s",
        "memory_peak_bytes",
        "swap_peak_bytes",
        "ttft_cov",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fields})


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_rate(value: Any) -> str:
    return fmt(value, 2) if value is not None else "n/a"


def trace_mix(config: dict[str, Any]) -> str:
    traces = config.get("traces", {})
    if not isinstance(traces, dict):
        return ""
    return ", ".join(
        f"{name}={float(trace.get('weight', 0)):.2g}"
        for name, trace in traces.items()
        if isinstance(trace, dict)
    )


def command_text(command: list[str] | str | None) -> str:
    argv = command_argv(command)
    if argv is None:
        return "<existing endpoint only>"
    return " ".join(shlex.quote(part) for part in argv)


def sample_target_text(config: dict[str, Any], concurrency: int) -> str:
    if config.get("measure_duration_s") is not None:
        warmup = config.get("warmup_duration_s")
        warmup_text = f"{warmup}s warmup, " if warmup is not None else ""
        return f"{warmup_text}{config['measure_duration_s']}s measured"
    per_worker = math.ceil(int(config["min_samples"]) / int(concurrency))
    return f"{int(config['min_samples'])} measured requests target ({per_worker}/worker)"


def progress_line(runtime: str, concurrency: int, rows: list[dict[str, Any]], done: bool = False) -> str:
    measured = [row for row in rows if row.get("phase") == "measure"]
    successes = sum(1 for row in measured if row.get("ok"))
    failures = len(measured) - successes
    if not measured:
        return f"{runtime} N={concurrency}: waiting for measured requests"
    summary = summarize_rows(rows)
    prefix = "done" if done else "progress"
    return (
        f"{runtime} N={concurrency} {prefix}: "
        f"{successes} ok/{failures} fail, "
        f"TTFT p50/p95={fmt_rate(summary.get('ttft_p50_s'))}/{fmt_rate(summary.get('ttft_p95_s'))}s, "
        f"warm p95={fmt_rate(summary.get('warm_prefix_ttft_p95_s'))}s, "
        f"ITL p95={fmt_rate(summary.get('itl_p95_s'))}s, "
        f"gen={fmt_rate(summary.get('gen_tok_s'))} tok/s/stream, "
        f"agg={fmt_rate(summary.get('aggregate_gen_tok_s'))} tok/s"
    )


def fmt_bytes(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    units = ["bytes", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    while number >= 1024 and idx < len(units) - 1:
        number /= 1024
        idx += 1
    if idx == 0:
        return f"{number:.0f} {units[idx]}"
    return f"{number:.2f} {units[idx]}"


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def runtime_config_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runtimes = config.get("runtimes") or []
    return {str(item.get("name")): item for item in runtimes if isinstance(item, dict) and item.get("name")}


def first_matching_log_line(log_path: Path, patterns: list[str]) -> str | None:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        if any(pattern in line for pattern in patterns):
            return line.strip()
    return None


def redact_private_text(value: str) -> str:
    value = re.sub(r"/var/mnt/data/users/[^/\s'\"<>]+", "/var/mnt/data/users/<user>", value)
    value = re.sub(r"/var/home/[^/\s'\"<>]+", "/var/home/<user>", value)
    value = re.sub(r"/home/[^/\s'\"<>]+", "/home/<user>", value)
    return value


def runtime_version_from_log(runtime_name: str, report_dir: Path) -> str:
    log_path = report_dir / runtime_name / "server.log"
    if runtime_name.startswith("vllm"):
        line = first_matching_log_line(log_path, ["version "])
        if line:
            match = re.search(r"version\s+([^ ]+)", line)
            if match:
                return f"vLLM {match.group(1)}"
    if runtime_name.startswith("llamacpp"):
        line = first_matching_log_line(log_path, ["build_info:"])
        if line:
            return redact_private_text(line.replace("build_info:", "llama.cpp build").strip())
    return "see server.log"


def runtime_model_details_from_log(runtime_name: str, report_dir: Path) -> list[str]:
    log_path = report_dir / runtime_name / "server.log"
    details: list[str] = []
    patterns = [
        "non-default args:",
        "Initializing a V1 LLM engine",
        "Maximum concurrency for",
        "Using max model len",
        "file type",
        "file size",
        "n_ctx_train",
        "Device 0:",
    ]
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return details
    for line in lines:
        cleaned = line.strip()
        if any(pattern in cleaned for pattern in patterns) and cleaned not in details:
            details.append(redact_private_text(cleaned))
        if len(details) >= 6:
            break
    return details


def html_list(items: list[str]) -> str:
    if not items:
        return "<p>n/a</p>"
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def report_html(path: Path, summaries: list[dict[str, Any]], config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = path.parent
    env = read_json_file(report_dir / "environment.json")
    runtime_configs = runtime_config_map(config)
    by_runtime: dict[str, list[dict[str, Any]]] = {}
    for row in summaries:
        by_runtime.setdefault(str(row["runtime"]), []).append(row)

    primary = set(config.get("primary_concurrency", []))
    nstar: dict[str, int | None] = {}
    for runtime, runtime_rows in by_runtime.items():
        baseline_rows = [r for r in runtime_rows if r.get("concurrency") == 1]
        baseline = baseline_rows[0].get("ttft_p95_s") if baseline_rows else None
        eligible = []
        for row in runtime_rows:
            ttft = row.get("ttft_p95_s")
            if (
                row.get("concurrency") in primary
                and baseline is not None
                and ttft is not None
                and ttft < 10 * baseline
                and row.get("success_rate") == 1.0
            ):
                eligible.append(int(row["concurrency"]))
        nstar[runtime] = max(eligible) if eligible else None

    rows_html = []
    for row in summaries:
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['runtime']))}</td>"
            f"<td>{row['concurrency']}</td>"
            f"<td>{fmt(nstar.get(str(row['runtime'])))}</td>"
            f"<td>{fmt(row['success_rate'])}</td>"
            f"<td>{fmt(row['ttft_p95_s'])}</td>"
            f"<td>{fmt(row['warm_prefix_ttft_p95_s'])}</td>"
            f"<td>{fmt(row['itl_p95_s'])}</td>"
            f"<td>{fmt(row['gen_tok_s'])}</td>"
            f"<td>{fmt(row.get('aggregate_gen_tok_s'))}</td>"
            f"<td>{fmt(row['memory_peak_bytes'], 0)}</td>"
            "</tr>"
        )

    primary_text = ", ".join(str(item) for item in config.get("primary_concurrency", []))
    overload_text = ", ".join(str(item) for item in config.get("overload_concurrency", [])) or "none"
    traces = config.get("traces") or {}
    trace_rows = []
    for name, trace in traces.items():
        if not isinstance(trace, dict):
            continue
        prompt_range = trace.get("prompt", ["?", "?"])
        output_range = trace.get("output", ["?", "?"])
        trace_rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{fmt(trace.get('weight'))}</td>"
            f"<td>{html.escape(str(prompt_range[0]))}-{html.escape(str(prompt_range[1]))}</td>"
            f"<td>{html.escape(str(output_range[0]))}-{html.escape(str(output_range[1]))}</td>"
            "</tr>"
        )

    runtime_cards = []
    for runtime_name, runtime_rows in by_runtime.items():
        runtime_config = runtime_configs.get(runtime_name, {})
        command = command_text(runtime_config.get("command"))
        env_text = runtime_config.get("env") or {}
        env_parts = [f"{key}={value}" for key, value in sorted(env_text.items())] if isinstance(env_text, dict) else []
        setup_items = [
            f"Endpoint: {runtime_rows[0].get('endpoint', runtime_config.get('endpoint', 'n/a')) if runtime_rows else runtime_config.get('endpoint', 'n/a')}",
            f"Served model: {runtime_rows[0].get('model', runtime_config.get('model', 'n/a')) if runtime_rows else runtime_config.get('model', 'n/a')}",
            f"Software: {runtime_version_from_log(runtime_name, report_dir)}",
            f"Launch command: {command}",
        ]
        if env_parts:
            setup_items.append("Environment overrides: " + ", ".join(env_parts))
        setup_items = [redact_private_text(str(item)) for item in setup_items]
        description = runtime_config.get("description")
        context_description = runtime_config.get("context_description")
        runtime_cards.append(
            "<section class=\"runtime-card\">"
            f"<h3>{html.escape(runtime_name)}</h3>"
            f"<p>{html.escape(redact_private_text(str(description or 'Inference stack details are from the benchmark config and server log.')))}</p>"
            f"<p><strong>Context/server settings:</strong> {html.escape(redact_private_text(str(context_description or 'see launch command and server log')))}</p>"
            f"{html_list(setup_items)}"
            "<details><summary>Server log facts used in this report</summary>"
            f"{html_list(runtime_model_details_from_log(runtime_name, report_dir))}"
            "</details>"
            "</section>"
        )

    machine_items = [
        f"Benchmark started at UTC: {env.get('timestamp_utc', 'n/a')}",
        f"Benchmark git commit: {env.get('benchmark_git_commit', 'n/a')}",
        f"Platform: {env.get('platform', 'n/a')}",
        f"Kernel: {env.get('kernel', 'n/a')}",
        f"Python: {env.get('python', 'n/a')}",
        f"Memory total: {fmt_bytes((env.get('memory') or {}).get('mem_total_bytes'))}",
        f"Memory available at benchmark start: {fmt_bytes((env.get('memory') or {}).get('mem_available_bytes'))}",
        f"Swap total: {fmt_bytes((env.get('memory') or {}).get('swap_total_bytes'))}",
    ]

    chart_data = json.dumps(summaries)
    primary_data = json.dumps(list(config.get("primary_concurrency", [])))
    runtime_names = json.dumps(list(by_runtime.keys()))
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Qwen Agentic Concurrency Benchmark</title>
  <style>
    body {{ color: #1b1b1b; font-family: system-ui, sans-serif; line-height: 1.45; margin: 2rem; max-width: 1200px; }}
    h1, h2, h3 {{ line-height: 1.15; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    canvas {{ display: block; max-width: 100%; border: 1px solid #ddd; margin: 0.5rem 0 2rem; }}
    code {{ background: #f5f5f5; padding: 0.1rem 0.25rem; }}
    .note {{ background: #f7f7f7; border-left: 4px solid #777; padding: 0.75rem 1rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; }}
    .runtime-card {{ border: 1px solid #ddd; padding: 1rem; }}
    .runtime-card h3 {{ margin-top: 0; }}
    .chart-label {{ font-weight: 600; margin-top: 1.5rem; }}
    details {{ margin-top: 0.75rem; }}
  </style>
</head>
<body>
  <h1>Qwen Agentic Concurrency Benchmark</h1>
  <p class="note">This is a generated benchmark report. It is intended to be readable without the repository open: the workload, inference-stack configuration, metric definitions, and chart units are included here.</p>

  <h2>Headline Result</h2>
  <p><code>N*</code> is the largest primary concurrency level where request success is 100% and <code>TTFT p95</code> is less than 10x that inference stack's N=1 baseline. In this report: {html.escape(", ".join(f"{name}: N*={value}" for name, value in nstar.items()))}.</p>

  <h2>Test Setup</h2>
  <p>The benchmark launched each inference stack as an OpenAI-compatible chat completions server, waited for <code>/v1/models</code> and a streamed chat health check, then ran the same synthetic agentic workload at primary concurrency levels <code>{html.escape(primary_text)}</code>. Optional overload levels: <code>{html.escape(overload_text)}</code>.</p>
  <div class="grid">
    <section>
      <h3>Machine</h3>
      {html_list(machine_items)}
    </section>
    <section>
      <h3>Workload</h3>
      <p>Each worker is an independent simulated local coding-agent session. A request is a real streamed <code>/v1/chat/completions</code> call, not llama-bench random token generation. The API payload contains a single user message; the first lines of that user message instruct the model to act as a local coding agent: inspect files, apply patches, and report results. The rest of the prompt is synthetic task material made from repeated technical words so prompt length is reproducible.</p>
      <p>Sessions have <code>{html.escape(str(config.get('session_turns')))}</code> turns. Later turns include a short text summary of earlier turns, which exercises prefix/cache reuse. Temperature is <code>{html.escape(str(config.get('temperature')))}</code>. Output is actual streamed model text capped by the per-trace max-token range below.</p>
      <p>Fixed-duration mode: <code>{html.escape(str(config.get('warmup_duration_s')))}</code>s warmup, then <code>{html.escape(str(config.get('measure_duration_s')))}</code>s measured window. Requests already in flight are allowed to finish, so the wall-clock time per N can exceed the measured window.</p>
    </section>
  </div>

  <h3>Trace Mix</h3>
  <table>
    <thead><tr><th>Trace</th><th>Weight</th><th>Prompt target tokens</th><th>Output cap tokens</th></tr></thead>
    <tbody>{''.join(trace_rows)}</tbody>
  </table>

  <h2>Inference Stacks</h2>
  <div class="grid">
    {''.join(runtime_cards)}
  </div>

  <h2>Metric Glossary</h2>
  <ul>
    <li><code>N</code>: concurrent simulated agent sessions/workers issuing chat requests to the inference server.</li>
    <li><code>N*</code>: largest primary-ramp N with 100% success and TTFT p95 below 10x the inference stack's N=1 TTFT p95.</li>
    <li><code>TTFT p95</code>: 95th percentile time to first streamed token, in seconds, across measured requests. Lower is better for interactive agents.</li>
    <li><code>Warm TTFT p95</code>: TTFT p95 for non-first turns in a session, where previous-turn text may be reusable by prefix/cache mechanisms.</li>
    <li><code>ITL p95</code>: 95th percentile inter-token latency, in seconds, between streamed output chunks after the first token.</li>
    <li><code>Gen tok/s/stream</code>: generated output chunks/tokens divided by summed successful request duration, in output tokens per request-second. It is normalized per request duration, so it approximates per-stream throughput.</li>
    <li><code>Agg gen tok/s</code>: generated output chunks/tokens from successful measured requests divided by the measured wall-clock span, from first measured request start to last measured request end. This is the server-level aggregate generation throughput.</li>
    <li><code>Prompt tok/s</code>: requested prompt tokens divided by summed successful request duration, in prompt tokens per request-second.</li>
    <li><code>Memory peak</code>: peak host memory used observed around benchmark requests, from <code>/proc/meminfo</code>; inference-server logs may include more specific GPU memory details.</li>
  </ul>

  <h2>Charts</h2>
  <p>Axes are linear. The x-axis is <code>N</code>, the configured concurrency level. Units are shown in each y-axis label.</p>
  <div class="chart-label">TTFT p95 vs N (seconds)</div>
  <canvas id="ttft" width="960" height="360"></canvas>
  <div class="chart-label">Aggregate generation throughput vs N (output tokens per wall-clock second)</div>
  <canvas id="aggregate-throughput" width="960" height="360"></canvas>
  <div class="chart-label">Per-stream generation throughput vs N (output tokens per request-second)</div>
  <canvas id="throughput" width="960" height="360"></canvas>
  <div class="chart-label">ITL p95 vs N (seconds)</div>
  <canvas id="itl" width="960" height="360"></canvas>
  <div class="chart-label">Memory peak vs N (GiB)</div>
  <canvas id="memory" width="960" height="360"></canvas>

  <h2>Summary</h2>
  <table>
    <thead>
      <tr><th>Inference stack</th><th>N</th><th>N*</th><th>Success</th><th>TTFT p95 s</th><th>Warm TTFT p95 s</th><th>ITL p95 s</th><th>Gen tok/s/stream</th><th>Agg gen tok/s</th><th>Memory peak bytes</th></tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
  <script>
    const data = {chart_data};
    const primaryN = {primary_data};
    const runtimeNames = {runtime_names};
    const colors = ["#005f73", "#9b2226", "#0a9396", "#ca6702", "#3a0ca3"];
    function niceTicks(maxValue, count) {{
      if (!Number.isFinite(maxValue) || maxValue <= 0) return [0, 1];
      const rawStep = maxValue / Math.max(1, count);
      const pow = Math.pow(10, Math.floor(Math.log10(rawStep)));
      const scaled = rawStep / pow;
      const nice = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
      const step = nice * pow;
      const top = Math.ceil(maxValue / step) * step;
      const ticks = [];
      for (let value = 0; value <= top + step / 2; value += step) ticks.push(value);
      return ticks;
    }}
    function formatTick(value, unitScale) {{
      const scaled = value / (unitScale || 1);
      if (Math.abs(scaled) >= 100) return scaled.toFixed(0);
      if (Math.abs(scaled) >= 10) return scaled.toFixed(1);
      return scaled.toFixed(2);
    }}
    function drawChart(id, key, title, yLabel, unitScale = 1) {{
      const canvas = document.getElementById(id);
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      const padLeft = 78, padRight = 28, padTop = 42, padBottom = 64;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#111";
      ctx.font = "16px sans-serif";
      ctx.fillText(title, padLeft, 24);
      const values = data.filter(d => d[key] !== null && d[key] !== undefined);
      if (!values.length) return;
      const xs = primaryN.length ? primaryN : values.map(d => d.concurrency);
      const ys = values.map(d => Number(d[key]));
      const minX = Math.min(...xs, 1);
      const maxX = Math.max(...xs, 1);
      const ticksY = niceTicks(Math.max(...ys, 1e-9) / unitScale, 5).map(v => v * unitScale);
      const maxY = Math.max(...ticksY, 1e-9);
      function xPos(n) {{
        if (maxX === minX) return (padLeft + w - padRight) / 2;
        return padLeft + (w - padLeft - padRight) * ((n - minX) / (maxX - minX));
      }}
      function yPos(v) {{
        return h - padBottom - (h - padTop - padBottom) * (Number(v) / maxY);
      }}
      ctx.strokeStyle = "#bbb";
      ctx.beginPath();
      ctx.moveTo(padLeft, h - padBottom);
      ctx.lineTo(w - padRight, h - padBottom);
      ctx.moveTo(padLeft, h - padBottom);
      ctx.lineTo(padLeft, padTop);
      ctx.stroke();
      ctx.font = "12px sans-serif";
      ctx.fillStyle = "#333";
      ctx.textAlign = "center";
      xs.forEach(n => {{
        const x = xPos(n);
        ctx.strokeStyle = "#eee";
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, h - padBottom);
        ctx.stroke();
        ctx.fillText(String(n), x, h - padBottom + 22);
      }});
      ctx.textAlign = "right";
      ticksY.forEach(t => {{
        const y = yPos(t);
        ctx.strokeStyle = "#eee";
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(w - padRight, y);
        ctx.stroke();
        ctx.fillStyle = "#333";
        ctx.fillText(formatTick(t, unitScale), padLeft - 8, y + 4);
      }});
      ctx.save();
      ctx.translate(18, (padTop + h - padBottom) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
      ctx.textAlign = "center";
      ctx.fillText("N (concurrent agent sessions)", (padLeft + w - padRight) / 2, h - 18);
      const runtimes = runtimeNames.length ? runtimeNames : [...new Set(data.map(d => d.runtime))];
      runtimes.forEach((runtime, idx) => {{
        const points = data.filter(d => d.runtime === runtime && d[key] !== null && d[key] !== undefined)
          .sort((a, b) => a.concurrency - b.concurrency);
        ctx.strokeStyle = colors[idx % colors.length];
        ctx.fillStyle = colors[idx % colors.length];
        ctx.beginPath();
        points.forEach((p, i) => {{
          const x = xPos(p.concurrency);
          const y = yPos(Number(p[key]));
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          ctx.fillRect(x - 3, y - 3, 6, 6);
        }});
        ctx.stroke();
        ctx.textAlign = "left";
        ctx.fillText(runtime, w - padRight - 180, padTop + 18 * idx);
      }});
    }}
    drawChart("ttft", "ttft_p95_s", "TTFT p95 vs N", "TTFT p95 (seconds)");
    drawChart("aggregate-throughput", "aggregate_gen_tok_s", "Aggregate generation throughput vs N", "Output tokens / wall-clock second");
    drawChart("throughput", "gen_tok_s", "Per-stream generation throughput vs N", "Output tokens / request-second");
    drawChart("itl", "itl_p95_s", "ITL p95 vs N", "ITL p95 (seconds)");
    drawChart("memory", "memory_peak_bytes", "Memory peak vs N", "Memory peak (GiB)", 1073741824);
  </script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def preflight_cache_dir(config: dict[str, Any]) -> Path:
    return Path(str(config.get("preflight_dir") or "results/preflight"))


def preflight_candidates(config: dict[str, Any], runtime: Runtime) -> list[dict[str, Any]]:
    configured = config.get("preflight_candidates", {})
    candidates = configured.get(runtime.name, []) if isinstance(configured, dict) else []
    if not candidates:
        return [{"name": "default", "env": {}}]
    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        name = str(candidate.get("name") or f"candidate-{index + 1}")
        env = candidate.get("env") or {}
        if not isinstance(env, dict):
            raise SystemExit(f"Preflight candidate {runtime.name}/{name} env must be a mapping")
        normalized.append({"name": name, "env": {str(k): str(v) for k, v in env.items()}})
    return normalized


def preflight_benchmark_config(config: dict[str, Any]) -> dict[str, Any]:
    tuned = deep_merge(config, {})
    tuned["use_preflight_results"] = False
    tuned["primary_concurrency"] = list(config.get("preflight_concurrency", [1, 2, 4]))
    tuned["overload_concurrency"] = []
    tuned["min_samples"] = int(config.get("preflight_min_samples", 6))
    tuned["max_samples"] = int(config.get("preflight_min_samples", 6))
    tuned["session_turns"] = int(config.get("preflight_session_turns", 3))
    tuned["warmup_sessions"] = int(config.get("preflight_warmup_sessions", 0))
    tuned["stop_on_saturation"] = False
    tuned.pop("warmup_duration_s", None)
    tuned.pop("measure_duration_s", None)
    return tuned


def candidate_score(summaries: list[dict[str, Any]]) -> tuple[bool, float, float, int]:
    valid = [
        row for row in summaries
        if int(row.get("requests") or 0) > 0 and float(row.get("success_rate") or 0) == 1.0
    ]
    if len(valid) != len(summaries) or not valid:
        return False, math.inf, 0.0, 0
    highest = max(valid, key=lambda row: int(row.get("concurrency") or 0))
    latency = highest.get("warm_prefix_ttft_p95_s") or highest.get("ttft_p95_s")
    if latency is None:
        return False, math.inf, 0.0, int(highest.get("concurrency") or 0)
    throughput = float(highest.get("aggregate_gen_tok_s") or highest.get("gen_tok_s") or 0.0)
    return True, float(latency), throughput, int(highest.get("concurrency") or 0)


def run_preflight(config: dict[str, Any], results_dir: Path) -> None:
    cache_dir = preflight_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_root = results_dir / "preflight-runs"
    benchmark_config = preflight_benchmark_config(config)

    for item in config["runtimes"]:
        runtime = Runtime(**item)
        records: list[dict[str, Any]] = []
        best_record: dict[str, Any] | None = None
        print(f"{runtime.name}: starting pre-flight candidate sweep", flush=True)

        for candidate in preflight_candidates(config, runtime):
            candidate_runtime = dataclasses.replace(
                runtime,
                env={**runtime.env, **candidate["env"]},
            )
            candidate_dir = run_root / runtime.name / candidate["name"]
            summaries = run_runtime(candidate_runtime, benchmark_config, candidate_dir)
            eligible, latency, throughput, highest_n = candidate_score(summaries)
            record = {
                "name": candidate["name"],
                "env": candidate["env"],
                "eligible": eligible,
                "score_latency_s": None if math.isinf(latency) else latency,
                "score_aggregate_gen_tok_s": throughput,
                "score_concurrency": highest_n,
                "summaries": summaries,
            }
            records.append(record)
            if eligible and (
                best_record is None
                or float(record["score_latency_s"]) < float(best_record["score_latency_s"])
                or (
                    record["score_latency_s"] == best_record["score_latency_s"]
                    and float(record["score_aggregate_gen_tok_s"])
                    > float(best_record["score_aggregate_gen_tok_s"])
                )
            ):
                best_record = record

        selected = {
            "runtime": runtime.name,
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "preflight_concurrency": benchmark_config["primary_concurrency"],
            "selection_metric": "lowest warm-prefix TTFT p95 at the highest preflight concurrency with 100% success; aggregate generation throughput breaks ties",
            "status": "selected" if best_record else "no_eligible_candidate",
            "selected_candidate": best_record["name"] if best_record else None,
            "selected_settings": {"env": best_record["env"]} if best_record else {},
            "candidates": records,
        }
        output_path = cache_dir / f"{runtime.name}.json"
        output_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{runtime.name}: wrote pre-flight selection to {output_path}", flush=True)


def apply_preflight_settings(runtime: Runtime, config: dict[str, Any]) -> Runtime:
    if not config.get("use_preflight_results", True):
        return runtime
    path = preflight_cache_dir(config) / f"{runtime.name}.json"
    if not path.exists():
        return runtime
    try:
        selected = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid preflight cache {path}: {exc}") from exc
    if selected.get("status") != "selected":
        print(f"{runtime.name}: ignoring pre-flight cache with status={selected.get('status')}", flush=True)
        return runtime
    selected_settings = selected.get("selected_settings") or {}
    selected_env = selected_settings.get("env") or {}
    if not isinstance(selected_env, dict):
        raise SystemExit(f"Invalid preflight env in {path}")
    merged_env = {**runtime.env, **{str(k): str(v) for k, v in selected_env.items()}}
    if merged_env != runtime.env:
        print(f"{runtime.name}: applying pre-flight settings from {path}", flush=True)
    return dataclasses.replace(runtime, env=merged_env)


def run_runtime(runtime: Runtime, config: dict[str, Any], results_dir: Path) -> list[dict[str, Any]]:
    runtime_dir = results_dir / runtime.name
    raw_path = runtime_dir / "raw.jsonl"
    log_path = runtime_dir / "server.log"
    if raw_path.exists():
        raw_path.unlink()
    summaries: list[dict[str, Any]] = []
    concurrencies = list(config["primary_concurrency"]) + list(config.get("overload_concurrency", []))
    print(
        f"{runtime.name}: endpoint={runtime.endpoint} model={runtime.model} "
        f"command={command_text(runtime.command)}",
        flush=True,
    )
    print(
        f"{runtime.name}: concurrency={','.join(str(c) for c in concurrencies)} "
        f"warmup_sessions={config.get('warmup_sessions')} trace_mix={trace_mix(config)}",
        flush=True,
    )

    if not runtime.allow_existing:
        cleanup_runtime_processes(runtime)

    preexisting, preexisting_error = endpoint_health(runtime)
    if preexisting and not runtime.allow_existing:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "runtime": runtime.name,
            "endpoint": runtime.endpoint,
            "model": runtime.model,
            "phase": "prelaunch",
            "ok": False,
            "error": "endpoint already healthy before benchmark launch; refusing to use a server this run did not spawn",
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        write_jsonl(raw_path, [row])
        print(f"{runtime.name}: refusing pre-existing endpoint at {runtime.endpoint}", flush=True)
        summary = empty_summary(runtime.name, 0)
        write_summary_csv(runtime_dir / "summary.csv", [summary])
        return [summary]

    process: subprocess.Popen[str] | None = None
    if not preexisting:
        process = launch_runtime(runtime, runtime_dir)

    health_details: dict[str, Any] = {}
    if preexisting and runtime.allow_existing:
        healthy, health_error = chat_health(runtime, int(runtime.readiness_chat_timeout_s))
        if not healthy:
            health_error = f"pre-existing endpoint passed /v1/models but failed chat readiness: {health_error}"
    elif process is None:
        healthy, health_error = False, f"endpoint is not healthy and no launch command is configured: {preexisting_error}"
    else:
        healthy, health_error, health_details = wait_for_runtime(runtime, process, log_path)
    if not healthy:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "runtime": runtime.name,
            "endpoint": runtime.endpoint,
            "model": runtime.model,
            "phase": "healthcheck",
            "ok": False,
            "error": health_error,
            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        row.update(health_details)
        write_jsonl(raw_path, [row])
        print(f"{runtime.name}: readiness failed: {health_error}", flush=True)
        summary = empty_summary(runtime.name, 0)
        write_summary_csv(runtime_dir / "summary.csv", [summary])
        terminate_runtime(process)
        cleanup_runtime_processes(runtime)
        return [summary]

    try:
        baseline_ttft: float | None = None
        previous_throughputs: list[float] = []

        for concurrency in concurrencies:
            if process is not None and process.poll() is not None:
                details = process_exit_summary(process, log_path)
                error_text = format_failure(
                    f"server process exited before concurrency {concurrency}",
                    details,
                )
                write_jsonl(
                    raw_path,
                    [
                        {
                            "runtime": runtime.name,
                            "endpoint": runtime.endpoint,
                            "model": runtime.model,
                            "concurrency": concurrency,
                            "phase": "server_exit",
                            "ok": False,
                            "error": error_text,
                            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                            **details,
                        }
                    ],
                )
                print(f"{runtime.name}: {error_text}", flush=True)
                if not summaries:
                    summaries.append(empty_summary(runtime.name, int(concurrency)))
                    write_summary_csv(runtime_dir / "summary.csv", summaries)
                break
            use_duration = config.get("measure_duration_s") is not None
            measured_target = None if use_duration else math.ceil(int(config["min_samples"]) / int(concurrency))
            rows: list[dict[str, Any]] = []
            progress_rows: list[dict[str, Any]] = []
            progress_updates: queue.Queue[dict[str, Any]] = queue.Queue()
            print(
                f"{runtime.name} N={concurrency}: benchmark started; "
                f"{sample_target_text(config, int(concurrency))}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=int(concurrency)) as executor:
                futures = [
                    executor.submit(
                        worker_session,
                        runtime,
                        config,
                        int(concurrency),
                        worker_id,
                        measured_target,
                        int(config["warmup_sessions"]),
                        progress_updates,
                    )
                    for worker_id in range(int(concurrency))
                ]
                pending = set(futures)
                started_at = time.monotonic()
                last_print_at = 0.0
                last_print_count = 0
                while pending:
                    try:
                        row = progress_updates.get(timeout=0.5)
                        progress_rows.append(row)
                    except queue.Empty:
                        pass

                    completed = {future for future in pending if future.done()}
                    for future in completed:
                        worker_rows = future.result()
                        rows.extend(worker_rows)
                        write_jsonl(raw_path, worker_rows)
                    pending -= completed

                    measured_seen = sum(1 for row in progress_rows if row.get("phase") == "measure")
                    now = time.monotonic()
                    if measured_seen and (
                        measured_seen >= last_print_count + max(1, int(concurrency))
                        or now - last_print_at >= 5
                        or not pending
                    ):
                        print(progress_line(runtime.name, int(concurrency), progress_rows), flush=True)
                        last_print_at = now
                        last_print_count = measured_seen
                    elif now - last_print_at >= 10:
                        completed_count = len(futures) - len(pending)
                        elapsed_s = int(now - started_at)
                        print(
                            f"{runtime.name} N={concurrency} progress: "
                            f"waiting for completions, completed_workers={completed_count}/{len(futures)}, "
                            f"pending_workers={len(pending)}, elapsed={elapsed_s}s",
                            flush=True,
                        )
                        last_print_at = now

                while True:
                    try:
                        progress_rows.append(progress_updates.get_nowait())
                    except queue.Empty:
                        break

            summary = summarize_rows(rows)
            summaries.append(summary)
            write_summary_csv(runtime_dir / "summary.csv", summaries)
            print(progress_line(runtime.name, int(concurrency), rows, done=True), flush=True)

            if process is not None and process.poll() is not None:
                details = process_exit_summary(process, log_path)
                error_text = format_failure(
                    f"server process exited during or after concurrency {concurrency}",
                    details,
                )
                write_jsonl(
                    raw_path,
                    [
                        {
                            "runtime": runtime.name,
                            "endpoint": runtime.endpoint,
                            "model": runtime.model,
                            "concurrency": concurrency,
                            "phase": "server_exit",
                            "ok": False,
                            "error": error_text,
                            "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                            **details,
                        }
                    ],
                )
                print(f"{runtime.name}: {error_text}", flush=True)
                break

            if int(concurrency) == 1:
                baseline_ttft = summary.get("ttft_p95_s")

            if config.get("stop_on_saturation", True):
                ttft_p95 = summary.get("ttft_p95_s")
                throughput = summary.get("aggregate_gen_tok_s") or summary.get("gen_tok_s") or 0
                previous_throughputs.append(float(throughput))
                ttft_saturated = (
                    baseline_ttft is not None
                    and ttft_p95 is not None
                    and ttft_p95 > 10 * baseline_ttft
                )
                failed = summary.get("success_rate", 0) < 1.0
                plateau = (
                    len(previous_throughputs) >= 3
                    and previous_throughputs[-1] <= previous_throughputs[-2] * 1.03
                    and previous_throughputs[-2] <= previous_throughputs[-3] * 1.03
                )
                if ttft_saturated or failed or plateau:
                    break
    finally:
        if not runtime.allow_existing:
            terminate_runtime(process)
            cleanup_runtime_processes(runtime)

    return summaries


def write_default_config(path: Path) -> None:
    path.write_text(json.dumps(DEFAULT_CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_int_csv(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("concurrency values must be >= 1")
    return parsed


def slug_part(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    normalized = normalized.strip("-._")
    return normalized or "run"


def results_slug(config: dict[str, Any], config_path: str | None, only_runtime: list[str] | None) -> str:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    parts = [timestamp]
    if config_path:
        parts.append(slug_part(Path(config_path).stem))
    else:
        parts.append("default")
    if only_runtime:
        parts.append(slug_part("-".join(sorted(only_runtime))))
    concurrency = "-".join(str(item) for item in config.get("primary_concurrency", []))
    if concurrency:
        parts.append(f"n{slug_part(concurrency)}")
    return "-".join(parts)


def unique_results_dir(base_dir: Path, slug: str) -> Path:
    candidate = base_dir / slug
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        candidate = base_dir / f"{slug}-{index:02d}"
        if not candidate.exists():
            return candidate
    raise SystemExit(f"Could not find unused results directory under {base_dir}")


def should_slug_results_dir(args: argparse.Namespace) -> bool:
    if args.no_results_slug:
        return False
    if not args.results_dir:
        return True
    return Path(args.results_dir).name == "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML or JSON config file. Defaults are built in.")
    parser.add_argument("--results-dir", help="Override results directory.")
    parser.add_argument(
        "--no-results-slug",
        action="store_true",
        help="Use the configured results directory exactly instead of creating a timestamped child.",
    )
    parser.add_argument("--preflight", action="store_true", help="Write/run optional pre-flight scaffold.")
    parser.add_argument("--write-default-config", help="Write default JSON config and exit.")
    parser.add_argument("--only-runtime", action="append", help="Limit to a runtime name. Repeatable.")
    parser.add_argument(
        "--concurrency",
        type=parse_int_csv,
        help="Override benchmark concurrency ramp, e.g. 1,2,4,8.",
    )
    parser.add_argument(
        "--primary-concurrency",
        type=parse_int_csv,
        help="Deprecated alias for --concurrency.",
    )
    parser.add_argument(
        "--overload-concurrency",
        type=parse_int_csv,
        help="Override optional overload concurrency ramp, e.g. 20,40.",
    )
    args = parser.parse_args()

    if args.write_default_config:
        write_default_config(Path(args.write_default_config))
        return 0

    config = load_config(args.config)
    if args.results_dir:
        config["results_dir"] = args.results_dir
    if args.preflight:
        config["preflight"] = True
    if args.concurrency and args.primary_concurrency:
        raise SystemExit("Use only one of --concurrency or --primary-concurrency.")
    if args.concurrency:
        config["primary_concurrency"] = args.concurrency
    if args.primary_concurrency:
        print(
            "warning: --primary-concurrency is deprecated; use --concurrency",
            file=sys.stderr,
            flush=True,
        )
        config["primary_concurrency"] = args.primary_concurrency
    if args.overload_concurrency:
        config["overload_concurrency"] = args.overload_concurrency
    if args.only_runtime:
        selected = set(args.only_runtime)
        config["runtimes"] = [r for r in config["runtimes"] if r["name"] in selected]
        missing = selected - {r["name"] for r in config["runtimes"]}
        if missing:
            raise SystemExit(f"Unknown runtime(s): {', '.join(sorted(missing))}")

    results_dir = Path(config["results_dir"])
    if should_slug_results_dir(args):
        results_dir = unique_results_dir(results_dir, results_slug(config, args.config, args.only_runtime))
        config["results_dir"] = str(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {results_dir}", flush=True)
    (results_dir / "environment.json").write_text(
        json.dumps(environment(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (results_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if config.get("preflight"):
        run_preflight(config, results_dir)
        return 0

    all_summaries: list[dict[str, Any]] = []
    configured_runtimes = [apply_preflight_settings(Runtime(**item), config) for item in config["runtimes"]]
    try:
        for runtime in configured_runtimes:
            print(f"Running {runtime.name} at {runtime.endpoint}", flush=True)
            summaries = run_runtime(runtime, config, results_dir)
            all_summaries.extend(summaries)
    except KeyboardInterrupt:
        print("Interrupted; cleaning up benchmark-owned runtime processes", flush=True)
        for runtime in configured_runtimes:
            if not runtime.allow_existing:
                cleanup_runtime_processes(runtime)
        raise

    write_summary_csv(results_dir / "summary.csv", all_summaries)
    report_html(results_dir / "report.html", all_summaries, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
