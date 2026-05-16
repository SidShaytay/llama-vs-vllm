from __future__ import annotations

import json
import socket
import sys
import textwrap
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import bench


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class RuntimeHealthTests(unittest.TestCase):
    def test_summarize_rows_reports_aggregate_generation_throughput(self) -> None:
        rows = [
            {
                "runtime": "fake-runtime",
                "concurrency": 2,
                "phase": "measure",
                "ok": True,
                "duration_s": 2.0,
                "request_start_s": 10.0,
                "request_end_s": 12.0,
                "ttft_s": 0.1,
                "itl_samples_s": [],
                "approx_output_tokens": 20,
                "requested_prompt_tokens": 100,
            },
            {
                "runtime": "fake-runtime",
                "concurrency": 2,
                "phase": "measure",
                "ok": True,
                "duration_s": 2.0,
                "request_start_s": 10.5,
                "request_end_s": 12.5,
                "ttft_s": 0.1,
                "itl_samples_s": [],
                "approx_output_tokens": 20,
                "requested_prompt_tokens": 100,
            },
        ]

        summary = bench.summarize_rows(rows)

        self.assertEqual(summary["gen_tok_s"], 10.0)
        self.assertEqual(summary["aggregate_gen_tok_s"], 16.0)
        self.assertEqual(summary["measure_wall_s"], 2.5)

    def test_endpoint_health_reports_connection_refused(self) -> None:
        runtime = bench.Runtime(
            name="fake-runtime",
            endpoint=f"http://127.0.0.1:{free_port()}/v1",
            model="fake-model",
        )

        healthy, error = bench.endpoint_health(runtime, timeout_s=1)

        self.assertFalse(healthy)
        self.assertIsNotNone(error)

    def test_worker_session_supports_fixed_duration_windows(self) -> None:
        runtime = bench.Runtime(
            name="fake-runtime",
            endpoint="http://127.0.0.1:1/v1",
            model="fake-model",
        )
        config = {
            **bench.DEFAULT_CONFIG,
            "warmup_duration_s": 0.01,
            "measure_duration_s": 0.03,
            "session_turns": 1,
            "think_time_ms": 0,
            "request_timeout_s": 1,
            "traces": {
                "tool_call": {"weight": 1.0, "prompt": [1, 1], "output": [1, 1]},
            },
        }

        def fake_request_chat(*args, **kwargs):
            time.sleep(0.005)
            return {
                "ok": True,
                "duration_s": 0.005,
                "ttft_s": 0.001,
                "chunks": 1,
                "approx_output_tokens": 1,
                "itl_samples_s": [],
            }

        with mock.patch.object(bench, "request_chat", side_effect=fake_request_chat):
            rows = bench.worker_session(
                runtime,
                config,
                concurrency=1,
                worker_id=0,
                measured_target=None,
                warmup_sessions=0,
            )

        phases = [row["phase"] for row in rows]
        self.assertIn("warmup", phases)
        self.assertIn("measure", phases)
        self.assertGreaterEqual(phases.count("measure"), 1)

    def test_models_ready_but_chat_unusable_is_health_failure(self) -> None:
        port = free_port()
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            server_script = tmp_path / "fake_models_only_server.py"
            server_script.write_text(
                textwrap.dedent(
                    f"""
                    from http.server import BaseHTTPRequestHandler, HTTPServer

                    class Handler(BaseHTTPRequestHandler):
                        def log_message(self, format, *args):
                            pass

                        def do_GET(self):
                            if self.path == "/v1/models":
                                self.send_response(200)
                                self.send_header("Content-Type", "application/json")
                                self.end_headers()
                                self.wfile.write(b'{{"data":[{{"id":"fake-model"}}]}}')
                                print("models endpoint returned healthy", flush=True)
                                self.server.should_stop = True
                            else:
                                self.send_response(404)
                                self.end_headers()

                    server = HTTPServer(("127.0.0.1", {port}), Handler)
                    server.should_stop = False
                    while not server.should_stop:
                        server.handle_request()
                    print("fake server exiting after models-only readiness", flush=True)
                    """
                ),
                encoding="utf-8",
            )

            runtime = bench.Runtime(
                name="fake-vllm-exit",
                endpoint=f"http://127.0.0.1:{port}/v1",
                model="fake-model",
                command=[sys.executable, str(server_script)],
                launch_timeout_s=5,
                readiness_chat_timeout_s=1,
            )
            config = {
                **bench.DEFAULT_CONFIG,
                "primary_concurrency": [1],
                "overload_concurrency": [],
                "min_samples": 1,
                "warmup_sessions": 0,
                "session_turns": 1,
                "request_timeout_s": 1,
            }

            summaries = bench.run_runtime(runtime, config, tmp_path / "results")

            self.assertEqual(summaries[0]["requests"], 0)
            raw_rows = [
                json.loads(line)
                for line in (tmp_path / "results" / runtime.name / "raw.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(raw_rows[0]["phase"], "healthcheck")
            self.assertFalse(raw_rows[0]["ok"])
            self.assertEqual(raw_rows[0]["server_exit_code"], 0)
            self.assertIn("server.log", raw_rows[0]["server_log_path"])
            self.assertTrue(
                any("fake server exiting" in line for line in raw_rows[0]["server_log_tail"])
            )
            self.assertIn("chat completion probe failed", raw_rows[0]["error"])


if __name__ == "__main__":
    unittest.main()
