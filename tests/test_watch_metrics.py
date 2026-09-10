import argparse
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from deploy.watch_ninfer import (GpuSnapshot, LogReader, ProcessCpu, RULE, WatchState, frame, gpu_status, launch_lines,
                                 parse_request, process_alive, row_lines, service_pid)


DONE = ("[2026-09-10 12:00:00.001] [info] ninfer-serve: [req 7] done finish=output_limit "
        "prompt=25000 gen=1000 think=800 cache=24500 reuse=append_frontier ttft=4800ms "
        "queue=3.00s prefill_time=1.50s prefill=333.3tok/s decode=25.0tok/s "
        "wall=44.76s speculative=mtp 3.10tok/round (70.0%)")


class WatchMetricsTest(unittest.TestCase):
    def test_completed_request_keeps_full_first_token_wait_and_separate_rates(self):
        row = parse_request(DONE)
        self.assertEqual(row.values, ["12:00:00", "25000/500", "3.00s", "1.50s", "4.8s",
                                      "333.3", "25.0", "1000", "98%", "70.0%", "44.76s", "max_tokens"])
        self.assertIn("Think=800", row.detail)
        self.assertNotIn("Snapshot", row.detail)
        self.assertNotIn("Think=800", "\n".join(row_lines(row, 160, False)))
        self.assertIn("Think=800", "\n".join(row_lines(row, 160, True)))

    def test_old_logs_do_not_invent_phase_durations_or_thinking_counts(self):
        old = DONE.replace("queue=3.00s ", "").replace("prefill_time=1.50s ", "").replace("think=800 ", "")
        row = parse_request(old)
        self.assertEqual(row.values[2:5], ["-", "-", "4.8s"])
        self.assertIn("Think=-", row.detail)

    def test_queue_timeout_and_preparation_rejection_are_visible(self):
        error = parse_request("[12:00:01] [req 9] error inference request expired while waiting for admission")
        self.assertEqual(error.values[-1], "queue_timeout")
        self.assertTrue(all(value == "-" for value in error.values[1:-1]))
        self.assertIn("expired", "\n".join(row_lines(error, 160, False)))
        rejected = parse_request("[12:00:02] [req 10] rejected phase=prepare status=400 "
                                 "code=context_length_exceeded message=prompt exceeds context")
        lines = row_lines(rejected, 160, False)
        self.assertEqual(len(lines[0]), len(RULE))
        self.assertIn("prompt exceeds context", "\n".join(lines))

    def test_cancellation_has_no_first_token_when_nothing_was_generated(self):
        row = parse_request(DONE.replace("finish=output_limit", "finish=cancelled").replace("gen=1000", "gen=0"))
        self.assertEqual(row.values[4], "-")
        self.assertEqual(row.values[-1], "cancelled")

    def test_live_counts_include_idle_transition_without_fake_request_rows(self):
        state = WatchState()
        state.feed("throughput interval=5.000s running=2 waiting=3")
        self.assertEqual((state.running, state.waiting), ("2", "3"))
        state.feed(DONE)
        state.feed("throughput interval=5.000s running=0 waiting=0")
        self.assertEqual((state.running, state.waiting), ("0", "0"))
        self.assertEqual(len(state.rows), 1)
        wide = frame(state, GpuSnapshot(metrics="GPU 0%"), 2, 160, 24, False, True)
        self.assertIn("Running 0/2 | Queued 0", wide)
        self.assertIn("Prefill/s", wide)
        narrow = frame(state, GpuSnapshot(metrics="GPU 0%"), 2, 80, 24, True, True)
        self.assertTrue(all(len(line) <= 80 for line in narrow.splitlines()))
        self.assertIn("Input/New=25000/500", narrow)

    def test_log_follower_handles_partial_lines_truncation_and_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "serve.log"
            path.write_text("first\npart")
            reader = LogReader(path)
            try:
                self.assertEqual(reader.read(), (True, ["first"]))
                with path.open("a") as log:
                    log.write("ial\n")
                self.assertEqual(reader.read(), (False, ["partial"]))
                path.write_text("new\n")
                self.assertEqual(reader.read(), (True, ["new"]))
                replacement = Path(directory) / "replacement"
                replacement.write_text("rotated\n")
                replacement.replace(path)
                self.assertEqual(reader.read(), (True, ["rotated"]))
            finally:
                reader.close()

    def test_missing_gpu_tool_does_not_stop_monitoring(self):
        with patch("deploy.watch_ninfer.subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(gpu_status(), GpuSnapshot())

    def test_startup_information_stays_above_live_status_and_empty_key_is_hidden(self):
        args = argparse.Namespace(server_url="http://0.0.0.0:8321/v1", api_key="fixture-key",
                                  model="/models/qwen.ninfer", max_context=245760,
                                  kv_capacity=245760, max_output=131072, draft_tokens=3,
                                  prefill_chunk=1024, max_concurrency=2, tool_replay_cache_mib=1024)
        startup = tuple(launch_lines(args))
        view = frame(WatchState(), GpuSnapshot(metrics="GPU 0%"), 2, 80, 30, False, False, startup)
        self.assertLess(view.index("API http"), view.index("Service stopped"))
        self.assertIn("Key fixture-key", view)
        self.assertIn("Max output 131,072", view)
        self.assertTrue(all(len(line) <= 80 for line in view.splitlines()))
        args.api_key = ""
        self.assertNotIn("Key", "\n".join(launch_lines(args)))

    def test_gpu_model_and_fan_remain_separate_from_live_utilization(self):
        reply = subprocess.CompletedProcess([], 0, 'NVIDIA GeForce RTX 2080 Ti, 90, 21616, 22528, 53, 49, 245\n', '')
        with patch("deploy.watch_ninfer.subprocess.run", return_value=reply):
            gpu = gpu_status()
        self.assertEqual(gpu.model, "NVIDIA GeForce RTX 2080 Ti")
        self.assertIn("GPU 90%", gpu.metrics)
        self.assertIn("FAN 53%", gpu.metrics)
        view = frame(WatchState(), gpu, 2, 160, 30, False, True, cpu="CPU 150%")
        self.assertIn("GPU model NVIDIA GeForce RTX 2080 Ti", view)
        self.assertIn("CPU 150% | GPU 90%", view)

    def test_process_cpu_reports_recent_total_across_cores_and_resets_on_restart(self):
        def stat(user, system, start):
            fields = ["0"] * 20
            fields[0], fields[11], fields[12], fields[19] = "S", str(user), str(system), str(start)
            return "123 (inference worker) " + " ".join(fields)
        with patch("deploy.watch_ninfer.os.sysconf", return_value=100):
            sampler = ProcessCpu()
        with patch("deploy.watch_ninfer.Path.read_text", side_effect=[stat(100, 50, 900), stat(350, 100, 900), stat(1, 1, 1000)]), \
             patch("deploy.watch_ninfer.time.monotonic", side_effect=[10.0, 12.0, 15.0]):
            self.assertEqual(sampler.sample(123), "CPU -")
            self.assertEqual(sampler.sample(123), "CPU 150%")
            self.assertEqual(sampler.sample(123), "CPU -")
        self.assertEqual(sampler.sample(None), "CPU -")

    def test_pid_file_tracks_service_start_after_watch_opens_offline(self):
        self.assertFalse(process_alive(None))
        self.assertFalse(process_alive(0))
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "service.pid"
            self.assertFalse(process_alive(service_pid(None, pid_file)))
            pid_file.write_text(str(os.getpid()))
            self.assertTrue(process_alive(service_pid(None, pid_file)))


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux deployment entry point")
class WatchLauncherTest(unittest.TestCase):
    script = Path(__file__).resolve().parents[1] / "deploy/restart-ninfer.sh"

    def fixture(self, root):
        fake_bin = root / "bin"
        fake_bin.mkdir()
        for name, body in {"pgrep": "exit 1\n", "curl": "exit 0\n",
                           "nvidia-smi": "printf '0\\n'\n"}.items():
            path = fake_bin / name
            path.write_text("#!/bin/sh\n" + body)
            path.chmod(0o755)
        model = root / "model.ninfer"
        model.touch()
        log = root / "serve.log"
        log.write_text(DONE + "\n")
        return dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
                    NINFER_BIN=str(root / "fake-server"), NINFER_MODEL=str(model),
                    NINFER_LOG=str(log), NINFER_PID_FILE=str(root / "server.pid"),
                    NINFER_GPU_POWER_LIMIT_W="")

    def test_watch_prints_offline_snapshot_instead_of_exiting_silently(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.fixture(Path(directory))
            for command in ("watch", "--watch"):
                with self.subTest(command=command):
                    result = subprocess.run(["bash", str(self.script), command, "--once"],
                                            env=env, capture_output=True, text=True, timeout=5)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("Service stopped", result.stdout)
                    self.assertIn("25000/500", result.stdout)
                    self.assertIn("Replay cache 1,024 MiB", result.stdout)

    def test_ctrl_c_after_restart_only_exits_monitor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = self.fixture(root)
            server = Path(env["NINFER_BIN"])
            ready = root / "handler-ready"
            arguments = root / "server-args.json"
            server.write_text(
                f"#!{sys.executable}\nimport json, signal, sys, time\nfrom pathlib import Path\n"
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(90))\n"
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
                f"Path({str(arguments)!r}).write_text(json.dumps(sys.argv[1:]))\n"
                f"Path({str(ready)!r}).touch()\nwhile True: time.sleep(1)\n")
            server.chmod(0o755)
            process = subprocess.Popen(["bash", str(self.script), "restart", "--tool-replay-cache-mib", "2048"], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            server_pid = None
            output = b""
            try:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    if select.select([process.stdout], [], [], 0.1)[0]:
                        output += os.read(process.stdout.fileno(), 65536)
                    if b"NInfer live request metrics" in output and ready.exists():
                        break
                    if process.poll() is not None:
                        self.fail(output.decode(errors="replace"))
                self.assertIn(b"NInfer live request metrics", output)
                server_pid = int(Path(env["NINFER_PID_FILE"]).read_text())
                server_args = json.loads(arguments.read_text())
                self.assertEqual(server_args[server_args.index("--tool-replay-cache-mib") + 1], "2048")
                self.assertIn(b"Replay cache 2,048 MiB", output)
                self.assertNotEqual(os.getpgid(server_pid), process.pid)
                os.killpg(process.pid, signal.SIGINT)
                tail, _ = process.communicate(timeout=5)
                output += tail
                self.assertIn(b"Monitoring stopped", output)
                self.assertTrue(process_alive(server_pid))
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.communicate(timeout=5)
                if server_pid is None:
                    try:
                        server_pid = int(Path(env["NINFER_PID_FILE"]).read_text())
                    except (OSError, ValueError):
                        pass
                if process_alive(server_pid):
                    os.kill(server_pid, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
