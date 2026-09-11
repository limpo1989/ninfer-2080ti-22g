#!/usr/bin/env python3
"""Read-only NInfer console-log dashboard; uses only the Python standard library."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
import time


FIELDS = re.compile(r"\b([a-z_]+)=([^\s]+)")
REQUEST = re.compile(r"\[req (\d+)\] (done|error|rejected)\b(.*)")
COLUMNS = (
    ("Time", 8), ("Input/New", 13), ("Queue", 7), ("Prefill", 7),
    ("TTFB", 7), ("Prefill/s", 9), ("Decode/s", 8), ("Output", 6),
    ("Cache hit", 9), ("MTP", 6), ("Wall", 8), ("Finish", 13),
)
RULE = "+" + "+".join("-" * (width + 2) for _, width in COLUMNS) + "+"
FINISH_NAMES = {"output_limit": "max_tokens", "tool_calls": "tool_calls"}
COLORS = {
    "muted": "90", "title": "1;36", "heading": "1;34", "accent": "36",
    "good": "32", "warning": "33", "error": "1;31", "mtp": "35", "rate": "1;36",
}


def paint(text: str, tone: str, color: bool) -> str:
    return f"\033[{COLORS[tone]}m{text}\033[0m" if color and tone else text


def metric_tone(name: str, value: str, error: bool = False) -> str:
    if value == "-":
        return "muted"
    if name == "Finish":
        if error or value in {"error", "rejected", "queue_timeout", "queue_full"}:
            return "error"
        if value in {"cancelled", "max_tokens", "context_limit"}:
            return "warning"
        return "accent" if value == "tool_calls" else "good"
    if name == "Queue":
        return "warning" if (numeric(value, "s") or 0) > 0 else "muted"
    if name in {"Prefill/s", "Decode/s"}:
        return "rate"
    if name == "Cache hit":
        return "good" if (numeric(value, "%") or 0) >= 95 else ""
    if name == "MTP":
        return "mtp"
    if name in {"Prefill", "TTFB"}:
        return "accent"
    return "muted" if name == "Time" else ""


def color_status(line: str, color: bool) -> str:
    parts = []
    for part in line.split(" | "):
        tone = ""
        if part.startswith("Service stopped"):
            tone = "error"
        elif part.startswith("Running"):
            tone = "good"
        elif part.startswith("Queued"):
            tone = "warning" if (numeric(part.removeprefix("Queued ")) or 0) > 0 else "muted"
        elif part == "GPU unavailable":
            tone = "muted"
        elif part.startswith(("CPU ", "GPU ", "VRAM ", "FAN ")):
            tone = "accent"
        elif part.endswith(" C"):
            temperature = numeric(part, " C")
            if temperature is not None:
                tone = "error" if temperature >= 80 else "warning" if temperature >= 75 else "good"
        parts.append(paint(part, tone, color))
    return paint(" | ", "muted", color).join(parts)


def color_config(line: str, color: bool) -> str:
    parts = []
    for part in line.split(" | "):
        for label in ("API", "Key", "Model", "Context", "KV", "Max output", "MTP",
                      "Turbo", "Concurrency", "Prefill chunk", "Replay cache", "State cache",
                      "GPU model"):
            if part.startswith(label + " "):
                value = part[len(label) + 1:]
                tone = "accent" if label == "API" else "good" if label == "Turbo" and value == "ON" else ""
                part = paint(label + " ", "muted", color) + paint(value, tone, color)
                break
        else:
            if part == "KVarN":
                part = paint(part, "accent", color)
        parts.append(part)
    return paint(" | ", "muted", color).join(parts)


def clean(text: str) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return "".join(c if c.isprintable() else " " for c in text)


def numeric(value: str | None, suffix: str = "") -> float | None:
    if value is None or (suffix and not value.endswith(suffix)):
        return None
    try:
        number = float(value[:-len(suffix)] if suffix else value)
        return number if math.isfinite(number) and number >= 0 else None
    except ValueError:
        return None


def duration(value: str | None) -> str:
    seconds = numeric(value, "s")
    return "-" if seconds is None else f"{seconds:.2f}s"


def rate(value: str | None) -> str:
    tokens = numeric(value, "tok/s")
    return "-" if tokens is None else f"{tokens:.1f}"


@dataclass
class Row:
    request_id: str
    values: list[str]
    detail: str = ""
    error: str = ""


def parse_request(line: str) -> Row | None:
    match = REQUEST.search(line)
    if match is None:
        return None
    request_id, event, body = match.groups()
    fields = dict(FIELDS.findall(body))
    stamp = re.search(r"\b\d{2}:\d{2}:\d{2}\b", line)
    timestamp = stamp.group() if stamp else "-"
    if event != "done":
        message = body.partition("message=")[2] if event == "rejected" else body.strip()
        finish = fields.get("code", "rejected") if event == "rejected" else "error"
        if "inference request expired" in message:
            finish = "queue_timeout"
        elif "inference request queue is full" in message:
            finish = "queue_full"
        return Row(request_id, [timestamp, *(["-"] * 10), finish],
                   error=clean(message or body.strip()))

    prompt = numeric(fields.get("prompt"))
    cached = numeric(fields.get("cache"))
    generated = numeric(fields.get("gen"))
    input_new = "-" if prompt is None else f"{int(prompt)}/-"
    hit = "-"
    if prompt is not None and cached is not None:
        input_new = f"{int(prompt)}/{max(0, int(prompt - cached))}"
        if prompt > 0:
            hit = f"{min(100, int(cached * 100 / prompt))}%"
    first_ms = numeric(fields.get("ttft"), "ms")
    ttfb = f"{first_ms / 1000:.1f}s" if first_ms is not None and generated else "-"
    mtp = re.search(r"speculative=\S+ \S+ \(([\d.]+%)\)", body)
    finish = fields.get("finish", "-")
    values = [
        timestamp, input_new, duration(fields.get("queue")),
        duration(fields.get("prefill_time")), ttfb, rate(fields.get("prefill")),
        rate(fields.get("decode")), "-" if generated is None else str(int(generated)),
        hit, mtp.group(1) if mtp else "-", duration(fields.get("wall")),
        FINISH_NAMES.get(finish, finish),
    ]
    detail = (f"#{request_id} Think={fields.get('think', '-')} tokens | "
              f"Reuse={fields.get('reuse', '-')} | Tool calls={fields.get('tool_calls', '0')} | "
              f"State={fields.get('state_source', '-')} Restore={fields.get('restore', '-')}")
    return Row(request_id, values, detail)


class WatchState:
    def __init__(self) -> None:
        self.rows: deque[Row] = deque(maxlen=100)
        self.running: str = "-"
        self.waiting: str = "-"

    def feed(self, line: str) -> Row | None:
        if "throughput interval=" in line:
            fields = dict(FIELDS.findall(line))
            self.running = fields.get("running", "-")
            self.waiting = fields.get("waiting", "-")
        row = parse_request(line)
        if row:
            self.rows.append(row)
        return row


class LogReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = None
        self.identity = None
        self.partial = b""

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None

    def read(self) -> tuple[bool, list[str]]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return False, []
        identity = (stat.st_dev, stat.st_ino)
        reset = self.file is None or identity != self.identity or stat.st_size < self.file.tell()
        if reset:
            self.close()
            self.file = self.path.open("rb")
            self.identity = identity
            self.partial = b""
            # Enough recent history to include request completions between throughput records.
            if stat.st_size > 256 * 1024:
                self.file.seek(stat.st_size - 256 * 1024)
                self.file.readline()
        chunks = (self.partial + self.file.read()).split(b"\n")
        self.partial = chunks.pop()
        return reset, [line.decode("utf-8", errors="replace") for line in chunks]


@dataclass
class GpuSnapshot:
    model: str = ""
    metrics: str = "GPU unavailable"


def gpu_status() -> GpuSnapshot:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
             "fan.speed,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        model, util, used, total, fan, temperature, power = [
            part.strip() for part in next(csv.reader([result.stdout.strip()]))]
        numbers = [numeric(value) for value in (util, used, total, fan, temperature, power)]
        def fmt(value, pattern):
            return "-" if value is None else pattern.format(value)
        metrics = (f"GPU {fmt(numbers[0], '{:.0f}%')} | "
                f"VRAM {fmt(None if numbers[1] is None else numbers[1] / 1024, '{:.1f}')}/"
                f"{fmt(None if numbers[2] is None else numbers[2] / 1024, '{:.1f}')} GiB | "
                f"FAN {fmt(numbers[3], '{:.0f}%')} | "
                f"{fmt(numbers[4], '{:.0f} C')} | {fmt(numbers[5], '{:.0f} W')}")
        return GpuSnapshot(clean(model), metrics)
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration, csv.Error):
        return GpuSnapshot()


def process_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def service_pid(pid: int | None, pid_file: Path | None) -> int | None:
    if pid_file is not None:
        try:
            pid = int(pid_file.read_text().strip())
        except (OSError, ValueError):
            pass
    return pid


class ProcessCpu:
    """Process-wide CPU ticks per wall second: one fully busy core is 100%."""
    def __init__(self) -> None:
        self.previous = None
        self.ticks_per_second = os.sysconf("SC_CLK_TCK")

    def sample(self, pid: int | None) -> str:
        if pid is None or pid <= 0:
            self.previous = None
            return "CPU -"
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            ticks = int(fields[11]) + int(fields[12])
            identity = (pid, int(fields[19]))
        except (OSError, ValueError, IndexError):
            self.previous = None
            return "CPU -"
        now = time.monotonic()
        previous = self.previous
        self.previous = (identity, ticks, now)
        if previous is None or previous[0] != identity or ticks < previous[1] or now <= previous[2]:
            return "CPU -"
        percent = 100 * (ticks - previous[1]) / self.ticks_per_second / (now - previous[2])
        return f"CPU {percent:.0f}%"


def table_line(values: list[str], color: bool = False, heading: bool = False,
               error: bool = False) -> str:
    cells = []
    for value, (name, width) in zip(values, COLUMNS):
        value = clean(value)
        tone = "heading" if heading else metric_tone(name, value, error)
        if len(value) > width:
            value = value[:width - 1] + "~"
        cells.append(paint(f"{value:>{width}}", tone, color))
    return paint("| ", "muted", color) + paint(" | ", "muted", color).join(cells) + paint(" |", "muted", color)


def row_lines(row: Row, width: int, details: bool, color: bool = False) -> list[str]:
    if width >= len(RULE):
        result = [table_line(row.values, color=color, error=bool(row.error))]
    else:
        parts = [f"{name}={value}" for (name, _), value in zip(COLUMNS[1:], row.values[1:])]
        result = textwrap.wrap(f"{row.values[0]} #{row.request_id} | " + " | ".join(parts),
                               width, subsequent_indent="  ")
        if color:
            names = "|".join(re.escape(name) for name, _ in COLUMNS[1:])
            def highlight(match):
                name, value = match.groups()
                return paint(name + "=", "muted", True) + paint(value, metric_tone(name, value, bool(row.error)), True)
            result = [re.sub(rf"({names})=([^ |]+)", highlight, line) for line in result]
    if row.error:
        result.extend(paint(line, "error", color) for line in
                      textwrap.wrap(f"  #{row.request_id} {row.error}", width, subsequent_indent="  "))
    elif details:
        result.extend(paint(line, "muted", color) for line in
                      textwrap.wrap("  " + clean(row.detail), width, subsequent_indent="  "))
    return result


def launch_lines(args: argparse.Namespace) -> list[str]:
    lines = []
    if args.server_url:
        endpoint = f"API {args.server_url}"
        if args.api_key:
            endpoint += f" | Key {args.api_key}"
        lines.append(endpoint)
    limits = []
    for label, value in (("Model", Path(args.model).stem if args.model else None),
                         ("Context", args.max_context), ("KV", args.kv_capacity),
                         ("Max output", args.max_output)):
        if value is not None:
            limits.append(f"{label} {value:,}" if isinstance(value, int) else f"{label} {value}")
    if limits:
        lines.append(" | ".join(limits))
    if args.draft_tokens is not None and args.prefill_chunk is not None:
        lines.append(f"KVarN | MTP {args.draft_tokens} | Turbo {'ON' if args.turbo else 'OFF'} | "
                     f"Concurrency {args.max_concurrency} | "
                     f"Prefill chunk {args.prefill_chunk}")
    if args.tool_replay_cache_mib is not None:
        lines.append(f"Replay cache {args.tool_replay_cache_mib:,} MiB")
    if getattr(args, "state_cache_max_mib", 0):
        lines.append(f"State cache RAM {args.state_cache_ram_mib:,} MiB / "
                     f"disk {args.state_cache_max_mib:,} MiB / "
                     f"idle {args.state_cache_idle_ms:,} ms")
    return lines


def runtime_status(state: WatchState, gpu: GpuSnapshot, capacity: int, alive: bool, cpu: str) -> str:
    service = f"Running {state.running}/{capacity} | Queued {state.waiting}" if alive else "Service stopped"
    return f"{service} | {cpu} | {gpu.metrics}"


def header(state: WatchState, gpu: GpuSnapshot, capacity: int, width: int, alive: bool,
           startup: tuple[str, ...] = (), color: bool = False, cpu: str = "CPU -") -> list[str]:
    status = runtime_status(state, gpu, capacity, alive, cpu)
    lines = [paint("NInfer live request metrics", "title", color)]
    for line in startup:
        lines.extend(color_config(part, color) for part in textwrap.wrap(clean(line), width))
    if gpu.model:
        lines.extend(color_config(part, color) for part in textwrap.wrap("GPU model " + gpu.model, width))
    lines.extend(color_status(part, color) for part in textwrap.wrap(status, width))
    lines.extend([paint("Ctrl+C exits this view; it does not stop the HTTP service.", "muted", color),
                  paint("Times: seconds | Rates: tokens/s | TTFB includes Queue and Prefill", "muted", color)])
    if width >= len(RULE):
        lines.extend([paint(RULE, "muted", color),
                      table_line([name for name, _ in COLUMNS], color=color, heading=True),
                      paint(RULE, "muted", color)])
    return lines


def frame(state: WatchState, gpu: GpuSnapshot, capacity: int, width: int, height: int,
          details: bool, alive: bool, startup: tuple[str, ...] = (), color: bool = False,
          cpu: str = "CPU -") -> str:
    lines = header(state, gpu, capacity, width, alive, startup, color, cpu)
    available = max(0, height - len(lines) - 1)
    blocks = []
    for row in reversed(state.rows):
        block = row_lines(row, width, details, color)
        if len(block) > available:
            break
        blocks.append(block)
        available -= len(block)
    for block in reversed(blocks):
        lines.extend(block)
    if not state.rows:
        lines.append(paint("Waiting for request results...", "muted", color))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--pid-file", type=Path, help="follow service restarts through its PID file")
    parser.add_argument("--server-url")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model")
    parser.add_argument("--max-context", type=int)
    parser.add_argument("--kv-capacity", type=int)
    parser.add_argument("--max-output", type=int)
    parser.add_argument("--draft-tokens", type=int)
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--tool-replay-cache-mib", type=int)
    parser.add_argument("--state-cache-dir")
    parser.add_argument("--state-cache-max-mib", type=int, default=0)
    parser.add_argument("--state-cache-ram-mib", type=int, default=4096)
    parser.add_argument("--state-cache-idle-ms", type=int, default=1000)
    parser.add_argument("--details", action="store_true", help="show thinking tokens and reuse details")
    parser.add_argument("--once", action="store_true", help="print a snapshot and exit")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                        help="terminal colors; auto respects NO_COLOR and redirected output")
    args = parser.parse_args()
    if not args.log.is_file():
        parser.error(f"Missing log: {args.log}")
    reader, state = LogReader(args.log), WatchState()
    startup = tuple(launch_lines(args))
    tty = sys.stdout.isatty() and not args.once
    color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty()
                                       and not os.environ.get("NO_COLOR"))
    gpu, cpu, next_refresh, last_frame = GpuSnapshot(), "CPU -", 0.0, ""
    cpu_sampler = ProcessCpu()
    initial = True
    if tty:
        print("\033[?1049h\033[?25l", end="", flush=True)
    try:
        while True:
            reset, lines = reader.read()
            if reset:
                state = WatchState()
            fresh = [row for line in lines if (row := state.feed(line)) is not None]
            now = time.monotonic()
            refresh = now >= next_refresh
            pid = service_pid(args.pid, args.pid_file)
            alive = process_alive(pid)
            if refresh:
                gpu = gpu_status()
                cpu = cpu_sampler.sample(pid if alive else None)
                next_refresh = now + 3.0
            width, height = shutil.get_terminal_size((160, 30))
            width = max(40, width)
            if tty or args.once:
                text = frame(state, gpu, args.max_concurrency, width,
                             10000 if args.once else height, args.details, alive, startup, color, cpu)
                if text != last_frame:
                    print(("\033[H\033[2J" if tty else "") + text, flush=True)
                    last_frame = text
            else:
                if initial or reset:
                    print("\n".join(header(state, gpu, args.max_concurrency, width, alive, startup, color, cpu)), flush=True)
                elif refresh:
                    status = runtime_status(state, gpu, args.max_concurrency, alive, cpu)
                    print("\n".join(color_status(line, color) for line in textwrap.wrap(status, width)), flush=True)
                for row in fresh:
                    print("\n".join(row_lines(row, width, args.details, color)), flush=True)
            if args.once:
                return 0
            initial = False
            time.sleep(0.2)
    except KeyboardInterrupt:
        return 0
    finally:
        reader.close()
        if tty:
            print("\033[0m\033[?25h\033[?1049l", end="", flush=True)
        if not args.once:
            print("Monitoring stopped; the HTTP service was not stopped.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
