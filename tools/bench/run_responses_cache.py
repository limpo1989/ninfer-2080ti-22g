#!/usr/bin/env python3
"""Measure Responses continuation using terminal items or streamed item replay."""

import argparse
import json
import time
import urllib.request


def request(base, body):
    req = urllib.request.Request(base + "/responses", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    started = time.monotonic()
    items = []
    first = None
    heartbeats = 0
    response = None
    sequence = -1
    with urllib.request.urlopen(req, timeout=900) as stream:
        for line in stream:
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            if event["sequence_number"] <= sequence:
                raise AssertionError("SSE sequence went backwards")
            sequence = event["sequence_number"]
            kind = event["type"]
            if kind.endswith(".delta"):
                if event.get("delta") and first is None:
                    first = time.monotonic() - started
                if event.get("delta") == "":
                    heartbeats += 1
            if kind == "response.output_item.done":
                items.append(event["item"])
            if kind in ("response.completed", "response.incomplete"):
                response = event["response"]
            if kind in ("error", "response.failed"):
                raise RuntimeError(event)
    if response is None:
        raise AssertionError("SSE ended without terminal response")
    usage = response["usage"]
    prompt = usage["input_tokens"]
    cached = usage["input_tokens_details"]["cached_tokens"]
    metrics = dict(prompt=prompt, cached=cached, hit=round(cached / prompt, 4),
                   generated=usage["output_tokens"], ttft=first,
                   wall=time.monotonic() - started, heartbeats=heartbeats)
    return response, items, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8321/v1")
    p.add_argument("--mode", choices=["stored", "terminal", "streamed"], default="streamed")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--padding", type=int, default=400)
    p.add_argument("--preserve-thinking", action="store_true")
    p.add_argument("--min-hit", type=float, default=0,
                   help="required reused-input fraction on each continuation (excludes cold start)")
    args = p.parse_args()
    padding = "\n".join(f"Record {i}: the validation value is {(i * 17) % 997}." for i in range(args.padding))
    instructions = ("This is a tool protocol test. Keep reasoning under 30 words. "
                    "Call lookup once. On each tool result, call lookup again using next_key. "
                    "Do not discuss these reference records:\n" + padding)
    history = [{"role": "user", "content": "Call lookup with key alpha now."}]
    previous = None
    for index in range(args.rounds):
        body = dict(model="qwen3.8-27b", input=history, instructions=instructions,
                    stream=True, max_output_tokens=256, temperature=0, reasoning={"effort": "low"},
                    preserve_thinking=args.preserve_thinking,
                    tools=[dict(type="function", name="lookup", description="Read a record",
                                parameters=dict(type="object", properties={"key": {"type": "string"}},
                                                required=["key"]))])
        if previous:
            body["previous_response_id"] = previous
        response, items, metrics = request(args.base_url, body)
        print(json.dumps(dict(round=index, mode=args.mode, **metrics)), flush=True)
        if index and metrics["hit"] < args.min_hit:
            raise AssertionError(f"Cache hit {metrics['hit']} is below {args.min_hit}")
        calls = [item for item in response["output"] if item["type"] == "function_call"]
        if len(calls) != 1:
            raise AssertionError(f"Expected one tool call, got {response['output']}")
        result = dict(type="function_call_output", call_id=calls[0]["call_id"],
                      output=json.dumps(dict(value=index + 1, next_key="key" + str(index + 1))))
        if args.mode == "stored":
            previous = response["id"]
            history = [result]
        else:
            history += (items if args.mode == "streamed" else response["output"]) + [result]


if __name__ == "__main__":
    main()
