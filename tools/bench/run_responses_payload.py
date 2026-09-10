#!/usr/bin/env python3
"""Measure one recorded Responses prompt; never execute returned tool calls."""
import argparse
import hashlib
import json
from run_responses_cache import request
from urllib.error import HTTPError

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("payload")
parser.add_argument("--base-url", default="http://127.0.0.1:8321/v1")
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--rounds", type=int, default=1,
                    help="simulate tool results for subsequent continuation requests")
parser.add_argument("--min-hit", type=float, default=0)
args = parser.parse_args()
with open(args.payload) as source:
    body = json.load(source)
body.update(stream=True, store=False, max_output_tokens=args.max_tokens,
            temperature=0.6)
# Responses does not expose seed/presence-penalty overrides. Use NINFER_SEED
# on the launcher for repeatable sampling; its default presence penalty is 1.
if isinstance(body["input"], str):
    body["input"] = [dict(role="user", content=body["input"])]
for round_index in range(args.rounds):
    try:
        terminal, items, metrics = request(args.base_url, body)
    except HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error
    semantic_output = []
    calls = []
    for item in terminal["output"]:
        if item["type"] == "reasoning":
            semantic_output.append(["reasoning", item.get("content") or item.get("summary")])
        elif item["type"] == "message":
            semantic_output.append(["text", item["content"]])
        elif item["type"] == "function_call":
            semantic_output.append(["call", item["name"], json.loads(item["arguments"])])
            calls.append(item)
    digest = hashlib.sha256(json.dumps(semantic_output, sort_keys=True).encode()).hexdigest()
    first, wall, generated = metrics["ttft"], metrics["wall"], metrics["generated"]
    print(json.dumps(dict(round=round_index, **metrics,
                          client_decode_tok_s=(generated - 1) / (wall - first) if first else None,
                          output_sha256=digest, status=terminal["status"])), flush=True)
    if round_index and metrics["hit"] < args.min_hit:
        raise AssertionError(f"Cache hit {metrics['hit']} below {args.min_hit}")
    if not calls or terminal["status"] != "completed":
        break
    for item in items:
        if item["type"] == "function_call":
            item["arguments"] = json.dumps(json.loads(item["arguments"]), ensure_ascii=False, separators=(",", ":"))
    body["input"] += items + [dict(type="function_call_output", call_id=call["call_id"],
                                  output="Done. Continue with one small read-only verification tool call.")
                              for call in calls]
