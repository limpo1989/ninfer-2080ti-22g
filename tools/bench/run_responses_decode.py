#!/usr/bin/env python3
"""Fixed local coding workloads for comparing MTP windows on the same server."""
import argparse
import json

from run_responses_cache import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8321/v1")
    args = parser.parse_args()
    prompts = [
        "Write a Python TTL cache with an injected monotonic clock, capacity bound, and LRU eviction. Include tests for expiry and replacement. Output code only.",
        "Write a C++17 bounded queue using mutex and condition_variable. Support close, blocking push, and blocking pop, including exception-safe wakeups. Output code only.",
        "Explain why a database transaction at repeatable-read isolation can still exhibit write skew. Give a detailed worked example and a correct solution using serialization or locking.",
    ]
    for index, prompt in enumerate(prompts):
        response, _, metrics = request(args.base_url, dict(
            model="qwen3.8-27b", input=prompt, reasoning={"effort": "none"},
            stream=True, max_output_tokens=384, temperature=0, store=False))
        if metrics["ttft"] is None:
            raise AssertionError("No generated output")
        metrics["decode_observed"] = (metrics["generated"] - 1) / (metrics["wall"] - metrics["ttft"])
        metrics["output"] = "".join(part["text"] for item in response["output"] if item["type"] == "message"
                                     for part in item["content"])
        print(json.dumps(dict(case=index, **metrics)), flush=True)


if __name__ == "__main__":
    main()
