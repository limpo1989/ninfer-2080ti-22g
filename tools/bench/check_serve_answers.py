#!/usr/bin/env python3
"""Small deterministic behavioral check for a local Responses service."""
import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8321/v1")
    args = parser.parse_args()
    cases = [
        ("Compute 37 * 23. Reply only with the integer.", "851"),
        ("Compute 144 / 12 + 19. Reply only with the integer.", "31"),
        ("Sort 9, 2, 7, 2, -1 ascending. Reply only with a JSON array.", [-1, 2, 2, 7, 9]),
        ("Evaluate Python: len(set([1, 1, 2, 3, 3])). Reply only with the integer.", "3"),
        ("Extract the value of port from {\"host\":\"local\",\"port\":8321}. Reply only with the integer.", "8321"),
        ("Which is larger, 9.11 or 9.9? Reply only with the larger number.", "9.9"),
    ]
    failures = 0
    for prompt, expected in cases:
        body = dict(model="qwen3.8-27b", input=prompt, reasoning={"effort": "none"},
                    temperature=0, max_output_tokens=64, store=False)
        req = urllib.request.Request(args.base_url + "/responses", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.load(response)
        text = "".join(part["text"] for item in result["output"] if item["type"] == "message"
                       for part in item["content"]).strip()
        actual = text
        if isinstance(expected, list):
            try:
                actual = json.loads(text)
            except ValueError:
                pass
        passed = actual == expected
        failures += not passed
        print(json.dumps(dict(prompt=prompt, actual=actual, expected=expected, passed=passed)), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
