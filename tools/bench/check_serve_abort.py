#!/usr/bin/env python3
"""Abort a cold Responses prefill and check the local service releases it."""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8321/v1")
    p.add_argument("--log", type=Path, required=True)
    args = p.parse_args()
    offset = args.log.stat().st_size
    prompt = "Cancellation probe. " + "alpha beta gamma delta omega " * 2500
    body = dict(model="qwen3.8-27b", input=prompt, max_output_tokens=256, stream=True,
                store=False, reasoning={"effort": "low"})
    req = urllib.request.Request(args.base_url + "/responses", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Connection": "close"})
    stream = urllib.request.urlopen(req, timeout=30)
    while True:
        line = stream.readline()
        if not line:
            raise AssertionError("No SSE start event")
        if line.startswith(b"data: "):
            break
    time.sleep(1)
    stream.close()
    closed = time.monotonic()
    while time.monotonic() - closed < 30:
        with args.log.open() as log:
            log.seek(offset)
            recent = log.read()
        submission = re.search(r"\[req (\d+)\].*submitted", recent)
        if submission:
            req_id = submission.group(1)
            if re.search(r"\[req " + req_id + r"\] error client disconnected", recent):
                print(json.dumps(dict(request_id=int(req_id), cancelled=True,
                                      cancellation_seconds=time.monotonic() - closed)))
                return
            if re.search(r"\[req " + req_id + r"\] done ", recent):
                raise AssertionError("Request completed generation instead of being cancelled")
        time.sleep(0.25)
    raise AssertionError("Request was not cancelled within 30 seconds")


if __name__ == "__main__":
    main()
