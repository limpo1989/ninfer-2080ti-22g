#!/usr/bin/env python3
"""Render a chat template with Python's jinja2, as the reference side of the
differential test described in docs/custom-chat-templates.md.

The environment deliberately mirrors the one transformers builds for chat
templates (trim_blocks and lstrip_blocks on, `tojson` using Python's default
separators, `raise_exception` available), because that is the rendering the
model's training data was produced with.

    reference_render.py <template.jinja> <context.json>
"""

import json
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment


def raise_exception(message):
    raise RuntimeError(message)


def tojson(value, ensure_ascii=False, indent=None):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: reference_render.py <template.jinja> <context.json>\n")
        return 2
    with open(sys.argv[1], encoding="utf-8") as handle:
        source = handle.read()
    with open(sys.argv[2], encoding="utf-8") as handle:
        context = json.load(handle)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    sys.stdout.write(env.from_string(source).render(**context))
    return 0


if __name__ == "__main__":
    sys.exit(main())
