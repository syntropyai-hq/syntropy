# SPDX-License-Identifier: MIT OR Apache-2.0
"""Records the chat-template rendering goldens with the reference engine (PROD-E3-T10).

The reference engine is the one model publishers render chat templates with: the
`transformers` library's `render_jinja_template`, which runs Jinja2 in a sandbox with
`trim_blocks`, `lstrip_blocks`, `raise_exception`, `strftime_now` and its own `tojson`.
`strftime_now` reads the clock there, so this script fixes its date first.

Usage, from this directory:

    python3 record.py                                   # the templates written for the tests
    python3 record.py --real <fixture cache>            # the M0 families' shipped template

`--real` reads `qwen3-0.6b/tokenizer_config.json` from the cache `cargo xtask fixtures fetch`
fills (objects named by BLAKE3; the name is taken from testdata/fixtures.toml) and writes
`expected-real-m0.json`. Both M0 checkpoints ship that file byte for byte.
"""

import argparse
import datetime as _dt
import json
import pathlib
import sys

import jinja2
import transformers
import transformers.utils.chat_template_utils as ctu

HERE = pathlib.Path(__file__).resolve().parent
DATE = _dt.datetime(2026, 11, 7, 12, 0, 0)


class _FixedClock(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return DATE


ctu.datetime = _FixedClock

# What each template is rendered with besides the conversation.
SYNTHETIC = {
    "qwen3-shaped": {"file": "qwen3-shaped.jinja", "kwargs": {}},
    "llama3-shaped": {
        "file": "llama3-shaped.jinja",
        "kwargs": {"bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>"},
    },
}


def render(template, conv, kwargs):
    extra = dict(kwargs)
    if "enable_thinking" in conv:
        extra["enable_thinking"] = conv["enable_thinking"]
    try:
        text = ctu.render_jinja_template(
            conversations=[conv["messages"]],
            chat_template=template,
            add_generation_prompt=conv["add_generation_prompt"],
            **extra,
        )
        if isinstance(text, tuple):
            text = text[0]
        if isinstance(text, list):
            text = text[0]
        return {"id": conv["id"], "rendered": text}
    except jinja2.exceptions.TemplateError as e:
        return {"id": conv["id"], "raised": str(e)}


def record(templates):
    convs = json.loads((HERE / "conversations.json").read_text("utf-8"))["conversations"]
    out = {
        "about": (
            "Chat-template renderings recorded by record.py with transformers "
            f"{transformers.__version__} and Jinja2 {jinja2.__version__}, "
            f"strftime_now fixed at {DATE.date().isoformat()} (PROD-E3-T10)."
        ),
        "date": DATE.date().isoformat(),
        "templates": [],
    }
    for name, (source, kwargs) in templates.items():
        out["templates"].append(
            {
                "name": name,
                "kwargs": kwargs,
                "cases": [render(source, c, kwargs) for c in convs],
            }
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", metavar="CACHE")
    args = p.parse_args()
    if args.real:
        manifest = (HERE / "../../../../../testdata/fixtures.toml").read_text("utf-8")
        name = 'name    = "qwen3-0.6b/tokenizer_config.json"'
        block = manifest.split(name, 1)[1].split("[[fixture]]", 1)[0]
        blake3 = block.split('blake3  = "', 1)[1].split('"', 1)[0]
        config = json.loads((pathlib.Path(args.real) / blake3).read_text("utf-8"))
        templates = {"qwen3": (config["chat_template"], {})}
        target = HERE / "expected-real-m0.json"
    else:
        templates = {
            n: ((HERE / t["file"]).read_text("utf-8"), t["kwargs"]) for n, t in SYNTHETIC.items()
        }
        target = HERE / "expected.json"
    data = record(templates)
    target.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(f"wrote {target}", file=sys.stderr)


if __name__ == "__main__":
    main()
