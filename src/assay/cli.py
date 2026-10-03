"""Command line interface.

Exit codes: ``0`` all contracts hold, ``1`` a contract failed (or ``diff`` found a regression),
``2`` usage or spec error, ``3`` nothing failed but some contracts are inconclusive (``--strict``).
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import checks, contracts, transforms
from ._version import __version__
from .demos import DEMOS, VARIANTS
from .diff import diff_results, load_result, render_diff
from .errors import AssayError, SpecError
from .report import (
    RENDERERS,
    render_html,
    render_json,
    render_junit,
    render_markdown,
    render_text,
)
from .results import SuiteResult
from .spec import build_model, load_spec

_FORMATS = {
    "json": render_json,
    "markdown": render_markdown,
    "html": render_html,
}


def _color(args: argparse.Namespace) -> bool:
    return not args.no_color and sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _render(result: SuiteResult, fmt: str, *, color: bool, strict: bool, examples: int) -> str:
    if fmt == "text":
        return render_text(result, color=color, examples=examples)
    if fmt == "junit":
        return render_junit(result, strict=strict)
    return _FORMATS[fmt](result)


def _write(path: str, text: str) -> None:
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot write '{path}': {exc}") from exc


# --------------------------------------------------------------------------- commands


def _cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec, load_model=not args.target)
    model = None
    if args.target:
        base = spec.path.parent.resolve() if spec.path else Path.cwd()
        model = build_model({"type": "python", "target": args.target}, base, "--target")
    suite = spec.to_suite(
        model=model, seed=args.seed, workers=args.workers, redact=True if args.redact else None
    )
    result = suite.run()

    sys.stdout.write(
        _render(result, args.format, color=_color(args), strict=args.strict, examples=args.examples)
    )
    for path, fmt in (
        (args.out_json, "json"), (args.out_junit, "junit"),
        (args.out_md, "markdown"), (args.out_html, "html"),
    ):  # fmt: skip
        if path:
            _write(
                path, _render(result, fmt, color=False, strict=args.strict, examples=args.examples)
            )
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if args.github_summary and summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(render_markdown(result) + "\n")
    return result.exit_code(strict=args.strict)


def _cmd_diff(args: argparse.Namespace) -> int:
    diffs = diff_results(load_result(args.baseline), load_result(args.candidate))
    sys.stdout.write(render_diff(diffs, color=_color(args)))
    regressed = any(d.is_regression for d in diffs)
    return 1 if regressed and not args.report_only else 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .model import import_target

    if args.name and args.name not in DEMOS:
        raise SpecError(f"unknown demo '{args.name}' (choose {', '.join(DEMOS)})")
    for name in [args.name] if args.name else list(DEMOS):
        demo = DEMOS[name]
        suite = load_spec(demo.spec, load_model=False).to_suite(
            model=import_target(demo.target(args.variant))
        )
        print(f"━━ {name} · {demo.industry} · {demo.summary}")
        print(f"   model under test: {demo.target(args.variant)}\n")
        sys.stdout.write(render_text(suite.run(), color=_color(args), examples=args.examples))
        print()
    other = "good" if args.variant == "regressed" else "regressed"
    print(
        f"Try the {other} model: assay demo {args.name or ''} --variant {other}".replace("  ", " ")
    )
    print(
        "Read the contracts: "
        + str(next(iter(DEMOS.values())).spec.parent.parent)
        + "/<demo>/assay.toml"
    )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec, resolve=False)
    print(f"{args.spec}: OK · {len(spec.contracts)} contracts · {len(spec.cases)} cases")
    return 0


def _summary(obj: Any) -> str:
    paragraph = " ".join((inspect.getdoc(obj) or "").split("\n\n")[0].split())
    paragraph = re.sub(r"\s*\([^)]*\)", "", paragraph).replace("``", "").replace("*", "")
    sentence = paragraph.split(". ")[0].rstrip(".")
    return sentence if len(sentence) <= 100 else sentence[:99] + "…"


def _cmd_list(args: argparse.Namespace) -> int:
    sections: dict[str, dict[str, Any]] = {
        "contracts": dict(contracts.REGISTRY),
        "transforms": dict(transforms.REGISTRY),
        "checks": dict(checks.REGISTRY),
    }
    for title, registry in sections.items():
        if args.what in (None, title):
            print(f"{title}:")
            width = max(map(len, registry))
            for name, obj in sorted(registry.items()):
                print(f"  {name.ljust(width)}  {_summary(obj)}")
            print()
    return 0


_INIT_MODEL = '''"""A tiny sentiment model. Swap `predict` for a call to your own system."""

import re

_POSITIVE = {"great", "love", "excellent", "good", "happy", "fantastic", "perfect"}
_NEGATIVE = {"terrible", "hate", "awful", "bad", "angry", "broken", "useless"}


def predict(text: str) -> dict:
    words = re.findall(r"[a-z']+", text.lower())
    balance = sum(w in _POSITIVE for w in words) - sum(w in _NEGATIVE for w in words)
    label = "positive" if balance > 0 else "negative" if balance < 0 else "neutral"
    return {"label": label, "score": round(min(1.0, max(0.0, 0.5 + 0.2 * balance)), 2)}
'''

_INIT_CASES = (
    "\n".join(
        json.dumps({"id": cid, "input": text, "expected": label})
        for cid, text, label in (
            ("c1", "I love this product, it is great", "positive"),
            ("c2", "Absolutely terrible service and a broken app", "negative"),
            ("c3", "The package arrived on Tuesday", "neutral"),
            ("c4", "Fantastic support team, very happy", "positive"),
            ("c5", "I hate the new update, it is useless", "negative"),
            ("c6", "Perfect, exactly what I needed", "positive"),
            ("c7", "The meeting is at noon", "neutral"),
            ("c8", "Awful experience, I am angry", "negative"),
            ("c9", "Love the new design, great work", "positive"),
            ("c10", "This is useless and the app is broken", "negative"),
            ("c11", "Please send the invoice by email", "neutral"),
            ("c12", "Good morning, happy to join the call", "positive"),
        )
    )
    + "\n"
)

_INIT_SPEC = """[suite]
name = "my-model"
seed = 0
cases_file = "cases.jsonl"

[model]
type = "python"
target = "model:predict"      # or: type = "http", url = "https://..." / type = "command"

[[contract]]
type = "golden"
name = "accuracy-on-labelled-cases"
output = "label"
tolerance = 0.25

[[contract]]
type = "property"
name = "score-is-a-probability"
output = "score"
check = { kind = "probability" }

[[contract]]
type = "invariance"
name = "case-does-not-matter"
output = "label"
transform = { kind = "case", mode = "upper" }

[[contract]]
type = "invariance"
name = "extra-whitespace-does-not-matter"
output = "label"
variants = 3
transform = { kind = "whitespace" }

[[contract]]
type = "invariance"
name = "typos-rarely-matter"
output = "label"
variants = 5
tolerance = 0.25
transform = { kind = "typo", rate = 0.05 }

[[contract]]
type = "sensitivity"
name = "flipping-the-sentiment-flips-the-label"
output = "label"

[contract.transform]
kind = "replace"

[contract.transform.mapping]
love = "hate"
great = "terrible"
fantastic = "awful"
perfect = "useless"
"""


def _cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.directory)
    files = {"assay.toml": _INIT_SPEC, "model.py": _INIT_MODEL, "cases.jsonl": _INIT_CASES}
    clash = [name for name in files if (target / name).exists()]
    if clash:
        raise SpecError(f"refusing to overwrite existing file(s) in '{target}': {', '.join(clash)}")
    target.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (target / name).write_text(content, encoding="utf-8")
    print(f"Created {', '.join(files)} in {target}/\nNext: assay run {target / 'assay.toml'}")
    return 0


# --------------------------------------------------------------------------- parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assay",
        description="Statistically rigorous behavioural contracts for AI and ML systems.",
    )
    parser.add_argument("--version", action="version", version=f"assay {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    def command(
        name: str, handler: Callable[[argparse.Namespace], int], help: str
    ) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help, description=help)
        sub.set_defaults(handler=handler)
        return sub

    run = command("run", _cmd_run, "Evaluate the contracts in a spec file against its model.")
    run.add_argument("spec", help="path to a .toml or .json spec")
    run.add_argument("-f", "--format", choices=RENDERERS, default="text", help="stdout format")
    run.add_argument(
        "--out-json", metavar="PATH", help="also write the JSON result (for `assay diff`)"
    )
    run.add_argument("--out-junit", metavar="PATH", help="also write JUnit XML")
    run.add_argument("--out-md", metavar="PATH", help="also write a Markdown summary")
    run.add_argument("--out-html", metavar="PATH", help="also write a self-contained HTML report")
    run.add_argument(
        "--github-summary", action="store_true", help="append Markdown to $GITHUB_STEP_SUMMARY"
    )
    run.add_argument(
        "--target", metavar="MODULE:FUNC", help="test this Python model instead of the spec's"
    )
    run.add_argument("--seed", type=int, help="override the spec's seed")
    run.add_argument("--workers", type=int, help="concurrent model calls (default: from spec)")
    run.add_argument(
        "--strict", action="store_true", help="treat inconclusive contracts as failures"
    )
    run.add_argument("--redact", action="store_true", help="hide inputs/outputs in counterexamples")
    run.add_argument(
        "--examples", type=int, default=2, help="counterexamples shown per contract (text)"
    )
    run.add_argument("--no-color", action="store_true")

    diff = command("diff", _cmd_diff, "Compare two JSON results and flag regressions.")
    diff.add_argument("baseline")
    diff.add_argument("candidate")
    diff.add_argument(
        "--report-only", action="store_true", help="exit 0 even when regressions exist"
    )
    diff.add_argument("--no-color", action="store_true")

    demo = command("demo", _cmd_demo, "Run the built-in demos (no setup, no API keys).")
    demo.add_argument("name", nargs="?", help=f"one of: {', '.join(DEMOS)} (default: all)")
    demo.add_argument("--variant", choices=VARIANTS, default="regressed")
    demo.add_argument("--examples", type=int, default=1)
    demo.add_argument("--no-color", action="store_true")

    init = command("init", _cmd_init, "Scaffold a runnable spec, model stub and cases.")
    init.add_argument("directory", nargs="?", default=".")

    check = command("check", _cmd_check, "Validate a spec without calling the model.")
    check.add_argument("spec")

    listing = command("list", _cmd_list, "List contracts, transforms and checks.")
    listing.add_argument("what", nargs="?", choices=("contracts", "transforms", "checks"))
    return parser


def _tolerate_narrow_encodings() -> None:
    """Reports use characters like ``≤`` and ``→``. On a console or redirect whose encoding cannot
    represent them (a legacy Windows code page), substitute ``?`` instead of crashing."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "utf-8").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _tolerate_narrow_encodings()
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except AssayError as exc:
        print(f"assay: error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # `assay list | head`: the reader left; stay quiet like other tools
        with contextlib.suppress(OSError, ValueError):
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
