"""Renderers: terminal, Markdown, JSON, JUnit XML and a self-contained HTML page.

Everything that came from the model or from your data is rendered through ``json.dumps``
first, which escapes control characters. A model that emits terminal escape sequences or fake
log lines therefore cannot garble or spoof the report.
"""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any

from ._version import __version__
from .results import ContractResult, SuiteResult
from .stats import Verdict

_LABEL = {Verdict.PASS: "PASS", Verdict.FAIL: "FAIL", Verdict.INCONCLUSIVE: "INCONCLUSIVE"}
_ANSI = {Verdict.PASS: "32", Verdict.FAIL: "31", Verdict.INCONCLUSIVE: "33"}
_HIDDEN = {"case", "reason"}


def pct(value: float | None, *, trim: bool = False) -> str:
    """Compact percentage: ``0%``, ``<0.1%``, ``3.7%``, ``100%`` (``trim`` drops a trailing .0)."""
    if value is None:
        return "n/a"
    if value == 0:
        return "0%"
    if value < 0.001:
        return "<0.1%"
    text = f"{value * 100:.1f}%"
    return text.replace(".0%", "%") if trim or value >= 1 else text


def short(value: Any, width: int = 96) -> str:
    """One-line, escape-safe rendering of any value."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= width else text[: width - 1] + "…"


def evidence(c: ContractResult) -> str:
    """One sentence describing what the trials showed."""
    if c.trials == 0:
        return "no applicable trials"
    base = f"{c.violations}/{c.trials} violations"
    if c.violations == 0:
        return f"{base} · true rate ≤ {pct(c.ci_high)} ({c.confidence:.0%} conf.)"
    return f"{base} · {pct(c.rate)} (CI {pct(c.ci_low)}–{pct(c.ci_high)})"


def _example_lines(example: Mapping[str, Any]) -> list[str]:
    lines = [f"case {example.get('case')}: {example.get('reason', 'violation')}"]
    changed = example.get("changed")
    if isinstance(changed, Mapping):
        for name, pair in changed.items():
            lines.append(f"  changed {name}: {short(pair[0], 40)} → {short(pair[1], 40)}")
    elif isinstance(changed, list):
        lines.append(f"  changed fields: {', '.join(map(str, changed))}")
    for key, value in example.items():
        if key in _HIDDEN or key == "changed" or (key == "variant" and changed):
            continue
        lines.append(f"  {key}: {short(value)}")
    return lines


def _detail_lines(c: ContractResult, examples: int = 3) -> list[str]:
    lines: list[str] = []
    if c.verdict is Verdict.INCONCLUSIVE and c.samples_to_decide:
        lines.append(f"needs ≈{c.samples_to_decide} trials to decide at this rate")
    if c.metrics:
        lines.append(" · ".join(f"{k} {v:g}" for k, v in c.metrics.items()))
    lines.extend(f"note: {note}" for note in c.notes)
    if c.verdict is not Verdict.PASS:
        for example in c.counterexamples[:examples]:
            lines.extend(_example_lines(example))
        if len(c.counterexamples) > examples:
            hidden = len(c.counterexamples) - examples
            lines.append(f"(+{hidden} more in the JSON/HTML/Markdown reports)")
    return lines


# --------------------------------------------------------------------------- text


def render_text(result: SuiteResult, *, color: bool = False, examples: int = 2) -> str:
    def paint(text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if color else text

    width = max((len(c.name) for c in result.contracts), default=8)
    counts = result.counts()
    out = [
        paint(f"assay {__version__}", "1")
        + f" · {result.suite} · {len(result.contracts)} contracts · {result.n_cases} cases"
        + f" · seed {result.seed} · {result.confidence:.0%} confidence",
        "",
    ]
    for c in result.contracts:
        label = paint(f"{_LABEL[c.verdict]:<12}", _ANSI[c.verdict])
        tolerance = f"tolerance {pct(c.tolerance, trim=True)}"
        out.append(f" {label} {paint(c.name.ljust(width), '1')}  {evidence(c)} · {tolerance}")
        for line in _detail_lines(c, examples):
            out.append(paint(f"{'':15}{line}", "2"))
    out += [
        "",
        f"{counts['pass']} passed · {counts['fail']} failed · {counts['inconclusive']} inconclusive"
        f" · {result.duration_s:.2f}s",
    ]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- json


def render_json(result: SuiteResult) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- markdown

_ICON = {
    Verdict.PASS: "✅ pass",
    Verdict.FAIL: "❌ **fail**",
    Verdict.INCONCLUSIVE: "⚠️ inconclusive",
}


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(result: SuiteResult) -> str:
    counts = result.counts()
    out = [
        f"### assay · {result.suite}",
        "",
        f"**{counts['pass']}** passed · **{counts['fail']}** failed · "
        f"**{counts['inconclusive']}** inconclusive — {result.n_cases} cases, "
        f"seed {result.seed}, {result.confidence:.0%} confidence",
        "",
        "| Verdict | Contract | Kind | Evidence | Tolerance |",
        "|---|---|---|---|---|",
    ]
    for c in result.contracts:
        out.append(
            f"| {_ICON[c.verdict]} | `{_cell(c.name)}` | {c.kind} | {_cell(evidence(c))} "
            f"| {pct(c.tolerance, trim=True)} |"
        )
    for c in result.contracts:
        body = _detail_lines(c)
        if c.verdict is Verdict.PASS or not body:
            continue
        fenced = "\n".join(body).replace("```", "'''")
        out += ["", f"<details><summary><code>{html.escape(c.name)}</code> details</summary>",
                "", "```text", fenced, "```", "", "</details>"]  # fmt: skip
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- junit

_XML_ILLEGAL = re.compile("[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def xml_safe(text: str) -> str:
    """Replace characters XML 1.0 cannot represent (model output may contain NUL and friends)."""
    return _XML_ILLEGAL.sub("\ufffd", text)


def render_junit(result: SuiteResult, *, strict: bool = False) -> str:
    """JUnit XML understood by GitHub, GitLab, Jenkins, Azure DevOps and most CI systems."""
    failing = {Verdict.FAIL, Verdict.INCONCLUSIVE} if strict else {Verdict.FAIL}
    suite = ET.Element("testsuite", name=xml_safe(result.suite), tests=str(len(result.contracts)))
    failures = skipped = 0
    props = ET.SubElement(suite, "properties")
    for key, value in (("seed", result.seed), ("confidence", result.confidence)):
        ET.SubElement(props, "property", name=key, value=str(value))
    for c in result.contracts:
        case = ET.SubElement(
            suite, "testcase", classname=xml_safe(result.suite), name=xml_safe(c.name),
            time=f"{c.duration_s:.3f}",
        )  # fmt: skip
        text = xml_safe("\n".join([evidence(c), *_detail_lines(c)]))
        if c.verdict in failing:
            failures += 1
            message = xml_safe(f"{c.verdict.value}: {evidence(c)}")
            failure = ET.SubElement(case, "failure", message=message, type=c.verdict.value)
            failure.text = text
        elif c.verdict is Verdict.INCONCLUSIVE:
            skipped += 1
            ET.SubElement(case, "skipped", message=xml_safe(f"inconclusive: {evidence(c)}"))
        else:
            ET.SubElement(case, "system-out").text = text
    suite.set("failures", str(failures))
    suite.set("skipped", str(skipped))
    suite.set("time", f"{result.duration_s:.3f}")
    ET.indent(suite)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(suite, encoding="unicode") + "\n"
    )


# --------------------------------------------------------------------------- html

_CSS = """
:root {
  --bg: #fff; --fg: #1b1f24; --muted: #5b6672; --line: #d8dee4; --card: #f6f8fa;
  --pass: #1a7f37; --fail: #cf222e; --warn: #9a6700;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e; --line: #30363d; --card: #161b22;
    --pass: #3fb950; --fail: #f85149; --warn: #d29922;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1000px; margin: 0 auto; padding: 32px 16px; }
h1 { margin: 0 0 4px; font-size: 1.5rem; }
.meta { color: var(--muted); margin: 0 0 20px; }
.chips { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
.chip {
  border: 1px solid var(--line); border-radius: 999px; padding: 2px 12px;
  background: var(--card);
}
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); }
th {
  color: var(--muted); font-weight: 600; font-size: .8rem;
  text-transform: uppercase; letter-spacing: .04em;
}
td:first-child { white-space: nowrap; padding-right: 20px; }
.v { font-weight: 700; font-size: .8rem; letter-spacing: .04em; }
.pass { color: var(--pass); } .fail { color: var(--fail); } .inconclusive { color: var(--warn); }
code { font: .88em ui-monospace, SFMono-Regular, Menlo, monospace; }
.kind { color: var(--muted); font-size: .85em; }
details { margin-top: 6px; } summary { cursor: pointer; color: var(--muted); }
pre {
  background: var(--card); border: 1px solid var(--line); border-radius: 6px;
  padding: 10px; overflow: auto; margin: 6px 0;
}
svg { display: block; max-width: 100%; }
svg .track { stroke: var(--line); } svg .tol { stroke: var(--fg); stroke-dasharray: 3 2; }
@media (max-width: 640px) { th:nth-child(3), td:nth-child(3) { display: none; } }
"""

_GAUGE_WIDTH = 208  # drawable width of the 220px gauge


def _gauge(c: ContractResult) -> str:
    """Violation rate with its confidence interval against the tolerance, as inline SVG."""
    axis = max(0.05, min(1.0, 1.25 * max(c.tolerance, c.ci_high)))

    def x(value: float) -> str:
        return f"{6 + _GAUGE_WIDTH * min(value, axis) / axis:.1f}"

    tone = "warn" if c.verdict is Verdict.INCONCLUSIVE else c.verdict.value
    title = html.escape(
        f"rate {pct(c.rate)}, CI {pct(c.ci_low)}–{pct(c.ci_high)}, "
        f"tolerance {pct(c.tolerance, trim=True)}"
    )
    return (
        f'<svg width="220" height="22" viewBox="0 0 220 22" role="img" aria-label="{title}">'
        f"<title>{title}</title>"
        '<line class="track" x1="6" y1="11" x2="214" y2="11" stroke-width="2"/>'
        f'<line x1="{x(c.ci_low)}" y1="11" x2="{x(c.ci_high)}" y2="11" '
        f'stroke="var(--{tone})" stroke-width="6" stroke-linecap="round"/>'
        f'<circle cx="{x(c.rate or 0.0)}" cy="11" r="4" fill="var(--bg)" '
        f'stroke="var(--{tone})" stroke-width="2"/>'
        f'<line class="tol" x1="{x(c.tolerance)}" y1="2" x2="{x(c.tolerance)}" y2="20" '
        'stroke-width="1.5"/></svg>'
    )


_ROW = (
    '<tr><td><span class="v {tone}">{label}</span></td>'
    '<td><code>{name}</code><div class="kind">{kind}</div></td>'
    "<td>{gauge}</td>"
    '<td>{evidence}<div class="kind">tolerance {tolerance}</div>{detail}</td></tr>'
)

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>assay · {suite}</title><style>{css}</style></head><body><main>
<h1>{suite}</h1>
<p class="meta">assay {version} · {cases} cases · seed {seed} · {confidence} confidence
· {seconds}s</p>
<div class="chips"><span class="chip pass">{passed} passed</span>
<span class="chip fail">{failed} failed</span>
<span class="chip inconclusive">{inconclusive} inconclusive</span></div>
<table><thead><tr><th>Verdict</th><th>Contract</th><th>Rate · CI · tolerance</th>
<th>Evidence</th></tr></thead>
<tbody>{rows}</tbody></table></main></body></html>
"""


def render_html(result: SuiteResult) -> str:
    """A single self-contained page: no scripts, no external requests, light and dark themes."""
    esc = html.escape
    rows: list[str] = []
    for c in result.contracts:
        lines = _detail_lines(c)
        detail = ""
        if lines and c.verdict is not Verdict.PASS:
            body = esc("\n".join(lines))
            detail = f"<details><summary>details</summary><pre>{body}</pre></details>"
        rows.append(
            _ROW.format(
                tone=c.verdict.value,
                label=_LABEL[c.verdict],
                name=esc(c.name),
                kind=esc(c.kind),
                gauge=_gauge(c),
                evidence=esc(evidence(c)),
                tolerance=pct(c.tolerance, trim=True),
                detail=detail,
            )
        )
    counts = result.counts()
    return _PAGE.format(
        suite=esc(result.suite),
        css=_CSS,
        version=__version__,
        cases=result.n_cases,
        seed=result.seed,
        confidence=f"{result.confidence:.0%}",
        seconds=f"{result.duration_s:.2f}",
        passed=counts["pass"],
        failed=counts["fail"],
        inconclusive=counts["inconclusive"],
        rows="".join(rows),
    )


RENDERERS = ("text", "json", "markdown", "junit", "html")
