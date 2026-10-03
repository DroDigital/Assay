import json
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from assay import Golden, Invariance, Property, Suite, checks
from assay import transforms as T
from assay.report import (
    evidence,
    pct,
    render_html,
    render_json,
    render_junit,
    render_markdown,
    render_text,
    short,
    xml_safe,
)


def make_result(model=None, cases=None):
    model = model or (lambda x: {"score": x["v"] / 10, "note": f"user {x['v']}@example.com"})
    cases = cases or [{"v": i} for i in range(1, 9)]
    suite = Suite("report-suite", model, cases, seed=4)
    return suite.add(
        Property("range-ok", check=checks.probability(), output="score"),
        Property("leaks", check=checks.no_pii(), output="note"),
        Property("thin", check=lambda y: y > 0.15 or "too small", output="score", tolerance=0.3),
        Invariance("stable", transform=T.shift_field("v", 1), output="score", tolerance=1.0),
    ).run()


def test_pct_formatting():
    assert [pct(v) for v in (None, 0, 0.0004, 0.037, 0.5, 1.0)] == [
        "n/a",
        "0%",
        "<0.1%",
        "3.7%",
        "50.0%",
        "100%",
    ]
    assert pct(0.05, trim=True) == "5%" and pct(0.052, trim=True) == "5.2%"


def test_short_is_escape_safe_and_bounded():
    assert short("a\x1b[31mred\nline") == '"a\\u001b[31mred\\nline"'
    assert len(short("x" * 500)) == 96 and short("x" * 500).endswith("…")


def test_text_report_lists_every_contract_with_its_verdict():
    text = render_text(make_result())
    for name in ("range-ok", "leaks", "thin", "stable"):
        assert name in text
    assert "PASS" in text and "FAIL" in text and "INCONCLUSIVE" in text
    assert "report-suite" in text and "seed 4" in text and "95% confidence" in text
    assert "2 passed · 1 failed · 1 inconclusive" in text
    assert "\x1b[" not in text


def test_text_report_colours_only_on_request():
    assert "\x1b[31m" in render_text(make_result(), color=True)
    assert "\x1b[" not in render_text(make_result(), color=False)


def test_text_report_limits_examples_and_points_to_other_formats():
    result = make_result()
    leaks = next(c for c in result.contracts if c.name == "leaks")
    assert len(leaks.counterexamples) == 3
    assert "(+1 more in the JSON/HTML/Markdown reports)" in render_text(result, examples=2)
    assert "more in the JSON" not in render_text(result, examples=3)


def test_hostile_model_output_cannot_inject_terminal_escapes_or_fake_lines():
    evil = "\x1b[2J\x1b[31mPASS\x1b[0m\n PASS         all-good   0/0 violations"
    result = Suite("t", lambda x: evil, ["a"]).add(Property("p", check=lambda y: "bad")).run()
    for rendered in (render_text(result), render_markdown(result)):
        assert "\x1b" not in rendered
        assert "\n PASS         all-good" not in rendered
    assert "\\u001b" in render_text(result)


def test_json_report_has_a_stable_schema():
    data = json.loads(render_json(make_result()))
    assert data["schema"] == "assay.result/v1" and data["tool"]["name"] == "assay"
    assert data["suite"] == {"name": "report-suite", "seed": 4, "confidence": 0.95, "cases": 8}
    assert data["summary"] == {"pass": 2, "fail": 1, "inconclusive": 1}
    first = data["contracts"][0]
    assert {
        "name",
        "kind",
        "verdict",
        "trials",
        "violations",
        "rate",
        "ci_low",
        "ci_high",
        "tolerance",
        "counterexamples",
    } <= set(first)


def test_markdown_has_a_table_details_and_escapes_pipes():
    result = (
        Suite("t", lambda x: "a|b", ["x"])
        .add(Property("p|ipe", check=lambda y: "pipe | in reason"))
        .run()
    )
    md = render_markdown(result)
    assert "| Verdict | Contract |" in md and "`p\\|ipe`" in md
    assert "<details>" in md and "```text" in md
    fenced = (
        Suite("t", lambda x: "```break```", ["x"]).add(Property("p", check=lambda y: "bad")).run()
    )
    assert (
        render_markdown(fenced).count("```") == 2
    )  # model output cannot close the code fence early


def test_junit_is_well_formed_and_maps_verdicts():
    root = ET.fromstring(render_junit(make_result()))
    assert root.tag == "testsuite" and root.get("tests") == "4" and root.get("failures") == "1"
    cases = {c.get("name"): c for c in root.iter("testcase")}
    assert cases["leaks"].find("failure") is not None
    assert cases["thin"].find("skipped") is not None
    assert cases["range-ok"].find("failure") is None and cases["range-ok"].find("skipped") is None
    assert {p.get("name") for p in root.iter("property")} == {"seed", "confidence"}


def test_junit_strict_turns_inconclusive_into_failures():
    root = ET.fromstring(render_junit(make_result(), strict=True))
    assert root.get("failures") == "2" and root.get("skipped") == "0"


def test_junit_survives_characters_xml_cannot_represent():
    nasty = "nul\x00 bell\x07 ퟿ end"
    result = (
        Suite("t\x00", lambda x: nasty, ["a"])
        .add(Property("p\x07", check=lambda y: "bad \x00 reason"))
        .run()
    )
    ET.fromstring(render_junit(result))  # would raise ParseError without sanitising
    assert xml_safe("a\x00b\x08c") == "a�b�c" and xml_safe("tab\tok\n") == "tab\tok\n"


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.scripts = [], 0

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.scripts += tag == "script"


def test_html_report_is_self_contained_and_escapes_everything():
    result = (
        Suite(
            "<b>suite</b>", lambda x: "<script>alert(1)</script>", ["<img src=x onerror=alert(1)>"]
        )
        .add(Property("<i>p</i>", check=lambda y: "reason <u>x</u>"))
        .run()
    )
    page = render_html(result)
    parser = _Collector()
    parser.feed(page)
    assert parser.scripts == 0 and "svg" in parser.tags and "<table>" in page
    assert "<img" not in page and "&lt;img" in page and "<b>suite" not in page
    assert (
        "http://" not in page.replace("http://www.w3.org", "") and "https://" not in page
    )  # no external assets


def test_html_gauge_reflects_verdicts():
    page = render_html(make_result())
    assert (
        'class="v fail"' in page and 'class="v pass"' in page and 'class="v inconclusive"' in page
    )
    assert "var(--warn)" in page


def test_evidence_wording():
    result = make_result()
    by_name = {c.name: c for c in result.contracts}
    assert "true rate ≤" in evidence(by_name["range-ok"])
    assert "violations ·" in evidence(by_name["leaks"]) and "CI" in evidence(by_name["leaks"])


def test_redacted_and_empty_results_still_render_sensibly():
    redacted = (
        Suite("t", lambda x: {"a": 1}, [{"g": "F"}], redact=True)
        .add(
            Invariance(
                "inv",
                transform=T.swap_field("g", {"F": "M"}),
                output="a",
                compare=lambda a, b: False,
            )
        )
        .run()
    )
    text = render_text(redacted)
    assert "changed fields: g" in text and "<redacted" in text and '"F"' not in text
    nothing = Suite("t", lambda x: x, [{"a": 1}]).add(Golden("g")).run()
    assert "no applicable trials" in evidence(nothing.contracts[0])


def test_model_controlled_keys_and_field_names_cannot_inject_terminal_escapes():
    hostile_key = "\x1b[2Jevil\nPASS fake-line"
    broken = (
        Suite("t", lambda x: {hostile_key: 1}, ["a"])
        .add(Property("p", check=lambda y: True, output="label"))
        .run()
    )
    row = {"a\x1b[31mred": 1, "g": "F"}
    swapped = (
        Suite("t", lambda r: r["g"], [row])
        .add(
            Invariance(
                "inv",
                transform=T.set_field("a\x1b[31mred", 2),
                tolerance=0.0,
                compare=lambda a, b: False,
            )
        )
        .run()
    )
    for result in (broken, swapped):
        for rendered in (
            render_text(result),
            render_markdown(result),
            render_junit(result),
            render_html(result),
        ):
            assert "\x1b" not in rendered and "\nPASS fake-line" not in rendered
    assert "\\u001b[2Jevil\\u000aPASS fake-line" in render_text(broken)
