"""Documentation is part of the product: it is executed, parsed and cross-checked."""

import re
import tomllib
from pathlib import Path

import pytest

from assay import Verdict, checks, contracts, transforms
from assay.spec import parse_spec

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "SECURITY.md", ROOT / "CHANGELOG.md",
        *sorted((ROOT / "docs").glob("*.md")), ROOT / "examples" / "README.md"]  # fmt: skip
FENCE = re.compile(r"```(\w+)[^\n]*\n(.*?)```", re.DOTALL)


def blocks(path: Path, language: str) -> list[str]:
    return [
        body for lang, body in FENCE.findall(path.read_text(encoding="utf-8")) if lang == language
    ]


def test_the_readme_quickstart_runs_and_tells_the_truth():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    snippets = re.findall(r"<!-- tested -->\n```python\n(.*?)```", text, re.DOTALL)
    assert len(snippets) == 1
    namespace: dict = {}
    exec(compile(snippets[0], "README.md", "exec"), namespace)  # noqa: S102 - our own documentation
    verdicts = {c.name: c.verdict for c in namespace["result"].contracts}
    assert verdicts == {
        "valid-score": Verdict.PASS,
        "income-never-hurts": Verdict.FAIL,
        "gender-blind": Verdict.FAIL,
    }
    # The numbers quoted in the README's sample output must be the real ones.
    by_name = {c.name: c for c in namespace["result"].contracts}
    assert (by_name["income-never-hurts"].violations, by_name["income-never-hurts"].trials) == (
        12,
        32,
    )
    assert (by_name["gender-blind"].violations, by_name["gender-blind"].trials) == (16, 16)
    assert (
        "12/32 violations · 37.5% (CI 22.9%–54.7%)" in text
        and "16/16 violations · 100% (CI 80.6%–100%)" in text
    )


FRAGMENT_DEFAULTS = {
    "[model]": '[model]\ntype = "python"\ntarget = "math:sqrt"\n',
    "[[case]]": "[[case]]\ninput = 1\n",
    "[[contract]]": '[[contract]]\ntype = "property"\ncheck = { kind = "finite" }\n',
}


def complete(snippet: str) -> str:
    """Add whatever a fragment lacks so it parses as a whole spec."""
    for marker, default in FRAGMENT_DEFAULTS.items():
        if marker not in snippet and not (marker == "[[case]]" and "cases_file" in snippet):
            snippet += "\n" + default
    return snippet


TOML_SNIPPETS = [(doc.name, i, body) for doc in DOCS for i, body in enumerate(blocks(doc, "toml"))]


@pytest.mark.parametrize(
    ("doc", "index", "snippet"), TOML_SNIPPETS, ids=[f"{d}#{i}" for d, i, _ in TOML_SNIPPETS]
)
def test_every_toml_snippet_in_the_docs_is_a_valid_spec(doc, index, snippet):
    data = tomllib.loads(complete(snippet))
    parse_spec(data, resolve=False)


def test_there_are_toml_snippets_to_check():
    assert len(TOML_SNIPPETS) >= 9


def test_the_reference_documents_every_contract_transform_check_and_command():
    reference = (ROOT / "docs" / "REFERENCE.md").read_text(encoding="utf-8")
    names = [
        *contracts.REGISTRY,
        *transforms.REGISTRY,
        *checks.REGISTRY,
        *checks.PII_KINDS,
        *checks.SECRET_KINDS,
    ]
    missing = [n for n in names if f"`{n}`" not in reference]
    assert missing == []
    for command in ("run", "diff", "demo", "init", "check", "list"):
        assert f"assay {command}" in reference


def test_numbers_quoted_in_the_docs_match_the_code():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{len(transforms.REGISTRY)} built-in transforms" in readme
    assert f"{len(checks.REGISTRY)} checks" in readme
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert (
        f"{len(transforms.REGISTRY)} input transforms and {len(checks.REGISTRY)} output checks"
        in changelog
    )
    assert len(contracts.REGISTRY) == 8 and "Eight contracts" in changelog


def slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", text))


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_and_anchors_resolve(doc):
    text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.DOTALL)
    for target, anchor in re.findall(r"\]\(([^)#\s]*)(?:#([^)\s]+))?\)", text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        destination = (doc.parent / target).resolve() if target else doc
        assert destination.exists(), f"{doc.name}: broken link {target!r}"
        if anchor and destination.suffix == ".md":
            headings = re.findall(
                r"^#+\s+(.*)$",
                re.sub(r"```.*?```", "", destination.read_text(encoding="utf-8"), flags=re.DOTALL),
                re.MULTILINE,
            )
            assert anchor in {slug(h) for h in headings}, (
                f"{doc.name}: no heading for #{anchor} in {destination.name}"
            )


def test_screenshots_referenced_by_the_readme_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for image in re.findall(r'src="(docs/assets/[^"]+)"', readme):
        assert (ROOT / image).stat().st_size > 10_000
