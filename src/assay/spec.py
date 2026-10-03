"""Declarative specs: a TOML (or JSON) file describing a model, cases and contracts.

A spec is *configuration*, but ``type = "python"`` models are imported, so treat spec files
with the same trust as the code they point at.
"""

from __future__ import annotations

import dataclasses
import difflib
import inspect
import json
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import checks as checks_mod
from . import contracts as contracts_mod
from . import transforms as transforms_mod
from .compare import make_comparator
from .contracts import Case, Contract
from .errors import SpecError
from .model import Model, ModelFn, command_model, http_model, import_target
from .runner import ON_ERROR, Suite
from .transforms import Transform

_SUITE_KEYS = (
    "name",
    "seed",
    "confidence",
    "workers",
    "on_error",
    "max_examples",
    "redact",
    "cases_file",
)
_TOP_KEYS = ("suite", "model", "case", "contract")


def _hint(word: str, valid: Sequence[str]) -> str:
    close = difflib.get_close_matches(word, list(valid), n=1)
    return f" - did you mean '{close[0]}'?" if close else ""


def _reject_unknown(given: Mapping[str, Any], valid: Sequence[str], where: str) -> None:
    unknown = [key for key in given if key not in valid]
    if unknown:
        key = unknown[0]
        raise SpecError(
            f"{where}: unknown option '{key}'{_hint(key, valid)} (valid: {', '.join(valid)})"
        )


def _call(factory: Callable[..., Any], options: Mapping[str, Any], where: str) -> Any:
    """Call a factory with spec options, turning every failure into a located SpecError.

    A variadic parameter (``one_of(*allowed)``, ``suffix(*texts)``) is filled from a list under
    its own name: ``allowed = ["a", "b"]``.
    """
    parameters = inspect.signature(factory).parameters
    _reject_unknown(options, list(parameters), where)
    keywords = dict(options)
    positional: list[Any] = []
    for name, parameter in parameters.items():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL and name in keywords:
            values = keywords.pop(name)
            if not isinstance(values, list):
                raise SpecError(f"{where}: '{name}' must be a list")
            positional = values
    try:
        return factory(*positional, **keywords)
    except (SpecError, TypeError) as exc:
        raise SpecError(f"{where}: {exc}") from exc


def _registry_lookup(registry: Mapping[str, Any], kind: Any, what: str, where: str) -> Any:
    if not isinstance(kind, str) or kind not in registry:
        shown = kind if isinstance(kind, str) else repr(kind)
        raise SpecError(
            f"{where}: unknown {what} '{shown}'{_hint(str(kind), list(registry))} "
            f"(available: {', '.join(sorted(registry))})"
        )
    return registry[kind]


# --------------------------------------------------------------------------- parts


def build_transform(value: Any, where: str) -> Transform:
    """``{kind = "typo", rate = 0.1}`` or a list of such tables (applied in order)."""
    if isinstance(value, list):
        if not value:
            raise SpecError(f"{where}: transform list is empty")
        return transforms_mod.chain(*(build_transform(item, where) for item in value))
    if not isinstance(value, Mapping):
        raise SpecError(f'{where}: transform must be a table like {{ kind = "typo" }}')
    options = dict(value)
    factory = _registry_lookup(
        transforms_mod.REGISTRY, options.pop("kind", None), "transform", where
    )
    return _call(factory, options, f"{where}, transform '{value.get('kind')}'")  # type: ignore[no-any-return]


def build_check(value: Any, where: str) -> checks_mod.Check:
    """``{kind = "in_range", lo = 0, hi = 1}`` or a list of such tables (all must hold)."""
    if isinstance(value, list):
        if not value:
            raise SpecError(f"{where}: check list is empty")
        return checks_mod.all_of(*(build_check(item, where) for item in value))
    if not isinstance(value, Mapping):
        raise SpecError(f'{where}: check must be a table like {{ kind = "probability" }}')
    options = dict(value)
    factory = _registry_lookup(checks_mod.REGISTRY, options.pop("kind", None), "check", where)
    return _call(factory, options, f"{where}, check '{value.get('kind')}'")  # type: ignore[no-any-return]


def build_model(
    config: Any, base_dir: Path, where: str = "[model]", *, resolve: bool = True
) -> ModelFn:
    """Create a model callable from a ``[model]`` table. ``resolve=False`` validates only."""
    if not isinstance(config, Mapping):
        raise SpecError(f"{where}: expected a table")
    options = dict(config)
    kind = options.pop("type", None)
    cache = options.pop("cache", True)
    if kind == "python":
        _reject_unknown(options, ("target",), where)
        target = options.get("target")
        if not isinstance(target, str):
            raise SpecError(f"{where}: type 'python' needs target = \"package.module:function\"")
        if not resolve:
            return _placeholder
        for directory in (str(base_dir), str(Path.cwd())):
            if directory not in sys.path:
                sys.path.insert(0, directory)
        fn = import_target(target)
    elif kind == "http":
        _reject_unknown(
            options,
            (
                "url",
                "headers",
                "input_key",
                "output_path",
                "timeout",
                "retries",
                "backoff",
                "max_bytes",
            ),
            where,
        )
        if "url" not in options:
            raise SpecError(f"{where}: type 'http' needs url = \"https://...\"")
        if not resolve:
            return _placeholder
        fn = _call(http_model, options, where)
    elif kind == "command":
        _reject_unknown(options, ("command", "timeout"), where)
        if "command" not in options:
            raise SpecError(f'{where}: type \'command\' needs command = ["program", "arg"]')
        if not resolve:
            return _placeholder
        arguments = {("argv" if key == "command" else key): value for key, value in options.items()}
        fn = _call(command_model, arguments, where)
    else:
        raise SpecError(f"{where}: type must be 'python', 'http' or 'command', got {kind!r}")
    return Model(fn, cache=bool(cache))


def _placeholder(_: Any) -> Any:  # pragma: no cover - only used when validating without a model
    raise RuntimeError("model was not resolved")


def _build_contract(
    config: Mapping[str, Any], index: int, base_dir: Path, resolve: bool
) -> Contract:
    options = dict(config)
    kind = options.pop("type", None)
    where = f"contract #{index + 1}"
    cls: type[Contract] = _registry_lookup(contracts_mod.REGISTRY, kind, "contract type", where)
    name = options.pop("name", f"{kind}-{index + 1}")
    where = f"contract #{index + 1} '{name}' ({kind})"

    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    accepts_compare = "compare" in fields
    valid = [*fields, *(["abs_tol", "rel_tol"] if accepts_compare else [])]
    _reject_unknown(options, valid, where)

    if accepts_compare:
        try:
            options["compare"] = make_comparator(
                options.pop("compare", None),
                options.pop("abs_tol", None),
                options.pop("rel_tol", None),
            )
        except SpecError as exc:
            raise SpecError(f"{where}: {exc}") from exc
    if "transform" in options:
        options["transform"] = build_transform(options["transform"], where)
    if "check" in options:
        options["check"] = build_check(options["check"], where)
    if "reference" in options:
        options["reference"] = build_model(
            options["reference"], base_dir, f"{where}, reference", resolve=resolve
        )
    if "steps" in options:
        options["steps"] = tuple(options["steps"])

    required = [
        f.name for f in fields.values()
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        and f.name != "name" and f.name not in options
    ]  # fmt: skip
    if required:
        raise SpecError(f"{where}: missing required option(s): {', '.join(required)}")
    try:
        return cls(name=str(name), **options)
    except SpecError as exc:
        raise SpecError(f"{where}: {exc}") from exc
    except TypeError as exc:
        raise SpecError(f"{where}: {exc}") from exc


def read_cases(path: Path) -> list[Case]:
    """Load cases from ``.jsonl`` (one ``{"input": ..., "expected": ..., "id": ...}`` per line)
    or a ``.json`` list of such objects."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read cases file '{path}': {exc}") from exc
    if path.suffix == ".json":
        try:
            rows = json.loads(text)
        except ValueError as exc:
            raise SpecError(f"{path}: invalid JSON: {exc}") from exc
        if not isinstance(rows, list):
            raise SpecError(f"{path}: expected a JSON list")
        entries = list(enumerate(rows, start=1))
    else:
        entries = []
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    entries.append((number, json.loads(line)))
                except ValueError as exc:
                    raise SpecError(f"{path}:{number}: invalid JSON: {exc}") from exc
    return [_case(row, f"{path.name}:{number}") for number, row in entries]


def _case(row: Any, where: str) -> Case:
    if not isinstance(row, Mapping) or "input" not in row:
        raise SpecError(f"{where}: each case must be an object with an 'input' key")
    _reject_unknown(row, ("input", "expected", "id"), where)
    return Case(
        input=row["input"],
        id=str(row["id"]) if "id" in row else "",
        expected=row.get("expected", contracts_mod.MISSING),
    )


# --------------------------------------------------------------------------- spec


@dataclass
class Spec:
    """A parsed spec file. ``model`` is ``None`` when parsed with ``resolve=False``."""

    name: str
    model: ModelFn | None
    cases: list[Case]
    contracts: list[Contract]
    seed: int = 0
    confidence: float = 0.95
    workers: int = 1
    on_error: str = "violation"
    max_examples: int = 3
    redact: bool = False
    path: Path | None = field(default=None, compare=False)

    def to_suite(self, *, model: ModelFn | None = None, **overrides: Any) -> Suite:
        chosen = model or self.model
        if chosen is None:
            raise SpecError("this spec was parsed without resolving its model")
        options: dict[str, Any] = {
            "seed": self.seed, "confidence": self.confidence, "workers": self.workers,
            "on_error": self.on_error, "max_examples": self.max_examples, "redact": self.redact,
        }  # fmt: skip
        options.update({k: v for k, v in overrides.items() if v is not None})
        return Suite(self.name, chosen, self.cases, contracts=self.contracts, **options)


def parse_spec(data: Mapping[str, Any], *, base_dir: Path | None = None, name: str = "suite",
               resolve: bool = True) -> Spec:  # fmt: skip
    """Build a :class:`Spec` from already-loaded TOML/JSON data."""
    base = base_dir or Path.cwd()
    _reject_unknown(data, _TOP_KEYS, "spec")
    suite_cfg = data.get("suite", {})
    if not isinstance(suite_cfg, Mapping):
        raise SpecError("[suite] must be a table")
    _reject_unknown(suite_cfg, _SUITE_KEYS, "[suite]")
    if "model" not in data:
        raise SpecError("spec needs a [model] table")
    if suite_cfg.get("on_error", "violation") not in ON_ERROR:
        raise SpecError(f"[suite]: on_error must be one of {ON_ERROR}")

    model = build_model(data["model"], base, resolve=resolve)

    cases: list[Case] = []
    if "cases_file" in suite_cfg:
        cases.extend(read_cases((base / str(suite_cfg["cases_file"])).resolve()))
    inline = data.get("case", [])
    if not isinstance(inline, list):
        raise SpecError("use [[case]] (an array of tables) for inline cases")
    cases.extend(_case(row, f"case #{i + 1}") for i, row in enumerate(inline))
    if not cases:
        raise SpecError(
            'spec has no cases: add [[case]] entries or [suite] cases_file = "cases.jsonl"'
        )

    raw_contracts = data.get("contract", [])
    if not isinstance(raw_contracts, list) or not raw_contracts:
        raise SpecError("spec needs at least one [[contract]]")
    contracts = [_build_contract(c, i, base, resolve) for i, c in enumerate(raw_contracts)]
    names = [c.name for c in contracts]
    duplicate = next((n for n in names if names.count(n) > 1), None)
    if duplicate:
        raise SpecError(f"duplicate contract name '{duplicate}'")

    return Spec(
        name=str(suite_cfg.get("name", name)),
        model=model if resolve else None,
        cases=cases,
        contracts=contracts,
        seed=int(suite_cfg.get("seed", 0)),
        confidence=float(suite_cfg.get("confidence", 0.95)),
        workers=int(suite_cfg.get("workers", 1)),
        on_error=str(suite_cfg.get("on_error", "violation")),
        max_examples=int(suite_cfg.get("max_examples", 3)),
        redact=bool(suite_cfg.get("redact", False)),
    )


def load_spec(path: str | Path, *, resolve: bool = True) -> Spec:
    """Read a ``.toml`` or ``.json`` spec file."""
    file = Path(path)
    try:
        raw = file.read_bytes()
    except OSError as exc:
        raise SpecError(f"cannot read spec '{file}': {exc}") from exc
    try:
        if file.suffix == ".json":
            data = json.loads(raw)
        elif file.suffix == ".toml":
            data = tomllib.loads(raw.decode("utf-8"))
        else:
            raise SpecError(f"unsupported spec type '{file.suffix}' (use .toml or .json)")
    except (ValueError, UnicodeDecodeError) as exc:
        raise SpecError(f"{file}: {exc}") from exc
    spec = parse_spec(
        data, base_dir=file.parent.resolve(), name=file.parent.name or file.stem, resolve=resolve
    )
    spec.path = file
    return spec
