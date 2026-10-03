import json
from pathlib import Path

import pytest

from assay import Verdict
from assay.contracts import MISSING, Consistency, Golden, Monotonicity, Property
from assay.errors import SpecError
from assay.spec import load_spec, parse_spec, read_cases

MODEL = {"type": "python", "target": "math:sqrt"}


def spec(contracts, **extra):
    data = {"model": MODEL, "case": [{"input": 4.0}], "contract": contracts, **extra}
    return parse_spec(data, resolve=False)


def test_a_complete_spec_round_trips_into_a_runnable_suite():
    parsed = parse_spec(
        {
            "suite": {
                "name": "demo",
                "seed": 9,
                "confidence": 0.9,
                "workers": 2,
                "on_error": "skip",
                "max_examples": 1,
                "redact": True,
            },
            "model": MODEL,
            "case": [{"id": "four", "input": 4.0, "expected": 2.0}, {"input": 9.0}],
            "contract": [
                {"type": "golden", "name": "acc"},
                {"type": "property", "name": "p", "check": {"kind": "probability"}},
                {"type": "consistency", "name": "c", "repeats": 2, "abs_tol": 0.1},
            ],
        }
    )
    assert (parsed.name, parsed.seed, parsed.confidence, parsed.workers) == ("demo", 9, 0.9, 2)
    assert (parsed.on_error, parsed.max_examples, parsed.redact) == ("skip", 1, True)
    assert [type(c) for c in parsed.contracts] == [Golden, Property, Consistency]
    assert [c.id for c in parsed.cases] == ["four", ""] and parsed.cases[1].expected is MISSING
    result = parsed.to_suite().run()
    assert result.contracts[0].trials == 1  # only the labelled case
    assert result.suite == "demo" and result.confidence == 0.9


def test_toml_files_resolve_relative_paths_and_default_the_suite_name(tmp_path):
    (tmp_path / "cases.jsonl").write_text(
        '{"input": 1}\n\n{"input": 4, "expected": 2, "id": "b"}\n'
    )
    (tmp_path / "myproject").mkdir()
    path = tmp_path / "myproject" / "assay.toml"
    path.write_text(
        '[suite]\ncases_file = "../cases.jsonl"\n[model]\ntype = "python"\ntarget = "math:sqrt"\n'
        '[[contract]]\ntype = "golden"\nname = "g"\ntolerance = 1.0\n'
    )
    loaded = load_spec(path)
    assert loaded.name == "myproject" and [c.input for c in loaded.cases] == [1, 4]
    assert loaded.path == path and loaded.to_suite().run().contracts[0].trials == 1


def test_json_specs_are_supported(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "case": [{"input": 4}],
                "contract": [{"type": "property", "check": {"kind": "finite"}}],
            }
        )
    )
    assert load_spec(path).contracts[0].name == "property-1"


def test_python_models_import_from_the_spec_directory(tmp_path):
    (tmp_path / "my_model_module.py").write_text("def predict(x):\n    return x * 2\n")
    path = tmp_path / "assay.toml"
    path.write_text(
        '[model]\ntype = "python"\ntarget = "my_model_module:predict"\n[[case]]\ninput = 2\nexpected = 4\n'
        '[[contract]]\ntype = "golden"\nname = "g"\n'
    )
    assert load_spec(path).to_suite().run().contracts[0].verdict is Verdict.PASS


@pytest.mark.parametrize(
    ("contract", "fragment"),
    [
        ({"type": "invarience"}, "unknown contract type 'invarience' - did you mean 'invariance'"),
        (
            {"type": "property", "check": {"kind": "probability"}, "tolrance": 0.1},
            "unknown option 'tolrance' - did you mean 'tolerance'",
        ),
        ({"type": "invariance"}, "missing required option(s): transform"),
        ({"type": "property"}, "missing required option(s): check"),
        (
            {"type": "invariance", "transform": {"kind": "tipo"}},
            "unknown transform 'tipo' - did you mean 'typo'",
        ),
        ({"type": "invariance", "transform": {"kind": "typo", "rte": 0.1}}, "unknown option 'rte'"),
        (
            {"type": "invariance", "transform": {"kind": "typo", "rate": 5}},
            "typo rate must be in (0, 1]",
        ),
        ({"type": "invariance", "transform": "typo"}, "transform must be a table"),
        ({"type": "invariance", "transform": []}, "transform list is empty"),
        ({"type": "property", "check": {"kind": "nope"}}, "unknown check 'nope'"),
        (
            {"type": "property", "check": {"kind": "one_of", "allowed": "abc"}},
            "'allowed' must be a list",
        ),
        ({"type": "property", "check": []}, "check list is empty"),
        (
            {"type": "property", "check": {"kind": "probability"}, "tolerance": 7},
            "tolerance must be in",
        ),
        ({"type": "property", "check": {"kind": "probability"}, "tolerance": "5%"}, "contract #1"),
        (
            {"type": "golden", "compare": "exact", "abs_tol": 0.1},
            "only apply to compare = 'approx'",
        ),
        ({"type": "latency"}, "missing required option(s): budget_ms"),
        ({"type": "monotonicity", "field": "x", "direction": "sideways"}, "direction must be"),
        ({}, "unknown contract type"),
    ],
)
def test_contract_errors_are_specific_and_located(contract, fragment):
    with pytest.raises(SpecError, match="contract #1") as info:
        spec([contract])
    assert fragment in str(info.value)


def test_variadic_options_take_lists_and_transform_lists_chain():
    parsed = spec(
        [
            {
                "type": "property",
                "name": "p",
                "check": [
                    {"kind": "one_of", "allowed": ["a", "b"]},
                    {"kind": "max_length", "limit": 3},
                ],
            },
            {
                "type": "invariance",
                "name": "i",
                "transform": [
                    {"kind": "case", "mode": "upper"},
                    {"kind": "suffix", "texts": ["!", "?"]},
                ],
            },
        ]
    )
    check = parsed.contracts[0].check
    assert check("a") is None and check("c") is not None
    assert parsed.contracts[1].transform.name == "case(upper)+suffix"


def test_comparators_and_steps_are_built_from_plain_settings():
    parsed = spec(
        [
            {
                "type": "invariance",
                "name": "i",
                "transform": {"kind": "case", "mode": "upper"},
                "compare": "casefold",
            },
            {"type": "golden", "name": "g", "abs_tol": 0.5},
            {
                "type": "monotonicity",
                "name": "m",
                "field": "x",
                "steps": [0.1, 0.2],
                "mode": "relative",
            },
        ]
    )
    assert parsed.contracts[0].compare("A  b", "a b") and parsed.contracts[1].compare(1.0, 1.4)
    assert isinstance(parsed.contracts[2], Monotonicity) and parsed.contracts[2].steps == (0.1, 0.2)


def test_agreement_reference_models_are_built_from_a_model_table():
    parsed = parse_spec({"model": MODEL, "case": [{"input": 4.0}],
                         "contract": [{"type": "agreement", "name": "a", "reference": {"type": "python", "target": "math:fabs"}}]})  # fmt: skip
    assert parsed.to_suite().run().contracts[0].violations == 1  # sqrt(4) != fabs(4)


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"case": [{"input": 1}], "contract": [{"type": "golden"}]}, r"needs a \[model\]"),
        ({"model": MODEL, "contract": [{"type": "golden"}]}, "no cases"),
        ({"model": MODEL, "case": [{"input": 1}]}, r"at least one \[\[contract\]\]"),
        ({"model": MODEL, "case": {"input": 1}, "contract": [{"type": "golden"}]}, r"\[\[case\]\]"),
        (
            {"model": MODEL, "case": [{"inputs": 1}], "contract": [{"type": "golden"}]},
            "'input' key",
        ),
        (
            {"model": MODEL, "case": [{"input": 1, "expect": 2}], "contract": [{"type": "golden"}]},
            "did you mean 'expected'",
        ),
        (
            {"model": MODEL, "case": [{"input": 1}], "contract": [{"type": "golden"}], "suit": {}},
            "did you mean 'suite'",
        ),
        (
            {
                "suite": {"seeds": 1},
                "model": MODEL,
                "case": [{"input": 1}],
                "contract": [{"type": "golden"}],
            },
            "did you mean 'seed'",
        ),
        (
            {
                "suite": {"on_error": "ignore"},
                "model": MODEL,
                "case": [{"input": 1}],
                "contract": [{"type": "golden"}],
            },
            "on_error",
        ),
        (
            {
                "model": MODEL,
                "case": [{"input": 1}],
                "contract": [{"type": "golden", "name": "a"}, {"type": "golden", "name": "a"}],
            },
            "duplicate contract name",
        ),
    ],
)
def test_top_level_validation(data, fragment):
    with pytest.raises(SpecError, match=fragment):
        parse_spec(data, resolve=False)


@pytest.mark.parametrize(
    ("model", "fragment"),
    [
        ({"type": "grpc"}, "type must be"),
        ({"type": "python"}, "needs target"),
        ({"type": "python", "target": "math:sqrt", "tareget": 1}, "unknown option"),
        ({"type": "http"}, "needs url"),
        ({"type": "command"}, "needs command"),
        ("math:sqrt", "expected a table"),
    ],
)
def test_model_table_validation(model, fragment):
    with pytest.raises(SpecError, match=fragment):
        parse_spec(
            {"model": model, "case": [{"input": 1}], "contract": [{"type": "golden"}]},
            resolve=False,
        )


def test_http_and_command_models_validate_on_resolve():
    ok_http = {"type": "http", "url": "http://localhost:1/x", "input_key": "prompt", "retries": 0}
    ok_cmd = {"type": "command", "command": ["python", "-c", "pass"]}
    for model in (ok_http, ok_cmd):
        assert (
            parse_spec(
                {"model": model, "case": [{"input": 1}], "contract": [{"type": "golden"}]}
            ).model
            is not None
        )
    with pytest.raises(SpecError, match="http"):
        parse_spec(
            {
                "model": {"type": "http", "url": "file:///x"},
                "case": [{"input": 1}],
                "contract": [{"type": "golden"}],
            }
        )
    with pytest.raises(SpecError, match="list of strings"):
        parse_spec(
            {
                "model": {"type": "command", "command": "python model.py"},
                "case": [{"input": 1}],
                "contract": [{"type": "golden"}],
            }
        )


def test_unresolved_specs_cannot_be_turned_into_suites():
    with pytest.raises(SpecError, match="without resolving"):
        spec([{"type": "golden"}]).to_suite()


def test_load_spec_errors(tmp_path):
    with pytest.raises(SpecError, match="cannot read spec"):
        load_spec(tmp_path / "missing.toml")
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("this is = = not toml")
    with pytest.raises(SpecError, match="bad.toml"):
        load_spec(bad_toml)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{nope")
    with pytest.raises(SpecError, match="bad.json"):
        load_spec(bad_json)
    other = tmp_path / "spec.yaml"
    other.write_text("a: 1")
    with pytest.raises(SpecError, match="unsupported spec type"):
        load_spec(other)


def test_read_cases_errors_name_the_file_and_line(tmp_path):
    bad = tmp_path / "c.jsonl"
    bad.write_text('{"input": 1}\n{oops}\n')
    with pytest.raises(SpecError, match=r"c.jsonl:2: invalid JSON"):
        read_cases(bad)
    bad.write_text('{"input": 1}\n{"x": 2}\n')
    with pytest.raises(SpecError, match=r"c.jsonl:2.*'input' key"):
        read_cases(bad)
    with pytest.raises(SpecError, match="cannot read cases"):
        read_cases(tmp_path / "missing.jsonl")
    as_json = tmp_path / "c.json"
    as_json.write_text('[{"input": "a", "expected": "b"}]')
    assert read_cases(as_json)[0].expected == "b"
    as_json.write_text('{"input": "a"}')
    with pytest.raises(SpecError, match="JSON list"):
        read_cases(as_json)
    as_json.write_text("[nope")
    with pytest.raises(SpecError, match="invalid JSON"):
        read_cases(as_json)


def test_shipped_demo_specs_are_valid():
    from assay.demos import DEMOS

    for demo in DEMOS.values():
        assert isinstance(load_spec(demo.spec, resolve=False).contracts[0].name, str)
        assert Path(demo.spec).exists()


def test_more_malformed_specs():
    base = {"model": MODEL, "case": [{"input": 1}]}
    with pytest.raises(SpecError, match=r"\[suite\] must be a table"):
        parse_spec({**base, "suite": "nope", "contract": [{"type": "golden"}]}, resolve=False)
    with pytest.raises(SpecError, match="check must be a table"):
        parse_spec(
            {**base, "contract": [{"type": "property", "check": "probability"}]}, resolve=False
        )


@pytest.mark.parametrize(
    ("contract", "fragment"),
    [
        ({"type": "property", "check": {"kind": "matches", "pattern": "("}}, "error"),
        ({"type": "monotonicity", "field": "x", "steps": 0.5}, "steps must be a list of numbers"),
        ({"type": "property", "check": {"kind": "probability"}, "tolerance": "5%"}, "TypeError"),
        ({"type": "latency", "budget_ms": "fast"}, "contract #1"),
    ],
)
def test_wrongly_typed_values_become_located_spec_errors_not_tracebacks(contract, fragment):
    with pytest.raises(SpecError, match="contract #1") as info:
        spec([contract])
    assert fragment in str(info.value)


@pytest.mark.parametrize(
    ("suite", "fragment"),
    [
        ({"seed": "abc"}, "'seed' must be int"),
        ({"seed": 1.5}, "'seed' must be int"),
        ({"seed": True}, "'seed' must be int"),
        ({"workers": "4"}, "'workers' must be int"),
        ({"confidence": "high"}, "'confidence' must be float"),
        ({"redact": "yes"}, "'redact' must be bool"),
        ({"name": 7}, "'name' must be str"),
    ],
)
def test_suite_settings_are_type_checked(suite, fragment):
    with pytest.raises(SpecError, match=fragment):
        spec([{"type": "golden"}], suite=suite)


def test_integer_valued_floats_are_accepted_where_floats_are_expected():
    assert spec([{"type": "golden"}], suite={"confidence": 1 - 0.05}).confidence == 0.95


def test_specs_with_the_same_module_name_each_get_their_own_module(tmp_path):
    """A second spec must not silently reuse the first one's cached module."""
    for name, value in (("a", 1), ("b", 2)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "clashing_model.py").write_text(f"def predict(x):\n    return {value}\n")
        (directory / "assay.toml").write_text(
            '[model]\ntype = "python"\ntarget = "clashing_model:predict"\n[[case]]\ninput = 0\nexpected = '
            f'{value}\n[[contract]]\ntype = "golden"\nname = "g"\n'
        )
    results = [
        load_spec(tmp_path / d / "assay.toml").to_suite().run().contracts[0].verdict
        for d in ("a", "b", "a")
    ]
    assert results == [Verdict.PASS, Verdict.PASS, Verdict.PASS]


def test_cwd_and_spec_directories_are_fallbacks_never_ahead_of_the_standard_library(
    tmp_path, monkeypatch
):
    import sys

    (tmp_path / "json.py").write_text("raise RuntimeError('shadowed the standard library')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", str(tmp_path))])
    before = list(sys.path)
    parse_spec(
        {"model": MODEL, "case": [{"input": 4.0}], "contract": [{"type": "golden"}]},
        base_dir=tmp_path,
    )
    assert sys.path[: len(before)] == before  # nothing was inserted ahead of the existing entries
    import json as stdlib_json

    assert stdlib_json.__file__ != str(tmp_path / "json.py")


def test_a_cache_false_model_setting_survives_a_model_override():
    data = {"model": {**MODEL, "cache": False}, "case": [{"input": "a"}], "contract": [
        {"type": "property", "name": "p", "check": {"kind": "finite"}},
        {"type": "invariance", "name": "i", "tolerance": 1.0, "transform": {"kind": "suffix", "texts": ["!"]}},
    ]}  # fmt: skip
    calls = []

    def model(x):
        calls.append(x)
        return 1.0

    parse_spec(data).to_suite(model=model).run()
    assert len(calls) == 3  # "a" is not shared between the two contracts: caching is off
    calls.clear()
    parse_spec({**data, "model": MODEL}).to_suite(model=model).run()
    assert len(calls) == 2  # with the default cache, "a" is computed once
