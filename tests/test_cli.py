import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from assay.cli import main
from assay.demos import DEMOS

LENDING = str(DEMOS["lending"].spec)
SUPPORT = str(DEMOS["support"].spec)


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_run_exit_codes(capsys):
    assert run(capsys, "run", LENDING)[0] == 0
    code, out, _ = run(capsys, "run", LENDING, "--target", "assay.demos.lending:regressed")
    assert code == 1 and "FAIL" in out and "more-income-never-hurts" in out


def test_strict_turns_inconclusive_into_exit_3(capsys, tmp_path):
    spec = tmp_path / "s.toml"
    spec.write_text(
        '[model]\ntype = "python"\ntarget = "math:fabs"\n[[case]]\ninput = 1\n'
        '[[contract]]\ntype = "property"\nname = "thin"\ncheck = { kind = "finite" }\ntolerance = 0.05\n'
    )
    assert run(capsys, "run", str(spec))[0] == 0
    code, out, _ = run(capsys, "run", str(spec), "--strict")
    assert code == 3 and "INCONCLUSIVE" in out


@pytest.mark.parametrize("fmt", ["text", "json", "markdown", "junit", "html"])
def test_every_stdout_format_works(capsys, fmt):
    code, out, _ = run(capsys, "run", LENDING, "--format", fmt)
    assert code == 0 and "lending-model" in out
    if fmt == "json":
        assert json.loads(out)["summary"]["fail"] == 0
    if fmt == "junit":
        ET.fromstring(out)


def test_report_files_can_be_written_in_the_same_run(capsys, tmp_path):
    paths = {k: tmp_path / f"r.{k}" for k in ("json", "xml", "md", "html")}
    code, out, _ = run(capsys, "run", LENDING, "--out-json", str(paths["json"]), "--out-junit", str(paths["xml"]),
                       "--out-md", str(paths["md"]), "--out-html", str(paths["html"]))  # fmt: skip
    assert code == 0 and "lending-model" in out
    assert json.loads(paths["json"].read_text())["schema"] == "assay.result/v1"
    assert ET.parse(paths["xml"]).getroot().tag == "testsuite"
    assert paths["md"].read_text().startswith("### assay") and paths["html"].read_text().startswith(
        "<!doctype html>"
    )


def test_github_step_summary_is_appended(capsys, tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    summary.write_text("earlier step\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    run(capsys, "run", LENDING, "--github-summary")
    text = summary.read_text()
    assert text.startswith("earlier step\n") and "### assay · lending-model" in text
    monkeypatch.delenv("GITHUB_STEP_SUMMARY")
    assert (
        run(capsys, "run", LENDING, "--github-summary")[0] == 0
    )  # silently ignored outside Actions


def test_seed_workers_and_redact_overrides(capsys):
    code, out, _ = run(capsys, "run", LENDING, "--seed", "99", "--workers", "4", "--redact",
                       "--target", "assay.demos.lending:regressed", "--format", "json")  # fmt: skip
    data = json.loads(out)
    assert code == 1 and data["suite"]["seed"] == 99
    example = next(c for c in data["contracts"] if c["name"] == "more-income-never-hurts")[
        "counterexamples"
    ][0]
    assert example["input"].startswith("<redacted")


def test_examples_flag_limits_counterexamples(capsys):
    _, one, _ = run(
        capsys, "run", SUPPORT, "--target", "assay.demos.support:regressed", "--examples", "1"
    )
    _, three, _ = run(
        capsys, "run", SUPPORT, "--target", "assay.demos.support:regressed", "--examples", "3"
    )
    assert one.count("case ticket") < three.count("case ticket")


def test_errors_exit_2_with_a_message_on_stderr(capsys, tmp_path):
    code, out, err = run(capsys, "run", str(tmp_path / "nope.toml"))
    assert code == 2 and out == "" and err.startswith("assay: error: cannot read spec")
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[model]\ntype = "python"\ntarget = "math:sqrt"\n[[case]]\ninput = 1\n[[contract]]\ntype = "propertyy"\n'
    )
    code, _, err = run(capsys, "run", str(bad))
    assert code == 2 and "did you mean 'property'" in err
    assert run(capsys, "run", LENDING, "--target", "nonexistent.module:f")[0] == 2


def test_argparse_errors_use_exit_code_2():
    with pytest.raises(SystemExit) as info:
        main(["run"])
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0 and capsys.readouterr().out.startswith("assay ")


def test_check_validates_without_calling_the_model(capsys, tmp_path):
    spec = tmp_path / "s.toml"
    spec.write_text('[model]\ntype = "python"\ntarget = "does.not.exist:f"\n[[case]]\ninput = 1\n'
                    '[[contract]]\ntype = "property"\ncheck = { kind = "finite" }\n')  # fmt: skip
    code, out, _ = run(capsys, "check", str(spec))
    assert code == 0 and "OK · 1 contracts · 1 cases" in out
    assert (
        run(capsys, "run", str(spec))[0] == 2
    )  # the model import only fails when actually running


def test_diff_command_gates_on_regressions(capsys, tmp_path):
    base, new = tmp_path / "base.json", tmp_path / "new.json"
    run(capsys, "run", LENDING, "--out-json", str(base))
    run(capsys, "run", LENDING, "--target", "assay.demos.lending:regressed", "--out-json", str(new))
    code, out, _ = run(capsys, "diff", str(base), str(new))
    assert code == 1 and "REGRESSED" in out and "3 regression(s)" in out
    assert run(capsys, "diff", str(base), str(new), "--report-only")[0] == 0
    assert run(capsys, "diff", str(new), str(base))[0] == 0  # fixing things is not a regression
    assert run(capsys, "diff", str(base), str(base))[0] == 0
    assert run(capsys, "diff", str(base), str(tmp_path / "missing.json"))[0] == 2


@pytest.mark.parametrize("name", list(DEMOS))
@pytest.mark.parametrize("variant", ["good", "regressed"])
def test_demo_command(capsys, name, variant):
    code, out, _ = run(capsys, "demo", name, "--variant", variant)
    assert code == 0 and name in out and f"assay.demos.{name}:{variant}" in out


def test_demo_without_a_name_runs_all(capsys):
    code, out, _ = run(capsys, "demo")
    assert code == 0 and all(f"━━ {name}" in out for name in DEMOS)
    assert run(capsys, "demo", "nonsense")[0] == 2


def test_list_command(capsys):
    code, out, _ = run(capsys, "list")
    assert code == 0 and "monotonicity" in out and "gender_swap" in out and "no_pii" in out
    assert "no_pii" not in run(capsys, "list", "transforms")[1]


def test_init_scaffolds_a_project_that_runs(capsys, tmp_path, monkeypatch):
    target = tmp_path / "proj"
    code, out, _ = run(capsys, "init", str(target))
    assert code == 0 and {p.name for p in target.iterdir()} == {
        "assay.toml",
        "model.py",
        "cases.jsonl",
    }
    monkeypatch.chdir(tmp_path)  # the spec directory is put on sys.path, not the cwd
    code, out, _ = run(capsys, "run", str(target / "assay.toml"))
    assert code == 0 and "FAIL" not in out
    code, _, err = run(capsys, "init", str(target))
    assert code == 2 and "refusing to overwrite" in err


def test_a_closed_stdout_pipe_exits_quietly():
    """`assay list | head` must not print a traceback (checked in a real subprocess)."""
    command = [sys.executable, "-m", "assay", "list"]
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as proc:
        assert proc.stdout is not None and proc.stderr is not None
        proc.stdout.close()  # the reader is gone before the child writes anything
        stderr = proc.stderr.read()
        assert proc.wait(timeout=30) == 0
    assert stderr == ""


def test_module_entry_point_and_console_script():
    env = {**os.environ, "NO_COLOR": "1"}
    done = subprocess.run(
        [sys.executable, "-m", "assay", "--version"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert done.stdout.startswith("assay ")
    done = subprocess.run(
        [sys.executable, "-m", "assay", "run", "/nonexistent.toml"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 2


def test_unwritable_report_paths_are_reported_cleanly(capsys, tmp_path):
    code, _, err = run(
        capsys, "run", LENDING, "--out-json", str(tmp_path / "no-such-dir" / "r.json")
    )
    assert code == 2 and "cannot write" in err


def test_reports_do_not_crash_on_consoles_that_cannot_encode_them(monkeypatch):
    import io

    raw = io.BytesIO()
    narrow = io.TextIOWrapper(raw, encoding="ascii", write_through=True)
    monkeypatch.setattr(sys, "stdout", narrow)
    assert main(["run", LENDING, "--no-color"]) == 0
    text = raw.getvalue().decode("ascii")
    assert "lending-model" in text and "PASS" in text and "?" in text  # "≤" etc. became "?"


def test_target_override_never_resolves_the_specs_own_model(capsys, tmp_path):
    """The spec's http model needs an env var that is not set; --target must not care."""
    spec = tmp_path / "s.toml"
    spec.write_text(
        '[model]\ntype = "http"\nurl = "http://127.0.0.1:1/x"\nheaders = { Authorization = "Bearer ${ASSAY_NOT_SET_XYZ}" }\n'
        '[[case]]\ninput = 4.0\n[[contract]]\ntype = "property"\nname = "p"\ncheck = { kind = "finite" }\n'
    )
    assert run(capsys, "run", str(spec))[0] == 2  # without the override it fails loudly
    code, out, _ = run(capsys, "run", str(spec), "--target", "math:sqrt")
    assert code == 0 and "PASS" in out


def test_target_override_still_resolves_agreement_references(capsys, tmp_path):
    spec = tmp_path / "s.toml"
    spec.write_text(
        '[model]\ntype = "python"\ntarget = "does.not.exist:f"\n[[case]]\ninput = 4.0\n'
        '[[contract]]\ntype = "agreement"\nname = "a"\nreference = { type = "python", target = "math:sqrt" }\n'
    )
    code, out, _ = run(capsys, "run", str(spec), "--target", "math:sqrt")
    assert code == 0 and "PASS" in out


@pytest.mark.parametrize("suite_line", ['seed = "abc"', "workers = 2.5", 'redact = "yes"'])
def test_malformed_suite_settings_exit_2_not_1(capsys, tmp_path, suite_line):
    spec = tmp_path / "s.toml"
    spec.write_text(
        f'[suite]\n{suite_line}\n[model]\ntype = "python"\ntarget = "math:sqrt"\n[[case]]\ninput = 1\n'
        '[[contract]]\ntype = "property"\ncheck = { kind = "finite" }\n'
    )
    code, _, err = run(capsys, "run", str(spec))
    assert code == 2 and "[suite]" in err
