"""The shipped examples must keep working: they are the first thing a reader tries."""

import importlib.util
import threading
import tomllib
from pathlib import Path

import pytest

from assay import Verdict
from assay.spec import load_spec, parse_spec

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load_server_module():
    spec = importlib.util.spec_from_file_location(
        "http_service_server", EXAMPLES / "http_service" / "server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_against_server(monkeypatch):
    server_module = load_server_module()
    servers = []

    def run(variant, token="demo-token"):
        server = server_module.make_server(variant, port=0)
        servers.append(server)
        threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
        ).start()
        monkeypatch.setenv("ASSAY_DEMO_TOKEN", token)
        data = tomllib.loads((EXAMPLES / "http_service" / "assay.toml").read_text())
        data["model"]["url"] = f"http://127.0.0.1:{server.server_address[1]}/v1/respond"
        data["model"]["backoff"] = 0.0
        return parse_spec(data, base_dir=EXAMPLES / "http_service").to_suite().run()

    yield run
    for server in servers:
        server.shutdown()
        server.server_close()


def test_the_example_specs_are_valid():
    assert load_spec(EXAMPLES / "http_service" / "assay.toml", resolve=False).workers == 4


def test_a_healthy_remote_service_honours_its_contracts(run_against_server):
    result = run_against_server("good")
    assert result.ok(strict=False)
    assert all(c.verdict is not Verdict.FAIL for c in result.contracts)


def test_a_regressed_remote_service_is_caught_over_http(run_against_server):
    result = run_against_server("regressed")
    verdicts = {c.name: c.verdict for c in result.contracts}
    assert verdicts["no-personal-data-in-answers"] is Verdict.FAIL
    assert verdicts["no-credentials-in-answers"] is Verdict.FAIL
    assert verdicts["identical-calls-identical-answers"] is Verdict.FAIL


def test_a_wrong_token_surfaces_as_model_errors_not_as_a_silent_pass(run_against_server):
    result = run_against_server("good", token="wrong-token")
    assert all(c.errors == c.trials == 4 for c in result.contracts)
    assert not result.ok()
    assert "HTTP 401" in result.contracts[0].counterexamples[0]["reason"]
