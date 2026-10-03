import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from assay.errors import ModelError, SpecError
from assay.model import Model, command_model, expand_env, fingerprint, http_model, import_target


def test_cache_serves_repeats_and_use_cache_false_bypasses_it():
    calls = []
    model = Model(lambda x: calls.append(x) or x * 2)
    assert model.call(2).output == 4 and model.call(2).cached is not False
    assert calls == [2]
    model.call(2, use_cache=False)
    assert calls == [2, 2]
    assert Model(lambda x: x, cache=False).call(1).cached is False


def test_cache_key_is_canonical_for_dicts():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    calls = []
    model = Model(lambda x: calls.append(1) or 1)
    model.call({"a": 1, "b": 2})
    model.call({"b": 2, "a": 1})
    assert len(calls) == 1


def test_timing_uses_the_injected_clock(clock):
    model = Model(lambda x: clock.advance(0.25), clock=clock)
    assert model.call("x", use_cache=False).seconds == pytest.approx(0.25)


def test_exceptions_become_model_errors_with_the_cause_attached():
    def broken(_):
        raise ValueError("boom")

    with pytest.raises(ModelError, match="ValueError: boom") as info:
        Model(broken).call(1)
    assert isinstance(info.value.__cause__, ValueError)


def test_failed_calls_are_not_cached():
    attempts = []

    def flaky(x):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("first call fails")
        return "ok"

    model = Model(flaky)
    with pytest.raises(ModelError):
        model.call(1)
    assert model.call(1).output == "ok"


def test_model_is_thread_safe_under_concurrent_use():
    model = Model(lambda x: x + 1)
    results = []
    threads = [
        threading.Thread(target=lambda i=i: results.append(model.call(i % 5).output))
        for i in range(200)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(set(results)) == [1, 2, 3, 4, 5]


def test_coerce_and_dunder_call():
    model = Model(len)
    assert Model.coerce(model) is model
    assert Model.coerce(len)("abc") == 3


def test_import_target_resolves_and_validates():
    assert import_target("math:sqrt")(9) == 3
    assert import_target("os.path:join") is not None
    for bad, fragment in [("nocolon", "must look like"), ("no.such.module:f", "cannot import"),
                          ("math:nope", "no attribute"), ("math:pi", "not callable")]:  # fmt: skip
        with pytest.raises(SpecError, match=fragment):
            import_target(bad)


def test_expand_env(monkeypatch):
    monkeypatch.setenv("ASSAY_T", "abc")
    assert expand_env("Bearer ${ASSAY_T}!") == "Bearer abc!"
    monkeypatch.delenv("ASSAY_MISSING", raising=False)
    with pytest.raises(SpecError, match="ASSAY_MISSING"):
        expand_env("${ASSAY_MISSING}")


# ------------------------------------------------------------------ http


class _Service:
    """A scripted local HTTP server: replies with the next status from `script`."""

    def __init__(self, script, body=None, location=None):
        self.script, self.requests = list(script), []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = self.rfile.read(int(self.headers["Content-Length"]))
                service.requests.append((self.headers.get("Authorization"), json.loads(payload)))
                status = service.script.pop(0) if service.script else 200
                raw = (
                    body if body is not None else json.dumps({"result": {"text": "hello", "n": 3}})
                )
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw.encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True
        ).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/x"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def service():
    made = []

    def make(script=(), body=None, location=None):
        made.append(_Service(script, body, location))
        return made[-1]

    yield make
    for item in made:
        item.close()


def test_http_model_posts_json_and_extracts_a_path(service, monkeypatch):
    monkeypatch.setenv("ASSAY_TOKEN", "s3cret")
    svc = service()
    call = http_model(
        svc.url,
        input_key="prompt",
        output_path="result.text",
        headers={"Authorization": "Bearer ${ASSAY_TOKEN}"},
    )
    assert call("hi") == "hello"
    assert svc.requests == [("Bearer s3cret", {"prompt": "hi"})]


def test_http_model_without_wrapping_returns_the_whole_body(service):
    svc = service()
    assert http_model(svc.url)({"a": 1}) == {"result": {"text": "hello", "n": 3}}
    assert svc.requests[0][1] == {"a": 1}


def test_http_model_returns_plain_text_bodies(service):
    assert http_model(service(body="just text").url)("x") == "just text"


def test_http_model_retries_transient_errors_with_backoff(service):
    svc = service([503, 429])
    sleeps = []
    assert http_model(svc.url, retries=2, backoff=0.5, sleep=sleeps.append)("x") == {
        "result": {"text": "hello", "n": 3}
    }
    assert len(svc.requests) == 3 and sleeps == [0.5, 1.0]


def test_http_model_gives_up_after_retries(service):
    svc = service([500, 500, 500])
    with pytest.raises(ModelError, match="HTTP 500"):
        http_model(svc.url, retries=2, sleep=lambda s: None)("x")
    assert len(svc.requests) == 3


def test_http_model_does_not_retry_client_errors(service):
    svc = service([404])
    with pytest.raises(ModelError, match="HTTP 404"):
        http_model(svc.url, retries=3, sleep=lambda s: None)("x")
    assert len(svc.requests) == 1


def test_http_model_reports_unreachable_hosts():
    with pytest.raises(ModelError, match="cannot reach"):
        http_model("http://127.0.0.1:1/x", retries=1, sleep=lambda s: None, timeout=1)("x")


def test_http_model_rejects_non_http_schemes_and_missing_env():
    for url in ("file:///etc/passwd", "ftp://example.com/x", "/relative", "http://"):
        with pytest.raises(SpecError, match="http"):
            http_model(url)
    with pytest.raises(SpecError, match="NOT_SET_ANYWHERE"):
        http_model("http://localhost/x", headers={"Authorization": "${NOT_SET_ANYWHERE}"})


# ------------------------------------------------------------------ command


def _py(code):
    return [sys.executable, "-c", code]


def test_command_model_speaks_json_over_stdio():
    echo = _py(
        "import sys, json; d = json.load(sys.stdin); print(json.dumps({'double': d['n'] * 2}))"
    )
    assert command_model(echo)({"n": 21}) == {"double": 42}


def test_command_model_falls_back_to_plain_text():
    assert command_model(_py("print('plain output')"))("x") == "plain output"


def test_command_model_failures():
    with pytest.raises(ModelError, match="exited with 3: oops"):
        command_model(_py("import sys; sys.stderr.write('oops'); sys.exit(3)"))("x")
    with pytest.raises(ModelError, match="not found"):
        command_model(["definitely-not-a-real-binary-xyz"])("x")
    with pytest.raises(ModelError, match="timed out"):
        command_model(_py("import time; time.sleep(5)"), timeout=0.3)("x")
    for bad in ([], [1, 2], "string"):
        with pytest.raises(SpecError):
            command_model(bad)


def test_model_errors_from_adapters_are_not_wrapped_twice():
    adapter = http_model("http://127.0.0.1:1/x", retries=0, timeout=1)
    with pytest.raises(ModelError) as info:
        Model(adapter).call("x")
    assert str(info.value).startswith("cannot reach") and info.value.__cause__ is not None


def test_http_responses_larger_than_the_cap_are_rejected(service):
    svc = service(body="x" * 500)
    with pytest.raises(ModelError, match="exceeds 100 bytes"):
        http_model(svc.url, max_bytes=100)("hi")
    assert http_model(svc.url, max_bytes=500)("hi") == "x" * 500


def test_redirects_are_never_followed_so_credentials_cannot_leak(service, monkeypatch):
    monkeypatch.setenv("ASSAY_TOKEN", "s3cret")
    elsewhere = service()  # a different host that must never see the request
    origin = service([307], location=elsewhere.url)
    call = http_model(
        origin.url,
        headers={"Authorization": "Bearer ${ASSAY_TOKEN}"},
        retries=3,
        sleep=lambda s: None,
    )
    with pytest.raises(ModelError, match="HTTP 307 redirect.*never forwarded"):
        call("x")
    assert len(origin.requests) == 1 and elsewhere.requests == []
