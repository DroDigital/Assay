"""The system under test, wrapped so every kind of model looks the same to a contract.

A model is any ``Callable[[input], output]``. Three adapters cover the common deployments:
:func:`import_target` (a Python function), :func:`http_model` (a JSON endpoint) and
:func:`command_model` (a program speaking JSON on stdin/stdout).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ModelError, SpecError
from .paths import get_path

ModelFn = Callable[[Any], Any]


def fingerprint(value: Any) -> str:
    """Canonical string for a JSON-like value; used for cache keys and de-duplication."""
    return json.dumps(value, sort_keys=True, default=repr, ensure_ascii=False)


@dataclass(frozen=True)
class Call:
    """One model invocation: its output and how long the real call took."""

    output: Any
    seconds: float
    cached: bool = False


class Model:
    """Thread-safe wrapper adding timing, error normalisation and an optional per-run cache.

    The cache stores outputs by canonical input, so a baseline ``f(x)`` shared by many contracts
    is paid for once (it matters when each call costs money). Contracts that measure variation
    or latency bypass it with ``use_cache=False``.
    """

    def __init__(
        self,
        fn: ModelFn,
        *,
        name: str | None = None,
        cache: bool = True,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._fn = fn
        self.name = name or getattr(fn, "__name__", type(fn).__name__)
        self._cache: dict[str, Any] | None = {} if cache else None
        self._lock = threading.Lock()
        self._clock = clock

    @classmethod
    def coerce(cls, model: ModelFn | Model) -> Model:
        return model if isinstance(model, Model) else cls(model)

    def call(self, value: Any, *, use_cache: bool = True) -> Call:
        key = fingerprint(value) if self._cache is not None and use_cache else None
        if key is not None and self._cache is not None:
            with self._lock:
                if key in self._cache:
                    return Call(self._cache[key], 0.0, cached=True)
        started = self._clock()
        try:
            output = self._fn(value)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        elapsed = self._clock() - started
        if key is not None and self._cache is not None:
            with self._lock:
                self._cache[key] = output
        return Call(output, elapsed)

    def __call__(self, value: Any) -> Any:
        return self.call(value).output


# --------------------------------------------------------------------------- adapters


def import_target(target: str) -> ModelFn:
    """Resolve ``"package.module:function"`` to a callable.

    Importing executes code. Spec files are therefore as trusted as any Python file you run.
    """
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise SpecError(f"model target must look like 'package.module:function', got '{target}'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SpecError(
            f"cannot import '{module_name}' for model target '{target}': {exc}"
        ) from exc
    obj: Any = module
    for part in attribute.split("."):
        if not hasattr(obj, part):
            raise SpecError(f"module '{module_name}' has no attribute '{attribute}'")
        obj = getattr(obj, part)
    if not callable(obj):
        raise SpecError(f"model target '{target}' is not callable")
    return obj  # type: ignore[no-any-return]


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(text: str) -> str:
    """Replace ``${NAME}`` with the environment variable; a missing variable is an error.

    Secrets such as API tokens stay out of spec files, and a missing one fails loudly instead
    of sending the literal string ``${TOKEN}`` to a server.
    """

    def lookup(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise SpecError(f"environment variable '{name}' is not set")
        return os.environ[name]

    return _ENV.sub(lookup, text)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow redirects: urllib would re-send the headers (API tokens) to the new host,
    and re-POST the body on 307/308."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_RETRYABLE = {408, 425, 429, 500, 502, 503, 504}
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def http_model(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    input_key: str | None = None,
    output_path: str | None = None,
    timeout: float = 30.0,
    retries: int = 2,
    backoff: float = 0.5,
    max_bytes: int = 10_000_000,
    sleep: Callable[[float], None] = time.sleep,
) -> ModelFn:
    """POST each input as JSON to ``url`` and return the parsed JSON response.

    ``input_key`` wraps the input as ``{input_key: input}``; ``output_path`` selects a field of
    the response. Rate limits and transient 5xx errors are retried with exponential backoff.
    Only ``http`` and ``https`` URLs are accepted (no ``file://``), redirects are never followed
    (so ``headers`` cannot leak to another host), and responses larger than ``max_bytes`` are
    rejected rather than held in memory.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise SpecError(f"model url must be an absolute http(s) URL, got '{url}'")
    resolved = {key: expand_env(value) for key, value in (headers or {}).items()}
    handlers: list[Any] = [_NoRedirect]
    if parsed.hostname in _LOOPBACK:
        handlers.append(urllib.request.ProxyHandler({}))  # never proxy local test servers
    opener = urllib.request.build_opener(*handlers)

    def call(value: Any) -> Any:
        body = json.dumps(value if input_key is None else {input_key: value}).encode()
        request = urllib.request.Request(  # noqa: S310 - scheme validated above
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json", **resolved},
        )
        for attempt in range(retries + 1):
            try:
                with opener.open(request, timeout=timeout) as response:
                    payload = response.read(max_bytes + 1)
                if len(payload) > max_bytes:
                    raise ModelError(f"response from {url} exceeds {max_bytes} bytes")
                raw = payload.decode("utf-8", errors="replace")
                break
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRYABLE and attempt < retries:
                    sleep(backoff * 2**attempt)
                    continue
                if 300 <= exc.code < 400:
                    raise ModelError(
                        f"HTTP {exc.code} redirect from {url}: redirects are not followed so "
                        "credentials are never forwarded; use the final URL"
                    ) from exc
                raise ModelError(f"HTTP {exc.code} from {url}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt < retries:
                    sleep(backoff * 2**attempt)
                    continue
                raise ModelError(f"cannot reach {url}: {exc}") from exc
        try:
            parsed_body: Any = json.loads(raw)
        except ValueError:
            parsed_body = raw
        return get_path(parsed_body, output_path) if output_path else parsed_body

    return call


def command_model(argv: list[str], *, timeout: float = 30.0) -> ModelFn:
    """Run ``argv`` once per input: JSON on stdin, JSON (or plain text) on stdout.

    The command is executed directly, never through a shell.
    """
    if not isinstance(argv, list) or not argv or not all(isinstance(part, str) for part in argv):
        raise SpecError(
            'command must be a non-empty list of strings, e.g. ["python", "model.py"] '
            "(it is run directly, not through a shell)"
        )

    def call(value: Any) -> Any:
        try:
            done = subprocess.run(  # noqa: S603 - argv list, no shell
                argv, input=json.dumps(value), capture_output=True, text=True,
                timeout=timeout, check=False,
            )  # fmt: skip
        except FileNotFoundError as exc:
            raise ModelError(f"command not found: {argv[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ModelError(f"command timed out after {timeout}s") from exc
        if done.returncode != 0:
            tail = done.stderr.strip()[-300:]
            raise ModelError(f"command exited with {done.returncode}: {tail}")
        text = done.stdout.strip()
        try:
            return json.loads(text)
        except ValueError:
            return text

    return call
