"""A tiny HTTP service around the demo assistant, to show assay testing a *remote* model.

    python examples/http_service/server.py --variant regressed &
    ASSAY_DEMO_TOKEN=demo-token assay run examples/http_service/assay.toml

``POST /v1/respond`` with ``{"prompt": "..."}`` and a bearer token returns
``{"text": "...", "refused": false}``. Any real service with a JSON API works the same way:
only the ``[model]`` table of the spec changes.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from assay.demos import assistant

TOKEN = "demo-token"


def make_server(
    variant: str = "good", host: str = "127.0.0.1", port: int = 0
) -> ThreadingHTTPServer:
    """Build (but do not start) the server. ``port=0`` picks a free port."""
    model = getattr(assistant, variant)

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                return self._reply(401, {"error": "missing or invalid bearer token"})
            if self.path != "/v1/respond":
                return self._reply(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                prompt = json.loads(self.rfile.read(length))["prompt"]
            except (ValueError, KeyError, TypeError):
                return self._reply(400, {"error": 'expected {"prompt": "..."}'})
            return self._reply(200, model(str(prompt)))

        def log_message(self, format: str, *args: Any) -> None:  # keep test output quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cli.add_argument("--variant", choices=("good", "regressed"), default="good")
    cli.add_argument("--port", type=int, default=8765)
    options = cli.parse_args()
    server = make_server(options.variant, port=options.port)
    print(f"serving the {options.variant} assistant on http://127.0.0.1:{options.port}/v1/respond")
    server.serve_forever()
