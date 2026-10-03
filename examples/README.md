# Examples

| Example | Shows |
|---|---|
| [`http_service/`](http_service) | testing a **remote** model over HTTP, with bearer-token auth from an environment variable and concurrent requests |
| [`pytest_integration/`](pytest_integration) | contracts as ordinary **pytest** tests, with `result.assert_ok()` |

The three built-in demos (a healthy and a regressed model each) live in
[`src/assay/demos`](../src/assay/demos); each one is a plain `assay.toml` you can read and copy:

```console
$ assay demo lending      # finance: monotonicity, counterfactual fairness, stability
$ assay demo support      # customer operations: case, whitespace, typos, Unicode, injection
$ assay demo assistant    # generative AI: PII and secret leaks, injection, flakiness, latency
```

Run the HTTP example:

```console
$ python examples/http_service/server.py --variant regressed &
$ ASSAY_DEMO_TOKEN=demo-token assay run examples/http_service/assay.toml
```
