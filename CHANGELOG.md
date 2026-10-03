# Changelog

## 0.1.0

First release.

* Eight contracts: invariance, sensitivity, monotonicity, property, golden, agreement,
  consistency, latency.
* One statistical decision rule: Wilson interval, three verdicts (`PASS`, `FAIL`,
  `INCONCLUSIVE`) and an estimate of the trials needed to decide.
* 16 input transforms and 13 output checks; Luhn- and mod-97-validated PII detectors and a
  credential detector that withholds the offending output.
* Python, HTTP and command-line model adapters with retries, caching and timing.
* TOML/JSON specs with located, suggestion-bearing validation errors.
* Text, Markdown, JSON, JUnit and self-contained HTML reports; `assay diff` for regression gates.
* Built-in offline demos (lending, support, assistant) and `assay init`.
