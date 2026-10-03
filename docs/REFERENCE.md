# Reference

Everything you can put in a spec file, plus the Python equivalents. Run `assay list` for the same
catalogue straight from the code. A test fails if a contract, transform or check is missing here.

## Spec file layout

A spec is TOML (or JSON with the same structure). Relative paths resolve against the spec's
directory.

```text
[suite]            optional
[model]            required
[[case]]           inline cases, and/or  [suite] cases_file = "cases.jsonl"
[[contract]]       at least one
```

### `[suite]`

| Key | Default | Meaning |
|---|---|---|
| `name` | spec directory name | shown in every report |
| `seed` | `0` | fixes every random perturbation; results never depend on `workers` |
| `confidence` | `0.95` | confidence level of the Wilson interval; a contract may override it |
| `workers` | `1` | concurrent model calls (latency contracts always run serially) |
| `on_error` | `"violation"` | how a crashing or unreachable model is scored: `violation`, `skip` or `raise` |
| `max_examples` | `3` | counterexamples kept per contract |
| `redact` | `false` | replace inputs and outputs in counterexamples with placeholders |
| `cases_file` | none | `.jsonl` (one `{"input": ..., "expected": ..., "id": ...}` per line) or a `.json` list |

### `[model]`

| `type` | Keys | Notes |
|---|---|---|
| `python` | `target = "package.module:function"` | the module is imported, so a spec is as trusted as code |
| `http` | `url`, `input_key`, `output_path`, `headers`, `timeout` (30), `retries` (2), `backoff` (0.5) | POSTs JSON; 408/425/429/5xx are retried with exponential backoff; only `http(s)` URLs; `${ENV_VAR}` in headers, a missing variable is an error |
| `command` | `command = ["program", "arg"]`, `timeout` (30) | JSON on stdin, JSON or text on stdout; run directly, never through a shell |

All types accept `cache = true` (per-run, in memory) so a baseline `f(x)` shared by several
contracts is computed once. Contracts that measure variation or latency bypass the cache.

`output` paths (`output = "choices.0.message.content"`) walk mappings by key and sequences by
index. A missing step is a *violation* ("the output schema changed"), with the available keys in the
message.

### `[[case]]`

`input` (required, any JSON value), `expected` (needed by `golden`), `id` (defaults to the position).

### `[[contract]]`

Common keys: `type`, `name`, `tolerance` (maximum acceptable violation *rate*, default `0`),
`confidence` (overrides the suite), `min_trials` (fewer applicable trials is `INCONCLUSIVE`).

Contracts that compare outputs take `compare = "exact" | "casefold" | "approx"` plus `abs_tol` /
`rel_tol` (which imply `approx`; numbers inside dicts and lists are compared recursively).
`output` selects the field to examine; without it the whole output is used.

| `type` | Required | Optional |
|---|---|---|
| `invariance` | `transform` | `variants` (1), `output`, `compare`, `abs_tol`, `rel_tol` |
| `sensitivity` | `transform` | as `invariance` |
| `monotonicity` | `field` | `direction` (`increasing`/`decreasing`), `output`, `steps` (`[0.1, 0.5]`), `mode` (`relative`/`absolute`), `slack` (0) |
| `property` | `check` | `transform`, `variants`, `output` |
| `golden` | none (cases need `expected`) | `output`, `compare`, `abs_tol`, `rel_tol` |
| `agreement` | `reference` (a `[model]`-style table) | `output`, `compare`, `abs_tol`, `rel_tol` |
| `consistency` |  | `repeats` (3), `output`, `compare`, `abs_tol`, `rel_tol` |
| `latency` | `budget_ms` | `repeats` (1) |

`direction = "increasing"` means *raising the field must not lower the output*; use `decreasing`
for features where more is worse (debt, missed payments).

`transform` and `check` take a table `{ kind = "...", ...options }`, or a **list** of tables
(transforms are applied in order; checks must all hold). Variadic options are lists:
`{ kind = "one_of", allowed = ["a", "b"] }`.

## Transforms

Text transforms accept `field = "name"` to act on one string field of a dict input. A transform that
changes nothing makes the trial *skipped*, never a pass.

| Kind | Options | Effect |
|---|---|---|
| `typo` | `rate` (0.05) | swapped, dropped, doubled or neighbouring-key letters; first letters kept |
| `case` | `mode` (`upper`, `lower`, `title`, `swap`) | change letter case |
| `whitespace` | | doubled spaces, tabs, newlines, leading and trailing blanks |
| `homoglyph` | `rate` (0.3) | Latin letters replaced by look-alike Cyrillic ones |
| `zero_width` | `rate` (0.15) | invisible zero-width spaces between letters |
| `replace` | `mapping`, `whole_word`, `ignore_case` | simultaneous, case-preserving substitution |
| `swap` | `mapping`, `both_ways`, `whole_word`, `ignore_case` | counterfactual substitution (names, regions, ...) |
| `gender_swap` | | he/she, man/woman, Mr/Ms, ... (`her` maps to `him`: a heuristic) |
| `suffix` | `texts` (list), `sep` | append one of the texts (distractors, prompt-injection probes) |
| `prefix` | `texts` (list), `sep` | prepend one of the texts |
| `set_field` | `field`, `value` or `choices` | set a field of a dict input |
| `swap_field` | `field`, `mapping` | map a categorical value, e.g. `{ F = "M", M = "F" }` |
| `shift_field` | `field`, `delta` | add to a numeric field |
| `scale_field` | `field`, `factor` | multiply a numeric field |
| `jitter_field` | `field`, `rel` (0.01) | multiply by `1 + U(-rel, rel)` |
| `drop_field` | `field` | remove a field (missing-data behaviour) |

## Checks

| Kind | Options | Passes when |
|---|---|---|
| `finite` | | a real, finite number (no NaN, infinity or bool) |
| `in_range` | `lo`, `hi` | a finite number within the bounds |
| `probability` | | a finite number in [0, 1] |
| `one_of` | `allowed` (list) | the value is in a closed set |
| `of_type` | `names` (list: `str`, `int`, `float`, `number`, `bool`, `dict`, `list`) | the type matches |
| `has_keys` | `keys` (list) | a mapping containing every key |
| `is_json` | | a string holding valid JSON |
| `max_length` / `min_length` | `limit` | length bound on strings or sequences |
| `matches` | `pattern`, `flags` | some string in the output matches |
| `not_matches` | `pattern`, `flags` | no string in the output matches |
| `no_pii` | `kinds` (`email`, `phone`, `ssn`, `credit_card`, `iban`) | no personal data found |
| `no_secrets` | `kinds` (`aws_access_key`, `github_token`, `private_key`, `jwt`, `api_key`, `slack_token`) | no credentials found |

`no_pii` and `no_secrets` are heuristics (patterns plus Luhn and mod-97 checksums). On failure they
report masked excerpts and **withhold the offending output** from every report format.

## Command line

```text
assay run SPEC [-f text|json|markdown|junit|html] [--out-json P] [--out-junit P] [--out-md P]
               [--out-html P] [--github-summary] [--target MODULE:FUNC] [--seed N]
               [--workers N] [--strict] [--redact] [--examples N] [--no-color]
assay diff BASELINE.json CANDIDATE.json [--report-only]
assay demo [lending|support|assistant] [--variant good|regressed]
assay init [DIR]          assay check SPEC          assay list [contracts|transforms|checks]
```

| Exit code | Meaning |
|---|---|
| 0 | every contract holds (or only inconclusive ones remain, without `--strict`) |
| 1 | a contract failed, or `diff` found a regression |
| 2 | usage or spec error |
| 3 | nothing failed, but some contracts are inconclusive and `--strict` is set |

`--target` runs the same contracts against a different Python model: the basis of "does the new
version regress?" when combined with `--out-json` and `assay diff`.

## Python API

```python
from assay import (Suite, Case, Invariance, Sensitivity, Monotonicity, Property, Golden,
                   Agreement, Consistency, Latency, checks, transforms, Verdict)

suite = Suite(name, model, cases, seed=0, confidence=0.95, workers=1,
              on_error="violation", max_examples=3, redact=False).add(contract, ...)
result = suite.run()                  # SuiteResult
result.ok(strict=False)               # bool
result.exit_code(strict=False)        # 0 / 1 / 3
result.assert_ok()                    # raises AssertionError carrying the text report
result.to_dict()                      # JSON-ready, schema "assay.result/v1"
print(result)                         # the text report
```

Contracts are keyword-friendly dataclasses: `Invariance("name", transform=..., variants=3,
output="label", compare=assay.approx(abs_tol=0.01), tolerance=0.02)`. `model` is any callable, a
`Model(fn, cache=False)`, or the result of `http_model(...)` / `command_model(...)`.
Custom transforms are `Transform("name", fn)` with `fn(value, rng) -> value`; custom checks are
functions returning `None` (ok) or a reason string.

## Result JSON (`assay.result/v1`)

```json
{
  "schema": "assay.result/v1",
  "tool": {"name": "assay", "version": "0.1.0"},
  "suite": {"name": "...", "seed": 7, "confidence": 0.95, "cases": 120},
  "summary": {"pass": 5, "fail": 2, "inconclusive": 0},
  "contracts": [{
    "name": "...", "kind": "invariance", "verdict": "fail",
    "trials": 104, "violations": 17, "skipped": 16, "errors": 0,
    "rate": 0.163, "ci_low": 0.105, "ci_high": 0.246,
    "tolerance": 0.0, "confidence": 0.95, "samples_to_decide": null,
    "metrics": {}, "counterexamples": [{"case": "applicant-043", "input": {}, "changed": {}}],
    "notes": [], "duration_s": 0.01
  }]
}
```
