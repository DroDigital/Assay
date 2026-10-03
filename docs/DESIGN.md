# Design

Why assay is built the way it is: the decisions, the alternatives rejected, and the limits.

## 1. Problem and goals

Teams ship models whose headline metric is healthy and whose *behaviour* is untested: the score
changes when someone types in capitals, depends on a protected attribute, leaks a customer's email,
or differs between two identical calls. Writing `assert model(x) == y` does not scale to that, and
fails outright for stochastic systems.

Goals, in priority order:

1. **Honest evidence.** Say how sure we are, and when we are not sure, say so.
2. **Model-agnostic.** A Python function, an HTTP endpoint, a CLI: anything that maps input to output.
3. **CI-native.** Deterministic runs, exit codes, JUnit, a regression gate.
4. **Cheap to adopt.** No dependencies, no services, readable specs, runs offline.

Non-goals: generating test inputs with an LLM, scoring open-ended text quality, replacing a
red-team.

## 2. Contracts as streams of trials

Every contract (invariance, monotonicity, property, golden, agreement, consistency, latency)
reduces to the same shape: *plan some trials; each trial either violates the contract or does not.*
So there is **one** statistical rule instead of eight ad-hoc ones, and every report, diff and CI
gate works for every contract.

This reduction is not free. Percentile latency becomes "the fraction of calls over the budget"
(`tolerance = 0.05` reads as p95), which is equivalent and easier to explain, but contracts about an
*aggregate* (group-level demographic parity) do not fit a per-trial violation and are not supported
yet.

## 3. The decision rule

For `k` violations in `n` trials, a Wilson score interval (Wilson, 1927) bounds the true violation
rate. Given a tolerance `τ` (the highest acceptable rate, default `0`):

| Condition | Verdict |
|---|---|
| lower bound > τ | `FAIL`: confidently above the tolerance |
| upper bound ≤ τ | `PASS`: confidently within the tolerance |
| otherwise | `INCONCLUSIVE` |

**Why Wilson.** It is closed-form, behaves at small `n` and at rates near 0 where AI suites live
(the textbook normal approximation gives nonsense there), and its coverage is easy to validate. A
test simulates 3000 experiments and asserts the 95% interval contains the truth 93–97% of the time.
Clopper-Pearson is more conservative; Bayesian intervals need a prior to defend. Both were rejected
for explainability.

**Zero tolerance is special.** No interval can certify a rate of exactly zero, so with `τ = 0` a
contract passes when *no violation was observed* and fails on any violation. The report still
prints the upper bound ("true rate ≤ 13.8%") so the strength of the claim stays visible, and
`min_trials` can force `INCONCLUSIVE` below a sample size you consider meaningful.

**Evidence has a size.** Zero violations out of `n` trials certifies a tolerance only when `n`
is large enough (95% / 99% confidence):

| Tolerance | 10% | 5% | 2% | 1% | 0.5% | 0.1% |
|---|---|---|---|---|---|---|
| trials with zero violations | 35 / 60 | 73 / 127 | 189 / 326 | 381 / 657 | 765 / 1321 | 3838 / 6629 |

For an `INCONCLUSIVE` result assay projects the same observed rate forward and reports how many
trials would decide it (`needs ≈332 trials`), so "run more" is an actionable instruction.

**`INCONCLUSIVE` does not fail the build by default**, because most teams would otherwise turn the
check off after the first false alarm. `--strict` (exit code 3) makes it a failure.

## 4. Architecture

```text
spec.py ─▶ contracts.py ─▶ runner.py ─▶ results.py ─▶ report.py / diff.py
 TOML/JSON    plan()          execute     immutable      text md json junit html
 validate     Trial list      decide()    dataclasses
                ▲
 transforms.py, checks.py, compare.py, paths.py      model.py (adapters, cache, timing)
```

**Plan, then execute.** `Contract.plan()` is a pure, single-threaded step that applies the
transforms with a seeded RNG and returns `Trial` objects. Execution only calls the model. Randomness
is therefore fixed before any concurrency, and results are identical for 1 or 32 workers (tested).
Each perturbation draws from `random.Random(f"{seed}|{contract}|{case}|{variant}")`; string seeds are
hashed with SHA-512, so the stream is stable across processes and Python versions.

**Trials that did not happen are not passes.** A counterfactual swap on a row without that field,
or a deterministic transform repeated by `variants`, yields the original input. The runner counts it
as *skipped* and says so. Counting no-ops as passes is the standard way a metamorphic suite reports
flattering numbers. If **no** trial applies, the verdict is `INCONCLUSIVE` with the reason, so a
misspelled field name cannot produce a green contract.

**Shortest counterexample first.** Violations are ranked by input size, then position, so the
example a human reads first is the easiest to understand.

**Caching.** Outputs are memoised per run by canonical JSON of the input. A baseline `f(x)` shared by
five contracts is paid for once, which matters when each call costs money. Consistency and latency
contracts bypass the cache by design.

**Latency contracts run serially**, even with many workers, so they do not contend with themselves.

## 5. Error policy

Three failure sources get three different treatments:

| Source | Example | Treatment |
|---|---|---|
| the model | exception, timeout, HTTP 503 after retries | a **violation** by default (`on_error`: `violation`, `skip`, `raise`) |
| the output's shape | `output = "score"` but the key is gone | a **violation**, with the available keys listed |
| your contract | a transform applied to the wrong input type, a check that raises | **aborts the run** with a located message |

A crash on a perturbed input *is* a broken promise, so it counts against the model. A bug in your own
test must never be quietly scored against the model.

## 6. Limits

* **Independence.** Wilson assumes independent trials. Several variants of one case are
  correlated, so intervals are somewhat optimistic when `variants` is large relative to the number
  of cases. Prefer more cases to more variants.
* **Multiple comparisons.** With many contracts, a contract whose true rate sits right on its
  tolerance has up to about a 2.5% chance of a spurious `FAIL`, and these add up across a large suite.
  Zero-tolerance contracts cannot fail by sampling noise alone: any violation is real. For large
  suites raise `confidence` (0.99 is a sensible default for gating).
* **Relations must be true in your domain.** assay will faithfully flag a model that changes its
  answer under a typo, but whether a typo *should* leave the answer unchanged is your call. Choose
  transforms and tolerances with someone who understands the domain.
* **Heuristic detectors.** `no_pii` and `no_secrets` are patterns plus checksums, not a DLP product.
* **Inputs are JSON-like.** Equality, caching and de-duplication use canonical JSON.

## 7. Security

* A spec file is configuration, but `type = "python"` imports code. Treat specs like scripts.
* HTTP models accept only `http(s)` URLs (no `file://`), send secrets via `${ENV_VAR}` expansion
  (a missing variable is an error, never sent literally), and never inherit a proxy for loopback.
* Command models are executed from an argument list, never through a shell.
* Model output is untrusted. Every value is rendered via `json.dumps` (escaping control and
  terminal-escape characters) so output cannot spoof verdict lines or recolour the terminal. Markdown
  code fences cannot be broken out of, HTML is escaped, and JUnit strips characters XML cannot carry.
* Sensitive-data checks report masked excerpts and **withhold the offending output** from every
  report. `redact = true` goes further and replaces all counterexample data with placeholders.

## 8. Alternatives considered

| Alternative | Why not |
|---|---|
| One `assert` per example | cannot express rates; flaky for stochastic models |
| A fixed pass-rate threshold ("95% must pass") | ignores sample size: 19/20 and 950/1000 look the same |
| A plugin architecture | premature; three small registries (contracts, transforms, checks) suffice |
| YAML specs | needs a dependency; TOML is in the standard library from Python 3.11 |
| LLM-generated perturbations | not reproducible, costs money, hard to reason about in CI |

## 9. Extending

* **A transform**: a function returning `Transform("name", fn)` where `fn(value, rng) -> value`;
  register it in `transforms.REGISTRY` to make it available in specs.
* **A check**: a function returning `None` or a reason; register in `checks.REGISTRY`. Detectors of
  sensitive data should return `Tagged(check, sensitive=True)`.
* **A contract**: subclass `Contract`, implement `plan()` returning `Trial` objects, and add it to
  `contracts.REGISTRY`. Spec parsing, reporting, diffing and the CLI work without further changes.

## References

* Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference.* JASA.
* Chen, T. Y., Cheung, S. C., Yiu, S. M. (1998). *Metamorphic testing: a new approach for generating
  next test cases.* HKUST technical report.
* Ribeiro, M. T., Wu, T., Guestrin, C., Singh, S. (2020). *Beyond accuracy: behavioral testing of NLP
  models with CheckList.* ACL.
