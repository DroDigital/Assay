# Contributing

## Setup

```console
$ git clone https://github.com/DroDigital/1 && cd 1
$ python -m venv .venv && . .venv/bin/activate
$ pip install -e ".[dev]"
$ make check          # ruff, mypy --strict, pytest with coverage
```

`make demo` runs the built-in demos. Python 3.11 or newer; the library has no runtime dependencies
and that is a deliberate constraint: please do not add one.

## Standards

* **Tests first for behaviour.** Every change ships with tests; coverage is gated at 90%.
  Prefer an exact assertion to `a or b`: if the expected value is deterministic, assert it.
* **Types.** `mypy --strict` must pass.
* **Style.** `ruff format` and `ruff check`; lines up to 100 characters.
* **Docs are tested.** The README quickstart (marked `<!-- tested -->`) is executed and every TOML
  block in `docs/` is parsed; adding a contract, transform or check without documenting it in
  `docs/REFERENCE.md` fails the build.
* **Honesty in output.** A new feature must not turn missing evidence into a pass: skipped,
  not-applicable and inconclusive states must stay visible.

## Adding things

See [docs/DESIGN.md §9](docs/DESIGN.md#9-extending). In short: a transform or check is a function
plus a registry entry; a contract is a dataclass with a `plan()` method.

## Pull requests

Describe the behaviour change and the motivating failure. Keep changes focused. Run `make check`
first; CI runs the same commands on Python 3.11, 3.12 and 3.13.
