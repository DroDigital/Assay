.PHONY: install lint format typecheck test check demo build clean

install:
	python -m pip install -e ".[dev]"

lint:
	python -m ruff check .
	python -m ruff format --check .

format:
	python -m ruff format .
	python -m ruff check --fix .

typecheck:
	python -m mypy

test:
	python -m pytest --cov

check: lint typecheck test

demo:
	assay demo

build:
	python -m build

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov src/*.egg-info
