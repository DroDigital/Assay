.PHONY: install lint format typecheck test check demo build clean

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy

test:
	python -m pytest --cov

check: lint typecheck test

demo:
	assay demo

build:
	python -m build

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov src/*.egg-info
