# Makefile for promptly development

.PHONY: help install install-dev test test-unit test-integration lint format type-check clean build docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	uv pip install -e .

install-dev:  ## Install package with development dependencies
	uv pip install -e .[dev,cli,ui]

test:  ## Run all tests
	pytest

test-unit:  ## Run unit tests only
	pytest -m "not integration"

test-integration:  ## Run integration tests only
	pytest -m integration

test-coverage:  ## Run tests with coverage report
	pytest --cov=promptly --cov-report=html --cov-report=term

test-verbose:  ## Run tests with verbose output
	pytest -v -s

lint:  ## Run linting
	ruff check promptly/ tests/
	ruff format --check promptly/ tests/

format:  ## Format code
	ruff format promptly/ tests/
	ruff check --fix promptly/ tests/

type-check:  ## Run type checking
	pyright promptly/

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	uv run python -m build

install-build:  ## Install the built package
	uv pip install dist/*.whl

docs:  ## Generate documentation (if sphinx is configured)
	@echo "Documentation generation not yet configured"

check: lint type-check test  ## Run all checks (lint, type-check, test)

ci:  ## Run CI pipeline locally
	uv pip install -e .[dev,cli,ui]
	ruff format --check promptly/ tests/
	ruff check promptly/ tests/
	pyright promptly/
	pytest --cov=promptly --cov-report=term

# Development shortcuts
run-cli:  ## Run the CLI
	promptly --help

run-tests: test  ## Alias for test

# Version management and releases
release-patch:  ## Release a patch version (0.1.0 -> 0.1.1)
	uv run cz bump --increment PATCH --yes
	git push --follow-tags

release-minor:  ## Release a minor version (0.1.0 -> 0.2.0)
	uv run cz bump --increment MINOR --yes
	git push --follow-tags

release-major:  ## Release a major version (0.1.0 -> 1.0.0)
	uv run cz bump --increment MAJOR --yes
	git push --follow-tags

release:  ## Auto-detect version bump based on commits
	uv run cz bump --yes
	git push --follow-tags

# Pre-commit hooks
pre-commit-install:  ## Install pre-commit hooks
	pre-commit install

pre-commit-run:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Package publishing
publish-test:  ## Publish to Test PyPI
	uv run python -m build
	uv run twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI (manual fallback)
	@echo "Warning: This will publish to PyPI. Use with caution!"
	@echo "Recommended: Use git tags (e.g., 'git tag v0.1.0 && git push --tags') to trigger automated release."
	@read -p "Are you sure you want to manually publish? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		uv run python -m build && uv run twine upload dist/*; \
	fi
