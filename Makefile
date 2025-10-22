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
	flake8 promptly/ tests/
	black --check promptly/ tests/

format:  ## Format code
	black promptly/ tests/

type-check:  ## Run type checking
	mypy promptly/

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
	python -m build

install-build:  ## Install the built package
	uv pip install dist/*.whl

docs:  ## Generate documentation (if sphinx is configured)
	@echo "Documentation generation not yet configured"

check: lint type-check test  ## Run all checks (lint, type-check, test)

ci:  ## Run CI pipeline locally
	uv pip install -e .[dev,cli,ui]
	black --check promptly/ tests/
	flake8 promptly/ tests/
	mypy promptly/
	pytest --cov=promptly --cov-report=term

# Development shortcuts
run-cli:  ## Run the CLI
	promptly --help

run-tests: test  ## Alias for test

# Database cleanup
clean-db:  ## Clean up test databases
	find . -name "*.db" -delete
	find . -name "promptly_traces.db" -delete
