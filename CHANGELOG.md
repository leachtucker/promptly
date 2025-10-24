# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


### Added
- Initial release of Promptly
- Core prompt management functionality
- Support for OpenAI and Anthropic clients
- Jinja2-based template system
- SQLite-based tracing and observability
- Command-line interface
- Comprehensive test suite
- Development tools (Ruff, mypy, pytest)
- Documentation and examples

### Features
- **Prompt Templates**: Jinja2-based templating with variable substitution
- **LLM Clients**: Unified interface for OpenAI and Anthropic APIs
- **Tracing**: SQLite-based observability and performance tracking
- **CLI**: Command-line interface for prompt execution
- **Batch Processing**: Run prompts with multiple variable sets
- **Error Handling**: Comprehensive error tracking and reporting

### Technical Details
- Python 3.8+ support
- Async/await throughout
- Type hints for all public APIs
- 90%+ test coverage
- Modern Python packaging with pyproject.toml

## v0.2.0 (2025-10-23)

### Feat

- :green_heart: Package publishing
- :green_heart: Fix build failure
- :white_check_mark: Fix failing tests in CI
- :green_heart: Fix failing CI (run formatter)
- :bug: fix py dep
- :fire: Remove duped env util
- :construction_worker: Pull black dep
- :construction_worker: Swap to UV
- :fire: Remove print
- :sparkles: Add gem client. re-expose sdk create types as options
- :sparkles: Setup gemini API
- :sparkles: Run eval/test cases in parallel for perf
- :sparkles: Prefer elite ratio for usability
- :sparkles: Remove simple non-llm fallbacks
- :sparkles: Setup res parsing
- :fire: Remove simple fallback
- :sparkles: add tracing to optimizer
- :sparkles: CLI optimizer out clean up
- :sparkles: Swap to pydantic for models
- :sparkles: Prompt Optimizers - Genetic Algo
- :sparkles: Enhance CLI trace views
- :white_check_mark: Ensure tests use tmp db
- :sparkles: Add tracing viewing within CLI
- :sparkles: Fix bug with CLI runner
- :white_check_mark: Adjust API. Update tests
- :sparkles: Set up base API and tests

### Fix

- :green_heart: Missing UV run for build failure
- :green_heart: Fix pip cache failure in CI
- :white_check_mark: Mock clients for CLI tests
