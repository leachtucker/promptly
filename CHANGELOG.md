# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project structure migration to `src/` layout
- Comprehensive documentation structure
- CONTRIBUTING.md with detailed contribution guidelines
- CHANGELOG.md for tracking changes
- Enhanced package management with pyproject.toml

### Changed
- Moved core modules to `src/promptly/core/`
- Moved CLI to `src/promptly/cli/`
- Reorganized tests into `tests/unit/` and `tests/integration/`
- Updated import paths throughout codebase
- Enhanced Makefile for new structure

### Fixed
- Fixed mypy configuration typo
- Improved package installation with src layout
- Updated entry points for CLI

## [0.1.0] - 2024-09-28

### Added
- Initial release of Promptly
- Core prompt management functionality
- Support for OpenAI and Anthropic clients
- Jinja2-based template system
- SQLite-based tracing and observability
- Command-line interface
- Comprehensive test suite
- Development tools (Black, flake8, mypy, pytest)
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
