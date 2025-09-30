# Contributing to Promptly

Thank you for your interest in contributing to Promptly! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and inclusive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/promptly.git
   cd promptly
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/leachtucker/promptly.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip or poetry for package management
- Git for version control

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the package in development mode**:
   ```bash
   pip install -e ".[dev,cli,ui]"
   ```

3. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

### Verify Installation

```bash
# Run tests
make test

# Run linting
make lint

# Run type checking
make type-check
```

## Making Changes

### Branch Strategy

- Create a new branch for each feature or bugfix
- Use descriptive branch names (e.g., `feature/add-new-client`, `fix/trace-error`)
- Keep branches focused on a single change

```bash
git checkout -b feature/your-feature-name
```

### Code Organization

The project follows tuhe layout:

```
promptly/
├── __init__.py
├── core/           # Core functionality
│   ├── clients.py  # LLM client implementations
│   ├── runner.py   # Prompt execution engine
│   ├── templates.py # Template management
│   └── tracer.py   # Observability and tracing
└── cli/            # Command-line interface
    └── main.py
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run tests with coverage
make test-coverage
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)
- Mock external dependencies

Example:
```python
def test_prompt_runner_execution():
    # Arrange
    mock_client = Mock()
    runner = PromptRunner(mock_client)
    
    # Act
    result = await runner.run(template, variables)
    
    # Assert
    assert result.content == expected_content
```

## Code Style

### Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
make format
```

### Linting

We use [flake8](https://flake8.pycqa.org/) for linting:

```bash
make lint
```

### Type Hints

We use [mypy](https://mypy.readthedocs.io/) for type checking:

```bash
make type-check
```

### Code Standards

1. **Type Hints**: Use type hints for all function parameters and return values
2. **Docstrings**: Use Google-style docstrings for all public functions and classes
3. **Error Handling**: Use specific exception types and provide meaningful error messages
4. **Async/Await**: Use async/await consistently for I/O operations
5. **Testing**: Maintain high test coverage (>90%)

## Submitting Changes

### Pull Request Process

1. **Ensure all checks pass**:
   ```bash
   make check  # Runs lint, type-check, and test
   ```

2. **Update documentation** if needed

3. **Create a pull request** with:
   - Clear title and description
   - Reference to any related issues
   - Screenshots for UI changes
   - Test instructions

4. **Respond to feedback** promptly

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for Claude 3.5 Sonnet
fix: resolve template variable parsing error
docs: update API documentation
test: add integration tests for batch processing
```

## Release Process

### Version Bumping

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped in `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] Release notes prepared

## Getting Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Email**: Contact the maintainer at leachtucker@gmail.com

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Promptly! 🚀
