# Developer Quick Start

## 🚀 One-Command Setup

```bash
# Complete development setup
python dev.py install && python dev.py test
```

## 📋 Common Commands

### Using the dev script (recommended)
```bash
python dev.py check          # Check environment
python dev.py install        # Install with dev dependencies
python dev.py test           # Run all tests
python dev.py test-unit      # Run unit tests only
python dev.py test-coverage  # Run tests with coverage
python dev.py lint           # Run code quality checks
python dev.py format         # Format code
python dev.py clean          # Clean build artifacts
python dev.py ci             # Run full CI pipeline
```

### Using Make (alternative)
```bash
make help                    # Show all commands
make install-dev             # Install with dev dependencies
make test                    # Run all tests
make test-coverage           # Run tests with coverage
make lint                    # Run linting
make format                  # Format code
make clean                   # Clean build artifacts
make ci                      # Run CI pipeline
```

### Using pytest directly
```bash
pytest                       # Run all tests
pytest -v                    # Verbose output
pytest -m "not integration"  # Unit tests only
pytest -m integration        # Integration tests only
pytest --cov=promptly        # With coverage
pytest tests/test_runner.py  # Specific test file
```

## 🏗️ Project Structure

```
promptly/
├── promptly/           # Main package
│   ├── __init__.py    # Package exports
│   ├── cli.py         # CLI interface
│   └── core/          # Core functionality
├── tests/             # Test suite
├── pyproject.toml     # Package configuration
├── setup.py           # Legacy setup
├── Makefile           # Development commands
├── dev.py             # Development script
└── README.md          # User documentation
```

## 🧪 Testing

- **Unit Tests**: `pytest -m "not integration"`
- **Integration Tests**: `pytest -m integration`
- **All Tests**: `pytest`
- **Coverage**: `pytest --cov=promptly --cov-report=html`

## 📝 Code Quality

- **Format**: `black promptly/ tests/`
- **Lint**: `flake8 promptly/ tests/`
- **Type Check**: `mypy promptly/`

## 🔧 Development Workflow

1. **Setup**: `python dev.py install`
2. **Make changes** to code
3. **Test**: `python dev.py test`
4. **Format**: `python dev.py format`
5. **Lint**: `python dev.py lint`
6. **Commit** your changes

## 📚 Documentation

- **[README.md](README.md)** - User documentation
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Complete development guide
- **[TESTING.md](TESTING.md)** - Detailed testing guide

## 🐛 Debugging

```bash
# Run tests with debug output
pytest -v -s

# Run specific test with debugger
pytest --pdb tests/test_runner.py::TestPromptRunner::test_runner_run_success

# Check package installation
pip show promptly

# Test CLI
promptly --help
```

## 🚨 Common Issues

1. **Import errors**: Ensure virtual environment is activated
2. **Test failures**: Run `python dev.py install` first
3. **CLI not found**: Install with `pip install -e .[cli]`
4. **Database errors**: Check file permissions for SQLite files

## 📦 Package Management

```bash
# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e .[dev,cli,ui]

# Build package
python -m build

# Install built package
pip install dist/*.whl
```
