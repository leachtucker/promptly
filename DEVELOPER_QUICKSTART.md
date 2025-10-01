# Developer Quick Start

## 🚀 One-Command Setup

```bash
# Complete development setup
python dev.py install && python dev.py test
```

## 📋 Common Commands


### Using Make 
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
├── promptly/          # Main package
│   ├── __init__.py    # Package exports
│   ├── cli/           # CLI interface
│   └── core/          # Core functionality
├── tests/             # Test suite
├── pyproject.toml     # Package configuration
├── Makefile           # Development commands
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

- **[Changelog](CHANGELOG.md)**

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
2. **Test failures**: Run `make install-dev` first
3. **CLI not found**: Install with `pip install -e .[cli]`

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
