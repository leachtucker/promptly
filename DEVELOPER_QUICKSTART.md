# Developer Quick Start

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

- **Format**: `ruff format promptly/ tests/`
- **Lint**: `ruff check promptly/ tests/`
- **Type Check**: `mypy promptly/`

## 🔧 Development Workflow

1. **Setup**: `uv pip install -e .[dev,cli,ui]`
2. **Make changes** to code
3. **Test**: `pytest`
4. **Format**: `ruff format promptly/ tests/` (or `make format`)
5. **Lint**: `ruff check promptly/ tests/` (or `make lint`)
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
uv pip show promptly

# Test CLI
promptly --help
```

## 🚨 Common Issues

1. **Import errors**: Ensure you've run `uv pip install -e .[dev,cli,ui]`
2. **Test failures**: Run `make install-dev` first
3. **CLI not found**: Install with `uv pip install -e .[cli]`
4. **UV not found**: Install UV with `curl -LsSf https://astral.sh/uv/install.sh | sh`

## 📦 Package Management

UV is a fast Python package installer and resolver, 10-100x faster than pip.

```bash
# Install in development mode
uv pip install -e .

# Install with all optional dependencies
uv pip install -e .[dev,cli,ui]

# Build package
python -m build

# Install built package
uv pip install dist/*.whl

# Create/sync virtual environment (UV handles this automatically)
# UV will create .venv if it doesn't exist when you run uv pip install
```
