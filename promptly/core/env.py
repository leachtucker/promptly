"""
Environment variable loading utilities
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def load_env_file(env_file: Optional[str] = None, override: bool = False) -> bool:
    """Load environment variables from a .env file

    Args:
        env_file: Path to the .env file. If None, searches for .env in current and parent directories
        override: Whether to override existing environment variables

    Returns:
        True if .env file was loaded successfully, False otherwise
    """
    if load_dotenv is None:
        raise ImportError("python-dotenv is required. Install it with: pip install python-dotenv")

    if env_file is None:
        # Search for .env file in current directory and parent directories
        current_dir = Path.cwd()
        for directory in [current_dir] + list(current_dir.parents):
            env_path = directory / ".env"
            if env_path.exists():
                env_file = str(env_path)
                break
        else:
            return False  # No .env file found

    # Load the .env file
    return load_dotenv(env_file, override=override)


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get an environment variable with optional validation

    Args:
        key: Environment variable name
        default: Default value if variable is not set
        required: Whether the variable is required (raises error if not set)

    Returns:
        Environment variable value or default

    Raises:
        ValueError: If required variable is not set
    """
    value = os.getenv(key, default)

    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")

    return value


def load_env_for_promptly() -> None:
    """Load environment variables specifically for promptly usage

    This function loads common API keys and configuration for promptly:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - PROMPTLY_DB_PATH (for tracer database)
    - PROMPTLY_LOG_LEVEL
    """
    load_env_file()

    # Set default values for promptly-specific variables if not already set
    if not os.getenv("PROMPTLY_DB_PATH"):
        os.environ["PROMPTLY_DB_PATH"] = "promptly_traces.db"

    if not os.getenv("PROMPTLY_LOG_LEVEL"):
        os.environ["PROMPTLY_LOG_LEVEL"] = "INFO"
