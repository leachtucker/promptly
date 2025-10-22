"""
Environment variable loading utilities
"""

import os
from typing import Optional

from dotenv import load_dotenv


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
    - GEMINI_API_KEY
    - PROMPTLY_DB_PATH (for tracer database)
    - PROMPTLY_LOG_LEVEL
    - PROMPTLY_TRACING_ENABLED
    """

    load_dotenv()

    # Set default values for promptly-specific variables if not already set
    if not os.getenv("PROMPTLY_DB_PATH"):
        os.environ["PROMPTLY_DB_PATH"] = "promptly_traces.db"
    if not os.getenv("PROMPTLY_LOG_LEVEL"):
        os.environ["PROMPTLY_LOG_LEVEL"] = "INFO"
    if not os.getenv("PROMPTLY_TRACING_ENABLED"):
        os.environ["PROMPTLY_TRACING_ENABLED"] = "false"
