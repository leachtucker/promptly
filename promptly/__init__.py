"""
Promptly - A lightweight library for LLM prompt management, observability/tracing, and optimization
"""

__version__ = "0.1.0"
__author__ = "Tucker Leach"
__email__ = "leachtucker@gmail.com"

# Load environment variables from .env file
from .core.utils.env import load_env_for_promptly

load_env_for_promptly()

# Import after env loading to ensure environment is set up first
from .core.client_types import AnthropicOptions, GoogleAIOptions, OpenAIOptions  # noqa: E402
from .core.clients import AnthropicClient, BaseLLMClient, LLMResponse, OpenAIClient  # noqa: E402
from .core.optimizer import (  # noqa: E402
    FitnessEvaluation,
    LLMComprehensiveFitnessFunction,
    LLMGeneticOptimizer,
    OptimizationResult,
    PromptTestCase,
)
from .core.runner import PromptRunner  # noqa: E402
from .core.templates import PromptMetadata, PromptTemplate  # noqa: E402
from .core.tracer import Tracer, TraceRecord  # noqa: E402

__all__ = [
    "PromptRunner",
    "PromptTemplate",
    "PromptMetadata",
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "LLMResponse",
    "OpenAIOptions",
    "AnthropicOptions",
    "GoogleAIOptions",
    "Tracer",
    "TraceRecord",
    "LLMGeneticOptimizer",
    "LLMComprehensiveFitnessFunction",
    "PromptTestCase",
    "OptimizationResult",
    "FitnessEvaluation",
]
