"""
Promptly - A lightweight library for LLM prompt management, observability/tracing, and optimization
"""

__version__ = "0.1.0"
__author__ = "Tucker Leach"
__email__ = "leachtucker@gmail.com"

# Load environment variables from .env file
from .core.utils.env import load_env_for_promptly
load_env_for_promptly()

from .core.runner import PromptRunner
from .core.templates import PromptTemplate, PromptMetadata
from .core.clients import BaseLLMClient, LLMResponse, OpenAIClient, AnthropicClient
from .core.tracer import Tracer, TraceRecord
from .core.optimizer import (
    LLMGeneticOptimizer,
    LLMAccuracyFitnessFunction,
    LLMSemanticFitnessFunction,
    TestCase,
    OptimizationResult,
    FitnessEvaluation,
)

__all__ = [
    "PromptRunner",
    "PromptTemplate",
    "PromptMetadata",
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "LLMResponse",
    "Tracer",
    "TraceRecord",
    "LLMGeneticOptimizer",
    "LLMAccuracyFitnessFunction",
    "LLMSemanticFitnessFunction",
    "TestCase",
    "OptimizationResult",
    "FitnessEvaluation",
]
