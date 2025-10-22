"""
Type re-exports from provider SDKs for type-safe client options.

This module re-exports the actual TypedDict types from each provider's SDK,
ensuring full type safety and compatibility with IDE autocomplete.

The types are aliased for cleaner imports and consistent naming across the codebase.
"""

# Re-export OpenAI types
# These come directly from the openai SDK and include all parameters for chat.completions.create()
# Re-export Anthropic types
# These come directly from the anthropic SDK and include all parameters for messages.create()
from anthropic.types.message_create_params import (
    MessageCreateParamsBase as AnthropicOptions,
)

# Re-export Google AI types
# These come directly from the google-genai SDK and include all parameters for generate_content()
from google.genai.types import (
    GenerateContentConfigDict as GoogleAIOptions,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase as OpenAIOptions,
)

__all__ = [
    "OpenAIOptions",
    "AnthropicOptions",
    "GoogleAIOptions",
]
