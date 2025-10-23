"""
LLM clients for Promptly
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar

import anthropic
import openai
from google import genai
from pydantic import BaseModel, Field

from .client_types import AnthropicOptions, GoogleAIOptions, OpenAIOptions
from .tracer import UsageData
from .utils.env import get_env_var, load_env_for_promptly

T = TypeVar("T", bound=BaseModel)


load_env_for_promptly()
ENV_OPENAI_API_KEY = get_env_var("OPENAI_API_KEY")
ENV_ANTHROPIC_API_KEY = get_env_var("ANTHROPIC_API_KEY")
ENV_GEMINI_API_KEY = get_env_var("GEMINI_API_KEY")


class LLMResponse(BaseModel):
    """Standardized response from any LLM"""

    content: Optional[str] = None
    model: str
    usage: UsageData = Field(default_factory=UsageData)  # tokens used
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients with provider-agnostic interface"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[Any] = None,
    ) -> LLMResponse:
        """
        Generate response from LLM.

        Args:
            prompt: The input prompt/instruction
            model: Model identifier (provider-specific)
            options: Provider-specific options dict

        Returns:
            LLMResponse: Standardized response object
        """
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        model: Optional[str] = None,
        options: Optional[Any] = None,
    ) -> T:
        """
        Generate structured response from LLM.

        Args:
            prompt: The input prompt/instruction
            response_model: Pydantic model for structured output
            model: Model identifier (provider-specific)
            options: Provider-specific options dict

        Returns:
            T: Instance of response_model with parsed data
        """
        pass

    @abstractmethod
    async def get_available_models(self) -> list[str]:
        """Get list of available models"""
        pass


class OpenAIClient(BaseLLMClient):
    """
    OpenAI client implementation.

    Supports all models from OpenAI's chat completions API.
    For full API documentation, see:
    https://platform.openai.com/docs/api-reference/chat/create
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.AsyncOpenAI(api_key=api_key or ENV_OPENAI_API_KEY)
        self.default_model = "gpt-3.5-turbo"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[OpenAIOptions] = None,
    ) -> LLMResponse:
        """
        Generate response using OpenAI's Chat Completions API.

        Args:
            prompt: The input prompt/instruction
            model: Model name (default: gpt-3.5-turbo)
            options: OpenAI-specific options (TypedDict from openai.types)
                    Full type: openai.types.chat.CompletionCreateParamsBase
                    Supports all parameters from OpenAI's chat.completions.create()
                    including temperature, max_tokens, top_p, frequency_penalty,
                    presence_penalty, stop, response_format, tools, etc.

        Returns:
            LLMResponse: Standardized response object
        """
        model = model or self.default_model

        # Build parameters dict, excluding already set ones
        params = {
            "model": model,
            "messages": [{"role": "system", "content": prompt}],
            **(options or {}),
        }
        response = await self.client.chat.completions.create(**params)

        usage_data = UsageData()
        if response.usage:
            usage_data = UsageData(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=usage_data,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id,
            },
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        model: Optional[str] = None,
        options: Optional[OpenAIOptions] = None,
    ) -> T:
        """
        Generate structured response using OpenAI's Structured Outputs.

        Args:
            prompt: The input prompt/instruction
            response_model: Pydantic model for structured output
            model: Model name (default: gpt-3.5-turbo)
            options: OpenAI-specific options (TypedDict from openai.types)
                    Note: response_format is set automatically to response_model

        Returns:
            T: Instance of response_model with parsed data
        """
        model = model or self.default_model

        # Build parameters dict, excluding already set ones
        params = {
            "model": model,
            "messages": [{"role": "system", "content": prompt}],
            "response_format": response_model,
            **(options or {}),
        }
        response = await self.client.beta.chat.completions.parse(**params)

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Failed to parse structured response")
        return parsed

    async def get_available_models(self) -> list[str]:
        models = await self.client.models.list()
        return [model.id for model in models.data]


class AnthropicClient(BaseLLMClient):
    """
    Anthropic client implementation.

    Supports all Claude models from Anthropic's messages API.
    For full API documentation, see:
    https://docs.anthropic.com/en/api/messages
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.AsyncAnthropic(api_key=api_key or ENV_ANTHROPIC_API_KEY)
        self.default_model = "claude-3-sonnet-20240229"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[AnthropicOptions] = None,
    ) -> LLMResponse:
        """
        Generate response using Anthropic's Messages API.

        Args:
            prompt: The input prompt/instruction (used as system message)
            model: Model name (default: claude-3-sonnet-20240229)
            options: Anthropic-specific options (TypedDict from anthropic.types)
                    Full type: anthropic.types.MessageCreateParamsBase
                    Supports all parameters from Anthropic's messages.create()
                    including max_tokens (required), messages (required),
                    temperature, top_p, top_k, stop_sequences, tools, etc.

        Returns:
            LLMResponse: Standardized response object

        Note:
            The Anthropic API requires 'max_tokens' and 'messages' parameters.
            If not provided in options, sensible defaults will be used:
            - max_tokens: 1024
            - messages: [] (will be populated from prompt)
        """
        model = model or self.default_model
        opts: dict[str, Any] = dict(options or {})

        # Anthropic requires max_tokens and messages
        if "max_tokens" not in opts:
            opts["max_tokens"] = 1024
        if "messages" not in opts:
            opts["messages"] = []

        response = await self.client.messages.create(
            model=model,
            system=prompt,
            **opts,
        )

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage=UsageData(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            metadata={"stop_reason": response.stop_reason, "response_id": response.id},
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        model: Optional[str] = None,
        options: Optional[AnthropicOptions] = None,
    ) -> T:
        """
        Generate structured response using Anthropic.

        Args:
            prompt: The input prompt/instruction
            response_model: Pydantic model for structured output
            model: Model name (default: claude-3-sonnet-20240229)
            options: Anthropic-specific options (TypedDict from anthropic.types)

        Returns:
            T: Instance of response_model with parsed data

        Note:
            Anthropic doesn't have native structured output support yet,
            so this method uses JSON schema prompting and parsing.
        """
        model = model or self.default_model
        opts: dict[str, Any] = dict(options or {})

        # Anthropic requires max_tokens and messages
        if "max_tokens" not in opts:
            opts["max_tokens"] = 1024
        if "messages" not in opts:
            opts["messages"] = []

        # Anthropic doesn't have native structured output, so we use JSON mode
        json_prompt = f"{prompt}\n\nPlease respond with valid JSON matching this schema: {response_model.model_json_schema()}"

        response = await self.client.messages.create(
            model=model,
            system=json_prompt,
            **opts,
        )

        # Parse the JSON response
        import json

        try:
            json_data = json.loads(response.content[0].text)
            return response_model(**json_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse structured response: {e}") from e

    async def get_available_models(self) -> list[str]:
        models = await self.client.models.list(limit=1000)
        return [model.id for model in models.data]


class GoogleAIClient(BaseLLMClient):
    """
    Google AI Studio (Gemini) client implementation.

    Supports all Gemini models from Google's Generative AI API.
    For full API documentation, see:
    https://ai.google.dev/api/generate-content
    """

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or ENV_GEMINI_API_KEY
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter."
            )

        self.client = genai.Client(api_key=api_key)
        self.default_model = "gemini-1.5-flash"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[GoogleAIOptions] = None,
    ) -> LLMResponse:
        """
        Generate response using Google AI Studio's generate_content API.

        Args:
            prompt: The input prompt/instruction
            model: Model name (default: gemini-1.5-flash)
            options: Google AI-specific options (TypedDict from google.genai.types)
                    Full type: google.genai.types.GenerateContentConfigDict
                    Supports all parameters from Google's generate_content()
                    including temperature, max_output_tokens, top_p, top_k,
                    candidate_count, stop_sequences, safety_settings, tools, etc.

        Returns:
            LLMResponse: Standardized response object
        """
        model_name = model or self.default_model

        config = genai.types.GenerateContentConfig(**options) if options else None  # type: ignore[arg-type]

        # Generate content - use async API
        response = await self.client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        # Extract usage metadata if available
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            completion_tokens = response.usage_metadata.candidates_token_count or 0
            total_tokens = response.usage_metadata.total_token_count or 0
        else:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

        # Get the text content
        content = response.text if hasattr(response, "text") else str(response)

        return LLMResponse(
            content=content,
            model=model_name,
            usage=UsageData(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            metadata={
                "finish_reason": response.candidates[0].finish_reason
                if response.candidates
                else None,
                "response_id": response.response_id,
            },
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        model: Optional[str] = None,
        options: Optional[GoogleAIOptions] = None,
    ) -> T:
        """
        Generate structured response using Google AI Studio.

        Args:
            prompt: The input prompt/instruction
            response_model: Pydantic model for structured output
            model: Model name (default: gemini-1.5-flash)
            options: Google AI-specific options (TypedDict from google.genai.types)
                    Note: response_mime_type and response_schema are set automatically

        Returns:
            T: Instance of response_model with parsed data
        """
        model_name = model or self.default_model
        opts: dict[str, Any] = dict(options) if options else {}

        # Get the JSON schema
        schema = response_model.model_json_schema()

        # Override response format settings for structured output
        opts["response_mime_type"] = "application/json"
        opts["response_schema"] = schema

        config = genai.types.GenerateContentConfig(**opts)  # type: ignore[arg-type]

        # Generate content
        response = await self.client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        # Parse the JSON response
        import json

        try:
            content = response.text if hasattr(response, "text") else str(response)
            json_data = json.loads(content or "")
            return response_model(**json_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse structured response: {e}") from e

    async def get_available_models(self) -> list[str]:
        """Get list of available Gemini models"""
        models = self.client.models.list(config={"query_base": True})
        return [model.name for model in models.page if model.name]


class LocalLLMClient(BaseLLMClient):
    """
    Client for local models (Ollama, etc.).

    This is a placeholder implementation for local model support.
    Full implementation would integrate with Ollama API or similar.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.default_model = "llama2"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Generate response using local model.

        Args:
            prompt: The input prompt/instruction
            model: Model name (default: llama2)
            options: Local model-specific options (dict)
                    Would support Ollama API parameters when implemented

        Returns:
            LLMResponse: Standardized response object

        Note:
            This is a placeholder implementation. Full Ollama integration
            would be implemented here.
        """
        # Placeholder - would implement actual Ollama API calls
        return LLMResponse(
            content=f"Local response for: {prompt[:50]}...",
            model=model or self.default_model,
            usage=UsageData(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        model: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> T:
        """
        Generate structured response using local model.

        Args:
            prompt: The input prompt/instruction
            response_model: Pydantic model for structured output
            model: Model name (default: llama2)
            options: Local model-specific options (dict)

        Returns:
            T: Instance of response_model with parsed data

        Note:
            This is not yet implemented. Would use Ollama's JSON mode
            or similar when implemented.
        """
        # TODO: Implement this
        raise NotImplementedError("LocalLLMClient does not support structured output")

    async def get_available_models(self) -> list[str]:
        return ["llama2", "mistral", "codellama"]
