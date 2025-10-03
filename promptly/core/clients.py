
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import openai
import anthropic

from .utils.env import get_env_var
from .tracer import UsageData

ENV_OPENAI_API_KEY = get_env_var("OPENAI_API_KEY")
ENV_ANTHROPIC_API_KEY = get_env_var("ANTHROPIC_API_KEY")


@dataclass
class LLMResponse:
    """Standardized response from any LLM"""

    content: str
    model: str
    usage: UsageData = field(default_factory=UsageData)  # tokens used
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # metadata has default_factory, so no initialization needed
        pass


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response from LLM"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI client implementation"""

    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.AsyncOpenAI(api_key=api_key or ENV_OPENAI_API_KEY)
        self.default_model = "gpt-3.5-turbo"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using OpenAI"""
        model = model or self.default_model

        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            **kwargs,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            usage=UsageData(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id,
            },
        )

    async def get_available_models(self) -> List[str]:
        models = await self.client.models.list()
        return [model.id for model in models.data]


class AnthropicClient(BaseLLMClient):
    """Anthropic client implementation"""

    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.AsyncAnthropic(api_key=api_key or ENV_ANTHROPIC_API_KEY)
        self.default_model = "claude-3-sonnet-20240229"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Anthropic"""
        model = model or self.default_model

        response = await self.client.messages.create(
            model=model,
            system=prompt,
            **kwargs,
        )

        return LLMResponse(
            content=response.content[0].text,
            model=model,
            usage=UsageData(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens
                + response.usage.output_tokens,
            ),
            metadata={"stop_reason": response.stop_reason, "response_id": response.id},
        )

    async def get_available_models(self) -> List[str]:
        models = await self.client.models.list(limit=1000)
        return [model.id for model in models.data]


class LocalLLMClient(BaseLLMClient):
    """Client for local models (Ollama, etc.)"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.default_model = "llama2"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using local model"""
        # Placeholder - would implement actual Ollama API calls
        return LLMResponse(
            content=f"Local response for: {prompt[:50]}...",
            model=model or self.default_model,
            usage=UsageData(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    def get_available_models(self) -> List[str]:
        return ["llama2", "mistral", "codellama"]
