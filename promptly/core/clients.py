
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, TypeVar
import openai
import anthropic
from pydantic import BaseModel, Field
from google import genai

from .utils.env import get_env_var
from .tracer import UsageData

T = TypeVar('T', bound=BaseModel)

ENV_OPENAI_API_KEY = get_env_var("OPENAI_API_KEY")
ENV_ANTHROPIC_API_KEY = get_env_var("ANTHROPIC_API_KEY")
ENV_GOOGLE_API_KEY = get_env_var("GOOGLE_API_KEY")


class LLMResponse(BaseModel):
    """Standardized response from any LLM"""
    content: Optional[str] = None
    model: str
    usage: UsageData = Field(default_factory=UsageData)  # tokens used
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> T:
        """Generate structured response from LLM"""
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
            model=response.model,
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

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """Generate structured response using OpenAI"""
        model = model or self.default_model

        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            response_format=response_model,
            **kwargs,
        )

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Failed to parse structured response")
        return parsed

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
            model=response.model,
            usage=UsageData(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens
                + response.usage.output_tokens,
            ),
            metadata={"stop_reason": response.stop_reason, "response_id": response.id},
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """Generate structured response using Anthropic"""
        model = model or self.default_model

        # Anthropic doesn't have native structured output, so we use JSON mode
        json_prompt = f"{prompt}\n\nPlease respond with valid JSON matching this schema: {response_model.model_json_schema()}"
        
        response = await self.client.messages.create(
            model=model,
            system=json_prompt,
            **kwargs,
        )

        # Parse the JSON response
        import json
        try:
            json_data = json.loads(response.content[0].text)
            return response_model(**json_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse structured response: {e}")

    async def get_available_models(self) -> List[str]:
        models = await self.client.models.list(limit=1000)
        return [model.id for model in models.data]


class GoogleAIClient(BaseLLMClient):
    """Google AI Studio (Gemini) client implementation using google-genai SDK"""

    def __init__(self, api_key: Optional[str] = None):        
        api_key = api_key or ENV_GOOGLE_API_KEY
        if not api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        self.client = genai.Client(api_key=api_key)
        self.default_model = "gemini-1.5-flash"

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Google AI Studio"""
        model_name = model or self.default_model
        
        # Generate content
        response = await self.client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(**kwargs) if kwargs else None,
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
        content = response.text if hasattr(response, 'text') else str(response)
        
        return LLMResponse(
            content=content,
            model=model_name,
            usage=UsageData(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            metadata={
                "finish_reason": response.candidates[0].finish_reason if response.candidates else None,
                "response_id": response.response_id,
            },
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """Generate structured response using Google AI Studio"""
        model_name = model or self.default_model
        
        # Get the JSON schema
        schema = response_model.model_json_schema()
        
        # Create generation config for structured output
        config = genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            **kwargs
        )
        
        # Generate content
        response = await self.client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        
        # Parse the JSON response
        import json
        try:
            content = response.text if hasattr(response, 'text') else str(response)
            json_data = json.loads(content or "")
            return response_model(**json_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse structured response: {e}")

    def get_available_models(self) -> List[str]:
        """Get list of available Gemini models"""
        models = self.client.models.list(config={'query_base': True})
        return [model.name for model in models.page if model.name]


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

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> T:
        """Generate structured response using local model"""
        # TODO: Implement this
        raise NotImplementedError("LocalLLMClient does not support structured output")

    def get_available_models(self) -> List[str]:
        return ["llama2", "mistral", "codellama"]
