"""
Tests for LLM client implementations
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from promptly.core.clients import (
    BaseLLMClient,
    OpenAIClient,
    AnthropicClient,
    LLMResponse,
)
from promptly.core.tracer import UsageData


class TestBaseLLMClient:
    """Test BaseLLMClient abstract class"""

    def test_base_client_is_abstract(self):
        """Test that BaseLLMClient cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseLLMClient()


class TestOpenAIClient:
    """Test OpenAIClient implementation"""

    def test_openai_client_initialization(self):
        """Test OpenAI client initialization"""
        client = OpenAIClient(api_key="test-key")

        assert client.default_model == "gpt-3.5-turbo"
        assert client.client is not None

    def test_openai_client_initialization_without_key(self):
        """Test OpenAI client initialization without API key"""
        client = OpenAIClient()

        assert client.default_model == "gpt-3.5-turbo"
        assert client.client is not None

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_openai_client_generate(self, mock_openai_class):
        """Test OpenAI client generate method"""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-3.5-turbo"

        mock_client.chat.completions.create.return_value = mock_response

        # Create client and test
        client = OpenAIClient(api_key="test-key")
        response = await client.generate(
            prompt="Test prompt", model="gpt-3.5-turbo", temperature=0.7, max_tokens=100
        )

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.model == "gpt-3.5-turbo"
        assert response.usage == UsageData(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

        # Verify OpenAI client was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-3.5-turbo"
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["max_tokens"] == 100
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["content"] == "Test prompt"

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_openai_client_generate_with_defaults(self, mock_openai_class):
        """Test OpenAI client generate with default parameters"""
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_response.model = "gpt-3.5-turbo"

        mock_client.chat.completions.create.return_value = mock_response

        # Create client and test with minimal parameters
        client = OpenAIClient(api_key="test-key")
        response = await client.generate(prompt="Test prompt")

        # Verify response
        assert response.content == "Test response"
        assert response.model == "gpt-3.5-turbo"

        # Verify default model was used
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_openai_client_get_available_models(self):
        """Test OpenAI client get available models"""
        client = OpenAIClient()
        models = await client.get_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-3.5-turbo" in models
        assert "gpt-4" in models

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_openai_client_generate_error(self, mock_openai_class):
        """Test OpenAI client error handling"""
        # Mock the OpenAI client to raise an error
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        # Create client and test error handling
        client = OpenAIClient(api_key="test-key")

        with pytest.raises(Exception, match="API Error"):
            await client.generate(prompt="Test prompt")


class TestAnthropicClient:
    """Test AnthropicClient implementation"""

    def test_anthropic_client_initialization(self):
        """Test Anthropic client initialization"""
        client = AnthropicClient(api_key="test-key")

        assert client.default_model == "claude-3-sonnet-20240229"
        assert client.client is not None

    def test_anthropic_client_initialization_without_key(self):
        """Test Anthropic client initialization without API key"""
        client = AnthropicClient()

        assert client.default_model == "claude-3-sonnet-20240229"
        assert client.client is not None

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic")
    async def test_anthropic_client_generate(self, mock_anthropic_class):
        """Test Anthropic client generate method"""
        # Mock the Anthropic client
        mock_client = AsyncMock()
        mock_anthropic_class.return_value = mock_client

        # Mock the response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Test response"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.model = "claude-3-sonnet-20240229"

        mock_client.messages.create.return_value = mock_response

        # Create client and test
        client = AnthropicClient(api_key="test-key")
        response = await client.generate(
            prompt="Test prompt",
            model="claude-3-sonnet-20240229",
            temperature=0.7,
            max_tokens=100,
        )

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.model == "claude-3-sonnet-20240229"
        assert response.usage == UsageData(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

        # Verify Anthropic client was called correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-sonnet-20240229"
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["max_tokens"] == 100
        assert call_args[1]["system"] == "Test prompt"

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic")
    async def test_anthropic_client_generate_with_defaults(self, mock_anthropic_class):
        """Test Anthropic client generate with default parameters"""
        # Mock the Anthropic client
        mock_client = AsyncMock()
        mock_anthropic_class.return_value = mock_client

        # Mock the response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Test response"
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 3
        mock_response.model = "claude-3-sonnet-20240229"

        mock_client.messages.create.return_value = mock_response

        # Create client and test with minimal parameters
        client = AnthropicClient(api_key="test-key")
        response = await client.generate(prompt="Test prompt")

        # Verify response
        assert response.content == "Test response"
        assert response.model == "claude-3-sonnet-20240229"

        # Verify default model was used
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-sonnet-20240229"

    @pytest.mark.asyncio
    async def test_anthropic_client_get_available_models(self):
        """Test Anthropic client get available models"""
        client = AnthropicClient()
        models = await client.get_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "claude-opus-4-1-20250805" in models
        assert "claude-sonnet-4-5-20250929" in models

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic")
    async def test_anthropic_client_generate_error(self, mock_anthropic_class):
        """Test Anthropic client error handling"""
        # Mock the Anthropic client to raise an error
        mock_client = AsyncMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")

        # Create client and test error handling
        client = AnthropicClient(api_key="test-key")

        with pytest.raises(Exception, match="API Error"):
            await client.generate(prompt="Test prompt")
