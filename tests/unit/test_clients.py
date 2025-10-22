"""
Tests for LLM client implementations
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptly.core.clients import (
    AnthropicClient,
    BaseLLMClient,
    GoogleAIClient,
    LLMResponse,
    OpenAIClient,
)
from promptly.core.tracer import UsageData


class TestBaseLLMClient:
    """Test BaseLLMClient abstract class"""

    def test_base_client_is_abstract(self):
        """Test that BaseLLMClient cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseLLMClient()  # type: ignore


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
            prompt="Test prompt",
            model="gpt-3.5-turbo",
            options={"temperature": 0.7, "max_tokens": 100},
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
            options={"temperature": 0.7, "max_tokens": 100},
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


class TestGoogleAIClient:
    """Test GoogleAIClient implementation"""

    def test_google_ai_client_initialization(self):
        """Test Google AI client initialization"""
        # Create mock modules
        mock_google = MagicMock()
        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_genai.types = mock_types
        mock_genai.Client = MagicMock()

        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            client = GoogleAIClient(api_key="test-key")
            assert client.default_model == "gemini-1.5-flash"

    def test_google_ai_client_initialization_without_key(self):
        """Test Google AI client initialization without API key raises error"""
        # Mock the environment variable to ensure it's not set
        with patch("promptly.core.clients.ENV_GEMINI_API_KEY", None):
            with pytest.raises(ValueError, match="Google API key is required"):
                GoogleAIClient()

    @pytest.mark.asyncio
    async def test_google_ai_client_generate(self):
        """Test Google AI client generate method"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].safety_ratings = []

        # Mock usage metadata
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 5
        mock_usage.total_token_count = 15
        mock_response.usage_metadata = mock_usage

        # Create mock modules and setup before import
        mock_google = MagicMock()
        mock_genai_module = MagicMock()
        mock_types = MagicMock()
        mock_genai_module.types = mock_types

        # Create properly structured async mock
        mock_client_instance = MagicMock()
        mock_client_instance.aio = MagicMock()
        mock_client_instance.aio.models = MagicMock()
        mock_client_instance.aio.models.generate_content = AsyncMock(return_value=mock_response)

        mock_genai_module.Client = MagicMock(return_value=mock_client_instance)

        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai_module}):
            # Create client
            client = GoogleAIClient(api_key="test-key")

            # Manually set the client.client to our mock since the import creates its own instance
            client.client = mock_client_instance

            # Test generate
            response = await client.generate(
                prompt="Test prompt",
                model="gemini-1.5-flash",
            )

            # Verify response
            assert isinstance(response, LLMResponse)
            assert response.content == "Test response"
            assert response.model == "gemini-1.5-flash"
            assert response.usage.prompt_tokens == 10
            assert response.usage.completion_tokens == 5
            assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_google_ai_client_generate_with_defaults(self):
        """Test Google AI client generate with default parameters"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.candidates[0].safety_ratings = []

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 5
        mock_usage.candidates_token_count = 3
        mock_usage.total_token_count = 8
        mock_response.usage_metadata = mock_usage

        # Create mock modules
        mock_google = MagicMock()
        mock_genai_module = MagicMock()
        mock_types = MagicMock()
        mock_genai_module.types = mock_types

        # Create properly structured async mock
        mock_client_instance = MagicMock()
        mock_client_instance.aio = MagicMock()
        mock_client_instance.aio.models = MagicMock()
        mock_client_instance.aio.models.generate_content = AsyncMock(return_value=mock_response)

        mock_genai_module.Client = MagicMock(return_value=mock_client_instance)

        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai_module}):
            # Create client
            client = GoogleAIClient(api_key="test-key")

            # Manually set the client.client to our mock
            client.client = mock_client_instance

            # Test generate with minimal parameters
            response = await client.generate(prompt="Test prompt")

            # Verify response
            assert response.content == "Test response"
            assert response.model == "gemini-1.5-flash"

    def test_google_ai_client_get_available_models(self):
        """Test Google AI client get available models"""
        # Create mock model objects
        mock_model_1 = MagicMock()
        mock_model_1.name = "gemini-1.5-flash"
        mock_model_2 = MagicMock()
        mock_model_2.name = "gemini-1.5-pro"
        mock_model_3 = MagicMock()
        mock_model_3.name = "gemini-2.0-flash-exp"

        # Create mock response with page attribute
        mock_models_response = MagicMock()
        mock_models_response.page = [mock_model_1, mock_model_2, mock_model_3]

        # Create mock client instance
        mock_client_instance = MagicMock()
        mock_client_instance.models.list = MagicMock(return_value=mock_models_response)

        with patch("google.genai.Client", return_value=mock_client_instance):
            client = GoogleAIClient(api_key="test-key")
            models = client.get_available_models()

            assert isinstance(models, list)
            assert len(models) == 3
            assert "gemini-1.5-flash" in models
            assert "gemini-1.5-pro" in models
            assert "gemini-2.0-flash-exp" in models

            # Verify the list method was called with correct config
            mock_client_instance.models.list.assert_called_once_with(config={"query_base": True})

    @pytest.mark.asyncio
    async def test_google_ai_client_generate_error(self):
        """Test Google AI client error handling"""
        # Create mock modules
        mock_google = MagicMock()
        mock_genai_module = MagicMock()
        mock_types = MagicMock()
        mock_genai_module.types = mock_types

        # Create properly structured async mock that raises an error
        mock_client_instance = MagicMock()
        mock_client_instance.aio = MagicMock()
        mock_client_instance.aio.models = MagicMock()
        mock_client_instance.aio.models.generate_content = AsyncMock(
            side_effect=Exception("API Error")
        )

        mock_genai_module.Client = MagicMock(return_value=mock_client_instance)

        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai_module}):
            # Create client
            client = GoogleAIClient(api_key="test-key")

            # Manually set the client.client to our mock
            client.client = mock_client_instance

            # Test error handling - match regex pattern properly
            with pytest.raises(Exception, match="API Error"):
                await client.generate(prompt="Test prompt")
