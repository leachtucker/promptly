"""
Tests for LLMResponse Pydantic model
"""

from promptly.core.clients import LLMResponse
from promptly.core.tracer import UsageData


class TestLLMResponse:
    """Test LLMResponse Pydantic model functionality"""

    def test_llm_response_creation(self):
        """Test basic LLMResponse creation"""
        response = LLMResponse(
            content="Test response", model="gpt-3.5-turbo", usage=UsageData(total_tokens=10)
        )

        assert response.content == "Test response"
        assert response.model == "gpt-3.5-turbo"
        assert response.usage.total_tokens == 10
        assert response.usage.prompt_tokens == 0
        assert response.usage.completion_tokens == 0
        assert response.metadata == {}

    def test_llm_response_with_metadata(self):
        """Test LLMResponse with custom metadata"""
        metadata = {"trace_id": "123", "custom": "value"}
        response = LLMResponse(
            content="Test response",
            model="gpt-3.5-turbo",
            usage=UsageData(total_tokens=10),
            metadata=metadata,
        )

        assert response.metadata == metadata

    def test_llm_response_metadata_default(self):
        """Test that metadata defaults to empty dict"""
        response = LLMResponse(
            content="Test response", model="gpt-3.5-turbo", usage=UsageData(total_tokens=10)
        )

        assert response.metadata == {}
        assert isinstance(response.metadata, dict)

    def test_llm_response_usage_types(self):
        """Test LLMResponse with different usage formats"""
        # Test with detailed usage
        response1 = LLMResponse(
            content="Test",
            model="gpt-3.5-turbo",
            usage=UsageData(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        assert response1.usage.total_tokens == 8
        assert response1.usage.prompt_tokens == 5
        assert response1.usage.completion_tokens == 3

        # Test with minimal usage
        response2 = LLMResponse(
            content="Test", model="gpt-3.5-turbo", usage=UsageData(total_tokens=1)
        )
        assert response2.usage.total_tokens == 1
        assert response2.usage.prompt_tokens == 0
        assert response2.usage.completion_tokens == 0
