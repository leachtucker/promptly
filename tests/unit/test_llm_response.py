"""
Tests for LLMResponse dataclass
"""

import pytest
from promptly.core.clients import LLMResponse


class TestLLMResponse:
    """Test LLMResponse dataclass functionality"""

    def test_llm_response_creation(self):
        """Test basic LLMResponse creation"""
        response = LLMResponse(
            content="Test response", model="gpt-3.5-turbo", usage={"total_tokens": 10}
        )

        assert response.content == "Test response"
        assert response.model == "gpt-3.5-turbo"
        assert response.usage == {"total_tokens": 10}
        assert response.metadata == {}

    def test_llm_response_with_metadata(self):
        """Test LLMResponse with custom metadata"""
        metadata = {"trace_id": "123", "custom": "value"}
        response = LLMResponse(
            content="Test response",
            model="gpt-3.5-turbo",
            usage={"total_tokens": 10},
            metadata=metadata,
        )

        assert response.metadata == metadata

    def test_llm_response_metadata_default(self):
        """Test that metadata defaults to empty dict"""
        response = LLMResponse(
            content="Test response", model="gpt-3.5-turbo", usage={"total_tokens": 10}
        )

        assert response.metadata == {}
        assert isinstance(response.metadata, dict)

    def test_llm_response_usage_types(self):
        """Test LLMResponse with different usage formats"""
        # Test with detailed usage
        response1 = LLMResponse(
            content="Test",
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )
        assert response1.usage["total_tokens"] == 8

        # Test with minimal usage
        response2 = LLMResponse(
            content="Test", model="gpt-3.5-turbo", usage={"total_tokens": 1}
        )
        assert response2.usage["total_tokens"] == 1
