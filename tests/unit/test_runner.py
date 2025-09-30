"""
Tests for PromptRunner
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from promptly.core.runner import PromptRunner
from promptly.core.templates import PromptTemplate
from promptly.core.clients import LLMResponse
from promptly.core.tracer import Tracer, TraceRecord


class TestPromptRunner:
    """Test PromptRunner functionality"""

    @pytest.mark.asyncio
    async def test_runner_initialization(self, mock_llm_client):
        """Test PromptRunner initialization"""
        runner = PromptRunner(mock_llm_client)

        assert runner.client == mock_llm_client
        assert isinstance(runner.tracer, Tracer)

    @pytest.mark.asyncio
    async def test_runner_with_custom_tracer(
        self, mock_llm_client, tracer_with_temp_db
    ):
        """Test PromptRunner with custom tracer"""
        runner = PromptRunner(mock_llm_client, tracer_with_temp_db)

        assert runner.client == mock_llm_client
        assert runner.tracer == tracer_with_temp_db

    @pytest.mark.asyncio
    async def test_runner_run_success(self, mock_llm_client, sample_prompt_template):
        """Test successful prompt execution"""
        runner = PromptRunner(mock_llm_client)

        # Mock the client response
        expected_response = LLMResponse(
            content="Hello Alice, how are you today?",
            model="gpt-3.5-turbo",
            usage={"total_tokens": 15},
        )
        mock_llm_client.generate.return_value = expected_response

        # Run the prompt
        response = await runner.run(
            prompt=sample_prompt_template,
            variables={"name": "Alice"},
            model="gpt-3.5-turbo",
        )

        # Verify response
        assert response == expected_response

        # Verify client was called correctly
        mock_llm_client.generate.assert_called_once()
        call_args = mock_llm_client.generate.call_args
        assert "Hello Alice, how are you today?" in call_args[0][0]
        assert call_args[1]["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_runner_run_with_tracing(
        self, mock_llm_client, sample_prompt_template, tracer_with_temp_db
    ):
        """Test prompt execution with tracing"""
        runner = PromptRunner(mock_llm_client, tracer_with_temp_db)

        # Mock the client response
        expected_response = LLMResponse(
            content="Hello Alice, how are you today?",
            model="gpt-3.5-turbo",
            usage={"total_tokens": 15},
        )
        mock_llm_client.generate.return_value = expected_response

        # Run the prompt
        response = await runner.run(
            prompt=sample_prompt_template,
            variables={"name": "Alice"},
            model="gpt-3.5-turbo",
        )

        # Verify response
        assert response == expected_response

        # Verify trace was recorded
        traces = tracer_with_temp_db.list_records()
        assert len(traces) == 1

        trace = traces[0]
        assert trace.prompt_name == sample_prompt_template.name
        assert "Hello Alice, how are you today?" in trace.rendered_prompt
        assert trace.response == "Hello Alice, how are you today?"
        assert trace.model == "gpt-3.5-turbo"
        assert trace.duration_ms > 0
        assert trace.error is None

    @pytest.mark.asyncio
    async def test_runner_run_template_render_error(
        self, mock_llm_client, tracer_with_temp_db
    ):
        """Test prompt execution with template rendering error"""
        runner = PromptRunner(mock_llm_client, tracer_with_temp_db)

        # Create template that will fail to render
        template = PromptTemplate(
            template="Hello {{ name }}, you are {{ age }} years old.",
            name="error_template",
        )

        # Run with missing required variable
        with pytest.raises(Exception):
            await runner.run(
                prompt=template,
                variables={"name": "Alice"},  # Missing 'age'
                model="gpt-3.5-turbo",
            )

        # Verify error trace was recorded
        traces = tracer_with_temp_db.list_records()
        assert len(traces) == 1

        trace = traces[0]
        assert trace.prompt_name == "error_template"
        assert trace.error is not None
        assert "age" in trace.error

    @pytest.mark.asyncio
    async def test_runner_run_llm_error(
        self, mock_llm_client, sample_prompt_template, tracer_with_temp_db
    ):
        """Test prompt execution with LLM error"""
        runner = PromptRunner(mock_llm_client, tracer_with_temp_db)

        # Mock LLM error
        mock_llm_client.generate.side_effect = Exception("LLM API error")

        # Run the prompt
        with pytest.raises(Exception, match="LLM API error"):
            await runner.run(
                prompt=sample_prompt_template,
                variables={"name": "Alice"},
                model="gpt-3.5-turbo",
            )

        # Verify error trace was recorded
        traces = tracer_with_temp_db.list_records()
        assert len(traces) == 1

        trace = traces[0]
        assert trace.prompt_name == sample_prompt_template.name
        assert trace.error == "LLM API error"
        assert trace.response == ""

    @pytest.mark.asyncio
    async def test_runner_run_simple(self, mock_llm_client):
        """Test simple prompt execution without template"""
        runner = PromptRunner(mock_llm_client)

        # Mock the client response
        expected_response = LLMResponse(
            content="This is a simple response",
            model="gpt-3.5-turbo",
            usage={"total_tokens": 10},
        )
        mock_llm_client.generate.return_value = expected_response


        template = PromptTemplate(template="What is the capital of France?", name="simple_question")
        # Run simple prompt
        response = await runner.run(
            prompt=template, model="gpt-3.5-turbo"
        )

        # Verify response
        assert response == expected_response

        # Verify client was called correctly
        mock_llm_client.generate.assert_called_once()
        call_args = mock_llm_client.generate.call_args
        assert call_args[0][0] == "What is the capital of France?"
        assert call_args[1]["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_runner_run_with_llm_kwargs(
        self, mock_llm_client, sample_prompt_template
    ):
        """Test prompt execution with additional LLM parameters"""
        runner = PromptRunner(mock_llm_client)

        # Mock the client response
        expected_response = LLMResponse(
            content="Response with custom params",
            model="gpt-3.5-turbo",
            usage={"total_tokens": 20},
        )
        mock_llm_client.generate.return_value = expected_response

        # Run with custom LLM parameters
        response = await runner.run(
            prompt=sample_prompt_template,
            variables={"name": "Alice"},
            model="gpt-3.5-turbo",
            temperature=0.9,
            max_tokens=100,
        )

        # Verify response
        assert response == expected_response

        # Verify client was called with custom parameters
        mock_llm_client.generate.assert_called_once()
        call_args = mock_llm_client.generate.call_args
        assert call_args[1]["temperature"] == 0.9
        assert call_args[1]["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_runner_run_without_variables(self, mock_llm_client):
        """Test prompt execution without variables"""
        runner = PromptRunner(mock_llm_client)

        # Create template that doesn't require variables
        template = PromptTemplate(
            template="What is the capital of France?", name="simple_question"
        )

        # Mock the client response
        expected_response = LLMResponse(
            content="The capital of France is Paris.",
            model="gpt-3.5-turbo",
            usage={"total_tokens": 15},
        )
        mock_llm_client.generate.return_value = expected_response

        # Run without variables
        response = await runner.run(prompt=template, model="gpt-3.5-turbo")

        # Verify response
        assert response == expected_response

        # Verify client was called with the template as-is
        mock_llm_client.generate.assert_called_once()
        call_args = mock_llm_client.generate.call_args
        assert call_args[0][0] == "What is the capital of France?"
