"""
Integration tests for promptly package
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from promptly import PromptRunner, PromptTemplate, PromptMetadata
from promptly.core.clients import OpenAIClient, LLMResponse
from promptly.core.tracer import Tracer, UsageData


class TestIntegration:
    """Integration tests for end-to-end workflows"""

    @pytest.mark.asyncio
    async def test_full_workflow_with_tracing(self, temp_db):
        """Test complete workflow from template to response with tracing"""
        # Create tracer with temp database
        tracer = Tracer(db_path=temp_db)

        # Create mock client
        mock_client = AsyncMock()
        mock_response = LLMResponse(
            content="Hello Alice, how are you today? I'm doing great, thank you for asking!",
            model="gpt-3.5-turbo",
            usage=UsageData(total_tokens=20),
        )
        mock_client.generate.return_value = mock_response

        # Create prompt template
        template = PromptTemplate(
            template="Hello {{ name }}, how are you today?",
            name="greeting",
            metadata=PromptMetadata(
                name="greeting",
                description="A friendly greeting",
                tags=["greeting", "conversation"],
            ),
        )

        # Create runner with tracing
        runner = PromptRunner(mock_client, tracer)

        # Execute the workflow
        response = await runner.run(
            prompt=template,
            variables={"name": "Alice"},
            model="gpt-3.5-turbo",
            temperature=0.7,
        )

        # Verify response
        assert (
            response.content
            == "Hello Alice, how are you today? I'm doing great, thank you for asking!"
        )
        assert response.model == "gpt-3.5-turbo"

        # Verify trace was recorded
        traces = tracer.list_records()
        assert len(traces) == 1

        trace = traces[0]
        assert trace.prompt_name == "greeting"
        assert trace.rendered_prompt == "Hello Alice, how are you today?"
        assert trace.model == "gpt-3.5-turbo"
        assert trace.duration_ms > 0
        assert trace.error is None

        # Verify client was called correctly
        mock_client.generate.assert_called_once()
        call_args = mock_client.generate.call_args
        assert call_args[0][0] == "Hello Alice, how are you today?"
        assert call_args[1]["model"] == "gpt-3.5-turbo"
        assert call_args[1]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_multiple_prompts_workflow(self, temp_db):
        """Test workflow with multiple prompts"""
        tracer = Tracer(db_path=temp_db)

        # Create mock client
        mock_client = AsyncMock()
        mock_responses = [
            LLMResponse(
                content="Hello Alice!",
                model="gpt-3.5-turbo",
                usage=UsageData(total_tokens=5),
            ),
            LLMResponse(
                content="The capital of France is Paris.",
                model="gpt-3.5-turbo",
                usage=UsageData(total_tokens=10),
            ),
        ]
        mock_client.generate.side_effect = mock_responses

        # Create templates
        greeting_template = PromptTemplate(
            template="Hello {{ name }}!", name="greeting"
        )
        question_template = PromptTemplate(
            template="What is the capital of {{ country }}?", name="capital_question"
        )

        # Create runner
        runner = PromptRunner(mock_client, tracer)

        # Execute multiple prompts
        greeting_response = await runner.run(
            prompt=greeting_template, variables={"name": "Alice"}, model="gpt-3.5-turbo"
        )

        question_response = await runner.run(
            prompt=question_template,
            variables={"country": "France"},
            model="gpt-3.5-turbo",
        )

        # Verify responses
        assert greeting_response.content == "Hello Alice!"
        assert question_response.content == "The capital of France is Paris."

        # Verify both traces were recorded
        traces = tracer.list_records()
        assert len(traces) == 2

        # Verify trace order (most recent first)
        assert traces[0].prompt_name == "capital_question"
        assert traces[1].prompt_name == "greeting"

        # Verify tracer stats
        stats = tracer.get_stats()
        assert stats["total_calls"] == 2
        assert stats["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, temp_db):
        """Test workflow with error handling"""
        tracer = Tracer(db_path=temp_db)

        # Create mock client that fails
        mock_client = AsyncMock()
        mock_client.generate.side_effect = Exception("API rate limit exceeded")

        # Create template
        template = PromptTemplate(template="Hello {{ name }}!", name="greeting")

        # Create runner
        runner = PromptRunner(mock_client, tracer)

        # Execute and expect error
        with pytest.raises(Exception, match="API rate limit exceeded"):
            await runner.run(
                prompt=template, variables={"name": "Alice"}, model="gpt-3.5-turbo"
            )

        # Verify error trace was recorded
        traces = tracer.list_records()
        assert len(traces) == 1

        trace = traces[0]
        assert trace.prompt_name == "greeting"
        assert trace.error == "API rate limit exceeded"
        assert trace.response == ""
        assert trace.duration_ms > 0

    @pytest.mark.asyncio
    async def test_simple_prompt_workflow(self, temp_db):
        """Test simple prompt workflow without templates"""
        tracer = Tracer(db_path=temp_db)

        # Create mock client
        mock_client = AsyncMock()
        mock_response = LLMResponse(
            content="The capital of France is Paris.",
            model="gpt-3.5-turbo",
            usage=UsageData(total_tokens=10),
        )
        mock_client.generate.return_value = mock_response

        # Create runner
        runner = PromptRunner(mock_client, tracer)

        # Execute simple prompt with template
        template = PromptTemplate(
            template="What is the capital of France?", name="simple_question"
        )
        response = await runner.run(prompt=template, model="gpt-3.5-turbo")

        # Verify response
        assert response.content == "The capital of France is Paris."

        # Verify trace was recorded
        traces = tracer.list_records()
        assert len(traces) == 1

        trace = traces[0]
        assert trace.rendered_prompt == "What is the capital of France?"
        assert trace.response == "The capital of France is Paris."
        assert trace.model == "gpt-3.5-turbo"

    def test_package_imports(self):
        """Test that all package imports work correctly"""
        # Test main package imports
        from promptly import (
            PromptRunner,
            PromptTemplate,
            Tracer,
            LLMResponse,
        )
        
        # Test core.clients imports
        from promptly.core.clients import (
            OpenAIClient,
            AnthropicClient,
        )

        # Verify classes can be instantiated (with mocks where needed)
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()

        runner = PromptRunner(mock_client)
        assert runner is not None

        template = PromptTemplate(template="Hello {{ name }}!")
        assert template is not None

        tracer = Tracer(db_path=":memory:")
        assert tracer is not None
