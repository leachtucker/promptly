import asyncio
import time
from typing import Any, Optional

from .clients import BaseLLMClient, LLMResponse
from .templates import PromptTemplate
from .tracer import Tracer, TraceRecord, UsageData


class PromptRunner:
    """Orchestrates prompt execution"""

    def __init__(
        self,
        client: BaseLLMClient,
        tracer: Optional[Tracer] = None,
        backup_client: Optional[BaseLLMClient] = None,
    ) -> None:
        self.client = client
        self.tracer = tracer or Tracer()
        self.backup_client = backup_client

    async def run(
        self,
        model: str,
        prompt: PromptTemplate,
        variables: Optional[dict[str, Any]] = None,
        **llm_kwargs: Any,
    ) -> LLMResponse:
        """Run a prompt template"""

        # Attempt with primary client
        try:
            response = await self._run_with_client(
                self.client, model, prompt, variables, **llm_kwargs
            )
            return response
        except Exception as e:
            if self.backup_client is None:
                raise e

        # Attempt with backup client
        try:
            response = await self._run_with_client(
                self.backup_client, model, prompt, variables, **llm_kwargs
            )
            return response
        except Exception as e:
            raise e

    async def run_batch(
        self,
        model: str,
        prompt: PromptTemplate,
        batch_variables: list[dict[str, Any]],
        concurrency: int = 5,
        **llm_kwargs: Any,
    ) -> list[LLMResponse]:
        """Run a batch of prompts in parallel"""
        semaphore = asyncio.Semaphore(concurrency)

        async def run_single(variables: dict[str, Any]) -> LLMResponse:
            async with semaphore:
                return await self.run(model, prompt, variables, **llm_kwargs)

        tasks = [run_single(variables) for variables in batch_variables]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out exceptions and return only LLMResponse objects
        return [result for result in results if isinstance(result, LLMResponse)]

    async def _run_with_client(
        self,
        client: BaseLLMClient,
        model: str,
        prompt: PromptTemplate,
        variables: Optional[dict[str, Any]] = None,
        **llm_kwargs: Any,
    ) -> LLMResponse:
        """Run a prompt template with a given client"""
        if variables is None:
            variables = {}
        variables = variables or {}

        # Render the prompt
        try:
            rendered_prompt = prompt.render(**variables)
        except Exception as e:
            # Log error and re-raise
            error_record = TraceRecord(
                prompt_name=prompt.name,
                prompt_template=prompt.template,
                rendered_prompt="",
                response="",
                model=model,
                duration_ms=0,
                error=str(e),
            )
            self.tracer.log(error_record)
            raise e

        # Call LLM
        start_time = time.time()
        generation_error = None
        response = None
        try:
            response = await client.generate(rendered_prompt, model=model, **llm_kwargs)
        except Exception as e:
            generation_error = e

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # If no response and no generation error was set, set generation error to a generic error
        if response is None and generation_error is None:
            generation_error = Exception("No response from LLM")

        # Log trace
        if self.tracer.is_tracing_enabled:
            trace_record = TraceRecord(
                prompt_name=prompt.name,
                prompt_template=prompt.template,
                rendered_prompt=rendered_prompt,
                response=response.content if response is not None else "",
                model=response.model if response is not None else "",
                duration_ms=duration_ms,
                usage=response.usage if response is not None else UsageData(),
                metadata=response.metadata if response is not None else {},
                error=str(generation_error) if generation_error is not None else None,
            )
            trace_record = self.tracer.log(trace_record)

            if response is not None:
                response.metadata["trace_id"] = trace_record.id

        if generation_error is not None:
            raise generation_error
        assert response is not None

        return response
