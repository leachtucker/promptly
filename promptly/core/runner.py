import time
import asyncio
from typing import Dict, Any, Optional, List
from .templates import PromptTemplate
from .clients import BaseLLMClient, LLMResponse
from .tracer import Tracer, TraceRecord


class PromptRunner:
    """Orchestrates prompt execution with tracing"""

    def __init__(self, client: BaseLLMClient, tracer: Optional[Tracer] = None) -> None:
        self.client = client
        self.tracer = tracer or Tracer()

    async def run(
        self,
        prompt: PromptTemplate,
        variables: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **llm_kwargs: Any,
    ) -> LLMResponse:
        """Run a prompt template with tracing"""
        variables = variables or {}
        start_time = time.time()

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
                model=model or "unknown",
                duration_ms=0,
                error=str(e),
            )
            self.tracer.log(error_record)
            raise

        # Call LLM
        try:
            response = await self.client.generate(
                rendered_prompt, model=model, **llm_kwargs
            )
            error = None
        except Exception as e:
            response = LLMResponse(content="", model=model or "unknown", usage={})
            error = str(e)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log trace
        trace_record = TraceRecord(
            prompt_name=prompt.name,
            prompt_template=prompt.template,
            rendered_prompt=rendered_prompt,
            response=response.content,
            model=response.model,
            duration_ms=duration_ms,
            usage=response.usage,
            metadata=response.metadata,
            error=error,
        )
        self.tracer.log(trace_record)

        if error:
            raise Exception(error)

        return response

    async def run_batch(
        self,
        prompt: PromptTemplate,
        batch_variables: List[Dict[str, Any]],
        model: Optional[str] = None,
        concurrency: int = 5,
        **llm_kwargs: Any,
    ) -> List[LLMResponse]:
        """Run prompt template with multiple variable sets"""
        semaphore = asyncio.Semaphore(concurrency)

        async def run_single(variables: Dict[str, Any]) -> LLMResponse:
            async with semaphore:
                return await self.run(prompt, variables, model, **llm_kwargs)

        tasks = [run_single(variables) for variables in batch_variables]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out exceptions and return only LLMResponse objects
        return [result for result in results if isinstance(result, LLMResponse)]
