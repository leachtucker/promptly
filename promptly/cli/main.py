"""
Command-line interface for promptly
"""

import click
import asyncio
import json
from typing import Optional
from dataclasses import asdict
from ..core.runner import PromptRunner
from ..core.tracer import Tracer
from ..core.clients import OpenAIClient, AnthropicClient


@click.group()
@click.version_option()
def main() -> None:
    """promptly - A lightweight library for LLM prompt management and optimization"""
    pass


@main.command()
@click.option("--template", "-t", help="Path to prompt template file")
@click.option("--model", "-m", default="gpt-3.5-turbo", help="Model to use")
@click.option(
    "--provider",
    "-p",
    default="openai",
    type=click.Choice(["openai", "anthropic"]),
    help="LLM provider",
)
@click.option("--api-key", help="API key for the provider")
@click.option("--trace", is_flag=True, help="Enable tracing", default=True)
@click.argument("prompt", required=False)
def run(
    template: Optional[str],
    model: str,
    provider: str,
    api_key: Optional[str],
    trace: bool,
    prompt: Optional[str],
) -> None:
    """Run a prompt with the specified model"""
    if not prompt and not template:
        click.echo("Error: Either --template or prompt argument is required")
        raise click.Abort()

    if prompt:
        # Simple prompt execution
        asyncio.run(_run_simple_prompt(prompt, model, provider, api_key, trace))
    else:
        # Template-based execution
        click.echo(f"Template execution not yet implemented: {template}")


async def _run_simple_prompt(
    prompt: str, model: str, provider: str, api_key: Optional[str], trace: bool
) -> None:
    """Run a simple prompt"""
    try:
        # Initialize client
        from ..core.clients import BaseLLMClient
        client: BaseLLMClient
        if provider == "openai":
            client = OpenAIClient(api_key=api_key)
        elif provider == "anthropic":
            client = AnthropicClient(api_key=api_key)
        else:
            click.echo(f"Unsupported provider: {provider}")
            return

        # Initialize tracer if requested
        tracer = Tracer() if trace else None

        # Create runner
        runner = PromptRunner(client, tracer)

        # Run prompt
        from ..core.templates import PromptTemplate
        template = PromptTemplate(template=prompt, name="cli_prompt")
        response = await runner.run(model, template)

        click.echo(f"Response: {response.content}")

        if trace and tracer:
            click.echo(f"Trace ID: {response.metadata.get('trace_id', 'N/A')}")

    except Exception as e:
        click.echo(f"Error: {e}")


@main.command()
@click.option("--trace-id", help="Trace ID to view")
def trace(trace_id: Optional[str]) -> None:
    """View trace information"""

    try:
        if trace_id:
            click.echo(f"Viewing trace: {trace_id}")
            tracer = Tracer()
            trace_record = tracer.get_record(trace_id)
            click.echo(f"Trace record: {json.dumps(asdict(trace_record), default=str, indent=2)}")
        else:
            click.echo("Listing recent traces...")
            tracer = Tracer()
            trace_records = tracer.list_records()
            trace_records = [asdict(record) for record in trace_records]
            click.echo_via_pager(f"Trace records: {json.dumps(trace_records, default=str, indent=2)}")

    except Exception as e:
        click.echo(f"Error: {e}")


if __name__ == "__main__":
    main()
