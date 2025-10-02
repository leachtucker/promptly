"""
Command-line interface for promptly
"""

from rich.console import Console
from rich.table import Table


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

    def _list_traces_table(tracer: Tracer) -> None:
            trace_records = tracer.list_records()
            
            if not trace_records:
                click.echo("No trace records found")
                return
            
            console = Console()
            table = Table(title="Trace Records")
            
            table.add_column("ID", style="cyan")
            table.add_column("Prompt", style="green", max_width=40)
            table.add_column("Model", style="yellow")
            table.add_column("Duration (ms)", justify="right")
            table.add_column("Error", style="red")
            
            for record in trace_records:
                table.add_row(
                    str(record.id or "N/A"),
                    record.prompt_name[:40] + "..." if len(record.prompt_name) > 40 else record.prompt_name,
                    record.model,
                    f"{record.duration_ms:.2f}",
                    str(record.error)[:30] if record.error else "None"
                )
            
            console.print(table)

    try:
        tracer = Tracer()

        if trace_id:
            click.echo(f"Viewing trace: {trace_id}")
            trace_record = tracer.get_record(trace_id)
            click.echo(f"Trace record: {json.dumps(asdict(trace_record), default=str, indent=2)}")
        else:
            click.echo("Listing recent traces...")
            # trace_records = tracer.list_records()
            # trace_records = [asdict(record) for record in trace_records]
            # click.echo_via_pager(f"Trace records: {json.dumps(trace_records, default=str, indent=2)}")
            _list_traces_table(tracer)

    except Exception as e:
        click.echo(f"Error: {e}")


if __name__ == "__main__":
    main()
