"""
Command-line interface for promptly
"""

import asyncio
import json
from typing import Any, Optional

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from ..core.clients import AnthropicClient, BaseLLMClient, GoogleAIClient, OpenAIClient
from ..core.optimizer import (
    LLMComprehensiveFitnessFunction,
    LLMGeneticOptimizer,
    OptimizationResult,
    ProgressCallback,
    PromptTestCase,
)
from ..core.runner import PromptRunner
from ..core.templates import PromptTemplate
from ..core.tracer import Tracer


class RichProgressCallback(ProgressCallback):
    """Rich-based progress callback for CLI optimization feedback"""

    def __init__(self, console: Console, progress: Progress, task_id: TaskID):
        self.console = console
        self.progress = progress
        self.task_id = task_id
        self.generation_count = 0
        self.best_candidates: list[dict[str, Any]] = []

    async def on_population_initialized(self, population_size: int) -> None:
        """Called when initial population is created"""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold blue]🧬 Population Ready![/bold blue]\n\n"
                f"[green]✨ Generated {population_size} diverse prompt variations[/green]\n"
                f"[dim]Each variation explores different approaches to your prompt[/dim]",
                border_style="blue",
                expand=False,
            )
        )
        self.console.print()

    async def on_generation_start(self, generation: int, total_generations: int) -> None:
        """Called at the start of each generation"""
        self.generation_count = generation + 1

        # Update progress description
        self.progress.update(
            self.task_id,
            description=f"🧬 Generation {self.generation_count}/{total_generations} - Evolving prompts...",
        )

        # Show generation start with excitement
        if self.generation_count == 1:
            self.console.print("[bold cyan]🚀 Starting evolution process...[/bold cyan]")
        else:
            self.console.print(
                f"[bold cyan]🔄 Generation {self.generation_count} - Let's evolve![/bold cyan]"
            )

    async def on_generation_complete(self, stats: dict) -> None:
        """Called when a generation completes with statistics"""
        generation = stats["generation"]
        best_fitness = stats["best_fitness"]
        best_prompt = stats["best_prompt"]
        reasoning = stats["reasoning"]
        errors = stats["errors"]

        # Store best candidate for later display
        self.best_candidates.append(
            {
                "generation": generation,
                "fitness": best_fitness,
                "prompt": best_prompt,
                "reasoning": reasoning,
                "errors": errors,
            }
        )

        # Update progress bar
        fitness_color = (
            "green" if best_fitness >= 0.8 else "yellow" if best_fitness >= 0.6 else "red"
        )
        self.progress.update(
            self.task_id,
            description=f"✨ Gen {generation} complete - Best: [{fitness_color}]{best_fitness:.3f}[/{fitness_color}]",
        )

        # Show exciting generation results
        self.console.print()

        # Generation champion display
        champion_emoji = "🏆" if best_fitness >= 0.8 else "🥇" if best_fitness >= 0.6 else "🔥"
        self.console.print(
            f"{champion_emoji} [bold green]Generation {generation} Champion![/bold green] [bold {fitness_color}]{best_fitness:.3f}[/bold {fitness_color}]"
        )

        # Show full best prompt
        self.console.print(
            Panel(
                f"[blue]{best_prompt}[/blue]",
                title="💡 Best Prompt",
                border_style="blue",
                expand=False,
            )
        )

        # Show improvement over previous generation
        if len(self.best_candidates) > 1:
            prev_fitness = self.best_candidates[-2]["fitness"]
            improvement = best_fitness - prev_fitness
            if improvement > 0:
                self.console.print(f"[bold green]📈 Improved by {improvement:.3f}![/bold green]")
            elif improvement < 0:
                self.console.print(f"[yellow]📉 Slight dip of {abs(improvement):.3f}[/yellow]")
            else:
                self.console.print("[dim]➡️ Maintaining performance[/dim]")

        # Show full LLM reasoning (if available)
        if reasoning and len(reasoning) > 0:
            self.console.print(
                Panel(
                    f"[white]{reasoning}[/white]",
                    title="🤖 LLM Insight",
                    border_style="cyan",
                    expand=False,
                )
            )

        self.console.print()

    async def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes"""
        # Complete the progress bar
        self.progress.update(self.task_id, completed=100, description="🎉 Evolution complete!")

        self.console.print()

        # Show evolution summary
        if len(self.best_candidates) > 1:
            self.console.print(
                Panel(
                    "[bold green]🎉 Evolution Complete![/bold green]\n\n"
                    "[blue]📊 Evolution Summary:[/blue]\n"
                    f"• Started with fitness: [yellow]{self.best_candidates[0]['fitness']:.3f}[/yellow]\n"
                    f"• Evolved to fitness: [green]{result.fitness_score:.3f}[/green]\n"
                    f"• Total improvement: [bold green]{result.fitness_score - self.best_candidates[0]['fitness']:.3f}[/bold green]\n"
                    f"• Generations evolved: [cyan]{result.generation + 1}[/cyan]",
                    border_style="green",
                    expand=False,
                )
            )
            self.console.print()

        # Show the final champion with celebration
        if result.fitness_score >= 0.8:
            celebration = "🌟 EXCELLENT! 🌟"
        elif result.fitness_score >= 0.6:
            celebration = "🎯 Good! 🎯"
        else:
            celebration = "🤏 Decent! 🤏"

        self.console.print(f"[bold green]{celebration}[/bold green]")
        self.console.print(
            f"[bold]Final Champion Fitness: [green]{result.fitness_score:.3f}[/green][/bold]"
        )


@click.group()
@click.version_option(version="0.1.0")
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
@click.option("--trace", is_flag=True, help="Enable tracing", default=False)
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

        if trace:
            click.echo(f"Trace ID: {response.metadata.get('trace_id', 'N/A')}")

    except Exception as e:
        click.echo(f"Error: {e}")


@main.command()
@click.option("--trace-id", help="Trace ID to view")
@click.option("--optimizer-only", is_flag=True, help="Show only optimizer traces")
@click.option("--optimization-id", help="Filter by specific optimization ID")
@click.option("--generation", type=int, help="Filter by generation number")
def trace(
    trace_id: Optional[str],
    optimizer_only: bool,
    optimization_id: Optional[str],
    generation: Optional[int],
) -> None:
    """View trace information"""

    def _list_traces_table(tracer: Tracer) -> None:
        if optimizer_only or optimization_id is not None or generation is not None:
            trace_records = tracer.list_optimizer_records(
                limit=20, optimization_id=optimization_id, generation=generation
            )
        else:
            trace_records = tracer.list_records(limit=20, optimizer_only=optimizer_only)

        if not trace_records:
            click.echo("No trace records found")
            return

        console = Console()
        table = Table(title="Trace Records", row_styles=["", "dim"])

        table.add_column("ID", style="cyan")
        table.add_column("Prompt Name", style="green")
        table.add_column("Prompt Template", style="purple")
        table.add_column("Response", style="blue", max_width=200)
        table.add_column("Model", style="yellow")
        table.add_column("Duration (ms)", justify="right")
        table.add_column("Error", style="red")
        table.add_column("Created At", style="purple")

        # Add optimizer-specific columns if showing optimizer traces
        if optimizer_only or optimization_id is not None or generation is not None:
            table.add_column("Optimization ID", style="red")
            table.add_column("Generation", style="red")

        for record in trace_records:
            response_text = record.response or ""
            row_data = [
                str(record.id or "N/A"),
                record.prompt_name,
                record.prompt_template[:400] + "..."
                if len(record.prompt_template) > 400
                else record.prompt_template,
                response_text[:400] + "..." if len(response_text) > 400 else response_text,
                record.model,
                f"{record.duration_ms:.2f}",
                str(record.error)[:30] if record.error else "",
                record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            ]

            # Add optimizer context if available
            if optimizer_only or optimization_id is not None or generation is not None:
                optimizer_context = record.metadata.get("optimizer_context", {})
                row_data.extend(
                    [
                        optimizer_context.get("optimization_id", "N/A"),
                        str(optimizer_context.get("generation", "N/A")),
                    ]
                )

            table.add_row(*row_data)

        console.print(table)

    def _view_trace(tracer: Tracer, trace_id: str) -> None:
        """View a trace record"""
        trace_record = tracer.get_record(trace_id)
        if not trace_record:
            click.echo(f"Trace {trace_id} not found")
            return

        console = Console()

        table = Table(title=f"Trace Record: {trace_record.id or 'N/A'}", show_lines=True)
        table.add_column("Field", style="cyan", width=25)
        table.add_column("Value", style="white", overflow="fold")

        table.add_row("ID", str(trace_record.id or "N/A"))
        table.add_row("Prompt Name", trace_record.prompt_name)
        table.add_row("Model", trace_record.model)
        table.add_row("Duration", f"{trace_record.duration_ms:.2f}ms")
        table.add_row("Total Tokens", str(trace_record.usage.total_tokens))
        table.add_row("Prompt Tokens", str(trace_record.usage.prompt_tokens))
        table.add_row("Completion Tokens", str(trace_record.usage.completion_tokens))
        table.add_row("Error", str(trace_record.error) if trace_record.error else "None")

        # Show optimizer context if available
        optimizer_context = trace_record.metadata.get("optimizer_context")
        if optimizer_context:
            table.add_row("", "")  # Empty row for separation
            table.add_row("OPTIMIZER CONTEXT", "")
            table.add_row("Optimization ID", optimizer_context.get("optimization_id", "N/A"))
            table.add_row("Generation", str(optimizer_context.get("generation", "N/A")))
            table.add_row("Population Size", str(optimizer_context.get("population_size", "N/A")))
            table.add_row("Base Prompt Name", optimizer_context.get("base_prompt_name", "N/A"))
            table.add_row("Start Time", optimizer_context.get("start_time", "N/A"))

        table.add_row("Prompt", trace_record.rendered_prompt)
        table.add_row("Response", trace_record.response)

        console.print(table)

    try:
        tracer = Tracer()

        if trace_id:
            _view_trace(tracer, trace_id)
        else:
            _list_traces_table(tracer)

    except Exception as e:
        click.echo(f"Error: {e}")


@main.command()
@click.option("--base-prompt", "-p", required=True, help="Base prompt template to optimize")
@click.option(
    "--test-cases",
    "-t",
    help="Path to JSON file containing test cases (optional for quality-based optimization)",
)
@click.option("--population-size", default=10, help="Population size for genetic algorithm")
@click.option("--generations", default=5, help="Number of generations to run")
@click.option(
    "--model",
    "-m",
    default="openai/gpt-3.5-turbo",
    help="Model to use for prompt execution, format: provider/model",
)
@click.option(
    "--eval-model",
    default="openai/gpt-5-mini-2025-08-07",
    help="Model to use for evaluation, format: provider/model",
)
@click.option("--api-key", help="API key for the provider")
@click.option("--mutation-rate", default=0.3, help="Mutation rate (0.0-1.0)")
@click.option("--crossover-rate", default=0.7, help="Crossover rate (0.0-1.0)")
@click.option("--elite-ratio", default=0.2, help="Ratio of elite individuals to preserve (0.0-1.0)")
@click.option(
    "--max-concurrent-evaluations",
    default=10,
    help="Maximum concurrent evaluation API calls (higher = faster but more load)",
)
@click.option(
    "--test-case-concurrency",
    default=10,
    help="Maximum concurrent test case executions per prompt (higher = faster)",
)
@click.option("--trace", is_flag=True, help="Enable tracing", default=False)
@click.option(
    "--trace-optimizer",
    is_flag=True,
    help="Enable tracing of optimizer prompts with separate context",
    default=False,
)
@click.option(
    "--output",
    "-o",
    help="Output file to save the optimized prompt",
    default="optimized_prompt.json",
)
@click.option(
    "--yes", "-y", is_flag=True, help="Skip confirmation prompt and proceed automatically"
)
@click.option(
    "--population-diversity",
    default=0.7,
    help="Diversity level for LLM population generation (0.0-1.0)",
)
def optimize(
    base_prompt: str,
    test_cases: str,
    population_size: int,
    generations: int,
    model: str,
    eval_model: str,
    api_key: Optional[str],
    mutation_rate: float,
    crossover_rate: float,
    elite_ratio: float,
    max_concurrent_evaluations: int,
    test_case_concurrency: int,
    trace: bool,
    trace_optimizer: bool,
    output: Optional[str],
    yes: bool,
    population_diversity: float,
) -> None:
    """Optimize a prompt using LLM-powered genetic algorithm (strictly LLM-driven)"""

    eval_model_provider, eval_model_name = eval_model.split("/")
    model_provider, model_name = model.split("/")

    async def _run_optimization():
        try:
            # Load test cases if provided
            test_cases_list = None
            if test_cases:
                try:
                    with open(test_cases) as f:
                        test_data = json.load(f)
                except Exception as e:
                    click.echo(f"Error loading test cases: {e}")
                    return

                # Parse test cases
                test_cases_list = []
                for test_case in test_data["test_cases"]:
                    test_cases_list.append(
                        PromptTestCase(
                            input_variables=test_case["input_variables"],
                            expected_output=test_case["expected_output"],
                            metadata=test_case.get("metadata", {}),
                        )
                    )

                click.echo(f"Loaded {len(test_cases_list)} test cases")
            else:
                click.echo("No test cases provided - using quality-based optimization")

            # Initialize eval clients
            eval_client: BaseLLMClient
            if eval_model_provider == "openai":
                eval_client = OpenAIClient(api_key=api_key)
            elif eval_model_provider == "anthropic":
                eval_client = AnthropicClient(api_key=api_key)
            elif eval_model_provider == "google":
                eval_client = GoogleAIClient(api_key=api_key)
            else:
                click.echo(f"Unsupported provider: {eval_model_provider}")
                return

            # Initialize main client
            main_client: BaseLLMClient
            if model_provider == "openai":
                main_client = OpenAIClient(api_key=api_key)
            elif model_provider == "anthropic":
                main_client = AnthropicClient(api_key=api_key)
            elif model_provider == "google":
                main_client = GoogleAIClient(api_key=api_key)
            else:
                click.echo(f"Unsupported provider: {model_provider}")
                return

            # Initialize runner (only needed if test cases are provided)
            tracer = Tracer() if (trace or trace_optimizer) else None
            runner = PromptRunner(main_client, tracer)

            # Create base prompt template
            base_template = PromptTemplate(template=base_prompt, name="base_prompt")

            fitness_function = LLMComprehensiveFitnessFunction(eval_client, eval_model_name)

            # Initialize optimizer (callback will be set later)
            optimizer = LLMGeneticOptimizer(
                eval_model=eval_model_name,
                population_size=population_size,
                generations=generations,
                fitness_function=fitness_function,
                tracer=tracer,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elite_ratio=elite_ratio,
                eval_client=eval_client,
                population_diversity_level=population_diversity,
                max_concurrent_evaluations=max_concurrent_evaluations,
                test_case_concurrency=test_case_concurrency,
            )

            # Calculate and display API call estimates
            api_calls = _calculate_api_calls(
                population_size=population_size,
                generations=generations,
                test_cases_count=len(test_cases_list) if test_cases_list else 0,
                has_test_cases=test_cases_list is not None,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
            )

            console = Console()

            # Welcome banner
            console.print(
                Panel.fit(
                    "[bold blue]🧬 Promptly Genetic Optimizer[/bold blue]\n"
                    "[dim]Powered by LLM-driven evolution[/dim]",
                    border_style="blue",
                )
            )
            console.print()

            # Configuration panel
            config_table = Table(title="⚙️ Optimization Configuration", box=box.ROUNDED)
            config_table.add_column("Parameter", style="cyan", no_wrap=True)
            config_table.add_column("Value", style="white")

            config_table.add_row("Population Size", f"[bold green]{population_size}[/bold green]")
            config_table.add_row("Generations", f"[bold green]{generations}[/bold green]")
            config_table.add_row(
                "Population Diversity", f"[bold green]{population_diversity}[/bold green]"
            )
            config_table.add_row(
                "Max Concurrent Evaluations", f"[bold cyan]{max_concurrent_evaluations}[/bold cyan]"
            )
            config_table.add_row(
                "Test Case Concurrency", f"[bold cyan]{test_case_concurrency}[/bold cyan]"
            )

            if test_cases_list:
                config_table.add_row(
                    "Test Cases", f"[bold green]{len(test_cases_list)}[/bold green]"
                )
                config_table.add_row("Mode", "[bold blue]Test case-based optimization[/bold blue]")
            else:
                config_table.add_row("Mode", "[bold blue]Quality-based optimization[/bold blue]")

            console.print(config_table)
            console.print()

            # API call estimates table
            api_table = Table(title="📊 API Call Estimates", box=box.ROUNDED)
            api_table.add_column("Operation", style="cyan", no_wrap=True)
            api_table.add_column("Calls", style="green", justify="right")

            api_table.add_row(
                "🧬 Population Generation",
                f"[bold green]{api_calls['population_generation']}[/bold green]",
            )
            api_table.add_row(
                "🎯 Evaluation", f"[bold green]{api_calls['evaluation']}[/bold green]"
            )
            api_table.add_row("🔄 Mutation", f"[bold green]{api_calls['mutation']}[/bold green]")
            api_table.add_row("🔀 Crossover", f"[bold green]{api_calls['crossover']}[/bold green]")
            api_table.add_row(
                "⚡ Prompt Execution", f"[bold green]{api_calls['execution']}[/bold green]"
            )
            api_table.add_row("", "", style="dim")  # Separator
            api_table.add_row(
                "[bold white]TOTAL API CALLS[/bold white]",
                f"[bold yellow]{api_calls['total']}[/bold yellow]",
            )

            console.print(api_table)
            console.print()

            # Cost estimation
            cost_estimate = _estimate_cost(api_calls, eval_model, model)
            if cost_estimate:
                cost_table = Table(title="💰 Estimated Cost Breakdown", box=box.ROUNDED)
                cost_table.add_column("Model/Operation", style="cyan", no_wrap=True)
                cost_table.add_column("Estimated Cost", style="green", justify="right")

                cost_table.add_row(
                    f"🎯 Evaluation ({eval_model})",
                    f"[bold green]~${cost_estimate['eval_cost']:.2f}[/bold green]",
                )
                if cost_estimate["execution_cost"] > 0:
                    cost_table.add_row(
                        f"⚡ Execution ({model})",
                        f"[bold green]~${cost_estimate['execution_cost']:.2f}[/bold green]",
                    )
                if cost_estimate["mutation_cost"] > 0:
                    cost_table.add_row(
                        "🔄 Mutation",
                        f"[bold green]~${cost_estimate['mutation_cost']:.2f}[/bold green]",
                    )
                if cost_estimate["crossover_cost"] > 0:
                    cost_table.add_row(
                        "🔀 Crossover",
                        f"[bold green]~${cost_estimate['crossover_cost']:.2f}[/bold green]",
                    )
                cost_table.add_row("", "", style="dim")  # Separator
                cost_table.add_row(
                    "[bold white]TOTAL ESTIMATED COST[/bold white]",
                    f"[bold yellow]~${cost_estimate['total_cost']:.2f}[/bold yellow]",
                )

                console.print(cost_table)
                console.print("[dim](Note: Actual costs may vary based on token usage)[/dim]")
                console.print()

            # Ask for confirmation (unless --yes flag is used)
            if not yes:
                console.print(
                    Panel(
                        "[bold green]🚀 Ready to evolve your prompt![/bold green]\n\n"
                        "The genetic algorithm will now begin optimizing your prompt through:\n"
                        "• 🧬 Intelligent population generation\n"
                        "• 🎯 LLM-powered fitness evaluation\n"
                        "• 🔄 Smart mutation and crossover\n"
                        "• 🏆 Elite selection and evolution",
                        title="Optimization Ready",
                        border_style="green",
                    )
                )
                console.print()

                console.print(
                    "[bold cyan]Do you want to proceed with the optimization?[/bold cyan]"
                )
                if not click.confirm(""):
                    console.print("[bold red]❌ Optimization cancelled.[/bold red]")
                    return
            else:
                console.print(
                    "[bold green]✅ Proceeding automatically (--yes flag provided)[/bold green]"
                )
                console.print()

            # Run optimization with progress indication
            console.print()
            console.print("[bold cyan]🚀 Starting genetic optimization...[/bold cyan]")
            console.print(
                "[dim]This may take a few minutes depending on population size and generations.[/dim]"
            )
            console.print()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                optimization_task = progress.add_task("Optimizing prompt...", total=100)

                # Create and set the progress callback
                progress_callback = RichProgressCallback(console, progress, optimization_task)
                optimizer.progress_callback = progress_callback

                # Start optimization
                result = await optimizer.optimize(
                    runner=runner,
                    base_prompt=base_template,
                    model=model_name,
                    test_cases=test_cases_list,
                )

            # Display results with celebration
            console.print()
            console.print(
                Panel.fit(
                    "[bold green]🎉 Optimization Complete![/bold green]\n"
                    "[dim]Your prompt has evolved to perfection![/dim]",
                    border_style="green",
                )
            )
            console.print()

            # Results summary table
            results_table = Table(title="🏆 Optimization Results", box=box.ROUNDED)
            results_table.add_column("Metric", style="cyan", no_wrap=True)
            results_table.add_column("Value", style="white", justify="right")

            # Fitness score with color coding
            fitness_color = (
                "green"
                if result.fitness_score >= 0.8
                else "yellow"
                if result.fitness_score >= 0.6
                else "red"
            )
            results_table.add_row(
                "🎯 Best Fitness Score",
                f"[bold {fitness_color}]{result.fitness_score:.3f}[/bold {fitness_color}]",
            )

            results_table.add_row(
                "🧬 Total Evaluations", f"[bold green]{result.total_evaluations}[/bold green]"
            )
            results_table.add_row(
                "⏱️ Optimization Time", f"[bold green]{result.optimization_time:.2f}s[/bold green]"
            )
            results_table.add_row(
                "🔄 Generations", f"[bold green]{result.generation + 1}[/bold green]"
            )
            results_table.add_row(
                "👥 Population Size", f"[bold green]{result.population_size}[/bold green]"
            )

            console.print(results_table)
            console.print()

            # Best prompt display
            console.print(
                Panel(
                    f"[bold blue]{result.best_prompt.template}[/bold blue]",
                    title="🏆 Optimized Prompt",
                    border_style="blue",
                    expand=False,
                )
            )
            console.print()

            # Save to file if requested
            if output:
                result.best_prompt.save(output)
                console.print(
                    f"[bold green]💾 Optimized prompt saved to: [cyan]{output}[/cyan][/bold green]"
                )
                console.print()

        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]❌ Error during optimization[/bold red]\n\n[red]{str(e)}[/red]",
                    title="Optimization Failed",
                    border_style="red",
                )
            )
            raise

    asyncio.run(_run_optimization())


def _calculate_api_calls(
    population_size: int,
    generations: int,
    test_cases_count: int,
    has_test_cases: bool,
    mutation_rate: float,
    crossover_rate: float,
) -> dict:
    """Calculate estimated API calls for optimization (strictly LLM-driven)"""

    # Base calculations
    total_evaluations = population_size * generations

    # Evaluation calls (one per individual per generation)
    evaluation_calls = total_evaluations

    # Prompt execution calls
    if has_test_cases and test_cases_count > 0:
        # If test cases are provided, run the prompt once for each test case
        execution_calls = total_evaluations * test_cases_count
    else:
        # If no test cases are provided, run the prompt once for quality evaluation on output
        execution_calls = 1 * total_evaluations

    # Mutation calls (based on actual mutation rate)
    mutation_calls = 0
    # Each generation, mutation_rate * population_size individuals get mutated
    mutation_calls = int(generations * population_size * mutation_rate)

    # Crossover calls (based on actual crossover rate)
    crossover_calls = 0
    # Each generation, crossover_rate * population_size individuals get crossed over
    # Each crossover produces 2 offspring, so we need crossover_rate * population_size / 2 crossover operations
    crossover_calls = int(generations * population_size * crossover_rate * 0.5)

    # Population generation calls (always uses LLM - one call at the beginning)
    population_generation_calls = 1

    total_calls = (
        evaluation_calls
        + execution_calls
        + mutation_calls
        + crossover_calls
        + population_generation_calls
    )

    return {
        "evaluation": evaluation_calls,
        "execution": execution_calls,
        "mutation": mutation_calls,
        "crossover": crossover_calls,
        "population_generation": population_generation_calls,
        "total": total_calls,
    }


def _estimate_cost(api_calls: dict, eval_model: str, exec_model: str) -> dict:
    """Estimate cost based on API calls and model pricing"""

    # Pricing per 1K tokens (input/output average)
    # Updated 2024/2025 pricing - actual costs may vary
    pricing = {
        "gpt-3.5-turbo": 0.0005,  # $0.0005 per 1K tokens (input: $0.0005, output: $0.0015)
        "gpt-4": 0.03,  # $0.03 per 1K tokens (input: $0.03, output: $0.06)
        "gpt-4-turbo": 0.01,  # $0.01 per 1K tokens (input: $0.01, output: $0.03)
        "gpt-4o": 0.005,  # $0.005 per 1K tokens (input: $0.005, output: $0.015)
        "gpt-4o-mini": 0.00015,  # $0.00015 per 1K tokens (input: $0.00015, output: $0.0006)
        "claude-3-sonnet-20240229": 0.003,  # $0.003 per 1K tokens (input: $0.003, output: $0.015)
        "claude-3-opus-20240229": 0.015,  # $0.015 per 1K tokens (input: $0.015, output: $0.075)
        "claude-3-haiku-20240307": 0.00025,  # $0.00025 per 1K tokens (input: $0.00025, output: $0.00125)
        "claude-3.5-sonnet": 0.003,  # $0.003 per 1K tokens (input: $0.003, output: $0.015)
    }

    # Estimate tokens per call (rough estimates)
    eval_tokens_per_call = 1000  # Evaluation prompts are typically longer
    exec_tokens_per_call = 500  # Execution calls vary based on test cases
    mutation_tokens_per_call = 800  # Mutation prompts are medium length
    crossover_tokens_per_call = 1200  # Crossover prompts are longer

    def get_price(model: str) -> float:
        # Try exact match first
        if model in pricing:
            return pricing[model]
        # Try partial matches
        if "gpt-4" in model.lower():
            return pricing["gpt-4"]
        elif "gpt-3.5" in model.lower():
            return pricing["gpt-3.5-turbo"]
        elif "claude" in model.lower():
            return pricing["claude-3-sonnet-20240229"]
        else:
            return 0.01  # Default estimate

    eval_price = get_price(eval_model)
    exec_price = get_price(exec_model)

    # Calculate costs
    eval_cost = (api_calls["evaluation"] * eval_tokens_per_call / 1000) * eval_price
    exec_cost = (api_calls["execution"] * exec_tokens_per_call / 1000) * exec_price
    mutation_cost = (api_calls["mutation"] * mutation_tokens_per_call / 1000) * eval_price
    crossover_cost = (api_calls["crossover"] * crossover_tokens_per_call / 1000) * eval_price

    total_cost = eval_cost + exec_cost + mutation_cost + crossover_cost

    return {
        "eval_cost": eval_cost,
        "execution_cost": exec_cost,
        "mutation_cost": mutation_cost,
        "crossover_cost": crossover_cost,
        "total_cost": total_cost,
    }


@main.command()
@click.option(
    "--test-cases",
    "-t",
    required=True,
    help="Path to JSON file containing test cases",
    default="./test_cases.json",
)
@click.option("--output", "-o", help="Output file to save the test cases template")
def init_test_cases(test_cases: str, output: Optional[str]) -> None:
    """Initialize a test cases file template"""

    template = {
        "description": "Test cases for prompt optimization",
        "test_cases": [
            {
                "input_variables": {"example_variable": "example_value"},
                "expected_output": "expected response",
                "metadata": {"description": "Description of this test case"},
            }
        ],
    }

    output_file = output or test_cases

    with open(output_file, "w") as f:
        json.dump(template, f, indent=2)

    click.echo(f"Test cases template created: {output_file}")
    click.echo("Edit the file to add your test cases and then run the optimize command.")


if __name__ == "__main__":
    main()
