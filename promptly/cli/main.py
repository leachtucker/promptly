"""
Command-line interface for promptly
"""

from rich.console import Console
from rich.table import Table


import click
import asyncio
import json
from typing import Optional
from ..core.runner import PromptRunner
from ..core.tracer import Tracer
from ..core.clients import OpenAIClient, AnthropicClient
from ..core.optimizer import (
    LLMGeneticOptimizer,
    LLMAccuracyFitnessFunction,
    LLMSemanticFitnessFunction,
    TestCase,
)
from ..core.templates import PromptTemplate



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
            trace_records = tracer.list_records(limit=20)
            
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
            
            for record in trace_records:
                table.add_row(
                    str(record.id or "N/A"),
                    record.prompt_name,
                    record.prompt_template[:400] + "..." if len(record.prompt_template) > 400 else record.prompt_template,
                    record.response[:400] + "..." if len(record.response) > 400 else record.response,
                    record.model,
                    f"{record.duration_ms:.2f}",
                    str(record.error)[:30] if record.error else "",
                    record.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                )
            
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
@click.option("--test-cases", "-t", help="Path to JSON file containing test cases (optional for quality-based optimization)")
@click.option("--population-size", default=10, help="Population size for genetic algorithm")
@click.option("--generations", default=5, help="Number of generations to run")
@click.option("--model", "-m", default="gpt-3.5-turbo", help="Model to use for prompt execution")
@click.option("--eval-model", default="gpt-4", help="Model to use for evaluation")
@click.option("--provider", default="openai", type=click.Choice(["openai", "anthropic"]), help="LLM provider")
@click.option("--api-key", help="API key for the provider")
@click.option("--mutation-rate", default=0.3, help="Mutation rate (0.0-1.0)")
@click.option("--crossover-rate", default=0.7, help="Crossover rate (0.0-1.0)")
@click.option("--elite-size", default=2, help="Number of elite individuals to preserve")
@click.option("--fitness-type", default="accuracy", type=click.Choice(["accuracy", "semantic"]), help="Fitness function type")
@click.option("--trace", is_flag=True, help="Enable tracing", default=True)
@click.option("--output", "-o", help="Output file to save the optimized prompt")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt and proceed automatically")
def optimize(
    base_prompt: str,
    test_cases: str,
    population_size: int,
    generations: int,
    model: str,
    eval_model: str,
    provider: str,
    api_key: Optional[str],
    mutation_rate: float,
    crossover_rate: float,
    elite_size: int,
    fitness_type: str,
    trace: bool,
    output: Optional[str],
    yes: bool,
) -> None:
    """Optimize a prompt using LLM-powered genetic algorithm"""
    
    async def _run_optimization():
        try:
            # Load test cases if provided
            test_cases_list = None
            if test_cases:
                with open(test_cases, 'r') as f:
                    test_data = json.load(f)
                
                # Parse test cases
                test_cases_list = []
                for test_case in test_data["test_cases"]:
                    test_cases_list.append(TestCase(
                        input_variables=test_case["input_variables"],
                        expected_output=test_case["expected_output"],
                        metadata=test_case.get("metadata", {})
                    ))
                
                click.echo(f"Loaded {len(test_cases_list)} test cases")
            else:
                click.echo("No test cases provided - using quality-based optimization")
            
            # Initialize clients
            if provider == "openai":
                main_client = OpenAIClient(api_key=api_key)
                eval_client = OpenAIClient(api_key=api_key)
                mutation_client = OpenAIClient(api_key=api_key)
                crossover_client = OpenAIClient(api_key=api_key)
            elif provider == "anthropic":
                main_client = AnthropicClient(api_key=api_key)
                eval_client = AnthropicClient(api_key=api_key)
                mutation_client = AnthropicClient(api_key=api_key)
                crossover_client = AnthropicClient(api_key=api_key)
            else:
                click.echo(f"Unsupported provider: {provider}")
                return
            
            # Initialize runner (only needed if test cases are provided)
            tracer = Tracer() if trace else None
            runner = PromptRunner(main_client, tracer)
            
            # Create base prompt template
            base_template = PromptTemplate(template=base_prompt, name="base_prompt")
            
            # Initialize fitness function
            if fitness_type == "accuracy":
                fitness_function = LLMAccuracyFitnessFunction(eval_client, eval_model)
            elif fitness_type == "semantic":
                fitness_function = LLMSemanticFitnessFunction(eval_client, eval_model)
            else:
                click.echo(f"Unsupported fitness type: {fitness_type}")
                return
            
            # Initialize optimizer
            optimizer = LLMGeneticOptimizer(
                population_size=population_size,
                generations=generations,
                fitness_function=fitness_function,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elite_size=elite_size,
                mutation_client=mutation_client,
                crossover_client=crossover_client,
                tracer=tracer
            )
            
            # Calculate and display API call estimates
            api_calls = _calculate_api_calls(
                population_size=population_size,
                generations=generations,
                test_cases_count=len(test_cases_list) if test_cases_list else 0,
                has_test_cases=test_cases_list is not None,
                mutation_client=mutation_client is not None,
                crossover_client=crossover_client is not None,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate
            )
            
            click.echo("Optimization Configuration:")
            click.echo(f"Population size: {population_size}")
            click.echo(f"Generations: {generations}")
            click.echo(f"Fitness type: {fitness_type}")
            if test_cases_list:
                click.echo(f"Test cases: {len(test_cases_list)}")
                click.echo("Mode: Test case-based optimization")
            else:
                click.echo("Mode: Quality-based optimization")
            click.echo()
            
            click.echo("API Call Estimates:")
            click.echo(f"Evaluation calls: {api_calls['evaluation']}")
            click.echo(f"Mutation calls: {api_calls['mutation']}")
            click.echo(f"Crossover calls: {api_calls['crossover']}")
            click.echo(f"Prompt execution calls: {api_calls['execution']}")
            click.echo(f"TOTAL API CALLS: {api_calls['total']}")
            
            # Add cost estimation
            cost_estimate = _estimate_cost(api_calls, eval_model, model)
            if cost_estimate:
                click.echo()
                click.echo("Estimated Cost:")
                click.echo(f"Evaluation model ({eval_model}): ~${cost_estimate['eval_cost']:.2f}")
                if cost_estimate['execution_cost'] > 0:
                    click.echo(f"Execution model ({model}): ~${cost_estimate['execution_cost']:.2f}")
                if cost_estimate['mutation_cost'] > 0:
                    click.echo(f"Mutation model: ~${cost_estimate['mutation_cost']:.2f}")
                if cost_estimate['crossover_cost'] > 0:
                    click.echo(f"Crossover model: ~${cost_estimate['crossover_cost']:.2f}")
                click.echo(f"TOTAL ESTIMATED COST: ~${cost_estimate['total_cost']:.2f}")
                click.echo("(Note: Actual costs may vary based on token usage)")
            click.echo()
            
            # Ask for confirmation (unless --yes flag is used)
            if not yes:
                if not click.confirm("Do you want to proceed with the optimization?"):
                    click.echo("Optimization cancelled.")
                    return
            else:
                click.echo("Proceeding automatically (--yes flag provided)")
            
            # Run optimization
            result = await optimizer.optimize(base_template, test_cases_list, runner, model=model)
            
            # Display results
            click.echo("Optimization completed!")
            click.echo(f"Best fitness score: {result.fitness_score:.3f}")
            click.echo(f"Total evaluations: {result.total_evaluations}")
            click.echo(f"Optimization time: {result.optimization_time:.2f}s")
            click.echo()
            click.echo("Best prompt:")
            click.echo("=" * 50)
            click.echo(result.best_prompt.template)
            click.echo("=" * 50)
            
            # Save to file if requested
            if output:
                result.best_prompt.save(output)
                click.echo(f"Optimized prompt saved to: {output}")
            
        except Exception as e:
            click.echo(f"Error during optimization: {e}")
            raise
    
    asyncio.run(_run_optimization())


def _calculate_api_calls(
    population_size: int,
    generations: int,
    test_cases_count: int,
    has_test_cases: bool,
    mutation_client: bool,
    crossover_client: bool,
    mutation_rate: float,
    crossover_rate: float
) -> dict:
    """Calculate estimated API calls for optimization"""
    
    # Base calculations
    total_evaluations = population_size * generations
    
    # Evaluation calls (one per individual per generation)
    evaluation_calls = total_evaluations
    
    # Prompt execution calls (only if test cases are provided)
    if has_test_cases and test_cases_count > 0:
        execution_calls = total_evaluations * test_cases_count
    else:
        execution_calls = 0
    
    # Mutation calls (based on actual mutation rate)
    mutation_calls = 0
    if mutation_client:
        # Each generation, mutation_rate * population_size individuals get mutated
        mutation_calls = int(generations * population_size * mutation_rate)
    
    # Crossover calls (based on actual crossover rate)
    crossover_calls = 0
    if crossover_client:
        # Each generation, crossover_rate * population_size individuals get crossed over
        # Each crossover produces 2 offspring, so we need crossover_rate * population_size / 2 crossover operations
        crossover_calls = int(generations * population_size * crossover_rate * 0.5)
    
    total_calls = evaluation_calls + execution_calls + mutation_calls + crossover_calls
    
    return {
        "evaluation": evaluation_calls,
        "execution": execution_calls,
        "mutation": mutation_calls,
        "crossover": crossover_calls,
        "total": total_calls
    }


def _estimate_cost(api_calls: dict, eval_model: str, exec_model: str) -> dict:
    """Estimate cost based on API calls and model pricing"""
    
    # Pricing per 1K tokens (input/output average)
    # Updated 2024/2025 pricing - actual costs may vary
    pricing = {
        "gpt-3.5-turbo": 0.0005,  # $0.0005 per 1K tokens (input: $0.0005, output: $0.0015)
        "gpt-4": 0.03,            # $0.03 per 1K tokens (input: $0.03, output: $0.06)
        "gpt-4-turbo": 0.01,      # $0.01 per 1K tokens (input: $0.01, output: $0.03)
        "gpt-4o": 0.005,          # $0.005 per 1K tokens (input: $0.005, output: $0.015)
        "gpt-4o-mini": 0.00015,   # $0.00015 per 1K tokens (input: $0.00015, output: $0.0006)
        "claude-3-sonnet-20240229": 0.003,  # $0.003 per 1K tokens (input: $0.003, output: $0.015)
        "claude-3-opus-20240229": 0.015,    # $0.015 per 1K tokens (input: $0.015, output: $0.075)
        "claude-3-haiku-20240307": 0.00025, # $0.00025 per 1K tokens (input: $0.00025, output: $0.00125)
        "claude-3.5-sonnet": 0.003,         # $0.003 per 1K tokens (input: $0.003, output: $0.015)
    }
    
    # Estimate tokens per call (rough estimates)
    eval_tokens_per_call = 1000    # Evaluation prompts are typically longer
    exec_tokens_per_call = 500     # Execution calls vary based on test cases
    mutation_tokens_per_call = 800 # Mutation prompts are medium length
    crossover_tokens_per_call = 1200 # Crossover prompts are longer
    
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
    eval_cost = (api_calls['evaluation'] * eval_tokens_per_call / 1000) * eval_price
    exec_cost = (api_calls['execution'] * exec_tokens_per_call / 1000) * exec_price
    mutation_cost = (api_calls['mutation'] * mutation_tokens_per_call / 1000) * eval_price
    crossover_cost = (api_calls['crossover'] * crossover_tokens_per_call / 1000) * eval_price
    
    total_cost = eval_cost + exec_cost + mutation_cost + crossover_cost
    
    return {
        "eval_cost": eval_cost,
        "execution_cost": exec_cost,
        "mutation_cost": mutation_cost,
        "crossover_cost": crossover_cost,
        "total_cost": total_cost
    }


@main.command()
@click.option("--test-cases", "-t", required=True, help="Path to JSON file containing test cases", default="./test_cases.json")
@click.option("--output", "-o", help="Output file to save the test cases template")
def init_test_cases(test_cases: str, output: Optional[str]) -> None:
    """Initialize a test cases file template"""
    
    template = {
        "description": "Test cases for prompt optimization",
        "test_cases": [
            {
                "input_variables": {
                    "example_variable": "example_value"
                },
                "expected_output": "expected response",
                "metadata": {
                    "description": "Description of this test case"
                }
            }
        ]
    }
    
    output_file = output or test_cases
    
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    click.echo(f"Test cases template created: {output_file}")
    click.echo("Edit the file to add your test cases and then run the optimize command.")


if __name__ == "__main__":
    main()