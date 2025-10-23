"""
Example of quality-based prompt optimization without test cases
"""

import asyncio

from promptly import (
    LLMComprehensiveFitnessFunction,
    LLMGeneticOptimizer,
    OpenAIClient,
    PromptRunner,
    PromptTemplate,
)


async def quality_optimization_example():
    """Example: Optimize a prompt based on quality criteria alone"""

    # Setup clients
    eval_client = OpenAIClient()

    # Create runner for optimization
    runner = PromptRunner(client=eval_client)

    # Create base prompt
    base_prompt = PromptTemplate(template="Write about {{topic}}", name="generic_writing_prompt")

    # Setup optimizer for quality-based optimization
    optimizer = LLMGeneticOptimizer(
        eval_model="gpt-4",
        population_size=6,
        generations=3,
        fitness_function=LLMComprehensiveFitnessFunction(eval_client, "gpt-4"),
        eval_client=eval_client,
        mutation_rate=0.4,
        crossover_rate=0.7,
    )

    print("Starting quality-based optimization...")
    print(f"Base prompt: {base_prompt.template}")
    print(f"Population size: {optimizer.population_size}")
    print(f"Generations: {optimizer.generations}")
    print("Mode: Quality-based optimization (no test cases)")
    print()

    # Run optimization without test cases (quality-based)
    result = await optimizer.optimize(runner, base_prompt, test_cases=None)

    print("Optimization completed!")
    print(f"Best quality score: {result.fitness_score:.3f}")
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Optimization time: {result.optimization_time:.2f}s")
    print()
    print("Best prompt:")
    print("=" * 60)
    print(result.best_prompt.template)
    print("=" * 60)

    return result


async def creative_writing_optimization():
    """Example: Optimize creative writing prompts"""

    # Setup clients
    eval_client = OpenAIClient()

    # Create runner for optimization
    runner = PromptRunner(client=eval_client)

    # Create base prompt for creative writing
    base_prompt = PromptTemplate(
        template="Write a story with {{character}} and {{setting}}", name="creative_story_prompt"
    )

    # Setup optimizer
    optimizer = LLMGeneticOptimizer(
        eval_model="gpt-4",
        population_size=8,
        generations=4,
        fitness_function=LLMComprehensiveFitnessFunction(eval_client, "gpt-4"),
        eval_client=eval_client,
        mutation_rate=0.5,
        crossover_rate=0.8,
    )

    print("Starting creative writing optimization...")
    print(f"Base prompt: {base_prompt.template}")
    print()

    # Run optimization
    result = await optimizer.optimize(runner, base_prompt, test_cases=None)

    print("Creative writing optimization completed!")
    print(f"Best quality score: {result.fitness_score:.3f}")
    print()
    print("Optimized creative writing prompt:")
    print("=" * 60)
    print(result.best_prompt.template)
    print("=" * 60)

    return result


async def instruction_optimization():
    """Example: Optimize general instruction prompts"""

    # Setup clients
    eval_client = OpenAIClient()

    # Create runner for optimization
    runner = PromptRunner(client=eval_client)

    # Create base prompt for instructions
    base_prompt = PromptTemplate(template="Help with {{task}}", name="instruction_prompt")

    # Setup optimizer
    optimizer = LLMGeneticOptimizer(
        eval_model="gpt-4",
        population_size=6,
        generations=3,
        fitness_function=LLMComprehensiveFitnessFunction(eval_client, "gpt-4"),
        eval_client=eval_client,
        mutation_rate=0.3,
        crossover_rate=0.7,
    )

    print("Starting instruction optimization...")
    print(f"Base prompt: {base_prompt.template}")
    print()

    # Run optimization
    result = await optimizer.optimize(runner, base_prompt, test_cases=None)

    print("Instruction optimization completed!")
    print(f"Best quality score: {result.fitness_score:.3f}")
    print()
    print("Optimized instruction prompt:")
    print("=" * 60)
    print(result.best_prompt.template)
    print("=" * 60)

    return result


async def main():
    """Run all quality optimization examples"""

    print("Quality-Based Prompt Optimization Examples")
    print("=" * 50)
    print()

    # Check for API key
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found. Examples may fail.")
        print("Set your API key: export OPENAI_API_KEY=your_key_here")
        print()

    try:
        # Example 1: Generic writing prompt
        print("Example 1: Generic Writing Prompt Optimization")
        print("-" * 40)
        await quality_optimization_example()
        print()

        # Example 2: Creative writing
        print("Example 2: Creative Writing Prompt Optimization")
        print("-" * 40)
        await creative_writing_optimization()
        print()

        # Example 3: Instructions
        print("Example 3: Instruction Prompt Optimization")
        print("-" * 40)
        await instruction_optimization()
        print()

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have a valid OPENAI_API_KEY set in your environment.")


if __name__ == "__main__":
    asyncio.run(main())
