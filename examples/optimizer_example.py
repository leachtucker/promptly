"""
Example usage of the LLM-powered prompt optimizer
"""

import asyncio
import os

from promptly import (
    AnthropicClient,
    LLMComprehensiveFitnessFunction,
    LLMGeneticOptimizer,
    OpenAIClient,
    PromptRunner,
    PromptTemplate,
    PromptTestCase,
)


async def math_qa_optimization_example():
    """Example: Optimize a math Q&A prompt"""

    # Setup clients
    main_client = OpenAIClient()  # For running prompts
    eval_client = OpenAIClient()  # For evaluation

    runner = PromptRunner(main_client)

    # Create test cases
    test_cases = [
        PromptTestCase(
            input_variables={"question": "What is 2+2?"},
            expected_output="4",
            metadata={"difficulty": "easy"},
        ),
        PromptTestCase(
            input_variables={"question": "What is 15-7?"},
            expected_output="8",
            metadata={"difficulty": "easy"},
        ),
        PromptTestCase(
            input_variables={"question": "What is 6*7?"},
            expected_output="42",
            metadata={"difficulty": "medium"},
        ),
        PromptTestCase(
            input_variables={"question": "What is 100/4?"},
            expected_output="25",
            metadata={"difficulty": "medium"},
        ),
        PromptTestCase(
            input_variables={"question": "What is 2^3?"},
            expected_output="8",
            metadata={"difficulty": "hard"},
        ),
    ]

    # Create base prompt
    base_prompt = PromptTemplate(
        template="Answer this question: {{question}}", name="math_qa_prompt"
    )

    # Setup optimizer
    optimizer = LLMGeneticOptimizer(
        eval_model="gpt-4",
        population_size=8,  # Smaller for demo
        generations=3,  # Fewer for demo
        fitness_function=LLMComprehensiveFitnessFunction(eval_client, "gpt-4"),
        eval_client=eval_client,
        mutation_rate=0.4,
        crossover_rate=0.6,
    )

    print("Starting math Q&A optimization...")
    print(f"Test cases: {len(test_cases)}")
    print(f"Population size: {optimizer.population_size}")
    print(f"Generations: {optimizer.generations}")
    print()

    # Run optimization
    result = await optimizer.optimize(runner, base_prompt, test_cases)

    print("Optimization completed!")
    print(f"Best fitness score: {result.fitness_score:.3f}")
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Optimization time: {result.optimization_time:.2f}s")
    print()
    print("Best prompt:")
    print("=" * 60)
    print(result.best_prompt.template)
    print("=" * 60)

    return result


async def creative_writing_optimization_example():
    """Example: Optimize a creative writing prompt"""

    # Setup clients
    main_client = AnthropicClient()  # Use Claude for creative tasks
    eval_client = OpenAIClient()  # Use GPT-4 for evaluation

    runner = PromptRunner(main_client)

    # Create test cases for creative writing
    test_cases = [
        PromptTestCase(
            input_variables={"genre": "sci-fi", "character": "robot"},
            expected_output="A story about a robot in a science fiction setting",
            metadata={"genre": "sci-fi"},
        ),
        PromptTestCase(
            input_variables={"genre": "fantasy", "character": "wizard"},
            expected_output="A story about a wizard in a fantasy setting",
            metadata={"genre": "fantasy"},
        ),
        PromptTestCase(
            input_variables={"genre": "mystery", "character": "detective"},
            expected_output="A story about a detective in a mystery setting",
            metadata={"genre": "mystery"},
        ),
    ]

    # Create base prompt
    base_prompt = PromptTemplate(
        template="Write a story about {{character}} in the {{genre}} genre.",
        name="creative_writing_prompt",
    )

    # Setup optimizer with semantic fitness
    optimizer = LLMGeneticOptimizer(
        eval_model="gpt-4",
        population_size=6,
        generations=2,
        fitness_function=LLMComprehensiveFitnessFunction(eval_client, "gpt-4"),
        eval_client=eval_client,
        mutation_rate=0.5,
        crossover_rate=0.8,
    )

    print("Starting creative writing optimization...")
    print(f"Test cases: {len(test_cases)}")
    print("Using semantic fitness evaluation")
    print()

    # Run optimization
    result = await optimizer.optimize(
        runner, base_prompt, test_cases, model="claude-3-sonnet-20240229"
    )

    print("Optimization completed!")
    print(f"Best fitness score: {result.fitness_score:.3f}")
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Optimization time: {result.optimization_time:.2f}s")
    print()
    print("Best prompt:")
    print("=" * 60)
    print(result.best_prompt.template)
    print("=" * 60)

    return result


async def code_generation_optimization_example():
    """Example: Optimize a code generation prompt"""

    # Setup clients
    main_client = OpenAIClient()
    eval_client = OpenAIClient()

    runner = PromptRunner(main_client)

    # Create test cases for code generation
    test_cases = [
        PromptTestCase(
            input_variables={"language": "Python", "task": "sort a list"},
            expected_output="Python code that sorts a list",
            metadata={"language": "Python"},
        ),
        PromptTestCase(
            input_variables={"language": "JavaScript", "task": "reverse a string"},
            expected_output="JavaScript code that reverses a string",
            metadata={"language": "JavaScript"},
        ),
        PromptTestCase(
            input_variables={"language": "Python", "task": "find maximum in array"},
            expected_output="Python code that finds the maximum value in an array",
            metadata={"language": "Python"},
        ),
    ]

    # Create base prompt
    base_prompt = PromptTemplate(
        template="Write {{language}} code to {{task}}.", name="code_generation_prompt"
    )

    # Setup optimizer
    optimizer = LLMGeneticOptimizer(
        eval_model="gpt-4",
        population_size=6,
        generations=2,
        fitness_function=LLMComprehensiveFitnessFunction(eval_client, "gpt-4"),
        eval_client=eval_client,
        mutation_rate=0.3,
        crossover_rate=0.7,
    )

    print("Starting code generation optimization...")
    print(f"Test cases: {len(test_cases)}")
    print("Using GPT-4 for evaluation")
    print()

    # Run optimization
    result = await optimizer.optimize(runner, base_prompt, test_cases, model="gpt-4")

    print("Optimization completed!")
    print(f"Best fitness score: {result.fitness_score:.3f}")
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Optimization time: {result.optimization_time:.2f}s")
    print()
    print("Best prompt:")
    print("=" * 60)
    print(result.best_prompt.template)
    print("=" * 60)

    return result


async def main():
    """Run all examples"""

    print("LLM-Powered Prompt Optimizer Examples")
    print("=" * 50)
    print()

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found. Examples may fail.")
        print("Set your API key: export OPENAI_API_KEY=your_key_here")
        print()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not found. Some examples may fail.")
        print("Set your API key: export ANTHROPIC_API_KEY=your_key_here")
        print()

    try:
        # Example 1: Math Q&A
        print("Example 1: Math Q&A Optimization")
        print("-" * 30)
        await math_qa_optimization_example()
        print()

        # Example 2: Creative Writing
        print("Example 2: Creative Writing Optimization")
        print("-" * 30)
        await creative_writing_optimization_example()
        print()

        # Example 3: Code Generation
        print("Example 3: Code Generation Optimization")
        print("-" * 30)
        await code_generation_optimization_example()
        print()

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have valid API keys set in your environment.")


if __name__ == "__main__":
    asyncio.run(main())
