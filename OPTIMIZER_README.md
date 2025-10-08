# LLM-Powered Genetic Prompt Optimizer

The Promptly optimizer module provides intelligent prompt optimization using LLM-powered genetic algorithms. This allows you to automatically improve your prompts through iterative evaluation and mutation.

## Features

- **LLM-Powered Evaluation**: Uses advanced LLMs to intelligently evaluate prompt performance
- **Intelligent Mutation**: LLM-driven prompt improvements and variations
- **Smart Crossover**: Combines the best elements from different prompts
- **Multiple Fitness Functions**: Accuracy-based and semantic similarity evaluation
- **Quality-Based Optimization**: Optimize prompts without test cases using general quality criteria
- **CLI Integration**: Easy-to-use command-line interface
- **Comprehensive Testing**: Full test coverage with mock implementations

## Quick Start

### 1. Install Dependencies

Make sure you have your API keys set:
```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"  # Optional
```

### 2. Create Test Cases

```bash
python -m promptly.cli.main init-test-cases -t my_test_cases.json
```

Edit `my_test_cases.json`:
```json
{
  "description": "Math Q&A test cases",
  "test_cases": [
    {
      "input_variables": {
        "question": "What is 2+2?"
      },
      "expected_output": "4",
      "metadata": {
        "difficulty": "easy"
      }
    },
    {
      "input_variables": {
        "question": "What is 15-7?"
      },
      "expected_output": "8",
      "metadata": {
        "difficulty": "easy"
      }
    }
  ]
}
```

### 3. Run Optimization

**With Test Cases:**
```bash
python -m promptly.cli.main optimize \
  --base-prompt "Answer this question: {{question}}" \
  --test-cases my_test_cases.json \
  --population-size 10 \
  --generations 5 \
  --provider openai
```

**Without Test Cases (Quality-Based):**
```bash
python -m promptly.cli.main optimize \
  --base-prompt "Write a {{genre}} story about {{character}}" \
  --population-size 8 \
  --generations 4 \
  --provider openai
```

## CLI Examples

### Basic Math Q&A Optimization

```bash
# Create test cases for math problems
python -m promptly.cli.main init-test-cases -t math_tests.json

# Edit math_tests.json with your test cases, then optimize
python -m promptly.cli.main optimize \
  --base-prompt "What is {{question}}?" \
  --test-cases math_tests.json \
  --population-size 8 \
  --generations 3 \
  --model gpt-3.5-turbo \
  --eval-model gpt-4 \
  --output optimized_math_prompt.json
```

### Creative Writing Prompt Optimization

```bash
# Create test cases for creative writing
python -m promptly.cli.main init-test-cases -t writing_tests.json

# Optimize with semantic evaluation
python -m promptly.cli.main optimize \
  --base-prompt "Write a {{genre}} story about {{character}}" \
  --test-cases writing_tests.json \
  --population-size 6 \
  --generations 4 \
  --provider anthropic \
  --model claude-3-sonnet-20240229 \
  --mutation-rate 0.5 \
  --crossover-rate 0.8
```

### Code Generation Optimization

```bash
# Optimize code generation prompts
python -m promptly.cli.main optimize \
  --base-prompt "Write {{language}} code to {{task}}" \
  --test-cases code_tests.json \
  --population-size 10 \
  --generations 5 \
  --model gpt-4 \
  --eval-model gpt-4 \
  --elite-size 3 \
  --mutation-rate 0.3 \
  --crossover-rate 0.7 \
  --trace \
  --output optimized_code_prompt.json
```

### High-Performance Optimization

```bash
# Larger population for better results (higher cost)
python -m promptly.cli.main optimize \
  --base-prompt "{{instruction}}" \
  --test-cases comprehensive_tests.json \
  --population-size 20 \
  --generations 10 \
  --model gpt-4 \
  --eval-model gpt-4 \
  --provider openai \
  --api-key $OPENAI_API_KEY \
  --elite-size 4 \
  --mutation-rate 0.4 \
  --crossover-rate 0.8 \
  --trace
```

### Quick Test Run

```bash
# Minimal configuration for testing
python -m promptly.cli.main optimize \
  --base-prompt "{{question}}" \
  --test-cases quick_tests.json \
  --population-size 4 \
  --generations 2 \
  --model gpt-3.5-turbo \
  --eval-model gpt-3.5-turbo
```

### Anthropic Claude Optimization

```bash
# Use Anthropic Claude for optimization
python -m promptly.cli.main optimize \
  --base-prompt "{{prompt}}" \
  --test-cases claude_tests.json \
  --population-size 8 \
  --generations 4 \
  --provider anthropic \
  --model claude-3-sonnet-20240229 \
  --eval-model claude-3-sonnet-20240229 \
  --api-key $ANTHROPIC_API_KEY
```

### Automated Optimization (Skip Confirmation)

```bash
# Skip confirmation prompt for automation/scripting
python -m promptly.cli.main optimize \
  --base-prompt "{{instruction}}" \
  --test-cases automation_tests.json \
  --population-size 6 \
  --generations 3 \
  --provider openai \
  --model gpt-3.5-turbo \
  --eval-model gpt-4 \
  --yes \
  --output optimized_prompt.json
```

### Quality-Based Optimization (No Test Cases)

```bash
# Optimize prompt quality without specific test cases
python -m promptly.cli.main optimize \
  --base-prompt "Write a creative story about {{topic}}" \
  --population-size 6 \
  --generations 3 \
  --provider openai \
  --model gpt-4 \
  --eval-model gpt-4 \
  --mutation-rate 0.4 \
  --crossover-rate 0.8
```

### Exploratory Prompt Discovery

```bash
# Discover new prompt patterns through quality optimization
python -m promptly.cli.main optimize \
  --base-prompt "{{instruction}}" \
  --population-size 12 \
  --generations 6 \
  --provider openai \
  --model gpt-4 \
  --eval-model gpt-4 \
  --elite-size 3 \
  --trace \
  --output discovered_prompt.json
```

## Optimization Modes

The optimizer supports two distinct modes:

### 1. Test Case-Based Optimization
- **Use Case**: When you have specific input/output examples
- **Evaluation**: Tests prompts against known correct answers
- **Best For**: Specific tasks like math problems, factual Q&A, code generation
- **Requirements**: Test cases file with input variables and expected outputs

### 2. Quality-Based Optimization
- **Use Case**: When you want to improve general prompt quality
- **Evaluation**: LLM evaluates prompt clarity, completeness, and effectiveness
- **Best For**: Creative writing, general instructions, exploratory optimization
- **Requirements**: Only a base prompt template needed

**Quality Evaluation Criteria:**
- Clarity: Is the prompt clear and unambiguous?
- Completeness: Does it provide sufficient context and instructions?
- Specificity: Is it specific enough to guide good responses?
- Structure: Is it well-structured and easy to follow?
- Effectiveness: Would this prompt likely produce high-quality responses?
- Template Variables: Are template variables appropriately used?

## Cost Management

The optimizer provides detailed API call estimation and cost analysis before running optimization to avoid costly accidents.

## Python API Usage

### Basic Optimization

```python
import asyncio
from promptly import (
    LLMGeneticOptimizer,
    LLMComprehensiveFitnessFunction,
    PromptTestCase,
    PromptTemplate,
    PromptRunner,
    OpenAIClient,
)

async def optimize_prompt():
    # Setup clients
    main_client = OpenAIClient()
    eval_client = OpenAIClient()
    
    runner = PromptRunner(main_client)
    
    # Create test cases
    test_cases = [
        PromptTestCase(
            input_variables={"question": "What is 2+2?"},
            expected_output="4"
        ),
        PromptTestCase(
            input_variables={"question": "What is 3+3?"},
            expected_output="6"
        ),
    ]
    
    # Create base prompt
    base_prompt = PromptTemplate(
        template="Answer this question: {{question}}",
        name="math_qa_prompt"
    )
    
    # Setup optimizer
    optimizer = LLMGeneticOptimizer(
        eval_model="gpt-4",
        population_size=10,
        generations=5,
        fitness_function=LLMComprehensiveFitnessFunction(eval_client, "gpt-4"),
        eval_client=eval_client,
    )
    
    # Run optimization
    result = await optimizer.optimize(runner, base_prompt, test_cases)
    
    print(f"Best prompt: {result.best_prompt.template}")
    print(f"Fitness score: {result.fitness_score}")

# Run the optimization
asyncio.run(optimize_prompt())
```

### Quality-Based Optimization (No Test Cases)

```python
import asyncio
from promptly import (
    LLMGeneticOptimizer,
    LLMComprehensiveFitnessFunction,
    PromptTemplate,
    PromptRunner,
    OpenAIClient,
)

async def optimize_prompt_quality():
    # Setup clients
    eval_client = OpenAIClient()
    
    # Create runner for optimization
    runner = PromptRunner(eval_client)
    
    # Create base prompt
    base_prompt = PromptTemplate(
        template="Write a story about {{character}} in {{setting}}",
        name="story_prompt"
    )
    
    # Setup optimizer for quality-based optimization
    optimizer = LLMGeneticOptimizer(
        eval_model="gpt-4",
        population_size=8,
        generations=4,
        fitness_function=LLMComprehensiveFitnessFunction(eval_client, "gpt-4"),
        eval_client=eval_client,
    )
    
    # Run optimization without test cases (quality-based)
    result = await optimizer.optimize(runner, base_prompt, test_cases=None)
    
    print(f"Best prompt: {result.best_prompt.template}")
    print(f"Quality score: {result.fitness_score}")

# Run the quality-based optimization
asyncio.run(optimize_prompt_quality())
```

### Advanced Configuration

```python
from promptly import (
    LLMGeneticOptimizer,
    LLMComprehensiveFitnessFunction,
    OpenAIClient,
    Tracer
)

# Setup client
eval_client = OpenAIClient()

# Configure comprehensive fitness function
fitness_function = LLMComprehensiveFitnessFunction(
    evaluation_client=eval_client,
    evaluation_model="gpt-4"  # Use GPT-4 for evaluation
)

# Configure optimizer with advanced options
optimizer = LLMGeneticOptimizer(
    eval_model="gpt-4",
    population_size=20,
    generations=10,
    fitness_function=fitness_function,
    eval_client=eval_client,
    mutation_rate=0.3,                    # 30% chance of mutation
    crossover_rate=0.7,                   # 70% chance of crossover
    elite_size=2,                         # Keep top 2 individuals
    population_diversity_level=0.7,       # Diversity level for LLM population generation
    tracer=Tracer()                       # Enable tracing
)
```

## CLI Commands

### Optimize Command

```bash
python -m promptly.cli.main optimize [OPTIONS]

Options:
  -p, --base-prompt TEXT          Base prompt template to optimize [required]
  -t, --test-cases TEXT           Path to JSON file containing test cases [required]
  --population-size INTEGER       Population size for genetic algorithm (default: 10)
  --generations INTEGER           Number of generations to run (default: 5)
  -m, --model TEXT                Model to use for prompt execution (default: gpt-3.5-turbo)
  --eval-model TEXT               Model to use for evaluation (default: gpt-5-mini-2025-08-07)
  --provider [openai|anthropic]   LLM provider (default: openai)
  --api-key TEXT                  API key for the provider
  --mutation-rate FLOAT           Mutation rate 0.0-1.0 (default: 0.3)
  --crossover-rate FLOAT          Crossover rate 0.0-1.0 (default: 0.7)
  --elite-size INTEGER            Number of elite individuals to preserve (default: 2)
  --use-llm-population            Use LLM to generate initial population (default: True)
  --population-diversity FLOAT    Diversity level for population generation 0.0-1.0 (default: 0.7)
  --trace                         Enable tracing (default: True)
  -o, --output TEXT               Output file to save the optimized prompt
  -y, --yes                       Skip confirmation prompt and proceed automatically
```

## Architecture

### Core Components

1. **LLMFitnessFunction**: Abstract base class for fitness evaluation
   - `LLMComprehensiveFitnessFunction`: Comprehensive evaluation combining accuracy and semantic similarity

2. **LLMPromptMutator**: LLM-powered prompt mutation
   - Intelligent prompt improvements using LLM
   - Strictly LLM-driven (no fallbacks)

3. **LLMPromptCrossover**: LLM-powered prompt crossover
   - Combines best elements from parent prompts using LLM
   - Strictly LLM-driven (no fallbacks)

4. **LLMPopulationGenerator**: LLM-powered population generation
   - Creates diverse initial populations using LLM
   - Configurable diversity levels

5. **LLMGeneticOptimizer**: Main optimization engine
   - Genetic algorithm implementation
   - Population management
   - Evolution loop
   - Progress callbacks for monitoring


## Examples

See `examples/optimizer_example.py` for comprehensive examples including:

- Math Q&A optimization
- Creative writing optimization  
- Code generation optimization


## Best Practices

1. **Test Case Quality**: Create diverse, representative test cases
2. **Base Prompt**: Start with a reasonable initial prompt
3. **Evaluation Model**: Use stronger models for better evaluation quality
4. **Iterative Approach**: Run multiple optimization cycles
5. **Manual Review**: Always review optimized prompts manually

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce population size or add delays
2. **Poor Results**: Check test case quality and base prompt
3. **High Costs**: Use smaller populations and fewer generations
4. **Evaluation Errors**: Ensure evaluation model has sufficient context

### Debug Mode

Enable tracing to monitor optimization progress:

```python
optimizer = LLMGeneticOptimizer(
    # ... other params ...
    tracer=Tracer()
)
```
