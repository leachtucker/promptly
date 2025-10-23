# promptly

A lightweight, developer-friendly library for LLM prompt management, observability/tracing, and optimization.
Currently with support for Python.

## Features

- **Prompt Templates**: Jinja2-based templating system for dynamic prompts
- **Multi-Provider Support**: OpenAI, Anthropic, Google AI (Gemini), and extensible client architecture
- **Built-in Tracing**: Comprehensive observability for prompt execution
- **Genetic Optimization**: LLM-powered genetic algorithms for automated prompt improvement
- **Async Support**: Full async/await support for high-performance applications
- **CLI Interface**: Command-line tools for prompt management
- **Type Safety**: Full type hints and Pydantic models

## Installation


```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install promptly
uv pip install promptly

# With development dependencies
uv pip install promptly[dev]

# With CLI tools
uv pip install promptly[cli]

# With UI components
uv pip install promptly[ui]
```

## Quick Start

```python
import asyncio
from promptly import PromptRunner, OpenAIClient, PromptTemplate

async def main():
    # Initialize client
    client = OpenAIClient(api_key="your-api-key")

    # Create a prompt template
    template = PromptTemplate(
        name="greeting",
        template="Hello {{ name }}, how are you today?",
        variables=["name"]
    )

    # Create runner with tracing
    runner = PromptRunner(client)

    # Execute prompt
    response = await runner.run(
        template=template,
        variables={"name": "Alice"},
        model="gpt-3.5-turbo"
    )

    print(response.content)

asyncio.run(main())
```

## Prompt Optimization

Promptly includes an advanced genetic algorithm optimizer that uses LLMs to automatically improve your prompts through iterative evaluation and mutation. The optimizer can work with test cases for accuracy-based optimization or without test cases for general quality improvement.

**Quick Example:**
```bash
# Optimize with test cases
promptly optimize \
  --base-prompt "Answer this question: {{question}}" \
  --test-cases my_tests.json \
  --population-size 10 \
  --generations 5

# Quality-based optimization (no test cases needed)
promptly optimize \
  --base-prompt "Write a {{genre}} story about {{character}}" \
  --population-size 8 \
  --generations 4
```

For complete documentation on optimization features, configuration options, and examples, see **[OPTIMIZER_README.md](OPTIMIZER_README.md)**.

## CLI Usage

```bash
# Run a simple prompt
promptly run "What is the capital of France?" --model="gpt-3.5-turbo"

# Run with tracing
promptly run "Explain quantum computing" --trace

# View traces
promptly trace
```

## Development

For developers who want to contribute to or extend promptly:

- **[Developer Quick Start.md](DEVELOPER_QUICKSTART.md)** - Complete development guide

## License

MIT License - see LICENSE file for details.
