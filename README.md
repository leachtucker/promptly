# promptly

A lightweight, developer-friendly library for LLM prompt management, observability/tracing, and optimization.

## Features

- **Prompt Templates**: Jinja2-based templating system for dynamic prompts
- **Multi-Provider Support**: OpenAI, Anthropic, and extensible client architecture
- **Built-in Tracing**: Comprehensive observability for prompt execution
- **Async Support**: Full async/await support for high-performance applications
- **CLI Interface**: Command-line tools for prompt management
- **Type Safety**: Full type hints and Pydantic models

## Installation

```bash
pip install promptly
```

For development dependencies:
```bash
pip install promptly[dev]
```

For CLI tools:
```bash
pip install promptly[cli]
```

For UI components:
```bash
pip install promptly[ui]
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
