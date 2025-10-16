# promptly

A lightweight, developer-friendly library for LLM prompt management, observability/tracing, and optimization.
Currently with support for Python.

## Features

- **Prompt Templates**: Jinja2-based templating system for dynamic prompts
- **Multi-Provider Support**: OpenAI, Anthropic, Google AI (Gemini), and extensible client architecture
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

### Using Google AI (Gemini)

```python
import asyncio
from promptly import GoogleAIClient

async def main():
    # Initialize Google AI client (requires GOOGLE_API_KEY environment variable)
    client = GoogleAIClient()
    
    # Simple generation
    response = await client.generate(
        prompt="Explain quantum computing in simple terms",
        model="gemini-1.5-flash"
    )
    
    print(response.content)

asyncio.run(main())
```

For more examples, see `examples/google_ai_example.py`.

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
