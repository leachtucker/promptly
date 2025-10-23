"""
Tests for CLI interface
"""

from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from promptly.cli.main import main, run, trace
from promptly.core.clients import LLMResponse
from promptly.core.tracer import UsageData


class TestCLI:
    """Test CLI functionality"""

    def test_cli_main_help(self):
        """Test CLI main help command"""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert (
            "promptly - A lightweight library for LLM prompt management and optimization"
            in result.output
        )

    def test_cli_version(self):
        """Test CLI version command"""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    @patch("promptly.cli.main.PromptRunner")
    @patch("promptly.core.clients.OpenAIClient")
    def test_cli_run_simple_prompt(self, mock_openai_class, mock_runner_class):
        """Test CLI run command with simple prompt"""
        # Mock the runner
        mock_runner = AsyncMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.run.return_value = LLMResponse(
            content="The capital of France is Paris.",
            model="gpt-3.5-turbo",
            usage=UsageData(total_tokens=10),
        )

        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "What is the capital of France?",
                "--model",
                "gpt-3.5-turbo",
                "--provider",
                "openai",
            ],
        )

        assert result.exit_code == 0
        assert "The capital of France is Paris." in result.output

    @patch("promptly.cli.main.PromptRunner")
    @patch("promptly.core.clients.OpenAIClient")
    def test_cli_run_with_trace(self, mock_openai_class, mock_runner_class):
        """Test CLI run command with tracing enabled"""
        # Mock the runner
        mock_runner = AsyncMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.run.return_value = LLMResponse(
            content="Test response",
            model="gpt-3.5-turbo",
            usage=UsageData(total_tokens=10),
            metadata={"trace_id": "test-trace-123"},
        )

        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(run, ["Test prompt", "--trace"])

        assert result.exit_code == 0
        assert "Test response" in result.output
        assert "test-trace-123" in result.output

    def test_cli_run_missing_prompt_and_template(self):
        """Test CLI run command with missing prompt and template"""
        runner = CliRunner()
        result = runner.invoke(run, [])

        assert result.exit_code != 0
        assert "Either --template or prompt argument is required" in result.output

    @patch("promptly.cli.main.PromptRunner")
    @patch("promptly.core.clients.OpenAIClient")
    def test_cli_run_with_anthropic(self, mock_openai_class, mock_runner_class):
        """Test CLI run command with Anthropic provider"""
        # Mock the runner
        mock_runner = AsyncMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.run.return_value = LLMResponse(
            content="Anthropic response",
            model="claude-3-sonnet-20240229",
            usage=UsageData(total_tokens=10),
        )

        # Mock OpenAI client (shouldn't be used)
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            run,
            [
                "Test prompt",
                "--provider",
                "anthropic",
                "--model",
                "claude-3-sonnet-20240229",
            ],
        )

        assert result.exit_code == 0
        assert "Anthropic response" in result.output

    def test_cli_trace_help(self):
        """Test CLI trace command help"""
        runner = CliRunner()
        result = runner.invoke(trace, ["--help"])

        assert result.exit_code == 0
        assert "View trace information" in result.output

    def test_cli_trace_list(self):
        """Test CLI trace command without specific trace ID"""
        runner = CliRunner()
        result = runner.invoke(trace, [])

        assert result.exit_code == 0

    def test_cli_trace_specific(self):
        """Test CLI trace command with specific trace ID"""
        runner = CliRunner()
        result = runner.invoke(trace, ["--trace-id", "test-123"])

        assert result.exit_code == 0
        assert "test-123" in result.output

    @patch("promptly.cli.main.PromptRunner")
    @patch("promptly.core.clients.OpenAIClient")
    def test_cli_run_error_handling(self, mock_openai_class, mock_runner_class):
        """Test CLI run command error handling"""
        # Mock the runner to raise an error
        mock_runner = AsyncMock()
        mock_runner_class.return_value = mock_runner
        mock_runner.run.side_effect = Exception("API Error")

        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(run, ["Test prompt"])

        assert result.exit_code == 0  # CLI should handle errors gracefully
        assert "Error: API Error" in result.output
