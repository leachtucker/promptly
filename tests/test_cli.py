"""
Tests for CLI interface
"""

from unittest.mock import AsyncMock

from click.testing import CliRunner

from promptly.core.clients import LLMResponse
from promptly.core.tracer import UsageData


class TestCLI:
    """Test CLI functionality"""

    def test_cli_main_help(self, cli_module):
        """Test CLI main help command"""
        runner = CliRunner()
        result = runner.invoke(cli_module["main"], ["--help"])

        assert result.exit_code == 0
        assert (
            "promptly - A lightweight library for LLM prompt management and optimization"
            in result.output
        )

    def test_cli_version(self, cli_module):
        """Test CLI version command"""
        runner = CliRunner()
        result = runner.invoke(cli_module["main"], ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_run_simple_prompt(self, cli_module):
        """Test CLI run command with simple prompt"""
        # Mock the runner
        mock_runner = AsyncMock()
        cli_module["mock_runner"].return_value = mock_runner
        mock_runner.run.return_value = LLMResponse(
            content="The capital of France is Paris.",
            model="gpt-3.5-turbo",
            usage=UsageData(total_tokens=10),
        )

        # Mock OpenAI client - return a mock instance when called
        mock_client_instance = AsyncMock()
        cli_module["mock_openai"].return_value = mock_client_instance

        runner = CliRunner()
        result = runner.invoke(
            cli_module["run"],
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

    def test_cli_run_with_trace(self, cli_module):
        """Test CLI run command with tracing enabled"""
        # Mock the runner
        mock_runner = AsyncMock()
        cli_module["mock_runner"].return_value = mock_runner
        mock_runner.run.return_value = LLMResponse(
            content="Test response",
            model="gpt-3.5-turbo",
            usage=UsageData(total_tokens=10),
            metadata={"trace_id": "test-trace-123"},
        )

        # Mock OpenAI client - return a mock instance when called
        mock_client_instance = AsyncMock()
        cli_module["mock_openai"].return_value = mock_client_instance

        runner = CliRunner()
        result = runner.invoke(cli_module["run"], ["Test prompt", "--trace"])

        assert result.exit_code == 0
        assert "Test response" in result.output
        assert "test-trace-123" in result.output

    def test_cli_run_missing_prompt_and_template(self, cli_module):
        """Test CLI run command with missing prompt and template"""
        runner = CliRunner()
        result = runner.invoke(cli_module["run"], [])

        assert result.exit_code != 0
        assert "Either --template or prompt argument is required" in result.output

    def test_cli_run_with_anthropic(self, cli_module):
        """Test CLI run command with Anthropic provider"""
        # Mock the runner
        mock_runner = AsyncMock()
        cli_module["mock_runner"].return_value = mock_runner
        mock_runner.run.return_value = LLMResponse(
            content="Anthropic response",
            model="claude-3-sonnet-20240229",
            usage=UsageData(total_tokens=10),
        )

        # Mock Anthropic client - return a mock instance when called
        mock_client_instance = AsyncMock()
        cli_module["mock_anthropic"].return_value = mock_client_instance

        runner = CliRunner()
        result = runner.invoke(
            cli_module["run"],
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

    def test_cli_trace_help(self, cli_module):
        """Test CLI trace command help"""
        runner = CliRunner()
        result = runner.invoke(cli_module["trace"], ["--help"])

        assert result.exit_code == 0
        assert "View trace information" in result.output

    def test_cli_trace_list(self, cli_module):
        """Test CLI trace command without specific trace ID"""
        runner = CliRunner()
        result = runner.invoke(cli_module["trace"], [])

        assert result.exit_code == 0

    def test_cli_trace_specific(self, cli_module):
        """Test CLI trace command with specific trace ID"""
        runner = CliRunner()
        result = runner.invoke(cli_module["trace"], ["--trace-id", "test-123"])

        assert result.exit_code == 0
        assert "test-123" in result.output

    def test_cli_run_error_handling(self, cli_module):
        """Test CLI run command error handling"""
        # Mock the runner to raise an error
        mock_runner = AsyncMock()
        cli_module["mock_runner"].return_value = mock_runner
        mock_runner.run.side_effect = Exception("API Error")

        # Mock OpenAI client - return a mock instance when called
        mock_client_instance = AsyncMock()
        cli_module["mock_openai"].return_value = mock_client_instance

        runner = CliRunner()
        result = runner.invoke(cli_module["run"], ["Test prompt"])

        assert result.exit_code == 0  # CLI should handle errors gracefully
        assert "Error: API Error" in result.output
