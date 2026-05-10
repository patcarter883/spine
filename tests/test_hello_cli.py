"""Tests for the 'spine hello' CLI command."""

import sys
from pathlib import Path

import click
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.cli import cli


class TestHelloCommand:
    """Tests for the 'spine hello' CLI command."""

    def test_hello_command_exists(self):
        """Test that the hello command is registered."""
        commands = cli.commands
        assert "hello" in commands, "hello command should be registered"

    def test_hello_command_prints_greeting(self):
        """Test that hello command prints a greeting message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["hello"])

        assert result.exit_code == 0
        assert "Hello" in result.output or "hello" in result.output.lower()

    def test_hello_command_includes_version(self):
        """Test that the greeting includes the version string."""
        runner = CliRunner()
        result = runner.invoke(cli, ["hello"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output, f"Expected version in output, got: {result.output}"

    def test_hello_command_help(self):
        """Test that --help works for the hello command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["hello", "--help"])

        assert result.exit_code == 0
        assert "hello" in result.output.lower() or "greeting" in result.output.lower()
