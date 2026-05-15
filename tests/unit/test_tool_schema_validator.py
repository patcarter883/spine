"""Tests for ToolSchemaValidator middleware."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from unittest.mock import MagicMock, AsyncMock

from spine.agents.tool_schema_validator import ToolSchemaValidator
from langchain_core.messages import ToolMessage


# ── Test Schemas ────────────────────────────────────────────────────────────


class SimpleSchema(BaseModel):
    """Simple test schema with a single required string parameter."""

    filepath: str


class SchemaWithInt(BaseModel):
    """Schema with an integer parameter for type validation testing."""

    filepath: str
    count: int


# ── Test Fixture Tools ──────────────────────────────────────────────────────


def make_mock_tool(name: str, schema_class: type[BaseModel]) -> MagicMock:
    """Create a mock BaseTool with the given schema.

    Args:
        name: The tool name.
        schema_class: A Pydantic model class to use as the input schema.

    Returns:
        A MagicMock configured to behave like a BaseTool.
    """
    tool = MagicMock()
    tool.name = name

    def get_input_schema():
        return schema_class

    tool.get_input_schema = get_input_schema
    return tool


# ── Test Cases ────────────────────────────────────────────────────────────────


class TestPassesThroughWhenToolIsNone:
    """When tool is None, should pass through to handler unchanged."""

    def test_passes_through_when_tool_is_none(self):
        """Test that handler is called when tool is None."""
        validator = ToolSchemaValidator()
        handler = MagicMock(return_value="result")
        request = MagicMock()
        request.tool = None
        request.tool_call = {"name": "some_tool", "args": {}, "id": "tc_1"}

        result = validator.wrap_tool_call(request, handler)

        handler.assert_called_once_with(request)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_async_passes_through_when_tool_is_none(self):
        """Async version: handler should be called when tool is None."""
        validator = ToolSchemaValidator()
        handler = AsyncMock(return_value="result")
        request = MagicMock()
        request.tool = None
        request.tool_call = {"name": "some_tool", "args": {}, "id": "tc_1"}

        result = await validator.awrap_tool_call(request, handler)

        handler.assert_called_once_with(request)
        assert result == "result"


class TestPassesThroughValidArgs:
    """When args are valid, should pass through to handler and return result."""

    def test_passes_through_valid_args(self):
        """Test that valid args call handler and return result."""
        validator = ToolSchemaValidator()
        handler = MagicMock(return_value="success")
        tool = make_mock_tool("read_file", SimpleSchema)
        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"filepath": "/home/user/test.txt"},
            "id": "tc_1",
        }

        result = validator.wrap_tool_call(request, handler)

        handler.assert_called_once_with(request)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_valid_passes_through(self):
        """Async version: valid args should pass through to handler."""
        validator = ToolSchemaValidator()
        handler = AsyncMock(return_value="success")
        tool = make_mock_tool("read_file", SimpleSchema)
        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"filepath": "/home/user/test.txt"},
            "id": "tc_1",
        }

        result = await validator.awrap_tool_call(request, handler)

        handler.assert_called_once_with(request)
        assert result == "success"


class TestReturnsErrorOnWrongParamName:
    """When an unknown parameter is provided, should return error ToolMessage."""

    def test_returns_error_on_wrong_param_name(self):
        """Test that wrong parameter name returns error ToolMessage."""
        validator = ToolSchemaValidator()
        handler = MagicMock(return_value="should_not_be_called")
        tool = make_mock_tool("read_file", SimpleSchema)
        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"file_names": "/home/user/test.txt"},  # Wrong param name
            "id": "tc_1",
        }

        result = validator.wrap_tool_call(request, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "read_file" in result.content
        assert "filepath" in result.content  # Shows what was expected
        assert result.tool_call_id == "tc_1"


class TestReturnsErrorOnWrongType:
    """When an argument has the wrong type, should return error ToolMessage."""

    def test_returns_error_on_wrong_type(self):
        """Test that wrong type returns error ToolMessage."""
        validator = ToolSchemaValidator()
        handler = MagicMock(return_value="should_not_be_called")
        tool = make_mock_tool("read_file", SchemaWithInt)
        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"filepath": "test.txt", "count": "not_an_int"},  # Wrong type
            "id": "tc_1",
        }

        result = validator.wrap_tool_call(request, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "read_file" in result.content
        assert "count" in result.content  # Field with wrong type


class TestReboundLimitExhaustion:
    """After max_rebound failures, should pass through to handler."""

    def test_rebound_limit_exhaustion(self):
        """Test that after max_rebound failures, handler is called."""
        validator = ToolSchemaValidator(max_rebound=2)
        handler = MagicMock(return_value="passed_through")
        tool = make_mock_tool("read_file", SimpleSchema)
        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"file_names": "/home/user/test.txt"},  # Wrong param
            "id": "tc_1",
        }

        # First two calls should return error messages
        result1 = validator.wrap_tool_call(request, handler)
        result2 = validator.wrap_tool_call(request, handler)

        assert isinstance(result1, ToolMessage)
        assert isinstance(result2, ToolMessage)
        assert result1.status == "error"
        assert result2.status == "error"

        # Third call should pass through (rebound exceeded)
        result3 = validator.wrap_tool_call(request, handler)
        handler.assert_called_once_with(request)
        assert result3 == "passed_through"


class TestReboundCounterResetsOnValidCall:
    """Valid calls should reset the rebound counter."""

    def test_rebound_counter_resets_on_valid_call(self):
        """Test that a valid call resets the rebound counter."""
        validator = ToolSchemaValidator(max_rebound=2)
        handler = MagicMock(return_value="success")
        tool = make_mock_tool("read_file", SimpleSchema)

        # Bad call 1
        request = MagicMock()
        request.tool = tool
        request.tool_call = {"name": "read_file", "args": {"file_names": "x"}, "id": "tc_1"}
        result1 = validator.wrap_tool_call(request, handler)
        assert isinstance(result1, ToolMessage)

        # Bad call 2
        request.tool_call["id"] = "tc_2"
        result2 = validator.wrap_tool_call(request, handler)
        assert isinstance(result2, ToolMessage)

        # Good call - should reset counter
        request.tool_call = {"name": "read_file", "args": {"filepath": "x"}, "id": "tc_3"}
        result3 = validator.wrap_tool_call(request, handler)
        assert result3 == "success"
        assert validator._rebound_counts.get("read_file", 0) == 0

        # Another bad call should work again (counter was reset)
        request.tool_call = {"name": "read_file", "args": {"file_names": "x"}, "id": "tc_4"}
        result4 = validator.wrap_tool_call(request, handler)
        assert isinstance(result4, ToolMessage)  # Should be error, not pass-through


class TestCatchesRuntimeError:
    """When catch_runtime_errors is True, should catch exceptions."""

    def test_catches_runtime_error(self):
        """Test that runtime errors are caught and returned as error."""
        validator = ToolSchemaValidator(catch_runtime_errors=True)
        handler = MagicMock(side_effect=FileNotFoundError("file not found"))
        tool = make_mock_tool("read_file", SimpleSchema)
        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"filepath": "/nonexistent.txt"},
            "id": "tc_1",
        }

        result = validator.wrap_tool_call(request, handler)

        handler.assert_called_once_with(request)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "FileNotFoundError" in result.content or "file not found" in result.content


class TestRuntimeErrorsDisabled:
    """When catch_runtime_errors is False, exceptions should propagate."""

    def test_runtime_errors_disabled(self):
        """Test that runtime errors propagate when disabled."""
        validator = ToolSchemaValidator(catch_runtime_errors=False)
        handler = MagicMock(side_effect=ValueError("runtime error"))
        tool = make_mock_tool("read_file", SimpleSchema)
        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"filepath": "/test.txt"},
            "id": "tc_1",
        }

        with pytest.raises(ValueError, match="runtime error"):
            validator.wrap_tool_call(request, handler)


class TestAsyncInvalidReturnsError:
    """Async validation failure should return error ToolMessage."""

    @pytest.mark.asyncio
    async def test_async_invalid_returns_error(self):
        """Test async version returns error on validation failure."""
        validator = ToolSchemaValidator()
        handler = AsyncMock(return_value="should_not_be_called")
        tool = make_mock_tool("read_file", SimpleSchema)
        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"file_names": "/home/user/test.txt"},  # Wrong param
            "id": "tc_1",
        }

        result = await validator.awrap_tool_call(request, handler)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "read_file" in result.content
        assert "filepath" in result.content  # Shows what was expected


class TestCoexistsWithToolOutputTrimmer:
    """ToolSchemaValidator and ToolOutputTrimmer use different hooks and don't conflict."""

    def test_different_hooks_no_conflict(self):
        """Validator uses wrap_tool_call, Trimmer uses awrap_model_call."""
        from spine.agents.tool_schema_validator import ToolSchemaValidator
        from spine.agents.context_editing import ToolOutputTrimmer

        validator = ToolSchemaValidator()
        trimmer = ToolOutputTrimmer()

        # Both implement wrap_tool_call (validator uses it, trimmer passes through)
        assert hasattr(validator, "wrap_tool_call")
        assert hasattr(validator, "awrap_tool_call")
        assert hasattr(trimmer, "wrap_tool_call")
        assert hasattr(trimmer, "awrap_tool_call")
        # Trimmer also has awrap_model_call (its real hook)
        assert hasattr(trimmer, "awrap_model_call")


class TestConfigToggle:
    """Test that the env var toggle works correctly."""

    def test_default_is_enabled(self):
        """By default, tool schema validation is enabled."""
        validator = ToolSchemaValidator()
        assert validator.max_rebound == 3
        assert validator.catch_runtime_errors is True

    def test_config_field_exists(self):
        """SpineConfig should have the tool_schema_validation field."""
        from spine.config import SpineConfig

        config = SpineConfig()
        assert hasattr(config, "tool_schema_validation")
        assert config.tool_schema_validation is True
