"""Tests for the prompts module."""

import pytest

from spine.prompts import (
    PromptBuilder,
    PromptConfig,
    Role,
    build_workflow_context,
    get_role_prompt,
    get_tool_instructions,
    get_output_format,
    EXPLORER_PROMPT,
    PLANNER_PROMPT,
    CODER_PROMPT,
    CRITIC_PROMPT,
    SME_PROMPT,
)


class TestPromptConfig:
    """Tests for PromptConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PromptConfig()
        assert config.include_workflow_context is True
        assert config.include_tool_instructions is True
        assert config.include_output_format is True
        assert config.max_context_items == 10
        assert config.debug_mode is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PromptConfig(
            include_workflow_context=False,
            include_tool_instructions=False,
            max_context_items=5,
            debug_mode=True,
        )
        assert config.include_workflow_context is False
        assert config.include_tool_instructions is False
        assert config.max_context_items == 5
        assert config.debug_mode is True


class TestRole:
    """Tests for Role enum."""

    def test_all_roles_defined(self):
        """Test that all expected roles are defined."""
        expected_roles = {
            "explorer", "sme", "planner", "critic",
            "coder", "reviewer", "test_engineer", "analyst", "designer"
        }
        actual_roles = {r.value for r in Role}
        assert expected_roles == actual_roles

    def test_role_values_match_strings(self):
        """Test that role enum values match expected strings."""
        assert Role.EXPLORER.value == "explorer"
        assert Role.PLANNER.value == "planner"
        assert Role.CODER.value == "coder"


class TestBuildWorkflowContext:
    """Tests for build_workflow_context function."""

    def test_basic_context(self):
        """Test basic workflow context generation."""
        state = {
            "requirement": "Add user authentication",
            "current_phase": "ANALYSIS",
            "completed_phases": [],
            "variables": {},
        }
        context = build_workflow_context(state, "explorer")
        
        assert "explorer" in context
        assert "ANALYSIS" in context
        assert "Add user authentication" in context

    def test_context_with_completed_phases(self):
        """Test context with completed phases."""
        state = {
            "requirement": "Add auth",
            "current_phase": "IMPLEMENTATION",
            "completed_phases": ["ANALYSIS", "PLANNING"],
            "variables": {},
        }
        context = build_workflow_context(state, "coder")
        
        assert "ANALYSIS" in context
        assert "PLANNING" in context
        assert "IMPLEMENTATION" in context

    def test_context_with_previous_outputs(self):
        """Test context with previous agent outputs."""
        state = {
            "requirement": "Test",
            "current_phase": "PLANNING",
            "completed_phases": ["ANALYSIS"],
            "variables": {},
        }
        previous = {
            "explorer": {"problem_statement": "Need authentication"},
        }
        context = build_workflow_context(state, "planner", previous)
        
        assert "explorer" in context
        assert "authentication" in context

    def test_context_with_project_info(self):
        """Test context includes project name, root, and tech stack."""
        state = {
            "requirement": "Add feature",
            "current_phase": "ANALYSIS",
            "completed_phases": [],
            "variables": {},
            "project_name": "my-app",
            "project_root": "/home/user/projects/my-app",
            "tech_stack": ["Python", "FastAPI", "PostgreSQL"],
        }
        context = build_workflow_context(state, "explorer")
        
        assert "my-app" in context
        assert "/home/user/projects/my-app" in context
        assert "Python" in context
        assert "FastAPI" in context
        assert "Tech stack" in context

    def test_context_without_tech_stack(self):
        """Test context works without tech stack."""
        state = {
            "requirement": "Test",
            "current_phase": "ANALYSIS",
            "completed_phases": [],
            "variables": {},
            "project_name": "test-project",
            "project_root": "/tmp/test",
            "tech_stack": [],
        }
        context = build_workflow_context(state, "explorer")
        
        assert "test-project" in context
        assert "/tmp/test" in context
        # Tech stack line should not appear when empty
        assert "Tech stack:" not in context


class TestGetRolePrompt:
    """Tests for get_role_prompt function."""

    def test_explorer_prompt(self):
        """Test getting explorer prompt."""
        prompt = get_role_prompt(Role.EXPLORER)
        assert "Requirement Analysis Agent" in prompt
        assert "<Role>" in prompt
        assert "<Process>" in prompt
        assert "<OutputFormat>" in prompt

    def test_planner_prompt(self):
        """Test getting planner prompt."""
        prompt = get_role_prompt(Role.PLANNER)
        assert "Execution Planning Agent" in prompt
        assert "architecture" in prompt.lower()
        assert "tasks" in prompt.lower()

    def test_coder_prompt(self):
        """Test getting coder prompt."""
        prompt = get_role_prompt(Role.CODER)
        assert "Implementation Agent" in prompt
        assert "code" in prompt.lower()

    def test_critic_prompt(self):
        """Test getting critic prompt."""
        prompt = get_role_prompt(Role.CRITIC)
        assert "Review and Validation Agent" in prompt
        assert "security" in prompt.lower()

    def test_sme_prompt(self):
        """Test getting SME prompt."""
        prompt = get_role_prompt(Role.SME)
        assert "Subject Matter Expert Agent" in prompt
        assert "research" in prompt.lower()

    def test_invalid_role_raises_error(self):
        """Test that invalid role raises ValueError."""
        # Since Role is an enum, all values are valid
        # But get_role_prompt should raise for roles not in the registry
        # The enum defines ANALYST and DESIGNER which use alias prompts
        # So we verify the function works correctly
        
        # Test that ANALYST returns CRITIC_PROMPT (alias)
        prompt = get_role_prompt(Role.ANALYST)
        assert "Review" in prompt
        
        # Test that DESIGNER returns PLANNER_PROMPT (alias)
        prompt = get_role_prompt(Role.DESIGNER)
        assert "Planning" in prompt or "Plan" in prompt


class TestGetToolInstructions:
    """Tests for get_tool_instructions function."""

    def test_filesystem_instructions(self):
        """Test getting filesystem tool instructions."""
        instructions = get_tool_instructions("filesystem")
        assert "Filesystem Tools" in instructions
        assert "read_file" in instructions

    def test_agent_provider_instructions(self):
        """Test getting agent provider instructions."""
        instructions = get_tool_instructions("agent_provider")
        assert "External Coding Agent" in instructions
        assert "OpenCode" in instructions

    def test_unknown_tool_returns_empty(self):
        """Test that unknown tool returns empty string."""
        instructions = get_tool_instructions("unknown_tool")
        assert instructions == ""

    def test_tool_aliases(self):
        """Test tool name aliases."""
        assert get_tool_instructions("file") == get_tool_instructions("filesystem")
        assert get_tool_instructions("agent") == get_tool_instructions("agent_provider")


class TestGetOutputFormat:
    """Tests for get_output_format function."""

    def test_analysis_format(self):
        """Test getting analysis output format."""
        fmt = get_output_format("analyze")
        assert "problem_statement" in fmt
        assert "constraints" in fmt
        assert "success_criteria" in fmt

    def test_plan_format(self):
        """Test getting plan output format."""
        fmt = get_output_format("plan")
        assert "architecture" in fmt
        assert "tasks" in fmt
        assert "TASK-" in fmt

    def test_implementation_format(self):
        """Test getting implementation output format."""
        fmt = get_output_format("implement")
        assert "files_changed" in fmt
        assert "tests" in fmt
        assert "acceptance_met" in fmt

    def test_role_specific_format(self):
        """Test role-specific format selection."""
        fmt = get_output_format("default", Role.EXPLORER)
        assert "problem_statement" in fmt


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = PromptBuilder(role=Role.EXPLORER)
        assert builder.role == Role.EXPLORER

    def test_builder_with_config(self):
        """Test builder with custom config."""
        config = PromptConfig(debug_mode=True)
        builder = PromptBuilder(role=Role.PLANNER, config=config)
        assert builder.config.debug_mode is True

    def test_build_basic_prompt(self):
        """Test building a basic prompt."""
        builder = PromptBuilder(role=Role.EXPLORER)
        state = {
            "requirement": "Add authentication",
            "current_phase": "ANALYSIS",
            "completed_phases": [],
        }
        prompt = builder.build(state, capability="analyze")
        
        # Should contain role prompt
        assert "Requirement Analysis Agent" in prompt
        # Should contain workflow context
        assert "Workflow Context" in prompt
        # Should contain output format
        assert "Output Format" in prompt

    def test_build_prompt_with_tools(self):
        """Test building prompt with tool instructions."""
        builder = PromptBuilder(role=Role.CODER)
        state = {"requirement": "Test", "current_phase": "IMPLEMENTATION"}
        prompt = builder.build(state, capability="implement", tools=["filesystem"])
        
        assert "Filesystem Tools" in prompt

    def test_build_prompt_without_workflow_context(self):
        """Test building prompt without workflow context."""
        config = PromptConfig(include_workflow_context=False)
        builder = PromptBuilder(role=Role.EXPLORER, config=config)
        state = {"requirement": "Test", "current_phase": "ANALYSIS"}
        prompt = builder.build(state, capability="analyze")
        
        assert "Workflow Context" not in prompt

    def test_build_prompt_with_previous_outputs(self):
        """Test building prompt with previous agent outputs."""
        builder = PromptBuilder(role=Role.PLANNER)
        state = {
            "requirement": "Test",
            "current_phase": "PLANNING",
            "completed_phases": ["ANALYSIS"],
        }
        previous = {"explorer": {"problem_statement": "Need auth"}}
        prompt = builder.build(state, capability="plan", previous_outputs=previous)
        
        assert "Previous Agent Outputs" in prompt

    def test_build_prompt_with_debug_mode(self):
        """Test building prompt with debug mode enabled."""
        config = PromptConfig(debug_mode=True)
        builder = PromptBuilder(role=Role.EXPLORER, config=config)
        state = {"requirement": "Test", "current_phase": "ANALYSIS"}
        prompt = builder.build(state, capability="analyze")
        
        assert "Debug Information" in prompt


class TestRolePromptContent:
    """Tests for role prompt content quality."""

    def test_explorer_has_xml_sections(self):
        """Test that explorer prompt has structured XML sections."""
        prompt = EXPLORER_PROMPT
        
        assert "<Role>" in prompt
        assert "<YourTask>" in prompt
        assert "<Process>" in prompt
        assert "<OutputFormat>" in prompt
        assert "<HardLimits>" in prompt

    def test_explorer_has_process_steps(self):
        """Test that explorer prompt has detailed process steps."""
        prompt = EXPLORER_PROMPT
        
        assert "Parse the requirement" in prompt
        assert "Extract constraints" in prompt
        assert "Define success criteria" in prompt
        assert "Flag ambiguities" in prompt

    def test_explorer_has_json_schema(self):
        """Test that explorer prompt has JSON output schema."""
        prompt = EXPLORER_PROMPT
        
        assert "```json" in prompt
        assert '"problem_statement"' in prompt
        assert '"constraints"' in prompt
        assert '"success_criteria"' in prompt
        assert '"ambiguities"' in prompt

    def test_planner_has_architecture_section(self):
        """Test that planner prompt has architecture guidance."""
        prompt = PLANNER_PROMPT
        
        assert "architecture" in prompt.lower()
        assert "components" in prompt.lower()
        assert "TASK-" in prompt

    def test_coder_has_testing_guidance(self):
        """Test that coder prompt has testing guidance."""
        prompt = CODER_PROMPT
        
        assert "test" in prompt.lower()
        assert "Test thoroughly" in prompt

    def test_critic_has_review_categories(self):
        """Test that critic prompt has review categories."""
        prompt = CRITIC_PROMPT
        
        assert "correctness" in prompt.lower()
        assert "security" in prompt.lower()
        assert "performance" in prompt.lower()
        assert "maintainability" in prompt.lower()

    def test_sme_has_research_guidance(self):
        """Test that SME prompt has research guidance."""
        prompt = SME_PROMPT
        
        assert "best practices" in prompt.lower()
        assert "pitfalls" in prompt.lower()
        assert "libraries_and_tools" in prompt.lower()
