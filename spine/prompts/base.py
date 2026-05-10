"""Base prompt utilities and PromptBuilder.

This module provides the core prompting infrastructure used by all SPINE agents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Role(Enum):
    """Standard agent roles in SPINE."""
    EXPLORER = "explorer"
    SME = "sme"
    PLANNER = "planner"
    CRITIC = "critic"
    CODER = "coder"
    REVIEWER = "reviewer"
    TEST_ENGINEER = "test_engineer"
    ANALYST = "analyst"
    DESIGNER = "designer"


@dataclass
class PromptConfig:
    """Configuration for prompt building.
    
    Attributes:
        include_workflow_context: Whether to inject workflow state context
        include_tool_instructions: Whether to include tool usage instructions
        include_output_format: Whether to include output format specification
        max_context_items: Maximum number of context items to include
        debug_mode: If True, include debug information in prompts
    """
    include_workflow_context: bool = True
    include_tool_instructions: bool = True
    include_output_format: bool = True
    max_context_items: int = 10
    debug_mode: bool = False


def build_workflow_context(
    state: dict[str, Any],
    role: str,
    previous_outputs: Optional[dict[str, Any]] = None,
) -> str:
    """Build context about the current workflow state.
    
    Creates a section that informs the agent about:
    - Their role in the workflow
    - What project they're working on
    - What phase they're in
    - What previous agents have produced
    - What's expected of them
    
    Args:
        state: Current SPINE state dictionary
        role: The agent's role name
        previous_outputs: Optional dict of previous agent outputs by role
        
    Returns:
        Formatted workflow context section
    """
    completed_phases = state.get("completed_phases", [])
    current_phase = state.get("current_phase", "initialization")
    requirement = state.get("requirement", "No requirement specified")
    variables = state.get("variables", {})
    
    # Project context
    project_name = state.get("project_name", "unknown-project")
    project_root = state.get("project_root", ".")
    tech_stack = state.get("tech_stack", [])
    
    parts = [
        "## Workflow Context",
        "",
        f"**Your role**: {role}",
        f"**Project**: {project_name}",
        f"**Project root**: {project_root}",
        f"**Current phase**: {current_phase}",
        f"**Completed phases**: {', '.join(completed_phases) if completed_phases else 'None yet'}",
    ]
    
    if tech_stack:
        parts.append(f"**Tech stack**: {', '.join(tech_stack)}")
    
    parts.extend([
        "",
        "### Requirement",
        requirement,
    ])
    
    # Include relevant variables
    if variables:
        relevant_vars = {
            k: v for k, v in variables.items()
            if k in {"workdir", "target_files", "constraints", "tech_stack"}
        }
        if relevant_vars:
            parts.extend([
                "",
                "### Variables",
                "```json",
                json.dumps(relevant_vars, indent=2, default=str),
                "```",
            ])
    
    # Include previous agent outputs
    if previous_outputs:
        parts.extend([
            "",
            "### Previous Agent Outputs",
        ])
        for agent_role, output in previous_outputs.items():
            if output:
                parts.extend([
                    f"**From {agent_role}**:",
                    "```json",
                    json.dumps(output, indent=2, default=str)[:1000],  # Truncate long outputs
                    "```",
                    "",
                ])
    
    return "\n".join(parts)


def format_state_summary(state: dict[str, Any], max_items: int = 5) -> str:
    """Create a concise summary of state for prompt inclusion.
    
    Args:
        state: SPINE state dictionary
        max_items: Maximum items to show in lists
        
    Returns:
        Formatted state summary
    """
    summary_parts = []
    
    if requirement := state.get("requirement"):
        summary_parts.append(f"**Requirement**: {requirement[:200]}")
    
    if completed := state.get("completed_phases"):
        summary_parts.append(f"**Completed**: {', '.join(completed[:max_items])}")
    
    if errors := state.get("errors", []):
        error_list = errors[:max_items]
        summary_parts.append(f"**Errors**: {len(errors)} total, showing {error_list}")
    
    return "\n".join(summary_parts) if summary_parts else "No state summary available."


class PromptBuilder:
    """Compose agent prompts from multiple sources.
    
    This class assembles prompts by combining:
    1. Role-specific instructions (from roles/ module)
    2. Tool instructions (from tools/ module)
    3. Workflow context (from current state)
    4. Output format specifications (from formats/ module)
    5. Task-specific context (capability and kwargs)
    
    Example:
        builder = PromptBuilder(role=Role.EXPLORER)
        prompt = builder.build(
            state={"requirement": "Add auth", "current_phase": "ANALYSIS"},
            capability="analyze",
            tools=["filesystem", "agent_provider"],
        )
    """
    
    def __init__(
        self,
        role: Role | str,
        config: Optional[PromptConfig] = None,
    ):
        """Initialize the prompt builder.
        
        Args:
            role: The agent's role (enum or string)
            config: Optional prompt configuration
        """
        self.role = Role(role) if isinstance(role, str) else role
        self.config = config or PromptConfig()
        self._role_prompts: dict[Role, str] = {}
        self._tool_instructions: dict[str, str] = {}
    
    def build(
        self,
        state: dict[str, Any],
        capability: str,
        tools: Optional[list[str]] = None,
        previous_outputs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Build complete prompt for agent execution.
        
        Args:
            state: Current SPINE state dictionary
            capability: The capability being executed
            tools: List of available tool names
            previous_outputs: Outputs from previous agents in workflow
            **kwargs: Additional context for the task
            
        Returns:
            Complete prompt string ready for LLM
        """
        parts = []
        
        # 1. Role-specific instructions
        role_prompt = self._get_role_prompt()
        if role_prompt:
            parts.append(role_prompt)
        
        # 2. Tool instructions (if tools available and enabled)
        if self.config.include_tool_instructions and tools:
            tool_instructions = self._build_tool_instructions(tools)
            if tool_instructions:
                parts.append(tool_instructions)
        
        # 3. Workflow context (if enabled)
        if self.config.include_workflow_context:
            parts.append(build_workflow_context(state, self.role.value, previous_outputs))
        
        # 4. Output format (if enabled)
        if self.config.include_output_format:
            output_format = self._get_output_format(capability)
            if output_format:
                parts.append(output_format)
        
        # 5. Task-specific context
        task_context = self._build_task_context(state, capability, **kwargs)
        if task_context:
            parts.append(task_context)
        
        # 6. Debug information (if enabled)
        if self.config.debug_mode:
            parts.append(self._build_debug_section(state, capability, tools, kwargs))
        
        return "\n\n".join(parts)
    
    def _get_role_prompt(self) -> str:
        """Get the role-specific prompt for this agent."""
        # Lazy import to avoid circular dependencies
        from spine.prompts.roles import get_role_prompt as _get_role_prompt
        return _get_role_prompt(self.role)
    
    def _build_tool_instructions(self, tools: list[str]) -> str:
        """Build combined tool instructions section."""
        from spine.prompts.tools import get_tool_instructions
        
        instructions = []
        for tool in tools:
            if tool_instruction := get_tool_instructions(tool):
                instructions.append(tool_instruction)
        
        if instructions:
            return "## Available Tools\n\n" + "\n\n".join(instructions)
        return ""
    
    def _get_output_format(self, capability: str) -> str:
        """Get output format specification for the capability."""
        from spine.prompts.formats import get_output_format as _get_output_format
        return _get_output_format(capability, self.role)
    
    def _build_task_context(
        self,
        state: dict[str, Any],
        capability: str,
        **kwargs,
    ) -> str:
        """Build task-specific context section."""
        parts = []
        
        # Add capability-specific instructions
        capability_instructions = self._get_capability_instructions(capability)
        if capability_instructions:
            parts.append(capability_instructions)
        
        # Add any additional context from kwargs
        if extra_context := kwargs.get("extra_context"):
            parts.extend([
                "## Additional Context",
                "",
                str(extra_context),
            ])
        
        # Add file scope if provided
        if files := kwargs.get("files"):
            parts.extend([
                "## File Scope",
                "",
                "Work within these files/directories:",
                "```",
                *files,
                "```",
            ])
        
        return "\n\n".join(parts) if parts else ""
    
    def _get_capability_instructions(self, capability: str) -> str:
        """Get instructions specific to a capability."""
        # Capability-specific instructions based on role
        capability_map = {
            "analyze": self._analyze_instructions(),
            "plan": self._plan_instructions(),
            "implement": self._implement_instructions(),
            "review": self._review_instructions(),
            "test": self._test_instructions(),
            "research": self._research_instructions(),
        }
        return capability_map.get(capability, "")
    
    def _analyze_instructions(self) -> str:
        return """## Task: Analyze

Focus on understanding the problem deeply. Do NOT jump to solutions.
Identify what's known, what's unknown, and what needs clarification."""
    
    def _plan_instructions(self) -> str:
        return """## Task: Plan

Create a detailed, step-by-step plan that can be executed by implementation agents.
Each step should be atomic and testable."""
    
    def _implement_instructions(self) -> str:
        return """## Task: Implement

Execute the plan step by step. Make small, testable changes.
Verify each change before moving to the next step."""
    
    def _review_instructions(self) -> str:
        return """## Task: Review

Evaluate the work for correctness, security, and completeness.
Provide actionable feedback, not just problems."""
    
    def _test_instructions(self) -> str:
        return """## Task: Test

Create comprehensive tests that verify the implementation.
Include edge cases and error conditions."""
    
    def _research_instructions(self) -> str:
        return """## Task: Research

Gather relevant information to inform the implementation.
Focus on best practices, existing patterns, and potential pitfalls."""
    
    def _build_debug_section(
        self,
        state: dict[str, Any],
        capability: str,
        tools: Optional[list[str]],
        kwargs: dict[str, Any],
    ) -> str:
        """Build debug information section."""
        return f"""## Debug Information

**Role**: {self.role.value}
**Capability**: {capability}
**Tools**: {tools or 'None'}
**State keys**: {list(state.keys())}
**Kwargs**: {list(kwargs.keys())}
"""
