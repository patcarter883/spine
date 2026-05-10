"""Agent prompt templates for SPINE.

This module provides structured, instructional prompts for each agent role,
following best practices from DeepAgents and LangChain:

1. Structured sections with XML-style tags
2. Process guidance (how to approach tasks)
3. Tool instructions (how to use available tools)
4. Output format specifications (what to return)
5. Hard limits and constraints
6. Workflow context injection

Usage:
    from spine.prompts import PromptBuilder, Role
    
    builder = PromptBuilder(role=Role.EXPLORER)
    prompt = builder.build(state, capability="analyze", tools=["filesystem"])
"""

from spine.prompts.base import (
    PromptBuilder,
    PromptConfig,
    Role,
    build_workflow_context,
)
from spine.prompts.formats import OutputFormat, get_output_format
from spine.prompts.roles import (
    CODER_PROMPT,
    CRITIC_PROMPT,
    EXPLORER_PROMPT,
    PLANNER_PROMPT,
    REVIEWER_PROMPT,
    SME_PROMPT,
    TEST_ENGINEER_PROMPT,
    get_role_prompt,
)
from spine.prompts.tools import (
    AGENT_PROVIDER_INSTRUCTIONS,
    FILESYSTEM_INSTRUCTIONS,
    get_tool_instructions,
)

__all__ = [
    # Builder
    "PromptBuilder",
    "PromptConfig",
    "Role",
    "build_workflow_context",
    # Role prompts
    "EXPLORER_PROMPT",
    "PLANNER_PROMPT",
    "CODER_PROMPT",
    "CRITIC_PROMPT",
    "SME_PROMPT",
    "REVIEWER_PROMPT",
    "TEST_ENGINEER_PROMPT",
    "get_role_prompt",
    # Tool instructions
    "FILESYSTEM_INSTRUCTIONS",
    "AGENT_PROVIDER_INSTRUCTIONS",
    "get_tool_instructions",
    # Output formats
    "OutputFormat",
    "get_output_format",
]
