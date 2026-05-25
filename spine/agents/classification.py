"""Task classification for early commitment in SPECIFY phase.

Uses a lightweight model to classify work descriptions into categories
that enable vector search filtering.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from spine.agents.helpers import resolve_model

logger = logging.getLogger(__name__)

# Fixed taxonomy of task categories for filtering vector search
TaskCategory = Literal[
    "Frontend/UI",
    "Backend/API",
    "Database",
    "Auth",
    "Testing",
    "Infrastructure",
    "Generic",
]

# Mapping from task category to symbol_type filters for vector search
CATEGORY_TO_SYMBOL_TYPES: dict[TaskCategory, list[str]] = {
    "Frontend/UI": ["component", "view", "page", "frontend", "ui"],
    "Backend/API": ["endpoint", "router", "handler", "controller", "api"],
    "Database": ["model", "schema", "migration", "repository", "dao"],
    "Auth": ["middleware", "guard", "auth", "authentication", "authorization"],
    "Testing": ["test", "spec", "fixture"],
    "Infrastructure": ["config", "script", "deploy", "ci", "infra"],
    "Generic": [],  # No filter - search all
}


class TaskClassificationResult(BaseModel):
    """Result of task classification."""

    category: TaskCategory = Field(description="The classified task category")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(default="", description="Brief reasoning for classification")


_CLASSIFICATION_SYSTEM = """\
You are a task classifier for an AI agent harness. Classify work descriptions
into one of these categories:

- Frontend/UI: User interfaces, components, views, pages, frontend logic
- Backend/API: Server endpoints, routers, handlers, controllers, API logic
- Database: Models, schemas, migrations, repositories, data access
- Auth: Authentication, authorization, security middleware, guards
- Testing: Unit tests, integration tests, test files, fixtures
- Infrastructure: Configs, scripts, deployments, CI/CD, server setup
- Generic: Non-code tasks or tasks that don't fit other categories

Return ONLY valid JSON with keys: category (string), confidence (0-1), reasoning (string).
"""


async def classify_task(description: str, config: dict | None = None) -> TaskClassificationResult:
    """Classify a work description into a task category.

    Uses a lightweight model to categorize the task, enabling vector search
    filtering by symbol_type in the recall tool.

    Args:
        description: The work description to classify.
        config: Optional LangGraph config for model resolution.

    Returns:
        TaskClassificationResult with category, confidence, and reasoning.
    """
    model = resolve_model(config, phase="classification")

    # Convert to BaseChatModel if string
    if isinstance(model, str):
        from langchain.chat_models import init_chat_model

        model = init_chat_model(model)

    prompt = f"Work description to classify:\n\n{description}"

    try:
        response = await model.ainvoke(
            [SystemMessage(content=_CLASSIFICATION_SYSTEM), HumanMessage(content=prompt)]
        )

        content = response.content if hasattr(response, "content") else str(response)

        # Extract JSON from response
        json_match = None
        if isinstance(content, str):
            import re

            json_match = re.search(r'\{\s*"category"\s*:.*"confidence"\s*:\s*[\d.]+', content)

        if json_match:
            result = json.loads(json_match.group(0))
        else:
            raise ValueError("No valid JSON in response")

        return TaskClassificationResult(
            category=result.get("category", "Generic"),  # type: ignore[arg-type]
            confidence=float(result.get("confidence", 0.5)),
            reasoning=result.get("reasoning", ""),
        )

    except Exception as e:
        logger.warning("Classification failed: %s — defaulting to Generic", e)
        return TaskClassificationResult(
            category="Generic",
            confidence=0.5,
            reasoning=f"Classification error: {e}",
        )


def get_symbol_type_filter(category: TaskCategory) -> list[str] | None:
    """Get the symbol_type filter for a task category.

    Args:
        category: The task category.

    Returns:
        List of symbol types to filter by, or None for no filter.
    """
    return CATEGORY_TO_SYMBOL_TYPES.get(category) or None