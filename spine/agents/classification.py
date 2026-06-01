"""Task classification for early commitment in SPECIFY phase.

Uses a lightweight model to classify work descriptions into categories
that downstream prompts use as metadata. The category does NOT filter
vector search — the AST extractor only emits function/class/method/
interface symbol types, so any category→type mapping would zero-result
on every non-Generic classification.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from spine.agents.helpers import resolve_model
from spine.agents.prompt_format import Tag, hostage_layout, xml_block, xml_blocks

logger = logging.getLogger(__name__)

TaskCategory = Literal[
    "Frontend/UI",
    "Backend/API",
    "Database",
    "Auth",
    "Testing",
    "Infrastructure",
    "Generic",
]


class TaskClassificationResult(BaseModel):
    """Result of task classification."""

    category: TaskCategory = Field(description="The classified task category")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(default="", description="Brief reasoning for classification")


_CLASSIFICATION_SYSTEM = (
    xml_block(
        Tag.ROLE,
        "You are a task classifier for an AI agent harness. Classify work "
        "descriptions into exactly one of the categories below.",
    )
    + "\n\n"
    + xml_block(
        Tag.CONSTRAINTS,
        "- Frontend/UI: User interfaces, components, views, pages, frontend logic\n"
        "- Backend/API: Server endpoints, routers, handlers, controllers, API logic\n"
        "- Database: Models, schemas, migrations, repositories, data access\n"
        "- Auth: Authentication, authorization, security middleware, guards\n"
        "- Testing: Unit tests, integration tests, test files, fixtures\n"
        "- Infrastructure: Configs, scripts, deployments, CI/CD, server setup\n"
        "- Generic: Non-code tasks or tasks that don't fit other categories",
    )
    + "\n\n"
    + xml_block(
        Tag.OUTPUT_SCHEMA,
        "Return ONLY valid JSON with keys: category (string), "
        "confidence (0-1), reasoning (string).",
    )
)


async def classify_task(description: str, config: dict | None = None) -> TaskClassificationResult:
    """Classify a work description into a task category.

    The category is recorded on workflow state as metadata for
    downstream prompts. It is NOT used as a recall filter.

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

    prompt = hostage_layout(
        xml_blocks((Tag.OBJECTIVE, description)),
        "Classify the work description above and return the JSON.",
    )

    try:
        response = await model.ainvoke(
            [SystemMessage(content=_CLASSIFICATION_SYSTEM), HumanMessage(content=prompt)]
        )

        content = response.content if hasattr(response, "content") else str(response)

        if isinstance(content, str):
            # Try direct JSON parse first (handles multi-line JSON with reasoning)
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Fall back to regex extraction for models that wrap JSON in markdown
                import re

                match = re.search(
                    r'\{\s*"category"\s*:.*?"confidence"\s*:\s*[\d.]+.*?\}',
                    content,
                    re.DOTALL,
                )
                if match:
                    try:
                        result = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        raise ValueError(f"No valid JSON in response: {content[:200]}")
                else:
                    raise ValueError(f"No valid JSON in response: {content[:200]}")
        else:
            raise ValueError(f"Unexpected response type: {type(content)}")

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


