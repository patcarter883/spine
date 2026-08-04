"""Task classification for early commitment in SPECIFY phase.

Uses a lightweight model to classify work descriptions into categories
that downstream prompts use as metadata. The category does NOT filter
vector search — the AST extractor only emits function/class/method/
interface symbol types, so any category→type mapping would zero-result
on every non-Generic classification.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from spine.agents.helpers import bind_structured_output, resolve_chat_model
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
    """Result of task classification.

    Reasoning-first field order (and matching JSON key order in the
    prompt's output schema) so the model writes its rationale before
    committing the ``category`` verdict. Consistent with the decision
    schemas ``CriticReview`` and ``ResearchManagerDecision``.
    """

    reasoning: str = Field(default="", description="Brief reasoning for classification")
    category: TaskCategory = Field(description="The classified task category")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


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
        "Return ONLY valid JSON with keys, in this order: reasoning "
        "(string), category (string), confidence (0-1). Write reasoning "
        "first, then commit to the category.",
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
    model = resolve_chat_model(config, phase="classification")

    prompt = hostage_layout(
        xml_blocks((Tag.OBJECTIVE, description)),
        "Classify the work description above and return the JSON.",
    )

    try:
        # Schema-bound, not prose-JSON: a verbose reasoning model (ZAYA)
        # won't reliably commit clean JSON from a prose instruction — its
        # deliberation spills into content and the scrape finds no object
        # (trace 019f5a37: two ~90s classification calls, zero parses,
        # silent Generic fallback). The grammar constrains only the FINAL
        # answer; RSA rollouts stay free-form for exploration.
        structured = bind_structured_output(model, TaskClassificationResult)
        result = await structured.ainvoke(
            [SystemMessage(content=_CLASSIFICATION_SYSTEM), HumanMessage(content=prompt)]
        )
        if not isinstance(result, TaskClassificationResult):
            raise ValueError(f"Unexpected structured result type: {type(result)}")
        return result

    except Exception as e:
        logger.warning("Classification failed: %s — defaulting to Generic", e)
        return TaskClassificationResult(
            category="Generic",
            confidence=0.5,
            reasoning=f"Classification error: {e}",
        )


