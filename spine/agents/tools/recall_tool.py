"""Native RAG Recall Tool for SPINE SPECIFY phase.

Retrieves relevant code chunks from the vector store based on semantic similarity,
with optional filtering by task classification.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from spine.agents.classification import get_symbol_type_filter
from spine.persistence.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RecallInput(BaseModel):
    """Input schema for the RecallTool."""

    query: str = Field(
        description="Natural language search query describing what you need to find"
    )
    k: int = Field(
        default=0,
        ge=1,
        le=50,
        description="Maximum number of chunks to return (0 = use config default, max 50)",
    )
    task_category: Optional[str] = Field(
        default=None,
        description="Optional task category for filtering (Frontend/UI, Backend/API, etc.)",
    )
    max_tokens: Optional[int] = Field(
        default=50000,
        description="Maximum total tokens for retrieved code (default 50k to stay under 100k context window)",
    )
    summaries_only: bool = Field(
        default=False,
        description="If True, return only natural language summaries (no raw code). "
        "Use this for high-level architectural discovery — much more token-efficient.",
    )


class RecallTool(BaseTool):
    """Retrieve relevant code chunks from the vector store.

    This tool performs semantic search against the vector store to find
    code chunks relevant to the work description. It uses the query
    embedding to find similar chunks and can optionally filter by
    task category to improve relevance.

    The retrieved chunks are returned as raw_code for the agent to
    analyze. Token budget control is applied to stay within context
    limits.
    """

    name: str = "recall"
    description: str = (
        "Semantic search against the vector knowledge base. "
        "Use natural language queries to find relevant functions, classes, "
        "and their LLM-generated summaries. "
        "Set summaries_only=True for high-level architectural discovery "
        "(returns file paths + symbol names + summaries only — no raw code). "
        "Set summaries_only=False when you need the actual source code inline."
    )
    args_schema: type[RecallInput] = RecallInput

    db_path: str = ".spine/spine.db"
    embedding_provider: str = "openai:text-embedding-3-large"

    def _run(
        self,
        query: str,
        k: int = 0,
        task_category: Optional[str] = None,
        max_tokens: Optional[int] = 50000,
        summaries_only: bool = False,
    ) -> str:
        """Execute the recall tool synchronously."""
        return self._recall_sync(query, k, task_category, max_tokens, summaries_only)

    async def _arun(
        self,
        query: str,
        k: int = 0,
        task_category: Optional[str] = None,
        max_tokens: Optional[int] = 50000,
        summaries_only: bool = False,
    ) -> str:
        """Execute the recall tool asynchronously."""
        return await self._recall_async(query, k, task_category, max_tokens, summaries_only)

    def _recall_sync(
        self,
        query: str,
        k: int,
        task_category: Optional[str],
        max_tokens: Optional[int],
        summaries_only: bool = False,
    ) -> str:
        """Synchronous recall - wraps async implementation."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run, self._recall_async(query, k, task_category, max_tokens, summaries_only)
                )
                return future.result(timeout=60)
        else:
            return asyncio.run(self._recall_async(query, k, task_category, max_tokens, summaries_only))

    async def _recall_async(
        self,
        query: str,
        k: int,
        task_category: Optional[str],
        max_tokens: Optional[int],
        summaries_only: bool = False,
    ) -> str:
        """Async recall implementation."""
        from spine.config import SpineConfig

        # Resolve k — 0 means "use config default"
        if k == 0:
            k = SpineConfig.load().recall_k

        # Get embedding for query
        embedding = await self._embed_query(query)

        # Get symbol type filter from category
        symbol_types = None
        if task_category:
            from spine.agents.classification import TaskCategory

            cat: TaskCategory = task_category  # type: ignore[assignment]
            symbol_types = get_symbol_type_filter(cat)

        # Search vector store
        store = VectorStore(self.db_path)
        store.ensure_schema()

        results = store.search_similar(embedding, k=k, filter_by_types=symbol_types)

        # Apply token budget
        results = self._apply_token_budget(results, max_tokens)

        # Strip raw_code when summaries_only
        if summaries_only:
            for r in results:
                r.pop("raw_code", None)

        return json.dumps(
            {
                "query": query,
                "category_filter": task_category,
                "chunks_found": len(results),
                "total_tokens": sum(self._estimate_tokens(r.get("raw_code", "")) for r in results),
                "summaries_only": summaries_only,
                "results": results,
            },
            indent=2,
        )

    async def _embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for the query text."""
        from spine.config import SpineConfig

        provider_cfg = SpineConfig.load().resolve_embedding_provider()
        if not provider_cfg:
            raise ValueError("No embedding provider configured")

        model_name = provider_cfg.get("model", "text-embedding-3-large")
        api_key = provider_cfg.get("api_key") or ""
        base_url = provider_cfg.get("base_url")

        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            **(base_url and {"base_url": base_url}) or {},
        )

        result = await embeddings.aembed_query(query)
        return np.array(result, dtype=np.float32)

    def _apply_token_budget(
        self, results: list[dict[str, Any]], max_tokens: Optional[int]
    ) -> list[dict[str, Any]]:
        """Apply token budget to results by truncating or filtering.

        Prioritizes results by similarity score.
        """
        if not max_tokens:
            return results

        filtered = []
        total_tokens = 0

        for result in results:
            tokens = self._estimate_tokens(result["raw_code"])
            if total_tokens + tokens <= max_tokens:
                filtered.append(result)
                total_tokens += tokens
            else:
                # Include truncated version
                remaining = max_tokens - total_tokens
                if remaining > 200:  # Only include if meaningful
                    result["raw_code"] = self._truncate_code(result["raw_code"], remaining)
                    filtered.append(result)
                break

        return filtered

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        # Rough approximation: 1 token ≈ 4 characters for code
        return len(text) // 4

    def _truncate_code(self, code: str, max_tokens: int) -> str:
        """Truncate code to fit within token limit."""
        max_chars = max_tokens * 4
        if len(code) <= max_chars:
            return code
        return code[:max_chars] + "\n\n// ... [truncated for token budget]"