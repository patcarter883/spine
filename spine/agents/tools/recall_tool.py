"""Native RAG Recall Tool for SPINE SPECIFY phase.

Retrieves relevant code chunks from the vector store based on semantic similarity.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from spine.agents._tokens import count_tokens
from spine.persistence.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RecallInput(BaseModel):
    """Input schema for the RecallTool."""

    query: str = Field(
        description="Natural language search query describing what you need to find"
    )
    k: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Maximum number of chunks to return (0 = use config default, max 50)",
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

    Performs semantic search against the vector store using the query
    embedding. Token budget control is applied to stay within context
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

    def _run(
        self,
        query: str,
        k: int = 0,
        max_tokens: Optional[int] = 50000,
        summaries_only: bool = False,
    ) -> str:
        """Execute the recall tool synchronously."""
        return self._recall_sync(query, k, max_tokens, summaries_only)

    async def _arun(
        self,
        query: str,
        k: int = 0,
        max_tokens: Optional[int] = 50000,
        summaries_only: bool = False,
    ) -> str:
        """Execute the recall tool asynchronously."""
        return await self._recall_async(query, k, max_tokens, summaries_only)

    def _recall_sync(
        self,
        query: str,
        k: int,
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
                    asyncio.run, self._recall_async(query, k, max_tokens, summaries_only)
                )
                return future.result(timeout=60)
        else:
            return asyncio.run(self._recall_async(query, k, max_tokens, summaries_only))

    async def _recall_async(
        self,
        query: str,
        k: int,
        max_tokens: Optional[int],
        summaries_only: bool = False,
    ) -> str:
        """Async recall implementation."""
        import os

        from spine.config import SpineConfig

        cfg = SpineConfig.load()
        if k == 0:
            k = cfg.recall_k

        embedding = await self._embed_query(query)

        store = VectorStore(self.db_path)
        store.ensure_schema()

        # Hybrid (BM25 + vector, RRF-fused) is the default: the local
        # embedding space is weak/anisotropic, so lexical match on
        # identifiers is what rescues exact-symbol queries. The raw query
        # text (no instruction prefix) drives BM25; the embedding drives
        # the vector side. RRF channel weights come from config (vector
        # down-weighted because it is noisy); env overrides for eval sweeps.
        v_w = float(os.getenv("SPINE_RRF_VECTOR_WEIGHT", str(cfg.rrf_vector_weight)))
        b_w = float(os.getenv("SPINE_RRF_BM25_WEIGHT", str(cfg.rrf_bm25_weight)))

        # Optional cross-encoder rerank: retrieve a larger candidate pool,
        # then re-order it to the final k. Disabled unless a reranker
        # provider is configured (env SPINE_RERANK=off force-disables).
        reranker_cfg = cfg.resolve_reranker_provider()
        rerank_on = reranker_cfg is not None and os.getenv("SPINE_RERANK", "").lower() != "off"
        fetch_k = max(k, cfg.rerank_pool) if rerank_on else k

        results = store.search_hybrid(
            embedding, query, k=fetch_k, vector_weight=v_w, bm25_weight=b_w
        )

        if rerank_on:
            from spine.agents.tools.reranker import rerank_hits

            results = await rerank_hits(reranker_cfg, query, results, top_k=k)

        results = self._apply_token_budget(results, max_tokens)

        if summaries_only:
            for r in results:
                r.pop("raw_code", None)

        return json.dumps(
            {
                "query": query,
                "chunks_found": len(results),
                "total_tokens": sum(count_tokens(r.get("raw_code", "")) for r in results),
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

        model_name = provider_cfg.get("model")
        if not model_name:
            raise ValueError(
                f"Embedding provider {provider_cfg.get('name')!r} has no 'model' set"
            )
        api_key = provider_cfg.get("api_key") or ""
        base_url = provider_cfg.get("base_url")

        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            **(base_url and {"base_url": base_url}) or {},
        )

        # Query-side embedding prefix, from the provider config. Asymmetric
        # models need a different prefix than indexed documents (nomic:
        # "search_query: " here vs "search_document: " at index time). The
        # prefix MUST match the one used during indexing for the same model.
        query_prefix = provider_cfg.get("query_prefix", "")
        result = await embeddings.aembed_query(query_prefix + query)
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
            tokens = count_tokens(result["raw_code"])
            if total_tokens + tokens <= max_tokens:
                filtered.append(result)
                total_tokens += tokens
            else:
                remaining = max_tokens - total_tokens
                if remaining > 200:
                    result["raw_code"] = self._truncate_code(result["raw_code"], remaining)
                    filtered.append(result)
                break

        return filtered

    @staticmethod
    def _truncate_code(code: str, max_tokens: int) -> str:
        """Truncate code to fit within token limit."""
        # Char-budget approximation is fine here — this is a hard cap on
        # output, not a ranking input. ~4 chars/token over-shoots slightly
        # for code, which is the safe direction.
        max_chars = max_tokens * 4
        if len(code) <= max_chars:
            return code
        return code[:max_chars] + "\n\n// ... [truncated for token budget]"
