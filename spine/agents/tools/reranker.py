"""Cross-encoder reranking for hybrid recall.

A cross-encoder scores each (query, candidate) pair *jointly*, which is
strictly more expressive than the bi-encoder cosine used for the vector
channel — so it can re-order BM25/hybrid candidates by true relevance even
when the standalone vector channel is weak. It is a second, expensive stage
applied to a small candidate pool, not a retriever.

Transport is the Cohere/Jina/vLLM-compatible ``/rerank`` HTTP API:

    POST {base_url}/rerank
    { "model": ..., "query": ..., "documents": ["...", ...] }
    -> { "results": [ { "index": i, "relevance_score": s }, ... ] }

The provider is configured under ``providers.reranker[]`` and selected by
``reranker_provider``. When unset or unreachable, callers fall back to the
fused order — reranking never hard-fails recall.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cross-encoders are typically capped near 512 tokens for the pair; keep the
# candidate text compact so the query half isn't crowded out.
_MAX_DOC_CHARS = 1200

# Rerank route differs by server: Cohere/Jina/vLLM use ``/rerank``,
# llama.cpp uses ``/reranking``. Probed in order against ``base_url`` and the
# first that answers (not 404) is cached per base_url so we probe only once.
_CANDIDATE_PATHS = ("/rerank", "/reranking")
_RESOLVED_PATH: dict[str, str] = {}


async def _post_rerank(client, base_url, provider_cfg, payload, headers):
    """POST to the rerank route, resolving the path once per base_url.

    Honors an explicit ``rerank_path`` from the provider config; otherwise
    probes ``_CANDIDATE_PATHS`` and caches the first that doesn't 404.
    Returns the parsed JSON, or None if no route answered.
    """
    explicit = provider_cfg.get("rerank_path")
    candidates = [explicit] if explicit else list(_CANDIDATE_PATHS)
    if base_url in _RESOLVED_PATH:
        candidates = [_RESOLVED_PATH[base_url]]

    for path in candidates:
        resp = await client.post(f"{base_url}{path}", json=payload, headers=headers)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        _RESOLVED_PATH[base_url] = path
        return resp.json()
    logger.warning("No rerank route found under %s (tried %s)", base_url, candidates)
    return None


def candidate_text(hit: dict[str, Any]) -> str:
    """Build the document text shown to the cross-encoder for one hit.

    Leads with the qualified name + file path (the strongest relevance
    anchors) followed by the identifier-dense summary, then a slice of raw
    code if room remains.
    """
    name = hit.get("symbol_name", "")
    path = hit.get("file_path", "")
    summary = hit.get("enriched_summary", "") or ""
    head = f"{name} ({path})\n{summary}".strip()
    if len(head) >= _MAX_DOC_CHARS:
        return head[:_MAX_DOC_CHARS]
    raw = hit.get("raw_code", "") or ""
    if raw:
        head = f"{head}\n{raw}"
    return head[:_MAX_DOC_CHARS]


async def rerank_hits(
    provider_cfg: dict[str, Any],
    query: str,
    hits: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Re-order ``hits`` by cross-encoder relevance and return the top_k.

    On any error (no httpx, endpoint down, malformed response) the original
    order is preserved and the first ``top_k`` returned — reranking is a
    best-effort enhancement, never a failure point.
    """
    if not hits:
        return hits
    base_url = (provider_cfg.get("base_url") or "").rstrip("/")
    model = provider_cfg.get("model")
    if not base_url or not model:
        logger.warning("Reranker provider missing base_url/model — skipping rerank")
        return hits[:top_k]

    documents = [candidate_text(h) for h in hits]
    payload = {"model": model, "query": query, "documents": documents}
    headers = {}
    api_key = provider_cfg.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            data = await _post_rerank(client, base_url, provider_cfg, payload, headers)
    except Exception as exc:  # noqa: BLE001 — never break recall on a rerank error
        logger.warning("Rerank request failed (%s) — using fused order", exc)
        return hits[:top_k]
    if data is None:
        return hits[:top_k]

    results = data.get("results")
    if not isinstance(results, list) or not results:
        logger.warning("Rerank response had no results — using fused order")
        return hits[:top_k]

    # results: [{index, relevance_score}], sorted or not — sort by score desc.
    ordered: list[dict[str, Any]] = []
    for r in sorted(results, key=lambda x: x.get("relevance_score", 0.0), reverse=True):
        idx = r.get("index")
        if isinstance(idx, int) and 0 <= idx < len(hits):
            item = dict(hits[idx])
            item["rerank_score"] = float(r.get("relevance_score", 0.0))
            ordered.append(item)
        if len(ordered) >= top_k:
            break
    return ordered or hits[:top_k]
