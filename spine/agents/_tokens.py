"""Shared token-counting utility used by context-management middleware
and the specify-phase context renderer.

Uses tiktoken's ``cl100k_base`` encoder when available; falls back to a
~4-chars-per-token heuristic so call sites never need to handle ImportError.
"""

from __future__ import annotations


def count_tokens(text: str) -> int:
    """Best-effort token count via tiktoken with a 4-char-per-token fallback."""
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4
