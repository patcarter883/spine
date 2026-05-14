"""SPINE UI page registry — shared page objects for navigation.

Stores the ``st.Page`` objects created by ``app.py`` so that individual
page modules can use ``st.switch_page(page)`` for programmatic
navigation (e.g. "click a work item → go to its detail page").

Pages are registered once by ``app.py`` after calling ``st.navigation``.
"""

from __future__ import annotations

from typing import Any

# ── Page object registry ──
# Keyed by url_path (e.g. "dashboard", "work-detail", "work-history")
_pages: dict[str, Any] = {}


def register(url_path: str, page: Any) -> None:
    """Register a page object for cross-page navigation."""
    _pages[url_path] = page


def get(url_path: str) -> Any:
    """Look up a page object by its url_path."""
    return _pages.get(url_path)


def all_pages() -> dict[str, Any]:
    """Return the full page registry."""
    return dict(_pages)
