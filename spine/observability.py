"""LangSmith tracing control for SPINE.

Tracing is disabled process-wide by default (see ``spine.config`` —
``_disable_global_tracing``) so that codebase indexing, the test suite,
onboarding repo analysis, and any ad-hoc LangGraph usage do not emit traces.

Only genuine **work task runs** opt in, by wrapping their graph execution in
``work_run_tracing``.  This uses a contextvar-scoped LangChain tracer rather
than flipping ``os.environ`` — so it is async- and concurrency-safe: a work
run streaming under a tracer in one task does not cause a concurrent indexing
job in the same process to start tracing.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import AsyncIterator, Iterator

logger = logging.getLogger(__name__)


@contextmanager
def work_run_tracing(work_id: str, work_type: str) -> Iterator[None]:
    """Enable LangSmith tracing for the duration of a work task run.

    Tracing is off process-wide by default; this context manager turns it on
    for exactly the wrapped LangGraph execution.  The tracer is installed via
    ``langchain_core.tracers.context.tracing_v2_enabled``, which sets a
    contextvar that always traces inside the block regardless of the ambient
    ``LANGSMITH_TRACING`` flag, and is scoped to the current async context.

    No-op (tracing stays off) when ``SPINE_TRACE_ALL`` is set — in that mode
    the global flag is already on, so wrapping would only double-install a
    tracer — or when no LangSmith API key is configured.

    Args:
        work_id: The work item ID — attached as a tag for filtering in the UI.
        work_type: The work type (e.g. ``"task"``) — attached as a tag.
    """
    trace_all = os.environ.get("SPINE_TRACE_ALL", "").lower() in ("1", "true", "yes")
    has_key = bool(
        os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    )
    if trace_all or not has_key:
        # Either tracing everything already (escape hatch) or no credentials to
        # trace to — nothing to scope here.
        yield
        return

    from langchain_core.tracers.context import tracing_v2_enabled

    project = (
        os.environ.get("LANGSMITH_PROJECT")
        or os.environ.get("LANGCHAIN_PROJECT")
        or "spine"
    )
    logger.debug("Enabling LangSmith tracing for work run %s (%s)", work_id, work_type)
    with tracing_v2_enabled(
        project_name=project,
        tags=[f"work_type:{work_type}", f"work_id:{work_id}"],
    ):
        yield


async def traced_astream(stream, work_id: str, work_type: str) -> AsyncIterator:
    """Wrap a LangGraph ``astream`` iterator so the run is traced to LangSmith.

    Yields chunks from ``stream`` with ``work_run_tracing`` active for the full
    lifetime of the iteration.  Because the tracer is contextvar-scoped to this
    generator frame, it covers exactly the wrapped stream and nothing else in
    the same process — a concurrent indexing job streaming a different graph
    stays untraced.

    Works for both ``async for`` consumption and manual ``__anext__()`` driving
    (the dispatcher's stall-timeout loop): on a stall the in-flight
    ``__anext__`` is cancelled, which propagates through this generator and
    runs the tracer's teardown, leaving the caller's context untouched.

    Args:
        stream: The async iterator returned by ``graph.astream(...)``.
        work_id: The work item ID — attached as a trace tag.
        work_type: The work type — attached as a trace tag.
    """
    with work_run_tracing(work_id, work_type):
        async for chunk in stream:
            yield chunk
