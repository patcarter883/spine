"""Parity tests for the distributed deterministic analysis map-reduce.

Covers Phase A of the onboarding graph (design Revision 2, §2.4, §4.2, §8): the
distributed map-reduce in :mod:`spine.work.onboarding.analysis_nodes` must
produce a :class:`RepoManifest` **byte-identical** (after a name-sort) to the
monolithic :meth:`spine.work.onboarding.analyzer.RepoAnalyzer.analyze` for the
same repository — including under SHUFFLED slice-completion order (the aggregator
must look slices up by key, never by index). A greenfield run yields zero units.

The tests build a small self-contained fixture repo on disk (several modules,
imports, and the convention markers the analyzer's pattern extractor keys on) so
they are hermetic — no MCP server, no vector index.
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.config import SpineConfig
from spine.work.onboarding import analysis_nodes
from spine.work.onboarding.analysis_nodes import (
    _aggregate_analysis_node,
    _analysis_explorer_node,
    _analysis_manager_node,
    build_analysis_graph,
)
from spine.work.onboarding.analyzer import RepoAnalyzer
from spine.work.onboarding.manifest import RepoManifest


# ── Fixture repo ─────────────────────────────────────────────────────────────


_FILES: dict[str, str] = {
    "pkg/work/dispatcher.py": (
        "from __future__ import annotations\n"
        "import logging\n"
        "from pkg.core import settings\n"
        "from pkg.ui import view\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "def submit_work(item: str) -> int:\n"
        "    logger = logging.getLogger(__name__)\n"
        "    try:\n"
        "        return view.render(item)\n"
        "    except Exception:\n"
        "        raise\n"
    ),
    "pkg/work/queue.py": (
        "from __future__ import annotations\n"
        "import logging\n"
        "from dataclasses import dataclass\n"
        "from pkg.core import settings\n"
        "\n"
        "@dataclass(frozen=True)\n"
        "class QueueItem:\n"
        "    name: str\n"
        "\n"
        "def enqueue(item: QueueItem) -> None:\n"
        "    logging.getLogger(__name__).info('q')\n"
    ),
    "pkg/core/settings.py": (
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class Settings:\n"
        "    debug: bool = False\n"
        "\n"
        "def load() -> Settings:\n"
        "    return Settings()\n"
    ),
    "pkg/ui/view.py": (
        "from __future__ import annotations\n"
        "import logging\n"
        "from pkg.work import dispatcher\n"
        "\n"
        "def render(item: str) -> int:\n"
        "    try:\n"
        "        return len(item)\n"
        "    except Exception:\n"
        "        logging.getLogger(__name__).exception('boom')\n"
        "        raise\n"
    ),
    "tests/test_work.py": (
        "from __future__ import annotations\n"
        "\n"
        "def test_submit() -> None:\n"
        "    assert True\n"
        "\n"
        "class TestQueue:\n"
        "    def test_enqueue(self) -> None:\n"
        "        assert True\n"
    ),
}


def _write_fixture(root: Path) -> None:
    for rel, content in _FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _hermetic_config(root: Path) -> SpineConfig:
    """No MCP (forces os.walk) + bogus vector db (no summaries)."""
    return SpineConfig(
        mcp_servers={},
        checkpoint_path="/nonexistent/spine-parity-test.db",
        workspace_root=str(root),
    )


def _normalise(manifest: RepoManifest) -> dict[str, Any]:
    """Manifest dict with the non-deterministic fields zeroed for comparison.

    ``generated_at`` is a wall-clock timestamp and ``workspace_root`` is the
    absolute path; both are intentionally excluded from the byte-parity check.
    Boundaries are name-sorted (the design's "after name-sort" qualifier).
    """
    data = manifest.to_dict()
    data["generated_at"] = ""
    data["workspace_root"] = ""
    data["module_boundaries"] = sorted(
        data.get("module_boundaries", []), key=lambda b: b["name"]
    )
    return data


# ── Distributed driver (manager → explorers → aggregate), order-controllable ──


def _run_distributed(
    root: Path,
    mode: str,
    config: SpineConfig,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> RepoManifest:
    """Drive the analysis nodes by hand so slice order can be shuffled.

    Mirrors what ``build_analysis_graph`` runs, but lets the test control the
    order explorer slices land in ``repo_slices`` to prove the aggregator looks
    up by key (never by index).
    """
    runnable_config = {"configurable": {"spine_config": config}}
    state: dict[str, Any] = {
        "work_id": "parity",
        "workspace_root": str(root),
        "mode": mode,
        "tech_stack": [],
    }
    manager_out = asyncio.run(_analysis_manager_node(state, runnable_config))
    state.update(manager_out)

    slices: list[dict[str, Any]] = []
    for unit in state.get("analysis_units", []) or []:
        explorer_state = dict(state)
        explorer_state["active_unit"] = unit
        out = _analysis_explorer_node(explorer_state, runnable_config)
        slices.extend(out["repo_slices"])

    if shuffle:
        random.Random(seed).shuffle(slices)

    state["repo_slices"] = slices
    agg_out = _aggregate_analysis_node(state, runnable_config)
    return RepoManifest.from_dict(agg_out["manifest"])


# ── Tests ────────────────────────────────────────────────────────────────────


def test_distributed_matches_monolith(tmp_path: Path) -> None:
    """Distributed deterministic analysis == monolithic analyze() (name-sorted)."""
    _write_fixture(tmp_path)
    config = _hermetic_config(tmp_path)

    monolith = asyncio.run(
        RepoAnalyzer(config=config).analyze(str(tmp_path), mode="brownfield")
    )
    distributed = _run_distributed(tmp_path, "brownfield", config)

    assert _normalise(distributed) == _normalise(monolith)


def test_distributed_matches_monolith_under_shuffled_slices(tmp_path: Path) -> None:
    """The aggregator looks up by key, so shuffled slice order changes nothing."""
    _write_fixture(tmp_path)
    config = _hermetic_config(tmp_path)

    monolith = asyncio.run(
        RepoAnalyzer(config=config).analyze(str(tmp_path), mode="brownfield")
    )
    monolith_norm = _normalise(monolith)

    for seed in range(5):
        distributed = _run_distributed(
            tmp_path, "brownfield", config, shuffle=True, seed=seed
        )
        assert _normalise(distributed) == monolith_norm, f"mismatch at seed {seed}"


def test_distributed_via_compiled_graph(tmp_path: Path) -> None:
    """The compiled ``build_analysis_graph`` produces the same manifest."""
    _write_fixture(tmp_path)
    config = _hermetic_config(tmp_path)

    monolith = asyncio.run(
        RepoAnalyzer(config=config).analyze(str(tmp_path), mode="brownfield")
    )

    graph = build_analysis_graph().compile()
    final = asyncio.run(
        graph.ainvoke(
            {
                "work_id": "parity-graph",
                "workspace_root": str(tmp_path),
                "mode": "brownfield",
                "tech_stack": [],
            },
            {"configurable": {"spine_config": config}},
        )
    )
    distributed = RepoManifest.from_dict(final["manifest"])
    assert _normalise(distributed) == _normalise(monolith)
    assert final["manifest_path"]


def test_greenfield_zero_units(tmp_path: Path) -> None:
    """Greenfield seeds zero analysis units and a greenfield manifest."""
    config = _hermetic_config(tmp_path)
    runnable_config = {"configurable": {"spine_config": config}}

    out = asyncio.run(
        _analysis_manager_node(
            {
                "work_id": "green",
                "workspace_root": str(tmp_path),
                "mode": "greenfield",
                "tech_stack": ["python", "fastapi"],
            },
            runnable_config,
        )
    )
    assert out["analysis_units"] == []
    assert out["prebuilt_manifest"]["mode"] == "greenfield"

    # Router routes straight to the aggregator with no units. It returns the
    # plain node name (not a Send) so the aggregator sees the full graph state
    # (incl. prebuilt_manifest) rather than an empty Send payload.
    route = analysis_nodes._analysis_router(out)
    assert route == "aggregate_analysis"

    # End-to-end greenfield graph: zero units, manifest passed through.
    graph = build_analysis_graph().compile()
    final = asyncio.run(
        graph.ainvoke(
            {
                "work_id": "green-graph",
                "workspace_root": str(tmp_path),
                "mode": "greenfield",
                "tech_stack": ["python", "fastapi"],
            },
            runnable_config,
        )
    )
    manifest = RepoManifest.from_dict(final["manifest"])
    assert manifest.mode == "greenfield"
    assert manifest.module_boundaries == []
    assert manifest.symbol_count == 0
    assert final["manifest_path"]


def test_monolithic_fallback_flag_off(tmp_path: Path) -> None:
    """``onboarding_distributed_analysis=False`` → 0 units, manifest inline."""
    _write_fixture(tmp_path)
    config = _hermetic_config(tmp_path)
    config.onboarding_distributed_analysis = False
    runnable_config = {"configurable": {"spine_config": config}}

    out = asyncio.run(
        _analysis_manager_node(
            {
                "work_id": "mono",
                "workspace_root": str(tmp_path),
                "mode": "brownfield",
                "tech_stack": [],
            },
            runnable_config,
        )
    )
    assert out["analysis_units"] == []
    assert out["prebuilt_manifest"]["mode"] == "brownfield"

    # And the passthrough manifest equals the monolith.
    monolith = asyncio.run(
        RepoAnalyzer(config=config).analyze(str(tmp_path), mode="brownfield")
    )
    distributed = RepoManifest.from_dict(out["prebuilt_manifest"])
    assert _normalise(distributed) == _normalise(monolith)
