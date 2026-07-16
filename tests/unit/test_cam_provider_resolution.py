"""CAM provider resolution: the organ is not always on the ACTIVE provider.

Live config regression (2026-07-16): openrouter is listed first (= the
"active" provider) while the cam: block sits on the minisgl lane that phase
routing actually uses — every CAM entry point keyed off
resolve_active_provider() and concluded CAM was unconfigured.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.agents.facts import _cam_provider
from spine.config import SpineConfig


def _load(tmp_path, yaml_text: str) -> SpineConfig:
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    return SpineConfig.load(path=str(p))


def test_cam_provider_found_behind_the_active_one(tmp_path):
    cfg = _load(
        tmp_path,
        "providers:\n  llm:\n"
        "  - name: openrouter\n    model: openai:remote\n"
        "    base_url: http://r:1/v1\n    enabled: true\n"
        "  - name: minisgl\n    model: openai:local\n"
        "    base_url: http://m:1919/v1\n    enabled: true\n"
        "    cam: {namespace: p}\n",
    )
    assert cfg.resolve_active_provider()["name"] == "openrouter"
    assert cfg.resolve_cam_provider()["name"] == "minisgl"


def test_cam_provider_skips_disabled_entries_and_disabled_cam(tmp_path):
    cfg = _load(
        tmp_path,
        "providers:\n  llm:\n"
        "  - name: off-provider\n    model: openai:a\n"
        "    base_url: http://a:1/v1\n    enabled: false\n"
        "    cam: true\n"
        "  - name: cam-disabled\n    model: openai:b\n"
        "    base_url: http://b:1/v1\n    enabled: true\n"
        "    cam: {enabled: false}\n"
        "  - name: cam-false\n    model: openai:c\n"
        "    base_url: http://c:1/v1\n    enabled: true\n"
        "    cam: false\n",
    )
    assert cfg.resolve_cam_provider() is None


def test_cam_true_shorthand_counts(tmp_path):
    cfg = _load(
        tmp_path,
        "providers:\n  llm:\n"
        "  - name: minisgl\n    model: openai:local\n"
        "    base_url: http://m:1919/v1\n    enabled: true\n    cam: true\n",
    )
    assert cfg.resolve_cam_provider()["name"] == "minisgl"


def test_facts_helper_prefers_cam_resolver_falls_back_to_active():
    cam_prov = {"name": "minisgl", "cam": {"namespace": "p"}}
    both = SimpleNamespace(
        resolve_cam_provider=lambda: cam_prov,
        resolve_active_provider=lambda: {"name": "active"},  # no cam
    )
    assert _cam_provider(both)["name"] == "minisgl"

    # Config objects that predate resolve_cam_provider (test doubles) still
    # work when their active provider carries the cam block.
    legacy = SimpleNamespace(resolve_active_provider=lambda: cam_prov)
    assert _cam_provider(legacy)["name"] == "minisgl"

    # No cam anywhere -> None.
    none = SimpleNamespace(resolve_active_provider=lambda: {"name": "a"})
    assert _cam_provider(none) is None
