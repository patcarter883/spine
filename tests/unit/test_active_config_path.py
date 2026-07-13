"""CLI --config becomes the process's ACTIVE config for bare load() callers.

Run d8bc459c (all-Zaya experiment): --config routed every lane to zaya,
but _active_provider_config re-read the default path on each model build —
the whole pipeline silently ran on the MAIN config's provider.
"""

import pytest

from spine.config import SpineConfig


@pytest.fixture(autouse=True)
def _reset_active_path():
    saved = SpineConfig._active_path
    SpineConfig._active_path = None
    yield
    SpineConfig._active_path = saved


def _write(tmp_path, name, provider_name):
    p = tmp_path / name
    p.write_text(
        "providers:\n  llm:\n"
        f"  - name: {provider_name}\n"
        f"    model: openai:{provider_name}-model\n"
        "    base_url: http://x:1/v1\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    return str(p)


def test_bare_load_follows_active_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, ".spine-main.yaml", "main-provider")
    override = _write(tmp_path, "config-override.yaml", "override-provider")

    SpineConfig.load_as_active(path=override)
    cfg = SpineConfig.load()  # bare — the model layer's call shape
    assert cfg.resolve_active_provider()["name"] == "override-provider"


def test_bare_load_without_active_uses_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".spine").mkdir()
    (tmp_path / ".spine" / "config.yaml").write_text(
        "providers:\n  llm:\n  - name: default-provider\n"
        "    model: openai:d\n    base_url: http://x:1/v1\n    enabled: true\n",
        encoding="utf-8",
    )
    cfg = SpineConfig.load()
    assert cfg.resolve_active_provider()["name"] == "default-provider"


def test_explicit_path_still_wins_over_active(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    override = _write(tmp_path, "config-override.yaml", "override-provider")
    other = _write(tmp_path, "config-other.yaml", "other-provider")

    SpineConfig.load_as_active(path=override)
    cfg = SpineConfig.load(path=other)
    assert cfg.resolve_active_provider()["name"] == "other-provider"
