"""Tests for the MCP result path-exclusion filter.

Covers the regression where ``codebase_query`` (and every other
``mcp_codebase-index_*`` tool routed through ``_post_process_result``)
was returning paths under dot-folders like ``.git/``, ``.venv/``,
``.spine/``, polluting research findings with noise.
"""

from __future__ import annotations

import json

from spine.mcp.client import (
    EXCLUDED_INDEX_PATHS,
    _is_excluded_path,
    _line_starts_with_excluded_path,
    _strip_excluded_paths,
)


class TestDotFolderExclusion:
    """Any path segment starting with '.' (other than '.' or '..') is dropped."""

    def test_drops_dot_folder_at_root(self):
        assert _line_starts_with_excluded_path(".git/HEAD")
        assert _line_starts_with_excluded_path(".venv/lib/python3.13/site-packages/x.py")
        assert _line_starts_with_excluded_path(".pytest_cache/CACHEDIR.TAG")
        assert _line_starts_with_excluded_path(".env")

    def test_drops_dot_folder_nested(self):
        assert _line_starts_with_excluded_path("spine/.cache/x.py")
        assert _line_starts_with_excluded_path("src/foo/.hidden/y")

    def test_keeps_normal_paths(self):
        assert not _line_starts_with_excluded_path("spine/agents/factory.py")
        assert not _line_starts_with_excluded_path("README.md")
        assert not _line_starts_with_excluded_path("tests/unit/test_foo.py")

    def test_keeps_current_and_parent_dir_markers(self):
        # ./foo and ../foo are NOT dot-folders.
        assert not _line_starts_with_excluded_path("./foo.py")
        assert not _line_starts_with_excluded_path("./spine/agents/factory.py")
        # Note: ../ is rejected at the workspace boundary elsewhere; the
        # path-line regex doesn't accept '..' as a leading segment anyway.

    def test_explicit_exclusions_still_apply(self):
        # The original EXCLUDED_INDEX_PATHS list keeps working alongside
        # the generic dot-folder rule.
        for prefix in EXCLUDED_INDEX_PATHS:
            sample = prefix + "anything"
            assert _line_starts_with_excluded_path(sample), (
                f"explicit exclusion '{prefix}' must still match"
            )


class TestStripExcludedPathsAggregate:
    """End-to-end: feed _strip_excluded_paths a mixed result blob and
    verify only the dot-folder lines disappear."""

    def test_mixed_blob_keeps_only_clean_paths(self):
        blob = "\n".join([
            "spine/agents/factory.py: def build_phase_agent",
            ".git/HEAD: ref refs/heads/main",
            ".venv/lib/python3.13/site-packages/foo.py: def bar",
            "tests/unit/test_x.py: def test_x",
            ".spine/artifacts/foo.md: some content",
            "spine/.cache/x.py: noise",
            "README.md: project intro",
        ])
        out, dropped = _strip_excluded_paths(blob)
        kept_lines = out.splitlines()
        assert "spine/agents/factory.py: def build_phase_agent" in kept_lines
        assert "tests/unit/test_x.py: def test_x" in kept_lines
        assert "README.md: project intro" in kept_lines
        for noisy in (".git/HEAD", ".venv/", ".spine/", "spine/.cache/"):
            assert not any(noisy in ln for ln in kept_lines), (
                f"line containing {noisy!r} should have been dropped"
            )
        assert dropped == 4

    def test_drops_dot_folders_without_explicit_spine_prefix(self):
        # Regression: the dot-folder rule must fire even when NONE of the
        # hard-coded EXCLUDED_INDEX_PATHS (.spine/...) prefixes appear in the
        # blob. Previously a guard short-circuited on those prefixes, leaking
        # .git/.venv/.pytest_cache paths into research findings.
        blob = "\n".join([
            "spine/agents/factory.py: def build_phase_agent",
            ".git/HEAD: ref refs/heads/main",
            ".venv/lib/python3.13/site-packages/foo.py: def bar",
            ".pytest_cache/CACHEDIR.TAG: noise",
            "  - .hidden/config.toml: leading-bullet hidden dir",
            "README.md: project intro",
        ])
        out, dropped = _strip_excluded_paths(blob)
        kept_lines = out.splitlines()
        assert kept_lines == [
            "spine/agents/factory.py: def build_phase_agent",
            "README.md: project intro",
        ]
        assert dropped == 4

    def test_clean_blob_passes_through_unchanged(self):
        # No dot-folders, no explicit prefixes → original text preserved
        # verbatim (including trailing newline) and nothing dropped.
        blob = "spine/agents/factory.py: def f\nREADME.md: intro\n"
        out, dropped = _strip_excluded_paths(blob)
        assert out == blob
        assert dropped == 0


class TestIsExcludedPath:
    """Unit tests for the raw-path helper used by both branches."""

    def test_dot_folder_at_root(self):
        assert _is_excluded_path(".agent/skills/x.md")
        assert _is_excluded_path(".deepagents/core/SKILL.md")
        assert _is_excluded_path(".git/HEAD")
        assert _is_excluded_path(".venv/lib/python3.13/site.py")

    def test_dot_folder_nested(self):
        assert _is_excluded_path("src/.cache/x.py")
        assert _is_excluded_path("spine/.hidden/y")

    def test_explicit_excluded_paths(self):
        for prefix in EXCLUDED_INDEX_PATHS:
            assert _is_excluded_path(prefix + "anything")

    def test_normal_paths_kept(self):
        assert not _is_excluded_path("spine/agents/factory.py")
        assert not _is_excluded_path("README.md")
        assert not _is_excluded_path("./spine/agents/factory.py")


class TestStripExcludedPathsJsonFormat:
    """_strip_excluded_paths must filter JSON-array results from codebase-index."""

    def _make_result(self, file: str, content: str = "x") -> dict:
        return {"file": file, "line_number": 1, "content": content}

    def test_all_hidden_entries_dropped(self):
        data = [
            self._make_result(".agent/skills/langsmith-trace-analysis/SKILL.md"),
            self._make_result(".deepagents/skills/deep-agents-core/SKILL.md"),
        ]
        out, dropped = _strip_excluded_paths(json.dumps(data))
        assert dropped == 2
        assert json.loads(out) == []

    def test_mixed_entries_only_hidden_dropped(self):
        data = [
            self._make_result("spine/agents/factory.py", "clean"),
            self._make_result(".agent/skills/x.md", "hidden"),
            self._make_result("README.md", "also clean"),
            self._make_result(".deepagents/core/SKILL.md", "hidden2"),
        ]
        out, dropped = _strip_excluded_paths(json.dumps(data))
        assert dropped == 2
        kept = json.loads(out)
        assert len(kept) == 2
        assert all(r["file"] in ("spine/agents/factory.py", "README.md") for r in kept)

    def test_no_hidden_entries_returns_original(self):
        data = [
            self._make_result("spine/agents/factory.py"),
            self._make_result("README.md"),
        ]
        blob = json.dumps(data)
        out, dropped = _strip_excluded_paths(blob)
        assert dropped == 0
        assert out is blob  # identical object — no copy made

    def test_empty_json_array_unchanged(self):
        out, dropped = _strip_excluded_paths("[]")
        assert dropped == 0

    def test_non_json_starting_with_bracket_falls_back_to_text(self):
        # Starts with '[' but isn't valid JSON → text-based filter fires.
        blob = "[not json\n.git/HEAD: ref\nREADME.md: intro"
        out, dropped = _strip_excluded_paths(blob)
        assert dropped == 1
        assert ".git/HEAD" not in out
        assert "README.md" in out
