"""Tests for _NormalizingLocalShellBackend path normalization."""

from pathlib import Path

from spine.agents.backend import _NormalizingLocalShellBackend


class TestPathNormalization:
    """Verify that absolute paths matching workspace root are stripped."""

    def test_relative_path_unchanged(self, tmp_path: Path) -> None:
        backend = _NormalizingLocalShellBackend(
            root_dir=str(tmp_path), virtual_mode=True
        )
        result = backend._resolve_path(".spine/artifacts/test/spec.md")
        assert result == tmp_path / ".spine" / "artifacts" / "test" / "spec.md"

    def test_workspace_root_prefix_stripped(self, tmp_path: Path) -> None:
        backend = _NormalizingLocalShellBackend(
            root_dir=str(tmp_path), virtual_mode=True
        )
        absolute = f"{tmp_path}/.spine/artifacts/test/spec.md"
        result = backend._resolve_path(absolute)
        # Should resolve to the same path as the relative version
        assert result == tmp_path / ".spine" / "artifacts" / "test" / "spec.md"

    def test_non_root_absolute_path_treated_as_virtual(self, tmp_path: Path) -> None:
        backend = _NormalizingLocalShellBackend(
            root_dir=str(tmp_path), virtual_mode=True
        )
        # /etc/passwd doesn't start with tmp_path, so it's treated as
        # a virtual path under tmp_path — existing DA behavior
        result = backend._resolve_path("/etc/passwd")
        assert result == tmp_path / "etc" / "passwd"

    def test_leading_slash_workspace_relative(self, tmp_path: Path) -> None:
        backend = _NormalizingLocalShellBackend(
            root_dir=str(tmp_path), virtual_mode=True
        )
        result = backend._resolve_path("/src/main.py")
        assert result == tmp_path / "src" / "main.py"

    def test_no_double_nesting(self, tmp_path: Path) -> None:
        """The primary bug: /home/pat/proj/.spine/... no longer double-nests."""
        backend = _NormalizingLocalShellBackend(
            root_dir=str(tmp_path), virtual_mode=True
        )
        # Simulate the exact pattern from the trace
        artifact_path = f"{tmp_path}/.spine/artifacts/aecf6210/specify/specification.md"
        result = backend._resolve_path(artifact_path)
        # Must NOT be tmp_path / "home" / "pat" / ...
        assert "home" not in result.parts
        assert result == tmp_path / ".spine" / "artifacts" / "aecf6210" / "specify" / "specification.md"
