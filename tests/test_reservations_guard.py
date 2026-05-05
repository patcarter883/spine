"""Tests for file reservation and write guard functionality."""

import sys
from pathlib import Path

import pytest

# Import modules directly without going through spine.__init__
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import reservations directly
import importlib.util
reservations_spec = importlib.util.spec_from_file_location(
    "spine.hive.reservations", "spine/hive/reservations.py"
)
reservations_module = importlib.util.module_from_spec(reservations_spec)
reservations_spec.loader.exec_module(reservations_module)
OwnedReservation = reservations_module.OwnedReservation
ResourceManager = reservations_module.ResourceManager

# Import storage directly
import importlib.util
storage_spec = importlib.util.spec_from_file_location(
    "spine.providers.storage", "spine/providers/storage.py"
)
storage_module = importlib.util.module_from_spec(storage_spec)

# Need to set up base module first
base_spec = importlib.util.spec_from_file_location(
    "spine.providers.base", "spine/providers/base.py"
)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["spine.providers.base"] = base_module
base_spec.loader.exec_module(base_module)

storage_spec.loader.exec_module(storage_module)
FileWriteGuard = storage_module.FileWriteGuard
LocalStorageProvider = storage_module.LocalStorageProvider


# --- OwnedReservation tests ---

class TestOwnedReservation:
    """Test the OwnedReservation dataclass."""

    def test_capture_content(self):
        """capture_content should store original content."""
        res = OwnedReservation(
            agent_id="agent1",
            paths=["file.py"],
            exclusive=True,
            reserved_at="2024-01-01T00:00:00",
        )
        res.capture_content("file.py", "original content")
        assert res.original_contents["file.py"] == "original content"

    def test_get_diff_identical(self):
        """get_diff should return empty list for identical content."""
        res = OwnedReservation(
            agent_id="agent1",
            paths=["file.py"],
            exclusive=True,
            reserved_at="2024-01-01T00:00:00",
        )
        res.capture_content("file.py", "same content\n")
        diff = res.get_diff("file.py", "same content\n")
        assert diff == []

    def test_get_diff_with_changes(self):
        """get_diff should return diff lines for changed content."""
        res = OwnedReservation(
            agent_id="agent1",
            paths=["file.py"],
            exclusive=True,
            reserved_at="2024-01-01T00:00:00",
        )
        res.capture_content("file.py", "line1\nline2\n")
        diff = res.get_diff("file.py", "line1\nmodified\n")
        assert len(diff) > 0
        assert any("modified" in line for line in diff)

    def test_get_diff_missing_content(self):
        """get_diff should handle missing original content."""
        res = OwnedReservation(
            agent_id="agent1",
            paths=["file.py"],
            exclusive=True,
            reserved_at="2024-01-01T00:00:00",
        )
        diff = res.get_diff("file.py", "new content")
        # Should treat empty original as all new
        assert len(diff) > 0


# --- ResourceManager tests ---

class TestResourceManagerOwnerReservation:
    """Test ResourceManager with OwnedReservation integration."""

    def test_reserve_returns_true(self):
        """reserve should return True for valid reservation."""
        rm = ResourceManager()
        result = rm.reserve("agent1", ["file.py"])
        assert result is True

    def test_reserve_adds_owned_reservation(self):
        """reserve should create OwnedReservation instance."""
        rm = ResourceManager()
        rm.reserve("agent1", ["file.py"])
        assert "agent1" in rm._reservations
        assert isinstance(rm._reservations["agent1"], OwnedReservation)

    def test_reserve_with_ttl(self):
        """reserve should accept ttl_seconds parameter."""
        rm = ResourceManager()
        rm.reserve("agent1", ["file.py"], ttl_seconds=300)
        assert rm._reservations["agent1"].ttl_seconds == 300

    def test_capture_original(self):
        """capture_original should store content in reservation."""
        rm = ResourceManager()
        rm.reserve("agent1", ["file.py"])
        rm.capture_original("agent1", "file.py", "original content")
        assert rm._reservations["agent1"].original_contents["file.py"] == "original content"

    def test_verify_diff(self):
        """verify_diff should return diff between original and new."""
        rm = ResourceManager()
        rm.reserve("agent1", ["file.py"])
        rm.capture_original("agent1", "file.py", "original\n")
        diff = rm.verify_diff("agent1", "file.py", "modified\n")
        assert len(diff) > 0

    def test_release_returns_verification_result(self):
        """release should return verification dict."""
        rm = ResourceManager()
        rm.reserve("agent1", ["file.py"])
        result = rm.release("agent1")
        assert result["agent"] == "agent1"
        assert result["verified"] is True
        assert "file.py" in result["paths"]

    def test_is_reserved_uses_dataclass(self):
        """is_reserved should work with OwnedReservation."""
        rm = ResourceManager()
        rm.reserve("agent1", ["file.py"])
        assert rm.is_reserved("file.py") is True
        assert rm.is_reserved("other.py") is False


# --- FileWriteGuard tests ---

class TestFileWriteGuard:
    """Test FileWriteGuard functionality."""

    def test_init_without_manager(self):
        """FileWriteGuard should work without resource manager."""
        guard = FileWriteGuard()
        assert guard._resource_manager is None

    def test_check_reservation_no_manager(self):
        """check_reservation should return True without manager."""
        guard = FileWriteGuard()
        assert guard.check_reservation("agent1", "file.py") is True

    def test_check_reservation_with_manager(self):
        """check_reservation should validate against resource manager."""
        rm = ResourceManager()
        rm.reserve("agent1", ["file.py"])
        guard = FileWriteGuard(resource_manager=rm)
        
        assert guard.check_reservation("agent1", "file.py") is True
        assert guard.check_reservation("agent2", "file.py") is False

    def test_guarded_write_without_reservation(self):
        """guarded_write should fail when another agent has reservation."""
        rm = ResourceManager()
        rm.reserve("agent2", ["file.py"])  # agent2 has reservation
        guard = FileWriteGuard(resource_manager=rm)
        
        result = guard.guarded_write("agent1", "file.py", b"content")
        assert result["success"] is False
        assert "error" in result

    def test_guarded_write_when_not_reserved(self):
        """guarded_write should succeed when file is not reserved by anyone."""
        rm = ResourceManager()
        guard = FileWriteGuard(resource_manager=rm)
        
        result = guard.guarded_write("agent1", "file.py", b"content")
        assert result["success"] is True  # No reservation = allowed

    def test_guarded_write_with_reservation(self):
        """guarded_write should succeed with reservation."""
        rm = ResourceManager()
        rm.reserve("agent1", ["file.py"])
        guard = FileWriteGuard(resource_manager=rm)
        
        result = guard.guarded_write("agent1", "file.py", b"new content")
        assert result["success"] is True

    def test_guarded_write_captures_diff(self):
        """guarded_write should capture and return diff."""
        rm = ResourceManager()
        rm.reserve("agent1", ["file.py"])
        guard = FileWriteGuard(resource_manager=rm)
        
        # Capture original content
        rm.capture_original("agent1", "file.py", "original\n")
        
        result = guard.guarded_write("agent1", "file.py", b"modified\n")
        assert result["success"] is True
        assert "diff" in result

    def test_guarded_write_logs_write(self):
        """guarded_write should log the write operation."""
        rm = ResourceManager()
        rm.reserve("agent1", ["file.py"])
        guard = FileWriteGuard(resource_manager=rm)
        
        guard.guarded_write("agent1", "file.py", b"content")
        
        assert len(guard._write_log) == 1
        assert guard._write_log[0]["agent"] == "agent1"
        assert guard._write_log[0]["path"] == "file.py"


# --- Integration tests ---

class TestReservationIntegration:
    """Integration tests for reservation and guard workflow."""

    def test_full_workflow(self):
        """Test complete reservation -> capture -> verify -> release workflow."""
        rm = ResourceManager()
        
        # Agent reserves files
        assert rm.reserve("agent1", ["src/app.py"], exclusive=True)
        
        # Capture original
        rm.capture_original("agent1", "src/app.py", "def old():\n    pass\n")
        
        # Verify diff
        diff = rm.verify_diff("agent1", "src/app.py", "def new():\n    return 1\n")
        assert len(diff) > 0
        
        # Release with verification
        result = rm.release("agent1")
        assert result["verified"] is True
        assert "src/app.py" in result["paths"]

    def test_exclusive_conflict(self):
        """Test that exclusive reservations conflict."""
        rm = ResourceManager()
        
        rm.reserve("agent1", ["shared.py"], exclusive=True)
        result = rm.reserve("agent2", ["shared.py"], exclusive=True)
        
        assert result is False

    def test_non_exclusive_no_conflict(self):
        """Test that non-exclusive reservations don't conflict."""
        rm = ResourceManager()
        
        rm.reserve("agent1", ["shared.py"], exclusive=False)
        result = rm.reserve("agent2", ["shared.py"], exclusive=False)
        
        assert result is True

    def test_storage_guard_integration(self, tmp_path):
        """Test FileWriteGuard with real storage provider."""
        rm = ResourceManager()
        storage = LocalStorageProvider(base_path=str(tmp_path))
        guard = FileWriteGuard(resource_manager=rm, storage_provider=storage)
        
        # Reserve and write
        rm.reserve("agent1", ["test.txt"])
        result = guard.guarded_write("agent1", "test.txt", b"new content")
        
        assert result["success"] is True