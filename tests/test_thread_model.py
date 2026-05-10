"""Tests for the Thread model and UUID4 thread ID generation."""

import re
import uuid
from datetime import datetime, timezone

from spine.models.thread import Thread, generate_thread_id


class TestGenerateThreadId:
    """Tests for the generate_thread_id function."""

    def test_generates_uuid4_string(self):
        thread_id = generate_thread_id()
        assert isinstance(thread_id, str)
        parts = thread_id.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

    def test_generates_valid_uuid4(self):
        thread_id = generate_thread_id()
        parsed = uuid.UUID(thread_id)
        assert parsed.version == 4

    def test_unique_ids(self):
        ids = {generate_thread_id() for _ in range(100)}
        assert len(ids) == 100

    def test_matches_uuid_pattern(self):
        thread_id = generate_thread_id()
        pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        assert re.match(pattern, thread_id), f"{thread_id} does not match UUID4 pattern"


class TestThreadModel:
    """Tests for the Thread dataclass."""

    def test_auto_generates_thread_id(self):
        thread = Thread()
        assert thread.thread_id is not None
        assert uuid.UUID(thread.thread_id).version == 4

    def test_unique_thread_ids(self):
        t1 = Thread()
        t2 = Thread()
        assert t1.thread_id != t2.thread_id

    def test_default_created_at(self):
        thread = Thread()
        assert thread.created_at is not None
        datetime.fromisoformat(thread.created_at)

    def test_default_status(self):
        thread = Thread()
        assert thread.status == "INIT"

    def test_custom_requirement(self):
        thread = Thread(requirement="Build auth module")
        assert thread.requirement == "Build auth module"

    def test_custom_thread_id(self):
        tid = str(uuid.uuid4())
        thread = Thread(thread_id=tid)
        assert thread.thread_id == tid

    def test_custom_status(self):
        thread = Thread(status="PLANNING")
        assert thread.status == "PLANNING"

    def test_to_dict(self):
        tid = str(uuid.uuid4())
        thread = Thread(thread_id=tid, requirement="test req", status="EXECUTION")
        d = thread.to_dict()
        assert d["thread_id"] == tid
        assert d["requirement"] == "test req"
        assert d["status"] == "EXECUTION"
        assert "created_at" in d
        assert "metadata" in d

    def test_from_dict(self):
        tid = str(uuid.uuid4())
        data = {
            "thread_id": tid,
            "requirement": "test",
            "status": "COMPLETE",
            "created_at": "2025-01-01T00:00:00+00:00",
            "metadata": {"key": "val"},
        }
        thread = Thread.from_dict(data)
        assert thread.thread_id == tid
        assert thread.requirement == "test"
        assert thread.status == "COMPLETE"
        assert thread.created_at == "2025-01-01T00:00:00+00:00"
        assert thread.metadata == {"key": "val"}

    def test_from_dict_empty(self):
        thread = Thread.from_dict({})
        assert thread.thread_id is not None
        assert uuid.UUID(thread.thread_id).version == 4
        assert thread.requirement == ""
        assert thread.status == "INIT"

    def test_metadata_default(self):
        thread = Thread()
        assert thread.metadata == {}
