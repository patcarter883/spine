"""Tests for thread-context propagation using context variables.

Verifies that:
- Unique thread IDs are generated and retrievable via context.
- Context is properly isolated across parallel threads.
- Each parallel work entry receives its own unique thread ID.
"""

import re
import threading
import uuid

from spine.utils.thread_context import (
    generate_thread_id,
    get_current_thread_id,
    set_current_thread_id,
    reset_thread_id,
    ensure_thread_id,
)


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


class TestThreadContext:
    """Tests for thread context variable propagation."""

    def test_get_current_default_generates_id(self):
        tid = get_current_thread_id()
        assert isinstance(tid, str)
        assert uuid.UUID(tid).version == 4

    def test_set_and_get(self):
        tid = "test-work-id"
        token = set_current_thread_id(tid)
        assert get_current_thread_id() == tid
        reset_thread_id(token)

    def test_reset_restores_previous_value(self):
        original = get_current_thread_id()
        token = set_current_thread_id("override")
        assert get_current_thread_id() == "override"
        reset_thread_id(token)
        restored = get_current_thread_id()
        assert restored == original

    def test_ensure_with_explicit_id(self):
        tid = "explicit-id"
        result = ensure_thread_id(tid)
        assert result == tid
        assert get_current_thread_id() == tid

    def test_ensure_without_arg_uses_existing(self):
        tid = "existing"
        set_current_thread_id(tid)
        result = ensure_thread_id()
        assert result == tid

    def test_consecutive_calls_in_same_context(self):
        tid1 = get_current_thread_id()
        tid2 = get_current_thread_id()
        assert tid1 == tid2

    def test_ensure_without_arg_generates_if_none(self):
        """ensure_thread_id() should generate a new ID if none is set."""
        tid = ensure_thread_id()
        assert uuid.UUID(tid).version == 4


class TestThreadContextParallel:
    """Tests for thread-ID isolation across parallel work entries."""

    def test_parallel_threads_all_have_unique_ids(self):
        results = []

        def worker():
            results.append(get_current_thread_id())

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert len(set(results)) == 20

    def test_parent_child_context_isolation(self):
        parent_id = get_current_thread_id()
        child_id = []

        def worker():
            child_id.append(get_current_thread_id())

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert parent_id != child_id[0]
        assert uuid.UUID(child_id[0]).version == 4

    def test_parallel_work_entries_dont_interfere(self):
        parent_id = get_current_thread_id()
        collected = []

        def worker():
            my_id = get_current_thread_id()
            parent_value = get_current_thread_id()
            collected.append((my_id, parent_value))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for my_id, parent_val in collected:
            assert my_id != parent_id
            assert my_id == parent_val

    def test_explicit_ids_in_parallel(self):
        thread_ids = [f"worker-{i}" for i in range(10)]
        collected = {}

        def worker(wid):
            set_current_thread_id(wid)
            collected[wid] = get_current_thread_id()

        threads = [threading.Thread(target=worker, args=(wid,)) for wid in thread_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for wid in thread_ids:
            assert collected[wid] == wid
        assert set(collected.keys()) == set(thread_ids)
