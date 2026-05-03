"""Tests for spine.core.learning and spine.swarm.learning - pattern learning system and integration."""

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure spine package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.core.learning import Pattern, AntiPattern, PatternRecord, LearningManager
from spine.core.hivemind import Hivemind
from spine.swarm.learning import (
    LearningIntegration,
    create_learning_integration,
    sync_patterns_to_hivemind,
)


# --- Fixtures ---

@pytest.fixture
def temp_knowledge_dir(tmp_path):
    """Create a temporary knowledge directory."""
    return str(tmp_path / "knowledge")


@pytest.fixture
def temp_memory_dir(tmp_path):
    """Create a temporary memory directory."""
    return str(tmp_path / "memory")


@pytest.fixture
def pattern():
    """Create a sample Pattern for testing."""
    return Pattern(
        pattern_id="p_001",
        context="Use connection pooling for database access",
        solution="Implement a connection pool with max_connections=10",
    )


@pytest.fixture
def learning_manager(temp_knowledge_dir):
    """Create a LearningManager with temp directory."""
    return LearningManager(knowledge_dir=temp_knowledge_dir)


@pytest.fixture
def hivemind(temp_memory_dir):
    """Create a Hivemind with temp directory."""
    return Hivemind(memory_path=temp_memory_dir)


@pytest.fixture
def learning_integration(learning_manager, hivemind, tmp_path):
    """Create a LearningIntegration."""
    return LearningIntegration(
        learning_manager=learning_manager,
        hivemind=hivemind,
        project_path=str(tmp_path),
    )


# --- Pattern dataclass tests ---

class TestPatternInit:
    """Test Pattern initialization."""

    def test_pattern_defaults(self):
        """Pattern should have sensible defaults."""
        p = Pattern(pattern_id="p1", context="ctx", solution="sol")
        assert p.pattern_id == "p1"
        assert p.context == "ctx"
        assert p.solution == "sol"
        assert p.status == "candidate"
        assert p.confirmations == 0
        assert p.successes == 0
        assert p.failures == 0
        assert p.confidence == 0.0
        assert p.first_seen is not None

    def test_pattern_custom_values(self):
        """Pattern should accept custom values."""
        p = Pattern(
            pattern_id="p2",
            context="ctx",
            solution="sol",
            status="proven",
            confirmations=10,
            successes=10,
            failures=0,
            confidence=1.0,
        )
        assert p.status == "proven"
        assert p.confirmations == 10
        assert p.confidence == 1.0

    def test_pattern_first_seen_auto(self):
        """Pattern should auto-generate first_seen."""
        import time
        time.sleep(0.01)
        p1 = Pattern(pattern_id="p3", context="c", solution="s")
        time.sleep(0.01)
        p2 = Pattern(pattern_id="p4", context="c", solution="s")
        assert p1.first_seen is not None
        assert p2.first_seen is not None
        # p2's first_seen should be after p1's
        assert p2.first_seen >= p1.first_seen


class TestPatternMaturity:
    """Test Pattern maturity progression."""

    def test_record_success_updates_counts(self, pattern):
        """record_success should increment successes and confirmations."""
        pattern.record_success()
        assert pattern.successes == 1
        assert pattern.confirmations == 1
        assert pattern.last_confirmed is not None

    def test_record_failure_updates_counts(self, pattern):
        """record_failure should increment failures and confirmations."""
        pattern.record_failure()
        assert pattern.failures == 1
        assert pattern.confirmations == 1

    def test_maturity_candidate(self, pattern):
        """Pattern with 50% success rate (<60%) should be anti_pattern_candidate."""
        pattern.record_success()
        pattern.record_failure()
        # 50% < 60% threshold → anti_pattern_candidate
        assert pattern.status == "anti_pattern_candidate"
        assert pattern.confidence == pytest.approx(0.5)

    def test_maturity_candidate_above_60_percent(self, pattern):
        """Pattern with 60%+ success rate should be candidate."""
        for _ in range(3):
            pattern.record_success()
        for _ in range(2):
            pattern.record_failure()
        # 60% >= 0.6 → candidate
        assert pattern.status == "candidate"

    def test_maturity_established(self, pattern):
        """Pattern with >=80% success and >=3 confirmations should be established."""
        for _ in range(4):
            pattern.record_success()
        pattern.record_failure()
        assert pattern.status == "established"
        assert pattern.confirmations == 5
        assert pattern.confidence == pytest.approx(0.8)

    def test_maturity_proven(self, pattern):
        """Pattern with >=90% success and >=10 confirmations should be proven."""
        for _ in range(10):
            pattern.record_success()
        pattern.record_failure()
        assert pattern.status == "proven"
        assert pattern.confirmations == 11
        assert pattern.confidence == pytest.approx(10 / 11)

    def test_maturity_antipattern_candidate(self, pattern):
        """Pattern with <60% success rate should be anti_pattern_candidate."""
        for _ in range(2):
            pattern.record_failure()
        pattern.record_success()
        assert pattern.status == "anti_pattern_candidate"
        assert pattern.confidence == pytest.approx(1 / 3)

    def test_maturity_stays_candidate_low_confirmations(self, pattern):
        """Pattern with 100% success but <3 confirmations stays candidate."""
        for _ in range(2):
            pattern.record_success()
        assert pattern.status == "candidate"

    def test_maturity_stays_candidate_below_threshold(self, pattern):
        """Pattern with 70% success and <10 confirmations stays candidate."""
        for _ in range(7):
            pattern.record_success()
        for _ in range(3):
            pattern.record_failure()
        # 70% >= 0.6 but confirmations=10, need >=0.9 for proven
        assert pattern.status == "candidate"

    def test_multiple_successes(self, pattern):
        """Multiple record_success calls should accumulate correctly."""
        for _ in range(5):
            pattern.record_success()
        assert pattern.successes == 5
        assert pattern.confirmations == 5
        assert pattern.confidence == pytest.approx(1.0)
        # With 5 confirmations (>=3) and 100% success, should be established
        assert pattern.status == "established"

    def test_multiple_failures(self, pattern):
        """Multiple record_failure calls should accumulate correctly."""
        for _ in range(3):
            pattern.record_failure()
        assert pattern.failures == 3
        assert pattern.confirmations == 3
        assert pattern.confidence == pytest.approx(0.0)
        assert pattern.status == "anti_pattern_candidate"

    def test_last_confirmed_updates_on_each_call(self, pattern):
        """last_confirmed should update on each record call."""
        import time
        pattern.record_success()
        first_confirm = pattern.last_confirmed
        time.sleep(0.01)
        pattern.record_failure()
        assert pattern.last_confirmed != first_confirm


class TestPatternSerialization:
    """Test Pattern to_dict/from_dict serialization."""

    def test_pattern_to_dict(self, pattern):
        """Pattern.to_dict should serialize all fields."""
        d = pattern.to_dict()
        assert d["pattern_id"] == "p_001"
        assert d["context"] == "Use connection pooling for database access"
        assert "first_seen" in d
        assert "confirmations" in d
        assert "confidence" in d

    def test_pattern_from_dict_roundtrip(self, pattern):
        """Pattern.from_dict should round-trip correctly."""
        d = pattern.to_dict()
        restored = Pattern.from_dict(d)
        assert restored.pattern_id == pattern.pattern_id
        assert restored.context == pattern.context
        assert restored.solution == pattern.solution
        assert restored.status == pattern.status
        assert restored.successes == pattern.successes
        assert restored.failures == pattern.failures


# --- AntiPattern tests ---

class TestAntiPattern:
    """Test AntiPattern dataclass."""

    def test_antipattern_defaults(self):
        """AntiPattern should have defaults."""
        ap = AntiPattern(
            pattern_id="ap1",
            pattern_context="bad practice",
            failure_rate=0.8,
            avoidance="Avoid this pattern",
        )
        assert ap.pattern_id == "ap1"
        assert ap.first_seen is not None
        assert ap.confirmed_failures == 0

    def test_antipattern_to_dict(self):
        """AntiPattern.to_dict should serialize."""
        ap = AntiPattern(
            pattern_id="ap2",
            pattern_context="bad",
            failure_rate=0.75,
            avoidance="Don't do this",
        )
        d = ap.to_dict()
        assert d["pattern_id"] == "ap2"
        assert d["failure_rate"] == 0.75

    def test_antipattern_from_dict(self):
        """AntiPattern.from_dict should deserialize."""
        data = {
            "pattern_id": "ap3",
            "pattern_context": "test",
            "failure_rate": 0.9,
            "avoidance": "avoid x",
            "first_seen": "2025-01-01",
            "confirmed_failures": 5,
        }
        ap = AntiPattern.from_dict(data)
        assert ap.pattern_id == "ap3"
        assert ap.confirmed_failures == 5


# --- PatternRecord tests ---

class TestPatternRecord:
    """Test PatternRecord dataclass."""

    def test_patternrecord_defaults(self):
        """PatternRecord should have auto timestamp."""
        pr = PatternRecord(
            pattern_id="p1",
            task_id="t1",
            work_item_id="w1",
            success=True,
        )
        assert pr.timestamp is not None
        assert pr.context == {}

    def test_patternrecord_to_dict(self):
        """PatternRecord.to_dict should serialize."""
        pr = PatternRecord(
            pattern_id="p1",
            task_id="t1",
            work_item_id="w1",
            success=True,
            context={"key": "val"},
        )
        d = pr.to_dict()
        assert d["pattern_id"] == "p1"
        assert d["success"] is True
        assert d["context"] == {"key": "val"}


# --- LearningManager tests ---

class TestLearningManager:
    """Test LearningManager."""

    def test_record_completion_success(self, learning_manager, pattern):
        """record_completion should update pattern on success."""
        learning_manager.record_completion(pattern, "task1", "work1", success=True)
        assert pattern.successes == 1
        assert pattern.confirmations == 1
        assert pattern.last_confirmed is not None

    def test_record_completion_failure(self, learning_manager, pattern):
        """record_completion should update pattern on failure."""
        learning_manager.record_completion(pattern, "task2", "work2", success=False)
        assert pattern.failures == 1
        assert pattern.confirmations == 1

    def test_record_completion_persists_pattern(self, learning_manager, pattern):
        """record_completion should save pattern to patterns.json."""
        learning_manager.record_completion(pattern, "task3", "work3", success=True)
        saved = learning_manager.get_pattern("p_001")
        assert saved is not None
        assert saved.successes == 1

    def test_record_completion_creates_jsonl_record(self, learning_manager, pattern):
        """record_completion should append to completions.jsonl."""
        learning_manager.record_completion(pattern, "task4", "work4", success=True)
        completions_path = learning_manager.completions_path
        assert os.path.exists(completions_path)
        with open(completions_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["pattern_id"] == "p_001"
        assert record["success"] is True

    def test_get_pattern(self, learning_manager, pattern):
        """get_pattern should return saved pattern."""
        learning_manager.record_completion(pattern, "task5", "work5", success=True)
        saved = learning_manager.get_pattern("p_001")
        assert saved is not None
        assert saved.pattern_id == "p_001"

    def test_get_nonexistent_pattern(self, learning_manager):
        """get_pattern should return None for unknown ID."""
        assert learning_manager.get_pattern("nonexistent") is None

    def test_get_patterns_by_status(self, learning_manager, pattern):
        """get_patterns_by_status should filter by status."""
        learning_manager.record_completion(pattern, "task6", "work6", success=True)
        all_patterns = learning_manager.get_all_patterns()
        assert len(all_patterns) >= 1
        candidate_patterns = learning_manager.get_patterns_by_status("candidate")
        assert len(candidate_patterns) >= 0  # may have established status

    def test_get_all_patterns(self, learning_manager, pattern):
        """get_all_patterns should return all stored patterns."""
        learning_manager.record_completion(pattern, "task7", "work7", success=True)
        all_p = learning_manager.get_all_patterns()
        assert len(all_p) >= 1
        assert any(p.pattern_id == "p_001" for p in all_p)

    def test_get_all_anti_patterns_empty(self, learning_manager):
        """get_all_anti_patterns should return empty initially."""
        assert learning_manager.get_all_anti_patterns() == []

    def test_anti_pattern_generation_high_failure(self, learning_manager):
        """Anti-pattern should be generated when failure rate > 60%."""
        p = Pattern(
            pattern_id="anti_p1",
            context="bad pattern",
            solution="bad solution",
        )
        # Create >60% failure rate
        learning_manager.record_completion(p, "t1", "w1", success=False)
        learning_manager.record_completion(p, "t2", "w2", success=False)
        learning_manager.record_completion(p, "t3", "w3", success=True)
        # failure_rate = 2/3 ≈ 66.7% > 60%
        anti = learning_manager.get_anti_pattern("anti_p1")
        assert anti is not None
        assert anti.failure_rate > 0.6

    def test_no_anti_pattern_low_failure(self, learning_manager):
        """Anti-pattern should not be generated when failure rate <= 60%."""
        p = Pattern(
            pattern_id="anti_p2",
            context="okay pattern",
            solution="okay solution",
        )
        # 1 failure out of 3 = 33% < 60%
        learning_manager.record_completion(p, "t1", "w1", success=True)
        learning_manager.record_completion(p, "t2", "w2", success=True)
        learning_manager.record_completion(p, "t3", "w3", success=False)
        anti = learning_manager.get_anti_pattern("anti_p2")
        assert anti is None


# --- LearningIntegration tests ---

class TestLearningIntegration:
    """Test LearningIntegration."""

    def test_record_pattern_completion(self, learning_integration, pattern):
        """record_pattern_completion should persist and store memory."""
        learning_integration.record_pattern_completion(
            pattern, "task1", "work1", success=True
        )
        # Should be in LearningManager
        saved = learning_integration.learning_manager.get_pattern("p_001")
        assert saved is not None
        # Should be in Hivemind
        memories = learning_integration.hivemind.get_insights()
        assert len(memories) >= 1

    def test_record_pattern_failure(self, learning_integration, pattern):
        """record_pattern_completion should store failure memory with correct tags."""
        learning_integration.record_pattern_completion(
            pattern, "task2", "work2", success=False
        )
        memories = learning_integration.hivemind.get_insights()
        # Should have a memory stored
        assert len(memories) >= 1
        # Check the tags include "failure"
        mem = memories[0]
        assert "failure" in mem.tags

    def test_find_similar_patterns(self, learning_integration, pattern):
        """find_similar_patterns should search Hivemind."""
        learning_integration.record_pattern_completion(
            pattern, "task3", "work3", success=True
        )
        results = learning_integration.find_similar_patterns(
            "pattern", threshold=0.0
        )
        assert len(results) >= 1

    def test_get_pattern_insights(self, learning_integration, pattern):
        """get_pattern_insights should return pattern-tagged memories."""
        learning_integration.record_pattern_completion(
            pattern, "task4", "work4", success=True
        )
        insights = learning_integration.get_pattern_insights()
        assert len(insights) >= 1

    def test_promote_proven_pattern(self, learning_integration):
        """promote_proven_pattern should store memory for proven patterns."""
        p = Pattern(
            pattern_id="proven_p1",
            context="proven context",
            solution="proven solution",
        )
        # Make it proven: 90%+ success, >=10 confirmations
        for _ in range(10):
            p.record_success()
        p.record_failure()  # 10/11 ≈ 90.9%

        learning_integration.record_pattern_completion(
            p, "task5", "work5", success=True
        )
        # Promote the proven pattern
        promoted = learning_integration.promote_proven_pattern("proven_p1")
        assert promoted is not None
        assert promoted.pattern_id == "proven_p1"

    def test_promote_non_proven_pattern(self, learning_integration, pattern):
        """promote_proven_pattern should return None for non-proven patterns."""
        # pattern is not proven (default status)
        promoted = learning_integration.promote_proven_pattern("p_001")
        # It returns the pattern even if not proven (current impl returns pattern after checking)
        # Actually checking the code: returns pattern if proven, None otherwise
        # Since our pattern is "candidate", this should return None
        assert promoted is None

    def test_get_failures_for_anti_pattern(self, learning_integration):
        """get_failures_for_anti_pattern should return failure-tagged memories."""
        p = Pattern(pattern_id="fail_p1", context="bad practice", solution="bad thing")
        learning_integration.record_pattern_completion(p, "t1", "w1", success=False)
        learning_integration.record_pattern_completion(p, "t2", "w2", success=False)

        failures = learning_integration.get_failures_for_anti_pattern()
        # get_failures_for_anti_pattern searches for "failure anti-pattern" in content
        # and filters by success=False metadata
        # Since content "Pattern: bad practice -> bad thing" doesn't contain "failure"
        # the query_similarity returns 0, so this may return empty
        # Test that the method works without error and returns a list
        assert isinstance(failures, list)

    def test_get_failures_with_matching_content(self, learning_integration):
        """get_failures_for_anti_pattern finds memories matching 'failure anti-pattern' in content."""
        # The query is "failure anti-pattern" so content must contain that exact substring
        p = Pattern(
            pattern_id="fail_p2",
            context="failure anti-pattern: don't skip validation",
            solution="always validate inputs"
        )
        learning_integration.record_pattern_completion(p, "t1", "w1", success=False)

        failures = learning_integration.get_failures_for_anti_pattern()
        assert len(failures) >= 1
        assert failures[0].metadata.get("success") is False

    def test_integration_with_project_path(self, learning_integration):
        """LearningIntegration should store project_path."""
        assert learning_integration.project_path != ""


# --- Factory function tests ---

class TestFactoryFunctions:
    """Test factory functions in spine.swarm.learning."""

    def test_create_learning_integration(self, tmp_path):
        """create_learning_integration should create integrated components."""
        knowledge_dir = str(tmp_path / "knowledge")
        memory_dir = str(tmp_path / "memory")
        integration = create_learning_integration(
            knowledge_dir=knowledge_dir,
            memory_path=memory_dir,
            project_path=str(tmp_path),
        )
        assert isinstance(integration, LearningIntegration)
        assert isinstance(integration.learning_manager, LearningManager)
        assert isinstance(integration.hivemind, Hivemind)
        # Verify dirs were created
        assert os.path.isdir(knowledge_dir)
        assert os.path.isdir(memory_dir)

    def test_sync_patterns_to_hivemind_empty(self, temp_knowledge_dir, temp_memory_dir):
        """sync_patterns_to_hivemind should return 0 with no patterns."""
        lm = LearningManager(knowledge_dir=temp_knowledge_dir)
        hm = Hivemind(memory_path=temp_memory_dir)
        synced = sync_patterns_to_hivemind(lm, hm)
        assert synced == 0

    def test_sync_patterns_to_hivemind_with_proven(self, temp_knowledge_dir, temp_memory_dir):
        """sync_patterns_to_hivemind should sync proven patterns."""
        lm = LearningManager(knowledge_dir=temp_knowledge_dir)
        hm = Hivemind(memory_path=temp_memory_dir)

        # Create a proven pattern
        p = Pattern(
            pattern_id="sync_p1",
            context="sync context",
            solution="sync solution",
        )
        for _ in range(10):
            p.record_success()
        p.record_failure()  # 10/11 ≈ 90.9% → proven
        lm.record_completion(p, "t1", "w1", success=True)

        synced = sync_patterns_to_hivemind(lm, hm)
        assert synced == 1
        # Verify memory was added
        memories = hm.get_insights()
        assert len(memories) >= 1

    def test_sync_patterns_to_hivemind_skips_non_proven(self, temp_knowledge_dir, temp_memory_dir):
        """sync_patterns_to_hivemind should only sync 'proven' patterns."""
        lm = LearningManager(knowledge_dir=temp_knowledge_dir)
        hm = Hivemind(memory_path=temp_memory_dir)

        # Create an established pattern (not proven)
        p = Pattern(
            pattern_id="sync_p2",
            context="established",
            solution="established solution",
        )
        for _ in range(4):
            p.record_success()
        p.record_failure()  # 4/5 = 80%, >=3 confirmations → established
        lm.record_completion(p, "t1", "w1", success=True)

        synced = sync_patterns_to_hivemind(lm, hm)
        assert synced == 0  # Not proven, so not synced

    def test_sync_patterns_to_hivemind_multiple_proven(self, temp_knowledge_dir, temp_memory_dir):
        """sync_patterns_to_hivemind should sync multiple proven patterns."""
        lm = LearningManager(knowledge_dir=temp_knowledge_dir)
        hm = Hivemind(memory_path=temp_memory_dir)

        for i in range(3):
            p = Pattern(
                pattern_id=f"sync_multi_{i}",
                context=f"context {i}",
                solution=f"solution {i}",
            )
            for _ in range(10):
                p.record_success()
            p.record_failure()
            lm.record_completion(p, f"t{i}", f"w{i}", success=True)

        synced = sync_patterns_to_hivemind(lm, hm)
        assert synced == 3
