"""Tests for learning.py - pattern learning system."""

import os
import tempfile
import importlib.util

spec = importlib.util.spec_from_file_location("learning", "spine/core/learning.py")
learning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learning)

Pattern = learning.Pattern
AntiPattern = learning.AntiPattern
PatternRecord = learning.PatternRecord
LearningManager = learning.LearningManager


class TestPattern:
    def test_init_defaults(self):
        pattern = Pattern(
            pattern_id="test-auth",
            context="JWT authentication",
            solution="Use RS256 keys"
        )
        assert pattern.pattern_id == "test-auth"
        assert pattern.context == "JWT authentication"
        assert pattern.solution == "Use RS256 keys"
        assert pattern.status == "candidate"
        assert pattern.confirmations == 0
        assert pattern.successes == 0
        assert pattern.failures == 0

    def test_record_success(self):
        pattern = Pattern("test", "ctx", "sol")
        pattern.record_success()
        assert pattern.successes == 1
        assert pattern.confirmations == 1
        assert pattern.confidence == 1.0

    def test_record_failure(self):
        pattern = Pattern("test", "ctx", "sol")
        pattern.record_failure()
        assert pattern.failures == 1
        assert pattern.confirmations == 1
        assert pattern.confidence == 0.0

    def test_maturity_candidate(self):
        pattern = Pattern("test", "ctx", "sol")
        pattern.record_success()
        pattern.record_success()
        assert pattern.status == "candidate"

    def test_maturity_established(self):
        pattern = Pattern("test", "ctx", "sol")
        for _ in range(3):
            pattern.record_success()
        for _ in range(2):
            pattern.record_failure()
        assert pattern.status == "candidate"
        for _ in range(5):
            pattern.record_success()
        assert pattern.status == "established"

    def test_maturity_proven(self):
        pattern = Pattern("test", "ctx", "sol")
        for _ in range(11):
            pattern.record_success()
        assert pattern.status == "proven"


class TestAntiPattern:
    def test_init(self):
        anti = AntiPattern(
            pattern_id="bad-pattern",
            pattern_context="poor approach",
            failure_rate=0.75,
            avoidance="Do something else instead"
        )
        assert anti.pattern_id == "bad-pattern"
        assert anti.failure_rate == 0.75
        assert anti.avoidance == "Do something else instead"


class TestLearningManager:
    def test_record_completion_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LearningManager(knowledge_dir=os.path.join(tmpdir, ".spine", "knowledge"))
            pattern = Pattern("test", "ctx", "sol")
            lm.record_completion(pattern, "task-1", "work-1", True)

            loaded = lm.get_pattern("test")
            assert loaded is not None
            assert loaded.successes == 1

    def test_record_completion_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LearningManager(knowledge_dir=os.path.join(tmpdir, ".spine", "knowledge"))
            pattern = Pattern("test", "ctx", "sol")
            lm.record_completion(pattern, "task-1", "work-1", False)

            loaded = lm.get_pattern("test")
            assert loaded.failures == 1

    def test_anti_pattern_generated_above_60_percent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LearningManager(knowledge_dir=os.path.join(tmpdir, ".spine", "knowledge"))
            pattern = Pattern("fail-pattern", "ctx", "sol")

            for _ in range(7):
                lm.record_completion(pattern, f"task-{_}", "work-1", False)

            anti = lm.get_anti_pattern("fail-pattern")
            assert anti is not None
            assert anti.failure_rate > 0.6

    def test_no_anti_pattern_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LearningManager(knowledge_dir=os.path.join(tmpdir, ".spine", "knowledge"))
            pattern = Pattern("ok-pattern", "ctx", "sol")

            for _ in range(5):
                lm.record_completion(pattern, f"task-{_}", "work-1", True)
            for _ in range(3):
                lm.record_completion(pattern, f"task-f-{_}", "work-1", False)

            anti = lm.get_anti_pattern("ok-pattern")
            assert anti is None

    def test_get_patterns_by_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LearningManager(knowledge_dir=os.path.join(tmpdir, ".spine", "knowledge"))

            pattern1 = Pattern("pat-1", "ctx", "sol")
            pattern1.record_success()
            pattern1.record_success()
            lm._save_pattern(pattern1)

            pattern2 = Pattern("pat-2", "ctx", "sol")
            for _ in range(11):
                pattern2.record_success()
            lm._save_pattern(pattern2)

            candidates = lm.get_patterns_by_status("candidate")
            proven = lm.get_patterns_by_status("proven")

            assert any(p.pattern_id == "pat-1" for p in candidates)
            assert any(p.pattern_id == "pat-2" for p in proven)