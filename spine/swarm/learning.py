"""Learning integration with Hivemind for semantic pattern memory."""

from typing import Any, Optional, List

from ..core.hivemind import Hivemind, Memory
from ..core.learning import LearningManager, Pattern


class LearningIntegration:
    """Integrates LearningManager with Hivemind for semantic pattern memory."""

    def __init__(
        self,
        learning_manager: LearningManager,
        hivemind: Hivemind,
        project_path: str = ""
    ):
        self.learning_manager = learning_manager
        self.hivemind = hivemind
        self.project_path = project_path

    def record_pattern_completion(
        self,
        pattern: Pattern,
        task_id: str,
        work_item_id: str,
        success: bool,
        context: dict[str, Any] | None = None
    ) -> None:
        """Record pattern completion and store semantic memory."""
        self.learning_manager.record_completion(
            pattern, task_id, work_item_id, success, context
        )
        self._store_pattern_memory(pattern, success, context)

    def _store_pattern_memory(
        self,
        pattern: Pattern,
        success: bool,
        context: dict[str, Any] | None = None
    ) -> Memory:
        """Store pattern as semantic memory in Hivemind."""
        memory = self.hivemind.add_memory(
            content=f"Pattern: {pattern.context} -> {pattern.solution}",
            context=f"pattern_completion:{pattern.pattern_id}",
            tags=["pattern", pattern.status, "success" if success else "failure"],
            metadata={
                "pattern_id": pattern.pattern_id,
                "success": success,
                "confidence": pattern.confidence,
                "context": context or {}
            }
        )
        return memory

    def find_similar_patterns(
        self,
        query: str,
        threshold: float = 0.5,
        limit: int = 10
    ) -> List[Memory]:
        """Find similar patterns using semantic similarity."""
        results = self.hivemind.query_similarity(query, threshold=threshold, limit=limit)
        return [r["memory"] for r in results]

    def get_pattern_insights(self, limit: int = 20) -> List[Memory]:
        """Get insights from pattern memories."""
        return self.hivemind.get_insights(tags=["pattern"], limit=limit)

    def promote_proven_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Promote a proven pattern to semantic memory with high confidence."""
        pattern = self.learning_manager.get_pattern(pattern_id)
        if pattern and pattern.status == "proven":
            self._store_pattern_memory(pattern, True, {"promoted": True})
        return pattern

    def get_failures_for_anti_pattern(self) -> List[Memory]:
        """Get pattern memories that represent failures."""
        results = self.hivemind.query_similarity(
            "failure anti-pattern",
            threshold=0.3,
            limit=50
        )
        return [
            r["memory"] for r in results
            if r["memory"].metadata.get("success") is False
        ]


def create_learning_integration(
    knowledge_dir: str = ".spine/knowledge",
    memory_path: str = ".spine/memory",
    project_path: str = ""
) -> LearningIntegration:
    """Factory to create LearningIntegration with LearningManager and Hivemind."""
    learning_manager = LearningManager(knowledge_dir=knowledge_dir)
    hivemind = Hivemind(memory_path=memory_path)
    return LearningIntegration(learning_manager, hivemind, project_path)


def sync_patterns_to_hivemind(
    learning_manager: LearningManager,
    hivemind: Hivemind
) -> int:
    """Sync all proven patterns to Hivemind as semantic memories."""
    proven_patterns = learning_manager.get_patterns_by_status("proven")
    synced = 0
    for pattern in proven_patterns:
        hivemind.add_memory(
            content=f"Proven pattern: {pattern.context} -> {pattern.solution}",
            context="proven_pattern",
            tags=["pattern", "proven"],
            metadata={
                "pattern_id": pattern.pattern_id,
                "confidence": pattern.confidence,
                "successes": pattern.successes,
                "failures": pattern.failures
            }
        )
        synced += 1
    return synced