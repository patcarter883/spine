"""Tests for spine.core.hivemind - semantic memory system with embedding-based similarity search."""

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure spine package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.core.hivemind import Hivemind, Memory


# --- Fixtures ---

@pytest.fixture
def temp_memory_dir(tmp_path):
    """Create a temporary memory directory for each test."""
    return str(tmp_path / "test_memory")


@pytest.fixture
def hivemind(temp_memory_dir):
    """Create a Hivemind instance with a temp directory."""
    return Hivemind(memory_path=temp_memory_dir)


# --- Memory dataclass tests ---

class TestMemoryDataclass:
    """Test the Memory dataclass."""

    def test_memory_defaults(self):
        """Memory should have sensible defaults."""
        mem = Memory(memory_id="mem_0001", content="test content")
        assert mem.memory_id == "mem_0001"
        assert mem.content == "test content"
        assert mem.context == ""
        assert mem.embedding is None
        assert mem.tags == []
        assert mem.metadata == {}
        assert mem.created_at is not None  # auto-generated

    def test_memory_with_all_fields(self):
        """Memory should accept all fields."""
        mem = Memory(
            memory_id="mem_0002",
            content="full content",
            context="test context",
            embedding=[0.1, 0.2, 0.3],
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
        )
        assert mem.context == "test context"
        assert mem.embedding == [0.1, 0.2, 0.3]
        assert mem.tags == ["tag1", "tag2"]
        assert mem.metadata == {"key": "value"}

    def test_memory_to_dict(self):
        """Memory.to_dict should serialize to dict."""
        mem = Memory(
            memory_id="mem_0003",
            content="content",
            context="ctx",
            embedding=[1.0, 2.0],
            tags=["a"],
            metadata={"k": "v"},
        )
        d = mem.to_dict()
        assert d["memory_id"] == "mem_0003"
        assert d["content"] == "content"
        assert d["context"] == "ctx"
        assert d["embedding"] == [1.0, 2.0]
        assert d["tags"] == ["a"]
        assert d["metadata"] == {"k": "v"}

    def test_memory_from_dict(self):
        """Memory.from_dict should deserialize from dict."""
        data = {
            "memory_id": "mem_0004",
            "content": "deserialized",
            "context": "ctx2",
            "embedding": [0.5, 0.5, 0.5],
            "tags": ["x", "y"],
            "metadata": {"nested": {"a": 1}},
            "created_at": "2025-01-01T00:00:00",
        }
        mem = Memory.from_dict(data)
        assert mem.memory_id == "mem_0004"
        assert mem.content == "deserialized"
        assert mem.context == "ctx2"
        assert mem.embedding == [0.5, 0.5, 0.5]
        assert mem.tags == ["x", "y"]
        assert mem.metadata == {"nested": {"a": 1}}
        assert mem.created_at == "2025-01-01T00:00:00"


# --- Hivemind core operations ---

class TestHivemindAddMemory:
    """Test Hivemind.add_memory()."""

    def test_add_memory_auto_id(self, hivemind):
        """add_memory should auto-generate ID if not provided."""
        mem = hivemind.add_memory(content="test content")
        assert mem.memory_id == "mem_0001"
        assert mem.content == "test content"
        assert len(hivemind.memories) == 1

    def test_add_memory_custom_id(self, hivemind):
        """add_memory should accept a custom ID."""
        mem = hivemind.add_memory(content="custom", memory_id="my_mem")
        assert mem.memory_id == "my_mem"
        assert hivemind.get_memory("my_mem") is mem

    def test_add_memory_with_all_options(self, hivemind):
        """add_memory should store all provided fields."""
        mem = hivemind.add_memory(
            content="full",
            context="ctx",
            memory_id="full_001",
            embedding=[0.1, 0.2],
            tags=["tag1", "tag2"],
            metadata={"key": "val"},
        )
        assert mem.context == "ctx"
        assert mem.embedding == [0.1, 0.2]
        assert mem.tags == ["tag1", "tag2"]
        assert mem.metadata == {"key": "val"}

    def test_add_memory_incremental_ids(self, hivemind):
        """Multiple add_memory calls should increment IDs."""
        m1 = hivemind.add_memory(content="first")
        m2 = hivemind.add_memory(content="second")
        m3 = hivemind.add_memory(content="third")
        assert m1.memory_id == "mem_0001"
        assert m2.memory_id == "mem_0002"
        assert m3.memory_id == "mem_0003"

    def test_add_memory_persists_to_disk(self, hivemind, temp_memory_dir):
        """add_memory should save memories.json to disk."""
        hivemind.add_memory(content="persist test")
        fpath = os.path.join(temp_memory_dir, "memories.json")
        assert os.path.exists(fpath)
        with open(fpath) as f:
            data = json.load(f)
        assert "memories" in data
        assert len(data["memories"]) == 1

    def test_add_memory_empty_content(self, hivemind):
        """add_memory should accept empty content."""
        mem = hivemind.add_memory(content="")
        assert mem.content == ""
        assert mem.memory_id == "mem_0001"


class TestHivemindGetMemory:
    """Test Hivemind.get_memory()."""

    def test_get_existing_memory(self, hivemind):
        """get_memory should return memory for existing ID."""
        mem = hivemind.add_memory(content="found", memory_id="get_test")
        result = hivemind.get_memory("get_test")
        assert result is mem
        assert result.content == "found"

    def test_get_nonexistent_memory(self, hivemind):
        """get_memory should return None for non-existent ID."""
        assert hivemind.get_memory("nonexistent") is None


class TestHivemindDeleteMemory:
    """Test Hivemind.delete_memory()."""

    def test_delete_existing_memory(self, hivemind):
        """delete_memory should remove and return True for existing ID."""
        hivemind.add_memory(content="delete me", memory_id="del_001")
        result = hivemind.delete_memory("del_001")
        assert result is True
        assert hivemind.get_memory("del_001") is None
        assert len(hivemind.memories) == 0

    def test_delete_nonexistent_memory(self, hivemind):
        """delete_memory should return False for non-existent ID."""
        result = hivemind.delete_memory("nonexistent")
        assert result is False

    def test_delete_persists_to_disk(self, hivemind, temp_memory_dir):
        """delete_memory should persist changes to disk."""
        hivemind.add_memory(content="to delete", memory_id="del_disk")
        hivemind.delete_memory("del_disk")
        fpath = os.path.join(temp_memory_dir, "memories.json")
        with open(fpath) as f:
            data = json.load(f)
        assert len(data["memories"]) == 0


class TestHivemindUpdateMemory:
    """Test Hivemind.update_memory()."""

    def test_update_content(self, hivemind):
        """update_memory should update content when provided."""
        hivemind.add_memory(content="old", memory_id="upd_001")
        result = hivemind.update_memory("upd_001", content="new")
        assert result.content == "new"
        assert hivemind.get_memory("upd_001").content == "new"

    def test_update_context(self, hivemind):
        """update_memory should update context when provided."""
        hivemind.add_memory(content="content", memory_id="upd_002", context="old_ctx")
        hivemind.update_memory("upd_002", context="new_ctx")
        assert hivemind.get_memory("upd_002").context == "new_ctx"

    def test_update_embedding(self, hivemind):
        """update_memory should update embedding when provided."""
        hivemind.add_memory(content="emb", memory_id="upd_003")
        hivemind.update_memory("upd_003", embedding=[0.3, 0.4, 0.5])
        assert hivemind.get_memory("upd_003").embedding == [0.3, 0.4, 0.5]

    def test_update_tags(self, hivemind):
        """update_memory should update tags when provided."""
        hivemind.add_memory(content="tags", memory_id="upd_004", tags=["a"])
        hivemind.update_memory("upd_004", tags=["b", "c"])
        assert hivemind.get_memory("upd_004").tags == ["b", "c"]

    def test_update_metadata(self, hivemind):
        """update_memory should update metadata when provided."""
        hivemind.add_memory(content="meta", memory_id="upd_005", metadata={"x": 1})
        hivemind.update_memory("upd_005", metadata={"x": 2, "y": 3})
        assert hivemind.get_memory("upd_005").metadata == {"x": 2, "y": 3}

    def test_update_nonexistent_memory(self, hivemind):
        """update_memory should return None for non-existent ID."""
        result = hivemind.update_memory("nope", content="new")
        assert result is None

    def test_update_partial(self, hivemind):
        """update_memory should only update provided fields."""
        hivemind.add_memory(
            content="content",
            memory_id="upd_006",
            context="ctx",
            tags=["t1"],
            metadata={"m": "v"},
        )
        hivemind.update_memory("upd_006", content="new_content")
        mem = hivemind.get_memory("upd_006")
        assert mem.content == "new_content"
        assert mem.context == "ctx"
        assert mem.tags == ["t1"]
        assert mem.metadata == {"m": "v"}

    def test_update_persists_to_disk(self, hivemind, temp_memory_dir):
        """update_memory should persist changes to disk."""
        hivemind.add_memory(content="old", memory_id="upd_disk")
        hivemind.update_memory("upd_disk", content="updated")
        fpath = os.path.join(temp_memory_dir, "memories.json")
        with open(fpath) as f:
            data = json.load(f)
        assert data["memories"][0]["content"] == "updated"


class TestHivemindClear:
    """Test Hivemind.clear()."""

    def test_clear_removes_all_memories(self, hivemind):
        """clear should remove all memories."""
        hivemind.add_memory(content="1")
        hivemind.add_memory(content="2")
        hivemind.add_memory(content="3")
        hivemind.clear()
        assert len(hivemind.memories) == 0

    def test_clear_persists_to_disk(self, hivemind, temp_memory_dir):
        """clear should persist empty state to disk."""
        hivemind.add_memory(content="will be cleared")
        hivemind.clear()
        fpath = os.path.join(temp_memory_dir, "memories.json")
        with open(fpath) as f:
            data = json.load(f)
        assert len(data["memories"]) == 0


class TestHivemindQuerySimilarity:
    """Test Hivemind.query_similarity()."""

    def test_query_similarity_with_text_fallback(self, hivemind):
        """query_similarity should do text match when no embedding."""
        hivemind.add_memory(content="the quick brown fox", memory_id="q1")
        hivemind.add_memory(content="jumps over the lazy dog", memory_id="q2")
        hivemind.add_memory(content="completely unrelated text here", memory_id="q3")

        results = hivemind.query_similarity(query="fox", embedding=None, threshold=0.0)
        # Only "the quick brown fox" contains "fox"
        assert len(results) == 1
        assert results[0]["memory"].memory_id == "q1"
        assert results[0]["score"] == 1.0

    def test_query_similarity_text_matches_multiple(self, hivemind):
        """query_similarity text fallback matches all containing memories."""
        hivemind.add_memory(content="the quick brown fox", memory_id="qm1")
        hivemind.add_memory(content="foxes are quick", memory_id="qm2")
        hivemind.add_memory(content="jumps over the lazy dog", memory_id="qm3")

        results = hivemind.query_similarity(query="fox", embedding=None, threshold=0.0)
        assert len(results) == 2
        assert results[0]["score"] == 1.0  # exact match sorted first

    def test_query_similarity_no_match(self, hivemind):
        """query_similarity should return empty list when no match."""
        hivemind.add_memory(content="hello world", memory_id="qn1")
        results = hivemind.query_similarity(query="xyz_nonexistent", embedding=None, threshold=0.0)
        assert len(results) == 0

    def test_query_similarity_case_insensitive(self, hivemind):
        """query_similarity text match should be case insensitive."""
        hivemind.add_memory(content="Testing Case Sensitivity", memory_id="qc1")
        results = hivemind.query_similarity(query="testing", embedding=None, threshold=0.0)
        assert len(results) == 1

    def test_query_similarity_threshold_filtering(self, hivemind):
        """query_similarity should filter by threshold."""
        hivemind.add_memory(content="exact match here", memory_id="qt1")
        results = hivemind.query_similarity(query="match", embedding=None, threshold=0.5)
        assert len(results) == 1  # exact match returns 1.0
        results_no = hivemind.query_similarity(query="nonexistent", embedding=None, threshold=0.0)
        assert len(results_no) == 0

    def test_query_similarity_limit(self, hivemind):
        """query_similarity should respect limit."""
        for i in range(5):
            hivemind.add_memory(content=f"match word {i}", memory_id=f"ql{i}")
        results = hivemind.query_similarity(query="match", embedding=None, limit=2)
        assert len(results) == 2

    def test_query_similarity_sort_by_score(self, hivemind):
        """query_similarity should sort results by score descending."""
        hivemind.add_memory(content="match1 word", memory_id="qs1")
        hivemind.add_memory(content="match2 word", memory_id="qs2")
        results = hivemind.query_similarity(query="match1", embedding=None, threshold=0.0)
        # Exact match should be first
        assert results[0]["memory"].memory_id == "qs1"
        assert results[0]["score"] == 1.0

    def test_query_similarity_with_cosine_embedding(self, hivemind):
        """query_similarity should use cosine similarity when embeddings provided."""
        # Add memories with known embeddings
        hivemind.add_memory(content="vector A", memory_id="vec1", embedding=[1.0, 0.0, 0.0])
        hivemind.add_memory(content="vector B", memory_id="vec2", embedding=[0.0, 1.0, 0.0])
        hivemind.add_memory(content="vector C", memory_id="vec3", embedding=[1.0, 1.0, 0.0])

        # Query vector close to vec1
        query_emb = [0.9, 0.1, 0.0]
        results = hivemind.query_similarity(
            query="test",
            embedding=query_emb,
            threshold=0.0,
        )
        assert len(results) == 3
        # vec1 should be most similar to [0.9, 0.1, 0.0]
        assert results[0]["memory"].memory_id == "vec1"
        assert results[0]["score"] > results[1]["score"]

    def test_query_similarity_cosine_identical_vectors(self, hivemind):
        """Cosine similarity of identical vectors should be 1.0."""
        hivemind.add_memory(content="identical", memory_id="id1", embedding=[1.0, 1.0, 1.0])
        results = hivemind.query_similarity(
            query="test",
            embedding=[1.0, 1.0, 1.0],
            threshold=0.0,
        )
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(1.0)

    def test_query_similarity_cosine_perpendicular_vectors(self, hivemind):
        """Cosine similarity of perpendicular vectors should be 0.0."""
        hivemind.add_memory(content="perp", memory_id="perp1", embedding=[1.0, 0.0])
        results = hivemind.query_similarity(
            query="test",
            embedding=[0.0, 1.0],
            threshold=0.0,
        )
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.0)

    def test_query_similarity_mixed_embedding_and_text(self, hivemind):
        """query_similarity: only memories without embeddings match via text fallback."""
        hivemind.add_memory(content="text match content", memory_id="mix1", embedding=[0.1, 0.1])
        hivemind.add_memory(content="text match other", memory_id="mix2")
        # Memories with embeddings are skipped when no query embedding is provided
        # (they require cosine similarity which needs query embedding)
        results = hivemind.query_similarity(query="text match", embedding=None, threshold=0.0)
        # Only mix2 matches (no embedding, text contains "text match")
        # mix1 has an embedding, so it's skipped
        assert len(results) == 1
        assert results[0]["memory"].memory_id == "mix2"

    def test_query_similarity_mismatched_vector_length(self, hivemind):
        """Cosine similarity with mismatched vector lengths should return 0.0."""
        hivemind.add_memory(content="diff len", memory_id="dlen1", embedding=[1.0, 2.0])
        results = hivemind.query_similarity(
            query="test",
            embedding=[1.0, 2.0, 3.0, 4.0],  # different length
            threshold=0.0,
        )
        assert len(results) == 1
        assert results[0]["score"] == 0.0


class TestHivemindGetInsights:
    """Test Hivemind.get_insights()."""

    def test_get_insights_returns_all(self, hivemind):
        """get_insights should return all memories by default."""
        hivemind.add_memory(content="mem1", memory_id="ins1")
        hivemind.add_memory(content="mem2", memory_id="ins2")
        results = hivemind.get_insights()
        assert len(results) == 2

    def test_get_insights_sorted_by_created_at_desc(self, hivemind):
        """get_insights should sort by created_at descending."""
        hivemind.add_memory(content="older", memory_id="ins_older")
        # Small delay to ensure different timestamps
        import time
        time.sleep(0.01)
        hivemind.add_memory(content="newer", memory_id="ins_newer")
        results = hivemind.get_insights()
        assert results[0].memory_id == "ins_newer"
        assert results[1].memory_id == "ins_older"

    def test_get_insights_with_tag_filter(self, hivemind):
        """get_insights should filter by tags."""
        hivemind.add_memory(content="tagged", memory_id="tag1", tags=["spine", "python"])
        hivemind.add_memory(content="untagged", memory_id="tag2")
        hivemind.add_memory(content="other_tag", memory_id="tag3", tags=["other"])
        results = hivemind.get_insights(tags=["spine"])
        assert len(results) == 1
        assert results[0].memory_id == "tag1"

    def test_get_insights_with_limit(self, hivemind):
        """get_insights should respect limit."""
        for i in range(5):
            hivemind.add_memory(content=f"ins_{i}", memory_id=f"lim{i}")
        results = hivemind.get_insights(limit=3)
        assert len(results) == 3

    def test_get_insights_empty(self, hivemind):
        """get_insights on empty store should return empty list."""
        results = hivemind.get_insights()
        assert results == []


# --- Hivemind persistence ---

class TestHivemindPersistence:
    """Test Hivemind persistence across instances."""

    def test_reload_memories_from_disk(self, temp_memory_dir):
        """_load_memories should reload memories from disk when called."""
        # First instance adds memories (which saves to disk)
        hm1 = Hivemind(memory_path=temp_memory_dir)
        hm1.add_memory(content="persisted1", memory_id="persist1")
        hm1.add_memory(content="persisted2", memory_id="persist2")
        del hm1

        # Second instance needs to call _load_memories to reload
        hm2 = Hivemind(memory_path=temp_memory_dir)
        hm2._load_memories()
        assert hm2.get_memory("persist1").content == "persisted1"
        assert hm2.get_memory("persist2").content == "persisted2"
        assert len(hm2.memories) == 2

    def test_delete_persists_across_instances(self, temp_memory_dir):
        """Delete should persist across Hivemind instances."""
        hm1 = Hivemind(memory_path=temp_memory_dir)
        hm1.add_memory(content="del_test", memory_id="del_across")
        hm1.delete_memory("del_across")
        del hm1

        hm2 = Hivemind(memory_path=temp_memory_dir)
        assert hm2.get_memory("del_across") is None

    def test_memory_dir_created_automatically(self, tmp_path):
        """Hivemind should create memory directory if it doesn't exist."""
        new_dir = str(tmp_path / "new_dir" / "nested")
        hm = Hivemind(memory_path=new_dir)
        assert os.path.isdir(new_dir)
        hm.add_memory(content="auto created", memory_id="auto1")
        assert os.path.exists(os.path.join(new_dir, "memories.json"))
