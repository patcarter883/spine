"""SPINE Vector Store - sqlite-vec backed vector storage and similarity search.

Provides async methods for storing code chunks with embeddings and 
retrieving similar chunks via cosine similarity.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Reciprocal-rank-fusion constant. 60 is the value from the original RRF
# paper (Cormack et al.) and the de-facto default; it damps the influence
# of any single list's top ranks so the two retrievers genuinely combine.
_RRF_K = 60

# Token pattern for turning a natural-language query into an FTS5 MATCH
# expression. unicode61 (the FTS tokenizer) splits on non-alphanumerics,
# so ``recall_tool`` and ``countTokens`` index as their word pieces; we
# mirror that by extracting alphanumeric runs from the query.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# FTS5 stop-ish tokens — ubiquitous query filler that matches everything
# and only adds noise to BM25 scoring.
_FTS_STOP: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "how", "what", "where", "does", "do",
    "of", "to", "in", "and", "or", "for", "with", "this", "that", "it",
    "currently", "used", "use",
})


class VectorStore:
    """Manages vector storage and similarity search using sqlite-vec.

    The store uses two tables:
    - symbol_metadata: stores file paths, symbol names, summaries, and raw code
    - symbol_vectors: virtual table for vector embeddings; its dimension is
      whatever the configured embedding model emits (768 for
      nomic-embed-text-v2, 4096 for Qwen3-Embedding-8B, etc.). The dimension
      is auto-detected from the live model at index time and the vec0 table
      is recreated if the model (and thus the dimension) changes.

    Attributes:
        db_path: Path to the SQLite database file.
    """

    # Fallback default only — the real dimension comes from the configured
    # model (via ``embedding_dim`` / probe) and from any existing vec0 table.
    EMBEDDING_DIM = 768

    def __init__(self, db_path: str = ".spine/spine.db", embedding_dim: int | None = None) -> None:
        """Initialize the vector store.

        Args:
            db_path: Path to the SQLite database file.
            embedding_dim: Expected embedding dimension. When provided (e.g.
                probed from the live model by the indexer), a fresh vec0 table
                is created at this dimension and an existing *empty* table of a
                different dimension is recreated to match. When ``None`` the
                store adopts whatever an existing table declares, falling back
                to :attr:`EMBEDDING_DIM`.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._dim_override = embedding_dim
        # Authoritative dimension of the live vec0 table; set in ensure_schema.
        self.embedding_dim = embedding_dim or self.EMBEDDING_DIM

    def set_embedding_dim(self, dim: int) -> None:
        """Set the expected embedding dimension before :meth:`ensure_schema`.

        Used by the indexer after probing the live model so the vec0 table
        is (re)created at the right width.
        """
        self._dim_override = dim
        self.embedding_dim = dim

    @staticmethod
    def _existing_vec_dim(conn: sqlite3.Connection) -> int | None:
        """Return the dimension declared by an existing symbol_vectors table."""
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'symbol_vectors'"
        ).fetchone()
        if not row or not row[0]:
            return None
        m = re.search(r"FLOAT\s*\[\s*(\d+)\s*\]", row[0])
        return int(m.group(1)) if m else None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the database connection with vec extension loaded."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.enable_load_extension(True)
            # Load sqlite-vec extension
            try:
                # First try bundled 'vec0' (newer sqlite-vec with Python 3.11+)
                self._conn.execute("SELECT load_extension('vec0')")
            except sqlite3.OperationalError:
                # Fallback: use sqlite_vec.load() helper
                try:
                    import sqlite_vec

                    self._conn.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
                except (ImportError, AttributeError) as e:
                    logger.error("Failed to load sqlite-vec extension: %s", e)
                    raise RuntimeError(
                        "sqlite-vec extension not available. Ensure sqlite-vec "
                        "is installed via pip install sqlite-vec"
                    ) from e
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def ensure_schema(self) -> None:
        """Ensure the vector store tables exist.

        Creates the tables if they don't exist. Should be called once
        during initialization. Idempotently migrates existing databases
        by adding the ``lang`` column when absent.
        """
        conn = self._get_connection()

        # Create metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS symbol_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                enriched_summary TEXT NOT NULL,
                raw_code TEXT NOT NULL,
                lang TEXT NOT NULL DEFAULT 'python',
                needs_enrichment BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrate older databases that pre-date the lang column.
        try:
            conn.execute(
                "ALTER TABLE symbol_metadata ADD COLUMN lang TEXT NOT NULL DEFAULT 'python'"
            )
        except sqlite3.OperationalError:
            # Column already exists — no-op.
            pass

        # Vector table (sqlite-vec vec0 virtual table). Its dimension must
        # match the embedding model. Resolve the target dimension from, in
        # order: an explicit override (probed from the live model), an
        # existing table's declared dimension, then the fallback default.
        existing_dim = self._existing_vec_dim(conn)
        target_dim = self._dim_override or existing_dim or self.EMBEDDING_DIM

        if existing_dim is not None and existing_dim != target_dim:
            vec_rows = conn.execute("SELECT COUNT(*) FROM symbol_vectors").fetchone()[0]
            if vec_rows == 0:
                # Safe to recreate at the new dimension (e.g. model swap
                # after a --wipe). A populated table is never silently
                # dropped — that would lose data; require an explicit wipe.
                conn.execute("DROP TABLE IF EXISTS symbol_vectors")
                logger.info(
                    "Recreating symbol_vectors: dim %d → %d", existing_dim, target_dim
                )
            else:
                logger.warning(
                    "symbol_vectors dim %d != expected %d but table has %d rows — "
                    "keeping existing dim. Run `spine index --wipe` to switch models.",
                    existing_dim, target_dim, vec_rows,
                )
                target_dim = existing_dim

        self.embedding_dim = target_dim
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS symbol_vectors
            USING vec0(embedding FLOAT[{target_dim}])
        """)

        # Create indexes for filtering
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_metadata_file_path ON symbol_metadata(file_path)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_metadata_symbol_type ON symbol_metadata(symbol_type)"
        )

        # Lexical (BM25) side of hybrid retrieval. External-content FTS5
        # table mirroring symbol_metadata — no data duplication; FTS reads
        # the indexed columns by rowid. Triggers keep it in sync across the
        # only two write paths (insert / delete_all). The local embedding
        # model produces a collapsed vector space (mean pairwise cosine
        # ~0.58), so lexical match on identifiers is what actually rescues
        # exact-symbol queries — see tests/recall_eval baseline.
        # Porter stemming on top of unicode61 so NL queries match code
        # morphologically: "budgeting"→budget, "counted"→count,
        # "implemented"→implement. Without it, lexical recall misses the
        # obvious symbol whenever the query inflects the identifier.
        existing_fts_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'symbol_fts'"
        ).fetchone()
        if existing_fts_sql and "porter" not in (existing_fts_sql[0] or ""):
            # Pre-porter table — drop and recreate (tokenizer is fixed at
            # create time; the backfill below repopulates it).
            conn.execute("DROP TABLE IF EXISTS symbol_fts")
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS symbol_fts USING fts5(
                symbol_name, file_path, raw_code,
                content='symbol_metadata', content_rowid='id',
                tokenize='porter unicode61'
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS symbol_metadata_ai
            AFTER INSERT ON symbol_metadata BEGIN
                INSERT INTO symbol_fts(rowid, symbol_name, file_path, raw_code)
                VALUES (new.id, new.symbol_name, new.file_path, new.raw_code);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS symbol_metadata_ad
            AFTER DELETE ON symbol_metadata BEGIN
                INSERT INTO symbol_fts(symbol_fts, rowid, symbol_name, file_path, raw_code)
                VALUES ('delete', old.id, old.symbol_name, old.file_path, old.raw_code);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS symbol_metadata_au
            AFTER UPDATE ON symbol_metadata BEGIN
                INSERT INTO symbol_fts(symbol_fts, rowid, symbol_name, file_path, raw_code)
                VALUES ('delete', old.id, old.symbol_name, old.file_path, old.raw_code);
                INSERT INTO symbol_fts(rowid, symbol_name, file_path, raw_code)
                VALUES (new.id, new.symbol_name, new.file_path, new.raw_code);
            END
        """)

        conn.commit()

        # Backfill the FTS index for stores populated before the lexical
        # table existed (the triggers only fire on new writes). 'rebuild'
        # is the documented external-content FTS5 repopulate command.
        # NOTE: ``COUNT(*) FROM symbol_fts`` reads the *content* table, so
        # it can't tell an empty index from a full one. The actual count
        # of indexed documents lives in the ``symbol_fts_docsize`` shadow
        # table (0 when the index is empty, N when built).
        meta_n = conn.execute("SELECT COUNT(*) FROM symbol_metadata").fetchone()[0]
        try:
            indexed_n = conn.execute("SELECT COUNT(*) FROM symbol_fts_docsize").fetchone()[0]
        except sqlite3.OperationalError:
            indexed_n = 0
        if meta_n and indexed_n != meta_n:
            conn.execute("INSERT INTO symbol_fts(symbol_fts) VALUES ('rebuild')")
            conn.commit()
            logger.info("Rebuilt FTS index: %d docs", meta_n)

        logger.info("Vector store schema ensured at %s", self._db_path)

    def insert(
        self,
        file_path: str,
        symbol_name: str,
        symbol_type: str,
        enriched_summary: str,
        raw_code: str,
        embedding: np.ndarray,
        needs_enrichment: bool = False,
        lang: str = "python",
    ) -> int:
        """Insert a symbol chunk with its embedding.

        Args:
            file_path: Path to the source file.
            symbol_name: Name of the function/class/symbol.
            symbol_type: Type of symbol (function, class, etc.).
            enriched_summary: Natural language summary of the code.
            raw_code: The raw source code.
            embedding: The embedding vector as numpy array.
            needs_enrichment: Flag for failed summarization.
            lang: Source language (``python``, ``php``, ``typescript``).

        Returns:
            The ID of the inserted row.

        Raises:
            ValueError: if the embedding width does not match the vec0 table.
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim {embedding.shape[0]} != store dim "
                f"{self.embedding_dim} for {symbol_name!r}. The vector store and "
                f"embedding model disagree — run `spine index --wipe` after a "
                f"model swap."
            )
        conn = self._get_connection()

        cursor = conn.execute(
            """
            INSERT INTO symbol_metadata (file_path, symbol_name, symbol_type, enriched_summary, raw_code, needs_enrichment, lang)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (file_path, symbol_name, symbol_type, enriched_summary, raw_code, needs_enrichment, lang),
        )
        row_id = cursor.lastrowid

        # Insert into vector table
        # Convert embedding to bytes (float32)
        embedding_bytes = embedding.astype(np.float32).tobytes()
        conn.execute(
            "INSERT INTO symbol_vectors (rowid, embedding) VALUES (?, ?)",
            (row_id, embedding_bytes),
        )

        conn.commit()
        return row_id

    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_by_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks using cosine similarity.

        Args:
            query_embedding: The query embedding as numpy array.
            k: Number of results to return.
            filter_by_types: Optional list of symbol types to filter by.

        Returns:
            List of dicts with file_path, symbol_name, symbol_type,
            enriched_summary, raw_code, and similarity score.
        """
        conn = self._get_connection()

        # Convert query embedding to bytes
        query_bytes = query_embedding.astype(np.float32).tobytes()

        # Build query with optional filter
        type_filter = ""
        params: list[Any] = [query_bytes, k]

        if filter_by_types:
            placeholders = ", ".join("?" for _ in filter_by_types)
            type_filter = f"WHERE symbol_type IN ({placeholders})"
            params[1:1] = filter_by_types

        query = f"""
            SELECT
                m.file_path,
                m.symbol_name,
                m.symbol_type,
                m.enriched_summary,
                m.raw_code,
                m.lang,
                1.0 - vec_distance_cosine(v.embedding, ?) as similarity
            FROM symbol_vectors v
            JOIN symbol_metadata m ON v.rowid = m.id
            {type_filter}
            ORDER BY similarity DESC
            LIMIT ?
        """

        cursor = conn.execute(query, params)
        results = []
        for row in cursor:
            results.append(
                {
                    "file_path": row["file_path"],
                    "symbol_name": row["symbol_name"],
                    "symbol_type": row["symbol_type"],
                    "enriched_summary": row["enriched_summary"],
                    "raw_code": row["raw_code"],
                    "lang": row["lang"],
                    "similarity": float(row["similarity"]),
                }
            )

        return results

    @staticmethod
    def _fts_match_expr(query_text: str) -> str | None:
        """Turn a natural-language query into a safe FTS5 OR-MATCH expression.

        Extracts alphanumeric tokens (mirroring the unicode61 tokenizer),
        drops filler stop-words, and OR-joins the rest as quoted terms.
        Returns ``None`` when nothing usable remains, so callers can skip
        the lexical search rather than issue a syntactically-invalid MATCH.
        """
        tokens = [t.lower() for t in _WORD_RE.findall(query_text or "")]
        tokens = [t for t in tokens if len(t) > 1 and t not in _FTS_STOP]
        if not tokens:
            return None
        # De-dup preserving order; quote each term to neutralize any FTS
        # operator characters and force literal token matching.
        seen: set[str] = set()
        terms = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                terms.append(f'"{t}"')
        return " OR ".join(terms)

    def search_bm25(self, query_text: str, k: int = 10) -> list[dict[str, Any]]:
        """Lexical BM25 search over symbol name, file path, and raw code.

        Symbol name and file path are weighted above raw code so a query
        that names a symbol or file ranks the defining chunk first. Returns
        the same payload shape as :meth:`search_similar`, with ``similarity``
        carrying the (negated) BM25 score for reference (higher = better).
        """
        match = self._fts_match_expr(query_text)
        if not match:
            return []
        conn = self._get_connection()
        # bm25() is ascending (lower = better); negate so higher = better,
        # consistent with the cosine ``similarity`` field.
        sql = """
            SELECT
                m.file_path, m.symbol_name, m.symbol_type,
                m.enriched_summary, m.raw_code, m.lang,
                -bm25(symbol_fts, 10.0, 5.0, 1.0) AS score
            FROM symbol_fts
            JOIN symbol_metadata m ON symbol_fts.rowid = m.id
            WHERE symbol_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?
        """
        try:
            cursor = conn.execute(sql, (match, k))
        except sqlite3.OperationalError as exc:
            logger.warning("BM25 search failed for %r: %s", match, exc)
            return []
        results = []
        for row in cursor:
            results.append(
                {
                    "file_path": row["file_path"],
                    "symbol_name": row["symbol_name"],
                    "symbol_type": row["symbol_type"],
                    "enriched_summary": row["enriched_summary"],
                    "raw_code": row["raw_code"],
                    "lang": row["lang"],
                    "similarity": float(row["score"]),
                }
            )
        return results

    def search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 10,
        pool: int = 50,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Fuse vector and BM25 results via reciprocal rank fusion (RRF).

        Each retriever contributes its top ``pool`` results; an item's
        fused score is ``sum(weight / (RRF_K + rank))`` over the lists it
        appears in (rank is 1-based per list). This is rank-based, so the
        two retrievers' incomparable score scales never need normalizing,
        and a result strong in *either* channel surfaces — which is the
        whole point when the vector space is weak but lexical match is
        sharp (or vice-versa).

        ``vector_weight`` / ``bm25_weight`` let a caller down-weight a
        channel that is known-weak. With the current local embedding model
        the vector space is near-random for many code queries, so weighting
        BM25 above vectors keeps lexical-exact hits from being diluted by
        vector noise (see tests/recall_eval).

        Args:
            query_embedding: Embedding of the (instruction-prefixed) query.
            query_text: The raw natural-language query, for BM25.
            k: Number of fused results to return.
            pool: Per-retriever candidate depth before fusion.
            vector_weight: RRF weight for the vector channel.
            bm25_weight: RRF weight for the lexical channel.

        Returns:
            Up to ``k`` result dicts (same shape as :meth:`search_similar`)
            ordered by fused score, which is carried in ``rrf_score``.
        """
        vec_hits = self.search_similar(query_embedding, k=pool)
        bm_hits = self.search_bm25(query_text, k=pool)

        def key(h: dict) -> tuple[str, str]:
            return (h["file_path"], h["symbol_name"])

        fused: dict[tuple[str, str], dict[str, Any]] = {}
        scores: dict[tuple[str, str], float] = {}
        for hits, weight in ((vec_hits, vector_weight), (bm_hits, bm25_weight)):
            for rank, h in enumerate(hits, 1):
                kk = key(h)
                scores[kk] = scores.get(kk, 0.0) + weight / (_RRF_K + rank)
                # First writer wins the payload; BM25 hits still carry full
                # metadata, so a vector-missed item keeps its fields.
                fused.setdefault(kk, h)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for kk, s in ranked[:k]:
            item = dict(fused[kk])
            item["rrf_score"] = float(s)
            out.append(item)
        return out

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dict with total_chunks, needs_enrichment_count, etc.
        """
        conn = self._get_connection()

        total = conn.execute("SELECT COUNT(*) as count FROM symbol_metadata").fetchone()
        needs_enrich = conn.execute(
            "SELECT COUNT(*) as count FROM symbol_metadata WHERE needs_enrichment = 1"
        ).fetchone()

        return {
            "total_chunks": total["count"] if total else 0,
            "needs_enrichment_count": needs_enrich["count"] if needs_enrich else 0,
            "embedding_dimension": self._existing_vec_dim(conn) or self.embedding_dim,
        }

    def mark_needs_enrichment(self, row_id: int) -> None:
        """Mark a chunk as needing re-enrichment.

        Args:
            row_id: The ID of the row to update.
        """
        conn = self._get_connection()
        conn.execute(
            "UPDATE symbol_metadata SET needs_enrichment = 1 WHERE id = ?",
            (row_id,),
        )
        conn.commit()

    def delete_all(self) -> None:
        """Delete all vectors and metadata from the store."""
        conn = self._get_connection()
        conn.execute("DELETE FROM symbol_vectors")
        conn.execute("DELETE FROM symbol_metadata")
        conn.commit()
        logger.info("Vector store cleared")