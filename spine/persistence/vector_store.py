"""SPINE Vector Store - sqlite-vec backed vector storage and similarity search.

Provides async methods for storing code chunks with embeddings and 
retrieving similar chunks via cosine similarity.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and similarity search using sqlite-vec.

    The store uses two tables:
    - symbol_metadata: stores file paths, symbol names, summaries, and raw code
    - symbol_vectors: virtual table for vector embeddings with 3072 dimensions
      (OpenAI text-embedding-3-large)

    Attributes:
        db_path: Path to the SQLite database file.
    """

    EMBEDDING_DIM = 768  # Default, overridden by provider config or insert() call

    def __init__(self, db_path: str = ".spine/spine.db") -> None:
        """Initialize the vector store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

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
        during initialization.
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
                needs_enrichment BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create vector table using sqlite-vec's vec0 virtual table
        # The embedding column stores the vector as a BLOB
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS symbol_vectors 
            USING vec0(embedding FLOAT[{self.EMBEDDING_DIM}])
        """)

        # Create indexes for filtering
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_metadata_file_path ON symbol_metadata(file_path)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_metadata_symbol_type ON symbol_metadata(symbol_type)"
        )

        conn.commit()
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

        Returns:
            The ID of the inserted row.
        """
        conn = self._get_connection()

        cursor = conn.execute(
            """
            INSERT INTO symbol_metadata (file_path, symbol_name, symbol_type, enriched_summary, raw_code, needs_enrichment)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (file_path, symbol_name, symbol_type, enriched_summary, raw_code, needs_enrichment),
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
        filter_by_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks using cosine similarity.

        Args:
            query_embedding: The query embedding as numpy array.
            k: Number of results to return.
            filter_by_type: Optional symbol type filter.

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

        if filter_by_type:
            type_filter = "WHERE symbol_type = ?"
            params.insert(1, filter_by_type)

        query = f"""
            SELECT 
                m.file_path,
                m.symbol_name,
                m.symbol_type,
                m.enriched_summary,
                m.raw_code,
                vec_cosine_similarity(v.embedding, ?) as similarity
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
                    "similarity": float(row["similarity"]),
                }
            )

        return results

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
            "embedding_dimension": self.EMBEDDING_DIM,
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