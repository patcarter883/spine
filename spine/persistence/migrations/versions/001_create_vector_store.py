"""Initial migration - create vector store table.

Revision ID: 001_create_vector_store
Revises: 
Create Date: 2026-05-25

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "001_create_vector_store"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the symbol_vectors table with sqlite-vec virtual table support."""
    # Create the main table for metadata
    op.create_table(
        "symbol_metadata",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column("symbol_name", sa.Text, nullable=False),
        sa.Column("symbol_type", sa.Text, nullable=False),
        sa.Column("enriched_summary", sa.Text, nullable=False),
        sa.Column("raw_code", sa.Text, nullable=False),
        sa.Column("needs_enrichment", sa.Boolean, default=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    
    # Create indexes for filtering
    op.create_index("idx_symbol_metadata_file_path", "symbol_metadata", ["file_path"])
    op.create_index("idx_symbol_metadata_symbol_type", "symbol_metadata", ["symbol_type"])
    
    # Create virtual table for vector storage using sqlite-vec
    # Note: sqlite-vec creates virtual tables with vec0 type
    # The embedding column uses BLOB for the vector data
    op.create_table(
        "symbol_vectors",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("embedding", sa.BLOB, nullable=False),
    )
    
    # Create a foreign key relationship
    op.create_foreign_key(
        "fk_symbol_vectors_id",
        "symbol_vectors",
        "symbol_metadata",
        ["id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Drop the vector store tables."""
    op.drop_table("symbol_vectors")
    op.drop_table("symbol_metadata")