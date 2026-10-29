"""Users, RBAC, audit integrity and hybrid-retrieval schema.

Revision ID: 0002_security_hybrid
Revises: 0001_initial
Create Date: 2026-10-29
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0002_security_hybrid"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- Users directory (RBAC) -------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.String(length=128), nullable=False),
        sa.Column("department", sa.String(length=64), nullable=False),
        sa.Column("clearance", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column(
            "is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_department_role", "users", ["department", "role"])

    # --- Audit trail: caller role + tamper-evident hash chain ---------------
    op.add_column("audit_logs", sa.Column("role", sa.String(length=16), nullable=True))
    op.add_column(
        "audit_logs", sa.Column("entry_hash", sa.String(length=64), nullable=True)
    )
    op.create_index(
        "ix_audit_logs_entry_hash", "audit_logs", ["entry_hash"], unique=True
    )

    # --- Hybrid retrieval: stored full-text column --------------------------
    op.add_column(
        "document_chunks",
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_document_chunks_content_tsv",
        "document_chunks",
        ["content_tsv"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_chunks_content_tsv", table_name="document_chunks"
    )
    op.drop_column("document_chunks", "content_tsv")

    op.drop_index("ix_audit_logs_entry_hash", table_name="audit_logs")
    op.drop_column("audit_logs", "entry_hash")
    op.drop_column("audit_logs", "role")

    op.drop_index("ix_users_department_role", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
