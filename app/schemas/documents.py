"""Document ingestion request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """A document to chunk, embed and store with permission metadata."""

    content: str = Field(
        min_length=1,
        examples=["The VPN concentrator is reachable at vpn.example.com."],
    )
    department_allowed: str = Field(
        min_length=1,
        max_length=64,
        examples=["Engineering"],
        description="Department allowed to read the document, or 'All' for public documents.",
    )
    clearance_level: int = Field(
        ge=1,
        le=5,
        examples=[2],
        description="Minimum clearance level required to read the document.",
    )
    source_url: str | None = Field(default=None, max_length=2048)
    document_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9._:-]+$",
        description="Optional caller-supplied document identifier (one is generated otherwise).",
    )


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    chunk_ids: list[int]
    embedding_model: str
