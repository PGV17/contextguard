"""Chat completion request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A natural-language query executed against the permission-filtered corpus."""

    query: str = Field(
        min_length=1,
        max_length=8000,
        examples=["What is the production database password?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Optional retrieval depth; clamped to the server-side maximum.",
    )


class Citation(BaseModel):
    """A retrieved source that the model cited in its answer."""

    marker: str = Field(examples=["[Doc 1]"])
    chunk_id: int
    document_id: str
    source_url: str | None = None
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: int
    model: str
