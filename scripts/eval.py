"""Offline retrieval evaluation: vector-only vs hybrid vs hybrid + re-rank.

Reports recall@k and MRR (Mean Reciprocal Rank) for every retrieval mode so
pipeline changes are measured, not guessed at. Admin credentials are used
internally so every seeded document is visible regardless of department.

Prerequisite: the corpus is seeded (``python -m scripts.seed``).

Usage:
    python -m scripts.eval [--k 5] [--pool 20]
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.security import CurrentUser
from app.db.session import create_engine, create_session_factory
from app.repositories.chunks import ChunkRepository
from app.services.embeddings import EmbeddingService, create_embedding_service
from app.services.reranking import create_reranker
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

# (question, documents that must appear in the top-k)
DATASET: tuple[tuple[str, frozenset[str]], ...] = (
    ("What is the production database password?", frozenset({"eng-db-runbook"})),
    ("What is the CEO's annual bonus?", frozenset({"hr-exec-compensation"})),
    ("When is the office closed?", frozenset({"company-holiday-notice"})),
    ("When does the Q3 marketing campaign launch?", frozenset({"marketing-q3-campaign"})),
    ("How many days of paid annual leave do employees get?", frozenset({"hr-leave-policy"})),
    ("What is the admin123 password used for?", frozenset({"eng-db-runbook"})),
    ("vacation days policy", frozenset({"hr-leave-policy"})),
    ("holiday closure announcement", frozenset({"company-holiday-notice"})),
)

MODE_SPECS: tuple[tuple[str, bool, bool], ...] = (
    # (label, hybrid_enabled, reranker_enabled)
    ("vector-only", False, False),
    ("hybrid (RRF)", True, False),
    ("hybrid + cross-encoder", True, True),
)


def _mode_settings(base: Settings, *, hybrid: bool, reranker: bool) -> Settings:
    return base.model_copy(
        update={"hybrid_enabled": hybrid, "reranker_enabled": reranker}
    )


async def evaluate_mode(
    *,
    label: str,
    hybrid: bool,
    reranker_enabled: bool,
    base_settings: Settings,
    embeddings: EmbeddingService,
    admin: CurrentUser,
    k: int,
) -> dict[str, object]:
    settings = _mode_settings(base_settings, hybrid=hybrid, reranker=reranker_enabled)
    service_reranker = create_reranker(settings)

    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            service = RetrievalService(
                chunks=ChunkRepository(session),
                reranker=service_reranker,
                settings=settings,
            )
            recall_hits = 0
            reciprocal_ranks: list[float] = []

            for question, expected in DATASET:
                query_embedding = await embeddings.embed_query(question)
                matches = await service.retrieve(
                    question=question,
                    query_embedding=query_embedding,
                    user=admin,
                    limit=k,
                )
                ranks = [
                    position
                    for position, match in enumerate(matches, start=1)
                    if match.document_id in expected
                ]
                first_rank = ranks[0] if ranks else None
                if first_rank is not None:
                    recall_hits += 1
                    reciprocal_ranks.append(1.0 / first_rank)
                else:
                    reciprocal_ranks.append(0.0)
                await session.commit()  # release transaction state between queries

            return {
                "mode": label,
                "recall": recall_hits / len(DATASET),
                "mrr": sum(reciprocal_ranks) / len(DATASET),
            }
    finally:
        await engine.dispose()


async def run_evaluation(base_settings: Settings, k: int) -> list[dict[str, object]]:
    configure_logging("WARNING")
    embeddings = create_embedding_service(base_settings)
    admin = CurrentUser(username="admin", department="All", clearance=5, role="admin")

    rows: list[dict[str, object]] = []
    for label, hybrid, reranker_enabled in MODE_SPECS:
        rows.append(
            await evaluate_mode(
                label=label,
                hybrid=hybrid,
                reranker_enabled=reranker_enabled,
                base_settings=base_settings,
                embeddings=embeddings,
                admin=admin,
                k=k,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes.")
    parser.add_argument("--k", type=int, default=5, help="cutoff for recall@k / MRR")
    parser.add_argument("--pool", type=int, default=20, help="candidate pool size")
    args = parser.parse_args()

    settings = get_settings().model_copy(update={"retrieval_pool_size": args.pool})
    rows = asyncio.run(run_evaluation(settings, args.k))

    header = f"{'retrieval mode':<26}{f'recall@{args.k}':>12}{'mrr':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['mode']:<26}{row['recall']:>11.2%} {row['mrr']:>10.4f}")


if __name__ == "__main__":
    main()
