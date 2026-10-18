"""
ContextGuard - database initialization and seeding script.

Creates the pgvector extension, the tables, and seeds 3 mock documents with
embeddings from sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).

Run with:  python db_setup.py   (re-running is safe; documents are reseeded)
"""

import asyncio

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from sentence_transformers import SentenceTransformer

DB_DSN = "postgresql://postgres:password@127.0.0.1:5432/rag_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SEED_DOCUMENTS = [
    {
        "content": "The production database password is 'admin123'.",
        "department_allowed": "Engineering",
        "clearance_level": 2,
    },
    {
        "content": "The CEO's annual bonus is $500,000.",
        "department_allowed": "HR",
        "clearance_level": 2,
    },
    {
        "content": "The office is closed for the Friday public holiday.",
        "department_allowed": "All",
        "clearance_level": 1,
    },
]

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id                SERIAL PRIMARY KEY,
    content           TEXT NOT NULL,
    embedding         VECTOR(384) NOT NULL,
    department_allowed VARCHAR(50) NOT NULL,
    clearance_level   INT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id             SERIAL PRIMARY KEY,
    username       VARCHAR(50) NOT NULL,
    query          TEXT NOT NULL,
    results_found  BOOLEAN NOT NULL,
    timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

INSERT_CHUNK_SQL = """
INSERT INTO document_chunks (content, embedding, department_allowed, clearance_level)
VALUES ($1, $2, $3, $4)
"""


async def main() -> None:
    print(f"Loading embedding model '{MODEL_NAME}' (first run downloads it)...")
    model = SentenceTransformer(MODEL_NAME)

    try:
        conn = await asyncpg.connect(DB_DSN)
    except OSError as exc:
        raise SystemExit(
            "Could not connect to PostgreSQL at 127.0.0.1:5432.\n"
            "Is the container running? Start it with:  docker compose up -d"
        ) from exc

    try:
        # Order matters: the vector type only exists after CREATE EXTENSION.
        await conn.execute(SCHEMA_SQL)
        await register_vector(conn)
        print("Schema ready: vector extension enabled, tables created.")

        # Reseed so the script is idempotent.
        await conn.execute("TRUNCATE TABLE document_chunks RESTART IDENTITY;")

        for doc in SEED_DOCUMENTS:
            embedding = np.asarray(model.encode(doc["content"]), dtype=np.float32)
            await conn.execute(
                INSERT_CHUNK_SQL,
                doc["content"],
                embedding,
                doc["department_allowed"],
                doc["clearance_level"],
            )
            print(
                f"  seeded [{doc['department_allowed']} / clearance {doc['clearance_level']}]: "
                f"{doc['content']}"
            )

        chunk_count = await conn.fetchval("SELECT COUNT(*) FROM document_chunks;")
        print(f"Done. {chunk_count} documents in 'document_chunks'.")
        print("Next step:  uvicorn main:app --reload")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
