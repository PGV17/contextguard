# ContextGuard V1 (archived)

This directory preserves the original single-file prototype for reference.

V1 consists of:

- `main.py` - FastAPI app with `/token` and `/search`, raw `asyncpg`, exact K-NN
  search and SQL `WHERE`-clause row-level security.
- `db_setup.py` - schema and seed script for the V1 tables.

The active V2 implementation lives at the repository root and uses a different
schema (additional `document_id`/`chunk_index` columns, an HNSW index and an
extended audit table). Do **not** run V1's `db_setup.py` against a V2 database:
it truncates `document_chunks` and inserts rows that do not satisfy the V2
schema. Run it only against a dedicated V1 database.

V1 dependencies: `fastapi`, `uvicorn`, `asyncpg`, `PyJWT`,
`sentence-transformers`, `pgvector`.
