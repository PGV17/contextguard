# ContextGuard V2

**Enterprise-grade, permission-filtered RAG engine with row-level security.**

ContextGuard answers questions over an internal document corpus — but only from
documents the caller is allowed to see. Department and clearance filtering is
enforced **inside the SQL query**, so restricted content is never retrieved into
the application at all. Every answer is grounded in the retrieved context, cites
its sources, and every request is audited.

## Highlights

- **N-tier architecture** — `api` / `services` / `repositories` / `schemas` with
  dependency injection; no business logic in route handlers.
- **SQLAlchemy 2.0 async + Alembic** — typed ORM models, versioned migrations,
  pgvector `VECTOR` columns and an **HNSW** index.
- **Row-level security in SQL** — `(department = user OR department = 'All') AND
  clearance <= user_clearance` is part of the retrieval statement.
- **Similarity threshold** (default `0.65`) applied as a distance predicate
  during the vector index scan, before any chunk reaches an LLM.
- **Cited generation** — a strict system prompt forces `[Doc N]` citations;
  the API maps markers back to structured sources.
- **Audit trail** — query, caller, permission context, retrieved chunk ids and
  the model response are stored for every request.
- **Production concerns** — structured JSON logs with request ids, uniform error
  envelopes, startup schema/dimension validation, retries and timeouts,
  readiness/liveness probes, multi-stage Docker image.

## Architecture

```
                 ┌──────────────────────── FastAPI (app/api) ────────────────────────┐
   JWT bearer ──▶ │  routers: /auth  /documents  /chat  /health                      │
                 │  deps: OAuth2 bearer → CurrentUser, request-scoped AsyncSession   │
                 └───────────────────────────────┬───────────────────────────────────┘
                                                 ▼
                 ┌──────────────────────── Services (app/services) ──────────────────┐
                 │  IngestionService: authorize → chunk → embed → persist            │
                 │  RagService: embed → retrieve → threshold → synthesize → audit    │
                 │  EmbeddingService / LLMService (OpenAI or local providers)        │
                 └───────────────────────────────┬───────────────────────────────────┘
                                                 ▼
                 ┌────────────────────── Repositories (app/repositories) ────────────┐
                 │  ChunkRepository: INSERT + RLS-filtered pgvector ANN search       │
                 │  AuditRepository: append-only audit rows                          │
                 └───────────────────────────────┬───────────────────────────────────┘
                                                 ▼
                 ┌──────────── PostgreSQL 16 + pgvector (HNSW, cosine) ─────────────┐
                 └───────────────────────────────────────────────────────────────────┘
```

| Layer | Responsibility |
| --- | --- |
| `app/api` | HTTP concerns: routing, auth dependency, validation, status codes |
| `app/services` | Business rules: chunking, authorization policy, RAG orchestration |
| `app/repositories` | SQL: models, inserts, the RLS-enforcing similarity query |
| `app/schemas` | Pydantic v2 request/response contracts |
| `app/core` | Configuration, JWT, structured logging, exceptions, middleware |
| `app/db` | Engine/session factories, ORM models, startup schema checks |

## Project layout

```
app/
  main.py                  # application factory + lifespan singletons
  core/                    # config, security, logging, errors, middleware, users
  db/                      # SQLAlchemy models, async engine, schema verification
  schemas/                 # Pydantic v2 API models
  repositories/            # chunk + audit persistence and vector search SQL
  services/                # chunking, embeddings, LLM, ingestion, RAG pipeline
  api/
    deps.py                # DI: session, current user, services
    routers/               # auth, documents, chat, health
alembic/                   # async migration environment + initial schema
scripts/seed.py            # demo corpus (trusted, system-level)
legacy/v1/                 # archived V1 prototype
```

## Quickstart

Prerequisites: **Docker** and **Python 3.11+**.

```bash
# 1. Configure the environment
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env

# 2. Install the app (choose your profile)
pip install -e ".[local]"        # zero-key local embeddings
# or: pip install -e .           # OpenAI embeddings + generation

# 3. Start PostgreSQL 16 with pgvector
docker compose up -d

# 4. Apply migrations (the vector dimension is baked in here)
alembic upgrade head

# 5. Seed the demo corpus
python -m scripts.seed

# 6. Run the API
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs.

> **Vector dimensions are deployment configuration.** `text-embedding-3-small`
> produces 1536 dimensions; `all-MiniLM-L6-v2` produces 384. Run migrations with
> the same `CG_EMBEDDING_PROVIDER` / `CG_EMBEDDING_DIMENSIONS` you will serve
> with. Changing providers after ingesting requires a new migration and a full
> re-embed of the corpus.

### Provider profiles

**OpenAI (production default)**

```dotenv
CG_EMBEDDING_PROVIDER=openai
CG_EMBEDDING_MODEL=text-embedding-3-small
CG_EMBEDDING_DIMENSIONS=1536
CG_LLM_PROVIDER=openai
CG_LLM_MODEL=gpt-4o-mini
CG_OPENAI_API_KEY=sk-...
```

**Offline / local (zero-cost, used for CI and demos)**

```dotenv
CG_EMBEDDING_PROVIDER=local
CG_LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CG_EMBEDDING_DIMENSIONS=384
CG_LLM_PROVIDER=extractive
```

The extractive fallback returns the most relevant permitted excerpts with
citations. It is deterministic and requires no model download beyond the local
embedder.

## API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/auth/token` | Exchange a mock username for a JWT (JSON or OAuth2 form) |
| `POST` | `/documents/ingest` | Chunk, embed and store a document with permissions |
| `POST` | `/chat/completions` | Permission-filtered retrieval + cited synthesis |
| `GET` | `/health` | Liveness |
| `GET` | `/health/ready` | Readiness (database check) |

Demo users: `intern` (Marketing / clearance 1), `hr_lead` (HR / 2),
`engineer` (Engineering / 2).

### 1. Get a token

```bash
curl -s -X POST http://127.0.0.1:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "engineer"}'
```

```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_in": 3600,
  "username": "engineer",
  "department": "Engineering",
  "clearance": 2
}
```

> Windows PowerShell: use `curl.exe` (PowerShell aliases `curl` to
> `Invoke-WebRequest`), or use the Swagger UI authorize dialog.

### 2. Ingest a document

Ingestion is least-privilege: callers may only write to their **own department**
at or below their **own clearance** (`CG_MIN_INGEST_CLEARANCE`, default 2).

```bash
curl -s -X POST http://127.0.0.1:8000/documents/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The corporate VPN endpoint is vpn.engineering.example.com and requires SSO.",
    "department_allowed": "Engineering",
    "clearance_level": 2,
    "source_url": "https://intranet.example.com/engineering/vpn"
  }'
```

```json
{"document_id": "5f1c...", "chunks_created": 1, "chunk_ids": [12], "embedding_model": "text-embedding-3-small"}
```

### 3. Ask a question

```bash
curl -s -X POST http://127.0.0.1:8000/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the production database password?"}'
```

```json
{
  "answer": "The production database password is 'admin123' [Doc 1].",
  "citations": [
    {
      "marker": "[Doc 1]",
      "chunk_id": 1,
      "document_id": "eng-db-runbook",
      "source_url": "https://intranet.example.com/engineering/db-runbook",
      "similarity": 0.8123
    }
  ],
  "retrieved_chunks": 1,
  "model": "gpt-4o-mini"
}
```

## How the security works

The retrieval query is the security boundary (`app/repositories/chunks.py`):

```sql
SELECT id, document_id, content, source_url, department_allowed,
       clearance_level, 1 - (embedding <=> :query) AS similarity
FROM document_chunks
WHERE (department_allowed = :department OR department_allowed = :public)
  AND clearance_level <= :clearance
  AND embedding <=> :query < 1 - :min_similarity
ORDER BY embedding <=> :query
LIMIT :limit;
```

Why this matters:

- **No post-filtering.** Restricted rows are never fetched into the process,
  so they cannot leak through logs, serialization bugs or prompts.
- **Threshold inside SQL.** The distance predicate runs during the HNSW scan;
  irrelevant chunks never reach the LLM (cost + prompt-injection surface).
- **JWT is the only authorization source.** `department`/`clearance` come from a
  verified signature, not from request parameters.
- **Least privilege on write.** Ingestion cannot create higher-clearance or
  cross-department documents.
- **Audit includes permission context.** Each row records the department and
  clearance under which the query ran, plus the chunk ids and response.

`SET LOCAL hnsw.iterative_scan = relaxed_order` is applied per request so
filtered HNSW scans keep scanning until enough permitted neighbours are found
(pgvector ≥ 0.8; older versions fall back to defaults).

## Configuration

All variables are prefixed with `CG_`; see `.env.example` for the full list.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CG_SECRET_KEY` | dev placeholder | JWT signing key (must be changed in production) |
| `CG_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Access-token lifetime |
| `CG_DATABASE_URL` | local Postgres | SQLAlchemy async DSN |
| `CG_EMBEDDING_PROVIDER` | `openai` | `openai` or `local` |
| `CG_EMBEDDING_DIMENSIONS` | provider default | pgvector dimension; must match the migration |
| `CG_LLM_PROVIDER` | `openai` | `openai` or `extractive` |
| `CG_RETRIEVAL_TOP_K` / `CG_RETRIEVAL_MAX_TOP_K` | `5` / `20` | Retrieval depth and server-side cap |
| `CG_RETRIEVAL_MIN_SIMILARITY` | `0.65` | Chunks below this cosine similarity are dropped |
| `CG_PUBLIC_DEPARTMENT` | `All` | Department value treated as public |
| `CG_CHUNK_SIZE` / `CG_CHUNK_OVERLAP` | `1000` / `150` | Ingestion chunking |
| `CG_MIN_INGEST_CLEARANCE` | `2` | Minimum clearance allowed to ingest |

## Code quality

```bash
ruff check app scripts alembic   # linting (config in pyproject.toml)
mypy app                         # static type checking
```

## Docker

The database is the default profile. To run the API in a container as well:

```bash
docker compose --profile app up -d --build
```

The container applies migrations on start. It installs runtime dependencies
only (OpenAI profile); local sentence-transformers is intentionally excluded
from the image.

## Production checklist

- Replace the mock identity provider (`app/core/users.py`) with your IdP and
  rotate `CG_SECRET_KEY` via a secrets manager.
- Add Postgres native RLS policies (`CREATE POLICY ... USING (...)`) on top of
  the application predicate for defence in depth.
- Re-index when changing embedding providers/dimensions; keep a migration per
  dimension.
- Put the service behind rate limiting and a WAF; add per-user quotas.
- Ship JSON logs to your aggregator (request ids are already propagated).
- Consider a cross-encoder re-ranker between retrieval and synthesis.

## V1

The original single-file prototype is archived under `legacy/v1/` for
reference. Its scripts target the V1 schema and must not be run against a V2
database.
