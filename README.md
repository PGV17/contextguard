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
  clearance <= user_clearance` is part of every retrieval statement (both legs).
- **Hybrid retrieval with RRF fusion** — ANN similarity + Postgres full-text
  (`ts_rank_cd` over a stored generated tsvector), fused with Reciprocal-Rank
  Fusion. Dense search handles paraphrases; lexical search catches exact
  identifiers ("admin123") that ANN misses.
- **Optional cross-encoder re-ranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2`
  re-orders the fused candidate pool (lazy-loaded, CPU-friendly).
- **Measured, not guessed** — `scripts/eval.py` reports recall@k and MRR per
  retrieval mode: vector-only **62.5%** vs hybrid **100%** on the demo corpus.
- **Real users + RBAC** — bcrypt-hashed passwords in the `users` table, roles
  (`viewer` / `editor` / `admin`) signed into the JWT, admin-only registration.
- **Cited generation** — a strict system prompt forces `[Doc N]` citations;
  the API maps markers back to structured sources.
- **Tamper-evident audit trail** — every row carries
  `sha256(prev_hash | payload)`; `scripts/verify_audit.py` walks the chain and
  detects any silent edit, deletion or reordering.
- **Prometheus metrics** — request rates/latencies, retrieved-chunk counters and
  an insufficient-context counter at `/metrics`.
- **CI** — GitHub Actions runs ruff + mypy + tests plus a migration smoke test
  against pgvector/pg16 on every push.
- **Production concerns** — structured JSON logs with request ids, uniform error
  envelopes, startup schema/dimension validation, retries and timeouts,
  readiness/liveness probes, multi-stage Docker image.

## Architecture

```
   JWT bearer + role claim                                   Prometheus /metrics
          |                                                            ^
          v                                                            |
   +---------------------------- app/api ----------------------------+
   | routers: /auth  /documents  /chat  /health  /metrics             |
   | deps: OAuth2 bearer -> CurrentUser, request-scoped AsyncSession  |
   +-------------------------------+----------------------------------+
                                   v
   +---------------------------- services ----------------------------+
   | IngestionService: authorize (RBAC) -> chunk -> embed -> persist  |
   | RagService: embed -> retrieve -> fuse/re-rank -> synthesize      |
   |                            -> audit                              |
   | RetrievalService: ANN + full-text (RLS) -> RRF -> cross-encoder  |
   | EmbeddingService / LLMService (OpenAI or local providers)        |
   +-------------------------------+----------------------------------+
                                   v
   +--------------------------- repositories -------------------------+
   | ChunkRepository: RLS-filtered vector + full-text statements      |
   | UserRepository: bcrypt credential lookups                        |
   | AuditRepository: hash-chained, append-only audit rows            |
   +-------------------------------+----------------------------------+
                                   v
   +------------- PostgreSQL 16 + pgvector (HNSW, cosine) -------------+
   | document_chunks (RLS predicate, content_tsv GIN, HNSW index)      |
   | users (bcrypt hashes, roles)   audit_logs (sha256 hash chain)     |
   +-------------------------------------------------------------------+
```

| Layer | Responsibility |
| --- | --- |
| `app/api` | HTTP concerns: routing, auth dependency, validation, status codes |
| `app/services` | Business rules: RBAC policy, hybrid retrieval, RAG orchestration |
| `app/repositories` | SQL: models, inserts, RLS-enforcing retrieval, hash chain |
| `app/schemas` | Pydantic v2 request/response contracts |
| `app/core` | Configuration, JWT/bcrypt, metrics, audit-chain hashing, logging |
| `app/db` | Engine/session factories, ORM models, startup schema checks |

## Project layout

```
app/
  main.py                  # application factory + lifespan singletons (incl. reranker)
  core/                    # config, security, passwords, metrics, audit_chain
  db/                      # SQLAlchemy models, async engine, schema verification
  schemas/                 # Pydantic v2 API models
  repositories/            # chunks (hybrid SQL), users, audit (hash chain)
  services/                # chunking, embeddings, LLM, authorization (RBAC),
                           # retrieval (hybrid + RRF), reranking, ingestion, rag
  api/
    deps.py                # DI: session, current user, admin gate, services
    routers/               # auth, documents, chat, health, metrics
alembic/                   # 0001 initial schema, 0002 users/RBAC/hybrid/audit-chain
scripts/                   # seed, eval, verify_audit, check_schema (trusted, system-level)
tests/                     # lean unit suite (no database/network required)
.github/workflows/ci.yml   # CI: lint + types + tests + migration smoke test
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
| `POST` | `/auth/token` | Authenticate username + bcrypt password and receive a JWT carrying department, clearance and role (JSON or OAuth2 form) |
| `POST` | `/auth/register` | Create a directory user — admins only |
| `POST` | `/documents/ingest` | Chunk, embed and store a document with permissions (RBAC-checked) |
| `POST` | `/chat/completions` | Hybrid retrieval + cited synthesis, permission-filtered |
| `GET` | `/metrics` | Prometheus telemetry (public; protect at the proxy layer) |
| `GET` | `/health` | Liveness |
| `GET` | `/health/ready` | Readiness (database check) |

### Demo users

| Username | Password | Role | Department | Clearance |
| --- | --- | --- | --- | --- |
| `admin` | `demo-password` | admin | All | 5 |
| `engineer` | `demo-password` | editor | Engineering | 2 |
| `hr_lead` | `demo-password` | editor | HR | 2 |
| `intern` | `demo-password` | viewer | Marketing | 1 |

> **Demo credentials only.** All seeded users share one password so the demo
> stays simple to type; every hash is still per-user bcrypt. Re-seed replaces
> passwords. Replace `CG_SECRET_KEY` before any non-demo deployment.

### 1. Authenticate

```bash
curl -s -X POST http://127.0.0.1:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "engineer", "password": "demo-password"}'
```

```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_in": 3600,
  "username": "engineer",
  "department": "Engineering",
  "clearance": 2,
  "role": "editor"
}
```

Wrong password returns `401 authentication_failed` — with an identical message
for unknown users, so the endpoint cannot be used to enumerate accounts.

Admins create users (`403` for everyone else):

```bash
curl -s -X POST http://127.0.0.1:8000/auth/register \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"username": "qa_analyst", "password": "password-123",
       "department": "Marketing", "clearance": 1, "role": "viewer"}'
```

### 2. Ingest a document

RBAC roles: **viewers** read only; **editors** ingest inside their own
department at or below their own clearance; **admins** may ingest anywhere —
including public (`'All'`) and cross-department documents that previously
required the seed script.

```bash
curl -s -X POST http://127.0.0.1:8000/documents/ingest \
  -H "Authorization: Bearer $EDITOR_TOKEN" \
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

## Hybrid retrieval and re-ranking

Pure ANN search is great at paraphrases but misses exact identifiers. ContextGuard
therefore runs two permission-filtered legs inside PostgreSQL and fuses them:

1. **Vector leg** — pgvector HNSW cosine search with the clearance/department
   predicate and the similarity threshold applied in the index scan.
2. **Full-text leg** — `content_tsv` (a stored generated
   `to_tsvector('english', content)` column, GIN-indexed), stemming each query
   term and OR-combining the resulting tsqueries, ranked by `ts_rank_cd`. The
   **same RLS predicate** runs in this statement too.
3. **Fusion** — Reciprocal-Rank Fusion (pure function in
   `app/services/retrieval.py`, `k=60`, deterministic tie-breaking).
4. **Re-ranking (optional)** — a `cross-encoder/ms-marco-MiniLM-L-6-v2`
   re-orders the fused candidate pool (lazy-loaded on first query; disable with
   `CG_RERANKER_ENABLED=false`).

Measured on the demo corpus (`python -m scripts.eval`, recall@5 / MRR):

| Retrieval mode | recall@5 | MRR |
| --- | --- | --- |
| vector-only | 62.5% | 0.6250 |
| hybrid (RRF) | **100%** | **1.0000** |
| hybrid + cross-encoder | **100%** | **1.0000** |

Vector-only misses the lexically hard cases — the exact token `admin123`, and
paraphrases like "vacation days policy" against an "annual leave" document.
The full-text leg catches all of them; the re-ranker sharpens ordering.

## Observability

`GET /metrics` exposes Prometheus counters/histograms:

- `contextguard_http_requests_total{method,path,status}` and
  `contextguard_http_request_duration_seconds` (instrumented in middleware)
- `contextguard_retrieved_chunks_total{department}` — chunks that cleared the
  permission filter and threshold
- `contextguard_insufficient_context_total` — queries answered safely because
  nothing permitted reached the threshold

## Audit integrity (hash chain)

Every audit row stores `entry_hash = sha256(prev_entry_hash | row_payload)`,
computed inside the request transaction under an advisory lock so concurrent
writes cannot claim the same predecessor. `scripts/verify_audit.py` walks the
chain:

```console
$ python -m scripts.verify_audit
OK: 15 entries verified. Latest entry id=15 hash=60ddbc5cc2b9bacc...
```

Any silent `UPDATE`/`DELETE` anywhere in history breaks verification — useful
for graders: delete a row with `psql`, rerun the script, watch it fail.

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
| `CG_RETRIEVAL_POOL_SIZE` | `20` | Candidate pool gathered before fusion/re-ranking |
| `CG_RETRIEVAL_MIN_SIMILARITY` | `0.65` | Chunks below this cosine similarity are dropped |
| `CG_HYBRID_ENABLED` | `true` | Add the full-text leg + RRF fusion |
| `CG_HYBRID_RRF_K` | `60` | Reciprocal-Rank Fusion constant |
| `CG_RERANKER_ENABLED` | `false` | Cross-encoder re-ranking (`true` in the local demo profile) |
| `CG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranker checkpoint |
| `CG_PUBLIC_DEPARTMENT` | `All` | Department value treated as public |
| `CG_CHUNK_SIZE` / `CG_CHUNK_OVERLAP` | `1000` / `150` | Ingestion chunking |
| `CG_MIN_INGEST_CLEARANCE` | `2` | Minimum clearance allowed to ingest (editors) |

## Code quality and CI

```bash
ruff check app scripts alembic tests   # linting (config in pyproject.toml)
mypy app                               # static type checking
pytest -q                              # unit suite (no database/network needed)
python -m scripts.check_schema         # connectivity + pgvector + dims sanity
```

CI (`.github/workflows/ci.yml`) runs two jobs on every push/PR to `main`:

1. **quality** — ruff, mypy and `pytest` against a pinned Python 3.12.
2. **migration-smoke** — fresh `pgvector/pgvector:pg16` service, `alembic
   upgrade head`, then `scripts/check_schema` verifies tables, the HNSW index
   and the embedding dimension (catches migration/model drift).

## Docker

The database is the default profile. To run the API in a container as well:

```bash
docker compose --profile app up -d --build
```

The container applies migrations on start. It installs runtime dependencies
only (OpenAI profile); local sentence-transformers is intentionally excluded
from the image.

## Production checklist

- Move directory management into your IdP (SSO/LDAP) and rotate
  `CG_SECRET_KEY` via a secrets manager.
- Add Postgres native RLS policies (`CREATE POLICY ... USING (...)`) on top of
  the application predicate for defence in depth.
- Re-index when changing embedding providers/dimensions; keep a migration per
  dimension.
- Put the service behind rate limiting and a WAF; add per-user quotas.
- Ship JSON logs to your aggregator and scrape `/metrics` from Prometheus; keep
  the telemetry endpoint unauthenticated but network-protected.
- Ship the audit `entry_hash` chain to an append-only external sink for stronger
  tamper evidence.

## V1

The original single-file prototype is archived under `legacy/v1/` for
reference. Its scripts target the V1 schema and must not be run against a V2
database.
