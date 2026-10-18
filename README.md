# ContextGuard

Minimal, runnable **Permission-Filtered RAG Engine with Row-Level Security**.

- **FastAPI** API with mock JWT auth (`PyJWT`)
- **PostgreSQL 16 + pgvector** storing embeddings, accessed via **asyncpg**
- **sentence-transformers/all-MiniLM-L6-v2** for local, free 384-dim embeddings
- Department + clearance filtering enforced **inside the SQL query**, not in Python

```
client ──/token──▶ JWT {sub, department, clearance}
client ──/search──▶ embed query ──▶ pgvector <=> search
                                   └─ WHERE (dept = user OR dept = 'All')
                                      AND clearance <= user_clearance
                                  ──▶ audit_logs ──▶ permitted chunks back
```

## Prerequisites

- Docker (Desktop or Engine) running
- Python 3.10+

## Install (once)

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate

# Recommended on Windows: CPU-only torch (~200 MB) instead of the multi-GB CUDA build
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

> The first run of `db_setup.py` also downloads the MiniLM model (~90 MB) from HuggingFace. Everything after that is fully offline/free.

## Run in 3 steps

```bash
# 1. Start PostgreSQL (pgvector/pgvector:pg16 on port 5432, user postgres / password / db rag_db)
docker compose up -d

# 2. Create the schema and seed the 3 mock documents
python db_setup.py

# 3. Start the API (Swagger UI at http://127.0.0.1:8000/docs)
uvicorn main:app --reload
```

## Try it

**Get a token** (or click through the `/docs` Swagger UI — no curl needed):

```bash
curl -s -X POST http://127.0.0.1:8000/token \
  -H "Content-Type: application/json" \
  -d '{"username": "engineer"}'
```

Returns `{"access_token": "eyJ...", "token_type": "bearer", "department": "Engineering", "clearance": 2}`.
Valid users: `intern` (Marketing/1), `hr_lead` (HR/2), `engineer` (Engineering/2).

**Search** with the token pasted in:

```bash
curl -s -X POST http://127.0.0.1:8000/search \
  -H "Authorization: Bearer PASTE_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{"query": "password"}'
```

> **Windows PowerShell**: `curl` is an alias of `Invoke-WebRequest` there — use `curl.exe` instead, or just use the `/docs` UI.

### Expected results for query `"password"`

| user      | JWT permissions      | sees                                          |
| --------- | -------------------- | --------------------------------------------- |
| engineer  | Engineering / 2      | **Doc 1 (password)** + Doc 3                  |
| hr_lead   | HR / 2               | Doc 2 (bonus) + Doc 3 — Doc 1 is Eng-only     |
| intern    | Marketing / 1        | Doc 3 only — Doc 1/2 blocked (dept+clearance) |

Doc 3 (`department_allowed = 'All'`, clearance 1) is visible to everyone; that is by design.
Note there is **no relevance threshold in V1** — the top-5 *permitted* chunks are always
returned, so this demo proves permission filtering, not relevance ranking. (An easy extension:
add `AND 1 - (embedding <=> $1) > 0.3` to `SEARCH_SQL` in `main.py`.)

## How the row-level security works

The filter lives inside the query itself, so a restricted row can never leak via
application-level post-filtering (`main.py`):

```sql
SELECT id, content, department_allowed, clearance_level,
       1 - (embedding <=> $1::vector) AS similarity
FROM document_chunks
WHERE (department_allowed = $2 OR department_allowed = 'All')
  AND clearance <= user_clearance  -- clearance_level <= $3
ORDER BY embedding <=> $1::vector
LIMIT 5;
```

The JWT's `department` and `clearance` claims are the only source of truth for `$2`/`$3`;
the client cannot influence them.

## Audit logs

Every `/search` call writes `username`, `query`, `results_found`, and `timestamp`:

```bash
docker exec -it contextguard-db psql -U postgres -d rag_db \
  -c "SELECT id, username, query, results_found, timestamp FROM audit_logs ORDER BY id DESC LIMIT 10;"
```

## Project layout

| file                 | purpose                                              |
| -------------------- | ---------------------------------------------------- |
| `docker-compose.yml` | pgvector/pgvector:pg16 database on port 5432         |
| `db_setup.py`        | creates extension + tables, seeds 3 embedded docs    |
| `main.py`            | FastAPI app: `/token` and `/search` with RLS query   |
| `requirements.txt`   | Python dependencies                                  |
