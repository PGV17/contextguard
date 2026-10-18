"""
ContextGuard - Permission-Filtered RAG Engine with Row-Level Security (V1).

Endpoints:
  POST /token  - exchange a mock username for a JWT carrying department + clearance.
  POST /search - embed the query, run a pgvector similarity search whose WHERE
                 clause enforces row-level security, audit-log the access, and
                 return the permitted chunks (no LLM generation in V1).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import List

import asyncpg
import jwt
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pgvector.asyncpg import register_vector
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SECRET_KEY = "contextguard-dev-secret-change-me"  # use an env var in production
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DB_DSN = "postgresql://postgres:password@127.0.0.1:5432/rag_db"
MAX_RESULTS = 5

# Mock identity directory. A real deployment would look these up in a user store.
MOCK_USERS = {
    "intern": {"department": "Marketing", "clearance": 1},
    "hr_lead": {"department": "HR", "clearance": 2},
    "engineer": {"department": "Engineering", "clearance": 2},
}

# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class TokenRequest(BaseModel):
    username: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    department: str
    clearance: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["production database password"])


class SearchResult(BaseModel):
    id: int
    content: str
    similarity: float
    department_allowed: str
    clearance_level: int


class SearchResponse(BaseModel):
    user: str
    department: str
    clearance: int
    query: str
    results_found: bool
    results: List[SearchResult]


# --------------------------------------------------------------------------- #
# SQL
# --------------------------------------------------------------------------- #
# The core row-level-security clause lives INSIDE the SQL, so a document can
# never leak through application-level post-filtering: a row is only visible if
# (department matches OR the document is public) AND the user's clearance is
# high enough for it.
SEARCH_SQL = """
SELECT
    id,
    content,
    department_allowed,
    clearance_level,
    1 - (embedding <=> $1::vector) AS similarity
FROM document_chunks
WHERE (department_allowed = $2 OR department_allowed = 'All')
  AND clearance_level <= $3
ORDER BY embedding <=> $1::vector
LIMIT $4
"""

AUDIT_SQL = """
INSERT INTO audit_logs (username, query, results_found)
VALUES ($1, $2, $3)
"""

# --------------------------------------------------------------------------- #
# Application state
# --------------------------------------------------------------------------- #
embedding_model: SentenceTransformer | None = None
pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, pool

    print(f"[ContextGuard] Loading embedding model '{MODEL_NAME}'...")
    embedding_model = SentenceTransformer(MODEL_NAME)

    try:
        pool = await asyncpg.create_pool(
            dsn=DB_DSN, min_size=1, max_size=5, init=register_vector
        )
    except OSError as exc:
        raise RuntimeError(
            "Could not connect to PostgreSQL at 127.0.0.1:5432. "
            "Start it with: docker compose up -d"
        ) from exc
    except ValueError as exc:
        # register_vector() fails before CREATE EXTENSION has run.
        raise RuntimeError(
            "The pgvector extension/tables are not initialized yet. "
            "Run: python db_setup.py"
        ) from exc

    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT COUNT(*) FROM document_chunks")
    except asyncpg.exceptions.UndefinedTableError as exc:
        raise RuntimeError(
            "Tables are missing. Initialize the database first: python db_setup.py"
        ) from exc

    print("[ContextGuard] Connected to PostgreSQL (pgvector). Ready.")
    yield

    if pool is not None:
        await pool.close()


app = FastAPI(
    title="ContextGuard",
    version="1.0.0",
    description=(
        "Permission-Filtered RAG Engine with Row-Level Security. "
        "Documents are filtered by department and clearance level "
        "inside the SQL query itself."
    ),
    lifespan=lifespan,
)

bearer_scheme = HTTPBearer(auto_error=True)


# --------------------------------------------------------------------------- #
# Auth dependency
# --------------------------------------------------------------------------- #
async def get_claims(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """Decode the bearer JWT and return its permission claims."""
    try:
        claims = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
        ) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {exc}"
        ) from exc

    missing = {"sub", "department", "clearance"} - set(claims)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token is missing required claims: {', '.join(sorted(missing))}",
        )
    return claims


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/")
async def root() -> dict:
    return {"service": "ContextGuard", "version": "1.0.0", "docs": "/docs"}


@app.post("/token", response_model=TokenResponse)
async def create_token(body: TokenRequest) -> TokenResponse:
    """Issue a mock JWT whose claims carry the user's department and clearance."""
    username = body.username.strip()
    user = MOCK_USERS.get(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                f"Unknown user '{username}'. "
                f"Valid users: {', '.join(sorted(MOCK_USERS))}"
            ),
        )

    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "department": user["department"],
        "clearance": user["clearance"],
        "iat": now,
        "exp": now + timedelta(minutes=TOKEN_EXPIRE_MINUTES),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return TokenResponse(
        access_token=token,
        department=user["department"],
        clearance=user["clearance"],
    )


@app.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    claims: dict = Depends(get_claims),
) -> SearchResponse:
    """Permission-filtered similarity search over the seeded document chunks."""
    if embedding_model is None or pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is still starting up, retry shortly.",
        )

    username = str(claims["sub"])
    department = str(claims["department"])
    clearance = int(claims["clearance"])

    query = body.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query must contain non-whitespace characters.",
        )

    # 1) Embed the query locally (free, no external API).
    embedding = np.asarray(embedding_model.encode(query), dtype=np.float32)

    # 2) Similarity search with row-level security enforced inside SQL,
    #    and 3) audit-log the access in the same transaction.
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                SEARCH_SQL, embedding, department, clearance, MAX_RESULTS
            )
            await conn.execute(AUDIT_SQL, username, query, len(rows) > 0)

    results = [
        SearchResult(
            id=row["id"],
            content=row["content"],
            similarity=round(float(row["similarity"]), 4),
            department_allowed=row["department_allowed"],
            clearance_level=row["clearance_level"],
        )
        for row in rows
    ]

    return SearchResponse(
        user=username,
        department=department,
        clearance=clearance,
        query=query,
        results_found=bool(results),
        results=results,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
