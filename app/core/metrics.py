"""Prometheus metric registry.

Metrics are declared once here; the HTTP middleware and the RAG service
instrument them. The ``/metrics`` endpoint exposes the default registry.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "contextguard_http_requests_total",
    "HTTP requests handled, by route template and status.",
    ["method", "path", "status"],
)

HTTP_DURATION = Histogram(
    "contextguard_http_request_duration_seconds",
    "End-to-end request latency in seconds.",
    ["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

RETRIEVED_CHUNKS = Counter(
    "contextguard_retrieved_chunks_total",
    "Chunks that cleared the permission filter and similarity threshold.",
    ["department"],
)

INSUFFICIENT_CONTEXT = Counter(
    "contextguard_insufficient_context_total",
    "Queries where no permitted chunk reached the similarity threshold.",
    ["department"],
)
