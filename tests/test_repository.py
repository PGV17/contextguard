"""Compiled-SQL tests for the RLS predicate and hybrid retrieval legs."""

from __future__ import annotations

from sqlalchemy.dialects import postgresql

from app.repositories.chunks import build_full_text_statement, build_search_statement


def _compile_vector(**overrides):
    parameters = {
        "query_embedding": [0.1] * 384,
        "department": "HR",
        "clearance_level": 2,
        "public_department": "All",
        "limit": 5,
        "min_similarity": 0.65,
    }
    parameters.update(overrides)
    statement = build_search_statement(**parameters)
    return statement.compile(dialect=postgresql.dialect())


def _compile_fts(**overrides):
    parameters = {
        "query": "admin123 password",
        "department": "HR",
        "clearance_level": 2,
        "public_department": "All",
        "limit": 5,
    }
    parameters.update(overrides)
    statement = build_full_text_statement(**parameters)
    return statement.compile(dialect=postgresql.dialect())


def test_statement_filters_by_department_and_clearance() -> None:
    compiled = _compile_vector()
    sql = str(compiled)
    values = list(compiled.params.values())

    assert "department_allowed" in sql
    assert "clearance_level <=" in sql
    assert "HR" in values
    assert "All" in values
    assert 2 in values


def test_admin_bypass_drops_only_the_department_predicate() -> None:
    sql = str(_compile_vector(include_all_departments=True))

    # One occurrence = the SELECT column; the RLS "=" predicate is gone.
    assert sql.count("department_allowed") == 1
    assert "clearance_level <=" in sql


def test_threshold_is_expressed_as_a_distance_predicate() -> None:
    compiled = _compile_vector(min_similarity=0.65)
    floats = [
        value for value in compiled.params.values() if isinstance(value, float)
    ]

    # cosine_distance < 1 - 0.65 = 0.35
    assert any(abs(value - 0.35) < 1e-9 for value in floats)


def test_statement_orders_by_distance_with_limit() -> None:
    sql = str(_compile_vector(limit=7)).upper()

    assert "ORDER BY" in sql
    assert "LIMIT" in sql


def test_full_text_leg_uses_tsvector_with_the_same_rls_predicate() -> None:
    compiled = _compile_fts()
    sql = str(compiled)
    values = list(compiled.params.values())

    assert "websearch_to_tsquery" in sql
    assert "@@" in sql
    assert "|" in sql  # terms are OR-combined
    assert "department_allowed" in sql
    assert "clearance_level <=" in sql
    assert "admin123" in values
    assert "HR" in values


def test_full_text_leg_admin_bypass_matches_vector_leg() -> None:
    sql = str(_compile_fts(include_all_departments=True))

    assert sql.count("department_allowed") == 1  # SELECT column only
    assert "websearch_to_tsquery" in sql
