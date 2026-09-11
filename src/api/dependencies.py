"""Shared dependencies: database sessions, caching, and the RAG pipeline.

The RAG pipeline is a module-level singleton on purpose. Constructing one loads a
sentence-transformer model and opens a ChromaDB client; doing that per request
would add seconds of latency to every question.

Caching is a small in-memory TTL cache rather than Redis. The read endpoints hit
SQLite, and the expensive calls downstream (FMP, Groq) are rate-limited free
tiers, so the thing worth protecting is the quota, not the database.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Callable

from cachetools import TTLCache
from fastapi import Depends, HTTPException, Path, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import settings
from src.db.database import get_session_factory
from src.db.models import Company

logger = logging.getLogger(__name__)

# Read-through cache for endpoints that hit the database. Short-lived: an
# ingestion run should be visible almost immediately.
_read_cache: TTLCache = TTLCache(maxsize=512, ttl=60)
# Generated summaries are slow and cost LLM quota, so they live longer.
_summary_cache: TTLCache = TTLCache(maxsize=128, ttl=900)

_pipeline = None


def get_db() -> Iterator[Session]:
    """One session per request, always closed.

    Rolls back on a failed request so a poisoned transaction can never ride a
    pooled connection into the next request (which then fails with
    PendingRollbackError far from the real cause).
    """
    session = get_session_factory()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_pipeline():
    """Process-wide RAG pipeline, built on first use."""
    global _pipeline
    if _pipeline is None:
        from src.rag.rag_pipeline import RagPipeline

        _pipeline = RagPipeline()
        logger.info("RAG pipeline initialised")
    return _pipeline


def reset_pipeline() -> None:
    """Drop the cached pipeline - used by tests to inject a fake."""
    global _pipeline
    _pipeline = None


def set_pipeline(pipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def clear_caches() -> None:
    _read_cache.clear()
    _summary_cache.clear()


def cached(cache: TTLCache, key: str, producer: Callable[[], Any]) -> Any:
    """Read-through cache helper."""
    if key in cache:
        return cache[key]
    value = producer()
    cache[key] = value
    return value


def read_cache() -> TTLCache:
    return _read_cache


def summary_cache() -> TTLCache:
    return _summary_cache


def normalise_ticker(
    ticker: str = Path(..., min_length=1, max_length=16, description="Ticker symbol")
) -> str:
    return ticker.strip().upper()


def get_company(
    ticker: str = Depends(normalise_ticker), db: Session = Depends(get_db)
) -> Company:
    """Resolve a ticker to a company, or 404 with an actionable hint.

    The hint matters: an empty database is the normal first-run state, and
    "not found" alone leaves the caller guessing what to do about it.
    """
    company = db.scalar(select(Company).where(Company.ticker == ticker))
    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"{ticker} is not in the database. Ingest it first: "
                f"POST /ingest/{ticker}"
            ),
        )
    return company


def require_llm() -> None:
    """Guard the endpoints that need Groq, with a clear reason when it is absent."""
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "GROQ_API_KEY is not configured, so generated answers are unavailable. "
                "Add it to .env (free key at https://console.groq.com). All non-generative "
                "endpoints work without it."
            ),
        )
