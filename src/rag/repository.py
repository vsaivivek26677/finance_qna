"""Persistence for RAG queries and evaluation runs.

Every question - production or evaluation - is logged with its retrieved context
and its groundedness score. That turns "the RAG is grounded" from a design claim
into a measurement over real traffic that can be queried later.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import Company, RagEvalLog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.rag.rag_pipeline import RagAnswer

logger = logging.getLogger(__name__)


def log_rag_query(
    session: Session,
    result: "RagAnswer",
    scores: dict[str, float] | None = None,
    query_type: str | None = None,
) -> int:
    """Write one query to `rag_eval_logs`. Returns the new row id."""
    company = session.scalar(select(Company).where(Company.ticker == result.ticker))
    entry = RagEvalLog(
        company_id=company.id if company else None,
        ticker=result.ticker,
        query=result.question,
        query_type=query_type or result.query_type,
        response=result.answer,
        retrieved_chunks_json={
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "citation": chunk.citation(),
                    "score": round(chunk.score, 6),
                    "chunk_type": chunk.chunk_type,
                    "fiscal_year": chunk.fiscal_year,
                }
                for chunk in result.chunks
            ]
        },
        scores_json=dict(scores or {}),
        groundedness=result.verification.groundedness,
        unverified_json=result.verification.to_json(),
        blocked=result.verification.blocked,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=int(result.latency_seconds * 1000),
    )
    session.add(entry)
    session.flush()
    return int(entry.id)


def get_logs(
    session: Session,
    ticker: str | None = None,
    query_type: str | None = None,
    limit: int = 50,
) -> list[RagEvalLog]:
    query = select(RagEvalLog).order_by(RagEvalLog.created_at.desc(), RagEvalLog.id.desc())
    if ticker:
        query = query.where(RagEvalLog.ticker == ticker.strip().upper())
    if query_type:
        query = query.where(RagEvalLog.query_type == query_type)
    return list(session.scalars(query.limit(limit)))


def groundedness_summary(session: Session, ticker: str | None = None) -> dict[str, Any]:
    """Aggregate groundedness across logged queries - the dashboard headline."""
    query = select(
        func.count(RagEvalLog.id),
        func.avg(RagEvalLog.groundedness),
        func.min(RagEvalLog.groundedness),
    ).where(RagEvalLog.groundedness.is_not(None))
    if ticker:
        query = query.where(RagEvalLog.ticker == ticker.strip().upper())

    count, average, minimum = session.execute(query).one()
    blocked_query = select(func.count(RagEvalLog.id)).where(RagEvalLog.blocked.is_(True))
    if ticker:
        blocked_query = blocked_query.where(RagEvalLog.ticker == ticker.strip().upper())

    return {
        "queries_logged": int(count or 0),
        "mean_groundedness": round(float(average), 4) if average is not None else None,
        "min_groundedness": round(float(minimum), 4) if minimum is not None else None,
        "blocked_answers": int(session.scalar(blocked_query) or 0),
    }
