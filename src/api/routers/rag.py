"""Generated summary and Q&A endpoints.

These are the only endpoints that call an LLM, and the only ones that can be
slow or fail for external reasons. Both are `async def` with the blocking work
pushed to a worker thread: the RAG call spends its time waiting on Groq, and
holding the event loop for it would stall every other request on the service.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from starlette.concurrency import run_in_threadpool

from src.api.dependencies import (
    get_company,
    get_pipeline,
    normalise_ticker,
    require_llm,
    summary_cache,
)
from src.api.schemas import AnswerResponse, AskRequest, SourceChunk
from src.db.database import session_scope
from src.db.models import Company
from src.rag.groq_client import GroqError
from src.rag.rag_pipeline import RagAnswer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ai"])

EXCERPT_CHARS = 600


def _to_response(ticker: str, result: RagAnswer) -> AnswerResponse:
    report = result.verification
    return AnswerResponse(
        ticker=ticker,
        question=result.question,
        answer=result.answer,
        groundedness=round(report.groundedness, 4),
        verified_claims=len(report.verified),
        total_claims=report.total,
        unverified=sorted({m.raw for m in report.unverified}) + report.unverified_periods,
        blocked=report.blocked,
        notes=report.notes,
        model=result.model,
        latency_seconds=round(result.latency_seconds, 3),
        sources=[
            SourceChunk(
                citation=chunk.citation(),
                chunk_type=chunk.chunk_type,
                fiscal_year=chunk.fiscal_year,
                similarity=round(chunk.score, 4),
                excerpt=chunk.text[:EXCERPT_CHARS],
            )
            for chunk in result.chunks
        ],
    )


@router.get(
    "/companies/{ticker}/summary",
    response_model=AnswerResponse,
    summary="Grounded executive summary",
    dependencies=[Depends(require_llm)],
)
async def get_summary(
    ticker: str = Depends(normalise_ticker),
    company: Company = Depends(get_company),
    fiscal_year: int | None = Query(None, description="Defaults to the latest year on file"),
    refresh: bool = Query(False, description="Bypass the cache and regenerate"),
) -> AnswerResponse:
    """Executive summary generated only from verified figures.

    Cached for 15 minutes: it is the slowest endpoint and every call spends free
    tier LLM quota, while the underlying data changes only on re-ingestion.
    """
    cache = summary_cache()
    key = f"summary:{ticker}:{fiscal_year}"
    if not refresh and key in cache:
        return cache[key]

    pipeline = get_pipeline()
    try:
        result = await run_in_threadpool(_run_summary, pipeline, ticker, fiscal_year)
    except GroqError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The language model is unavailable: {exc}",
        ) from exc

    response = _to_response(ticker, result)
    cache[key] = response
    return response


def _run_summary(pipeline, ticker: str, fiscal_year: int | None) -> RagAnswer:
    """Generation plus its own DB session, both on the worker thread.

    The request-scoped session must not cross the threadpool boundary - a
    SQLAlchemy Session is not safe to touch from two threads, and the query-log
    write inside the pipeline is what would corrupt it.
    """
    with session_scope() as session:
        return pipeline.summarize(ticker, fiscal_year=fiscal_year, session=session)


@router.post(
    "/companies/{ticker}/ask",
    response_model=AnswerResponse,
    summary="Ask a grounded question about a company",
    dependencies=[Depends(require_llm)],
)
async def ask(
    payload: AskRequest,
    ticker: str = Depends(normalise_ticker),
    company: Company = Depends(get_company),
) -> AnswerResponse:
    """Answer a question using only retrieved, verified context.

    The response always carries its own verification: `groundedness` is the share
    of numeric claims traceable back to the retrieved chunks, and `unverified`
    names any that were not. Answers are not cached - the same question can be
    asked about newly ingested data.
    """
    pipeline = get_pipeline()
    try:
        result = await run_in_threadpool(
            _run_ask, pipeline, payload.question, ticker, payload.fiscal_year, payload.top_k
        )
    except GroqError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The language model is unavailable: {exc}",
        ) from exc

    return _to_response(ticker, result)


def _run_ask(pipeline, question: str, ticker: str, fiscal_year: int | None, top_k: int | None):
    """Q&A plus its own worker-thread DB session (see _run_summary)."""
    with session_scope() as session:
        return pipeline.ask(
            question, ticker, fiscal_year=fiscal_year, session=session, top_k=top_k
        )
