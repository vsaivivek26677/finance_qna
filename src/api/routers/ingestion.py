"""Ingestion endpoint: run Layers 1-3 for a ticker on demand.

This is the one write endpoint. It runs the whole chain - fetch, validate, store,
compute ratios, detect flags, index for retrieval - because a half-ingested
company is a trap: it looks present in `/companies` but every analysis endpoint
returns nothing.

It is synchronous rather than a background job. The call takes tens of seconds,
but the caller gets a real report of what landed and what was missing, which for
a single-user analytics tool is more useful than a job id to poll. A production
deployment with many users would move this to a queue.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from src.api.dependencies import clear_caches, get_pipeline, normalise_ticker
from src.api.schemas import IngestRequest, IngestResponse
from src.config import settings
from src.db.database import session_scope
from src.ingestion.fmp_client import FMPAuthError, FMPError
from src.ingestion.ingest_pipeline import IngestionPipeline
from src.ingestion.schemas import PeriodType
from src.ratios.ratio_engine import run_analysis

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingestion"])


def _run_chain(ticker: str, payload: IngestRequest) -> IngestResponse:
    """Ingest, analyse and index one ticker. Runs in a worker thread.

    Each post-fetch step gets its own ``session_scope`` rather than sharing the
    request session: they run on this worker thread, a Session must not be
    touched from two threads, and isolating them means one step failing cannot
    poison the transaction the next step commits.
    """
    pipeline = IngestionPipeline(include_market_data=payload.include_market_data)
    try:
        report = pipeline.run(
            ticker, period=PeriodType(payload.period), years=payload.years
        )
    finally:
        pipeline.fmp.close()

    response = IngestResponse(
        ticker=ticker,
        status=report.status,
        records_written=report.records_written,
        critical_missing=report.critical_missing,
        errors=report.errors,
    )
    if report.status == "failed":
        return response

    if payload.run_ratios:
        try:
            with session_scope() as db:
                analysis = run_analysis(db, ticker, period="FY")
                response.ratios_computed = analysis.ratios_computed
                response.red_flags_raised = analysis.red_flags_raised
                response.errors.extend(analysis.warnings)
        except Exception as exc:  # noqa: BLE001 - ratios failing must not lose the data
            logger.exception("Ratio analysis failed for %s", ticker)
            response.errors.append(f"ratios: {exc}")

    if payload.index_for_rag:
        try:
            with session_scope() as db:
                response.chunks_indexed = get_pipeline().index_company(db, ticker)
        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG indexing failed for %s", ticker)
            response.errors.append(f"rag_index: {exc}")

    if payload.run_ratios:
        try:
            from src.prediction import repository as prediction_repository
            from src.prediction.predict import ModelUnavailable, score

            with session_scope() as db:
                rows = prediction_repository.build_feature_rows(db, ticker)
                estimates = [(row, score(row.features)) for row in rows]
                response.distress_estimates = prediction_repository.save_estimates(
                    db, ticker, estimates
                )
        except ModelUnavailable:
            pass  # no model trained yet - not an ingestion failure
        except Exception as exc:  # noqa: BLE001
            logger.exception("Distress scoring failed for %s", ticker)
            response.errors.append(f"distress: {exc}")

    return response


@router.post(
    "/ingest/{ticker}",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Fetch, analyse and index a company",
)
async def ingest(
    payload: IngestRequest | None = None,
    ticker: str = Depends(normalise_ticker),
) -> IngestResponse:
    """Run the full chain for one ticker.

    Costs 4 requests against the Financial Modeling Prep free tier (250/day).
    """
    if not settings.fmp_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "FMP_API_KEY is not configured, so new companies cannot be ingested. "
                "Add it to .env (free key at site.financialmodelingprep.com)."
            ),
        )

    payload = payload or IngestRequest()
    try:
        response = await run_in_threadpool(_run_chain, ticker, payload)
    except FMPAuthError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except FMPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream data provider error: {exc}",
        ) from exc

    clear_caches()  # newly ingested data must be visible immediately

    if response.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Ingestion failed for {ticker}: {'; '.join(response.errors) or 'no data returned'}"
            ),
        )
    return response
