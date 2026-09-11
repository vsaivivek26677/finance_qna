"""FastAPI application.

The service owns all business logic; the Streamlit dashboard is a thin HTTP
client with no database or LLM access of its own. That separation is the point of
having an API layer at all - the same backend could serve a different frontend
without change.

Run it with:  uvicorn src.api.main:app --reload
Docs at:      http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select

from src.api.dependencies import get_db
from src.api.routers import companies, ingestion, predictions, ratios, rag
from src.api.schemas import HealthResponse
from src.config import settings
from src.db.database import init_db, session_scope
from src.db.models import Company, Ratio
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

API_VERSION = "0.6.0"

DESCRIPTION = """
Financial statement analysis over verified data.

**What makes this API unusual:** it reports the absence of data as explicitly as
its presence. Ratios that could not be computed come back with
`is_calculable: false` and a reason naming the missing input. Statement fields the
data provider never reported are listed in `missing_fields`. Generated answers
carry a `groundedness` score and name any figure that could not be traced back to
the source data.

Nothing is estimated to fill a gap.

* **Layer 1** - ingestion from Financial Modeling Prep and yfinance
* **Layer 2** - 35 ratios, Altman Z / Z'', Piotroski F, Beneish M, and 15 rule-based red flags
* **Layer 3** - retrieval-augmented Q&A with post-generation number verification
* **Layer 4** - a calibrated distress-probability model trained on ~78.7k labelled American-company firm-years
"""

TAGS_METADATA = [
    {"name": "companies", "description": "Profiles and filed financial statements."},
    {"name": "analysis", "description": "Ratios, distress scores and rule-based red flags."},
    {"name": "ai", "description": "Grounded summaries and Q&A. Requires a Groq API key."},
    {"name": "ingestion", "description": "Fetch, analyse and index a company on demand."},
    {"name": "system", "description": "Health and readiness."},
]


def _warm_retrieval() -> None:
    """Load the embedding model off the request path.

    The first question after a restart otherwise waits ~30s while the
    sentence-transformer loads, which reads as a hang. Warming it in a daemon
    thread means start-up is unaffected and the first real query is fast.
    """
    try:
        from src.api.dependencies import get_pipeline

        get_pipeline().embedder.embed(["warmup"])
        logger.info("Retrieval model warmed")
    except Exception as exc:  # noqa: BLE001 - warmup is best-effort
        logger.info("Retrieval warmup skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare the schema once at start-up rather than per request."""
    setup_logging()
    init_db()
    threading.Thread(target=_warm_retrieval, name="retrieval-warmup", daemon=True).start()
    logger.info("API ready (version %s)", API_VERSION)
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Financial Analytics Platform API",
    description=DESCRIPTION,
    version=API_VERSION,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# The dashboard runs as a separate process (locally, or on Streamlit Community
# Cloud), so it is a cross-origin caller. Credentials are not used, and the API
# holds no user data, so a permissive origin policy is appropriate here; a
# deployment with auth would pin this to the known frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(companies.router)
app.include_router(ratios.router)
app.include_router(predictions.router)
app.include_router(rag.router)
app.include_router(ingestion.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a clean error rather than leaking a stack trace to the client."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal error: {type(exc).__name__}"},
    )


@app.get("/health", response_model=HealthResponse, tags=["system"], summary="Service health")
def health() -> HealthResponse:
    """Readiness of every dependency, so a failure is diagnosable at a glance."""
    companies_count = ratios_count = analysed_count = 0
    database_state = "ok"
    try:
        with session_scope() as session:
            companies_count = int(session.scalar(select(func.count()).select_from(Company)) or 0)
            ratios_count = int(session.scalar(select(func.count()).select_from(Ratio)) or 0)
            analysed_count = int(
                session.scalar(select(func.count(func.distinct(Ratio.company_id)))) or 0
            )
    except Exception as exc:  # noqa: BLE001
        database_state = f"error: {type(exc).__name__}"
        logger.warning("Health check could not reach the database: %s", exc)

    indexed_count = 0
    try:
        from src.rag.vector_store import ChromaVectorStore

        store = ChromaVectorStore()
        indexed_count = len({c.ticker for c in store.get({}, limit=5000)})
    except Exception:  # noqa: BLE001 - an unbuilt index is a normal state
        indexed_count = 0

    try:
        from src.prediction.predict import model_is_ready

        distress_ready = model_is_ready()
    except Exception:  # noqa: BLE001 - an untrained model is a normal state
        distress_ready = False

    return HealthResponse(
        status="ok" if database_state == "ok" else "degraded",
        database=database_state,
        companies=companies_count,
        ratios=ratios_count,
        rag_indexed=indexed_count > 0,
        companies_analysed=analysed_count,
        companies_indexed=indexed_count,
        distress_model_ready=distress_ready,
        llm_configured=bool(settings.groq_api_key),
        version=API_VERSION,
    )


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "Financial Analytics Platform API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
