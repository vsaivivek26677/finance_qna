"""Layer 3 orchestration: index -> retrieve -> generate -> verify -> log.

The pipeline never lets the model near raw filings. It sees only chunks built by
Layer 3's chunker from figures Layer 1 ingested and Layer 2 computed, and
everything it says is checked back against those figures afterwards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from sqlalchemy.orm import Session

from src.config import settings
from src.rag import repository
from src.rag.chunker import DATA_AVAILABILITY, DISTRESS, PROFILE, RATIOS, RED_FLAG, build_chunks
from src.rag.embedder import Embedder, get_embedder
from src.rag.groq_client import GroqClient
from src.rag.guardrails import (
    SYSTEM_PROMPT,
    VerificationReport,
    apply_guardrails,
    build_summary_prompt,
    build_user_prompt,
)
from src.rag.vector_store import RetrievedChunk, VectorStore, get_vector_store

logger = logging.getLogger(__name__)

# Question intents whose answers live in a specific chunk type, regardless of how
# the question happens to score against everything else. Keys are keyword sets;
# matching one guarantees those chunk types appear in the retrieved context.
INTENT_CHUNK_TYPES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("risk", "bankrupt", "distress", "insolven", "solvency", "default", "altman", "z-score"): (
        DISTRESS,
        RED_FLAG,
    ),
    ("red flag", "warning", "concern", "worrying", "trouble"): (RED_FLAG, DISTRESS),
    ("manipulat", "earnings quality", "beneish", "aggressive accounting", "fraud"): (
        DISTRESS,
        RED_FLAG,
    ),
    ("piotroski", "f-score", "fundamental strength"): (DISTRESS,),
    ("missing", "not available", "unavailable", "limitation", "gap", "incomplete"): (
        DATA_AVAILABILITY,
    ),
}


@dataclass
class RagAnswer:
    """A generated answer with everything needed to audit it."""

    question: str
    answer: str
    ticker: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    verification: VerificationReport = field(default_factory=VerificationReport)
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    query_type: str = "ask"
    log_id: int | None = None

    @property
    def groundedness(self) -> float:
        return self.verification.groundedness

    @property
    def citations(self) -> list[str]:
        return [chunk.citation() for chunk in self.chunks]

    def sources_block(self) -> str:
        if not self.chunks:
            return "Sources: none retrieved."
        return "Sources:\n" + "\n".join(
            f"  [{i}] {c.citation()} (similarity {c.score:.3f})"
            for i, c in enumerate(self.chunks, start=1)
        )


class RagPipeline:
    """Indexing and question answering over one or more companies."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        llm: GroqClient | None = None,
        top_k: int | None = None,
        block_on_unverified: bool | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self.top_k = top_k or settings.rag_top_k
        self.block_on_unverified = (
            settings.rag_block_on_unverified if block_on_unverified is None else block_on_unverified
        )

    # Built lazily so indexing does not pay for an LLM client, and answering
    # does not pay for a model load until it is actually needed.
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = get_vector_store("chroma")
        return self._store

    @property
    def llm(self) -> GroqClient:
        if self._llm is None:
            self._llm = GroqClient()
        return self._llm

    # --- indexing ----------------------------------------------------------

    def index_company(self, session: Session, ticker: str, period: str = "FY") -> int:
        """Chunk, embed and store one company. Re-running replaces its chunks."""
        ticker = ticker.strip().upper()
        chunks = build_chunks(session, ticker, period=period)
        if not chunks:
            logger.warning("Nothing to index for %s", ticker)
            return 0

        embeddings = self.embedder.embed([c.text for c in chunks])
        written = self.store.upsert(chunks, embeddings)
        logger.info("Indexed %d chunks for %s using %s", written, ticker, self.embedder.name)
        return written

    # --- retrieval ---------------------------------------------------------

    def retrieve(
        self,
        question: str,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        chunk_types: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Top-k chunks for a question, filtered to the company and period."""
        where: dict[str, Any] = {}
        if ticker:
            where["ticker"] = ticker.strip().upper()
        if fiscal_year is not None:
            # -1 is the profile chunk, which has no fiscal year but is always
            # relevant - excluding it would strip the company's own identity.
            where["fiscal_year"] = {"$in": [fiscal_year, -1]}
        if chunk_types:
            where["chunk_type"] = {"$in": list(chunk_types)}

        embedding = self.embedder.embed([question])[0]
        top_k = top_k or self.top_k
        results = self.store.query(embedding, top_k=top_k, where=where or None)

        if not chunk_types:
            results = self._ensure_intent_coverage(question, embedding, results, where, top_k)
        return results

    def _ensure_intent_coverage(
        self,
        question: str,
        embedding: Sequence[float],
        results: list[RetrievedChunk],
        where: dict[str, Any],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Guarantee the chunks a question's *intent* requires are present.

        Semantic similarity alone is unreliable for risk questions. "Is this
        company at risk of bankruptcy?" scores highly against the balance sheet
        while the chunk that actually answers it - the Altman Z-Score and its
        zone - can fall outside the top-k. Answering from the balance sheet means
        the model reasoning about solvency itself, which is exactly what the
        deterministic scores exist to prevent.

        So for a small set of recognisable intents the relevant chunk types are
        fetched directly and merged in. Deterministic, inspectable, and it only
        ever adds context that the rule engine already computed.
        """
        required = {
            chunk_type
            for keywords, types in INTENT_CHUNK_TYPES.items()
            if any(keyword in question.lower() for keyword in keywords)
            for chunk_type in types
        }
        if not required:
            return results

        present = {chunk.chunk_type for chunk in results}
        missing = required - present
        if not missing:
            return results

        # Re-query rather than plain fetch, so the added chunks carry a real
        # similarity score and the best one of each type is the one included.
        filters = dict(where)
        filters["chunk_type"] = {"$in": sorted(missing)}
        extra = self.store.query(embedding, top_k=len(missing), where=filters)
        if not extra:
            return results

        # Keep the ranked results first; append the intent chunks, then trim the
        # weakest similarity matches so the context stays within budget.
        known = {chunk.chunk_id for chunk in results}
        additions = [c for c in extra if c.chunk_id not in known]
        if not additions:
            return results

        keep = max(0, top_k - len(additions))
        merged = results[:keep] + additions
        logger.debug(
            "Intent coverage added %d chunk(s) of type %s", len(additions), sorted(missing)
        )
        return merged

    def context_for_summary(self, ticker: str, fiscal_year: int | None = None) -> list[RetrievedChunk]:
        """Chunks for an executive summary, selected structurally not by similarity.

        A summary must cover profile, ratios, distress scores, red flags and data
        limitations for the period. Similarity search would happily return five
        ratio chunks and no risk chunk, so the selection is explicit instead.
        """
        ticker = ticker.strip().upper()
        everything = self.store.get({"ticker": ticker}, limit=500)
        if not everything:
            return []

        if fiscal_year is None:
            years = [c.fiscal_year for c in everything if c.fiscal_year is not None]
            if not years:
                return everything[: self.top_k]
            fiscal_year = max(years)

        wanted = {PROFILE, RATIOS, DISTRESS, RED_FLAG, DATA_AVAILABILITY}
        selected = [
            chunk
            for chunk in everything
            if chunk.chunk_type in wanted
            and (chunk.fiscal_year == fiscal_year or chunk.fiscal_year is None)
        ]
        # Income statement gives the summary its headline revenue and profit.
        selected += [
            chunk
            for chunk in everything
            if chunk.chunk_type == "income_statement" and chunk.fiscal_year == fiscal_year
        ]
        return selected

    # --- generation --------------------------------------------------------

    def diagnose(self, session: Session | None, ticker: str) -> str:
        """Explain precisely which step is missing for this ticker.

        The generic "run ingestion first" message is actively misleading once a
        company has been ingested and analysed but never indexed - which is the
        normal state, because indexing is a separate command. Naming the actual
        missing step is the difference between a dead end and a fix.
        """
        ticker = ticker.strip().upper()
        if session is None:
            return (
                f"No indexed data for {ticker}. Index it with: "
                f"python -m src.rag.cli index {ticker}"
            )

        from sqlalchemy import select

        from src.db.models import Company, Ratio

        company = session.scalar(select(Company).where(Company.ticker == ticker))
        if company is None:
            return (
                f"{ticker} has not been ingested. Fetch it with: "
                f"python -m src.ingestion.cli {ticker}"
            )

        has_ratios = session.scalar(
            select(Ratio.id).where(Ratio.company_id == company.id).limit(1)
        )
        if not has_ratios:
            return (
                f"{ticker} is ingested but its ratios have not been computed. Run: "
                f"python -m src.ratios.cli {ticker}"
            )

        if self.store.count(ticker) == 0:
            return (
                f"{ticker} has been ingested and analysed but is not indexed for "
                f"retrieval yet, so there is nothing to ground an answer in. Index it "
                f"with: python -m src.rag.cli index {ticker}"
            )

        years = sorted(
            {c.fiscal_year for c in self.store.get({"ticker": ticker}, limit=500)}
            - {None}
        )
        return (
            f"No indexed data for {ticker} in the requested period. "
            f"Indexed fiscal years: {', '.join(f'FY{y}' for y in years) or 'none'}."
        )

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        chunks: list[RetrievedChunk],
        question: str,
        ticker: str,
        query_type: str,
        session: Session | None = None,
    ) -> RagAnswer:
        if not chunks:
            return RagAnswer(
                question=question,
                answer=self.diagnose(session, ticker),
                ticker=ticker,
                query_type=query_type,
                verification=VerificationReport(notes=["No context retrieved."]),
            )

        completion = self.llm.complete(system_prompt, user_prompt)
        answer, report = apply_guardrails(
            completion.text, chunks, block_on_unverified=self.block_on_unverified
        )
        if completion.was_truncated:
            report.notes.append(
                "The model hit its output limit; the answer may be cut off mid-sentence."
            )

        return RagAnswer(
            question=question,
            answer=answer,
            ticker=ticker,
            chunks=chunks,
            verification=report,
            model=completion.model,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            latency_seconds=completion.latency_seconds,
            query_type=query_type,
        )

    def ask(
        self,
        question: str,
        ticker: str,
        fiscal_year: int | None = None,
        session: Session | None = None,
        top_k: int | None = None,
    ) -> RagAnswer:
        """Answer a question about one company, grounded and verified."""
        ticker = ticker.strip().upper()
        chunks = self.retrieve(question, ticker=ticker, fiscal_year=fiscal_year, top_k=top_k)
        result = self._generate(
            SYSTEM_PROMPT,
            build_user_prompt(question, chunks),
            chunks,
            question,
            ticker,
            "ask",
            session=session,
        )
        self._log(session, result)
        return result

    def summarize(
        self,
        ticker: str,
        fiscal_year: int | None = None,
        session: Session | None = None,
    ) -> RagAnswer:
        """Generate a grounded executive summary for the latest period on file."""
        ticker = ticker.strip().upper()
        chunks = self.context_for_summary(ticker, fiscal_year)
        question = f"Executive summary for {ticker}" + (f" FY{fiscal_year}" if fiscal_year else "")
        result = self._generate(
            SYSTEM_PROMPT,
            build_summary_prompt(ticker, chunks),
            chunks,
            question,
            ticker,
            "summary",
            session=session,
        )
        self._log(session, result)
        return result

    # --- logging -----------------------------------------------------------

    def _log(self, session: Session | None, result: RagAnswer) -> None:
        """Persist the query, its context and its scores for later analysis."""
        if session is None or not settings.rag_log_queries:
            return
        try:
            result.log_id = repository.log_rag_query(session, result)
        except Exception as exc:  # noqa: BLE001 - logging must never break answering
            logger.warning("Could not log RAG query: %s", exc)
            # A failed flush leaves the session in a rolled-back-pending state;
            # clear it so a later commit() on the same session does not blow up
            # with PendingRollbackError.
            try:
                session.rollback()
            except Exception:  # noqa: BLE001
                pass


def index_all(session: Session, tickers: Sequence[str], pipeline: RagPipeline | None = None) -> dict[str, int]:
    """Index several companies, reporting chunks written per ticker."""
    pipeline = pipeline or RagPipeline()
    return {t.upper(): pipeline.index_company(session, t) for t in tickers}
