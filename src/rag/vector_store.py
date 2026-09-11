"""Vector storage and retrieval.

Two backends behind one interface. ChromaDB is the persistent local store used in
anger; an in-memory store with exact cosine search backs the tests, which keeps
them fast, hermetic, and free of on-disk state that could leak between cases.

Chunk ids are deterministic (`TICKER:FY2024:income_statement`), so re-indexing a
company overwrites its entries rather than accumulating duplicates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

from src.rag.chunker import Chunk
from src.rag.embedder import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned from a search, with its similarity score."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float

    @property
    def ticker(self) -> str:
        return str(self.metadata.get("ticker", ""))

    @property
    def fiscal_year(self) -> int | None:
        year = self.metadata.get("fiscal_year")
        return None if year in (None, -1) else int(year)

    @property
    def chunk_type(self) -> str:
        return str(self.metadata.get("chunk_type", ""))

    @property
    def period_label(self) -> str:
        year = self.fiscal_year
        if year is None:
            return "all periods"
        period = self.metadata.get("period", "FY")
        return f"FY{year}" if period == "FY" else f"{period} {year}"

    def source_values(self) -> dict[str, float]:
        """The verified numbers behind this chunk, as stored at index time."""
        raw = self.metadata.get("source_values_json")
        if not raw:
            return {}
        try:
            return {k: float(v) for k, v in json.loads(raw).items()}
        except (ValueError, TypeError):
            return {}

    def citation(self) -> str:
        label = self.chunk_type.replace("_", " ")
        return f"{self.ticker} {self.period_label} {label}".strip()


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int: ...

    def query(
        self,
        embedding: Sequence[float],
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...

    def get(self, where: dict[str, Any] | None = None, limit: int = 200) -> list[RetrievedChunk]: ...

    def count(self, ticker: str | None = None) -> int: ...

    def delete_ticker(self, ticker: str) -> int: ...


def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Equality / `$in` filtering, shared by both backends."""
    if not where:
        return True
    for key, condition in where.items():
        value = metadata.get(key)
        if isinstance(condition, dict):
            if "$in" in condition and value not in condition["$in"]:
                return False
            if "$ne" in condition and value == condition["$ne"]:
                return False
        elif value != condition:
            return False
    return True


class InMemoryVectorStore:
    """Exact cosine search over a dict. Used by the tests."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        for chunk, embedding in zip(chunks, embeddings):
            self._items[chunk.chunk_id] = (chunk, list(embedding))
        return len(chunks)

    def query(
        self,
        embedding: Sequence[float],
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        scored = []
        for chunk, vector in self._items.values():
            metadata = chunk.metadata()
            if not _matches(metadata, where):
                continue
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=metadata,
                    score=cosine_similarity(embedding, vector),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def get(self, where: dict[str, Any] | None = None, limit: int = 200) -> list[RetrievedChunk]:
        found = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id, text=chunk.text, metadata=chunk.metadata(), score=1.0
            )
            for chunk, _ in self._items.values()
            if _matches(chunk.metadata(), where)
        ]
        found.sort(key=lambda c: c.chunk_id)
        return found[:limit]

    def count(self, ticker: str | None = None) -> int:
        if ticker is None:
            return len(self._items)
        return sum(1 for chunk, _ in self._items.values() if chunk.ticker == ticker.upper())

    def delete_ticker(self, ticker: str) -> int:
        ticker = ticker.upper()
        doomed = [cid for cid, (chunk, _) in self._items.items() if chunk.ticker == ticker]
        for chunk_id in doomed:
            del self._items[chunk_id]
        return len(doomed)


class ChromaVectorStore:
    """Persistent local ChromaDB collection.

    Embeddings are always supplied explicitly rather than letting Chroma pick its
    own embedding function, so the same model is guaranteed to be used at index
    time and at query time.
    """

    def __init__(self, path: str | None = None, collection: str | None = None) -> None:
        from src.config import settings  # local import keeps config out of the import cycle

        try:
            import chromadb  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "chromadb is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self.path = path or settings.chroma_dir
        self.collection_name = collection or settings.chroma_collection
        self._client = chromadb.PersistentClient(path=self.path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB collection %r ready at %s", self.collection_name, self.path)

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
            embeddings=[list(e) for e in embeddings],
        )
        return len(chunks)

    def query(
        self,
        embedding: Sequence[float],
        top_k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where=_to_chroma_where(where),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=document or "",
                metadata=dict(metadata or {}),
                # Chroma returns cosine *distance*; convert back to similarity.
                score=1.0 - float(distance),
            )
            for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
        ]

    def get(self, where: dict[str, Any] | None = None, limit: int = 200) -> list[RetrievedChunk]:
        """Fetch by metadata filter rather than by similarity."""
        found = self._collection.get(
            where=_to_chroma_where(where), limit=limit, include=["documents", "metadatas"]
        )
        chunks = [
            RetrievedChunk(
                chunk_id=chunk_id, text=document or "", metadata=dict(metadata or {}), score=1.0
            )
            for chunk_id, document, metadata in zip(
                found.get("ids") or [],
                found.get("documents") or [],
                found.get("metadatas") or [],
            )
        ]
        chunks.sort(key=lambda c: c.chunk_id)
        return chunks

    def count(self, ticker: str | None = None) -> int:
        if ticker is None:
            return int(self._collection.count())
        found = self._collection.get(where={"ticker": ticker.upper()}, include=[])
        return len(found.get("ids") or [])

    def delete_ticker(self, ticker: str) -> int:
        existing = self.count(ticker)
        if existing:
            self._collection.delete(where={"ticker": ticker.upper()})
        return existing

    def reset(self) -> None:
        """Drop and recreate the collection. Used by the reindex CLI flag."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )


def _to_chroma_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Chroma needs an explicit `$and` once a filter has more than one key."""
    if not where:
        return None
    if len(where) == 1:
        return where
    return {"$and": [{key: value} for key, value in where.items()]}


def get_vector_store(backend: str = "chroma", **kwargs: Any) -> VectorStore:
    if backend == "memory":
        return InMemoryVectorStore()
    if backend == "chroma":
        return ChromaVectorStore(**kwargs)
    raise ValueError(f"unknown vector store backend: {backend!r}")
