"""Text embedding, behind a small interface with a swappable backend.

The production backend is `sentence-transformers/all-MiniLM-L6-v2` running
locally: free, no API cost, no data leaving the machine.

A deterministic hashing backend is provided alongside it, and the test suite uses
it exclusively. Tests that silently download a 90MB model are slow, fail offline,
and make retrieval assertions depend on a model's behaviour rather than on the
pipeline's. The hashing embedder is a real bag-of-words vector space - good
enough that keyword retrieval genuinely works in tests - while staying instant
and dependency-free.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9._%-]*")


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into fixed-length vectors."""

    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


# Function words, plus the boilerplate this project's own chunker emits into
# almost every chunk ("not available", "reported by the data source", the ticker
# and period headers). Without stripping these, plain TF cosine scores every
# chunk highly on the words they all share, and topical terms stop deciding the
# ranking. A real sentence embedder handles this on its own; a bag-of-words one
# needs the help.
_STOPWORDS = frozenset(
    """
    a an and any are as at be been by did do does for from had has have how in
    is it its of on or that the their there these this to was were what when
    which who why will with
    above all also available been calculated could estimated field fields
    following have here listed must not period rather reported source sources
    statement than these those value values were
    """.split()
)


class HashingEmbedder:
    """Deterministic hashed bag-of-words vectors. No model download, no network.

    Term frequencies are hashed into a fixed number of buckets with sub-linear
    damping and L2 normalisation, so cosine similarity behaves like a classic TF
    vector space. Stopwords are dropped so that topical terms - "margin",
    "inventory", "bankruptcy" - actually decide the ranking.

    Lexical rather than semantic, which is what makes it a stable substrate for
    testing retrieval plumbing. It is not a substitute for the real model.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.name = f"hashing-{dimension}"

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return [
            token
            for token in _TOKEN_PATTERN.findall(text.lower())
            if token not in _STOPWORDS and not token.isdigit()
        ]

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            counts: dict[int, float] = {}
            for token in self._tokenise(text):
                bucket = self._bucket(token)
                counts[bucket] = counts.get(bucket, 0.0) + 1.0

            vector = [0.0] * self.dimension
            for bucket, count in counts.items():
                vector[bucket] = 1.0 + math.log(count)  # damp repeated terms

            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
            vectors.append(vector)
        return vectors


class SentenceTransformerEmbedder:
    """Local `sentence-transformers` model. Loaded lazily on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.name = model_name
        self._model = None
        self._dimension: int | None = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "sentence-transformers is not installed. Run "
                    "`pip install -r requirements.txt`, or set EMBEDDING_BACKEND=hashing."
                ) from exc
            logger.info("Loading embedding model %s (first run downloads it)", self.name)
            self._model = SentenceTransformer(self.name)
            self._dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load()
        return int(self._dimension or 384)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return [[float(x) for x in row] for row in vectors]


def get_embedder(backend: str | None = None, model_name: str | None = None) -> Embedder:
    """Build the configured embedder. `backend` is 'sentence-transformers' or 'hashing'."""
    from src.config import settings  # local import avoids a config import cycle

    backend = (backend or settings.embedding_backend).strip().lower()
    if backend in {"hashing", "hash", "test"}:
        return HashingEmbedder()
    if backend in {"sentence-transformers", "sentence_transformers", "st", "default"}:
        return SentenceTransformerEmbedder(model_name or settings.embedding_model)
    raise ValueError(f"unknown embedding backend: {backend!r}")


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, safe on zero vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
