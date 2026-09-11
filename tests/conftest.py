"""Shared pytest fixtures.

Each test that touches the database gets its own throwaway SQLite file, so the
suite never reads or writes the developer's real `data/financials.db`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from src.config import settings
from src.db.database import init_db, reset_engine, session_scope
from src.ingestion.fmp_client import FMPClient


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank out real API keys for every test.

    Without this the suite behaves differently depending on whether the
    developer has a populated `.env`, and a mis-mocked HTTP test could spend
    real free-tier quota against the live API.
    """
    monkeypatch.setattr(settings, "fmp_api_key", None, raising=False)
    monkeypatch.setattr(settings, "groq_api_key", None, raising=False)


@pytest.fixture(autouse=True)
def deterministic_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the hashing embedder throughout the suite.

    Downloading a 90MB transformer would make the tests slow, fail offline,
    and tie retrieval assertions to a model's behaviour rather than to the
    pipeline's.
    """
    monkeypatch.setattr(settings, "embedding_backend", "hashing", raising=False)


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Point the app at an isolated SQLite database for the duration of a test."""
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setattr(settings, "database_url", url, raising=False)
    monkeypatch.setattr(settings, "ingest_cache_raw_json", False, raising=False)
    reset_engine()
    init_db()
    yield url
    reset_engine()


@pytest.fixture
def db_session(temp_db: str):
    """A transactional session against the temporary database."""
    with session_scope() as session:
        yield session


@pytest.fixture
def fmp_client() -> FMPClient:
    """Client with a dummy key and caching off, for respx-mocked HTTP tests."""
    return FMPClient(
        api_key="test-key",
        base_url="https://financialmodelingprep.com",
        max_retries=2,
        cache_raw=False,
    )
