"""Central configuration, loaded from environment / .env.

Every layer imports `settings` from here rather than reading os.environ directly,
so there is exactly one place that knows how the app is configured.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CACHE_DIR = DATA_DIR / "raw"


class Settings(BaseSettings):
    """Application settings. Values come from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Secrets -----------------------------------------------------------
    fmp_api_key: str | None = Field(default=None, description="Financial Modeling Prep API key")
    groq_api_key: str | None = Field(default=None, description="Groq API key (Layer 3)")

    # --- Storage -----------------------------------------------------------
    database_url: str = Field(default=f"sqlite:///{DATA_DIR.as_posix()}/financials.db")

    # --- FMP client --------------------------------------------------------
    fmp_base_url: str = "https://financialmodelingprep.com"
    fmp_timeout_seconds: float = 30.0
    fmp_max_retries: int = 3
    # Free tier allows 250 requests/day. The client refuses to exceed this in a
    # single process run so a bad loop cannot silently burn the whole quota.
    fmp_daily_request_budget: int = 250

    # --- Ingestion defaults ------------------------------------------------
    ingest_default_years: int = 5
    ingest_cache_raw_json: bool = True

    # --- Layer 3: RAG ------------------------------------------------------
    # 'sentence-transformers' runs all-MiniLM-L6-v2 locally; 'hashing' is the
    # deterministic offline backend the test suite uses.
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = str(DATA_DIR / "chroma")
    chroma_collection: str = "financial_facts"

    # Groq retires model ids on a rolling basis - the Llama 3.x ids this project
    # was originally specified against are already gone. Run
    # `python -c "from src.rag.groq_client import GroqClient; print(GroqClient().list_models())"`
    # to see what a key can currently reach, and override via GROQ_MODEL.
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "openai/gpt-oss-20b"
    # Factual extraction, not prose: near-greedy decoding keeps numbers stable.
    groq_temperature: float = 0.1
    groq_max_tokens: int = 1200
    groq_timeout_seconds: float = 60.0
    groq_max_retries: int = 3

    rag_top_k: int = 8
    # Flag unverifiable numbers by default; set true to suppress the answer
    # entirely when the model states a figure absent from the context.
    rag_block_on_unverified: bool = False
    rag_log_queries: bool = True

    # --- Layer 4: Distress prediction ------------------------------------
    # A calibrated classifier trained on the sowide/bankruptcy_dataset American
    # public-company set (~78.7k firm-years, ~6.6% bankrupt, NYSE/NASDAQ
    # 1999-2018), applied to liability-scaled ratios our engine already
    # computes. See src/prediction/.
    distress_model_path: str = str(PROJECT_ROOT / "models" / "distress_model.joblib")
    distress_data_dir: str = str(DATA_DIR / "reference" / "us_bankruptcy")
    distress_dataset_url: str = (
        "https://raw.githubusercontent.com/sowide/bankruptcy_dataset/main/"
        "american_bankruptcy_dataset.csv"
    )

    # --- Misc --------------------------------------------------------------
    api_base_url: str = "http://localhost:8000"
    log_level: str = "INFO"

    @property
    def raw_cache_dir(self) -> Path:
        return RAW_CACHE_DIR

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
