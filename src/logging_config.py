"""Single logging setup used by CLI entry points."""

from __future__ import annotations

import logging
import sys

from src.config import settings

_CONFIGURED = False


def _make_stdout_unicode_safe() -> None:
    """Stop a Windows console from crashing on non-cp1252 output.

    LLM responses routinely contain characters the legacy Windows codepage
    cannot encode (narrow no-break spaces, en dashes, curly quotes). Without
    this, printing a perfectly good answer raises UnicodeEncodeError and the
    command dies after the model has already been paid for.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - redirected streams
            pass


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Safe to call repeatedly."""
    global _CONFIGURED
    _make_stdout_unicode_safe()
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, (level or settings.log_level).upper(), logging.INFO))
    root.handlers = [handler]
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)
    _CONFIGURED = True
