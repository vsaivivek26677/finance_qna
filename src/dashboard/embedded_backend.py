"""Run the FastAPI backend in-process, for hosts that only run the dashboard.

Streamlit Community Cloud starts a single process: ``streamlit run app.py``.
There is nowhere to also run ``uvicorn``. So when the dashboard boots and finds
no backend reachable at a loopback ``API_BASE_URL``, it starts one here in a
daemon thread. The dashboard's HTTP client then talks to ``127.0.0.1`` exactly
as it would to a separately hosted API - nothing else in the app changes.

Locally (where you run ``uvicorn`` yourself) or with ``API_BASE_URL`` pointed at
a real remote backend, this module does nothing but return that URL.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time

import httpx

logger = logging.getLogger(__name__)

_LOOPBACK = "127.0.0.1"
_PORT = int(os.environ.get("EMBEDDED_API_PORT", "8000"))
_HEALTH_TIMEOUT_S = 120
_start_lock = threading.Lock()
_started = False


def _is_loopback(url: str) -> bool:
    return "127.0.0.1" in url or "localhost" in url


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) == 0


def _health_ok(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/health", timeout=3.0).status_code == 200
    except Exception:  # noqa: BLE001 - any failure means "not ready"
        return False


def _bridge_secrets() -> None:
    """Copy Streamlit's ``st.secrets`` into ``os.environ`` so ``src.config``
    (which reads the environment) picks up keys set in the Cloud UI. Scalars
    only; a ``[section]`` in secrets.toml is left alone."""
    try:
        import streamlit as st

        items = list(st.secrets.items())
    except Exception:  # noqa: BLE001 - no secrets file locally is normal
        return

    for key, value in items:
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value)

    # settings is an lru_cache'd singleton; drop it so the bridged values win
    # even if something imported the config module before now.
    try:
        from src.config import get_settings

        get_settings.cache_clear()
    except Exception:  # noqa: BLE001
        pass


def _serve() -> None:
    import uvicorn

    config = uvicorn.Config(
        "src.api.main:app",
        host=_LOOPBACK,
        port=_PORT,
        log_level="warning",
        access_log=False,
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    # Signal handlers can only be installed on the main thread.
    server.install_signal_handlers = lambda: None
    server.run()


def ensure_backend() -> str:
    """Return the base URL the dashboard should call, starting a local API in a
    background thread first if one is needed and not already running."""
    global _started

    _bridge_secrets()

    configured = os.environ.get("API_BASE_URL", "").strip().rstrip("/")
    if configured and not _is_loopback(configured):
        return configured  # a real remote backend is configured; use it as-is

    base_url = f"http://{_LOOPBACK}:{_PORT}"

    if _health_ok(base_url):
        return base_url  # already running (local dev, or a warm rerun)

    with _start_lock:
        if not _started and not _port_open(_LOOPBACK, _PORT):
            threading.Thread(target=_serve, name="embedded-api", daemon=True).start()
            _started = True
            logger.info("Started embedded FastAPI backend on %s", base_url)

    deadline = time.time() + _HEALTH_TIMEOUT_S
    while time.time() < deadline:
        if _health_ok(base_url):
            return base_url
        time.sleep(1.0)

    logger.warning("Embedded backend did not report healthy within %ss", _HEALTH_TIMEOUT_S)
    return base_url  # hand it back anyway; the API client will show a clear error
