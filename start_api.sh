#!/usr/bin/env bash
# Start the FastAPI backend. Docs at http://localhost:8000/docs
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn src.api.main:app --reload --host 0.0.0.0 --port "${PORT:-8000}"
