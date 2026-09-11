#!/usr/bin/env bash
# Start the Streamlit dashboard. Needs the API running (see start_api.sh).
set -euo pipefail
cd "$(dirname "$0")"
export API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
exec streamlit run src/dashboard/app.py --server.port "${STREAMLIT_PORT:-8501}"
