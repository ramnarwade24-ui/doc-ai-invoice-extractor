#!/usr/bin/env bash
set -euo pipefail

# Live demo launcher (Streamlit).
# Opens the Streamlit app which includes the "Judge Demo" tab.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8501}"

cd "$REPO_ROOT"

echo "[INFO] Launching Streamlit Judge Demo..."
echo "[INFO] URL: http://localhost:${PORT}"
echo "       If running in a container, forward port ${PORT}."

export STREAMLIT_SERVER_PORT="$PORT"
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

streamlit run doc_ai_project/app.py
