# Deploying to Streamlit Community Cloud

Streamlit Community Cloud runs **one process**: `streamlit run <entrypoint>`.
This project needs the FastAPI backend too, so the dashboard starts it itself.

## How the single-process setup works

`src/dashboard/embedded_backend.py` runs at dashboard start-up:

1. Copies `st.secrets` into the process environment, so `src/config.py` (which
   reads env vars) sees `FMP_API_KEY`, `GROQ_API_KEY`, etc.
2. Checks whether an API is already reachable at `API_BASE_URL`
   (default `http://127.0.0.1:8000`).
3. If not, and the URL is loopback, it starts `uvicorn` on `127.0.0.1:8000` in a
   **daemon thread** and waits (up to 120 s) for `/health` to go green.
4. The dashboard's HTTP client then calls `127.0.0.1:8000` exactly as it would a
   remote backend — no other code changes.

Point `API_BASE_URL` at a real URL (secret or env var) and the embedded backend
is skipped entirely — that's the two-host mode described at the bottom.

## What is committed for the deploy

| Path | Why it's in the repo |
|---|---|
| `data/financials.db` | 47 companies × 5 years. FMP's free tier no longer serves most symbols, so it can't be fully rebuilt. |
| `data/chroma/` | The 2,556-chunk retrieval index. Committed so RAG works on boot without a 2-minute rebuild. |
| `models/distress_model.joblib` | The trained Layer 4 model. |

## Steps

1. **Push the repo to GitHub** (public, or private with Streamlit Cloud granted access).

2. Go to **share.streamlit.io** → *Create app* → *Deploy from a repo*.
   - Repository: `<your-username>/finance_qna`
   - Branch: `main`
   - **Main file path: `src/dashboard/app.py`**
   - Python version: **3.12** (also set in `runtime.txt`)

3. **Advanced settings → Secrets.** Paste as TOML (no `[section]` header):

   ```toml
   FMP_API_KEY = "your_free_fmp_key"
   GROQ_API_KEY = "your_free_groq_key"
   ```

   `FMP_API_KEY` is only needed if you want the in-app *Ingest* button to work;
   every other tab runs off the committed database. `GROQ_API_KEY` powers the AI
   Summary / Ask AI tabs — without it those tabs show a clear "not configured"
   notice and the rest of the app is unaffected.

4. **Deploy.** First boot takes a while: pip installs torch + chromadb +
   sentence-transformers (~3–5 min), then the first AI question downloads the
   90 MB embedding model. After that it's fast. Reboots reuse the pip cache.

## Resource limits — read this

Community Cloud gives each app about **1 GB RAM**. This app loads Streamlit,
FastAPI, ChromaDB, a sentence-transformer (via PyTorch) and scikit-learn in **one
process**. It usually fits, but a cold RAG query can push it close to the limit,
and you may see the app restart under load.

If it won't stay up:

- **Drop the AI layer.** Remove `chromadb`, `sentence-transformers` and the
  `--extra-index-url` line from `requirements.txt`, and set a secret
  `GROQ_API_KEY = ""`. The AI Summary / Ask AI tabs then show "not configured";
  Statements, Ratio analytics, the distress model and Red flags all keep
  working, at roughly half the memory footprint.
- **Or split the hosts** (below), which is the robust option.

## Alternative: two hosts

Run the API on a small always-on host and keep only the dashboard on Streamlit
Cloud.

1. Deploy the FastAPI app anywhere that runs a container or a Python web service
   (Render / Railway / Fly.io free tiers): start command
   `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`, same
   `requirements.txt`, same secrets.
2. In the Streamlit Cloud app's secrets, add:
   ```toml
   API_BASE_URL = "https://your-api-host.onrender.com"
   ```
   The dashboard detects the non-loopback URL and skips the embedded backend.

## Testing deploy mode locally

```bash
# no separate `uvicorn` running — the dashboard should start one itself
python -m streamlit run src/dashboard/app.py
```

Put your keys in `.env` (local) rather than `secrets.toml`; the bridge reads
`st.secrets` only when a secrets file exists, and `src/config.py` loads `.env`
regardless.
