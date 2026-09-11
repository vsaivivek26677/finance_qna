# Financial Analytics Platform — As-Built Design

### Financial statement analysis, ratio engine, grounded LLM Q&A, and a calibrated distress-probability model

This document describes the system **as it was actually built**. It began as a
forward-looking plan; that plan changed in several places during
implementation, and the changes are recorded in
[§9 Deviations from the original plan](#9-deviations-from-the-original-plan).
The [README](README.md) is the usage guide; this is the design rationale.

---

## 1. What it does

1. Ingests raw **financial statements** (income statement, balance sheet, cash
   flow) plus company profile and price history for a ticker, from the Financial
   Modeling Prep free tier and `yfinance`.
2. Computes **35 financial ratios** across seven categories, the **Altman Z /
   Z''**, **Piotroski F** and **Beneish M** composite scores, and **15
   rule-based red flags**. Nothing is estimated to fill a gap — an input that is
   missing produces an explicit "not calculable" with the reason named.
3. Runs a **RAG pipeline** (ChromaDB + local embeddings + a free Groq LLM) that
   answers questions and writes executive summaries **only** from the verified
   figures above, with every number in the output checked against its source
   after generation.
4. Estimates the **probability of financial distress** with a classifier trained
   on ~78.7k labelled American-company firm-years from an external dataset,
   applied to ratios scaled by total liabilities.
5. Serves all of it through a **FastAPI** backend, presented by a **Streamlit**
   dashboard that holds no database, model or API key of its own.

---

## 2. Architecture

```
        Financial Modeling Prep  +  yfinance
                     │
                     ▼
        ┌───────────────────────────────┐
        │ Layer 1  Ingestion            │  fetch → validate → store
        │  - explicit missing-data map  │  (never drops or zero-fills)
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
             SQLite  (SQLAlchemy ORM)     ◄─────────────────┐
        └──────────────┬────────────────┘                   │
                       ▼                                     │
        ┌───────────────────────────────┐                   │
        │ Layer 2  Ratio engine         │  35 ratios,       │
        │  - Altman / Piotroski / Beneish│  15 red-flag rules│
        └──────────────┬────────────────┘                   │
             ┌─────────┴──────────┐                          │
             ▼                    ▼                          │
   ┌──────────────────┐  ┌──────────────────────┐            │
   │ Layer 3  RAG+LLM │  │ Layer 4  Distress    │            │
   │  ChromaDB        │  │  probability model   │            │
   │  Groq (free)     │  │  trained on US firms  │            │
   │  number + period │  │  calibrated, banded  │            │
   │  verification    │  │  (scikit-learn)      │            │
   └────────┬─────────┘  └──────────┬───────────┘            │
            └──────────┬────────────┘                        │
                       ▼                                     │
        ┌───────────────────────────────┐                   │
        │ Layer 5a  FastAPI backend     │───────────────────┘
        │  (owns all business logic)    │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Layer 5b  Streamlit dashboard │  thin HTTP client
        └───────────────────────────────┘
```

**The Streamlit app never touches the database, the model or Groq.** It calls
the FastAPI backend over HTTP. That split is the point of having a service layer:
the backend could serve a different frontend unchanged, and the business logic is
in one place and independently testable.

Everything runs as plain Python processes — `uvicorn` for the API, `streamlit
run` for the dashboard. No Docker.

---

## 3. Tech stack (all free / open-source)

| Purpose | Tool |
|---|---|
| Backend API | FastAPI + `uvicorn`, `cachetools` for TTL caching |
| HTTP | `httpx` (+ `respx` for tests), `tenacity` for retries |
| Validation | `pydantic` v2 / `pydantic-settings` — also the FastAPI request/response models |
| Storage | SQLite via SQLAlchemy 2.0 ORM |
| Data | `pandas`, `numpy` |
| Statements | Financial Modeling Prep free tier; `yfinance` for price / market cap |
| Vector store | ChromaDB (local, persistent) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local); a deterministic hashing backend for tests |
| LLM | Groq free tier — `openai/gpt-oss-120b`, fallback `openai/gpt-oss-20b` |
| RAG evaluation | a deterministic reference-based harness (not `ragas` — see §9) |
| ML | `scikit-learn` (`HistGradientBoostingClassifier` + isotonic calibration); `joblib` for the artifact; `pandas` reads the training CSV |
| Dashboard | Streamlit + Plotly |
| Tests | `pytest` (510, no network) |

---

## 4. Layer 1 — Data Ingestion

```
src/ingestion/
    field_maps.py       canonical field name → source aliases
    schemas.py          pydantic records (reused as FastAPI models)
    normalizer.py       raw JSON → validated record + missing map (pure, no I/O)
    fmp_client.py       FMP HTTP client: retries, quota guard, API-variant fallback
    yfinance_client.py  price history, share count, market cap
    repository.py       upserts keyed on natural business keys
    ingest_pipeline.py  fetch → normalize → store, with an auditable report
    cli.py              command-line entry point
```

Three decisions the later layers depend on:

- **Nothing is silently dropped or zero-filled.** Every numeric field is
  optional; a field without a value gets an entry in the record's missing map
  naming *why* — `not_reported`, `null_in_source` or `unparseable`. That map is
  persisted per row. A `NULL` in the database is therefore never ambiguous.
- **Both FMP API generations are supported.** FMP is migrating `/api/v3/...` →
  `/stable/...` with different field spellings; `field_maps.py` lists the aliases
  each canonical field may arrive under, and the client retries against the other
  generation on a plan-availability error, then sticks with whichever worked.
  (FMP has since been disabling legacy `/api/v3` endpoints on the free tier
  entirely for many symbols — those come back as an HTTP 403 plan limit and are
  recorded, not retried into the ground.)
- **Failures degrade, they do not abort.** A missing profile or one
  unnormalizable row produces a `partial` run with the problem recorded, and
  every attempt writes an `ingestion_runs` audit row.

**Schema.** `companies`, `income_statements`, `balance_sheets`,
`cash_flow_statements`, `market_data`, `ingestion_runs`, plus the downstream
tables `ratios`, `red_flags`, `rag_eval_logs`, `distress_predictions` — all
declared in `src/db/models.py` and created up front, so later layers add logic,
not migrations. Upserts key on the natural business key, so re-running a ticker
refreshes rows rather than duplicating history. `companies.is_financial_sector`
is set at ingestion because the Altman model is not defined for banks.

**Quota safety.** The FMP client counts its own requests and refuses to exceed
`FMP_DAILY_REQUEST_BUDGET` (250) in one process, honours `Retry-After` on 429,
retries transient 5xx, and detects the error objects FMP returns inside HTTP 200
bodies. Raw responses are cached under `data/raw/<TICKER>/`.

---

## 5. Layer 2 — Ratio Engine, Distress Scores & Red Flags

```
src/ratios/
    base.py             RatioResult, PeriodBundle, NotCalculable, the registry
    liquidity.py profitability.py leverage.py efficiency.py valuation.py cash_flow.py
    distress_scores.py  Altman Z / Z'', Piotroski F, Beneish M
    red_flags.py        15 deterministic rules with tunable thresholds
    ratio_engine.py     load → compute → detect → persist
    repository.py       persistence and read helpers
    cli.py              command-line entry point
```

| Category | Ratios |
|---|---|
| Liquidity | Current, Quick, Cash, Working Capital |
| Profitability | Gross / Operating / Net Margin, ROE, ROA, ROIC |
| Leverage | Debt-to-Equity, Debt-to-Assets, Liabilities-to-Assets, Interest Coverage, Net Debt/EBITDA |
| Efficiency | Asset / Inventory / Receivables Turnover, DSO, DIO |
| Valuation | P/E, P/B, P/S, Enterprise Value, EV/EBITDA, Diluted EPS |
| Cash Flow | FCF, FCF Margin, OCF Margin, Cash Conversion, Capex/OCF |
| Distress | Altman Z, Altman Z'', Piotroski F, Beneish M |

- **A ratio is computed or explicitly refused — never estimated.** Refusals also
  cover ratios that would be *misleading*: P/E on negative earnings, D/E on
  negative equity, inventory turnover on zero inventory.
- **Composite scores persist their components.** Every Altman term, all nine
  Piotroski signals, all eight Beneish indices are stored, so Layer 3 narrates
  pre-computed facts rather than inferring them.
- **Sector applicability is enforced and disclosed.** Banks and insurers are
  excluded from both Altman variants and from 8 of the 15 red-flag rules (a
  bank's interest expense is a cost of funding, its balance sheet is not
  split current/non-current, lending drives operating cash flow negative). An
  `Info` flag names exactly which rules were skipped.
- **Market data is paired to the period, not to today** — the price observation
  nearest the fiscal period end, within 45 days, or no valuation ratio.
- **Flags are evaluated per period against only that period's history** — no
  lookahead.

The Altman Z''-Score's four components (X1–X5 minus the asset-turnover term)
are also the backbone of the Layer 4 feature set.

---

## 6. Layer 3 — RAG, Grounded Q&A & Evaluation

```
src/rag/
    chunker.py        DB rows → labelled, retrievable fact chunks
    embedder.py       local all-MiniLM-L6-v2, plus a deterministic test backend
    vector_store.py   ChromaDB (persistent) and an in-memory store for tests
    groq_client.py    retries, rate-limit handling, model fallback
    guardrails.py     system prompt + post-generation verification
    rag_pipeline.py   index → retrieve → generate → verify → log
    evaluation.py     ground-truth eval set and scoring
    repository.py     query logging and groundedness aggregation
    cli.py            command-line entry point
```

- **Missing data is stated, never omitted.** Every chunk names what is absent
  (`NOT AVAILABLE ... interest_expense (not reported)`), and each period gets a
  dedicated data-availability chunk.
- **Every number the model writes is checked against the numbers it was given.**
  Matching is precision-aware (`391,035,000,000` → `391.04B` passes, `395B`
  does not). Period citations are verified separately. Groundedness is reported
  per answer and logged for every query.
- **Risk questions always retrieve the risk chunks**, and **summaries are
  selected structurally** (performance + position + risk + data limitations)
  rather than by similarity, which would happily return five ratio chunks and no
  risk chunk.
- **Indexing is a separate step from ingestion.** `/health` reports
  `companies_indexed` against `companies_analysed`; the dashboard warns when they
  differ; a query that retrieves nothing names the step that is actually missing.

**Evaluation is reference-based and deterministic.** The questions are generated
*from the database*, so both the correct answer and the chunk that should supply
it are known exactly. `faithfulness` is the guardrail check (no second LLM),
`answer_correctness` is a precision-aware numeric match against the DB,
`context_recall` / `context_precision` measure retrieval, and
`refusal_correctness` — which has no `ragas` equivalent — checks that a refusal
does not still slip in a number. A live Apple run (five years, 63 chunks, 25
questions) scores 1.000 on every metric except context precision (0.809).

---

## 7. Layer 4 — Distress Prediction

```
src/prediction/
    features.py     the 8-feature spec, defined once; shared by training and scoring
    dataset.py      download, cache and parse the American bankruptcy CSV
    model.py        train, calibrate, cross-validate, compare to baseline, persist
    predict.py      load the artifact and score one feature vector
    repository.py   build features from stored statements (reuses Layer 2's market-data pairing); persist estimates
    cli.py          train / score / report
```

**The question.** "Which zone does the textbook formula put this company in" is
Layer 2's job. Layer 4 asks a different one: "how often did firms that looked
like this one actually fail". It is a learned weighting of a liability-scaled
ratio set, not a second formula.

**Why not train on our own data.** ~235 company-years across ~47 companies, with
panels four steps long, cannot support a credible forecasting model. Layer 4
borrows scale from an external labelled set instead.

**Second training set.** The first version of this layer trained on the UCI
*Polish companies bankruptcy data* (2000s Polish SMEs) — a working model, but a
domain mismatch against the US large-caps this platform scores. It was replaced
with [sowide/bankruptcy_dataset](https://github.com/sowide/bankruptcy_dataset),
American public companies (NYSE/NASDAQ, 1999–2018). That source has a verified
defect in its asset-side columns (`Total Current Liabilities` is byte-identical
to `Total Liabilities` in every row; `Total Assets` is smaller than
`Current Assets` in 72% of rows), so every feature here is scaled by total
liabilities instead — the one size-anchor column that behaves consistently.

| | |
|---|---|
| Training data | 78,682 firm-years, 8,971 NYSE/NASDAQ companies, 1999–2018, 6.6% bankrupt within the forecast horizon |
| Features | 8 liability-scaled ratios: net income, EBIT, EBITDA, retained earnings, revenue and market value each over total liabilities, plus gross margin and the operating-expense ratio |
| Model | `HistGradientBoostingClassifier` (`max_depth=3`, class-weighted) + isotonic calibration; extreme ratios winsorised to a learned 1st–99th percentile band |
| Persistence | one `joblib` artifact carrying the estimator, feature list, per-feature median/IQR, permutation importances, CV metrics and provenance |

**Measured** (5-fold CV, out of fold, vs a naive unweighted linear baseline on the same features):

| Metric | Model | Baseline |
|---|---|---|
| ROC-AUC | 0.699 | 0.520 |
| PR-AUC (base rate 6.6%) | 0.138 | 0.071 |
| Brier | 0.060 | — |

The learned model nearly doubles the baseline's PR-AUC. ROC-AUC in the
high-0.6s is modest — the cost of deliberately excluding the dataset's broken
columns rather than using all 18 raw variables the source paper did — but it is
now trained on the population it actually scores.

**Design rules, consistent with the rest of the stack:**

- **A period with fewer than 5 of 8 features is refused**, `is_scored: false`
  with the missing inputs named, rather than scored off imputed medians.
- **The band, not the decimal, is the signal.** Isotonic calibration is
  piecewise-constant; two nearby estimates can land on the same calibrated
  value. Factor attribution is only shown once an estimate is off the floor.
- **Scoring is wired into ingestion** — `POST /ingest/{ticker}` fetches,
  analyses, indexes *and* scores — so a freshly added company is never a dead end.

---

## 8. Layer 5 — API & Dashboard

```
src/api/
    main.py               app instance, /health, CORS, exception handler
    dependencies.py       DB session, TTL caches, RAG pipeline singleton
    routers/
        companies.py ratios.py predictions.py rag.py ingestion.py
    schemas.py            response contract

src/dashboard/
    app.py                page layout and tab routing
    api_client.py         typed HTTP client; every failure becomes a message
    components/
        theme.py          palette, type scale, CSS, shared Plotly layout
        charts.py         figures and all number formatting
        panels.py         KPI rows, flag cards, statement tables, model readouts
```

**Endpoints.** `/health`; `/companies` and `/companies/{ticker}` (+ `/statements`,
`/ratios`, `/ratios/{name}/series`, `/distress-score`, `/distress-prediction`,
`/red-flags`, `/summary`); `POST /companies/{ticker}/ask`; `POST /ingest/{ticker}`.
OpenAPI docs at `/docs`.

**API design:**

- **The contract reports absence as explicitly as presence** —
  `is_calculable: false` with a reason, `missing_fields` on statements,
  `groundedness` on answers, `is_scored: false` with a reason on predictions.
- **Only the generative endpoints are `async def`**; the blocking Groq call is
  pushed to a worker thread. Database endpoints are plain `def` (FastAPI already
  threadpools them) — making them `async` would block the loop on sync SQLAlchemy.
- **Caching protects quota, not the database** — 60 s for reads, 15 min for
  generated summaries; ingestion clears the caches.
- **Errors carry the fix** — a 404 names the ingest call, a missing Groq key is a
  503 explaining the rest still works, an upstream LLM failure is a 502.
- The embedding model is **warmed in a background thread at start-up**.

**Dashboard.** Six tabs (Overview, Statements, Ratio analytics, Risk & red flags,
AI summary, Ask AI). Dark navy ground, one teal accent, colour reserved for
severity. Three details that matter more than they look:

- **Units are decided per field, not per type** — a margin is `26.92%`, a
  multiple `1.52`, an absolute `-17.67B`, EPS `7.24`, a share count `14.95B`, and
  a percentage ratio's YoY change is in points (`+2.94 pts`), not a relative
  percent.
- **Deltas are coloured by meaning, not by sign** — falling leverage reads green.
- **Absence is visible** — statement gaps as `N/A — not reported`, uncomputable
  ratios behind an expander with reasons, every answer with a groundedness pill
  and its sources.

The distress probability appears on the Overview (next to the Altman gauge) and
in full on the Risk tab (probability by year, the risky-side features, and an
expander with the model's CV metrics and limitations).

**Deployment (documented, not built).** Streamlit Community Cloud hosts only the
Streamlit process, so the FastAPI backend needs its own host (Render / Railway /
Fly.io free tier) with `API_BASE_URL` pointed at it; SQLite will not persist on
Streamlit Cloud, so a public demo needs free-tier hosted Postgres via
`DATABASE_URL`; secrets go in Streamlit's Secrets Manager.

---

## 9. Deviations from the original plan

The original plan specified more than what shipped. Each of these was a
deliberate call; the README's
[What was planned but not built](README.md#what-was-planned-but-not-built-and-why)
section carries the full rationale.

| Planned | Built instead | Reason (short) |
|---|---|---|
| `ragas` RAG evaluation | Deterministic reference-based harness | `ragas` pins pre-1.0 LangChain; DB-generated questions make the metrics deterministic anyway |
| XGBoost forecasting (revenue growth, future ROE) + walk-forward backtesting | Layer 4 distress model trained on an external labelled set | ~190 company-years cannot support a credible forecast |
| MLflow tracking + model registry | Metrics + provenance inside the `joblib` artifact; `cli report` | Lifecycle tooling for one model with one training routine is ceremony |
| SHAP explainability | Permutation importance (global) + median-deviation attribution (per company) | Ten features; SHAP's plots would not add signal here |
| Peer clustering, anomaly detection, sector classifier, peer benchmarks, temporal deviation | Removed entirely — tables dropped, artifacts deleted | Clusters of 3–4 on ~28 companies; anomaly scores near random on ~140 rows; the sector classifier re-predicted a field FMP already returns |
| CI (GitHub Actions) + live Streamlit Cloud deployment | `python -m pytest` as the gate; deployment documented | Not built |

---

## 10. Anti-hallucination — cross-cutting

1. Missing / null fields flagged at ingestion, never dropped.
2. Ratios needing a missing input are `is_calculable = False` with the input named.
3. RAG chunks state `NOT AVAILABLE` explicitly rather than omitting a field.
4. The system prompt hard-instructs the LLM to admit missing data, not estimate.
5. Low temperature (0.1) for factual generation.
6. Post-generation number *and period* matching against source-of-truth values.
7. Groundedness is measured per answer and logged for every query.
8. Red flags and distress scores are rule-based; the LLM only narrates them.
9. The distress model refuses a company-year it cannot properly describe rather
   than scoring it off imputed medians.
10. Every AI-generated claim in the UI shows its source period and a
    groundedness pill.
