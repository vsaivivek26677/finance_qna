# Financial Analytics Platform

Financial statement analysis over verified data: a ratio engine with distress
scoring and rule-based red flags, grounded RAG summaries and Q&A, and a
calibrated bankruptcy-probability model — served through a FastAPI backend and a
Streamlit dashboard.

Built in layers. See [the as-built design doc](financial_analytics_project_plan.md)
for the full picture, and [What was planned but not built](#what-was-planned-but-not-built-and-why)
for where this deviates from the original plan.

| Layer | What it does | Status |
|---|---|---|
| 1 — Data ingestion | Fetch, validate and store statements; nothing dropped silently | ✅ |
| 2 — Ratio engine | 35 ratios, Altman Z / Z'', Piotroski F, Beneish M, 15 red-flag rules | ✅ |
| 3 — RAG + Groq LLM | Grounded summaries and Q&A with post-generation number verification | ✅ |
| 4 — Distress prediction | Calibrated P(financial distress) from a model trained on ~78.7k labelled American-company firm-years | ✅ |
| 5a — FastAPI backend | REST layer owning all business logic | ✅ |
| 5b — Streamlit dashboard | Thin HTTP client over the API | ✅ |

The database currently holds **47 companies** (five fiscal years each), all
analysed, indexed for retrieval and scored by the distress model.

---

## Setup

The repository ships with a populated `data/financials.db` (47 companies, five
fiscal years each), the retrieval index at `data/chroma/`, and the trained
distress model at `models/distress_model.joblib`, so the app has data to show
on a fresh clone with no rebuild step.

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env          # then fill in the keys below
# The retrieval index (data/chroma/) is committed, so no indexing step is
# needed on a fresh clone. Only re-run this after ingesting a new company:
python -m src.rag.cli index --all
```

Keys in `.env`:

- `FMP_API_KEY` — free key from [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs). 250 requests/day; one ticker costs 4 (profile + three statements). FMP has been retiring its legacy `/api/v3` endpoints, and the free tier now returns HTTP 403 for many large-cap symbols — ingestion records those as a plan limit and moves on. Most companies in the database were ingested before that cut-off.
- `GROQ_API_KEY` — free key from [console.groq.com](https://console.groq.com), used from Layer 3 onward.

Market data comes from `yfinance` and embeddings run locally, so neither needs a key.

**Groq retires model ids regularly** — the `llama-3.x` ids this project was
originally specified against no longer exist. Check what your key can reach:

```bash
python -c "from src.rag.groq_client import GroqClient; print(GroqClient().list_models())"
```

Set `GROQ_MODEL` in `.env` if the default is unavailable; the client also falls
back automatically to `GROQ_FALLBACK_MODEL`.

Starting from an empty database instead? `python -m src.ingestion.cli --init-only`
creates the schema; then ingest, analyse, index and score a ticker with
`POST /ingest/{ticker}` or the per-layer CLIs.

---

## Layer 1 — Data Ingestion

### Usage

Ingest one ticker (5 years of annual statements plus monthly price history):

```bash
python -m src.ingestion.cli AAPL
```

Several tickers, quarterly statements, or machine-readable output:

```bash
python -m src.ingestion.cli AAPL MSFT NVDA --years 5
```

```bash
python -m src.ingestion.cli AAPL --period quarter --years 8 --json
```

Skip market data (Layer 2 then falls back to the book-value Z''-Score):

```bash
python -m src.ingestion.cli AAPL --no-market-data
```

From Python:

```python
from src.ingestion.ingest_pipeline import ingest_ticker

report = ingest_ticker("AAPL", years=5)
print(report.summary_line())      # [SUCCESS] AAPL (annual): income_statements=5, ...
print(report.critical_missing)    # {} when nothing important is absent
```

### Design

```
src/ingestion/
    field_maps.py       canonical field name -> source aliases
    schemas.py          pydantic records (reused as FastAPI models in Layer 5)
    normalizer.py       raw JSON -> validated record + missing map (pure, no I/O)
    fmp_client.py       FMP HTTP client: retries, quota guard, API-variant fallback
    yfinance_client.py  price history, share count, market cap
    repository.py       upserts keyed on natural business keys
    ingest_pipeline.py  fetch -> normalize -> store, with an auditable report
    cli.py              command-line entry point
```

Three decisions are worth calling out, because they are what the later layers
depend on.

**Nothing is silently dropped or zero-filled.** Every numeric field is optional,
and any field without a value gets an entry in the record's missing map naming
*why*: `not_reported` (the key was absent), `null_in_source` (present but null),
or `unparseable`. That map is persisted per row in `is_missing_json`. So a `NULL`
in the database is never ambiguous, and "revenue was zero" is never confused with
"revenue was not reported". Layer 2 uses this to mark ratios
`is_calculable = False` with a named reason instead of quietly producing a wrong
number; Layer 3 chunks the field as "NOT AVAILABLE" rather than omitting it.

Booleans are explicitly rejected by the numeric coercion, so a stray flag can
never become a `1.0` on a balance sheet.

**Both FMP API generations are supported.** FMP is migrating from `/api/v3/...`
to `/stable/...`, and which one a key can reach depends on when it was issued —
the spellings differ (`epsdiluted` vs `epsDiluted`, `netReceivables` vs
`accountsReceivables`, `fillingDate` vs `filingDate`). Rather than branching on
the endpoint, `field_maps.py` lists the aliases each canonical field may arrive
under, and the client retries against the other generation when a plan-availability
error comes back, then sticks with whichever worked. Adding a second data source
later means adding aliases, not rewriting the pipeline.

**Failures degrade, they do not abort.** A missing profile, a broken yfinance
scrape, or one unnormalizable statement row produces a `partial` run with the
problem recorded — not a lost ingestion. Every attempt, successful or not, writes
an `ingestion_runs` audit row. A run is only `success` when statements landed,
no errors occurred, no critical field is missing, and market data was retrieved.

### What gets stored

`companies`, `income_statements`, `balance_sheets`, `cash_flow_statements`,
`market_data`, plus the `ingestion_runs` audit trail. Tables for later layers
(`ratios`, `red_flags`, `rag_eval_logs`, `distress_predictions`) are declared in
`src/db/models.py` and created up front.

Upserts are keyed on the natural business key (ticker; company + period + fiscal
year; company + date), so re-running a ticker refreshes existing rows rather than
duplicating history — restatements overwrite stale figures, which is what an
analyst would expect. A partial profile response will not erase good data already
on file.

Company rows carry `is_financial_sector`, set at ingestion, because the Altman
Z-Score is not defined for banks and insurers — Layer 2 returns
"Not Applicable — Financial Sector" instead of a misleading number.

### Quota safety

The FMP client counts its own requests and refuses to exceed
`FMP_DAILY_REQUEST_BUDGET` (default 250) in a single process, so a runaway loop
cannot burn a day's quota. It honours `Retry-After` on HTTP 429, retries
transient 5xx with exponential backoff, and detects the error objects FMP returns
inside HTTP 200 bodies. Raw JSON responses are cached under `data/raw/<TICKER>/`
so re-runs and debugging cost nothing.

---

## Layer 2 — Ratio Engine, Distress Scores & Red Flags

35 ratios across seven categories, plus rule-based risk detection. Reads only
what Layer 1 stored; writes to the `ratios` and `red_flags` tables.

### Usage

```bash
python -m src.ratios.cli AAPL
```

Detail for one year, including everything that could *not* be computed and why:

```bash
python -m src.ratios.cli AAPL --year 2024 --show-missing
```

```bash
python -m src.ratios.cli AAPL MSFT --json
```

From Python:

```python
from src.db.database import session_scope
from src.ratios.ratio_engine import run_analysis, analyze_ticker

with session_scope() as session:
    report = run_analysis(session, "AAPL")
    print(report.summary_line())   # 171 ratios computed, 4 not calculable (98% coverage)

    analyses, flags = analyze_ticker(session, "AAPL")
    latest = analyses[-1]
    z = latest.result("altman_z_score")
    print(z.value, z.details["zone"], z.inputs)   # 10.08 Safe {'x1_...': ...}
```

### What it computes

| Category | Ratios |
|---|---|
| Liquidity | Current, Quick, Cash, Working Capital |
| Profitability | Gross/Operating/Net Margin, ROE, ROA, ROIC |
| Leverage | Debt-to-Equity, Debt-to-Assets, Liabilities-to-Assets, Interest Coverage, Net Debt/EBITDA |
| Efficiency | Asset/Inventory/Receivables Turnover, DSO, DIO |
| Valuation | P/E, P/B, P/S, Enterprise Value, EV/EBITDA, Diluted EPS |
| Cash Flow | FCF, FCF Margin, OCF Margin, Cash Conversion, Capex/OCF |
| Distress | Altman Z, Altman Z'', Piotroski F, Beneish M |

### Design

```
src/ratios/
    base.py             RatioResult, PeriodBundle, NotCalculable, the registry
    liquidity.py profitability.py leverage.py
    efficiency.py valuation.py cash_flow.py
    distress_scores.py  Altman Z / Z'', Piotroski F, Beneish M
    red_flags.py        15 deterministic rules with tunable thresholds
    ratio_engine.py     load -> compute -> detect -> persist
    repository.py       ratio/flag persistence and read helpers
    cli.py              command-line entry point
```

**A ratio is computed or explicitly refused — never estimated.** Ratio functions
raise `NotCalculable` with a reason naming the specific input, and the engine
stores that as `is_calculable = False` with the reason attached. The reason
recorded at ingestion carries through, so `revenue not reported` and
`revenue null in source` stay distinguishable all the way to the dashboard. Refusals
also cover ratios that would be *misleading* rather than impossible: P/E on
negative earnings, D/E on negative equity, and inventory turnover on zero
inventory all produce numbers that look valid and rank nonsensically.

Real example from the run above — Apple has no reported interest expense in FY2024:

```
interest_coverage   no interest expense reported in FY2024; interest coverage is
                    undefined (which is not itself a sign of distress)
```

**Composite scores persist their components.** Every Altman term, all nine
Piotroski signals, and all eight Beneish indices are stored alongside the score
in `inputs_json`/`details_json`. Layer 3 narrates those pre-computed facts rather
than inferring why a company sits where it does.

**Where more than one basis is defensible, the choice is recorded.** ROE, ROA and
the turnover ratios use average balances when a prior year is on file and closing
balances otherwise; `method` says which, so two periods are never silently
compared on different bases. ROIC's effective tax rate comes from the company's
own reported tax expense, and in a pre-tax loss year — where the rate is
undefined — it is set to zero and the substitution is stated.

**Sector applicability is enforced, and the skip is disclosed.** Banks and
insurers are excluded from both Altman variants, which return
"Not Applicable - Financial Sector". The same reasoning extends to eight of the
fifteen red-flag rules: a bank's interest expense is a cost of funding, its
balance sheet is not split into current and non-current, and lending routinely
drives operating cash flow negative. Running those rules over JPM produced four
confident false positives, so they are now skipped — and an `Info` flag names
exactly which rules were skipped and why, rather than leaving a silent gap.

**Market data is paired to the period, not to today.** Valuation ratios use the
price observation nearest the fiscal period end, within 45 days. Outside that
window there is no pairing and the ratio is unavailable — a 2021 balance sheet is
never valued at a 2026 share price. Without market data the market-value Altman Z
is unavailable and the book-value Z''-Score takes over, as the plan specifies.

Worth knowing when reading the output: the two Altman variants can disagree
sharply and legitimately. Apple scores Z = 10.08 (Safe) but Z'' = 2.31 (Grey),
because its market value vastly exceeds its book equity. They are different
models with different coefficients and different zone boundaries, not two
estimates of one number.

### Red flags

Fifteen deterministic rules across solvency, liquidity, profitability, leverage,
earnings quality and data integrity. Each flag records the values that triggered
it, so it is auditable and safe to hand to an LLM as ground truth. Thresholds
live in one `THRESHOLDS` dict.

Two rules deserve note. The sector-median leverage comparison is **skipped
entirely** when no median is supplied — inventing a benchmark is exactly the
fabrication this layer exists to prevent. And missing critical data is itself a
visible `Info` flag, because a ratio table with quiet gaps invites the reader to
assume the gaps are zeros.

Flags are evaluated per period against only that period's history, so a flag on
FY2022 reflects what an analyst could have known at the time — no lookahead.

---

## Layer 3 — RAG, Grounded Q&A & Evaluation

Retrieval-augmented generation over the verified figures from Layers 1-2, with a
free Groq LLM, local embeddings, and a quantitative evaluation harness.

### Usage

```bash
python -m src.rag.cli index AAPL
```

Indexing is a **separate step from ingestion**, so a company can have statements
and ratios yet still be unanswerable. Index everything that has been analysed:

```bash
python -m src.rag.cli index --all
```

`/health` reports `companies_indexed` against `companies_analysed`, and the
dashboard sidebar warns when they differ. When retrieval finds nothing, the
answer names the step that is actually missing - ingestion, ratios, or indexing -
rather than telling you to start over.

```bash
python -m src.rag.cli ask AAPL "What was the net margin in FY2024?" --year 2024
```

```bash
python -m src.rag.cli summary AAPL
```

```bash
python -m src.rag.cli eval AAPL --years 2 --show-failures
```

```bash
python -m src.rag.cli stats --ticker AAPL
```

Add `--show-context` to `ask` to print the retrieved chunks, and `--block` to
withhold any answer containing a figure that cannot be verified.

### Measured results

A live run against Apple — five fiscal years, 63 indexed chunks, 25 generated
evaluation questions:

| Metric | Score |
|---|---|
| Faithfulness (numeric claims traceable to context) | **1.000** |
| Answer correctness (matches the database) | **1.000** (25/25) |
| Context recall (answer-bearing chunk retrieved) | **1.000** |
| Refusal correctness (says NOT AVAILABLE when it is) | **1.000** |
| Context precision (rank of that chunk) | 0.809 |

Model: `openai/gpt-oss-120b` on Groq's free tier, temperature 0.1.

### Design

```
src/rag/
    chunker.py        DB rows -> labelled, retrievable fact chunks
    embedder.py       local all-MiniLM-L6-v2, plus a deterministic test backend
    vector_store.py   ChromaDB (persistent) and an in-memory store for tests
    groq_client.py    retries, rate-limit handling, model fallback
    guardrails.py     system prompt + post-generation verification
    rag_pipeline.py   index -> retrieve -> generate -> verify -> log
    evaluation.py     ground-truth eval set and scoring
    repository.py     query logging and groundedness aggregation
    cli.py            command-line entry point
```

**Missing data is stated, never omitted.** A chunk that quietly leaves out
`interest_expense` invites the model to fill the gap. Every chunk instead names
what is absent — `NOT AVAILABLE ... interest_expense (not reported)` — and each
period gets a dedicated data-availability chunk. Asked for Apple's FY2024
interest coverage, the system answers:

> The interest coverage ratio for **AAPL FY2024** is **NOT AVAILABLE** because no
> interest expense was reported in FY2024, making the ratio undefined.

**Every number the model writes is checked against the numbers it was given.**
After generation, each figure in the answer is matched against the machine-readable
`source_values` carried by the retrieved chunks. Matching is precision-aware: if
the source is `391,035,000,000` then `391.04B` is a faithful rounding and passes,
while `395B` matches nothing and is flagged. Period citations are verified
separately — attributing a real figure to a year that was never retrieved reads
as authoritative and is invisible to a purely numeric check. Groundedness is
reported per answer and logged for every query.

**Risk questions always retrieve the risk chunks.** Semantic similarity ranks
"is this company at risk of bankruptcy?" against the balance sheet, leaving the
Altman Z-Score outside the top-k — which would leave the model reasoning about
solvency itself, the exact judgement Layer 2 exists to make instead. A small,
inspectable intent map guarantees the distress and red-flag chunks are present
for those questions.

**Summaries are selected structurally, not by similarity.** An executive summary
must cover performance, position, risk and data limitations; similarity search
would happily return five ratio chunks and no risk chunk. The summary path fetches
the required chunk types directly.

### Evaluation, and why not `ragas`

The plan specified `ragas`. Every released version pins the pre-1.0 LangChain
line — they import `langchain_community.chat_models.vertexai`, removed in
LangChain 1.x — so installing it means downgrading LangChain globally or
maintaining a parallel virtualenv. It was dropped.

The replacement is not a weaker stand-in for this domain. `ragas` scores
faithfulness by asking a second LLM to judge the first: expensive,
non-deterministic, and itself capable of hallucinating. Here the questions are
generated *from the database*, so both the correct answer and the chunk that
should supply it are known exactly, and the metrics become reference-based and
deterministic:

| Metric | How it is computed |
|---|---|
| `faithfulness` | Guardrail verification — no LLM involved |
| `answer_correctness` | Precision-aware numeric match against the DB value |
| `context_recall` | Was the known answer-bearing chunk retrieved? |
| `context_precision` | Reciprocal rank of that chunk |
| `refusal_correctness` | Does it say NOT AVAILABLE when the value genuinely is? |

`refusal_correctness` has no `ragas` equivalent and is the one that matters most
here — a refusal that still slips in a number scores zero. The eval set includes
metrics that genuinely cannot be computed (the earliest year has no prior year,
so Piotroski and Beneish are unavailable) and a period outside the data entirely.

Every query, in evaluation or in production, is logged to `rag_eval_logs` with
its retrieved context and groundedness, so the score is a measurement over real
traffic rather than a claim.

---

## Layer 4 — Distress Prediction

A calibrated classifier that estimates the probability a company is in financial
distress. It is deliberately **not** trained on this project's ~190 company-years
— that is far too little data for a credible model, and it was the reason the
original XGBoost forecasting plan was dropped. Instead it learns from an external
labelled dataset and is then applied to the same ratios Layer 2 already computes.

### What it is

| | |
|---|---|
| Training data | [sowide/bankruptcy_dataset](https://github.com/sowide/bankruptcy_dataset) (Lombardo et al., *Future Internet* 2022): 78,682 firm-years from 8,971 NYSE/NASDAQ public companies, 1999–2018, 6.6% bankrupt (Chapter 7/11 filed the following year). CC-BY 4.0. |
| Features | 8 ratios scaled by **total liabilities**, not total assets — see "Why liability-scaled" below |
| Model | `HistGradientBoostingClassifier` (shallow, class-weighted) wrapped in isotonic probability calibration |
| Output | A calibrated probability, a risk band (Low / Moderate / Elevated / High), and the features sitting furthest onto the risky side of the training distribution |

This is the second training set this layer has used. The first version was
trained on the UCI *Polish companies bankruptcy data* (2000s Polish SMEs) —
functional, but a poor match for the US large-caps this platform actually
scores. This dataset is the real target population: American public companies,
scored on liabilities/profitability/market-value ratios computed by the exact
same function (`build_features_from_figures`) whether it is training on a
historical firm-year or scoring a live company.

**Why liability-scaled, not asset-scaled.** The source CSV has a verified data
defect: its `Total Current Liabilities` column is byte-identical to
`Total Liabilities` in all 78,682 rows, and its `Total Assets` column is
*smaller* than `Current Assets` in 72% of rows — both accounting
impossibilities. Total assets is therefore never used; every feature is scaled
by total liabilities, the one size-anchor column that behaves consistently.
One upside: `market_value_to_liabilities` is now an *exact* match for Altman's
original 1968 X4 term (market value of equity / total liabilities), rather
than the book-value stand-in the asset-scaled version used.

### Measured results

Five-fold cross-validation, out of fold, against a naive unweighted linear
baseline built from the same features (there is no published coefficient set,
like Altman's, for these particular ratios):

| Metric | Model | Naive linear baseline |
|---|---|---|
| ROC-AUC | **0.699** | 0.520 |
| PR-AUC (base rate 6.6%) | **0.138** | 0.071 |
| Brier score | 0.060 | — |

The learned model nearly doubles the baseline's PR-AUC. ROC-AUC in the high-0.6s
is modest — the price of deliberately excluding the dataset's broken columns
rather than the 18 raw variables the source paper used — but it is now trained
on the actual population it scores, not a different country's SMEs two decades
removed. See the "What could be done differently" list this project has kept
throughout for what a fuller fix would look like (a corrected data source,
more features, a held-out temporal test set).

### Usage

```bash
python -m src.prediction.cli train          # downloads the CSV, ~1 min to fit
python -m src.prediction.cli score --all    # score every ingested company
python -m src.prediction.cli score AAPL --year 2024
python -m src.prediction.cli report         # print the trained model's metrics
```

```python
from src.prediction.predict import score
from src.prediction.repository import build_feature_rows
from src.db.database import session_scope

with session_scope() as session:
    rows = build_feature_rows(session, "WBA")
    est = score(rows[-1].features)
    print(est.probability, est.risk_band, est.factors)
```

### Design

```
src/prediction/
    features.py     the 8-feature spec, defined once; shared by training and scoring
    dataset.py      download, cache and parse the American bankruptcy CSV
    model.py        train, calibrate, cross-validate, compare to baseline, persist
    predict.py      load the artifact and score one feature vector
    repository.py   build features from stored statements (reuses Layer 2's market-data pairing); persist estimates
    cli.py          train / score / report
```

**A period without enough inputs is refused, not guessed.** Scoring needs at
least 5 of the 8 features; below that the estimate leans too hard on imputed
medians. Those company-years come back `is_scored: false` with the missing
inputs named — the same rule the rest of the stack follows.

**The band is the signal, not the decimal.** Isotonic calibration is
piecewise-constant, so two nearby estimates can land on the same calibrated
value. The dashboard leads with the band and reserves factor attribution for
estimates that are off the floor.

---

## Layer 5 — API & Dashboard

A FastAPI service owns all business logic; the Streamlit dashboard is a thin HTTP
client with no database session, no model and no API key of its own. That split
is the point of having an API layer — the same backend could serve a different
frontend unchanged.

### Running it

```bash
./start_api.sh
```

```bash
./start_dashboard.sh
```

The API serves OpenAPI docs at <http://localhost:8000/docs>; the dashboard runs
at <http://localhost:8501>.

### Endpoints

| Endpoint | Returns |
|---|---|
| `GET /health` | Database, retrieval index and LLM readiness |
| `GET /companies` | Every ingested company with its fiscal years |
| `GET /companies/{ticker}` | Profile and classification |
| `GET /companies/{ticker}/statements` | Statements, with `missing_fields` per period |
| `GET /companies/{ticker}/ratios` | All ratios, including uncomputable ones with reasons |
| `GET /companies/{ticker}/ratios/{name}/series` | One ratio across fiscal years |
| `GET /companies/{ticker}/distress-score` | Altman Z / Z'', Piotroski F, Beneish M with zones |
| `GET /companies/{ticker}/distress-prediction` | Model-estimated P(financial distress) per year, with provenance |
| `GET /companies/{ticker}/red-flags` | Rule-based flags with severity and trigger values |
| `GET /companies/{ticker}/summary` | Grounded executive summary |
| `POST /companies/{ticker}/ask` | Grounded Q&A |
| `POST /ingest/{ticker}` | Fetch, analyse, index and score a company |

### API design

**The contract reports absence as explicitly as presence.** A ratio that could
not be computed is returned with `is_calculable: false` and a reason naming the
missing input, rather than omitted — omission would let a client read a gap as a
zero. Statement fields the provider never sent are listed in `missing_fields`.
Generated answers carry a `groundedness` score and name any figure that could not
be traced back to source data. A client rendering the response faithfully cannot
present an estimate as a fact.

**Only the generative endpoints are `async def`.** They spend their time waiting
on Groq, and the blocking call is pushed to a worker thread with
`run_in_threadpool`, so one slow generation does not stall the event loop. The
database endpoints are plain `def`, which FastAPI already runs in a threadpool —
making them `async` would block the loop on synchronous SQLAlchemy calls, which is
the more common mistake.

**Caching protects quota, not the database.** Reads use a 60-second TTL cache;
generated summaries use 15 minutes because each one costs free-tier LLM quota.
Ingestion clears the caches so new data is visible immediately.

**Errors carry the fix.** A 404 says `POST /ingest/AAPL`; a missing Groq key
returns 503 explaining that every non-generative endpoint still works; an upstream
LLM failure is a 502, not a 500.

The embedding model is warmed in a background thread at start-up. Without it the
first question after a restart waits ~30 seconds for the sentence-transformer to
load, which reads as a hang.

### Dashboard

Six tabs: Overview, Statements, Ratio analytics, Risk & red flags, AI summary,
Ask AI. Dark navy ground, a single teal accent, and colour reserved for severity
so that when something turns amber or red it means something. The Overview and
Risk tabs show the Layer 4 distress probability next to the Altman gauge — the
learned estimate and the textbook formula side by side, never one instead of the
other.

```
src/dashboard/
    app.py                  page layout and tab routing
    api_client.py           typed HTTP client; every failure becomes a message
    components/
        theme.py            palette, type scale, CSS, shared Plotly layout
        charts.py           figures and all number formatting
        panels.py           KPI rows, flag cards, statement tables, sources
```

Three details that matter more than they look:

**Units are decided per field, not per type.** A margin renders as `26.92%`, a
multiple as `1.52`, an absolute as `-17.67B`, EPS as `7.24` and a share count as
`14.95B`. Showing a margin as `0.2692` or EPS as `USD 7` is the classic way a
finance dashboard becomes quietly wrong.

**Deltas are coloured by meaning, not by sign.** Streamlit paints any negative
delta red; for leverage and collection-period ratios a fall is an improvement, so
those use inverted colouring. Falling debt-to-equity reads green. A percentage
ratio's year-over-year change is shown in points (`+2.94 pts`), not a relative
percent — a margin moving from 24% to 26% went up two points, not eight percent.

**Absence is visible.** Statement gaps render as `N/A — not reported`, ratios that
could not be computed sit behind an expander with the reason for each, and every
generated answer shows a groundedness pill plus its retrieved sources.

### Deployment

Streamlit Community Cloud runs one process. `src/dashboard/embedded_backend.py`
starts the FastAPI backend in a daemon thread on first load when no API is
reachable, so the whole stack deploys as a single app. The database, retrieval
index and trained model are committed, so a fresh deploy has data immediately.
Full walkthrough, secrets and the resource-limit caveats: **[DEPLOYMENT.md](DEPLOYMENT.md)**.

For a separately hosted backend, set `API_BASE_URL` to its URL and the embedded
server is skipped.

---

## What was planned but not built, and why

The [original plan](financial_analytics_project_plan.md) specified more than what
shipped. Each omission below was a deliberate call, not an unfinished corner.

**`ragas` for RAG evaluation.** Every released version pins the pre-1.0 LangChain
line (`langchain_community.chat_models.vertexai`, removed in LangChain 1.x), so
installing it means downgrading LangChain globally. Replaced with a
reference-based harness: because the eval questions are generated *from the
database*, the correct answer and the chunk that should supply it are both known
exactly, so faithfulness / answer-correctness / context-recall become
deterministic instead of an LLM judging an LLM. Detail in the Layer 3 section.

**XGBoost next-period forecasting (revenue growth, future ROE) with walk-forward
backtesting.** The data does not support it. Five annual periods per company
across ~47 companies is ~235 rows and panels four steps long — any
cross-validated forecast would be a toy, and walk-forward validation on four
usable folds says almost nothing. Layer 4 answers a question the data *can*
support: it borrows 78.7k labelled outcomes from an external dataset and applies a
learned model to ratios we compute, rather than trying to learn from our own
thin history.

**MLflow experiment tracking and model registry.** One model with one training
routine does not need a tracking server or a staged registry — that is lifecycle
tooling for a fleet of competing experiments. The `joblib` artifact carries its
own metrics, feature list, training date and row counts; `python -m
src.prediction.cli report` prints them.

**SHAP explainability.** The distress model has ten features. Global permutation
importance (stored in the artifact) plus a per-company "which values sit furthest
onto the risky side of the training distribution" attribution cover what a SHAP
summary and force plot would show here, without the dependency or the plot
rendering. On a hundred-feature model the trade-off would go the other way.

**Peer clustering, anomaly detection, a sector classifier, peer benchmarks and
temporal-deviation scoring.** A mid-build Layer 4 attempt, before the pivot to
the bankruptcy model. K-means peer groups over ~28 companies gave clusters of
three or four — too small to benchmark a company against. Isolation-forest
anomaly scores over ~140 company-years flagged rows at close to random. The
sector classifier was predicting a label FMP already returns in the company
profile. All of it was removed — the four tables dropped from the schema and the
`.joblib` artifacts deleted — rather than shipped as a dashboard ornament.

**CI (GitHub Actions).** Not built. `python -m pytest` (510 tests, no network)
is the regression gate. The Streamlit Community Cloud deployment *was* built —
the dashboard self-hosts the API in a background thread, see
[DEPLOYMENT.md](DEPLOYMENT.md).

---

## Tests

```bash
python -m pytest
```

510 tests. Layer 1 covers numeric/date coercion and missing-reason
classification, v3-vs-stable payload equivalence, HTTP error mapping and quota
enforcement (via `respx`), and pipeline persistence including upsert idempotency
and restatement handling. Layer 2 asserts every ratio against hand-calculated
literals derived from Apple's FY2023 filing independently of `src/`, and checks
each red-flag rule both fires and stays silent. Layer 3 tests the guardrail from
the attacker's side — what wrong number could a model emit, and is it caught.
Layer 4 checks that training and scoring use the same feature definitions, that
an under-specified company-year is refused rather than guessed, and that the
learned model beats the Altman baseline on a synthetic separable set. Layer 5
tests that the API reports absence faithfully (reasons on uncomputable ratios,
`missing_fields` on statements, groundedness on answers, model provenance on
predictions) and that the dashboard's formatters render each field in its own
units.

No test touches the network or the real database — the bankruptcy training set is never
downloaded in the suite; every model test builds a small synthetic frame. API
keys are blanked suite-wide so a mis-mocked test can never spend live quota,
embeddings use the deterministic hashing backend so nothing downloads a model,
and the LLM is always scripted.

---

## Project layout

```
src/
  config.py           settings from .env (single source of configuration)
  db/                 SQLAlchemy models + engine/session management
  ingestion/          Layer 1
  ratios/             Layer 2
  rag/                Layer 3
  prediction/         Layer 4 - distress-probability model
  api/                Layer 5a - FastAPI service
  dashboard/          Layer 5b - Streamlit client
models/               the trained distress model (distress_model.joblib)
data/                 SQLite database, raw JSON cache, ChromaDB, bankruptcy training set (gitignored)
tests/
```
