"""Fetch and parse the training set: American public-company bankruptcies.

Source: sowide/bankruptcy_dataset (Lombardo, Pellegrino, Adosoglou, Cagnoni,
Pardalos & Poggi, "Machine Learning for Bankruptcy Prediction in the American
Stock Market", Future Internet 2022). 78,682 firm-year observations from 8,971
NYSE/NASDAQ companies, 1999-2018, ~6.6% labelled bankrupt (Chapter 7/11 filed
the following year). CC-BY 4.0.

This replaces an earlier version of this layer trained on the UCI "Polish
companies bankruptcy" dataset - 2000s Polish SMEs, a poor match for the US
large-caps this platform actually scores. This dataset is the real target
population.

The raw CSV reports 18 anonymised accounting variables (`X1`..`X18`; see the
paper's Table 2) with **no company or ticker identity** - by design, since the
non-anonymised version cannot be redistributed under the source license. Two of
those columns have a verified data defect: `X14` ("Total Current Liabilities")
is byte-identical to `X17` ("Total Liabilities") in every one of the 78,682
rows, and `X10` ("Total Assets") is smaller than `X1` ("Current Assets") in 72%
of rows - both accounting impossibilities. `X10` and `X14` are therefore never
used here; every feature in `features.py` is scaled by the one column that
behaves consistently, `X17` (total liabilities), via `build_features_from_figures`
- the same function used to score a live company - so training and scoring can
never compute a feature differently.

The CSV is cached under `settings.distress_data_dir` so training is offline
after the first run. Nothing here is imported by the test suite - tests build
synthetic frames instead, so CI never reaches out to GitHub.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings
from src.prediction.features import build_features_from_figures

logger = logging.getLogger(__name__)

LABEL_COLUMN = "label"
HORIZON_COLUMN = "years_before_outcome"

# The raw dataset's columns, per the paper's Table 2. Only the ones that feed a
# feature in features.py (via build_features_from_figures) are used; X1, X2,
# X5, X7, X10, X11, X14 are read from the file but never consumed - X10/X14 are
# the defective columns described above, the rest simply aren't part of the
# liability-scaled feature set.
_STATUS_COLUMN = "status_label"
_FAILED_VALUE = "failed"


def _download_csv(dest: Path) -> None:
    import httpx

    url = settings.distress_dataset_url
    logger.info("Downloading the bankruptcy training set from %s", url)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(url)
        response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    logger.info("Saved %s (%d KB)", dest, len(response.content) // 1024)


def _build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw X1..X18 columns into the eight canonical features, one row
    at a time conceptually (vectorised) but via the exact same guarded-division
    function `predict.score` uses to score a real company."""
    rows = []
    for record in raw.itertuples(index=False):
        rows.append(
            build_features_from_figures(
                total_liabilities=record.X17,
                net_income=record.X6,
                operating_income=record.X12,
                ebitda=record.X4,
                retained_earnings=record.X15,
                revenue=record.X16,
                market_value=record.X8,
                gross_profit=record.X13,
                operating_expenses=record.X18,
            )
        )
    frame = pd.DataFrame(rows)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame[LABEL_COLUMN] = (raw[_STATUS_COLUMN] == _FAILED_VALUE).astype(int)
    # Every row already represents "the fiscal year before the outcome" (or a
    # healthy year) per the source paper's labelling convention - there is no
    # multi-horizon structure to preserve, unlike the earlier UCI version.
    frame[HORIZON_COLUMN] = 1
    return frame


def load_training_frame(data_dir: str | Path | None = None, *, download: bool = True) -> pd.DataFrame:
    """Return one frame: the eight feature columns, `label` (1 = bankrupt), and
    `years_before_outcome`. Downloads and caches the CSV on first use."""
    directory = Path(data_dir or settings.distress_data_dir)
    csv_path = directory / "american_bankruptcy_dataset.csv"

    if not csv_path.exists():
        if not download:
            raise FileNotFoundError(
                f"{csv_path} not found and download=False. Run "
                f"`python -m src.prediction.cli train` with network access, or "
                f"place the CSV from https://github.com/sowide/bankruptcy_dataset there."
            )
        _download_csv(csv_path)

    raw = pd.read_csv(csv_path)
    frame = _build_features(raw)
    logger.info(
        "Loaded %d firm-years (%d bankrupt, %.1f%%) from %d companies",
        len(frame), int(frame[LABEL_COLUMN].sum()),
        100 * frame[LABEL_COLUMN].mean(), raw["company_name"].nunique(),
    )
    return frame
