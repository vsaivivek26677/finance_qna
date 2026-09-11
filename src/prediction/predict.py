"""Load the trained model once and score a feature vector.

Kept separate from `model.py` so the API and CLI can import scoring without
pulling in scikit-learn's training stack or the dataset loader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import settings
from src.prediction.features import FEATURE_NAMES, usable_feature_count
from src.prediction.model import DistressModel

logger = logging.getLogger(__name__)

# A period needs at least this many of the eight features to be scored. Below
# it the estimate leans too hard on imputed medians to mean anything.
MIN_USABLE_FEATURES = 5


class ModelUnavailable(RuntimeError):
    """Raised when no trained artifact is on disk."""


@dataclass
class DistressEstimate:
    is_scored: bool
    probability: float | None = None
    risk_band: str | None = None
    factors: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    features: dict[str, float | None] = field(default_factory=dict)
    model_name: str = ""
    model_version: str = ""


@lru_cache(maxsize=1)
def load_model(path: str | None = None) -> DistressModel:
    target = Path(path or settings.distress_model_path)
    if not target.exists():
        raise ModelUnavailable(
            f"No distress model at {target}. Train one with "
            f"`python -m src.prediction.cli train`."
        )
    return DistressModel.load(target)


def model_is_ready(path: str | None = None) -> bool:
    return Path(path or settings.distress_model_path).exists()


def score(features: dict[str, float | None], *, path: str | None = None) -> DistressEstimate:
    """Score one company-year. Missing inputs are stated, never guessed around."""
    model = load_model(path)
    usable = usable_feature_count(features)
    if usable < MIN_USABLE_FEATURES:
        present = [n for n in FEATURE_NAMES if features.get(n) is not None]
        return DistressEstimate(
            is_scored=False,
            reason=(
                f"only {usable} of {len(FEATURE_NAMES)} model inputs are available "
                f"({', '.join(present) or 'none'}); need at least {MIN_USABLE_FEATURES}"
            ),
            features=features,
            model_name=model.model_name,
            model_version=model.model_version,
        )

    # Isotonic calibration is piecewise-constant, so the raw float carries
    # false precision; round to keep the stored value honest.
    probability = round(model.probability(features), 5)
    band = model.band(probability)
    # Factor attribution is only meaningful once the estimate is off the floor.
    # For a company the model puts at ~0, naming "what raised it" would be noise.
    factors = model.explain(features) if band != model.bands[0][1] else []
    return DistressEstimate(
        is_scored=True,
        probability=probability,
        risk_band=band,
        factors=factors,
        features=features,
        model_name=model.model_name,
        model_version=model.model_version,
    )
