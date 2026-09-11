"""Train, calibrate and persist the financial-distress classifier.

The shipped estimator is a shallow gradient-boosted tree wrapped in isotonic
probability calibration, so its output can be read as an actual probability
rather than a score. Discrimination and calibration are both measured out of
fold, and both are reported against a naive linear baseline built from the same
features - a learned model that cannot beat a simple unweighted rule would not
be worth shipping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.prediction.dataset import LABEL_COLUMN
from src.prediction.features import FEATURE_NAMES, SAFE_DIRECTION

logger = logging.getLogger(__name__)

MODEL_NAME = "distress-hgb-isotonic"
# 2.0.0: retrained on the American public-company dataset (sowide/bankruptcy_
# dataset) with a redesigned, liability-scaled feature set - a breaking change
# from 1.0.0's UCI-Polish-trained, asset-scaled model.
MODEL_VERSION = "2.0.0"

# p < 0.05 -> Low, < 0.12 -> Moderate, < 0.30 -> Elevated, else High. The
# training base rate is ~5%, so "Elevated" means several times more likely to
# fail than a typical firm in the sample - not a prediction that it will.
DEFAULT_BANDS: tuple[tuple[float, str], ...] = (
    (0.05, "Low"),
    (0.12, "Moderate"),
    (0.30, "Elevated"),
    (1.01, "High"),
)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to a learned percentile range.

    The training ratios contain genuine extreme outliers (small-cap firms with
    near-zero denominators). Left raw they dominate calibration and the tree
    splits; the same clip is applied when scoring a real company so the two
    stay comparable.
    """

    def __init__(self, lower: float = 1.0, upper: float = 99.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(arr, self.lower, axis=0)
        self.hi_ = np.nanpercentile(arr, self.upper, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float).copy()
        return np.clip(arr, self.lo_, self.hi_)


def _base_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("winsor", Winsorizer()),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_depth=3,
                    learning_rate=0.05,
                    max_iter=400,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


@dataclass
class DistressModel:
    """A fitted, calibrated estimator plus everything needed to score and
    explain one company-year without re-reading the training data."""

    estimator: Any
    features: list[str]
    bands: list[tuple[float, str]]
    feature_median: dict[str, float]
    feature_iqr: dict[str, float]
    drivers: list[dict[str, float]]
    metrics: dict[str, Any]
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    n_train: int = 0
    n_positive: int = 0

    @property
    def prevalence(self) -> float:
        return self.n_positive / self.n_train if self.n_train else 0.0

    def _row(self, features: dict[str, float | None]) -> pd.DataFrame:
        return pd.DataFrame([[features.get(name) for name in self.features]], columns=self.features)

    def probability(self, features: dict[str, float | None]) -> float:
        return float(self.estimator.predict_proba(self._row(features))[0, 1])

    def band(self, probability: float) -> str:
        for cutoff, label in self.bands:
            if probability < cutoff:
                return label
        return self.bands[-1][1]

    def explain(self, features: dict[str, float | None], top: int = 3) -> list[dict[str, Any]]:
        """The features whose values are furthest onto the risky side of the
        training median, weighted by how much the model relies on each. This is
        an indicative attribution, not a SHAP decomposition."""
        importance = {d["feature"]: d["importance"] for d in self.drivers}
        scored: list[tuple[float, dict[str, Any]]] = []
        for name in self.features:
            value = features.get(name)
            if value is None:
                continue
            iqr = self.feature_iqr.get(name) or 1.0
            deviation = (value - self.feature_median.get(name, 0.0)) / iqr
            # risky side: below median when higher_is_safer, above when not
            risky = -deviation if SAFE_DIRECTION.get(name, True) else deviation
            weight = max(importance.get(name, 0.0), 0.0)
            contribution = risky * weight
            if contribution > 0:
                scored.append(
                    (contribution, {
                        "feature": name,
                        "value": round(float(value), 4),
                        "training_median": round(self.feature_median.get(name, 0.0), 4),
                        "direction": "raises estimate",
                    })
                )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [payload for _, payload in scored[:top]]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved distress model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "DistressModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain a DistressModel")
        return model


def _naive_linear_baseline_score(X: pd.DataFrame) -> np.ndarray:
    """A simple, unweighted linear solvency score - the model must beat this to
    justify its complexity. There is no published coefficient set (like
    Altman's) for these liability-scaled ratios, so each feature is simply
    signed by its known direction (SAFE_DIRECTION) and summed; the equivalent
    of a logistic-regression rule with every weight fixed to +-1. Returned
    negated so it reads as a distress score, like the model's."""
    z = sum(
        (X[name] if higher_is_safer else -X[name]).fillna(0)
        for name, higher_is_safer in SAFE_DIRECTION.items()
    )
    return (-z).to_numpy()


def train_model(
    frame: pd.DataFrame, *, calibration: str = "isotonic", seed: int = 0, folds: int = 5
) -> DistressModel:
    X = frame[list(FEATURE_NAMES)].astype(float)
    y = frame[LABEL_COLUMN].astype(int).to_numpy()
    logger.info("Training on %d rows, %d positive (%.2f%%)", len(y), y.sum(), 100 * y.mean())

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    calibrated = CalibratedClassifierCV(_base_pipeline(seed), method=calibration, cv=3)

    logger.info("Cross-validating out of fold (%d folds)...", folds)
    oof = cross_val_predict(calibrated, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    model_metrics = {
        "roc_auc": float(roc_auc_score(y, oof)),
        "pr_auc": float(average_precision_score(y, oof)),
        "brier": float(brier_score_loss(y, oof)),
    }
    base_score = _naive_linear_baseline_score(X)
    baseline_metrics = {
        "roc_auc": float(roc_auc_score(y, base_score)),
        "pr_auc": float(average_precision_score(y, base_score)),
    }
    logger.info(
        "  model    ROC-AUC %.3f  PR-AUC %.3f  Brier %.4f",
        model_metrics["roc_auc"], model_metrics["pr_auc"], model_metrics["brier"],
    )
    logger.info(
        "  baseline ROC-AUC %.3f  PR-AUC %.3f  (naive linear rule)",
        baseline_metrics["roc_auc"], baseline_metrics["pr_auc"],
    )

    logger.info("Fitting the shipped estimator on all rows...")
    calibrated.fit(X, y)

    # Permutation importance on the uncalibrated base (cheaper, same ranking).
    base = _base_pipeline(seed).fit(X, y)
    y_series = pd.Series(y, index=X.index)
    sample = X.sample(min(6000, len(X)), random_state=seed)
    sample_y = y_series.loc[sample.index].to_numpy()
    perm = permutation_importance(
        base, sample, sample_y, scoring="average_precision",
        n_repeats=5, random_state=seed, n_jobs=-1,
    )
    drivers = sorted(
        (
            {"feature": name, "importance": float(mean)}
            for name, mean in zip(FEATURE_NAMES, perm.importances_mean)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )

    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    return DistressModel(
        estimator=calibrated,
        features=list(FEATURE_NAMES),
        bands=list(DEFAULT_BANDS),
        feature_median={k: float(v) for k, v in X.median().items()},
        feature_iqr={k: float(q3[k] - q1[k]) for k in FEATURE_NAMES},
        drivers=drivers,
        metrics={
            "cv_folds": folds,
            "model": model_metrics,
            "baseline": baseline_metrics,
            "lift_pr_auc": model_metrics["pr_auc"] - baseline_metrics["pr_auc"],
        },
        n_train=int(len(y)),
        n_positive=int(y.sum()),
    )
