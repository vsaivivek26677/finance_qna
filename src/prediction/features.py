"""The feature set, defined once so training and scoring cannot drift apart.

Every feature is scaled by **total liabilities** rather than total assets. That
is a deliberate departure from a first version of this layer (trained on the
UCI "Polish companies bankruptcy" dataset, asset-scaled): the American-market
training set this layer now uses has a documented, verified data defect in its
asset-side columns (`Total Current Liabilities` is byte-identical to
`Total Liabilities` in every row, and `Total Assets` is smaller than `Current
Assets` in 72% of rows - both accounting impossibilities). Total liabilities is
the one column in that dataset that behaves consistently, so every ratio here
is built on it. One upside: `market_value_to_liabilities` is now an *exact*
match for Altman's original 1968 X4 definition (market value of equity / total
liabilities), rather than the book-value stand-in the asset-scaled version used.

Definitions are taken from the training set, so a scored company is described
the same way the training rows were.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str          # canonical name used everywhere downstream
    description: str
    # True when a higher value means a *healthier* firm. Used only to phrase
    # the "what pushed this estimate up" explanation, never in the model.
    higher_is_safer: bool


# Order matters: the model is trained and scored on this exact sequence.
FEATURES: tuple[Feature, ...] = (
    Feature("net_income_to_liabilities",
            "Net income / total liabilities", True),
    Feature("ebit_to_liabilities",
            "EBIT / total liabilities (operating profitability vs. debt load)", True),
    Feature("ebitda_to_liabilities",
            "EBITDA / total liabilities (cash-generating capacity vs. debt load)", True),
    Feature("retained_earnings_to_liabilities",
            "Retained earnings / total liabilities (cumulative profitability, Altman X2 analogue)", True),
    Feature("revenue_to_liabilities",
            "Revenue / total liabilities (scale and turnover vs. debt load, Altman X5 analogue)", True),
    Feature("market_value_to_liabilities",
            "Market capitalisation / total liabilities (Altman X4, exact original definition)", True),
    Feature("gross_margin",
            "Gross profit / revenue", True),
    Feature("operating_expense_ratio",
            "Operating expenses / revenue", False),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
SAFE_DIRECTION: dict[str, bool] = {f.name: f.higher_is_safer for f in FEATURES}


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """A guarded division. Missing input or a zero/near-zero denominator gives
    None, which the model pipeline imputes - a wrong number would be worse."""
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < 1.0:  # denominators here are balance-sheet/income totals
        return None
    return numerator / denominator


def build_features_from_figures(
    *,
    total_liabilities: float | None,
    net_income: float | None,
    operating_income: float | None,   # EBIT
    ebitda: float | None,
    retained_earnings: float | None,
    revenue: float | None,
    market_value: float | None,        # market capitalisation
    gross_profit: float | None,
    operating_expenses: float | None,
) -> dict[str, float | None]:
    """Compute the eight features from one period's raw figures.

    Returns every feature name as a key; a value that could not be computed is
    None rather than absent, so a caller can count how many inputs were usable.
    """
    return {
        "net_income_to_liabilities": _ratio(net_income, total_liabilities),
        "ebit_to_liabilities": _ratio(operating_income, total_liabilities),
        "ebitda_to_liabilities": _ratio(ebitda, total_liabilities),
        "retained_earnings_to_liabilities": _ratio(retained_earnings, total_liabilities),
        "revenue_to_liabilities": _ratio(revenue, total_liabilities),
        "market_value_to_liabilities": _ratio(market_value, total_liabilities),
        "gross_margin": _ratio(gross_profit, revenue),
        "operating_expense_ratio": _ratio(operating_expenses, revenue),
    }


def usable_feature_count(features: dict[str, float | None]) -> int:
    return sum(1 for name in FEATURE_NAMES if features.get(name) is not None)
