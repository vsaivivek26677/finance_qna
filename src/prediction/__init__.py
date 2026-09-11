"""Layer 4 - financial-distress probability.

A calibrated classifier trained on an external labelled bankruptcy dataset
(sowide/bankruptcy_dataset, ~78.7k American-company firm-years, 1999-2018)
applied to ratios scaled by total liabilities. It answers a different question
from the rule-based Altman/Piotroski/Beneish scores: not "which zone does the
textbook formula put this in" but "how often did firms that looked like this
one actually fail".
"""

from src.prediction.features import FEATURES, build_features_from_figures

__all__ = ["FEATURES", "build_features_from_figures"]
