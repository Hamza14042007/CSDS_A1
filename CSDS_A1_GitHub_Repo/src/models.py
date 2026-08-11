"""The two models, each as a self-contained scikit-learn Pipeline.

Both share the same preprocessing: median imputation with explicit missingness
indicators and balanced class weights. The elastic-net model additionally
standardises inputs; gradient boosting does not need scaling.
"""
from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

import config


def make_elasticnet() -> Pipeline:
    """Regularised linear baseline (20% L1 / 80% L2)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(**config.ELASTICNET_PARAMS)),
    ])


def make_hgb() -> Pipeline:
    """Histogram gradient boosting.

    HGB handles NaNs natively, but we keep the missingness indicators so both
    models see the same information. No scaling is applied.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("clf", HistGradientBoostingClassifier(**config.HGB_PARAMS)),
    ])


MODEL_FACTORIES = {
    "elasticnet": make_elasticnet,
    "hgb": make_hgb,
}

MODEL_LABELS = {
    "elasticnet": "Elastic-net logistic",
    "hgb": "Histogram gradient boosting",
}
