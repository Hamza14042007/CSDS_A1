"""Permutation importance for the integrated boosting model.

Importance is the average drop in test PR-AUC when a single feature column is
shuffled, so it is expressed in the same units as the headline metric.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, average_precision_score

import config
from src.build_dataset import split_xy
from src.models import make_hgb

_pr_auc_scorer = make_scorer(average_precision_score, response_method="predict_proba")


def run_permutation_importance(df, meta) -> pd.DataFrame:
    cols = meta["finra_cols"] + meta["sec_cols"]
    model = make_hgb()
    Xtr, ytr = split_xy(df, cols, "train")
    Xte, yte = split_xy(df, cols, "test")
    model.fit(Xtr, ytr)

    result = permutation_importance(
        model, Xte, yte,
        scoring=_pr_auc_scorer,
        n_repeats=config.PERMUTATION_REPEATS,
        random_state=config.RANDOM_SEED,
    )
    imp = (pd.DataFrame({
                "feature": cols,
                "importance": result.importances_mean,
                "std": result.importances_std,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
    return imp
