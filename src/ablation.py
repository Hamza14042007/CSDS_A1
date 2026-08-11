"""Source ablation: score each model on FINRA-only, SEC-only and integrated
feature sets to test whether the two sources are complementary.
"""
from __future__ import annotations
import pandas as pd

from src.build_dataset import split_xy
from src.models import MODEL_FACTORIES, MODEL_LABELS
from src.evaluate import evaluate


def run_ablation(df, meta) -> pd.DataFrame:
    finra_cols = meta["finra_cols"]
    sec_cols = meta["sec_cols"]
    feature_sets = {
        "FINRA": finra_cols,
        "SEC": sec_cols,
        "Integrated": finra_cols + sec_cols,
    }

    rows = []
    for model_key, factory in MODEL_FACTORIES.items():
        for set_name, cols in feature_sets.items():
            model = factory()
            Xtr, ytr = split_xy(df, cols, "train")
            Xte, yte = split_xy(df, cols, "test")
            model.fit(Xtr, ytr)
            score = model.predict_proba(Xte)[:, 1]
            m = evaluate(yte, score)
            rows.append({
                "model": MODEL_LABELS[model_key],
                "model_key": model_key,
                "feature_set": set_name,
                "pr_auc": m["pr_auc"],
            })
    wide = (pd.DataFrame(rows)
              .pivot(index=["model_key", "model"],
                     columns="feature_set", values="pr_auc")
              .reset_index())
    wide["gain_vs_finra"] = wide["Integrated"] - wide["FINRA"]
    return wide
