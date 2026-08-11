"""End-to-end driver: ingest -> features -> models -> evaluation -> figures.

Run from the repository root:

    python -m scripts.run_experiment

Outputs metrics to results/ and the four figures to figures/. Requires the raw
FINRA and SEC files to be present (see scripts/download_data.py).
"""
from __future__ import annotations
import json
import os
import sys

import pandas as pd

# Allow "python scripts/run_experiment.py" as well as "-m scripts.run_experiment".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_ingest import load_finra, load_sec, load_ticker_map
from src.build_dataset import build_modelling_table, split_xy
from src.models import MODEL_FACTORIES, MODEL_LABELS
from src.evaluate import evaluate
from src.ablation import run_ablation
from src.importance import run_permutation_importance
from src import plots

NAVY, BERRY = "#1C7293", "#8E1F3C"


def main():
    print("[1/6] Loading raw sources ...")
    finra = load_finra()
    sub, num = load_sec()
    ticker_map = load_ticker_map()

    print("[2/6] Building modelling table ...")
    df, meta = build_modelling_table(finra, sub, num, ticker_map)
    print(f"      rows={len(df):,}  symbols={meta['n_symbols']}  "
          f"threshold={meta['threshold']:.3f}")
    print(f"      split counts: {meta['counts']}")
    print(f"      test prevalence: {meta['test_prevalence']:.3f}")

    feature_cols = meta["finra_cols"] + meta["sec_cols"]

    print("[3/6] Training and scoring both models ...")
    metrics = {}
    pr_data = []
    for key, factory in MODEL_FACTORIES.items():
        model = factory()
        Xtr, ytr = split_xy(df, feature_cols, "train")
        Xte, yte = split_xy(df, feature_cols, "test")
        model.fit(Xtr, ytr)
        score = model.predict_proba(Xte)[:, 1]
        metrics[key] = evaluate(yte, score)
        color = NAVY if key == "elasticnet" else BERRY
        pr_data.append((MODEL_LABELS[key], yte.to_numpy(), score, color))
        print(f"      {MODEL_LABELS[key]:<30} "
              f"PR-AUC={metrics[key]['pr_auc']:.3f} "
              f"ROC-AUC={metrics[key]['roc_auc']:.3f} "
              f"Brier={metrics[key]['brier']:.3f} "
              f"P@20={metrics[key]['precision@budget']:.3f} "
              f"lift={metrics[key]['lift@budget']:.2f}x")

    print("[4/6] Running source ablation ...")
    ablation = run_ablation(df, meta)
    print(ablation[["model", "FINRA", "SEC", "Integrated", "gain_vs_finra"]]
          .to_string(index=False))

    print("[5/6] Computing permutation importance ...")
    importance = run_permutation_importance(df, meta)
    print(importance.head(6).to_string(index=False))

    print("[6/6] Writing figures and results ...")
    plots.plot_target_and_split(df, meta)
    plots.plot_precision_recall(pr_data)
    plots.plot_ablation(ablation)
    plots.plot_permutation_importance(importance)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w") as fh:
        json.dump({"meta": {k: v for k, v in meta.items()
                            if k not in ("finra_cols", "sec_cols")},
                   "models": metrics}, fh, indent=2)
    ablation.to_csv(os.path.join(config.RESULTS_DIR, "ablation.csv"), index=False)
    importance.to_csv(os.path.join(config.RESULTS_DIR, "permutation_importance.csv"),
                      index=False)
    print("Done. See results/ and figures/.")


if __name__ == "__main__":
    main()
