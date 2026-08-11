"""End-to-end smoke test on SYNTHETIC data.

This does not reproduce the report's numbers (it uses random data); it only
proves the pipeline wiring is correct: features build, the time split works, both
models train and score, and the ablation, importance and figures all run.

    python -m tests.test_smoke
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.build_dataset import build_modelling_table, split_xy
from src.models import MODEL_FACTORIES, MODEL_LABELS
from src.evaluate import evaluate
from src.ablation import run_ablation
from src.importance import run_permutation_importance
from src import plots

RNG = np.random.default_rng(0)


def _synth_finra(n_symbols=60, n_days=22):
    dates = pd.bdate_range("2026-07-01", periods=n_days)
    rows = []
    for i in range(n_symbols):
        sym = f"SYM{i:03d}"
        base = RNG.uniform(0.2, 0.6)
        for d in dates:
            total = RNG.integers(10_000, 500_000)
            share = np.clip(base + RNG.normal(0, 0.1), 0.01, 0.99)
            short = int(total * share * RNG.uniform(0.8, 1.0))
            exempt = int(total * share * RNG.uniform(0.0, 0.05))
            rows.append((sym, d, short, exempt, total))
    df = pd.DataFrame(rows, columns=["symbol", "date", "short_volume",
                                     "short_exempt_volume", "total_volume"])
    df["short_share"] = (df.short_volume + df.short_exempt_volume) / df.total_volume
    df["exempt_share"] = df.short_exempt_volume / df.total_volume
    return df


def _synth_sec(n_symbols=60):
    sub = pd.DataFrame({
        "adsh": [f"000-{i:04d}" for i in range(n_symbols)],
        "cik": list(range(1, n_symbols + 1)),
        "sic": RNG.integers(1000, 9000, n_symbols).astype(str),
        "form": RNG.choice(["10-K", "10-Q"], n_symbols),
        "period": "20260630",
        "filed": "20260615",
    })
    tags = list(config.SEC_TAGS.values())
    num_rows = []
    for i in range(n_symbols):
        for tag in tags:
            num_rows.append({
                "adsh": f"000-{i:04d}", "tag": tag, "version": "us-gaap/2026",
                "ddate": "20260630", "qtrs": "0", "uom": "USD",
                "segments": "", "value": float(RNG.uniform(1e6, 1e10)),
            })
    num = pd.DataFrame(num_rows)
    ticker_map = pd.DataFrame({"cik": list(range(1, n_symbols + 1)),
                               "ticker": [f"SYM{i:03d}" for i in range(n_symbols)]})
    return sub, num, ticker_map


def main():
    finra = _synth_finra()
    sub, num, ticker_map = _synth_sec()

    df, meta = build_modelling_table(finra, sub, num, ticker_map)
    assert set(df["split"]) <= {"train", "valid", "test"}
    assert meta["counts"].get("train", 0) > 0
    assert meta["counts"].get("test", 0) > 0
    assert 0.0 < meta["test_prevalence"] < 1.0

    feature_cols = meta["finra_cols"] + meta["sec_cols"]
    assert len(meta["finra_cols"]) >= 6
    assert len(meta["sec_cols"]) >= 6

    pr_data = []
    for key, factory in MODEL_FACTORIES.items():
        model = factory()
        Xtr, ytr = split_xy(df, feature_cols, "train")
        Xte, yte = split_xy(df, feature_cols, "test")
        model.fit(Xtr, ytr)
        score = model.predict_proba(Xte)[:, 1]
        m = evaluate(yte, score)
        assert 0 <= m["pr_auc"] <= 1
        assert 0 <= m["precision@budget"] <= 1
        pr_data.append((MODEL_LABELS[key], yte.to_numpy(), score, "#1C7293"))

    ablation = run_ablation(df, meta)
    assert {"FINRA", "SEC", "Integrated"} <= set(ablation.columns)

    importance = run_permutation_importance(df, meta)
    assert len(importance) == len(feature_cols)

    # Write throwaway figures to a temp dir so committed report figures in
    # figures/ are never overwritten by the test.
    import tempfile
    tmp = tempfile.mkdtemp(prefix="smoke_fig_")
    plots.plot_target_and_split(df, meta, os.path.join(tmp, "target_and_split.png"))
    plots.plot_precision_recall(pr_data, os.path.join(tmp, "precision_recall.png"))
    plots.plot_ablation(ablation, os.path.join(tmp, "ablation.png"))
    plots.plot_permutation_importance(importance,
                                      path=os.path.join(tmp, "importance.png"))

    print("SMOKE TEST PASSED")
    print("  rows:", len(df), "| finra feats:", len(meta["finra_cols"]),
          "| sec feats:", len(meta["sec_cols"]))
    print("  test prevalence:", round(meta["test_prevalence"], 3))


if __name__ == "__main__":
    main()
