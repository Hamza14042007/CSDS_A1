"""Generate the four report figures from computed results.

Produces: target_and_split.png, precision_recall_curves.png,
ablation_pr_auc.png, permutation_importance.png in the figures/ directory.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

import config

NAVY, GOLD, TEAL, BERRY = "#10263F", "#B9822B", "#1C7293", "#8E1F3C"
plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def plot_target_and_split(df, meta, path=None):
    path = path or os.path.join(config.FIGURES_DIR, "target_and_split.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax1.hist(df["next_share"], bins=40, color=NAVY, alpha=0.85)
    ax1.axvline(meta["threshold"], color=BERRY, lw=2,
                label=f"Training 75th percentile = {meta['threshold']:.3f}")
    ax1.set_title("Target definition")
    ax1.set_xlabel("Next-day FINRA short-sale share")
    ax1.set_ylabel("Stock-days")
    ax1.legend(fontsize=8)

    counts = meta["counts"]
    order = ["train", "valid", "test"]
    labels = ["Train", "Validate", "Test"]
    vals = [counts.get(k, 0) for k in order]
    bars = ax2.bar(labels, vals, color=[NAVY, GOLD, BERRY])
    ax2.set_title("Chronological split")
    ax2.set_ylabel("Stock-days")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                 ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_precision_recall(pr_data, path=None):
    """pr_data: list of (label, y_true, y_score, color)."""
    path = path or os.path.join(config.FIGURES_DIR, "precision_recall_curves.png")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    prevalence = None
    for label, y_true, y_score, color in pr_data:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(rec, prec, color=color, lw=2, label=f"{label} (AP={ap:.3f})")
        prevalence = float(np.mean(y_true))
    if prevalence is not None:
        ax.axhline(prevalence, ls="--", color="grey",
                   label=f"Prevalence ({prevalence:.3f})")
    ax.set_title("Next-day high short-share detection: integrated models")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_ablation(ablation_df, path=None):
    path = path or os.path.join(config.FIGURES_DIR, "ablation_pr_auc.png")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    models = ablation_df["model"].tolist()
    sets = ["FINRA", "SEC", "Integrated"]
    colors = {"FINRA": NAVY, "SEC": GOLD, "Integrated": BERRY}
    x = np.arange(len(models)); width = 0.26
    for i, s in enumerate(sets):
        vals = ablation_df[s].tolist()
        bars = ax.bar(x + (i - 1) * width, vals, width, label=s, color=colors[s])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(0.248, ls="--", color="grey", label="No-skill prevalence")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Test PR-AUC")
    ax.set_title("Ablation test: what each data source contributes")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_permutation_importance(imp_df, top_n=10, path=None):
    path = path or os.path.join(config.FIGURES_DIR, "permutation_importance.png")
    top = imp_df.head(top_n).iloc[::-1]
    colors = [TEAL if f.startswith(config.FINRA_PREFIX) else BERRY
              for f in top["feature"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(top["feature"], top["importance"], xerr=top["std"],
            color=colors, capsize=3)
    ax.set_xlabel("Decrease in PR-AUC after permutation")
    ax.set_title("Integrated boosting model: test-set permutation importance")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path
