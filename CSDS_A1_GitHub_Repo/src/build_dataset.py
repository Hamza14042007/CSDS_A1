"""Join FINRA behaviour with SEC issuer context and split in time.

Steps (order matters and prevents leakage):
  1. Build trailing FINRA features + next-day share.
  2. Attach frozen SEC issuer features by ticker (inner join).
  3. Split by trading date into train / validation / test.
  4. Define the positive label from the TRAIN-only target quantile.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config
from src.finra_features import build_finra_features, finra_feature_columns
from src.sec_features import build_sec_features, sec_feature_columns


def _between(df: pd.DataFrame, window: tuple[str, str]) -> pd.Series:
    lo, hi = pd.to_datetime(window[0]), pd.to_datetime(window[1])
    return (df["date"] >= lo) & (df["date"] <= hi)


def build_modelling_table(finra, sub, num, ticker_map):
    """Return (df, meta) where df has features, split tag and label."""
    fin = build_finra_features(finra)
    sec = build_sec_features(sub, num, ticker_map)

    fin["ticker"] = fin["symbol"].str.upper()
    df = fin.merge(sec, on="ticker", how="inner")   # only symbols with a filing

    # Chronological split.
    df["split"] = np.where(_between(df, config.TRAIN_DATES), "train",
                    np.where(_between(df, config.VALID_DATES), "valid",
                    np.where(_between(df, config.TEST_DATES), "test", "drop")))
    df = df[df["split"] != "drop"].reset_index(drop=True)

    # Target threshold from TRAIN next-day share only.
    train_share = df.loc[df["split"] == "train", "next_share"]
    threshold = float(train_share.quantile(config.TARGET_QUANTILE))
    df["label"] = (df["next_share"] > threshold).astype(int)

    meta = {
        "threshold": threshold,
        "finra_cols": finra_feature_columns(df),
        "sec_cols": sec_feature_columns(df),
        "n_symbols": df["symbol"].nunique(),
        "counts": df["split"].value_counts().to_dict(),
        "test_prevalence": float(df.loc[df["split"] == "test", "label"].mean()),
    }
    return df, meta


def split_xy(df: pd.DataFrame, feature_cols: list[str], split: str):
    part = df[df["split"] == split]
    X = part[feature_cols].astype(float)
    y = part["label"].astype(int)
    return X, y
