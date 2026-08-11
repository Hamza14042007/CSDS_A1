"""Trailing market features and the next-day target from FINRA daily data.

Each feature uses only information visible on the scoring date; the label is the
NEXT trading day's short-sale share. Features are grouped under the ``finra: ``
prefix so the ablation and permutation-importance plots can pick them out.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config

P = config.FINRA_PREFIX


def build_finra_features(finra: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (symbol, date) with trailing features + next_share.

    Parameters
    ----------
    finra : tidy consolidated frame from :func:`data_ingest.load_finra`.
    """
    df = finra.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", group_keys=False)

    # Current-day, information-as-of-close features.
    df[P + "short share"] = df["short_share"]
    df[P + "short exempt share"] = df["exempt_share"]
    df[P + "log total volume"] = np.log1p(df["total_volume"])

    # Trailing behaviour (min_periods keeps early days but flags them as NaN
    # where a full window is not yet available).
    df[P + "short share ma3"] = g["short_share"].transform(
        lambda s: s.rolling(3, min_periods=2).mean())
    df[P + "short share sd5"] = g["short_share"].transform(
        lambda s: s.rolling(5, min_periods=2).std())
    df[P + "short share change 1d"] = g["short_share"].transform(
        lambda s: s.diff(1))

    # Next-day outcome. shift(-1) on a per-symbol, date-sorted series gives the
    # next available trading day; only consecutive pairs get a label.
    df["next_share"] = g["short_share"].shift(-1)
    df["next_date"] = g["date"].shift(-1)

    feature_cols = [c for c in df.columns if c.startswith(P)]
    keep = ["symbol", "date", "next_date", "next_share", *feature_cols]
    out = df[keep].dropna(subset=["next_share"]).reset_index(drop=True)
    return out


def finra_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(config.FINRA_PREFIX)]
