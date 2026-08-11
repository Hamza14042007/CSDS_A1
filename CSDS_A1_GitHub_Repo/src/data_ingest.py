"""Load and validate the two raw public sources.

FINRA: daily consolidated (CNMS) regulation-SHO short-sale volume files.
       Six fields per row: Date, Symbol, ShortVolume, ShortExemptVolume,
       TotalVolume, Market.
SEC:   Financial Statement and Notes monthly archive (sub.tsv, num.tsv) plus
       the SEC CIK -> ticker map (company_tickers.json).

Nothing here fabricates data: if the raw files are missing the functions raise
so the experiment cannot silently run on nothing.
"""
from __future__ import annotations
import glob
import json
import os

import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------- #
# FINRA
# --------------------------------------------------------------------------- #
def load_finra(finra_dir: str = config.FINRA_RAW) -> pd.DataFrame:
    """Read every FINRA daily file in ``finra_dir`` into one tidy frame.

    FINRA daily files are pipe-delimited with a trailing summary line that we
    drop. Rows with non-positive total volume are rejected and the derived
    short-sale share is quarantined to the ``[0, 1]`` interval.
    """
    paths = sorted(glob.glob(os.path.join(finra_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(
            f"No FINRA daily files found in {finra_dir}. "
            f"Run scripts/download_data.py first."
        )

    frames = []
    for p in paths:
        df = pd.read_csv(p, sep="|", dtype=str)
        # Drop the trailing footer row FINRA appends (no valid Symbol).
        df = df[df.get("Symbol").notna() & (df.get("Symbol") != "")]
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    raw = raw.rename(columns={
        "Date": "date", "Symbol": "symbol",
        "ShortVolume": "short_volume",
        "ShortExemptVolume": "short_exempt_volume",
        "TotalVolume": "total_volume", "Market": "market",
    })

    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m%d", errors="coerce")
    for c in ("short_volume", "short_exempt_volume", "total_volume"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.dropna(subset=["date", "symbol", "total_volume"])
    raw = raw[raw["total_volume"] > 0]                      # reject non-positive volume

    raw["short_share"] = (
        (raw["short_volume"].fillna(0) + raw["short_exempt_volume"].fillna(0))
        / raw["total_volume"]
    )
    raw = raw[(raw["short_share"] >= 0) & (raw["short_share"] <= 1)]  # quarantine

    # Collapse any duplicate market rows to one consolidated record per day.
    agg = (raw.groupby(["symbol", "date"], as_index=False)
              .agg(short_volume=("short_volume", "sum"),
                   short_exempt_volume=("short_exempt_volume", "sum"),
                   total_volume=("total_volume", "sum")))
    agg["short_share"] = (
        (agg["short_volume"] + agg["short_exempt_volume"]) / agg["total_volume"]
    )
    agg["exempt_share"] = agg["short_exempt_volume"] / agg["total_volume"]
    return agg.sort_values(["symbol", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SEC
# --------------------------------------------------------------------------- #
def load_ticker_map(path: str = config.TICKER_MAP) -> pd.DataFrame:
    """SEC company_tickers.json -> DataFrame[cik, ticker]."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing. Download https://www.sec.gov/files/company_tickers.json"
        )
    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    rows = [{"cik": int(v["cik_str"]), "ticker": v["ticker"].upper()}
            for v in blob.values()]
    return pd.DataFrame(rows).drop_duplicates("cik")


def load_sec(sec_dir: str = config.SEC_RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``sub.tsv`` and ``num.tsv`` from an unzipped SEC notes archive."""
    sub_path = os.path.join(sec_dir, "sub.tsv")
    num_path = os.path.join(sec_dir, "num.tsv")
    for p in (sub_path, num_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing. Unzip the SEC monthly notes archive into {sec_dir}."
            )
    sub = pd.read_csv(sub_path, sep="\t", dtype=str, low_memory=False)
    num = pd.read_csv(num_path, sep="\t", dtype=str, low_memory=False)
    return sub, num
