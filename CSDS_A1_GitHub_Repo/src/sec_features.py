"""Issuer-level features from the SEC Financial Statement and Notes archive.

For each CIK we keep the latest accepted 10-K/10-Q/20-F/40-F, extract a small
set of non-dimensional financial facts from num.tsv, and engineer size,
leverage, liquidity and profitability ratios plus SIC-division indicators.
All columns are grouped under the ``sec: `` prefix.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config

P = config.SEC_PREFIX

# First digit of the SIC code -> broad economic division (US SIC structure).
SIC_DIVISION = {
    0: "A_agri", 1: "B_mining_construction", 2: "C_manufacturing_light",
    3: "D_manufacturing_heavy", 4: "E_transport_utilities", 5: "F_trade",
    6: "G_finance", 7: "H_services", 8: "I_services_prof", 9: "J_public",
}


def _latest_submissions(sub: pd.DataFrame) -> pd.DataFrame:
    """One row per CIK: the most recently filed eligible form."""
    s = sub.copy()
    s["cik"] = pd.to_numeric(s["cik"], errors="coerce")
    s = s[s["form"].isin(config.SEC_FORMS)]
    s["filed"] = pd.to_datetime(s["filed"], format="%Y%m%d", errors="coerce")
    s = s.dropna(subset=["cik", "filed"])
    s = s.sort_values("filed").groupby("cik", as_index=False).last()
    keep = ["adsh", "cik", "sic", "period", "filed", "form"]
    return s[[c for c in keep if c in s.columns]]


def _extract_facts(num: pd.DataFrame, submissions: pd.DataFrame) -> pd.DataFrame:
    """Pivot the required XBRL tags to one row per submission (adsh)."""
    n = num.copy()
    n = n[n["adsh"].isin(submissions["adsh"])]

    # Keep only non-dimensional facts (no segment breakdown). Across archive
    # versions the marker is an empty ``segments`` or a zero ``dimh`` hash.
    for dim_col in ("segments", "dimh", "dimn"):
        if dim_col in n.columns:
            blank = n[dim_col].isna() | n[dim_col].astype(str).str.strip().isin(
                ["", "0", "0x00000000", "nan"])
            n = n[blank]
            break

    n["value"] = pd.to_numeric(n["value"], errors="coerce")
    n["ddate"] = pd.to_datetime(n.get("ddate"), format="%Y%m%d", errors="coerce")
    n = n.dropna(subset=["value"])

    wanted = set(config.SEC_TAGS.values())
    n = n[n["tag"].isin(wanted)]

    # Most recent period per (adsh, tag); prefer full-year/quarter durations.
    n = n.sort_values("ddate").groupby(["adsh", "tag"], as_index=False).last()
    wide = n.pivot_table(index="adsh", columns="tag", values="value",
                         aggfunc="last")
    return wide.reset_index()


def build_sec_features(sub: pd.DataFrame, num: pd.DataFrame,
                       ticker_map: pd.DataFrame) -> pd.DataFrame:
    """Return one row per ticker with engineered ``sec: `` features."""
    submissions = _latest_submissions(sub)
    facts = _extract_facts(num, submissions)
    df = submissions.merge(facts, on="adsh", how="left")
    df = df.merge(ticker_map, on="cik", how="inner")   # only filers we can map

    t = config.SEC_TAGS
    def col(name):  # tag column may be absent if never reported this month
        return df[t[name]] if t[name] in df.columns else pd.Series(np.nan, index=df.index)

    assets = col("assets")
    revenue = col("revenue")
    liabilities = col("liabilities")
    cur_assets = col("current_assets")
    cur_liab = col("current_liabilities")
    cash = col("cash")
    net_income = col("net_income")
    op_income = col("operating_income")

    out = pd.DataFrame({"ticker": df["ticker"].str.upper()})
    out[P + "log assets"] = np.log1p(assets.clip(lower=0))
    out[P + "log revenue"] = np.log1p(revenue.clip(lower=0))
    out[P + "liabilities to assets"] = liabilities / assets
    out[P + "current ratio"] = cur_assets / cur_liab
    out[P + "cash to assets"] = cash / assets
    out[P + "return on assets"] = net_income / assets
    out[P + "operating margin"] = op_income / revenue

    # SIC division indicators.
    sic = pd.to_numeric(df.get("sic"), errors="coerce")
    division = (sic // 1000).map(SIC_DIVISION)
    dummies = pd.get_dummies(division, prefix=(P + "div").strip(), dtype=float)
    out = pd.concat([out, dummies], axis=1)

    # Replace infinities from ratio division-by-zero with NaN (imputed later).
    out = out.replace([np.inf, -np.inf], np.nan)
    # One row per ticker (latest filing already selected upstream).
    return out.drop_duplicates("ticker").reset_index(drop=True)


def sec_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(config.SEC_PREFIX)]
