"""Central configuration for the FINRA + SEC next-day short-sale experiment.

All paths, date windows, the target threshold rule and the two models'
hyper-parameters are defined here so the pipeline is fully reproducible and
every number in the report can be traced back to a single place.
"""
from __future__ import annotations
import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
FIGURES_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")

FINRA_RAW = os.path.join(DATA_RAW, "finra")   # daily CNMSshvol*.txt files
SEC_RAW = os.path.join(DATA_RAW, "sec")       # unzipped monthly notes archive (sub.tsv, num.tsv, ...)
TICKER_MAP = os.path.join(DATA_RAW, "company_tickers.json")

for _d in (DATA_RAW, DATA_PROCESSED, FIGURES_DIR, RESULTS_DIR, FINRA_RAW, SEC_RAW):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Experiment windows  (edit these to roll the study forward)
# --------------------------------------------------------------------------- #
# FINRA daily consolidated (CNMS) short-sale files to ingest.
FINRA_MONTH = "2026-07"          # July 2026 trading month

# SEC Financial Statement and Notes monthly archive that is frozen BEFORE the
# FINRA prediction window (prevents look-ahead of issuer information).
SEC_ARCHIVE = "2026_06_notes"    # June 2026 monthly archive

# Chronological split by trading date (inclusive). No shuffling: later market
# patterns must never leak into training.
TRAIN_DATES = ("2026-07-01", "2026-07-17")   # -> ~3,945 rows in the report
VALID_DATES = ("2026-07-20", "2026-07-23")   # -> ~1,307 rows
TEST_DATES  = ("2026-07-24", "2026-07-30")   # -> ~1,639 rows

# --------------------------------------------------------------------------- #
# Target definition
# --------------------------------------------------------------------------- #
# Positive class = next trading day's off-exchange short-sale share exceeds the
# training-period quantile below. Computing the cut-off on TRAIN ONLY avoids
# using test labels to define what counts as "unusual".
TARGET_QUANTILE = 0.75           # 75th percentile -> ~0.642 cut-off in the report
REVIEW_BUDGET = 0.20             # operational metrics evaluate the top 20% queue

RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Feature groups (column-name prefixes drive the ablation and the plots)
# --------------------------------------------------------------------------- #
FINRA_PREFIX = "finra: "
SEC_PREFIX = "sec: "

# SEC XBRL tags pulled from num.tsv to engineer issuer-level features.
SEC_TAGS = {
    "assets": "Assets",
    "revenue": "Revenues",
    "liabilities": "Liabilities",
    "current_assets": "AssetsCurrent",
    "current_liabilities": "LiabilitiesCurrent",
    "cash": "CashAndCashEquivalentsAtCarryingValue",
    "net_income": "NetIncomeLoss",
    "operating_income": "OperatingIncomeLoss",
}
# Forms treated as the issuer's latest accepted financial picture.
SEC_FORMS = ("10-K", "10-Q", "20-F", "40-F")

# --------------------------------------------------------------------------- #
# Model hyper-parameters (exactly as reported)
# --------------------------------------------------------------------------- #
ELASTICNET_PARAMS = dict(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.20,           # 20% L1 / 80% L2
    C=0.5,
    class_weight="balanced",
    max_iter=5000,
    random_state=RANDOM_SEED,
)

HGB_PARAMS = dict(
    learning_rate=0.06,
    max_iter=220,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.15,
    class_weight="balanced",
    random_state=RANDOM_SEED,
)

PERMUTATION_REPEATS = 20
