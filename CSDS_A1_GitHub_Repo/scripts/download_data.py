"""Fetch the raw public inputs for the experiment.

Downloads:
  1. FINRA daily consolidated (CNMS) short-sale files for the configured month.
  2. The SEC CIK -> ticker map (company_tickers.json).
  3. Prints instructions for the SEC Financial Statement and Notes archive,
     which is a large ZIP that you unzip into data/raw/sec/.

The SEC requires a descriptive User-Agent with contact details; edit SEC_UA
below before running. Nothing here is scraped aggressively: one request per file.

    python -m scripts.download_data
"""
from __future__ import annotations
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# SEC's fair-access policy requires a real contact address here.
SEC_UA = "CSDS-A1 research (your.email@student.rmit.edu.au)"

FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_NOTES_URL = (
    "https://www.sec.gov/files/dera/data/financial-statement-and-notes-data-sets/"
    "{archive}.zip"
)


def _get(url: str, dest: str, ua: str = SEC_UA):
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as fh:
        fh.write(r.read())


def download_finra_month(month: str = config.FINRA_MONTH):
    """Download every weekday FINRA file in ``month`` (YYYY-MM)."""
    import pandas as pd
    start = pd.Timestamp(month + "-01")
    days = pd.date_range(start, start + pd.offsets.MonthEnd(0), freq="B")
    for d in days:
        stamp = d.strftime("%Y%m%d")
        dest = os.path.join(config.FINRA_RAW, f"CNMSshvol{stamp}.txt")
        if os.path.exists(dest):
            continue
        url = FINRA_URL.format(yyyymmdd=stamp)
        try:
            _get(url, dest, ua="Mozilla/5.0 (CSDS-A1 research)")
            print("  finra", stamp, "ok")
        except Exception as e:  # holidays / weekends return 404 -> skip
            print("  finra", stamp, "skip", e)
        time.sleep(0.5)


def download_ticker_map():
    print("Downloading SEC ticker map ...")
    _get(TICKER_URL, config.TICKER_MAP)
    print("  ->", config.TICKER_MAP)


def main():
    print("Downloading FINRA daily files ...")
    download_finra_month()
    download_ticker_map()
    print()
    print("SEC Financial Statement and Notes archive is large; download and "
          "unzip it manually into", config.SEC_RAW)
    print("  URL:", SEC_NOTES_URL.format(archive=config.SEC_ARCHIVE))
    print("  After unzipping you should have sub.tsv and num.tsv in that folder.")


if __name__ == "__main__":
    main()
