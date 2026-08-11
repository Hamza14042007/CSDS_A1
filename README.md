# Next-Day Short-Sale Monitoring — FINRA + SEC

Reproducible code for **Case Studies in Data Science (Individual Task 1)**.

This project supports the data-science case study written around Vanguard
Australia's *Data Scientist, Specialist* role. It asks a single, practical
question:

> At the end of a FINRA trading day, can we rank which securities are likely to
> record an **unusually high off-exchange short-sale share on the next trading
> day** — accurately enough to build a fixed-capacity review queue?

The output is a **market-activity monitoring flag**, not a price forecast and not
a claim about investor intent. A human analyst reviews the ranked queue; the model
only directs attention.

> **Note on scope.** This repository accompanies the written report and reproduces
> its methodology end to end. Exact counts and metrics depend on the specific FINRA
> daily files and SEC monthly archive you download, which can be revised at source.

---

## What the pipeline does

1. **Ingest** 22 daily consolidated FINRA short-sale files (July 2026) and the
   SEC *Financial Statement and Notes* monthly archive (June 2026), frozen
   **before** the prediction window to prevent look-ahead.
2. **Engineer features**
   - *FINRA (`finra:`)* — current short share, exempt share, log volume, 3-day
     mean, 5-day volatility and 1-day change of short-sale share.
   - *SEC (`sec:`)* — log assets, log revenue, liabilities/assets, current ratio,
     cash/assets, return on assets, operating margin, and SIC-division indicators.
3. **Join and split in time** — inner join on ticker, then a strict chronological
   split (no shuffling): train → validation → test.
4. **Define the target** — positive class = next-day short-sale share above the
   **75th percentile of the training period only**.
5. **Train two models** — an elastic-net logistic regression (linear baseline)
   and a histogram gradient boosting classifier (non-linear challenger).
6. **Evaluate, ablate, explain** — headline metrics, a FINRA/SEC/integrated source
   ablation, and permutation importance for the integrated boosting model.

## Repository layout

```
config.py                  # all paths, date windows, target rule, hyper-parameters
src/
  data_ingest.py           # load + validate FINRA files, SEC sub/num, ticker map
  finra_features.py        # trailing market features + next-day target
  sec_features.py          # issuer financial ratios + SIC-division indicators
  build_dataset.py         # join, chronological split, train-only threshold
  models.py                # the two sklearn pipelines (exact hyper-parameters)
  evaluate.py              # PR-AUC, ROC-AUC, Brier, precision/recall/lift@20%
  ablation.py              # FINRA-only vs SEC-only vs integrated
  importance.py            # permutation importance (drop in PR-AUC)
  plots.py                 # the four report figures
scripts/
  download_data.py         # fetch FINRA daily files + SEC ticker map
  run_experiment.py        # end-to-end driver -> results/ + figures/
tests/
  test_smoke.py            # runs the whole pipeline on synthetic data
figures/                   # the four figures used in the report
```

## Quick start

```bash
# 1. Install dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 2. Verify the pipeline wiring on synthetic data (no downloads needed)
python -m tests.test_smoke

# 3. Fetch the raw public inputs
#    - edit the User-Agent in scripts/download_data.py to your email first
python -m scripts.download_data
#    Then download + unzip the SEC monthly notes archive into data/raw/sec/
#    (the script prints the exact URL). You should end up with sub.tsv and num.tsv.

# 4. Run the full experiment
python -m scripts.run_experiment
```

Results are written to `results/` (`metrics.json`, `ablation.csv`,
`permutation_importance.csv`) and figures to `figures/`.

## Data sources

| Source | What it provides | Link |
|---|---|---|
| FINRA Reg SHO daily short-sale volume (consolidated CNMS) | Date, Symbol, ShortVolume, ShortExemptVolume, TotalVolume, Market | <https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data> |
| SEC Financial Statement and Notes Data Sets | XBRL filings (`sub.tsv`, `num.tsv`, …) — issuer identity, industry, financials | <https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets> |
| SEC company tickers | CIK → ticker map | <https://www.sec.gov/files/company_tickers.json> |

Raw data is **not** committed (see `.gitignore`); download it with the script.
The SEC requires a descriptive `User-Agent` with contact details — set yours in
`scripts/download_data.py` before running.

## Headline results (as reported)

Integrated models, held-out test (1,639 stock-days, positive rate 0.248):

| Model | PR-AUC | ROC-AUC | Brier | P@20% | R@20% | Lift@20% |
|---|---:|---:|---:|---:|---:|---:|
| Elastic-net logistic | 0.517 | 0.777 | 0.192 | 0.558 | 0.451 | 2.25× |
| **Histogram gradient boosting** | **0.580** | **0.793** | **0.174** | **0.576** | **0.466** | **2.33×** |

Source ablation (test PR-AUC):

| Model | FINRA | SEC | Integrated | Gain vs FINRA |
|---|---:|---:|---:|---:|
| Elastic-net logistic | 0.502 | 0.334 | 0.517 | +0.015 |
| Histogram gradient boosting | 0.567 | 0.491 | 0.580 | +0.012 |

**Takeaways.** Boosting is stronger on every threshold-free score. The sources are
*complementary, not contradicting*: recent FINRA activity — especially the 3-day
mean short-sale share — carries most of the signal, while SEC filings add only a
small (+0.012) but positive contribution.

## Why these evaluation metrics

- **PR-AUC (primary)** — only ~25% of test days are positive; PR-AUC rewards
  recovering the rare positives while holding precision. Accuracy is uninformative
  here (predicting "normal" everywhere is already ~75% accurate).
- **ROC-AUC** — threshold-free ranking across both classes.
- **Brier score** — probability calibration quality (lower is better).
- **Precision / recall / lift @ 20%** — operational value under a fixed review
  budget: how concentrated a top-20% queue is versus reviewing at random.

## Limitations

- Coverage is selective — only symbols whose issuer filed an eligible form in the
  chosen SEC month are included; not a full market cross-section.
- FINRA reports **publicly disseminated off-exchange volume**, not
  exchange-consolidated trading or short-interest positions. A high share is not a
  bearish position by itself.
- The reported test spans five scoring dates — a prototype, not a proven system.
- Repeated tickers across dates are intentional (the application scores known
  securities over time); an **issuer-holdout** test is required before claiming
  transfer to unseen companies.

## Reproducibility notes

- All windows, the target quantile, and both models' hyper-parameters live in
  `config.py`; change them there to roll the study forward.
- The chronological split and train-only threshold prevent leakage; the test block
  is scored once and never used for fitting.
- `random_state` is fixed throughout.

## License

MIT — see [LICENSE](LICENSE).
