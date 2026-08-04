# K-Index Macro/Asset-Price Regressions

Responds to the advisor feedback (in addition to Prof. Girand's comments):

1. Does the K-Index explain asset prices? Regress % change in stock market
   indexes, bond total returns (e.g. 10Y Treasury), % change in the dollar
   vs. JPY/EUR/GBP, and % change in gold, on a constant, the current value
   of K, and lags.
2. Estimate a model of economic growth (unemployment rate, GDP, industrial
   production) as a function of K.

## Status

**The K-Index itself is real and rebuilt from Pillar 1's original source
files** (`k_index_builder.py`, a port of `KIndex_Complete_1.ipynb`). It is
currently a **2-pillar version (wealth + income)** — the consumer pillar
needs `bluebk02n.xls` (U. Michigan sentiment by income tercile), which
hasn't been added to the repo yet. `build_k_index()` upgrades to the full
3-pillar version automatically the moment that file is added — no code
changes needed, and every printed/saved output labels which version ran.

**Every regression the advisor asked for is unrun.** They need real
quarterly data for 9 series (stock index, 10Y Treasury total return,
USD/JPY, USD/EUR, USD/GBP, gold, unemployment rate, GDP, industrial
production), and this sandbox cannot fetch any of them — confirmed directly
via `curl` against FRED, Yahoo Finance, Stooq, Alpha Vantage, an
exchange-rate API, BLS, and BEA; all return a 403 from the network policy,
not a code bug. Running `run_k_regressions.py` right now produces a clean
"skipped" message per missing series rather than a fabricated result (see
below) — the code is complete and correct, just waiting on data.

## Validation: the wealth and income pillars are exact

`dfa-networth-levels-detail.csv` and `wage-growth-data.xlsx` were added to
the repo and run through the notebook's exact logic
(`build_wealth_pillar()`, `build_income_pillar()`). Both match the
notebook's own printed output to six decimal places:

| Quarter | Wealth (notebook) | Wealth (this repo) | Income (notebook) | Income (this repo) |
|---|---|---|---|---|
| 2025:Q3 | 139.0827 | 139.0827 | — | — |
| 2025:Q4 | 139.305421 | 139.305421 | 0.600000 | 0.600000 |
| 2026:Q1 | 138.860484 | 138.860484 | 0.366667 | 0.366667 |
| 2026:Q2 | — | — | 0.200000 | 0.200000 |

Before these two files existed, this repo's own `dfa-networth-shares-detail.csv`
(percentage shares, not dollar levels) was used to *approximate* the wealth
pillar by multiplying the rounded published percentages by a total net
worth derived from `dfa-generation-levels-detail.csv`. That got within ~2%
(136x vs. 139x) — close enough to confirm the approach was sound, but the
~2% gap (traced to rounding in the published 1-decimal percentages, not a
scope mismatch — the two levels files' totals reconcile to $2M on a $174T
series) is exactly why the actual `dfa-networth-levels-detail.csv` file
matters and the approximation isn't used anywhere in this module.

## What's built

- `k_index_builder.py` — `build_wealth_pillar()`, `build_income_pillar()`,
  `build_consumer_pillar()` (returns `None` if `bluebk02n.xls` is absent),
  and `build_k_index()` (equal-weight `K`, plus `K_pca` via direct SVD and
  `K_invvar`, mirroring the notebook's robustness checks).
- `target_data.py` — loads each of the 9 target series from
  `data_cache/<name>.csv` (columns `Date,Value`); computes % change or
  first-difference as appropriate. Raises a clear per-series error if the
  file is missing rather than attempting (and failing) a live pull.
- `regressions.py` — `run_k_regression(target, k, n_lags=4)`: OLS of the
  transformed target on a constant, current `K`, and `n_lags` lagged values
  of `K`, with HAC (Newey-West) standard errors.
- `run_k_regressions.py` — entry point; runs both regression families,
  skipping (not fabricating) any target whose CSV isn't present yet.

## What's needed to get real numbers

**To complete the K-Index (3rd pillar):** `bluebk02n.xls` (U. Michigan
Surveys of Consumers, income-tercile table) added to the repo root.

**For the regressions themselves**, drop CSVs (columns `Date,Value`, a
level not a % change — the code computes the transform) into
`k_index_model/data_cache/`:

| File | Suggested source | Notes |
|---|---|---|
| `sp500.csv` | S&P 500 index level (or another broad index) | |
| `treasury_10y_total_return.csv` | A real total-return series/index/ETF NAV | The constant-maturity *yield* (e.g. FRED `DGS10`) is not a total return and needs converting via duration first — a real total-return series is much cleaner if available |
| `usdjpy.csv`, `usdeur.csv`, `usdgbp.csv` | Spot exchange rates | State which direction (USD per FX unit, or vice versa) in how the file's produced — just be consistent across the three |
| `gold.csv` | USD/oz spot or futures | |
| `unemployment_rate.csv` | U-3 rate (%) | Regressed as first-difference, not % change (see below) |
| `gdp.csv` | Real GDP level (e.g. FRED `GDPC1`) | |
| `industrial_production.csv` | IP index level (e.g. FRED `INDPRO`) | |

Quarterly frequency to match K; monthly series (unemployment, IP, FX, gold)
get resampled to quarter-end automatically by `target_data.py`.

**A methodology note on unemployment:** the advisor's brief says "percentage
change" for all the market variables, but for a rate like unemployment
(already in percentage points), a first difference (e.g. 4.0% → 4.2%, a
+0.2 point move) is the standard, more interpretable transform — "%
change in a percent" is a less standard measure and can behave oddly near
zero. `target_data.load_diff()` is used for unemployment for this reason;
flag if a literal % change is wanted instead, it's a one-line change.

## Running it

```bash
cd k_index_model
pip install -r requirements.txt
python k_index_builder.py       # rebuild and inspect K on its own
python run_k_regressions.py     # the actual advisor-requested regressions
```
