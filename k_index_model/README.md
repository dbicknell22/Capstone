# K-Index Macro/Asset-Price Regressions

Responds to the advisor feedback (in addition to Prof. Girand's comments):

1. Does the K-Index explain asset prices? Regress % change in stock market
   indexes, bond total returns (e.g. 10Y Treasury), % change in the dollar
   vs. JPY/EUR/GBP, and % change in gold, on a constant, the current value
   of K, and lags.
2. Estimate a model of economic growth (unemployment rate, GDP, industrial
   production) as a function of K.

## Status

**The K-Index is complete and fully validated — all 3 pillars, exact match
to the original notebook.** `bluebk02n.xls` (U. Michigan sentiment) has been
added alongside the other two source files, so `build_k_index()` now
produces the real wealth + income + consumer composite, not the 2-pillar
placeholder from before. Every number it produces — z-scored pillars, K,
K_pca, K_invvar, pillar correlations, leave-one-out, and all three Chow
tests — was checked directly against the notebook's own printed output and
matches (see "Validation" below). This is not a proxy or a rebuild that
happens to look similar; it reproduces Pillar 1's original K-Index exactly.

**Every regression the advisor asked for is still unrun.** They need real
quarterly data for 9 series (stock index, 10Y Treasury total return,
USD/JPY, USD/EUR, USD/GBP, gold, unemployment rate, GDP, industrial
production), and this sandbox cannot fetch any of them — confirmed directly
via `curl` against FRED, Yahoo Finance, Stooq, Alpha Vantage, an
exchange-rate API, BLS, and BEA; all return a 403 from the network policy,
not a code bug. Running `run_k_regressions.py` right now produces a clean
"skipped" message per missing series rather than a fabricated result (see
below) — the code is complete and correct, just waiting on data.

## Validation: all three pillars, and the full composite, are exact

`dfa-networth-levels-detail.csv`, `wage-growth-data.xlsx`, and `bluebk02n.xls`
are all in the repo now, run through the notebook's exact logic. Every
figure below is checked directly against the notebook's own printed output:

| Metric | Notebook | This repo |
|---|---|---|
| Wealth, 2025:Q3/Q4, 2026:Q1 | 139.0827 / 139.305421 / 138.860484 | match |
| Income, 2025:Q4-2026:Q2 | 0.600000 / 0.366667 / 0.200000 | match |
| Consumer, 2025:Q3/Q4, 2026:Q1 | 12.1 / 10.1 / 14.4 | match |
| K, 2025:Q3/Q4, 2026:Q1 | 0.194171 / -0.086627 / 0.066567 | match |
| PCA loadings (wealth, income, consumer) | 0.616, 0.749, 0.242 | match |
| Pillar correlations (wealth-income, wealth-consumer, income-consumer) | 0.45, -0.19, 0.31 | match |
| Leave-one-out corr (drop wealth/income/consumer) | 0.874, 0.918, 0.871 | match |
| corr(K, K_pca), corr(K, K_invvar) | 0.959, 0.784 | match |
| Chow test F/p (GFC 2008:Q3, COVID 2020:Q2, 2022:Q3) | 38.94/0.0000, 33.10/0.0000, 9.25/0.0002 | match |

Before all three files existed, this repo's own `dfa-networth-shares-detail.csv`
(percentage shares, not dollar levels) was used to *approximate* the wealth
pillar by multiplying the rounded published percentages by a total net
worth derived from `dfa-generation-levels-detail.csv`. That got within ~2%
(136x vs. 139x) — close enough to confirm the approach was sound, but the
~2% gap (traced to rounding in the published 1-decimal percentages, not a
scope mismatch — the two levels files' totals reconcile to $2M on a $174T
series) is exactly why the actual `dfa-networth-levels-detail.csv` file
mattered, and the approximation isn't used anywhere in this module.

## What's built

- `k_index_builder.py` — `build_wealth_pillar()`, `build_income_pillar()`,
  `build_consumer_pillar()`, and `build_k_index()` (equal-weight `K`, plus
  `K_pca` via direct SVD and `K_invvar`, mirroring the notebook's robustness
  checks). All three pillars are present; `build_consumer_pillar()` would
  return `None` (falling back to a labeled 2-pillar `K`) only if
  `bluebk02n.xls` were ever removed.
- `k_index_analysis.py` — the notebook's remaining robustness section:
  pillar correlations, leave-one-pillar-out, cross-scheme correlation, Chow
  tests for a structural break at 2008:Q3/2020:Q2/2022:Q3, and the headline
  chart + pillars chart. Saves `output/kindex.csv` (the full assembled
  series, same file the notebook itself saves).
- `target_data.py` — loads each of the 9 regression target series from
  `data_cache/<name>.csv` (columns `Date,Value`); computes % change or
  first-difference as appropriate. Raises a clear per-series error if the
  file is missing rather than attempting (and failing) a live pull.
- `regressions.py` — `run_k_regression(target, k, n_lags=4)`: OLS of the
  transformed target on a constant, current `K`, and `n_lags` lagged values
  of `K`, with HAC (Newey-West) standard errors.
- `run_k_regressions.py` — entry point; runs both regression families,
  skipping (not fabricating) any target whose CSV isn't present yet.

## What's needed to get real numbers

**The K-Index itself needs nothing further** — it's complete and validated.

**For the regressions**, drop CSVs (columns `Date,Value`, a
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
python k_index_analysis.py      # full robustness section + charts + kindex.csv
python run_k_regressions.py     # the actual advisor-requested regressions
```
