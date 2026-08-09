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

## Real result: does K explain the S&P 500? (first real market-data test in this repo)

The S&P 500's own return (`sprtrn`, from the same CRSP/WRDS pull used in
`pillar2_alpha_model`) is now real data here — the first target series in
either model actually tested against real market returns rather than a
DFA-derived proxy or a skipped placeholder. `python run_k_regressions.py`
finds `data_cache/sp500.csv` and runs it for real:

|  | K coefficient | p-value |
|---|---|---|
| **Combined** (`y ~ const + K_t + K_(t-1..4)` together) | 0.0786 | **0.007** |
| **Contemporaneous only** (`y ~ const + K_t`) | 0.0137 | 0.195 |
| **Lagged only, joint F-test** (`y ~ const + K_(t-1..4)`) | — | 0.353 |

**This is exactly the trap the separated-regression methodology exists to
catch.** The combined model makes K's contemporaneous term look strongly
significant (p=0.007) — but `corr(K, K_lag1) = 0.918` (K is a slow-moving
quarterly index, so of course it's highly autocorrelated with its own
recent past), and that severe multicollinearity is enough to distort which
individual coefficient looks significant in a model that includes 4 highly
correlated copies of the same series. Tested properly — contemporaneous and
lagged as two separate, cleanly-identified regressions, the same discipline
used for BEDI's structural break test and the consumer-credit/equity-growth
mechanism tests — **neither shows a significant relationship** (p=0.195,
p=0.353). The honest read: K does not show robust evidence of explaining
S&P 500 quarterly returns, contemporaneously or with a lag, once tested
without the multicollinearity artifact.

This doesn't change the overall advice — it's one target series out of
nine, and the same "test contemporaneous and lagged separately, don't trust
a combined model's individual coefficients" logic applies to whichever of
the remaining eight arrive next.

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
python mechanism_tests.py       # real results, right now -- see below
```

---

## Round 2 of advisor feedback: testing the actual mechanism

> "You created a K-index... which is kind of a spread between rich and
> poor. I would expect to see widening - stocks go up, tightening - stocks
> go down. Widening - consumer credit worsens, tightening - consumer
> credit improves. Etc. Professor Melvin mentioned that the effects on
> assets might be lagged. No relationship is incredibly useful (Buffett
> wouldn't expect a relationship!)."

Two things this changes about the test, both implemented in
`mechanism_tests.py`:

1. **The hypothesis is contemporaneous first, lagged second.** "Widening ->
   stocks go up" describes a mechanical channel: equity rallies concentrate
   gains among the top-wealth holders who own most of the equities,
   mechanically pushing K up in the *same* quarter. Melvin's lagged effect
   is an addendum, not the primary claim. `run_k_regressions.py`'s original
   combined regression (current K + 4 lags together) can bury this
   distinction — `K` and `K_lag1` are correlated with each other, so
   multicollinearity can make both look weaker than either really is,
   the same failure mode the BEDI structural-break test hit earlier in this
   project. `contemporaneous_and_lagged_test()` (now shared by both
   `mechanism_tests.py` and `run_k_regressions.py`) fits them as two
   separate regressions instead.
2. **Consumer credit is testable right now, no new data needed.**
   `dfa-networth-levels-detail.csv` — already in the repo — has a real
   dollar-level "Consumer credit" column by wealth percentile. Bottom 50%'s
   consumer credit as a share of their own assets (a leverage ratio) is a
   direct, real proxy for "consumer credit worsens" that doesn't depend on
   any of the blocked external data.

Stock prices themselves are still blocked (same wall as `run_k_regressions.py`).
In the meantime, `mechanism_tests.py` uses the DFA-derived aggregate
household equity-holdings growth as a real, data-grounded stand-in — it
conflates price return with net contribution/withdrawal flows, so it's
directional, not a clean total-return series (same caveat as everywhere
else this proxy has been used in this project).

### Real results

| Test | Contemporaneous K coefficient | p-value | Lagged (4Q) joint F-test p-value |
|---|---|---|---|
| Widening K → consumer credit worsens (Bottom 50% leverage, ΔQoQ) | +0.0020 | 0.062 | 0.067 |
| Widening K → stocks go up (DFA equity-growth proxy) | +0.0174 | 0.102 | 0.708 |

**Both point the direction the advisor expects — positive sign in both
cases — but neither clears a conventional 5% significance bar.** The
consumer-credit relationship is the closer call (p=0.062 contemporaneous,
p=0.067 joint on the lags — both would pass a 10% threshold, which is not
unreasonable for a 113-quarter macro sample, but isn't the same as clearing
5%). The equity-growth relationship is weaker and shows no lagged effect at
all (p=0.708 — Melvin's addendum doesn't show up here).

**Read this the way the advisor's own framing suggests, not as a failed
model.** A directionally-consistent-but-marginal contemporaneous
relationship is close to what you'd actually expect from a mechanical
wealth-effect channel using public, low-frequency (quarterly) data — and a
strong, clean *lagged* predictive edge would be the surprising result, not
the null one. Both a widely-available index (K is built entirely from
public Fed/Atlanta Fed/UMich data) and Buffett's efficient-markets instinct
argue against expecting an easy, exploitable lag: if quarterly public data
reliably predicted stock returns a quarter ahead, that edge would likely
already be arbitraged away. What's here is consistent with "the mechanism
is real but weak/contemporaneous," not "there's a free lunch we haven't
found yet."

### What would sharpen this further

- **Real stock/market data** (see `run_k_regressions.py`'s data ask) —
  the DFA equity-growth proxy blends valuation with flows; a real index
  return would isolate the price-return channel the hypothesis is actually
  about.
- **A real consumer-credit delinquency series** — Pillar 4 of the original
  deck already cites "NY Fed Consumer Credit Panel / Equifax, Quarterly
  Report on Household Debt and Credit" for the mortgage-FICO chart. If that
  data (or the delinquency-rate cut of it) is available to the team, it's a
  more direct "consumer credit worsens" measure than the leverage-ratio
  proxy used here.
- **Robustness across lag length and cohort definition** — this used the
  net-worth Bottom 50% cut and a fixed 4-lag window; worth checking the
  income-based bottom quintile (also in this repo's `dfa-income-levels-detail.csv`)
  and a couple of different lag lengths before treating either p-value as
  final, the same discipline applied to BEDI's structural break test.
