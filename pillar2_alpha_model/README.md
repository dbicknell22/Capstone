# Pillar 2 Alpha Model — Demographic Ownership Tilt

Translates Pillar 2's descriptive finding — Baby Boomers hold ~51% of U.S.
household wealth and ~70% of public equities against a ~20% population share,
and are entering the retirement distribution phase — into a systematic,
testable long/short strategy, following the applied-finance-project feedback
to build a model that attempts to produce alpha from that framework.

## Thesis

Lifecycle-investing theory (Bodie & Merton) predicts that as an investor ages
past peak earning years, optimal portfolios rotate from growth/high-duration
equity exposure toward income-generating, lower-volatility assets — driven by
declining human capital, shrinking risk-bearing horizon, and (mechanically)
Required Minimum Distributions once retirement accounts hit age 73.

Pillar 2 shows that shift is happening in a specific, very large cohort all
at once: Boomers are ~20% of the population but hold the majority of
household wealth and a supermajority of equities. If that rotation is real
and not yet fully priced, it should show up as a **persistent return
differential** between:

- **Long basket** — sectors/styles that structurally benefit from retiree
  income demand and defensive positioning: Healthcare (`XLV`), Utilities
  (`XLU`), Consumer Staples (`XLP`), High Dividend Yield (`VYM`).
- **Short basket** — higher-duration, growth-oriented exposure more
  associated with younger, still-accumulating cohorts' portfolios and
  household-formation-driven consumption: Technology (`XLK`), Consumer
  Discretionary (`XLY`), Small-Cap Growth (`IWO`).

Both legs are liquid, low-cost sector/style ETFs — chosen deliberately over
individual stocks so the strategy is directly investable and avoids
single-name idiosyncratic risk and survivorship bias in the backtest.

## What the model does

1. `factor_construction.py` — builds monthly equal-weight returns for each
   basket and the dollar-neutral long-short spread (100% long / 100% short,
   0% net, 200% gross).
2. `backtest.py` — computes CAGR, annualized vol, Sharpe, max drawdown, hit
   rate, and (the actual alpha test) regresses the long-short series on the
   **Fama-French 5 factors + momentum** with Newey-West (HAC) standard
   errors. The regression intercept is the model's claimed alpha: the return
   left over after controlling for market, size, value, profitability,
   investment, and momentum exposure. If the intercept isn't statistically
   significant, the "alpha" is just repackaged factor exposure.
3. `cross_sectional_score.py` — a separate, current-data sanity check: scores
   ~20 well-known large caps on `z(dividend yield) − z(beta)` and confirms
   the top-scoring names concentrate in the sectors Pillar 2 flags as
   Boomer-heavy (healthcare, staples, utilities, real estate), rather than
   growth/tech.
4. `run_backtest.py` — the real entry point. Ties 1–2 together and writes
   `output/performance_stats.csv`, `output/alpha_regression_summary.txt`, and
   `output/cumulative_returns.png`.

## Data & how to run it for real

The model needs two free data sources: Yahoo Finance (via `yfinance`, for
ETF prices back to 1999) and Ken French's data library at Dartmouth (via
`pandas-datareader`, for the factor returns used in the alpha test).

**This was built and validated in a sandboxed session whose network policy
blocks both of those hosts** (confirmed via direct `curl` — 403 from the
egress proxy, not a code bug). No live backtest results exist in this repo
as a result. To get real numbers:

```bash
cd pillar2_alpha_model
pip install -r requirements.txt
python run_backtest.py        # needs normal internet access
```

Run that on your laptop, Colab, or any machine that isn't network-restricted.
If you only have offline price/factor files (e.g. from WRDS/Bloomberg
exports), drop them into `data_cache/` instead — `data_sources.py` checks the
cache before attempting a live pull:

- `data_cache/<TICKER>.csv` — columns `Date, Adj Close`, one file per ticker
  in `factor_construction.LONG_BASKET + SHORT_BASKET + ["SPY"]`.
- `data_cache/ff_factors_monthly.csv` — columns
  `Mkt-RF, SMB, HML, RMW, CMA, RF, UMD`, monthly %, indexed by month-end date.

## Validating the code without market data: `smoke_test_synthetic.py`

To prove the pipeline (basket math, Sharpe/drawdown, HAC regression) is
implemented correctly even without data access, `smoke_test_synthetic.py`
fabricates 25 years of fake monthly factor and long-short returns with a
**known, injected 2%/year alpha**, then runs the same regression code and
checks it recovers something close to that 2% with a significant t-stat.

```
python smoke_test_synthetic.py
```

Sample output from this session:

```
True injected annual alpha:      2.00%
Recovered annual alpha (OLS):    3.66%
Alpha t-stat:                    2.77
```

**This is a unit test, not a finding.** Every number in it is fabricated —
it says nothing about whether the real XLV/XLU/XLP/VYM vs XLK/XLY/IWO spread
actually earns alpha. Do not cite `output/SYNTHETIC_smoke_test.png` or its
numbers in the deck; it only demonstrates the code runs and the math is
correct. Real results require running `run_backtest.py` with data access.

## Known limitations / what to check once real numbers exist

- **Sector ETFs are a proxy, not direct ownership data.** Pillar 2's ~70%
  equity-ownership figure is Boomers' share of *all* U.S. equities, not
  specifically these sectors. The basket choice is a theoretically-motivated
  proxy (lifecycle theory), not a measured fact — treat the regression's
  alpha significance as the real test of whether the proxy holds up.
- **Real estate is deliberately excluded from the long basket.** Boomers'
  dominant asset is owner-occupied housing, but that doesn't map cleanly to
  publicly-traded REIT returns (different holder base, different drivers).
  Adding a REIT leg (e.g. `VNQ`, or senior-housing-specific `REZ`) is a
  reasonable extension but should be tested as its own leg, not folded in
  blind.
- **RMD seasonality is not modeled here.** A natural follow-on (Signal
  thesis option not chosen this round) is a calendar overlay: forced Q4
  selling of equities held in IRAs by 73+ retirees, with reversal into
  January — testable with the same monthly return data by adding a
  month-of-year dummy.
- **Start date (1999) is set by ETF inception, not by theory.** `VYM`
  (2006) and some others have shorter histories than `XLV/XLU/XLP/XLK/XLY`
  (1998); `build_long_short` will silently shrink the sample to the
  shortest-history ticker's start date via `.dropna()` — check
  `output/performance_stats.csv`'s `N Months` before trusting the Sharpe
  ratio on a short sample.
- **HAC lag choice (`maxlags=3`) is a reasonable default for monthly data,
  not tuned.** Worth a robustness check across a couple of lag lengths
  before treating the alpha t-stat as final.
