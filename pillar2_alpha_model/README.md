# Pillar 2 Alpha Model — Demographic Ownership Tilt

Translates Pillar 2's descriptive finding — Baby Boomers hold ~51% of U.S.
household wealth and ~70% of public equities against a ~20% population share,
and are entering the retirement distribution phase — into a systematic,
testable long/short strategy, following the applied-finance-project feedback
to build a model that attempts to produce alpha from that framework.

The repo has two parts, built in two passes:

- **Part 1 — ETF-based tradeable strategy.** A long/short factor construction
  (`factor_construction.py`, `backtest.py`, `run_backtest.py`) ready to run
  the moment it has market data. No live results exist yet — this sandbox's
  network policy blocks every price/factor data host tried (see below).
- **Part 2 — Fed DFA fundamentals analysis.** `dfa_signals.py`,
  `predictive_test.py`, `dfa_alpha_analysis.py` — once the repo's own
  `dfa-*.csv` files (the Fed's Distributional Financial Accounts, 1989:Q3 to
  2026:Q1) were added, this became fully computable **with real numbers,
  right now, with no external data at all.** Jump to
  ["Part 2 results"](#part-2-results--what-the-dfa-data-actually-shows) for
  what it actually found — including a specification that looked significant
  at first and did not survive a robustness check, kept in because that's an
  honest part of the result, not a bug to quietly remove.

## Part 1: ETF strategy — Thesis

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

## Part 1: what the ETF model does

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

---

## Part 2: Fed DFA fundamentals analysis

The repo root now has six real Fed DFA "detail" files — quarterly, 1989:Q3
through 2026:Q1 (147 quarters), each cutting household-sector balance sheets
(broken into ~20 asset/liability categories) a different way:

| File | Cut |
|---|---|
| `dfa-age-levels-detail.csv` / `-shares-` | age70plus, age55to69, age40to54, ageunder40 |
| `dfa-generation-levels-detail.csv` | Silent, BabyBoom, GenX, Millennial |
| `dfa-income-levels-detail.csv` / `-shares-` | 6 income percentile bands |
| `dfa-networth-shares-detail.csv` | TopPt1, RemainingTop1, Next9, Next40, Bottom50 |

Validated on load: summing "Corporate equities and mutual fund shares" across
the 4 generation buckets reproduces the same total as summing across the 4
age buckets (max abs diff = $2M on a ~$55T series — rounding), confirming
these are consistent cuts of the same underlying totals, not independently
sourced data that might disagree.

Unlike Part 1, **this needs no external network access** — the signals,
the charts, and the regression results below are all real, computed from
these files, reproducible by running:

```bash
cd pillar2_alpha_model
python dfa_alpha_analysis.py
```

### What it builds

- `dfa_signals.py` — the core derived series:
  - `rotation_signal(generation)`: that generation's share of household
    equities minus its share of "safe" assets (deposits, money-market funds,
    government/municipal + corporate/foreign bonds, annuities) — a direct,
    real measure of whether a cohort is net-tilted toward risk or income
    assets, and how that tilt is moving quarter to quarter.
  - `k_shape_intensity()`: Top 1% share of net worth (`TopPt1 +
    RemainingTop1`) minus Bottom 50% share — the wealth-concentration gap,
    computed directly from the DFA net-worth-share cut rather than Pillar 1's
    3-indicator composite z-score.
  - `aggregate_equity_growth()`: QoQ % growth in total household-sector
    equity holdings, summed across all generations — a real, DFA-derived
    stand-in for "the market went up," used because this environment can't
    reach an actual price index. It conflates price return with net
    contributions/withdrawals and new issuance, so treat it as directional,
    not a clean total-return series.
- `predictive_test.py` / `dfa_alpha_analysis.py` — lagged OLS regressions
  (HAC standard errors) testing whether a cohort's rotation signal predicts
  subsequent k-shape or aggregate-equity moves, swept across cohort cut and
  lag length (2 vs. 4 quarters) for robustness.

**A correctness bug caught by looking at the chart, not just the numbers:**
the first version of the safe-asset share summed each component column's
*own* percentage share (deposits %, bonds %, annuities %, …) and added them —
which double-counts, since each is a share of a *different* total. The
resulting series ran past 100%, up to ~290%, visible immediately on plotting
it. Fixed by summing dollar levels across the safe-asset columns first, then
taking one share of that combined total (see the correctness note in
`dfa_signals.generation_asset_shares`). Worth flagging because it silently
changed a downstream result: the pre-fix version showed the BabyBoom
rotation signal strongly (and spuriously) trend-correlated with calendar
time (r = -0.685) and with the k-shape gap in levels (r = -0.813); after the
fix, both collapse to essentially zero (r = 0.039, r = 0.003) — the
"finding" was an artifact of the bug, not a real relationship. This is why
`dfa_alpha_analysis.py`'s first two chart outputs get eyeballed, not just
numerically asserted, before anything downstream is trusted.

### Part 2 results — what the DFA data actually shows

Two charts, both real (`output/dfa_boomer_rotation.png`,
`output/dfa_k_shape_gap.png`):

- **BabyBoom's household-equity share climbed from ~18% (1989) to a plateau
  around 53-56% since roughly 2015** — a precise, data-grounded version of
  the "wealth trend has stuck" observation already in the Pillar 2 slide,
  now dated and quantified from the raw series rather than eyeballed off a
  chart image.
- **The wealth-concentration gap (Top 1% − Bottom 50% net worth share)**
  trended from ~19.5% (1989) to ~29% (2026), with clear cyclical dips around
  2009, 2020, and 2022-23 — a real, direct validation of what Pillar 1's
  composite K-Index was already proxying for.

The predictive-regression sweep (`output/dfa_predictive_test_results.csv`,
32 specification/outcome combinations: 8 cohort cuts × 2 lag lengths × 2
outcomes) is the actual alpha test, and the honest result is a **null**:

- **3 of 32 combinations clear p < 0.05** — in line with the ~1.6 expected
  by chance alone at a 5% threshold if there were no real effect at all.
- The one combination that looked most interesting on first pass — BabyBoom
  generation, 4-quarter lags, joint F p = 0.038 (k-shape) and p = 0.038
  (equity growth) — **does not survive shortening to 2 lags** (p = 0.708 and
  p = 0.716). A real economic relationship should not flip from significant
  to nowhere-close on a specification choice this minor; that pattern is the
  signature of noise found by testing enough specifications, not a genuine
  signal (a textbook multiple-comparisons / "garden of forking paths" case —
  worth keeping in the writeup precisely because it's the kind of result
  that's tempting to report as "the finding" if you stop at the first
  significant p-value instead of checking whether it holds up).
- Level correlations between rotation signals and the k-shape gap are
  unstable in sign across cuts (from -0.68 to +0.83 depending which age
  bucket or generation), which is itself evidence against a stable causal
  relationship — a real effect should point the same direction across
  related cuts.

**Bottom line: this linear, single-cohort rotation signal does not show
robust evidence of predicting either wealth-concentration shifts or
aggregate equity growth at 1-4 quarter horizons.** That is a legitimate
research finding, not a dead end reported as a failure — it means the
"Boomers rotating out of equities predicts X" mechanism, at least in this
simple form, isn't showing up in 36 years of the Fed's own household
balance-sheet data. It does not falsify the broader Pillar 2 thesis (Boomer
wealth/equity concentration is real and well-documented); it means *this
particular linear signal construction* isn't demonstrated to carry
exploitable predictive content on its own.

### Honest next steps, in priority order

1. **Test against real tradeable returns, not another fundamentals series.**
   The actual test of "alpha" is whether these DFA signals predict the Part 1
   ETF long/short book's *forward returns* — that needs the market data this
   sandbox can't reach. This is the highest-value next step.
2. **Out-of-sample / expanding-window validation**, not more in-sample
   F-tests — the robustness sweep here already shows in-sample significance
   is fragile; a signal worth trading should predict *forward*, not fit
   backward.
3. **Regime-conditional specifications** — e.g., does the rotation signal
   matter more around recessions or high-volatility quarters, rather than
   as a constant-coefficient linear relationship across 36 years spanning
   very different monetary and demographic regimes?
4. **Unused columns in the same files** — DC vs. DB pension entitlements
   (a genuine glide-path signal, since DC balances are participant-directed
   and DB are not), mortgage debt paydown pace, consumer credit — none of
   this is touched by the current signal set.
