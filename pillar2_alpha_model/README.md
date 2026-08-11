# Pillar 2 Alpha Model — Demographic Ownership Tilt

Translates Pillar 2's descriptive finding — Baby Boomers hold ~51% of U.S.
household wealth and ~70% of public equities against a ~20% population share,
and are entering the retirement distribution phase — into a systematic,
testable long/short strategy, following the applied-finance-project feedback
to build a model that attempts to produce alpha from that framework.

The repo has two parts, built in two passes:

- **Part 1 — ETF-based tradeable strategy.** A long/short factor construction
  (`factor_construction.py`, `backtest.py`, `run_backtest.py`). **Real
  performance results now exist**, from CRSP/WRDS price data — see
  ["Part 1 results"](#part-1-results--real-etf-returns-via-wrdscrsp). The
  alpha regression against Fama-French factors is still blocked (this
  sandbox's network policy blocks Dartmouth's data library, same as every
  other live-data host tried).
- **Part 2 — Fed DFA fundamentals analysis.** `dfa_signals.py`,
  `predictive_test.py`, `dfa_alpha_analysis.py` — once the repo's own
  `dfa-*.csv` files (the Fed's Distributional Financial Accounts, 1989:Q3 to
  2026:Q1) were added, this became fully computable **with real numbers,
  right now, with no external data at all.** Jump to
  ["Part 2 results"](#part-2-results--what-the-dfa-data-actually-shows) for
  what it actually found — including a specification that looked significant
  at first and did not survive a robustness check, kept in because that's an
  honest part of the result, not a bug to quietly remove.
- **Part 3 — BEDI (Boomer Equity Displacement Index) and the forward-return
  test.** `bedi_index.py`, `bedi_analysis.py`, `bedi_forward_return_test.py`
  — combines Part 2's two signals into one composite index and tests it
  against forward returns of real ETF pairs. The composite and its
  structural-break test are real, computed now (see
  ["Part 3 results"](#part-3-results--bedi-and-the-forward-return-test));
  the forward-return regression against real prices hits the same network
  wall as Part 1.

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

## Part 1 results — real ETF returns, via WRDS/CRSP

**Update: real price data now exists.** Daily total returns (`RET`, dividends
included, splits already adjusted) for all 9 tickers plus `sprtrn` (S&P 500
index return) were pulled from CRSP via WRDS, 1998/2000/2002/2006-2025
depending on each ETF's inception date. Converted to a synthetic total-return
price index per ticker (cumulative product of `1 + daily return`) so it
slots into `load_prices()` with no code changes.

**Full-sample performance (1999-2025, 323 months):**

| Leg | CAGR | Sharpe | Max Drawdown |
|---|---|---|---|
| Long (healthcare/utilities/staples/high-div) | 7.8% | **0.71** | -38% |
| Short (tech/discretionary/small-cap growth) | 8.8% | 0.53 | -58% |
| **Long-short** | **-3.3%** | **-0.15** | -68% |
| Benchmark (SPY) | 8.3% | 0.61 | -51% |

**The headline long-short number is negative — but it's driven almost
entirely by the last ~9 years, not a persistent 27-year effect:**

| Period | Long-short CAGR | Long-short Sharpe |
|---|---|---|
| Pre-2017 (1999-2016) | -1.4% | **-0.02** (essentially flat) |
| 2017-2025 | -6.9% | **-0.42** |

The strategy was roughly breakeven for 18 years and only really lost money
during the mega-cap tech/AI-driven stretch since 2017 — a period when the
S&P 500 itself had an unusually high Sharpe (0.98), a favorable regime for
anything carrying long market exposure. **The long leg alone is a genuinely
solid standalone result** — 0.71 Sharpe over the full 27 years, beating the
benchmark's 0.61 — it's specifically the short-growth leg that's been the
costly side of this trade.

**What's still missing to properly settle this**: Fama-French factors, to
test whether the long-short's negative return reflects picking the wrong
sectors or just carrying negative net market beta (from the short leg's
typically higher beta) through an unusually strong bull run. That's exactly
what `alpha_regression()` tests, and it's the one piece still blocked —
`run_backtest.py` now degrades gracefully (reports the performance stats
above, using RF=0 as an approximation, and clearly skips just the alpha
regression) rather than failing outright when this file is missing.

**Licensing note**: the CRSP/WRDS data itself is not committed to this repo
(`data_cache/*.csv` stays gitignored) per typical WRDS data-use agreements —
only the resulting statistics and charts above are checked in.

## Does the strategy's own return correlate with Boomer data or the K-Index?

The result above tests one thing: is this specific basket profitable,
unconditional on any signal. It says nothing about whether the Boomer
thesis the basket is *based on* actually shows up statistically. That's a
different, more direct question — and nothing in this project tested it
until now. `strategy_predictor_test.py` regresses the strategy's own
quarterly return (the monthly `long_short` series above, compounded to
quarterly) against three real predictors, using the same
contemporaneous/lagged-separated, lag-length-swept methodology as every
other regression in this project:

| Predictor | Contemporaneous p | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|---|
| Rotation signal (Boomer equity share − safe-asset share) | 0.868 | 0.308 | 0.546 | 0.216 | 0.319 |
| K-Index | 0.528 | 0.556 | 0.639 | 0.506 | **0.080** |
| Real-estate rotation (Boomer real estate share of assets) | 0.607 | 0.706 | 0.925 | 0.922 | 0.969 |

**All three are clean nulls.** The rotation signal and real-estate
rotation are null at every horizon, no fragile pattern. K-Index's 4-lag
reading (p=0.080) is the only one that even approaches significance, and
it does so at just one out of four lag lengths tried — the exact "only
significant at one arbitrary lag choice" red flag this project has
flagged elsewhere (USD/JPY, BEDI's structural break test) as a
specification-mined result rather than a real one, not a finding to
report as robust.

**Read plainly: the demographic story behind this basket's ETF selection
does not show up as a statistically detectable relationship in the
basket's own returns**, whether tested against the raw Boomer rotation
signal, the K-Index, or a literal "are they selling their homes" proxy.
This is a different (and arguably more direct) test than the BEDI →
LQD-SPY / K → Treasury results elsewhere in this project — those work on
narrower, more targeted instrument pairs (credit vs. equity, Treasuries),
while this basket is a broader multi-sector bet. The pattern across all of
this project's tests together: K-shape/Boomer signals show real
relationships with specific, narrowly-targeted rate-sensitive instruments
(Treasuries, investment-grade credit spreads) but not with broader,
multi-sector baskets like this one or with general asset prices.

Caveats carried over from elsewhere in this project: K-Index here uses its
only existing form (the full-sample z-score), the same mild look-ahead
caveat every other K regression in this project already carries. The
rotation signal and real-estate share are raw DFA levels, never
normalized over any rolling window, so neither has a look-ahead concern.
Full output: `output/strategy_predictor_test_results.txt` and
`output/strategy_predictor_test_summary.csv`. A one-page visual scorecard
combining this with the performance table above (`strategy_predictor_scorecard.py`)
is saved at `output/strategy_predictor_scorecard.png`.

## Does the Boomer real-estate rotation signal, or K, predict REIT returns?

`reit_basket_quarterly.csv` (added to the repo root, 140 quarters
1990Q1-2024Q4, single `reit_ret` column) is the first real, investable
real-estate price series in this project — everything about Boomer real
estate up to now was DFA dollar levels, never a tradeable return. That
makes it the natural target for `real_estate_rotation()`, the most literal
"are Boomers selling their homes" proxy built in this project, instead of
testing it only against the (non-real-estate) long/short strategy basket.
K-Index is tested against it too — REITs were the one major asset class K
hadn't been tested against yet (stocks, Treasuries, FX, gold are already
done), and real estate is arguably the asset class most directly exposed
to a Boomer-retirement/K-shape story. Note: the file has no
composition/source documentation beyond its column name — treat this as
"a REIT basket return series" pending confirmation of its exact source.

Same methodology as everywhere else in this project — contemporaneous and
lagged tested separately, swept across lags 1-4:

| Test | Real-estate rotation | K-Index |
|---|---|---|
| N | 140 quarters (1990-2024) | 109 quarters (1998-2024) |
| Contemporaneous | coef=0.0003, p=0.859 | coef=0.0236, **p=0.021** |
| Lagged joint F, n=1 | p=0.815 | p=0.109 |
| n=2 | p=0.172 | p=0.231 |
| n=3 | p=0.161 | p=0.447 |
| n=4 | p=0.236 | p=0.574 |

**Real-estate rotation is a clean null, cleanly so** — no near-miss, no
fragile one-lag pattern. Boomers' real estate share of their own assets
does not predict REIT basket returns, at any horizon tested.

**K-Index's contemporaneous reading (p=0.021) looked like a real hit at
first — the opposite pattern from the Treasury result (which is lagged,
not contemporaneous)** — but it doesn't survive the same outlier check
that the BEDI→LQD-SPY result survived. Using a pre-specified, objective
rule (drop quarters where the REIT return itself exceeds 2 standard
deviations — not hand-picked dates), 6 of 109 quarters get dropped: the
2008Q4–2009Q3 financial crisis and the 2020Q1 COVID crash:

| | Full sample | Ex-outliers (6 of 109 dropped) |
|---|---|---|
| Coefficient | 0.0236 | 0.0185 |
| p-value | 0.021 | **0.075** |

Removing under 6% of the sample — the handful of quarters where global
financial crises hit both REITs and the wealth distribution at once —
erases the significance entirely. **Read honestly, this is not a robust
structural relationship; it's two series both reacting to the same
macro shocks (2008–09, 2020) at the same time, not K leading REITs.**
The lagged relationship (the more interesting "does K predict future REIT
returns" question) is a clean null at every horizon regardless (p=0.11–0.57).

This adds a third data point to the pattern already emerging across this
project's asset-price tests: K's one **robust** relationship remains the
lagged Treasury result (survives every lag length, no outlier-sensitivity
issue reported there); REITs join the S&P 500, FX, and gold as targets
where an initially-interesting reading did not hold up once tested
properly. Full output: `output/reit_predictor_test_results.txt` and
`output/reit_predictor_test_summary.csv`.

**Next candidate**, not yet run: the isolated rotation signal
(Boomer equity − safe-asset share) against this same REIT series, for
completeness with the same 3-predictor pattern used against the long/short
strategy above.

## Data & how to run it for real

The model needs two free data sources: Yahoo Finance (via `yfinance`, for
ETF prices — though CRSP/WRDS, as used above, is arguably the better
source) and Ken French's data library at Dartmouth (via `pandas-datareader`,
for the factor returns used in the alpha test — **still the missing piece**).

**This was built and validated in a sandboxed session whose network policy
blocks both of those hosts** (confirmed via direct `curl` — 403 from the
egress proxy, not a code bug). To get the alpha regression too:

```bash
cd pillar2_alpha_model
pip install -r requirements.txt
python run_backtest.py        # needs normal internet access, or ff_factors_monthly.csv in data_cache/
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

## "Are we entering uncharted territory?" — Artisan's generational-precedent question

`generational_precedent_analysis.py` answers this directly, using data
already in this repo (`dfa-generation-levels-detail.csv`) — no new sourcing
needed. For each generation, `wealth_share ÷ household_share` gives a
concentration ratio (1.0 = holding exactly its proportional share of
wealth; above 1.0 = overrepresented).

**Note on "Silent" in this data**: the Fed's category is really "everyone
born before 1946" — a fixed cohort that only shrinks over the sample (no
new members ever join it), not the narrower 1928-1945 academic definition.
That turns out to be useful here: it traces one real cohort's concentration
ratio from late-career through deep retirement and mortality-driven wealth
transfer — the same arc Boomers are now entering, observed a generation
earlier.

**Result**: Baby Boomers' current ratio (1.73x, as of 2026:Q1) already
exceeds the Silent Generation's entire all-time peak (1.66x, reached in
1997) — and Boomers are still climbing, with no plateau yet. The shape of
the two trajectories differs even more than the peak numbers: Silent's
ratio held a stable plateau (1.5-1.66x) for nearly 24 years (1989-2013)
before declining, while Boomers have climbed almost without interruption
for 37 years and just kept going past where Silent's plateau sat. A
generation-agnostic cross-check (the age-70-plus bracket, which avoids any
ambiguity in exactly how "Silent" is defined) shows the same still-rising,
not-yet-plateaued pattern independently.

**Caveat, stated plainly**: Boomers have only partly aged into retirement
(the youngest turn 65 in 2029) — this reading is an early-to-mid-transition
snapshot, not Boomers' completed arc the way Silent's is fully observed.
Fair to say Boomers already exceed Silent's peak; not yet fair to say how
their full trajectory compares, since it hasn't finished.

See `output/generational_precedent_ratio.png` (by generation),
`output/generational_precedent_age70plus.png` (the cross-check), and
`output/generational_precedent_summary.csv` for the full numbers.

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

---

## Part 3: BEDI and the forward-return test

Combines Part 2's two components into a single composite — the **Boomer
Equity Displacement Index (BEDI)** — and runs the actual alpha test: does it
predict forward returns of real, tradeable long/short pairs.

### Building BEDI (`bedi_index.py`, `bedi_analysis.py`)

BEDI rises when Boomers are de-risking (falling rotation spread) **and**
wealth concentration is widening (rising k-shape gap) at the same time — the
thesis being that the combination is more meaningful than either series
alone. Both components are z-scored so they're on a comparable scale before
combining, both equal-weight and via PCA (mirroring Pillar 1's own
equal-weight-vs-PCA check).

Two versions exist, for two different purposes — **this distinction matters
and is easy to get wrong:**

- `build_bedi_full_sample()` — z-scores against the full 1989-2026 mean/std.
  Fine for the descriptive chart. **Do not use for the regression**: a
  full-sample z-score at, say, 2005:Q1 is computed using data through
  2026:Q1 — information that didn't exist yet in 2005. A "predictive" result
  built on that is partly look-ahead bias by construction.
- `build_bedi_expanding()` — z-scores (and, for the PCA composite, even the
  PCA loadings) using only data available up to and including each quarter.
  This is the version `bedi_forward_return_test.py` actually regresses.

PCA is done via a direct SVD on the two-column z-scored matrix rather than
adding scikit-learn as a dependency for one principal component.

### Part 3 results — BEDI and the forward-return test

**The PCA-weighting request surfaced something worth reporting on its own:**
Pillar 1's three indicators (wealth, income, sentiment) correlated 0.87-0.96
with each other — a real, robust common factor, which is why equal-weight
and PCA gave nearly the same index there. BEDI's two components do not do
that: `corr(rotation_spread, k_shape_gap)` over the full 1989-2026 sample is
**+0.064** — essentially zero. Mechanically, that means:

- The full-sample PCA's explained-variance ratio is 0.532 (50% is what two
  *uncorrelated*, equal-variance series would give) — there isn't a
  dominant shared axis for PCA to find.
- `corr(BEDI_equal_weight, BEDI_pca)` is **0.000** in the full-sample
  version and only **0.719** in the point-in-time version (vs. Pillar 1's
  0.96) — equal-weight and PCA are not telling the same story here.

This doesn't mean the index is broken — averaging two weakly-correlated
series is still a valid (if noisier) way to combine them, and the code is
correct (verified: `corr(rotation_spread, k_shape_gap) = 0.064` checks out
directly against the raw series). It means the premise "these two moving
together is more meaningful than either alone" isn't well-supported by 37
years of the actual data — they mostly don't move together. Worth knowing
before leaning on the composite as if it captured one dominant mechanism.

**Structural break test.** Testing for a break at 2019:Q4 in BEDI's linear
time trend, the combined model (level-shift dummy + trend-interaction
together) gives a significant joint F-test (F=27.4, p<0.0001) but both
individual coefficients look insignificant (p=0.50, p=0.95) — a
multicollinearity artifact: `post` and `t_post` correlate **0.998** in this
sample (a trend-interaction term is nearly proportional to the level dummy
alone when the break sits in a narrow late range of the series), not
evidence of "no break." Fitting the level-shift and trend-change terms
**separately** (avoiding the collinearity) gives two cleanly identified,
highly significant results:

- Level shift at 2019:Q4: **-1.01 std dev, t=-7.43, p<0.001**
- Trend-slope change at 2019:Q4: **-0.0076/quarter, t=-7.39, p<0.001**

**Both are the opposite sign from what "the retirement wave accelerated
BEDI" would predict.** The chart (`output/bedi_full_sample.png`) confirms
this isn't a regression artifact: BEDI drops sharply starting 2020, bottoms
around a deep trough near 2022-23, and by 2026:Q1 still sits below its
pre-2020 peak. There is a real, statistically robust structural break at
2019-2020 — it just runs in the direction of a **level drop and trend
deceleration**, not the acceleration the "retirement wave" framing assumed.
(Consistent with the mechanism visible in the individual-component chart
from Part 2: 2020's stimulus-era relief measures temporarily narrowed
wealth concentration and disrupted the smooth equity-share climb, before
both resumed drifting up later.)

**The forward-return regression (`bedi_forward_return_test.py`)** — the
actual test that converts this from descriptive to investable — implements
exactly the requested specification, using the look-ahead-free
`build_bedi_expanding()` version of BEDI:

```
Forward_Return(XLV - XLY) ~ BEDI_lag1 + controls
Forward_Return(LQD - SPY)  ~ BEDI_lag1 + controls
```

**Update: this now has a real result**, using the CRSP/WRDS ETF price data
pulled for Part 1. Fama-French `Mkt-RF` as a control is still unavailable
(same missing file as Part 1), so this ran without controls:

| Target | Horizon | BEDI_lag coefficient | p-value |
|---|---|---|---|
| XLV − XLY | 1Q | 0.0201 | 0.166 (not significant) |
| XLV − XLY | 2Q | 0.0123 | 0.308 (not significant) |
| **LQD − SPY** | **1Q** | **0.0345** | **0.012** |
| LQD − SPY | 2Q | 0.0298 | 0.048 |

**The sector-rotation pair (XLV vs. XLY) shows nothing. The bonds-vs-stocks
pair (LQD vs. SPY) does** — BEDI rising predicts LQD outperforming SPY the
following quarter, both at 1Q and (more weakly) 2Q horizons. This is
directionally consistent with the K-Index model's independent finding that
K predicts 10-year Treasury returns one quarter out (see
`k_index_model/README.md`) — two different composite indices, built from
different source data, both pointing to the same "divergence/de-risking
signals precede a bond-favoring quarter" story.

**Caveat before treating this as confirmed**: the LQD-SPY regression's
residuals are strongly non-normal (skew 1.85, kurtosis 9.1, Jarque-Bera
p≈0) — a sign a few extreme quarters may be doing a lot of work. The most
extreme quarter is 2008:Q3 (the GFC, LQD beat SPY by 37 points the
following quarter). Dropping the 2 most extreme quarters, the result
survives but weakens: coefficient falls from 0.0345 to 0.0233, p rises from
0.012 to 0.025 — still significant, so this isn't purely a single-crisis-
quarter artifact, but the crisis quarter is a real contributor to the
effect's magnitude, not just noise around a stable estimate.

**This is the one regression in the repo that actually earns the word
"alpha" if it holds up** — it's a test against real tradeable returns, it
comes back significant, it survives a basic outlier check, and it
independently corroborates a separate finding from the K-Index model. It
is not yet a finished result: no out-of-sample validation, no risk-factor
controls (Fama-French still missing), and N=93 quarters is a moderate
sample for a claim this specific. Treat it as the most promising lead in
this project, not as a proven strategy.

### Closing the loop: BEDI (and the isolated rotation signal) directly against the 10Y Treasury

BEDI had been tested against LQD-SPY and XLV-XLY, and the isolated
rotation signal against the long/short strategy and REIT returns — but
neither had been tested against the 10Y Treasury directly, the one
instrument K itself has a robust relationship with
(`k_index_model/README.md`). `bedi_treasury_test.py` fills that gap, using
BEDI's point-in-time/expanding version (no look-ahead) and the same
contemporaneous/lagged-separated, lag-swept methodology as everywhere
else:

| | Contemporaneous | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|---|
| BEDI (expanding) | p=0.158 | **p=0.012** | **p=0.021** | **p=0.029** | p=0.060 |
| Rotation signal alone | p=0.362 | **p=0.015** | **p=0.020** | **p=0.015** | **p=0.034** |

**Both clear this project's own bar for "robust"** — significant (or
nearly so) at every lag length tried, 1 through 4, not just one arbitrary
choice, the same signature that made the K→Treasury result trustworthy in
the first place. Both are null contemporaneously, significant only with a
lag — the effect is Boomers' behavior *this* quarter predicting Treasury
returns *next* quarter, not a same-quarter co-movement.

**Outlier check (same objective >2-std rule as elsewhere): survives.**
Dropping the 4 quarters where Treasury returns themselves were most
extreme (2008Q4, 2010Q2, 2011Q3, 2020Q1):

| | Full sample | Ex-outliers (4 of 93 dropped) |
|---|---|---|
| Rotation signal, coef / p | -0.0046 / 0.013 | -0.0041 / 0.016 |
| BEDI, coef / p | 0.0159 / 0.011 | 0.0123 / 0.042 |

Neither loses significance — a meaningfully cleaner result than the
REIT/K-Index contemporaneous reading (`k_index_model/README.md`), which
*did* fail this exact check.

**Economically, the direction makes sense and is internally consistent**:
rotation_signal's coefficient is negative — as Boomers' equity-minus-safe-
asset spread *falls* (they rotate toward safety), Treasury returns rise
next quarter. BEDI's coefficient is positive, which says the same thing
through BEDI's sign-flipped convention (BEDI rises when Boomers de-risk).
Both agree with each other, and both agree with K's independent finding.

**This is now the third independent corroboration of the same story** —
K → Treasury (k_index_model), BEDI → LQD-SPY (above), and now BEDI/rotation
signal → Treasury directly, from three different constructions of
"Boomer/wealth-divergence signal" against two different (but related)
fixed-income instruments. Full output: `output/bedi_treasury_test_results.txt`,
`output/bedi_treasury_test_summary.csv`.

**Ideas for other ways to measure "Boomer rotation into fixed income" that
haven't been built yet**, roughly in order of how much new work they'd take:

1. **The raw Boomer safe-asset share level** (not the equity-minus-safe
   spread) — `generation_asset_shares()` already computes this column
   (`safe_share_pct`), it's just never been pulled out and tested on its
   own. A rising level is the most literal "how much of their portfolio is
   now parked in safety," separate from what's happening to their equity
   side. Trivial to test — no new construction needed.
2. **A "true fixed income" share, excluding cash-like assets** —
   `SAFE_COLS` currently bundles deposits and money-market funds (cash
   equivalents) together with actual bonds (`U.S. government and municipal
   securities`, `Corporate and foreign bonds`) and annuities. Splitting
   out just the two bond columns would isolate literal fixed-income-
   security exposure from cash-parking, which are arguably different
   behaviors (defensive cash buildup vs. actually buying duration).
3. **A real Boomer bond-buying *flow* measure** (QoQ change in the bond
   columns' dollar level, not just their share of a shifting total) —
   closer to "are they actively buying" than a share level, which can
   drift just from other assets changing size. More construction work,
   same data.
4. **Turn this into an actual backtest**, the way `k_timed_treasury_backtest.py`
   did for K — hold Treasury when the rotation signal (or BEDI) crossed
   its threshold last quarter, and see if it holds up as CAGR/Sharpe/
   drawdown the same way K's did.
