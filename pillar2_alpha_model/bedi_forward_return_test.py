"""The actual alpha test: does BEDI predict forward returns of real,
tradeable long/short pairs?

    Forward_Return(XLV - XLY) ~ BEDI_lag1 + controls
    Forward_Return(LQD - SPY) ~ BEDI_lag1 + controls

XLV-XLY: healthcare minus discretionary — a sector-level read on the same
rotation-toward-defensives thesis BEDI is built from.
LQD-SPY: investment-grade bonds minus broad equities — the direct
fixed-income-rotation read (Boomers selling equities, buying bonds).

Uses BEDI's point-in-time/expanding version (bedi_index.build_bedi_expanding)
specifically because it has no look-ahead — the full-sample z-scored version
would let a "predictive" regression cheat by using future data to define
BEDI's own scale.

Requires real ETF price history (XLV, XLY, LQD, SPY) via data_sources.py —
same network dependency, and same limitation, as run_backtest.py: this
sandboxed environment cannot reach Yahoo Finance (confirmed via direct curl,
see README), so this fails with an instructive error here rather than a
fabricated result. Run on a machine with normal internet access to get real
numbers.

Run: python bedi_forward_return_test.py
"""
import pandas as pd
import statsmodels.api as sm

from bedi_index import build_bedi_expanding
from data_sources import load_prices, load_ff_factors

PAIRS = {
    "XLV_minus_XLY": ("XLV", "XLY"),
    "LQD_minus_SPY": ("LQD", "SPY"),
}


def quarterly_returns(tickers, start, end) -> pd.DataFrame:
    prices = load_prices(tickers, start, end)
    q = prices.resample("QE").last()
    return q.pct_change()


def build_panel(start="1995-01-01", end=None):
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    bedi = build_bedi_expanding()["BEDI_equal_weight"]

    all_tickers = sorted({t for pair in PAIRS.values() for t in pair})
    rets = quarterly_returns(all_tickers, start, end)

    spreads = pd.DataFrame({
        name: rets[a] - rets[b] for name, (a, b) in PAIRS.items()
    })
    # forward return: the spread realized AFTER the BEDI reading it's tested against
    forward = spreads.shift(-1).add_suffix("_fwd1")
    forward2 = spreads.shift(-2).add_suffix("_fwd2")

    panel = pd.concat([bedi.rename("BEDI"), forward, forward2], axis=1).dropna(how="all")
    return panel


def run_regression(panel: pd.DataFrame, target_col: str, n_lags: int = 1, controls: pd.DataFrame = None):
    df = panel[[target_col, "BEDI"]].copy()
    df["BEDI_lag"] = df["BEDI"].shift(n_lags)
    df = df[[target_col, "BEDI_lag"]].dropna()

    X = df[["BEDI_lag"]]
    if controls is not None:
        X = X.join(controls, how="inner")
        df = df.loc[X.index]
    X = sm.add_constant(X)
    model = sm.OLS(df[target_col], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return model


def main():
    panel = build_panel()
    print(f"Panel built: {len(panel)} quarters "
          f"({panel.index.min().date()} -> {panel.index.max().date()})\n")

    try:
        controls = (load_ff_factors(panel.index.min(), panel.index.max()) / 100.0)[["Mkt-RF"]]
        controls = controls.resample("QE").sum()  # sum of monthly -> quarterly approx
    except RuntimeError as e:
        print(f"[controls unavailable: {e}]\n")
        controls = None

    for pair_name in PAIRS:
        for lag in [1, 2]:
            target = f"{pair_name}_fwd{lag}"
            print(f"=== Forward_Return({pair_name.replace('_minus_', ' - ')}), "
                  f"{lag}Q ahead ~ BEDI_lag{lag} ===")
            model = run_regression(panel, target, n_lags=lag, controls=controls)
            print(model.summary())
            print()


if __name__ == "__main__":
    main()
