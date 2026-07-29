"""Tests whether the Baby Boomer rotation signal (dfa_signals.rotation_signal)
has real predictive content — using ONLY the provided Fed DFA data, so this
runs and produces genuine numbers without any external market data.

Two outcomes tested, each regressed on 1-4 quarter lags of the rotation
signal's QoQ change, with Newey-West (HAC) standard errors for
autocorrelation:

  1. k_shape_gap_qoq_chg   — does Boomer rotation lead subsequent shifts in
                              wealth concentration (Top 1% vs Bottom 50% net
                              worth share)?
  2. agg_equity_qoq_growth — does Boomer rotation lead subsequent growth in
                              aggregate household equity holdings (a real,
                              DFA-derived market-value proxy — see caveat in
                              dfa_signals.aggregate_equity_growth)?

This measures whether the *fundamental* signal is informative. It is not a
trading backtest: converting "this predicts X" into realized P&L still
requires actual ETF/security prices, which this environment cannot reach
(see README). Treat this module as the evidence for whether pursuing that
next step is worthwhile.
"""
import pandas as pd
import statsmodels.api as sm

from dfa_signals import rotation_signal, k_shape_intensity, aggregate_equity_growth

MAX_LAG = 4


def build_panel(generation: str = "BabyBoom") -> pd.DataFrame:
    rot = rotation_signal(generation)[["rotation_spread_qoq_chg"]]
    k = k_shape_intensity()[["k_shape_gap_qoq_chg"]]
    eq = aggregate_equity_growth()
    panel = rot.join(k, how="inner").join(eq, how="inner").dropna()
    return panel


def lagged_regression(panel: pd.DataFrame, outcome_col: str, predictor_col: str, max_lag: int = MAX_LAG):
    """outcome_t ~ const + predictor_{t-1} + ... + predictor_{t-max_lag}."""
    df = panel[[outcome_col, predictor_col]].copy()
    lag_cols = []
    for lag in range(1, max_lag + 1):
        col = f"{predictor_col}_lag{lag}"
        df[col] = df[predictor_col].shift(lag)
        lag_cols.append(col)
    df = df.dropna()

    X = sm.add_constant(df[lag_cols])
    y = df[outcome_col]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return model, df


def run(generation: str = "BabyBoom"):
    panel = build_panel(generation)
    results = {}
    for outcome in ["k_shape_gap_qoq_chg", "agg_equity_qoq_growth"]:
        model, _ = lagged_regression(panel, outcome, "rotation_spread_qoq_chg")
        results[outcome] = model
    return panel, results


if __name__ == "__main__":
    panel, results = run("BabyBoom")
    print(f"Panel: {len(panel)} quarterly observations, "
          f"{panel.index.min().date()} -> {panel.index.max().date()}\n")
    for outcome, model in results.items():
        print(f"=== Outcome: {outcome} ~ lags of BabyBoom rotation_spread_qoq_chg ===")
        print(model.summary())
        print()
