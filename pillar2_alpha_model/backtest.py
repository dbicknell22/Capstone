"""Performance stats and the alpha test: does the Pillar 2 long-short series
earn a return not already explained by known risk factors?
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from data_sources import load_ff_factors

MONTHS_PER_YEAR = 12
FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]


def perf_stats(returns: pd.Series, rf: pd.Series = None) -> dict:
    r = returns.dropna()
    rf_aligned = rf.reindex(r.index).fillna(0) if rf is not None else 0
    excess = r - rf_aligned

    cagr = (1 + r).prod() ** (MONTHS_PER_YEAR / len(r)) - 1
    vol = r.std() * np.sqrt(MONTHS_PER_YEAR)
    sharpe = (excess.mean() * MONTHS_PER_YEAR) / vol if vol > 0 else np.nan
    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    return {
        "CAGR": cagr,
        "Ann. Vol": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Hit Rate": (r > 0).mean(),
        "N Months": len(r),
    }


def alpha_regression(long_short: pd.Series, start, end):
    """OLS of the long-short series on Fama-French 5 + momentum, with
    Newey-West (HAC) standard errors to account for return autocorrelation."""
    ff = load_ff_factors(start, end) / 100.0  # FF library reports percent, not decimal
    cols = [c for c in FACTOR_COLS if c in ff.columns]

    df = pd.concat([long_short.rename("ls"), ff[cols]], axis=1).dropna()
    X = sm.add_constant(df[cols])
    model = sm.OLS(df["ls"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return model
