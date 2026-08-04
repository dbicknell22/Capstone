"""Loads the market/macro series the professor's regressions need, from
data_cache/<name>.csv (columns: Date, Value). This repo cannot fetch any of
these live — Yahoo Finance, FRED, Stooq, Alpha Vantage, exchange-rate APIs,
BLS, and BEA are all blocked by this sandbox's network policy (confirmed
directly, not assumed) — so every series must be supplied as a CSV, same
pattern as the dfa-*.csv and wage-growth-data.xlsx files already added.

Expected cache files (suggested source in parens; any comparable source is
fine as long as the file is Date,Value with a level, not a % change —
% change is computed here so it's calculated the same way for every series):

  sp500.csv                    (S&P 500 index level; or another broad index)
  treasury_10y_total_return.csv (a real total-return series if you have one,
                                  e.g. a Treasury total-return index or ETF
                                  NAV; the 10Y constant-maturity YIELD alone
                                  is not a total return and needs converting
                                  first -- see README)
  usdjpy.csv, usdeur.csv, usdgbp.csv   (USD per unit of foreign currency, or
                                         foreign currency per USD -- just be
                                         consistent; note which in the file)
  gold.csv                     (USD/oz)
  unemployment_rate.csv        (U-3 rate, %, not seasonally-adjusted vs SA
                                 doesn't matter much at quarterly frequency)
  gdp.csv                      (real GDP level, e.g. GDPC1)
  industrial_production.csv    (IP index level, e.g. INDPRO)

All are resampled to quarter-end to align with K's quarterly frequency.
"""
from pathlib import Path
import pandas as pd

from _pathutil import find_dir_containing

CACHE_DIR = find_dir_containing("k_index_builder.py") / "data_cache"

SERIES_FILES = {
    "sp500": "sp500.csv",
    "treasury_10y_total_return": "treasury_10y_total_return.csv",
    "usdjpy": "usdjpy.csv",
    "usdeur": "usdeur.csv",
    "usdgbp": "usdgbp.csv",
    "gold": "gold.csv",
    "unemployment_rate": "unemployment_rate.csv",
    "gdp": "gdp.csv",
    "industrial_production": "industrial_production.csv",
}


def load_level(name: str) -> pd.Series:
    if name not in SERIES_FILES:
        raise ValueError(f"Unknown series '{name}'. Known: {list(SERIES_FILES)}")
    path = CACHE_DIR / SERIES_FILES[name]
    if not path.exists():
        raise RuntimeError(
            f"Missing {path}. This sandbox cannot fetch '{name}' live (every "
            "market/macro data host tried is blocked by network policy) -- "
            f"drop a CSV with columns Date,Value at {path} to proceed. See "
            "target_data.py's module docstring for suggested sources."
        )
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Value"].resample("QE").last()


def load_pct_change(name: str) -> pd.Series:
    return load_level(name).pct_change().rename(name)


def load_diff(name: str) -> pd.Series:
    """First difference rather than % change -- appropriate for a rate like
    unemployment, where "% change in a percent" is a less standard measure
    than the level change in percentage points."""
    return load_level(name).diff().rename(name)
