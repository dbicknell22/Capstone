"""Data loaders for the Pillar 2 demographic-tilt model.

Every loader checks a local CSV cache first, then falls back to a live pull
(yfinance for prices, Ken French's data library via pandas-datareader for
factors). In network-restricted environments the live pull will fail with an
instructive error — populate data_cache/ manually in that case (see README).
"""
from pathlib import Path
import pandas as pd

CACHE_DIR = Path(__file__).parent / "data_cache"


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.csv"


def load_prices(tickers, start, end) -> pd.DataFrame:
    """Adjusted close prices for `tickers`. Columns = tickers, index = date."""
    frames = {}
    missing = []
    for t in tickers:
        p = _cache_path(t)
        if p.exists():
            frames[t] = pd.read_csv(p, index_col=0, parse_dates=True)["Adj Close"]
        else:
            missing.append(t)

    if missing:
        try:
            import yfinance as yf
        except ImportError as e:
            raise RuntimeError(
                "yfinance not installed and no cache for: " + ", ".join(missing)
            ) from e

        raw = yf.download(missing, start=start, end=end, auto_adjust=False, progress=False)
        if raw.empty:
            raise RuntimeError(
                "Live price download returned nothing for: " + ", ".join(missing) + ". "
                "Most likely this environment's network policy blocks Yahoo Finance. "
                "Run this script somewhere with normal internet access, or drop a CSV "
                f"(columns: Date, Adj Close) per ticker into {CACHE_DIR}/<TICKER>.csv."
            )

        adj = raw["Adj Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Adj Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            adj.columns = [missing[0]]

        for t in missing:
            series = adj[t]
            frames[t] = series
            series.to_frame("Adj Close").to_csv(_cache_path(t))

    df = pd.DataFrame(frames).sort_index()
    return df.loc[start:end]


def load_ff_factors(start, end, momentum: bool = True) -> pd.DataFrame:
    """Monthly Fama-French 5 factors (+ momentum), in percent, from Ken French's library."""
    cache = _cache_path("ff_factors_monthly")
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True).loc[start:end]

    try:
        import pandas_datareader.data as pdr
    except ImportError as e:
        raise RuntimeError(
            "pandas_datareader not installed and no cached factor file found."
        ) from e

    try:
        out = pdr.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start, end)[0].copy()
        if momentum:
            mom = pdr.DataReader("F-F_Momentum_Factor", "famafrench", start, end)[0]
            mom.columns = ["UMD"]
            out = out.join(mom, how="inner")
    except Exception as e:
        raise RuntimeError(
            "Live Fama-French factor download failed. This environment likely blocks "
            "outbound calls to Dartmouth's data library (mba.tuck.dartmouth.edu). Run "
            f"this on a machine with normal internet access, or drop a CSV at {cache} "
            "with columns [Mkt-RF, SMB, HML, RMW, CMA, RF, UMD] (monthly %, indexed "
            "by month-end date)."
        ) from e

    out.index = out.index.to_timestamp()
    out.to_csv(cache)
    return out
