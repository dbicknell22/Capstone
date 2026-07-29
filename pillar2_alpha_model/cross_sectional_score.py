"""Snapshot cross-sectional check: does a simple demographic-tilt score
(high dividend yield, low beta) actually concentrate in the sectors Pillar 2
flags as Boomer-heavy (healthcare, staples, utilities, real estate) versus
growth/youth-skewing sectors (tech, discretionary)?

This is a point-in-time validation of the SCORE, not a backtest — it uses
current data only, pulled live (see README for the same network caveat as
the rest of the model).
"""
import pandas as pd

# Hand-tagged sector map: yfinance's `sector` field is inconsistent/missing for
# some tickers, so sector identity here is fixed rather than trusted to the API.
UNIVERSE = {
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
    "PG": "Staples", "KO": "Staples", "PEP": "Staples", "CL": "Staples",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    "O": "Real Estate", "WELL": "Real Estate", "VTR": "Real Estate",
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "META": "Technology",
    "AMZN": "Discretionary", "TSLA": "Discretionary", "NKE": "Discretionary", "SBUX": "Discretionary",
}
DEFENSIVE_SECTORS = {"Healthcare", "Staples", "Utilities", "Real Estate"}


def fetch_snapshot(tickers) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for t in tickers:
        info = yf.Ticker(t).info
        rows.append({
            "ticker": t,
            "sector": UNIVERSE[t],
            "dividend_yield": info.get("dividendYield") or 0.0,
            "beta": info.get("beta"),
        })
    return pd.DataFrame(rows).set_index("ticker")


def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["beta"]).copy()
    z = lambda s: (s - s.mean()) / s.std()
    df["score"] = z(df["dividend_yield"]) - z(df["beta"])
    df["defensive_sector"] = df["sector"].isin(DEFENSIVE_SECTORS)
    return df.sort_values("score", ascending=False)


if __name__ == "__main__":
    snap = score(fetch_snapshot(list(UNIVERSE)))
    print(snap)
    top_q = snap.head(len(snap) // 4)
    print("\nTop-quartile score, share in defensive sectors:", top_q["defensive_sector"].mean())
