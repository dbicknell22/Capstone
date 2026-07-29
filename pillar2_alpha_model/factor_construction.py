"""Builds the Pillar 2 demographic long/short baskets and the resulting
long-short return series.

Thesis (from Pillar 2): Baby Boomers hold ~51% of U.S. household wealth and
close to 70% of public equities against a ~20% population share, and are
entering the distribution phase of the lifecycle (RMDs, retirement spending,
declining risk tolerance). Lifecycle-investing theory (Bodie & Merton's
glide-path work) says that shift shows up as a durable rotation toward
income/low-volatility/defensive exposure and away from higher-duration growth
exposure that skews toward younger, still-accumulating cohorts. LONG_BASKET
proxies the former, SHORT_BASKET the latter, using liquid sector/style ETFs
so the strategy is directly investable (no single-name concentration risk).
"""
import pandas as pd
from data_sources import load_prices

LONG_BASKET = ["XLV", "XLU", "XLP", "VYM"]   # healthcare, utilities, staples, high dividend
SHORT_BASKET = ["XLK", "XLY", "IWO"]          # tech, discretionary, small-cap growth
BENCHMARK = "SPY"


def basket_returns(prices: pd.DataFrame, tickers) -> pd.Series:
    monthly = prices[tickers].resample("ME").last()
    rets = monthly.pct_change().dropna(how="all")
    return rets.mean(axis=1)  # equal-weight, monthly rebalance


def build_long_short(start="1999-01-01", end=None) -> pd.DataFrame:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = sorted(set(LONG_BASKET + SHORT_BASKET + [BENCHMARK]))
    prices = load_prices(tickers, start, end)

    long_ret = basket_returns(prices, LONG_BASKET)
    short_ret = basket_returns(prices, SHORT_BASKET)
    bench_ret = basket_returns(prices, [BENCHMARK])

    out = pd.DataFrame({
        "long_leg": long_ret,
        "short_leg": short_ret,
        "long_short": long_ret - short_ret,   # dollar-neutral: 100% long / 100% short
        "benchmark": bench_ret,
    }).dropna()
    return out
