"""Builds the luxury/discount consumer long-short basket: a K-Index-motivated
alternative to the original lifecycle-rotation long/short in
factor_construction.py.

Thesis: rather than proxying "the K-shaped economy" through an age/lifecycle
rotation (retirees moving from growth to defensive sectors -- the original
basket's logic), this targets K's *consumer* pillar directly -- top-decile
discretionary spending on premium/luxury goods vs. bottom-cohort trade-down
behavior into discount/value retail. Unlike growth/tech (the original
basket's short leg), luxury and discount retailer stock prices are not
inputs into how K itself is computed, so there's no mechanical/reverse-
causation overlap with K's own construction.

All 10 names trade under a single, never-renamed ticker over this window
(RH's two CUSIPs are the same company's 2016 corporate restructuring, not a
different one -- verified via WRDS PERMNO lookup). Sample is bounded to
2013-01-01 onward because BURL (Burlington Stores) IPO'd Oct 2013 -- the
binding constraint once the renamed/recycled tickers (TPR, CPRI, and the
pre-2013 recycled segments of RL/RH/FIVE) are excluded.

Same equal-weight, monthly-rebalance, dollar-neutral (100% long / 100%
short) mechanics as factor_construction.py -- no timing rule, always fully
invested in both legs.
"""
import pandas as pd
from data_sources import load_prices

LONG_BASKET = ["RL", "EL", "RH", "ULTA", "LULU"]      # premium/luxury consumer
SHORT_BASKET = ["DLTR", "ROST", "DG", "FIVE", "BURL"]  # discount/value consumer

START = "2013-01-01"


def basket_returns(prices: pd.DataFrame, tickers) -> pd.Series:
    monthly = prices[tickers].resample("ME").last()
    rets = monthly.pct_change().dropna(how="all")
    return rets.mean(axis=1)  # equal-weight, monthly rebalance


def build_luxury_discount(start=START, end=None) -> pd.DataFrame:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = sorted(set(LONG_BASKET + SHORT_BASKET))
    prices = load_prices(tickers, start, end)

    long_ret = basket_returns(prices, LONG_BASKET)
    short_ret = basket_returns(prices, SHORT_BASKET)

    out = pd.DataFrame({
        "long_leg": long_ret,
        "short_leg": short_ret,
        "long_short": long_ret - short_ret,
    })
    return out.dropna(how="all")


def quarterly_luxury_discount_return(start=START, end=None) -> pd.Series:
    """Compounds the monthly long_short return into a quarterly return, to
    match the quarterly frequency every K/BEDI regression in this project
    uses."""
    ls = build_luxury_discount(start=start, end=end)["long_short"]
    q = ls.resample("QE").apply(lambda s: (1 + s).prod() - 1)
    return q.rename("luxury_discount_qtr_return")


if __name__ == "__main__":
    df = build_luxury_discount()
    df.to_csv("output/luxury_discount_monthly_returns.csv")
    q = quarterly_luxury_discount_return()
    q.to_csv("output/luxury_discount_quarterly_returns.csv")
    print(f"N monthly = {len(df)} ({df.index.min().date()} -> {df.index.max().date()})")
    print(f"N quarterly = {len(q)} ({q.index.min().date()} -> {q.index.max().date()})")
    print("Saved output/luxury_discount_monthly_returns.csv, "
          "output/luxury_discount_quarterly_returns.csv")
