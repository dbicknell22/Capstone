"""Entry point for the real backtest.

    python run_backtest.py

Requires outbound internet access to Yahoo Finance and Ken French's data
library (or pre-populated CSVs in data_cache/ — see README). Writes
performance_stats.csv, alpha_regression_summary.txt, and
cumulative_returns.png to output/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from factor_construction import build_long_short
from backtest import perf_stats, alpha_regression
from data_sources import load_ff_factors

START, END = "1999-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")
OUT = "output"


def main():
    rets = build_long_short(START, END)
    ff = load_ff_factors(START, END) / 100.0

    stats_df = pd.DataFrame({leg: perf_stats(rets[leg], ff.get("RF")) for leg in rets.columns}).T
    stats_df.to_csv(f"{OUT}/performance_stats.csv")
    print(stats_df, "\n")

    model = alpha_regression(rets["long_short"], START, END)
    with open(f"{OUT}/alpha_regression_summary.txt", "w") as f:
        f.write(str(model.summary()))
    print(model.summary())

    cum = (1 + rets).cumprod()
    cum.plot(figsize=(10, 6), title="Pillar 2 Demographic Tilt — Cumulative Growth of $1")
    plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(f"{OUT}/cumulative_returns.png", dpi=150)
    print(f"\nSaved outputs to {OUT}/")


if __name__ == "__main__":
    main()
