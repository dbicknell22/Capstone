"""Backtests an actual K-timed Treasury allocation strategy: does the
K -> Treasury lagged relationship (run_k_regressions.py: K_lag1 is
significant at every lag length 1-4, the one robust result in this whole
project) survive being turned into a real, investable strategy with
performance stats -- not just a regression coefficient?

Signal timing: K observed at the end of quarter t decides quarter t+1's
Treasury exposure. This matches the specification that's actually
significant -- K_lag1, not K_t itself (the contemporaneous test is null,
p=0.262, per README).

Uses the EXPANDING (point-in-time) version of K
(k_index_builder.build_k_index_expanding), NOT the full-sample z-score
used in every K regression elsewhere in this project. A real backtest
making capital-allocation decisions needs a no-look-ahead signal in a way
a regression coefficient's statistical significance does not -- this is
stricter than the rest of this project's K usage, not laxer.

Rule (binary -- the simplest possible implementation of the signal):
  K_t (point-in-time z-score) > 0  -> long Treasury (IEF) in quarter t+1
  K_t <= 0                          -> hold cash (0% return) in quarter t+1

"0% cash return" is the same RF=0 approximation used elsewhere in this
project (run_backtest.py) when a real risk-free series isn't available.

Benchmark: buy-and-hold Treasury (IEF), same period.

Run: python k_timed_treasury_backtest.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from k_index_builder import build_k_index_expanding
from target_data import load_pct_change

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"

OUT = "output"
QUARTERS_PER_YEAR = 4


def perf_stats(returns: pd.Series) -> dict:
    r = returns.dropna()
    cagr = (1 + r).prod() ** (QUARTERS_PER_YEAR / len(r)) - 1
    vol = r.std() * np.sqrt(QUARTERS_PER_YEAR)
    sharpe = (r.mean() * QUARTERS_PER_YEAR) / vol if vol > 0 else np.nan
    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return {"CAGR": cagr, "Ann. Vol": vol, "Sharpe": sharpe,
            "Max Drawdown": max_dd, "Hit Rate": (r > 0).mean(), "N Quarters": len(r)}


def build_backtest(min_periods: int = 20) -> pd.DataFrame:
    k = build_k_index_expanding(min_periods=min_periods)["K"]
    treasury_ret = load_pct_change("treasury_10y_total_return")

    df = pd.concat([k.rename("K"), treasury_ret.rename("treasury_ret")], axis=1, sort=True).dropna()
    df["K_lag1"] = df["K"].shift(1)
    df = df.dropna()

    df["signal"] = (df["K_lag1"] > 0).astype(int)
    df["strategy_ret"] = df["signal"] * df["treasury_ret"]
    df["benchmark_ret"] = df["treasury_ret"]
    return df


def robustness_check(df: pd.DataFrame, lines: list):
    """Does the signal have real directional information, or is any Sharpe
    gain just a generic 'being in cash sometimes lowers vol' artifact?
    Three checks: (1) the INVERTED signal should do meaningfully worse than
    buy-and-hold if the real signal is doing real work, not the opposite;
    (2) the raw split in average Treasury return between "on" and "off"
    quarters; (3) turnover, since a strategy that flips every quarter isn't
    realistic even before costs are modeled."""
    inverted_ret = (1 - df["signal"]) * df["treasury_ret"]
    inv_stats = perf_stats(inverted_ret)
    on = df.loc[df["signal"] == 1, "treasury_ret"]
    off = df.loc[df["signal"] == 0, "treasury_ret"]
    switches = int((df["signal"].diff().abs() == 1).sum())

    lines.append("Robustness checks:")
    lines.append(f"  Inverted signal (long only when K_lag1<=0): CAGR={inv_stats['CAGR']:.4f}, "
                 f"Sharpe={inv_stats['Sharpe']:.4f}  <- should be much worse than buy-and-hold if signal is real")
    lines.append(f"  Avg Treasury return, K_lag1>0 quarters (n={len(on)}): {on.mean() * 100:.2f}% "
                 f"(std {on.std() * 100:.2f}%)")
    lines.append(f"  Avg Treasury return, K_lag1<=0 quarters (n={len(off)}): {off.mean() * 100:.2f}% "
                 f"(std {off.std() * 100:.2f}%)")
    lines.append(f"  Position switches: {switches} over {len(df)} quarters "
                 f"(avg {len(df) / max(switches, 1):.1f} quarters per regime) -- low turnover")
    worst5 = df.sort_values("treasury_ret").head(5)[["treasury_ret", "signal"]]
    avoided = int((worst5["signal"] == 0).sum())
    lines.append(f"  Of the 5 worst Treasury quarters in-sample, the strategy was OUT for {avoided} of them")
    lines.append("")
    return inv_stats


def plot_cumulative(df: pd.DataFrame, path: str):
    cum_strategy = (1 + df["strategy_ret"]).cumprod()
    cum_bench = (1 + df["benchmark_ret"]).cumprod()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(cum_bench.index, cum_bench.values, color=SLATE, lw=2, label="Buy & hold Treasury (IEF)")
    ax.plot(cum_strategy.index, cum_strategy.values, color=NAVY, lw=2.25, label="K-timed strategy")
    for start, end in _off_spans(df):
        ax.axvspan(start, end, color=GOLD, alpha=0.12, lw=0)
    ax.set_title("K-timed Treasury allocation vs. buy-and-hold (shaded = strategy in cash)",
                 color=NAVY, weight="bold", fontsize=13)
    ax.set_ylabel("Cumulative growth of $1")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(alpha=0.15)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _off_spans(df: pd.DataFrame):
    off = df["signal"] == 0
    spans = []
    start = None
    for i, (date, is_off) in enumerate(off.items()):
        if is_off and start is None:
            start = date
        elif not is_off and start is not None:
            spans.append((start, date))
            start = None
    if start is not None:
        spans.append((start, df.index[-1]))
    return spans


def main():
    df = build_backtest()
    stats_strategy = perf_stats(df["strategy_ret"])
    stats_bench = perf_stats(df["benchmark_ret"])

    lines = []
    lines.append("K-timed Treasury allocation backtest")
    lines.append(f"N = {len(df)} quarters ({df.index.min().date()} -> {df.index.max().date()})")
    lines.append(f"Time in market (quarters K_lag1 > 0): {df['signal'].mean() * 100:.1f}%")
    lines.append("")
    header = f"{'':25s}" + "".join(f"{k:>14s}" for k in stats_strategy)
    lines.append(header)
    lines.append(f"{'Strategy (K-timed)':25s}" + "".join(
        f"{v:14.4f}" if isinstance(v, float) else f"{v:14d}" for v in stats_strategy.values()))
    lines.append(f"{'Benchmark (buy&hold)':25s}" + "".join(
        f"{v:14.4f}" if isinstance(v, float) else f"{v:14d}" for v in stats_bench.values()))
    lines.append("")
    inv_stats = robustness_check(df, lines)

    text = "\n".join(lines)
    print(text)

    pd.DataFrame([{"series": "strategy", **stats_strategy},
                  {"series": "benchmark", **stats_bench},
                  {"series": "inverted_signal", **inv_stats}]).to_csv(
        f"{OUT}/k_timed_treasury_backtest_stats.csv", index=False)
    df.to_csv(f"{OUT}/k_timed_treasury_backtest_returns.csv")
    with open(f"{OUT}/k_timed_treasury_backtest_results.txt", "w") as f:
        f.write(text + "\n")
    plot_cumulative(df, f"{OUT}/k_timed_treasury_backtest.png")
    print(f"\nSaved to {OUT}/k_timed_treasury_backtest_results.txt, _stats.csv, _returns.csv, .png")


if __name__ == "__main__":
    main()
