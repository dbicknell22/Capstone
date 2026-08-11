"""Backtests BEDI-timed and rotation-signal-timed Treasury allocation
strategies: does bedi_treasury_test.py's regression finding (both BEDI and
the isolated rotation signal predict the 10Y Treasury return, lagged one
quarter, robust across lags 1-4 and an outlier check) survive being turned
into an actual strategy with performance stats -- not just a p-value?

Same structure as k_index_model/k_timed_treasury_backtest.py, applied to
two predictors instead of one:

Signal timing: the predictor observed at the end of quarter t decides
quarter t+1's Treasury exposure -- matches the lag-1 specification that's
actually significant (both are null contemporaneously, per
bedi_treasury_test.py).

Both predictors are already point-in-time/expanding z-scores with no new
construction needed: `BEDI_equal_weight` and `z_rotation` (the isolated
rotation signal, already sign-flipped and expanding-z-scored inside
bedi_index.build_bedi_expanding() -- BEDI rises, and z_rotation rises,
when Boomers de-risk).

Rule (binary, same as the K-timed backtest):
  signal_t (point-in-time z-score) > 0  -> long Treasury in quarter t+1
  signal_t <= 0                          -> hold cash (0% return)

Benchmark: buy-and-hold Treasury, same period.

Run: python bedi_timed_treasury_backtest.py
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _pathutil import find_dir_containing
from bedi_index import build_bedi_expanding

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
K_INDEX_DIR = REPO_ROOT / "k_index_model"
if str(K_INDEX_DIR) not in sys.path:
    sys.path.insert(0, str(K_INDEX_DIR))

OUT = "output"
QUARTERS_PER_YEAR = 4
NAVY = "#0B1F3A"
GOLD = "#C8A24D"
CRIMSON = "#C8102E"
SLATE = "#6C757D"


def load_treasury_return() -> pd.Series:
    """Reads k_index_model/data_cache/treasury_10y_total_return.csv
    directly -- same reason as bedi_treasury_test.py's version of this
    function: target_data.py's own path resolution breaks when imported
    cross-directory (both k_index_model and pillar2_alpha_model have their
    own same-named _pathutil.py module, and Python's import cache keeps
    whichever one loaded first)."""
    df = pd.read_csv(K_INDEX_DIR / "data_cache" / "treasury_10y_total_return.csv",
                      parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Value"].resample("QE").last().pct_change().rename("treasury_10y_total_return")


def perf_stats(returns: pd.Series) -> dict:
    r = returns.dropna()
    cagr = (1 + r).prod() ** (QUARTERS_PER_YEAR / len(r)) - 1
    vol = r.std() * np.sqrt(QUARTERS_PER_YEAR)
    sharpe = (r.mean() * QUARTERS_PER_YEAR) / vol if vol > 0 else np.nan
    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return {"CAGR": cagr, "Ann. Vol": vol, "Sharpe": sharpe,
            "Max Drawdown": max_dd, "Hit Rate": (r > 0).mean(), "N Quarters": len(r)}


def build_backtest(signal: pd.Series, treasury_ret: pd.Series) -> pd.DataFrame:
    df = pd.concat([signal.rename("signal_raw"), treasury_ret.rename("treasury_ret")],
                    axis=1, sort=True).dropna()
    df["signal_lag1"] = df["signal_raw"].shift(1)
    df = df.dropna()

    df["signal"] = (df["signal_lag1"] > 0).astype(int)
    df["strategy_ret"] = df["signal"] * df["treasury_ret"]
    df["benchmark_ret"] = df["treasury_ret"]
    return df


def robustness_check(df: pd.DataFrame, lines: list):
    inverted_ret = (1 - df["signal"]) * df["treasury_ret"]
    inv_stats = perf_stats(inverted_ret)
    on = df.loc[df["signal"] == 1, "treasury_ret"]
    off = df.loc[df["signal"] == 0, "treasury_ret"]
    switches = int((df["signal"].diff().abs() == 1).sum())

    lines.append(f"  Inverted signal (long only when signal_lag1<=0): CAGR={inv_stats['CAGR']:.4f}, "
                 f"Sharpe={inv_stats['Sharpe']:.4f}  <- should be much worse than buy-and-hold if signal is real")
    lines.append(f"  Avg Treasury return, signal_lag1>0 quarters (n={len(on)}): {on.mean() * 100:.2f}% "
                 f"(std {on.std() * 100:.2f}%)")
    lines.append(f"  Avg Treasury return, signal_lag1<=0 quarters (n={len(off)}): {off.mean() * 100:.2f}% "
                 f"(std {off.std() * 100:.2f}%)")
    lines.append(f"  Position switches: {switches} over {len(df)} quarters "
                 f"(avg {len(df) / max(switches, 1):.1f} quarters per regime)")
    worst5 = df.sort_values("treasury_ret").head(5)[["treasury_ret", "signal"]]
    avoided = int((worst5["signal"] == 0).sum())
    lines.append(f"  Of the 5 worst Treasury quarters in-sample, the strategy was OUT for {avoided} of them")
    return inv_stats


def _off_spans(df: pd.DataFrame):
    off = df["signal"] == 0
    spans, start = [], None
    for date, is_off in off.items():
        if is_off and start is None:
            start = date
        elif not is_off and start is not None:
            spans.append((start, date))
            start = None
    if start is not None:
        spans.append((start, df.index[-1]))
    return spans


def plot_cumulative(dfs: dict, path: str):
    fig, axes = plt.subplots(1, len(dfs), figsize=(15, 5.5), sharey=True)
    for ax, (name, df) in zip(axes, dfs.items()):
        cum_strategy = (1 + df["strategy_ret"]).cumprod()
        cum_bench = (1 + df["benchmark_ret"]).cumprod()
        ax.plot(cum_bench.index, cum_bench.values, color=SLATE, lw=2, label="Buy & hold Treasury")
        ax.plot(cum_strategy.index, cum_strategy.values, color=NAVY, lw=2.25, label=f"{name}-timed")
        for start, end in _off_spans(df):
            ax.axvspan(start, end, color=GOLD, alpha=0.12, lw=0)
        ax.set_title(f"{name}-timed vs. buy-and-hold", color=NAVY, weight="bold", fontsize=12)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.grid(alpha=0.15)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Cumulative growth of $1")
    fig.suptitle("Boomer-rotation-timed Treasury allocation vs. buy-and-hold (shaded = strategy in cash)",
                 color=NAVY, weight="bold", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    treasury_ret = load_treasury_return()
    bedi = build_bedi_expanding()

    signals = {
        "BEDI": bedi["BEDI_equal_weight"],
        "Rotation signal": bedi["z_rotation"],
    }

    lines = []
    lines.append("BEDI-timed and rotation-signal-timed Treasury allocation backtests")
    lines.append("")

    dfs, all_stats = {}, []
    for name, sig in signals.items():
        df = build_backtest(sig, treasury_ret)
        dfs[name] = df
        stats_strategy = perf_stats(df["strategy_ret"])
        stats_bench = perf_stats(df["benchmark_ret"])

        lines.append(f"=== {name}-timed ===")
        lines.append(f"N = {len(df)} quarters ({df.index.min().date()} -> {df.index.max().date()})")
        lines.append(f"Time in market (quarters signal_lag1 > 0): {df['signal'].mean() * 100:.1f}%")
        header = f"{'':25s}" + "".join(f"{k:>14s}" for k in stats_strategy)
        lines.append(header)
        lines.append(f"{'Strategy (' + name + '-timed)':25s}" + "".join(
            f"{v:14.4f}" if isinstance(v, float) else f"{v:14d}" for v in stats_strategy.values()))
        lines.append(f"{'Benchmark (buy&hold)':25s}" + "".join(
            f"{v:14.4f}" if isinstance(v, float) else f"{v:14d}" for v in stats_bench.values()))
        lines.append("Robustness checks:")
        inv_stats = robustness_check(df, lines)
        lines.append("")

        all_stats.append({"signal": name, "series": "strategy", **stats_strategy})
        all_stats.append({"signal": name, "series": "benchmark", **stats_bench})
        all_stats.append({"signal": name, "series": "inverted_signal", **inv_stats})

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/bedi_timed_treasury_backtest_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(all_stats).to_csv(f"{OUT}/bedi_timed_treasury_backtest_stats.csv", index=False)
    plot_cumulative(dfs, f"{OUT}/bedi_timed_treasury_backtest.png")
    print(f"\nSaved to {OUT}/bedi_timed_treasury_backtest_results.txt, _stats.csv, .png")


if __name__ == "__main__":
    main()
