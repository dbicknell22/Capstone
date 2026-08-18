"""Consolidated view for the 'show the lack of equity evidence' section:
regressions of the long/short strategy's own return against K-Index and
BEDI, plus the actual backtested return series if each signal is used to
time the strategy (hold long/short when the signal was positive last
quarter, hold cash otherwise) -- same rule as the K-timed/BEDI-timed
Treasury backtests, applied here to the long/short strategy instead.

This does not introduce a new finding -- strategy_predictor_test.py and
bedi_long_short_test.py already showed both regressions are null at every
lag. This script exists to make that null visually concrete: the
regression numbers, side by side, and what actually happens to the
strategy's return if you time it with either signal anyway.

Run: python k_bedi_long_short_backtest.py
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _pathutil import find_dir_containing
from bedi_index import build_bedi_expanding
from strategy_predictor_test import quarterly_long_short_return

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
K_INDEX_DIR = REPO_ROOT / "k_index_model"
if str(K_INDEX_DIR) not in sys.path:
    sys.path.insert(0, str(K_INDEX_DIR))
from k_index_builder import build_k_index, build_k_index_expanding  # noqa: E402
from regressions import contemporaneous_and_lagged_test    # noqa: E402

OUT = "output"
QUARTERS_PER_YEAR = 4

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
GREEN = "#2E7D32"
CRIMSON = "#C8102E"


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    cagr = (1 + r).prod() ** (QUARTERS_PER_YEAR / len(r)) - 1
    vol = r.std() * np.sqrt(QUARTERS_PER_YEAR)
    sharpe = (r.mean() * QUARTERS_PER_YEAR) / vol if vol > 0 else float("nan")
    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return {"CAGR": cagr, "Ann. Vol": vol, "Sharpe": sharpe, "Max Drawdown": max_dd, "N": len(r)}


def build_timed(target, signal_expanding, lag=1):
    df = pd.concat([target, signal_expanding.rename("signal")], axis=1, sort=True).dropna()
    df["signal_lag"] = df["signal"].shift(lag)
    df = df.dropna()
    df["on"] = (df["signal_lag"] > 0).astype(int)
    df["timed_ret"] = df["on"] * df[target.name]
    df["always_on_ret"] = df[target.name]
    return df


def main():
    target = quarterly_long_short_return()
    K_exp = build_k_index_expanding()["K"]
    bedi = build_bedi_expanding()["BEDI_equal_weight"]

    # ---- Regressions (reads what strategy_predictor_test.py / bedi_long_short_test.py already found) ----
    K_level = build_k_index()[0]["K"]

    reg_lines = []
    reg_rows = []
    for name, sig in [("K-Index", K_level), ("BEDI", bedi)]:
        contemp, _ = contemporaneous_and_lagged_test(target, sig, n_lags=1)
        contemp_p = contemp.pvalues["K"]
        reg_rows.append({"signal": name, "test": "contemporaneous", "n_lags": 0, "p_value": contemp_p})
        reg_lines.append(f"{name} -> long/short strategy: contemporaneous p={contemp_p:.4f}")
        for n in [1, 2, 3, 4]:
            _, lagged = contemporaneous_and_lagged_test(target, sig, n_lags=n)
            reg_rows.append({"signal": name, "test": "lagged_joint_F", "n_lags": n, "p_value": lagged.f_pvalue})
            reg_lines.append(f"  n_lags={n}: p={lagged.f_pvalue:.4f}")

    # ---- Backtests: hold long/short when signal (expanding, lag-1) was positive, else cash ----
    df_k = build_timed(target, K_exp, lag=1)
    df_bedi = build_timed(target, bedi, lag=1)

    stats_always_on = perf_stats(target)
    stats_k_timed = perf_stats(df_k["timed_ret"])
    stats_bedi_timed = perf_stats(df_bedi["timed_ret"])

    lines = ["Regressions: K-Index and BEDI vs. the long/short strategy's own return", ""]
    lines += reg_lines
    lines.append("")
    lines.append("Backtests: hold long/short when signal_lag1 > 0, else cash")
    lines.append(f"Always-on (no timing):  CAGR={stats_always_on['CAGR']*100:+.2f}%  Sharpe={stats_always_on['Sharpe']:.2f}  MaxDD={stats_always_on['Max Drawdown']*100:.1f}%")
    lines.append(f"K-timed:                CAGR={stats_k_timed['CAGR']*100:+.2f}%  Sharpe={stats_k_timed['Sharpe']:.2f}  MaxDD={stats_k_timed['Max Drawdown']*100:.1f}%  (N={stats_k_timed['N']})")
    lines.append(f"BEDI-timed:             CAGR={stats_bedi_timed['CAGR']*100:+.2f}%  Sharpe={stats_bedi_timed['Sharpe']:.2f}  MaxDD={stats_bedi_timed['Max Drawdown']*100:.1f}%  (N={stats_bedi_timed['N']})")

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/k_bedi_long_short_backtest_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(reg_rows).to_csv(f"{OUT}/k_bedi_long_short_regression_summary.csv", index=False)
    pd.DataFrame([{"series": "always_on", **stats_always_on},
                  {"series": "K_timed", **stats_k_timed},
                  {"series": "BEDI_timed", **stats_bedi_timed}]).to_csv(
        f"{OUT}/k_bedi_long_short_backtest_stats.csv", index=False)

    # ---- Chart: regression p-values (left) + cumulative return, both timed vs always-on (right) ----
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.28, left=0.06, right=0.97, top=0.85, bottom=0.13)

    ax0 = fig.add_subplot(gs[0])
    x = np.arange(5)
    width = 0.35
    k_ps = [reg_rows[i]["p_value"] for i in range(5)]
    bedi_ps = [reg_rows[i]["p_value"] for i in range(5, 10)]
    ax0.bar(x - width/2, k_ps, width, color=NAVY, label="K-Index")
    ax0.bar(x + width/2, bedi_ps, width, color=GOLD, label="BEDI")
    ax0.axhline(0.05, color=CRIMSON, lw=1.1, ls="--")
    ax0.text(4.4, 0.05, " 5%", color=CRIMSON, fontsize=8, va="center")
    ax0.set_xticks(x, ["Contemp.", "n=1", "n=2", "n=3", "n=4"])
    ax0.set_ylim(0, 1.0)
    ax0.set_ylabel("p-value")
    ax0.set_title("Regression: does K or BEDI predict\nthe long/short strategy's own return?", fontsize=12, fontweight="bold", color=NAVY)
    ax0.legend(frameon=False, fontsize=9.5)
    ax0.grid(axis="y", alpha=0.15)
    for s in ["top", "right"]:
        ax0.spines[s].set_visible(False)

    ax1 = fig.add_subplot(gs[1])
    cum_always = (1 + target.reindex(df_k.index).fillna(0)).cumprod()
    cum_k = (1 + df_k["timed_ret"]).cumprod()
    cum_bedi = (1 + df_bedi["timed_ret"]).cumprod()
    ax1.plot(cum_always.index, cum_always.values, color=SLATE, lw=2, label="Always-on (no timing)")
    ax1.plot(cum_k.index, cum_k.values, color=NAVY, lw=2.25, label="K-timed")
    ax1.plot(cum_bedi.index, cum_bedi.values, color=GOLD, lw=2.25, label="BEDI-timed")
    ax1.axhline(1.0, color="#AAAAAA", lw=0.8, ls=":")
    ax1.set_title("Long/short strategy: cumulative return,\ntimed vs. always-on", fontsize=12, fontweight="bold", color=NAVY)
    ax1.set_ylabel("Growth of $1")
    ax1.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax1.grid(alpha=0.15)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)

    fig.suptitle("Long/short strategy vs. K-Index and BEDI: no relationship, in regression or in a backtest",
                 fontsize=14, fontweight="bold", color=NAVY, y=0.98)

    plt.savefig(f"{OUT}/k_bedi_long_short_backtest.png", dpi=150, facecolor="white")
    print(f"\nSaved to {OUT}/k_bedi_long_short_backtest_results.txt, _regression_summary.csv, _backtest_stats.csv, .png")


if __name__ == "__main__":
    main()
