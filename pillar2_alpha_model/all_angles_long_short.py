"""Every angle this project attacked the long/short strategy from, in one
place: regressions and backtests for Level, Direction, and Difference of
both K and BEDI, against the long/short strategy's own return.

Level: build_k_index_expanding()["K"] / build_bedi_expanding()["BEDI_equal_weight"]
        -- already-existing point-in-time (expanding) indices.
Direction: sign of the quarter-over-quarter change in the same expanding
        index (1 = rising, 0 = falling).
Difference: the quarter-over-quarter change itself (continuous).

All three specs are derived from the same underlying expanding/point-in-
time index for both K and BEDI, so nothing here mixes a look-ahead-biased
version with a safe one. Backtest rule, uniform across all 6: hold the
long/short position in quarter t when the spec (observed at the end of
quarter t-1) was positive; hold cash otherwise.

This does not introduce a new finding -- every regression and backtest in
this file has already been shown null or fragile-and-rejected elsewhere in
this project (strategy_predictor_test.py, bedi_long_short_test.py,
direction_equity_test.py, k_bedi_long_short_backtest.py). This script
exists purely to produce one consistent, complete set of exhibits showing
every angle side by side.

Run: python all_angles_long_short.py
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
from regressions import contemporaneous_and_lagged_test     # noqa: E402

OUT = "output"
QUARTERS_PER_YEAR = 4

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
GREEN = "#2E7D32"
CRIMSON = "#C8102E"
OFFWHITE = "#F5F6F8"
INK = "#1A1A1A"


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    cagr = (1 + r).prod() ** (QUARTERS_PER_YEAR / len(r)) - 1
    vol = r.std() * np.sqrt(QUARTERS_PER_YEAR)
    sharpe = (r.mean() * QUARTERS_PER_YEAR) / vol if vol > 0 else float("nan")
    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return {"CAGR": cagr, "Ann. Vol": vol, "Sharpe": sharpe, "Max Drawdown": max_dd,
            "Hit Rate": (r > 0).mean(), "N": len(r)}


def timed_return(target, spec_series, lag=1):
    df = pd.concat([target, spec_series.rename("spec")], axis=1, sort=True).dropna()
    df["spec_lag"] = df["spec"].shift(lag)
    df = df.dropna()
    on = (df["spec_lag"] > 0).astype(int)
    return (on * df[target.name]).rename("timed_ret"), df


def main():
    target = quarterly_long_short_return()

    # Regressions use the SAME signal construction as every other regression
    # published in this project: full-sample K (matching strategy_predictor_test.py,
    # k_specification_test.py, direction_equity_test.py), expanding BEDI (matching
    # bedi_long_short_test.py -- BEDI's full-sample version is documented as
    # descriptive-only / look-ahead-biased and has never been used in a regression).
    K_reg = build_k_index()[0]["K"]
    bedi_reg = build_bedi_expanding()["BEDI_equal_weight"]
    REG_SPECS = {
        "Level": {"K": K_reg, "BEDI": bedi_reg},
        "Direction": {"K": (K_reg.diff() > 0).astype(float), "BEDI": (bedi_reg.diff() > 0).astype(float)},
        "Difference": {"K": K_reg.diff(), "BEDI": bedi_reg.diff()},
    }

    # Backtests use the point-in-time (expanding) version of BOTH indices --
    # matching k_timed_treasury_backtest.py / bedi_timed_treasury_backtest.py's
    # no-look-ahead standard for anything making a capital-allocation decision.
    K_bt = build_k_index_expanding()["K"]
    bedi_bt = build_bedi_expanding()["BEDI_equal_weight"]
    BT_SPECS = {
        "Level": {"K": K_bt, "BEDI": bedi_bt},
        "Direction": {"K": (K_bt.diff() > 0).astype(float), "BEDI": (bedi_bt.diff() > 0).astype(float)},
        "Difference": {"K": K_bt.diff(), "BEDI": bedi_bt.diff()},
    }

    always_on_stats = perf_stats(target)

    # ---- Regressions: contemporaneous + lag sweep, all 3 specs x 2 indices ----
    reg_rows = []
    for spec_name, idx in REG_SPECS.items():
        for idx_name, series in idx.items():
            contemp, _ = contemporaneous_and_lagged_test(target, series, n_lags=1)
            reg_rows.append({"spec": spec_name, "index": idx_name, "test": "contemporaneous",
                              "n_lags": 0, "p_value": contemp.pvalues["K"]})
            for n in [1, 2, 3, 4]:
                _, lagged = contemporaneous_and_lagged_test(target, series, n_lags=n)
                reg_rows.append({"spec": spec_name, "index": idx_name, "test": "lagged_joint_F",
                                  "n_lags": n, "p_value": lagged.f_pvalue})
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(f"{OUT}/all_angles_regression_summary.csv", index=False)

    # ---- Backtests: all 6 combos ----
    bt_rows = [{"spec": "Always-on", "index": "-", **always_on_stats}]
    bt_series = {}
    for spec_name, idx in BT_SPECS.items():
        for idx_name, series in idx.items():
            ret, _ = timed_return(target, series, lag=1)
            stats = perf_stats(ret)
            bt_rows.append({"spec": spec_name, "index": idx_name, **stats})
            bt_series[(spec_name, idx_name)] = ret
    bt_df = pd.DataFrame(bt_rows)
    bt_df.to_csv(f"{OUT}/all_angles_backtest_stats.csv", index=False)

    with open(f"{OUT}/all_angles_results.txt", "w") as f:
        f.write("Regressions (K/BEDI x Level/Direction/Difference vs. long/short strategy):\n")
        f.write(reg_df.to_string(index=False))
        f.write("\n\nBacktests (hold long/short when spec_lag1 > 0, else cash):\n")
        f.write(bt_df.to_string(index=False))
    print("Saved all_angles_regression_summary.csv, all_angles_backtest_stats.csv, all_angles_results.txt")

    # =====================================================
    # INDIVIDUAL SNAPSHOT CHARTS -- one file per exhibit
    # =====================================================
    def reg_chart(spec_name, filename):
        sub = reg_df[reg_df["spec"] == spec_name]
        fig, ax = plt.subplots(figsize=(8, 5.2))
        x = np.arange(5)
        width = 0.35
        k_ps = sub[sub["index"] == "K"].sort_values("n_lags")["p_value"].values
        bedi_ps = sub[sub["index"] == "BEDI"].sort_values("n_lags")["p_value"].values
        bars_k = ax.bar(x - width / 2, k_ps, width, color=NAVY, label="K-Index")
        bars_bedi = ax.bar(x + width / 2, bedi_ps, width, color=GOLD, label="BEDI")
        for bars, vals in [(bars_k, k_ps), (bars_bedi, bedi_ps)]:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, max(v, 0.012) + 0.015, f"{v:.3f}",
                        ha="center", fontsize=8, color=INK)
        ax.axhline(0.05, color=CRIMSON, lw=1.1, ls="--")
        ax.text(4.4, 0.05, " 5%", color=CRIMSON, fontsize=9, va="center")
        ax.set_xticks(x, ["Contemp.", "n=1", "n=2", "n=3", "n=4"])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("p-value")
        ax.set_title(f"{spec_name} of K / BEDI vs. long/short strategy's own return",
                     fontsize=13, fontweight="bold", color=NAVY, pad=12)
        ax.legend(frameon=False, fontsize=10.5)
        ax.grid(axis="y", alpha=0.15)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{OUT}/{filename}", dpi=150, facecolor="white")
        plt.close()

    def returns_chart(spec_name, filename):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        cum_always = (1 + target).cumprod()
        ax.plot(cum_always.index, cum_always.values, color=SLATE, lw=2, label="Always-on (no timing)")
        for idx_name, color in [("K", NAVY), ("BEDI", GOLD)]:
            ret = bt_series[(spec_name, idx_name)]
            cum = (1 + ret).cumprod()
            ax.plot(cum.index, cum.values, color=color, lw=2.25, label=f"{idx_name}-timed ({spec_name.lower()})")
        ax.axhline(1.0, color="#AAAAAA", lw=0.8, ls=":")
        ax.set_title(f"Long/short strategy: cumulative return,\n{spec_name.lower()}-timed vs. always-on",
                     fontsize=13, fontweight="bold", color=NAVY, pad=12)
        ax.set_ylabel("Growth of $1")
        ax.legend(frameon=False, fontsize=10.5, loc="upper right")
        ax.grid(alpha=0.15)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{OUT}/{filename}", dpi=150, facecolor="white")
        plt.close()

    for spec_name in BT_SPECS:
        reg_chart(spec_name, f"reg_{spec_name.lower()}_chart.png")
        returns_chart(spec_name, f"returns_{spec_name.lower()}_chart.png")

    # ---- Consolidated scorecard table (one image, all 6 + baseline) ----
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    col_labels = ["Specification", "Index", "CAGR", "Ann. Vol", "Sharpe", "Max Drawdown"]
    rows = []
    for r in bt_rows:
        rows.append([r["spec"], r["index"], f"{r['CAGR']*100:+.2f}%", f"{r['Ann. Vol']*100:.1f}%",
                     f"{r['Sharpe']:.2f}", f"{r['Max Drawdown']*100:.1f}%"])
    tab = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center",
                   bbox=[0, 0, 1, 1], colWidths=[0.22, 0.13, 0.16, 0.16, 0.14, 0.19])
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#E3E6EA")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
            continue
        bg = OFFWHITE if r % 2 == 0 else "white"
        cell.set_facecolor(bg)
        row = bt_rows[r - 1]
        if c in (0, 1):
            cell.set_text_props(ha="left" if c == 0 else "center", color=INK,
                                 fontweight="bold" if row["spec"] == "Always-on" else "normal")
        elif c == 4:
            cell.set_text_props(color=GREEN if row["Sharpe"] >= always_on_stats["Sharpe"] else CRIMSON, fontweight="bold")
        else:
            cell.set_text_props(color=INK)
    fig.suptitle("Every angle: long/short strategy timed by K-Index and BEDI", fontsize=14,
                 fontweight="bold", color=NAVY, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{OUT}/all_angles_scorecard_table.png", dpi=150, facecolor="white")
    plt.close()

    print("Saved 6 chart PNGs (reg_*, returns_* x 3 specs) + all_angles_scorecard_table.png")


if __name__ == "__main__":
    main()
