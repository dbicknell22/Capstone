"""Re-renders all_angles_long_short.py's 7 exhibits with the K-Index series
stripped out, leaving only BEDI -- same regressions/backtests already
computed there (regression p-values reused from
output/all_angles_regression_summary.csv; backtest series recomputed from
the same BEDI expanding index, identical methodology).

Run: python bedi_only_all_angles.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bedi_index import build_bedi_expanding
from strategy_predictor_test import quarterly_long_short_return

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
    return (on * df[target.name]).rename("timed_ret")


def main():
    target = quarterly_long_short_return()
    reg_df = pd.read_csv(f"{OUT}/all_angles_regression_summary.csv")
    reg_df = reg_df[reg_df["index"] == "BEDI"]

    bedi_bt = build_bedi_expanding()["BEDI_equal_weight"]
    BT_SPECS = {
        "Level": bedi_bt,
        "Direction": (bedi_bt.diff() > 0).astype(float),
        "Difference": bedi_bt.diff(),
    }

    always_on_stats = perf_stats(target)
    bt_rows = [{"spec": "Always-on", **always_on_stats}]
    bt_series = {}
    for spec_name, series in BT_SPECS.items():
        ret = timed_return(target, series, lag=1)
        stats = perf_stats(ret)
        bt_rows.append({"spec": spec_name, **stats})
        bt_series[spec_name] = ret

    def reg_chart(spec_name, filename):
        sub = reg_df[reg_df["spec"] == spec_name]
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        x = np.arange(5)
        ps = sub.sort_values("n_lags")["p_value"].values
        colors = [GOLD if v < 0.05 else SLATE for v in ps]
        bars = ax.bar(x, ps, color=colors, width=0.5)
        for b, v in zip(bars, ps):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, color=INK)
        ax.axhline(0.05, color=CRIMSON, lw=1.1, ls="--")
        ax.text(4.35, 0.05, " 5%", color=CRIMSON, fontsize=9, va="center")
        ax.set_xticks(x, ["Contemp.", "n=1", "n=2", "n=3", "n=4"])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("p-value")
        ax.set_title(f"{spec_name} of BEDI vs. long/short strategy's own return",
                     fontsize=13, fontweight="bold", color=NAVY, pad=12)
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
        ret = bt_series[spec_name]
        cum = (1 + ret).cumprod()
        ax.plot(cum.index, cum.values, color=GOLD, lw=2.25, label=f"BEDI-timed ({spec_name.lower()})")
        ax.axhline(1.0, color="#AAAAAA", lw=0.8, ls=":")
        ax.set_title(f"Long/short strategy: cumulative return,\n{spec_name.lower()}-timed BEDI vs. always-on",
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
        reg_chart(spec_name, f"reg_{spec_name.lower()}_bedi_only.png")
        returns_chart(spec_name, f"returns_{spec_name.lower()}_bedi_only.png")

    # ---- Scorecard table (Always-on + 3 BEDI specs only) ----
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.axis("off")
    col_labels = ["Specification", "CAGR", "Ann. Vol", "Sharpe", "Max Drawdown"]
    rows = []
    for r in bt_rows:
        rows.append([r["spec"], f"{r['CAGR']*100:+.2f}%", f"{r['Ann. Vol']*100:.1f}%",
                     f"{r['Sharpe']:.2f}", f"{r['Max Drawdown']*100:.1f}%"])
    tab = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center",
                   bbox=[0, 0, 1, 1], colWidths=[0.28, 0.18, 0.18, 0.16, 0.2])
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
        if c == 0:
            cell.set_text_props(ha="left", color=INK, fontweight="bold" if row["spec"] == "Always-on" else "normal")
        elif c == 3:
            cell.set_text_props(color=GREEN if row["Sharpe"] >= always_on_stats["Sharpe"] else CRIMSON, fontweight="bold")
        else:
            cell.set_text_props(color=INK)
    fig.suptitle("BEDI-timed long/short strategy: every angle", fontsize=14,
                 fontweight="bold", color=NAVY, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(f"{OUT}/all_angles_scorecard_bedi_only.png", dpi=150, facecolor="white")
    plt.close()

    print("Saved 6 chart PNGs (reg_*, returns_* x 3 specs, _bedi_only) + all_angles_scorecard_bedi_only.png")


if __name__ == "__main__":
    main()
