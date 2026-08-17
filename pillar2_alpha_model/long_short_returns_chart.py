"""Restyles run_backtest.py's cumulative_returns.png to match this
project's visual language, paired with the performance stats table.
Reads real cached CRSP/WRDS ETF prices directly -- no network needed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from factor_construction import build_long_short

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
GREEN = "#2E7D32"
CRIMSON = "#C8102E"
OFFWHITE = "#F5F6F8"
INK = "#1A1A1A"

START, END = "1999-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")

LABELS = {
    "long_leg": "Long leg (defensives: XLV, XLU, XLP, VYM)",
    "short_leg": "Short leg (growth: XLK, XLY, IWO)",
    "long_short": "Long / short (the strategy)",
    "benchmark": "Benchmark (SPY)",
}
COLORS = {"long_leg": GREEN, "short_leg": GOLD, "long_short": NAVY, "benchmark": SLATE}
STYLE = {"long_leg": "-", "short_leg": "-", "long_short": "-", "benchmark": "--"}


def perf_stats(r, periods_per_year=12):
    r = r.dropna()
    cagr = (1 + r).prod() ** (periods_per_year / len(r)) - 1
    vol = r.std() * (periods_per_year ** 0.5)
    sharpe = (r.mean() * periods_per_year) / vol if vol > 0 else float("nan")
    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return {"CAGR": cagr, "Ann. Vol": vol, "Sharpe": sharpe, "Max Drawdown": max_dd, "Hit Rate": (r > 0).mean()}


def main():
    rets = build_long_short(START, END)
    stats = {leg: perf_stats(rets[leg]) for leg in rets.columns}
    cum = (1 + rets).cumprod()

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.35, top=0.92, bottom=0.06, left=0.07, right=0.97)

    ax = fig.add_subplot(gs[0])
    for leg in ["benchmark", "long_leg", "short_leg", "long_short"]:
        ax.plot(cum.index, cum[leg], color=COLORS[leg], ls=STYLE[leg], lw=2.25, label=LABELS[leg])
        ax.annotate(f"${cum[leg].iloc[-1]:.2f}", xy=(cum.index[-1], cum[leg].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9.5,
                    color=COLORS[leg], fontweight="bold")
    ax.set_title(f"Pillar 2 Long/Short Strategy — Cumulative Growth of $1 ({rets.index.min().date()} → {rets.index.max().date()})",
                 fontsize=14, fontweight="bold", color=NAVY, pad=14)
    ax.set_ylabel("Growth of $1")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.grid(alpha=0.15)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    ax_tab = fig.add_subplot(gs[1])
    ax_tab.axis("off")
    col_labels = ["", "CAGR", "Ann. Vol", "Sharpe", "Max Drawdown", "Hit Rate"]
    rows = []
    for leg in ["long_leg", "short_leg", "long_short", "benchmark"]:
        s = stats[leg]
        rows.append([LABELS[leg], f"{s['CAGR']*100:+.1f}%", f"{s['Ann. Vol']*100:.1f}%",
                     f"{s['Sharpe']:.2f}", f"{s['Max Drawdown']*100:.1f}%", f"{s['Hit Rate']*100:.1f}%"])
    tab = ax_tab.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center",
                        bbox=[0, 0.05, 1, 0.85], colWidths=[0.36, 0.14, 0.14, 0.12, 0.14, 0.1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#E3E6EA")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
            continue
        leg = ["long_leg", "short_leg", "long_short", "benchmark"][r - 1]
        cell.set_facecolor(OFFWHITE if r % 2 == 0 else "white")
        if c == 0:
            cell.set_text_props(ha="left", color=INK, fontweight="bold" if leg == "long_short" else "normal")
        elif c in (1, 3):
            val = stats[leg]["CAGR"] if c == 1 else stats[leg]["Sharpe"]
            cell.set_text_props(color=GREEN if val >= 0 else CRIMSON, fontweight="bold")
        elif c == 4:
            cell.set_text_props(color=CRIMSON if leg in ("long_short",) else INK)
        else:
            cell.set_text_props(color=INK)

    plt.savefig("output/long_short_returns_chart.png", dpi=150, facecolor="white")
    print("Saved output/long_short_returns_chart.png")


if __name__ == "__main__":
    main()
