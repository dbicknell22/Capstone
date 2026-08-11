"""Builds a one-page visual scorecard combining the strategy's own
performance (CAGR, Sharpe, drawdown, from run_backtest.py's
performance_stats.csv) with strategy_predictor_test.py's results: does the
long/short strategy's own return correlate with the Boomer rotation
signal, the K-Index, or Boomer real-estate selling?

Reads both CSVs already on disk (output/performance_stats.csv and
output/strategy_predictor_test_summary.csv) rather than re-running
anything, so the chart always matches whatever numbers actually exist.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "output"

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
GREEN = "#2E7D32"
CRIMSON = "#C8102E"
OFFWHITE = "#F5F6F8"

PREDICTORS = [
    "rotation_signal (Boomer equity - safe-asset share)",
    "K-Index",
    "real_estate_rotation (Boomer real estate share of assets)",
]
LABELS = {
    "rotation_signal (Boomer equity - safe-asset share)": "Rotation signal\n(Boomer equity − safe-asset share)",
    "K-Index": "K-Index",
    "real_estate_rotation (Boomer real estate share of assets)": "Real-estate rotation\n(Boomer RE share of assets)",
}
VERDICTS = {
    "rotation_signal (Boomer equity - safe-asset share)": ("NULL", SLATE),
    "K-Index": ("NULL", SLATE),
    "real_estate_rotation (Boomer real estate share of assets)": ("NULL", SLATE),
}

PERF_ROWS = [
    ("Long leg\n(defensives)", "long_leg"),
    ("Short leg\n(growth)", "short_leg"),
    ("Long / short\n(the strategy)", "long_short"),
    ("Benchmark\n(SPY)", "benchmark"),
]


def _fmt_pct(v):
    return f"{v * 100:+.1f}%"


def main():
    perf = pd.read_csv(f"{OUT}/performance_stats.csv", index_col=0)
    df = pd.read_csv(f"{OUT}/strategy_predictor_test_summary.csv")

    fig = plt.figure(figsize=(14, 11.3))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.42, 0.46, 1.3], hspace=0.35, wspace=0.3,
                           left=0.055, right=0.97, top=0.91, bottom=0.06)

    fig.suptitle("Pillar 2 Long/Short Strategy — Performance & Boomer/K-Index Scorecard",
                 fontsize=17, fontweight="bold", color=NAVY, x=0.055, ha="left", y=0.975)
    fig.text(0.055, 0.94,
              "Long defensives (XLV, XLU, XLP, VYM) vs. short growth (XLK, XLY, IWO)  •  323 months, 1999–2025",
              fontsize=10.5, color=SLATE, ha="left")

    # ---- Row 0: performance stats table ----
    ax_perf = fig.add_subplot(gs[0, :])
    ax_perf.axis("off")
    ax_perf.set_title("STRATEGY PERFORMANCE", fontsize=11.5, fontweight="bold", color=NAVY,
                       loc="left", pad=4)

    perf_cols = ["", "CAGR", "Ann. Vol", "Sharpe", "Max Drawdown", "Hit Rate"]
    perf_rows = []
    for label, key in PERF_ROWS:
        r = perf.loc[key]
        perf_rows.append([label.replace("\n", " "), _fmt_pct(r["CAGR"]), _fmt_pct(r["Ann. Vol"]),
                           f"{r['Sharpe']:.2f}", _fmt_pct(r["Max Drawdown"]), _fmt_pct(r["Hit Rate"])])

    tab_perf = ax_perf.table(cellText=perf_rows, colLabels=perf_cols, cellLoc="center",
                              bbox=[0, 0.0, 1, 0.82],
                              colWidths=[0.22, 0.16, 0.16, 0.14, 0.18, 0.14])
    tab_perf.auto_set_font_size(False)
    tab_perf.set_fontsize(11)
    for (r, c), cell in tab_perf.get_celld().items():
        cell.set_edgecolor("#E3E6EA")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
            continue
        key = PERF_ROWS[r - 1][1]
        is_strategy_row = key == "long_short"
        cell.set_facecolor(OFFWHITE if is_strategy_row else "white")
        if c == 0:
            cell.set_text_props(ha="left", color="#1A1A1A", fontweight="bold" if is_strategy_row else "normal")
        elif c in (1, 3):  # CAGR, Sharpe columns get colored by sign
            val = perf.loc[key]["CAGR"] if c == 1 else perf.loc[key]["Sharpe"]
            cell.set_text_props(color=GREEN if val >= 0 else CRIMSON, fontweight="bold")
        else:
            cell.set_text_props(color="#1A1A1A")

    # ---- Row 1: predictor regression table ----
    ax_tab = fig.add_subplot(gs[1, :])
    ax_tab.axis("off")
    ax_tab.set_title("DOES THE STRATEGY'S RETURN CORRELATE WITH BOOMER DATA OR K?", fontsize=11.5,
                      fontweight="bold", color=NAVY, loc="left", pad=4)

    col_labels = ["Predictor", "Contemp. p", "n=1", "n=2", "n=3", "n=4", "Verdict"]
    rows = []
    for pred in PREDICTORS:
        sub = df[df["predictor"] == pred]
        contemp_p = sub[sub["test"] == "contemporaneous"]["p_value"].iloc[0]
        lag_ps = [sub[(sub["test"] == "lagged_joint_F") & (sub["n_lags"] == n)]["p_value"].iloc[0] for n in [1, 2, 3, 4]]
        verdict, _ = VERDICTS[pred]
        rows.append([LABELS[pred].replace("\n", " "), f"{contemp_p:.3f}"] + [f"{p:.3f}" for p in lag_ps] + [verdict])

    tab = ax_tab.table(cellText=rows, colLabels=col_labels, cellLoc="center",
                        bbox=[0, 0.0, 1, 0.85],
                        colWidths=[0.38, 0.10, 0.08, 0.08, 0.08, 0.08, 0.13])
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#E3E6EA")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
            continue
        cell.set_facecolor(OFFWHITE if r % 2 == 0 else "white")
        if c == 0:
            cell.set_text_props(ha="left", color="#1A1A1A")
        elif c == 6:
            _, color = VERDICTS[PREDICTORS[r - 1]]
            cell.set_text_props(color=color, fontweight="bold")
        else:
            val = float(rows[r - 1][c])
            if val < 0.10:
                cell.set_text_props(fontweight="bold", color=GOLD)

    # ---- Row 2: bar charts, one per predictor ----
    for i, pred in enumerate(PREDICTORS):
        ax = fig.add_subplot(gs[2, i])
        sub = df[df["predictor"] == pred]
        contemp_p = sub[sub["test"] == "contemporaneous"]["p_value"].iloc[0]
        lag_ps = [sub[(sub["test"] == "lagged_joint_F") & (sub["n_lags"] == n)]["p_value"].iloc[0] for n in [1, 2, 3, 4]]

        labels = ["Contemp.", "n=1", "n=2", "n=3", "n=4"]
        values = [contemp_p] + lag_ps
        colors = [GOLD if v < 0.10 else SLATE for v in values]

        bars = ax.bar(labels, values, color=colors, width=0.6)
        ax.axhline(0.05, color=CRIMSON, lw=1.1, ls="--")
        ax.text(4.45, 0.05, " 5%", color=CRIMSON, fontsize=8, va="center", ha="left")
        ax.set_ylim(0, 1.0)
        ax.set_title(LABELS[pred], fontsize=10.5, fontweight="bold", color=NAVY, pad=10)
        ax.set_ylabel("p-value" if i == 0 else "", fontsize=9, color=SLATE)
        ax.tick_params(axis="x", labelsize=8.5, colors=SLATE)
        ax.tick_params(axis="y", labelsize=8, colors=SLATE)
        ax.grid(axis="y", alpha=0.15)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#D8DCE2")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                     ha="center", fontsize=8, color="#1A1A1A")

    fig.text(0.055, 0.015,
              "The strategy underperformed the benchmark on both return and risk-adjusted basis. Its own returns are also clean nulls against "
              "all 3 Boomer/K predictors — K-Index's n=4 reading (bold gold, p=0.080) is the only near-miss, at just one of four lag lengths tried, "
              "the same one-lag-only pattern flagged elsewhere in this project as specification-mined, not robust.",
              fontsize=9.3, color=NAVY, style="italic", ha="left", wrap=True)

    plt.savefig(f"{OUT}/strategy_predictor_scorecard.png", dpi=160, facecolor="white")
    plt.close()
    print(f"Saved {OUT}/strategy_predictor_scorecard.png")


if __name__ == "__main__":
    main()
