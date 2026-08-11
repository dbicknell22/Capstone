"""One-page visual scorecard covering every regression, mechanism test,
strategy-correlation check, and backtest run across this entire project
(k_index_model + pillar2_alpha_model combined) -- not just K's original
9 advisor-requested targets.

All numbers here are transcribed from results already written up and
verified in k_index_model/README.md and pillar2_alpha_model/README.md
(and their underlying output/*.txt, *.csv files) -- this script does not
re-run any regression itself, it only renders what's already been found.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "capstone_scorecard.png"

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
GREEN = "#2E7D32"
CRIMSON = "#C8102E"
OFFWHITE = "#F5F6F8"
INK = "#1A1A1A"

VERDICT_COLOR = {
    "ROBUST": GREEN, "WORKS": GREEN, "WORKS (BEST)": GREEN,
    "BORDERLINE": GOLD,
    "NULL": SLATE,
    "FAILS CHECK": CRIMSON, "LOSES MONEY": CRIMSON,
}

LEFT_ROWS = [
    ("10Y Treasury total return", "Asset price", "ROBUST", "Significant, all 4 lag lengths (p<0.02)"),
    ("Industrial production", "Econ growth", "BORDERLINE", "p=0.057 — close, not significant"),
    ("S&P 500", "Asset price", "NULL", "Combined-model \"hit\" was multicollinearity"),
    ("USD / JPY", "Asset price", "NULL", "Only \"sig.\" at 3–4 lags — fragile pattern"),
    ("USD / EUR", "Asset price", "NULL", "Not significant at any lag"),
    ("USD / GBP", "Asset price", "NULL", "Not significant at any lag"),
    ("Gold", "Asset price", "NULL", "Not significant at any lag"),
    ("Unemployment rate (Δ)", "Econ growth", "NULL", "Null at every lag, 1–4"),
    ("GDP (% chg)", "Econ growth", "NULL", "Combined-model \"hit\" was multicollinearity"),
    ("Real-estate rotation → REIT", "Real estate", "NULL", "Null at every lag, 1–4"),
    ("K-Index → REIT", "Real estate", "FAILS CHECK", "p=0.021 → 0.075 once outliers dropped"),
]

RIGHT_ROWS = [
    ("BEDI → 10Y Treasury", "Fixed income", "ROBUST", "Sig. nearly every lag, survives outlier check"),
    ("Rotation signal → 10Y Treasury", "Fixed income", "ROBUST", "Sig. every lag, survives outlier check"),
    ("BEDI → Credit vs. Equity (LQD−SPY)", "Fixed income", "ROBUST", "p=0.012 (1Q), survives outlier check"),
    ("BEDI → Healthcare vs. Discretionary", "Sector rotation", "NULL", "p=0.166, not significant"),
    ("Rotation signal → consumer credit", "Mechanism", "BORDERLINE", "p=0.062, right direction"),
    ("K-Index → equity holdings growth", "Mechanism", "NULL", "p=0.102, weak"),
    ("Long/short ETF → rotation signal", "Strategy corr.", "NULL", "p=0.868 contemporaneous"),
    ("Long/short ETF → K-Index", "Strategy corr.", "NULL", "One-lag fluke, p=0.080"),
    ("Long/short ETF → real estate rotation", "Strategy corr.", "NULL", "p=0.607 contemporaneous"),
    ("K-timed Treasury allocation", "Backtest", "WORKS", "Sharpe 0.60 vs. 0.48 benchmark"),
    ("BEDI-timed Treasury allocation", "Backtest", "WORKS", "Sharpe 0.58 vs. 0.48 benchmark"),
    ("Rotation-signal-timed Treasury alloc.", "Backtest", "WORKS (BEST)", "Sharpe 0.64 vs. 0.48 benchmark"),
    ("Long/short ETF strategy (Pillar 2)", "Backtest", "LOSES MONEY", "CAGR -3.3% vs. benchmark +8.3%"),
]


def draw_table(ax, rows, title):
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=NAVY, loc="left", pad=8)

    col_labels = ["Test", "Category", "Result", "Note"]
    cell_text = [[r[0], r[1], r[2], r[3]] for r in rows]

    tab = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="left", loc="upper left",
                   bbox=[0, 0, 1, 1], colWidths=[0.40, 0.13, 0.13, 0.34])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.7)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#E3E6EA")
        cell.PAD = 0.02
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold", ha="left" if c == 0 else "center")
            continue
        bg = OFFWHITE if r % 2 == 0 else "white"
        cell.set_facecolor(bg)
        if c == 0:
            cell.set_text_props(ha="left", color=INK, fontsize=8.8)
        elif c == 1:
            cell.set_text_props(ha="left", color=SLATE, fontsize=8.2)
        elif c == 2:
            verdict = rows[r - 1][2]
            cell.set_text_props(ha="center", color=VERDICT_COLOR[verdict], fontweight="bold", fontsize=8.0)
        else:
            cell.set_text_props(ha="left", color="#4A4A4A", fontsize=8.0)


def main():
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(1, 2, wspace=0.06, left=0.03, right=0.985, top=0.88, bottom=0.08)

    fig.suptitle("Complete Scorecard — Every Test Run Across This Project", fontsize=18,
                 fontweight="bold", color=NAVY, x=0.03, ha="left", y=0.965)
    fig.text(0.03, 0.925,
             "K-Index, BEDI, rotation signal, real estate, long/short strategy correlation, and backtested Treasury-timing strategies",
             fontsize=11, color=SLATE, ha="left")

    ax_left = fig.add_subplot(gs[0, 0])
    draw_table(ax_left, LEFT_ROWS, "K-INDEX  ·  ASSET PRICES, ECON GROWTH & REAL ESTATE")

    ax_right = fig.add_subplot(gs[0, 1])
    draw_table(ax_right, RIGHT_ROWS, "BEDI / ROTATION SIGNAL  ·  STRATEGY CORRELATION  ·  BACKTESTS")

    fig.text(0.03, 0.025,
             "4 robust regression hits (all fixed income)  •  3 backtested strategies beat buy-and-hold  •  2 borderline  •  "
             "15 clean/fragile nulls  •  1 failed an outlier check  •  1 built strategy lost money",
             fontsize=10.5, color=NAVY, style="italic", fontweight="bold", ha="left")

    plt.savefig(OUT, dpi=160, facecolor="white")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
