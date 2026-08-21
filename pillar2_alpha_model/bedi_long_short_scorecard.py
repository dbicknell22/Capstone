"""Recreates the "Does the strategy's return correlate with Boomer data or
K?" exhibit, scoped to just BEDI (single predictor, single panel) -- reads
the already-saved bedi_long_short_test_summary.csv, doesn't re-run
anything.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
CRIMSON = "#C8102E"
OFFWHITE = "#F5F6F8"
INK = "#1A1A1A"

df = pd.read_csv("output/bedi_long_short_test_summary.csv")

contemp_p = df[df["test"] == "contemporaneous"]["p_value"].iloc[0]
lag_ps = [df[(df["test"] == "lagged_joint_F") & (df["n_lags"] == n)]["p_value"].iloc[0] for n in [1, 2, 3, 4]]
values = [contemp_p] + lag_ps
verdict = "NULL" if min(values) >= 0.05 else "BORDERLINE"

fig = plt.figure(figsize=(9.5, 8.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.1], hspace=0.32, left=0.09, right=0.93, top=0.90, bottom=0.08)

# ---- Title bar ----
fig.text(0.09, 0.955, "DOES THE STRATEGY'S RETURN CORRELATE WITH BEDI?",
          fontsize=15, fontweight="bold", color=NAVY, ha="left", va="top")

# ---- Table ----
ax_tab = fig.add_subplot(gs[0])
ax_tab.axis("off")
col_labels = ["Predictor", "Contemp. p", "n=1", "n=2", "n=3", "n=4", "Verdict"]
cell_text = [["BEDI (expanding, equal-weight)", f"{contemp_p:.3f}", f"{lag_ps[0]:.3f}",
              f"{lag_ps[1]:.3f}", f"{lag_ps[2]:.3f}", f"{lag_ps[3]:.3f}", verdict]]
tab = ax_tab.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center",
                    bbox=[0, 0.15, 1, 0.85], colWidths=[0.33, 0.15, 0.1, 0.1, 0.1, 0.1, 0.14])
tab.auto_set_font_size(False)
tab.set_fontsize(10.5)
for (r, c), cell in tab.get_celld().items():
    cell.set_edgecolor("#E3E6EA")
    if r == 0:
        cell.set_facecolor(NAVY)
        cell.set_text_props(color="white", fontweight="bold")
        continue
    cell.set_facecolor("white")
    if c == 0:
        cell.set_text_props(ha="left", color=INK)
    elif c == 6:
        cell.set_text_props(color=SLATE, fontweight="bold")
    else:
        v = values[c - 1]
        color = GOLD if v < 0.10 else INK
        cell.set_text_props(color=color, fontweight="bold" if v < 0.10 else "normal")

# ---- Bar chart ----
ax = fig.add_subplot(gs[1])
labels = ["Contemp.", "n=1", "n=2", "n=3", "n=4"]
colors = [GOLD if v < 0.10 else SLATE for v in values]
bars = ax.bar(labels, values, color=colors, width=0.55)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10.5, color=INK)

ax.axhline(0.05, color=CRIMSON, lw=1.2, ls="--")
ax.text(4.32, 0.05, "5%", color=CRIMSON, fontsize=10, va="center")
ax.set_ylim(0, 1.0)
ax.set_ylabel("p-value")
ax.set_title("BEDI (expanding, equal-weight)", fontsize=13, fontweight="bold", color=NAVY, pad=12)
ax.grid(axis="y", alpha=0.15)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

plt.savefig("output/bedi_long_short_scorecard.png", dpi=150, facecolor="white")
print("saved output/bedi_long_short_scorecard.png")
