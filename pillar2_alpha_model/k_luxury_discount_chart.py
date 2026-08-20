"""Chart for k_luxury_discount_test.py's results -- reads the summary CSV
already on disk, doesn't re-run anything."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
CRIMSON = "#C8102E"

df = pd.read_csv("output/k_luxury_discount_test_summary.csv")

contemp_p = df[df["test"] == "contemporaneous"]["p_value"].iloc[0]
lag_ps = [df[(df["test"] == "lagged_joint_F") & (df["n_lags"] == n)]["p_value"].iloc[0] for n in [1, 2, 3, 4]]
labels = ["Contemp.", "n=1", "n=2", "n=3", "n=4"]
values = [contemp_p] + lag_ps

fig, ax = plt.subplots(figsize=(8.5, 5.5))
# Gray = clean null. Hatched crimson = clears 5% numerically but flagged as a
# fragile, spec-mined artifact (not colored gold -- gold elsewhere in this
# project's palette means "real signal," which this explicitly is not).
colors = [SLATE, SLATE, SLATE, CRIMSON, CRIMSON]
hatches = ["", "", "", "//", "//"]
bars = ax.bar(labels, values, color=colors, width=0.55)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)
    bar.set_edgecolor("white")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, max(v, 0.012) + 0.025, f"{v:.3f}", ha="center", fontsize=9.5)

ax.axhline(0.05, color=CRIMSON, lw=1.1, ls="--")
ax.text(4.35, 0.05, " 5%", color=CRIMSON, fontsize=9, va="center")
ax.set_ylim(0, 1.0)
ax.set_ylabel("p-value")
ax.set_title("Does K predict the luxury vs. discount consumer basket's return?",
             fontsize=13.5, fontweight="bold", color=NAVY, pad=14)
ax.grid(axis="y", alpha=0.15)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# Annotate the fragile-pattern read directly on the chart
ax.annotate("null at n=1, n=2 -- only clears 5%\nonce 3-4 lag terms are added:\nsame fragile signature as USD/JPY,\nnot a real relationship",
            xy=(3.5, 0.06), xytext=(1.7, 0.72),
            fontsize=9.3, color=CRIMSON, ha="center",
            arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.1))

fig.text(0.5, -0.02, "N=52 quarters (2013-2025). Lag-1 outlier check (the standard spec): "
                     "p=0.424 full sample -> p=0.161 ex-outliers, already null before and after.",
         ha="center", fontsize=9.5, style="italic", color=SLATE)
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig("output/k_luxury_discount_chart.png", dpi=150, facecolor="white")
print("saved output/k_luxury_discount_chart.png")
