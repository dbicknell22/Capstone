"""Chart for k_specs_luxury_discount_test.py's results -- reads the summary
CSV already on disk, doesn't re-run anything."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
GREEN = "#2E7D32"
CRIMSON = "#C8102E"

df = pd.read_csv("output/k_specs_luxury_discount_summary.csv")

SPECS = ["Level of K", "Difference of K", "Direction of K"]
labels = ["Contemp.", "n=1", "n=2", "n=3", "n=4"]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.3), sharey=True)

for ax, spec in zip(axes, SPECS):
    sub = df[df["spec"] == spec]
    contemp_p = sub[sub["test"] == "contemporaneous"]["p_value"].iloc[0]
    lag_ps = [sub[(sub["test"] == "lagged_joint_F") & (sub["n_lags"] == n)]["p_value"].iloc[0] for n in [1, 2, 3, 4]]
    values = [contemp_p] + lag_ps

    # Contemporaneous-only hits (single term, no lag-stacking) -> gold, the
    # genuine result. Any lagged pattern that's null at n=1 and only clears
    # 5% once more lag terms are added -> hatched crimson: the same fragile,
    # spec-mined signature already rejected for USD/JPY and for Level of K
    # here, not a second confirmed finding.
    if spec == "Level of K":
        colors = [SLATE, SLATE, SLATE, CRIMSON, CRIMSON]
        hatches = ["", "", "", "//", "//"]
    elif spec == "Difference of K":
        colors = [GOLD, SLATE, CRIMSON, CRIMSON, CRIMSON]
        hatches = ["", "", "//", "//", "//"]
    else:  # Direction of K -- contemporaneous hit, lags cleanly null throughout
        colors = [GOLD, SLATE, SLATE, SLATE, SLATE]
        hatches = ["" for _ in values]

    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
        bar.set_edgecolor("white")
    for b, v in zip(bars, values):
        label_y = v + 0.025 if v > 0.09 else 0.115
        ax.text(b.get_x() + b.get_width() / 2, label_y, f"{v:.3f}", ha="center", fontsize=8.5)

    ax.axhline(0.05, color=CRIMSON, lw=1.1, ls="--")
    ax.set_ylim(0, 1.0)
    ax.set_title(spec, fontsize=12.5, fontweight="bold", color=NAVY, pad=10)
    ax.grid(axis="y", alpha=0.15)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("p-value")
axes[2].text(4.35, 0.065, "5%", color=CRIMSON, fontsize=9.5, va="center", ha="left")

fig.suptitle("Level, Difference, and Direction of K vs. the luxury/discount consumer basket",
             fontsize=14.5, fontweight="bold", color=NAVY, y=0.985)
fig.text(0.5, 0.045,
         "N=52 quarters (2013-2025). Gold = genuine, outlier-robust contemporaneous hit (same-quarter,\n"
         "not a leading/tradeable signal). Hatched crimson = clears 5% only once multiple lag terms are\n"
         "jointly added -- the same fragile, spec-mined signature already rejected for USD/JPY.",
         ha="center", fontsize=10, style="italic", color=SLATE)
plt.tight_layout(rect=[0, 0.135, 1, 0.92])
plt.savefig("output/k_specs_luxury_discount_chart.png", dpi=150, facecolor="white")
print("saved output/k_specs_luxury_discount_chart.png")
