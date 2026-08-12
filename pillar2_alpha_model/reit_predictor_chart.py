"""Chart for reit_predictor_test.py's results -- reads the summary CSV
already on disk, doesn't re-run anything."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
CRIMSON = "#C8102E"

df = pd.read_csv("output/reit_predictor_test_summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))

for ax, (pred, label) in zip(axes, [
    ("real_estate_rotation (Boomer real estate share of assets)", "Real-estate rotation → REIT"),
    ("K-Index", "K-Index → REIT"),
]):
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
    ax.set_title(label, fontsize=12.5, fontweight="bold", color=NAVY, pad=10)
    ax.set_ylabel("p-value")
    ax.grid(axis="y", alpha=0.15)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=8.5)

ex_row = df[(df["predictor"] == "K-Index") & (df["test"] == "contemporaneous_ex_outliers")]
ex_p = ex_row["p_value"].iloc[0]
axes[1].bar(["Contemp.\n(ex-outliers)"], [ex_p], color=CRIMSON, width=0.6, alpha=0.85,
            hatch="//", edgecolor="white")
axes[1].text(5.0, ex_p + 0.03, f"{ex_p:.3f}", ha="center", fontsize=8.5)
axes[1].set_xlim(-0.6, 5.6)

fig.suptitle("Does Boomer real-estate behavior, or K, predict REIT returns?", fontsize=14, fontweight="bold", color=NAVY, y=0.995)
fig.text(0.5, 0.01, "N=140 (real-estate rotation), N=109 (K-Index). K's contemporaneous p=0.021 rises to p=0.075 once the 6 most extreme outlier quarters are dropped (hatched bar).",
          ha="center", fontsize=9.5, style="italic", color=SLATE)
plt.tight_layout(rect=[0, 0.05, 1, 0.93])
plt.savefig("output/reit_predictor_chart.png", dpi=150, facecolor="white")
print("saved output/reit_predictor_chart.png")
