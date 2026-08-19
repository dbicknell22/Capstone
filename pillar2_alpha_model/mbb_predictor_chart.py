"""Chart for mbb_predictor_test.py's results -- reads the summary CSV
already on disk, doesn't re-run anything."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
CRIMSON = "#C8102E"

df = pd.read_csv("output/mbb_predictor_test_summary.csv")

PREDICTORS = [
    ("K-Index (level)", "K-Index"),
    ("BEDI (expanding, equal-weight)", "BEDI"),
    ("Rotation signal (Boomer equity - safe-asset share)", "Rotation signal"),
    ("Real-estate rotation (Boomer real estate share of assets)", "Real-estate rotation"),
]

fig, axes = plt.subplots(1, 4, figsize=(19, 5), sharey=True)

for ax, (pred, label) in zip(axes, PREDICTORS):
    sub = df[df["predictor"] == pred]
    contemp_p = sub[sub["test"] == "contemporaneous"]["p_value"].iloc[0]
    lag_ps = [sub[(sub["test"] == "lagged_joint_F") & (sub["n_lags"] == n)]["p_value"].iloc[0] for n in [1, 2, 3, 4]]
    labels = ["Contemp.", "n=1", "n=2", "n=3", "n=4"]
    values = [contemp_p] + lag_ps
    colors = [GOLD if v < 0.10 else SLATE for v in values]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.axhline(0.05, color=CRIMSON, lw=1.1, ls="--")
    ax.set_ylim(0, 1.0)
    ax.set_title(label, fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax.grid(axis="y", alpha=0.15)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=8)

    ex_row = df[(df["predictor"] == pred) & (df["test"] == "contemporaneous_ex_outliers")]
    if len(ex_row):
        ex_p = ex_row["p_value"].iloc[0]
        ax.bar(["Ex-outliers\n(n=1)"], [ex_p], color=CRIMSON, width=0.6, alpha=0.85, hatch="//", edgecolor="white")
        ax.text(5.0, ex_p + 0.03, f"{ex_p:.3f}", ha="center", fontsize=8)
        ax.set_xlim(-0.6, 5.6)

axes[0].set_ylabel("p-value")

fig.suptitle("Does K, BEDI, or a Boomer rotation signal predict MBB (mortgage-backed securities) returns?",
             fontsize=14, fontweight="bold", color=NAVY, y=1.03)
fig.text(0.5, -0.03, "N=75 quarters (2007-2025). K's lag-1/2 reading (p=0.049/0.048) fails the outlier check "
                     "(hatched bar, p=0.093) -- everything else is a clean null at every horizon.",
         ha="center", fontsize=10, style="italic", color=SLATE)
plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.savefig("output/mbb_predictor_chart.png", dpi=150, facecolor="white")
print("saved output/mbb_predictor_chart.png")
