"""Return-side visuals for the two genuine (contemporaneous) hits found in
k_specs_luxury_discount_test.py: Direction of K and Difference of K against
the luxury/discount consumer basket.

This is NOT a backtest -- both relationships are contemporaneous, and the
lagged version of Direction of K (the only kind you could actually trade)
already came back null at every lag (n=1-4, all p>0.4). These charts show
the same-quarter co-movement that IS real, for the descriptive/validating
story ("K shows up in equity prices"), not a tradeable strategy.

Run: python luxury_discount_significant_returns_chart.py
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _pathutil import find_dir_containing
from luxury_discount_construction import quarterly_luxury_discount_return

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
sys.path.insert(0, str(REPO_ROOT / "k_index_model"))
from k_index_builder import build_k_index  # noqa: E402

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
GREEN = "#2E7D32"
CRIMSON = "#C8102E"
OFFWHITE = "#F5F6F8"

target = quarterly_luxury_discount_return()
K = build_k_index()[0]["K"]
df = pd.concat([target, K.rename("K"), K.diff().rename("dK")], axis=1, sort=True).dropna()
df["direction"] = (df["dK"] > 0).astype(int)

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28,
                       left=0.07, right=0.97, top=0.90, bottom=0.08)

# ---- Panel 1 (top, full width): cumulative basket return, shaded by K direction ----
ax1 = fig.add_subplot(gs[0, :])
cum = (1 + df["luxury_discount_qtr_return"]).cumprod()
ax1.plot(cum.index, cum.values, color=NAVY, lw=2.2, zorder=3)

# Shade each quarter's span green (K rose that quarter) or crimson (K fell)
edges = list(df.index)
for i, dt in enumerate(edges):
    start = edges[i - 1] if i > 0 else dt - pd.tseries.frequencies.to_offset("QE")
    color = GREEN if df.loc[dt, "direction"] == 1 else CRIMSON
    ax1.axvspan(start, dt, color=color, alpha=0.10, lw=0)

ax1.set_title("Luxury/discount basket cumulative return, shaded by K's direction that same quarter\n"
              "(green = K rose, red = K fell -- co-movement, not a timing signal)",
              fontsize=12.5, fontweight="bold", color=NAVY, pad=12)
ax1.set_ylabel("Growth of $1")
ax1.grid(alpha=0.15)
for s in ["top", "right"]:
    ax1.spines[s].set_visible(False)

# ---- Panel 2 (bottom-left): avg return by K direction bucket ----
ax2 = fig.add_subplot(gs[1, 0])
grp = df.groupby("direction")["luxury_discount_qtr_return"]
means = grp.mean()
counts = grp.count()
bars = ax2.bar(["K fell\n(n={})".format(counts[0]), "K rose\n(n={})".format(counts[1])],
               [means[0], means[1]], color=[CRIMSON, GREEN], width=0.55)
for b, v in zip(bars, [means[0], means[1]]):
    va = "bottom" if v > 0 else "top"
    offset = 0.003 if v > 0 else -0.003
    ax2.text(b.get_x() + b.get_width() / 2, v + offset,
              f"{v*100:+.1f}%", ha="center", va=va, fontsize=11, fontweight="bold", color=NAVY)
ax2.margins(y=0.15)
ax2.axhline(0, color="#AAAAAA", lw=0.9)
ax2.set_title("Average quarterly return by K's direction\n(Direction of K, contemp. p=0.0006)",
              fontsize=11.5, fontweight="bold", color=NAVY, pad=10)
ax2.set_ylabel("Avg. quarterly return")
ax2.grid(axis="y", alpha=0.15)
for s in ["top", "right"]:
    ax2.spines[s].set_visible(False)

# ---- Panel 3 (bottom-right): scatter of Delta-K vs. quarterly return ----
ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(df["dK"], df["luxury_discount_qtr_return"], color=NAVY, alpha=0.75, s=42, zorder=3)
coef = np.polyfit(df["dK"], df["luxury_discount_qtr_return"], 1)
xs = np.linspace(df["dK"].min(), df["dK"].max(), 100)
ax3.plot(xs, coef[0] * xs + coef[1], color=GOLD, lw=2.2, zorder=2)
ax3.axhline(0, color="#CCCCCC", lw=0.8)
ax3.axvline(0, color="#CCCCCC", lw=0.8)
ax3.set_title("Quarterly change in K vs. basket return\n(Difference of K, contemp. p=0.026)",
              fontsize=11.5, fontweight="bold", color=NAVY, pad=10)
ax3.set_xlabel("ΔK (quarter-over-quarter)")
ax3.set_ylabel("Basket quarterly return")
ax3.grid(alpha=0.15)
for s in ["top", "right"]:
    ax3.spines[s].set_visible(False)

fig.suptitle("Where the significant K results actually show up: same-quarter co-movement, not a trading signal",
             fontsize=15, fontweight="bold", color=NAVY, y=0.975)

plt.savefig("output/luxury_discount_significant_returns_chart.png", dpi=150, facecolor="white")
print("saved output/luxury_discount_significant_returns_chart.png")
