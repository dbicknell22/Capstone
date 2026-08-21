"""Chart for k_timed_luxury_discount_backtest.py: cumulative return of the
K-timed strategy at each lag vs. always-on, plus the inverted lag-3/lag-4
comparison that makes the "not a real signal" point directly in return
terms.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _pathutil import find_dir_containing
from luxury_discount_construction import quarterly_luxury_discount_return

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
sys.path.insert(0, str(REPO_ROOT / "k_index_model"))
from k_index_builder import build_k_index_expanding  # noqa: E402

NAVY = "#0B1F3A"
GOLD = "#C8A24D"
SLATE = "#6C757D"
GREEN = "#2E7D32"
CRIMSON = "#C8102E"

target = quarterly_luxury_discount_return()
K = build_k_index_expanding()["K"]


def run(target, signal, lag, invert=False):
    df = pd.concat([target, signal.rename("signal")], axis=1, sort=True).dropna()
    df["signal_lag"] = df["signal"].shift(lag)
    df = df.dropna()
    on = (df["signal_lag"] < 0) if invert else (df["signal_lag"] > 0)
    return (on.astype(int) * df[target.name]).rename("ret")


fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# ---- Left: all 4 lags vs always-on ----
ax = axes[0]
cum_always = (1 + target).cumprod()
ax.plot(cum_always.index, cum_always.values, color=SLATE, lw=2.4, label="Always-on (no timing)", zorder=5)
lag_colors = {1: NAVY, 2: GOLD, 3: CRIMSON, 4: "#8B5E3C"}
for lag in [1, 2, 3, 4]:
    ret = run(target, K, lag)
    cum = (1 + ret).cumprod()
    ax.plot(cum.index, cum.values, color=lag_colors[lag], lw=1.6, alpha=0.85, label=f"K-timed, lag={lag}")
ax.axhline(1.0, color="#AAAAAA", lw=0.8, ls=":")
ax.set_title("K-timed luxury/discount basket, all 4 lags vs. always-on\n"
             "every lag underperforms -- none of this is tradeable",
             fontsize=12.5, fontweight="bold", color=NAVY, pad=12)
ax.set_ylabel("Growth of $1")
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.grid(alpha=0.15)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# ---- Right: lag=3 as-is vs. inverted (the key "this isn't real" exhibit) ----
ax2 = axes[1]
ret3 = run(target, K, 3)
ret3_inv = run(target, K, 3, invert=True)
cum3 = (1 + ret3).cumprod()
cum3_inv = (1 + ret3_inv).cumprod()
ax2.plot(cum_always.index, cum_always.values, color=SLATE, lw=2, label="Always-on (no timing)")
ax2.plot(cum3.index, cum3.values, color=CRIMSON, lw=2.3, label="K-timed, lag=3 (as regression implies)")
ax2.plot(cum3_inv.index, cum3_inv.values, color=GREEN, lw=2.3, label="Inverted signal, lag=3")
ax2.axhline(1.0, color="#AAAAAA", lw=0.8, ls=":")
ax2.set_title("Lag=3: the \"significant\" specification vs. its own inverse\n"
              "inverting the signal does BETTER -- the tell that it isn't real",
              fontsize=12.5, fontweight="bold", color=NAVY, pad=12)
ax2.set_ylabel("Growth of $1")
ax2.legend(frameon=False, fontsize=9.5, loc="upper right")
ax2.grid(alpha=0.15)
for s in ["top", "right"]:
    ax2.spines[s].set_visible(False)

fig.suptitle("Can you trade on K's lag-3/lag-4 \"significance\"? No -- and the backtest shows exactly why",
             fontsize=14.5, fontweight="bold", color=NAVY, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("output/k_timed_luxury_discount_chart.png", dpi=150, facecolor="white")
print("saved output/k_timed_luxury_discount_chart.png")
