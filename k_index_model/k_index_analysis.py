"""Reproduces the notebook's robustness section (4.1 pillar correlations,
5.1 leave-one-pillar-out, 7 structural-break Chow tests) against the now-
complete 3-pillar K-Index, plus the headline chart. All real, all validated
against the notebook's own printed output (see README).

Run: python k_index_analysis.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from k_index_builder import build_k_index

OUT = "output"
NAVY, RED, GREEN, GREY = "#0B1F3A", "#C8102E", "#2E7D32", "#6C757D"


def chow_test(y, t, split):
    def rss(yy, tt):
        X = np.column_stack([np.ones(len(tt)), tt])
        b, *_ = np.linalg.lstsq(X, yy, rcond=None)
        r = yy - X @ b
        return float(r @ r)
    if split < 4 or len(y) - split < 4:
        return np.nan, np.nan
    rp = rss(y, t)
    rs = rss(y[:split], t[:split]) + rss(y[split:], t[split:])
    k = 2
    N = len(y)
    F = ((rp - rs) / k) / (rs / (N - 2 * k))
    return F, 1 - stats.f.cdf(F, k, N - 2 * k)


def main():
    Z, pillars_used, loadings = build_k_index()
    if len(pillars_used) != 3:
        print(f"WARNING: only {len(pillars_used)} pillars available "
              f"({pillars_used}) -- robustness checks below assume 3.")

    z_cols = [f"z_{p}" for p in pillars_used]

    print("=== Pillar correlations ===")
    print(Z[z_cols].corr().round(2).to_string())

    print("\n=== Leave-one-pillar-out ===")
    for drop in z_cols:
        keep = [c for c in z_cols if c != drop]
        k_loo = Z[keep].mean(axis=1)
        rho = k_loo.corr(Z["K"])
        print(f"drop {drop.replace('z_', ''):9s} -> corr with full K = {rho:.3f}")

    print(f"\n=== Cross-scheme correlation ===")
    print(Z[["K", "K_pca", "K_invvar"]].corr().round(3).to_string())

    K = Z["K"].dropna()
    y, t = K.values.astype(float), np.arange(len(K), dtype=float)
    print("\n=== Structural break (Chow test, H0: no break at date) ===")
    break_rows = []
    for label, ts in {"GFC 2008Q3": "2008-09-30", "COVID 2020Q2": "2020-06-30", "2022Q3": "2022-09-30"}.items():
        pos = K.index.get_indexer([pd.Timestamp(ts)], method="nearest")[0]
        F, p = chow_test(y, t, pos)
        flag = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"  {label:14s} F={F:7.2f}  p={p:.4f}  {flag}")
        break_rows.append({"break_date": label, "F": F, "p_value": p})
    pd.DataFrame(break_rows).to_csv(f"{OUT}/k_index_chow_tests.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color="#888", lw=0.8)
    ax.plot(Z.index, Z["K"], color=NAVY, lw=2.4, label="K-Index (equal weight)")
    ax.plot(Z.index, Z["K_pca"], color=GREY, lw=1.1, ls="--", label="K-Index (PCA)")
    ax.set_title("The K-Index — U.S. Cohort Divergence (z-score)", color=NAVY, weight="bold", fontsize=13)
    ax.set_ylabel("Standardized divergence")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/k_index.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color="#888", lw=0.8)
    colors = {"z_wealth": NAVY, "z_income": RED, "z_consumer": GREEN}
    for c in z_cols:
        ax.plot(Z.index, Z[c], color=colors.get(c, GREY), lw=1.8, label=c.replace("z_", "").capitalize())
    ax.set_title("Three pillars (z-scored)", color=NAVY, weight="bold", fontsize=13)
    ax.set_ylabel("Standardized spread")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/k_index_pillars.png", dpi=150)
    plt.close()

    Z.to_csv(f"{OUT}/kindex.csv")
    print(f"\nSaved charts and kindex.csv to {OUT}/")


if __name__ == "__main__":
    main()
