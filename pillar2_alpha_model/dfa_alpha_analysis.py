"""Consolidated analysis: builds every dfa_signals.py series, runs the
predictive tests in predictive_test.py across BOTH cohort cuts (generation
and age), and writes the real charts + a results table to output/. This is
the single script to run to reproduce every number in the README's
"Results — what the DFA data actually shows" section.

Run: python dfa_alpha_analysis.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from dfa_signals import (
    rotation_signal, k_shape_intensity, aggregate_equity_growth,
    generation_asset_shares, EQUITY_COL, SAFE_COLS, _load,
)

OUT = "output"


def age_rotation_signal(age_bucket: str) -> pd.DataFrame:
    """Same combined-bucket-then-share logic as generation_asset_shares in
    dfa_signals.py (see the correctness note there): sum dollar levels across
    the safe-asset columns first, THEN take one share of that total — summing
    per-column percentage shares directly would double-count and can exceed
    100%."""
    df = _load("dfa-age-levels-detail.csv").copy()
    df["safe_assets_total"] = df[SAFE_COLS].sum(axis=1)

    cols = [EQUITY_COL, "safe_assets_total"]
    totals = df.groupby("Date")[cols].transform("sum")
    shares = df[cols] / totals * 100
    df["equity_share_pct"] = shares[EQUITY_COL]
    df["safe_share_pct"] = shares["safe_assets_total"]

    out = df[df["Category"] == age_bucket].set_index("Date").sort_index()
    out["rotation_spread"] = out["equity_share_pct"] - out["safe_share_pct"]
    out["rotation_spread_qoq_chg"] = out["rotation_spread"].diff()
    return out[["equity_share_pct", "safe_share_pct", "rotation_spread", "rotation_spread_qoq_chg"]]


def lagged_regression(y: pd.Series, x: pd.Series, max_lag: int = 4, hac: int = 3):
    df = pd.DataFrame({"y": y, "x": x})
    cols = []
    for lag in range(1, max_lag + 1):
        c = f"x_lag{lag}"
        df[c] = df["x"].shift(lag)
        cols.append(c)
    df = df.dropna()
    X = sm.add_constant(df[cols])
    return sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac})


def main():
    # --- Chart 1: BabyBoom equity share vs safe-asset share, 1989-2026 ---
    boomer = rotation_signal("BabyBoom")
    plt.figure(figsize=(10, 6))
    plt.plot(boomer.index, boomer["equity_share_pct"], label="BabyBoom share of household equities (%)")
    plt.plot(boomer.index, boomer["safe_share_pct"], label="BabyBoom share of deposits/bonds/annuities (%)")
    plt.title("Baby Boomer share of U.S. household equities vs. safe assets")
    plt.ylabel("Share of total household-sector holdings (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/dfa_boomer_rotation.png", dpi=150)
    plt.close()

    # --- Chart 2: K-shape gap (Top1% - Bottom50% net worth share) ---
    k = k_shape_intensity()
    plt.figure(figsize=(10, 6))
    plt.plot(k.index, k["top1_share_pct"], label="Top 1% share of net worth (%)")
    plt.plot(k.index, k["bottom50_share_pct"], label="Bottom 50% share of net worth (%)")
    plt.plot(k.index, k["k_shape_gap"], label="Gap (Top 1% - Bottom 50%)", linestyle="--")
    plt.title("Wealth concentration gap, from DFA net-worth shares")
    plt.ylabel("Share of total net worth (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/dfa_k_shape_gap.png", dpi=150)
    plt.close()

    # --- Predictive tests across cohort cuts AND lag lengths (robustness sweep) ---
    # A specification that's only significant at one lag length and not others
    # is a red flag for a spuriously-mined result rather than a real signal —
    # so lag length is swept explicitly instead of reporting a single choice.
    eq_growth = aggregate_equity_growth()
    rows = []
    cuts = {f"generation:{g}": rotation_signal(g) for g in ["BabyBoom", "GenX", "Millennial", "Silent"]}
    for bucket in ["age70plus", "age55to69", "age40to54", "ageunder40"]:
        cuts[f"age:{bucket}"] = age_rotation_signal(bucket)

    for name, sig in cuts.items():
        panel = sig.join(k, how="inner").join(eq_growth, how="inner").dropna()
        trend_corr = pd.Series(range(len(panel)), index=panel.index).corr(panel["rotation_spread"])
        level_corr_k = panel["rotation_spread"].corr(panel["k_shape_gap"])

        for max_lag in [2, 4]:
            m_k = lagged_regression(panel["k_shape_gap_qoq_chg"], panel["rotation_spread_qoq_chg"], max_lag=max_lag)
            m_eq = lagged_regression(panel["agg_equity_qoq_growth"], panel["rotation_spread_qoq_chg"], max_lag=max_lag)
            rows.append({
                "cohort_cut": name,
                "max_lag": max_lag,
                "corr(rotation_spread, calendar_time)": round(trend_corr, 3),
                "corr(rotation_spread, k_shape_gap) [level]": round(level_corr_k, 3),
                "predict k_shape_gap_chg: F p-value": round(m_k.f_pvalue, 3),
                "predict k_shape_gap_chg: R2": round(m_k.rsquared, 3),
                "predict agg_equity_growth: F p-value": round(m_eq.f_pvalue, 3),
                "predict agg_equity_growth: R2": round(m_eq.rsquared, 3),
            })

    results = pd.DataFrame(rows).set_index(["cohort_cut", "max_lag"])
    results.to_csv(f"{OUT}/dfa_predictive_test_results.csv")
    pd.set_option("display.width", 160)
    print(results.to_string())

    n_tests = len(results) * 2  # k_shape + eq_growth columns, both tested per row
    n_sig = ((results.filter(like="F p-value") < 0.05).sum().sum())
    print(f"\n{n_sig} of {n_tests} specification/outcome combinations clear p<0.05 "
          f"— consistent with the ~{0.05*n_tests:.1f} expected by chance alone at a 5% "
          "threshold with no true effect. See README for the robustness read.")
    print(f"\nSaved charts and results table to {OUT}/")


if __name__ == "__main__":
    main()
