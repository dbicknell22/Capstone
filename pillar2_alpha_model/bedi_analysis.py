"""Charts BEDI (both versions), reports the equal-weight-vs-PCA comparison
Pillar 1 also ran, and tests for the structural break around 2019-2020 that
the retirement-wave thesis predicts. All real, all computed from the DFA
data — no external market data needed.

Run: python bedi_analysis.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from bedi_index import build_bedi_full_sample, build_bedi_expanding, _raw_components

OUT = "output"


def structural_break_test(bedi: pd.Series, break_date="2019-10-01"):
    """Tests for a structural break at `break_date` in BEDI's linear time
    trend. The combined model (post dummy AND trend-interaction together)
    is included for completeness but its individual coefficients are not
    interpretable here: `post` and `t_post` correlate 0.998 in this sample
    (t_post = t*post, and post-2019 t values span a narrow, late range, so
    the two are nearly collinear) — the joint F-test can be significant while
    both individual t-stats look insignificant, which is a multicollinearity
    artifact, not evidence of "no break." So a level-shift-only and a
    trend-only model are also fit separately, each identifiable on its own."""
    df = pd.DataFrame({"y": bedi})
    df["t"] = range(len(df))
    df["post"] = (df.index >= pd.Timestamp(break_date)).astype(int)
    df["t_post"] = df["t"] * df["post"]

    restricted = sm.OLS(df["y"], sm.add_constant(df[["t"]])).fit()
    combined = sm.OLS(df["y"], sm.add_constant(df[["t", "post", "t_post"]])).fit()
    level_only = sm.OLS(df["y"], sm.add_constant(df[["t", "post"]])).fit()
    trend_only = sm.OLS(df["y"], sm.add_constant(df[["t", "t_post"]])).fit()
    f_test = combined.compare_f_test(restricted)

    return {
        "restricted": restricted, "combined": combined,
        "level_only": level_only, "trend_only": trend_only,
        "f_test": f_test, "collinearity_post_vs_tpost": df["post"].corr(df["t_post"]),
    }


def main():
    full, loadings, evr = build_bedi_full_sample()
    exp = build_bedi_expanding()
    raw = _raw_components()

    corr_components = raw["rotation_spread"].corr(raw["k_shape_gap"])
    corr_full = full["BEDI_equal_weight"].corr(full["BEDI_pca"])
    corr_exp = exp["BEDI_equal_weight"].corr(exp["BEDI_pca"])

    print("=== Component co-movement check (mirrors Pillar 1's robustness check) ===")
    print(f"corr(rotation_spread, k_shape_gap), raw levels, full sample: {corr_components:.3f}")
    print("Pillar 1's three indicators correlated 0.87-0.96 with each other (a real common "
          "factor). These two components correlate far weaker, so PCA weighting here is not "
          "confirming a dominant shared driver the way it did in Pillar 1 — it's revealing "
          "that there mostly isn't one, at least not one this pair of signals picks up.")
    print(f"corr(BEDI_equal_weight, BEDI_pca), full-sample z-score: {corr_full:.3f}")
    print(f"corr(BEDI_equal_weight, BEDI_pca), expanding/point-in-time z-score: {corr_exp:.3f}")
    print(f"PCA explained variance ratio (full-sample): {evr:.3f} "
          f"(50% would mean the two components are equally-weighted noise with no shared axis)")
    print(f"PCA loadings (full-sample): {loadings.to_dict()}\n")

    # --- Chart: components + BEDI, full-sample z-score ---
    plt.figure(figsize=(10, 6))
    plt.plot(full.index, full["z_rotation"], label="z(Boomer de-risking)", alpha=0.6)
    plt.plot(full.index, full["z_k_shape"], label="z(K-shape gap)", alpha=0.6)
    plt.plot(full.index, full["BEDI_equal_weight"], label="BEDI (equal-weight)", linewidth=2, color="black")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.title("Boomer Equity Displacement Index (BEDI) — full-sample z-score")
    plt.ylabel("Standard deviations from full-sample mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/bedi_full_sample.png", dpi=150)
    plt.close()

    # --- Chart: BEDI equal-weight vs PCA, expanding/point-in-time version ---
    plt.figure(figsize=(10, 6))
    plt.plot(exp.index, exp["BEDI_equal_weight"], label="BEDI (equal-weight, point-in-time)")
    plt.plot(exp.index, exp["BEDI_pca"], label="BEDI (PCA, point-in-time)", linestyle="--")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(pd.Timestamp("2019-10-01"), color="red", linewidth=0.8, linestyle=":",
                label="proposed 2019-2020 structural break")
    plt.title("BEDI — point-in-time (no look-ahead) version used in the forward-return regression")
    plt.ylabel("Standard deviations (expanding-window)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/bedi_expanding.png", dpi=150)
    plt.close()

    # --- Structural break test ---
    print("=== Structural break test: does BEDI's level/trend shift at 2019:Q4? ===")
    res = structural_break_test(full["BEDI_equal_weight"])
    print(f"Joint F-test (combined vs. trend-only restricted model): "
          f"F={res['f_test'][0]:.3f}, p={res['f_test'][1]:.6f}")
    print(f"corr(post, t_post) in the combined model: {res['collinearity_post_vs_tpost']:.3f} "
          "— this near-perfect collinearity is why the combined model's individual "
          "coefficients below are NOT reliable on their own; see level_only/trend_only instead.\n")
    print("--- Combined model (post + t_post together — collinear, do not read individually) ---")
    print(res["combined"].summary().tables[1])
    print("\n--- Level-shift-only model (cleanly identified) ---")
    print(res["level_only"].summary().tables[1])
    print("\n--- Trend-slope-change-only model (cleanly identified) ---")
    print(res["trend_only"].summary().tables[1])

    with open(f"{OUT}/bedi_structural_break_test.txt", "w") as f:
        f.write(f"Joint F-test: F={res['f_test'][0]:.4f}, p={res['f_test'][1]:.6f}\n")
        f.write(f"corr(post, t_post): {res['collinearity_post_vs_tpost']:.4f}\n\n")
        f.write("=== Combined model (collinear — do not read individual coefficients) ===\n")
        f.write(str(res["combined"].summary()) + "\n\n")
        f.write("=== Level-shift-only model ===\n")
        f.write(str(res["level_only"].summary()) + "\n\n")
        f.write("=== Trend-slope-change-only model ===\n")
        f.write(str(res["trend_only"].summary()))

    full.to_csv(f"{OUT}/bedi_full_sample.csv")
    exp.to_csv(f"{OUT}/bedi_expanding.csv")
    print(f"\nSaved charts, CSVs, and structural-break test to {OUT}/")


if __name__ == "__main__":
    main()
