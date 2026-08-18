"""Advisor comment: "It is important to clarify what measure of K you use
for the asset price effects. It could be the level of K, the difference of
K, whether K is growing or shrinking. All make sense and might be tried."

Every K regression elsewhere in this project (run_k_regressions.py,
mechanism_tests.py, the K-timed Treasury backtest) uses the LEVEL of K --
K's own z-scored value at each quarter, and its lags. That was never made
explicit as a deliberate choice among alternatives; this script makes the
choice explicit and tests the two alternatives the advisor named, against
the one target where K already has a real, robust relationship (the 10Y
Treasury), so specification choice is checked exactly where it matters most:

  1. Level of K       -- K_t itself (the existing approach everywhere else)
  2. Difference of K   -- K_t - K_{t-1}, the size and direction of the
                          quarter-over-quarter change
  3. Direction of K    -- a binary indicator, 1 if K rose from the prior
                          quarter, 0 if it fell (magnitude-free; a purely
                          qualitative "widening vs. tightening" test)

Same discipline as every other regression in this project: contemporaneous
and lagged tested as two SEPARATE regressions, swept across lag lengths 1-4.

Run: python k_specification_test.py
"""
import pandas as pd
import statsmodels.api as sm

from k_index_builder import build_k_index
from target_data import load_pct_change
from regressions import contemporaneous_and_lagged_test

OUT = "output"


def outlier_robustness_check(target: pd.Series, pred: pd.Series, lines: list):
    """Same objective, pre-specified outlier rule used elsewhere in this
    project: drop quarters where the TARGET (Treasury return) itself
    exceeds 2 standard deviations, re-fit the contemporaneous
    specification, and see if the result survives."""
    df = pd.concat([target, pred.rename("spec")], axis=1, sort=True).dropna()
    thresh = 2 * target.std()
    outliers = df[df[target.name].abs() > thresh]
    df_ex = df.drop(index=outliers.index)

    X = sm.add_constant(df[["spec"]])
    full = sm.OLS(df[target.name], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    X_ex = sm.add_constant(df_ex[["spec"]])
    ex = sm.OLS(df_ex[target.name], X_ex).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    lines.append(f"  outlier-robustness check on contemporaneous ({len(outliers)} quarters with "
                 f"|Treasury return| > 2 std dropped): {list(outliers.index.strftime('%Y-%m-%d'))}")
    lines.append(f"    full sample:  coef={full.params['spec']:.4f}, p={full.pvalues['spec']:.4f} (N={len(df)})")
    lines.append(f"    ex-outliers:  coef={ex.params['spec']:.4f}, p={ex.pvalues['spec']:.4f} (N={len(df_ex)})")
    return full, ex


def main():
    K = build_k_index()[0]["K"]
    target = load_pct_change("treasury_10y_total_return")

    specs = {
        "Level of K": K,
        "Difference of K (K_t - K_t-1)": K.diff(),
        "Direction of K (1=rising, 0=falling)": (K.diff() > 0).astype(float),
    }

    lines = []
    lines.append("Does the choice of K specification change the K -> 10Y Treasury result?")
    lines.append("Target: treasury_10y_total_return_pct_chg (same target as run_k_regressions.py)")
    lines.append("Methodology: contemporaneous_and_lagged_test() -- two SEPARATE regressions,")
    lines.append("swept across lag lengths 1-4, same discipline as every other regression")
    lines.append("in this project.")
    lines.append("")

    summary_rows = []
    for name, pred in specs.items():
        df = pd.concat([target, pred.rename("spec")], axis=1, sort=True).dropna()
        lines.append(f"=== {name} ===")
        lines.append(f"  N = {len(df)} quarters ({df.index.min().date()} -> {df.index.max().date()})")
        contemp, _ = contemporaneous_and_lagged_test(target, pred, n_lags=1)
        contemp_coef, contemp_p = contemp.params["K"], contemp.pvalues["K"]
        lines.append(f"  contemporaneous: coef={contemp_coef:.4f}, p={contemp_p:.4f}")
        summary_rows.append({"spec": name, "test": "contemporaneous", "n_lags": 0,
                              "coef_or_F": contemp_coef, "p_value": contemp_p})
        for n in [1, 2, 3, 4]:
            _, lagged = contemporaneous_and_lagged_test(target, pred, n_lags=n)
            lines.append(f"  n_lags={n}: joint F={lagged.fvalue:.3f}, p={lagged.f_pvalue:.4f}")
            summary_rows.append({"spec": name, "test": "lagged_joint_F", "n_lags": n,
                                  "coef_or_F": lagged.fvalue, "p_value": lagged.f_pvalue})
        if name != "Level of K" and contemp_p < 0.10:
            _, ex = outlier_robustness_check(target, pred, lines)
            summary_rows.append({"spec": name, "test": "contemporaneous_ex_outliers", "n_lags": 0,
                                  "coef_or_F": ex.params["spec"], "p_value": ex.pvalues["spec"]})
        lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/k_specification_test_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/k_specification_test_summary.csv", index=False)
    print(f"Saved to {OUT}/k_specification_test_results.txt and {OUT}/k_specification_test_summary.csv")


if __name__ == "__main__":
    main()
