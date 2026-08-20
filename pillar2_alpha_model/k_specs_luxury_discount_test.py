"""Level, Difference, and Direction of K against the luxury/discount
consumer basket -- same three specifications as k_specification_test.py
(run against Treasury, where K already has a real relationship) and
all_angles_long_short.py (run against the original long/short), applied
here to the purpose-built basket from luxury_discount_construction.py.

  1. Level of K      -- K_t itself (k_luxury_discount_test.py's spec)
  2. Difference of K  -- K_t - K_{t-1}
  3. Direction of K   -- 1 if K rose from the prior quarter, 0 if it fell

All three use the full-sample level of K, matching every other K
regression in this project. Same discipline: contemporaneous and lagged
tested as two SEPARATE regressions, swept across lag lengths 1-4, with an
outlier-robustness check on anything that clears 10%.

Run: python k_specs_luxury_discount_test.py
"""
import sys

import pandas as pd
import statsmodels.api as sm

from _pathutil import find_dir_containing
from luxury_discount_construction import quarterly_luxury_discount_return

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
K_INDEX_DIR = REPO_ROOT / "k_index_model"
if str(K_INDEX_DIR) not in sys.path:
    sys.path.insert(0, str(K_INDEX_DIR))
from k_index_builder import build_k_index                  # noqa: E402
from regressions import contemporaneous_and_lagged_test     # noqa: E402

OUT = "output"


def outlier_robustness_check(target: pd.Series, pred: pd.Series, lines: list):
    df = pd.concat([target, pred.rename("spec")], axis=1, sort=True).dropna()
    thresh = 2 * target.std()
    outliers = df[df[target.name].abs() > thresh]
    df_ex = df.drop(index=outliers.index)

    X = sm.add_constant(df[["spec"]])
    full = sm.OLS(df[target.name], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    X_ex = sm.add_constant(df_ex[["spec"]])
    ex = sm.OLS(df_ex[target.name], X_ex).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    lines.append(f"  outlier-robustness check on contemporaneous ({len(outliers)} quarters with "
                 f"|basket return| > 2 std dropped): {list(outliers.index.strftime('%Y-%m-%d'))}")
    lines.append(f"    full sample:  coef={full.params['spec']:.4f}, p={full.pvalues['spec']:.4f} (N={len(df)})")
    lines.append(f"    ex-outliers:  coef={ex.params['spec']:.4f}, p={ex.pvalues['spec']:.4f} (N={len(df_ex)})")
    return full, ex


def main():
    target = quarterly_luxury_discount_return()
    K = build_k_index()[0]["K"]

    SPECS = {
        "Level of K": K,
        "Difference of K": K.diff(),
        "Direction of K": (K.diff() > 0).astype(float),
    }

    lines = []
    lines.append("Level, Difference, and Direction of K against the luxury/discount")
    lines.append("consumer basket's return. Methodology: contemporaneous_and_lagged_test()")
    lines.append("-- two SEPARATE regressions (contemporaneous-only, lagged-only joint")
    lines.append("F-test), swept across lag lengths 1-4, same discipline as every other")
    lines.append("regression in this project.")
    lines.append("")

    summary_rows = []
    for spec_name, pred in SPECS.items():
        df = pd.concat([target, pred.rename("K")], axis=1, sort=True).dropna()
        lines.append(f"=== {spec_name} ===")
        lines.append(f"  N = {len(df)} quarters ({df.index.min().date()} -> {df.index.max().date()})")
        contemp, _ = contemporaneous_and_lagged_test(target, pred, n_lags=1)
        contemp_coef, contemp_p = contemp.params["K"], contemp.pvalues["K"]
        lines.append(f"  contemporaneous: coef={contemp_coef:.4f}, p={contemp_p:.4f}")
        summary_rows.append({"spec": spec_name, "test": "contemporaneous", "n_lags": 0,
                              "coef_or_F": contemp_coef, "p_value": contemp_p})
        min_lag_p = contemp_p
        for n in [1, 2, 3, 4]:
            _, lagged = contemporaneous_and_lagged_test(target, pred, n_lags=n)
            lines.append(f"  n_lags={n}: joint F={lagged.fvalue:.3f}, p={lagged.f_pvalue:.4f}")
            summary_rows.append({"spec": spec_name, "test": "lagged_joint_F", "n_lags": n,
                                  "coef_or_F": lagged.fvalue, "p_value": lagged.f_pvalue})
            min_lag_p = min(min_lag_p, lagged.f_pvalue)

        if min_lag_p < 0.10:
            _, ex = outlier_robustness_check(target, pred, lines)
            summary_rows.append({"spec": spec_name, "test": "contemporaneous_ex_outliers", "n_lags": 0,
                                  "coef_or_F": ex.params["spec"], "p_value": ex.pvalues["spec"]})
        lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/k_specs_luxury_discount_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/k_specs_luxury_discount_summary.csv", index=False)
    print(f"\nSaved to {OUT}/k_specs_luxury_discount_results.txt and {OUT}/k_specs_luxury_discount_summary.csv")


if __name__ == "__main__":
    main()
