"""Does K predict the luxury/discount consumer basket's return?

Motivation: the original long/short (defensive sectors vs. growth sectors)
tests against K came back null, and on reflection the pairing had a real
conceptual problem -- K's wealth pillar is partly composed of top-decile
equity holdings, which overlap with exactly the growth/tech names the
original short leg bets against. That creates a mechanical, same-quarter
link running the wrong direction (growth rallying pushes K up, rather than
K predicting growth's future underperformance).

This basket targets K's *consumer* pillar directly instead: premium/luxury
consumer names (long) vs. discount/value consumer names (short) -- neither
side is an input into how K is computed, so there's no equivalent overlap.
See luxury_discount_construction.py for the basket itself and its data
provenance.

Same discipline as every other regression in this project: contemporaneous
and lagged tested as two SEPARATE regressions, swept across lag lengths
1-4, with an outlier-robustness check (objective >2-std rule) on anything
that clears 10%. K uses the full-sample level, matching every other K
regression published in this project (strategy_predictor_test.py,
reit_predictor_test.py, mbb_predictor_test.py, k_specification_test.py).

Run: python k_luxury_discount_test.py
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
    df = pd.concat([target, pred.rename("pred")], axis=1, sort=True).dropna()
    df["pred_lag1"] = df["pred"].shift(1)
    df = df.dropna()

    thresh = 2 * target.std()
    outliers = df[df[target.name].abs() > thresh]
    df_ex = df.drop(index=outliers.index)

    X = sm.add_constant(df[["pred_lag1"]])
    full = sm.OLS(df[target.name], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    X_ex = sm.add_constant(df_ex[["pred_lag1"]])
    ex = sm.OLS(df_ex[target.name], X_ex).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    lines.append(f"  outlier-robustness check on n_lags=1 ({len(outliers)} quarters with "
                 f"|basket return| > 2 std dropped): {list(outliers.index.strftime('%Y-%m-%d'))}")
    lines.append(f"    full sample:  coef={full.params['pred_lag1']:.4f}, p={full.pvalues['pred_lag1']:.4f} (N={len(df)})")
    lines.append(f"    ex-outliers:  coef={ex.params['pred_lag1']:.4f}, p={ex.pvalues['pred_lag1']:.4f} (N={len(df_ex)})")
    return full, ex


def main():
    target = quarterly_luxury_discount_return()
    K = build_k_index()[0]["K"]

    df = pd.concat([target, K], axis=1, sort=True).dropna()
    lines = []
    lines.append("Does K predict the luxury/discount consumer long-short basket's return?")
    lines.append("Methodology: contemporaneous_and_lagged_test() -- two SEPARATE")
    lines.append("regressions (contemporaneous-only, lagged-only joint F-test),")
    lines.append("swept across lag lengths 1-4, same discipline as every other")
    lines.append("regression in this project. K = full-sample level.")
    lines.append("")
    lines.append(f"N = {len(df)} quarters ({df.index.min().date()} -> {df.index.max().date()})")

    summary_rows = []
    contemp, _ = contemporaneous_and_lagged_test(target, K, n_lags=1)
    contemp_coef, contemp_p = contemp.params["K"], contemp.pvalues["K"]
    lines.append(f"contemporaneous: coef={contemp_coef:.4f}, p={contemp_p:.4f}")
    summary_rows.append({"test": "contemporaneous", "n_lags": 0, "coef_or_F": contemp_coef, "p_value": contemp_p})

    min_lag_p = contemp_p
    for n in [1, 2, 3, 4]:
        _, lagged = contemporaneous_and_lagged_test(target, K, n_lags=n)
        lines.append(f"n_lags={n}: joint F={lagged.fvalue:.3f}, p={lagged.f_pvalue:.4f}")
        summary_rows.append({"test": "lagged_joint_F", "n_lags": n, "coef_or_F": lagged.fvalue, "p_value": lagged.f_pvalue})
        min_lag_p = min(min_lag_p, lagged.f_pvalue)

    if min_lag_p < 0.10:
        _, ex = outlier_robustness_check(target, K, lines)
        summary_rows.append({"test": "contemporaneous_ex_outliers", "n_lags": 0,
                              "coef_or_F": ex.params["pred_lag1"], "p_value": ex.pvalues["pred_lag1"]})

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/k_luxury_discount_test_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/k_luxury_discount_test_summary.csv", index=False)
    print(f"\nSaved to {OUT}/k_luxury_discount_test_results.txt and {OUT}/k_luxury_discount_test_summary.csv")


if __name__ == "__main__":
    main()
