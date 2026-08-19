"""Does K, BEDI, the rotation signal, or the real-estate rotation signal
predict MBB (iShares MBS ETF) returns?

Motivation: every real, robust result in this project involves a
rate-sensitive fixed-income instrument (10Y Treasury, IG credit spread).
REITs -- tested earlier -- are equity, not fixed income, and that test's
one promising reading (K contemporaneous) turned out to be a crisis-period
co-movement, not a real relationship. MBS are a much closer analog to
Treasuries: fixed income, priced primarily off mortgage rates and the
Treasury curve, with a duration profile (roughly 5-6 years for MBB) not
far from IEF's (roughly 7-8 years) -- while still being literally about
real estate (mortgages), unlike a generic bond index. This tests whether
the rate-sensitivity story extends to mortgage-backed fixed income.

Caveat carried into the interpretation, not just the caveats list: MBS
have negative convexity (prepayment risk) -- when rates fall, refinancing
accelerates, capping price appreciation compared to a Treasury of similar
duration. So even a real relationship here might show up dampened
relative to the Treasury result, not equally strong.

Tests all four predictors already validated elsewhere in this project,
using the same predictor construction as their most recent use:
  - K-Index (full-sample level, matching every K regression in this project)
  - BEDI (expanding/point-in-time, matching every BEDI regression)
  - Rotation signal, isolated (raw DFA level, matching strategy_predictor_test.py)
  - Real-estate rotation (raw DFA level, matching reit_predictor_test.py)

Same discipline as every other regression in this project: contemporaneous
and lagged tested as two SEPARATE regressions, swept across lag lengths
1-4, with an outlier-robustness check (objective >2-std rule) on anything
that clears 10%.

Run: python mbb_predictor_test.py
"""
import sys

import pandas as pd
import statsmodels.api as sm

from _pathutil import find_dir_containing
from bedi_index import build_bedi_expanding
from dfa_signals import rotation_signal, real_estate_rotation
from data_sources import load_prices

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
K_INDEX_DIR = REPO_ROOT / "k_index_model"
if str(K_INDEX_DIR) not in sys.path:
    sys.path.insert(0, str(K_INDEX_DIR))
from k_index_builder import build_k_index                  # noqa: E402
from regressions import contemporaneous_and_lagged_test      # noqa: E402

OUT = "output"


def load_mbb_return() -> pd.Series:
    prices = load_prices(["MBB"], "2007-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
    q = prices["MBB"].resample("QE").last()
    return q.pct_change().rename("mbb_qtr_return")


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
                 f"|MBB return| > 2 std dropped): {list(outliers.index.strftime('%Y-%m-%d'))}")
    lines.append(f"    full sample:  coef={full.params['pred_lag1']:.4f}, p={full.pvalues['pred_lag1']:.4f} (N={len(df)})")
    lines.append(f"    ex-outliers:  coef={ex.params['pred_lag1']:.4f}, p={ex.pvalues['pred_lag1']:.4f} (N={len(df_ex)})")
    return full, ex


def main():
    target = load_mbb_return()

    predictors = {
        "K-Index (level)": build_k_index()[0]["K"],
        "BEDI (expanding, equal-weight)": build_bedi_expanding()["BEDI_equal_weight"],
        "Rotation signal (Boomer equity - safe-asset share)": rotation_signal("BabyBoom")["rotation_spread"],
        "Real-estate rotation (Boomer real estate share of assets)": real_estate_rotation("BabyBoom")["real_estate_share_pct"],
    }

    lines = []
    lines.append("Does K, BEDI, the rotation signal, or real-estate rotation predict MBB (MBS) returns?")
    lines.append("Methodology: contemporaneous_and_lagged_test() -- two SEPARATE")
    lines.append("regressions (contemporaneous-only, lagged-only joint F-test),")
    lines.append("swept across lag lengths 1-4, same discipline as every other")
    lines.append("regression in this project.")
    lines.append("")

    summary_rows = []
    for name, pred in predictors.items():
        df = pd.concat([target, pred], axis=1, sort=True).dropna()
        lines.append(f"=== {name} ===")
        lines.append(f"  N = {len(df)} quarters ({df.index.min().date()} -> {df.index.max().date()})")
        contemp, _ = contemporaneous_and_lagged_test(target, pred, n_lags=1)
        contemp_coef, contemp_p = contemp.params["K"], contemp.pvalues["K"]
        lines.append(f"  contemporaneous: coef={contemp_coef:.4f}, p={contemp_p:.4f}")
        summary_rows.append({"predictor": name, "test": "contemporaneous", "n_lags": 0,
                              "coef_or_F": contemp_coef, "p_value": contemp_p})
        min_lag_p = contemp_p
        for n in [1, 2, 3, 4]:
            _, lagged = contemporaneous_and_lagged_test(target, pred, n_lags=n)
            lines.append(f"  n_lags={n}: joint F={lagged.fvalue:.3f}, p={lagged.f_pvalue:.4f}")
            summary_rows.append({"predictor": name, "test": "lagged_joint_F", "n_lags": n,
                                  "coef_or_F": lagged.fvalue, "p_value": lagged.f_pvalue})
            min_lag_p = min(min_lag_p, lagged.f_pvalue)

        if min_lag_p < 0.10:
            _, ex = outlier_robustness_check(target, pred, lines)
            summary_rows.append({"predictor": name, "test": "contemporaneous_ex_outliers", "n_lags": 0,
                                  "coef_or_F": ex.params["pred_lag1"], "p_value": ex.pvalues["pred_lag1"]})
        lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/mbb_predictor_test_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/mbb_predictor_test_summary.csv", index=False)
    print(f"Saved to {OUT}/mbb_predictor_test_results.txt and {OUT}/mbb_predictor_test_summary.csv")


if __name__ == "__main__":
    main()
