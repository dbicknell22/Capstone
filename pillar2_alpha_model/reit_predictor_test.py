"""Does the Boomer real-estate rotation signal, or the K-Index, predict
REIT returns?

The most direct pairing available in this project: real_estate_rotation()
(dfa_signals.py) is a real-estate-selling proxy built from DFA data that
was never tradeable on its own -- reit_basket_quarterly.csv (added to the
repo root) is the first real, investable real-estate price series in this
project, making it the natural target for that signal instead of the
strategy's own (non-real-estate) basket returns.

K-Index is added as a second predictor here for the same reason it's been
tested against every other asset class in this project (stocks, Treasuries,
FX, gold) -- REITs were the one major asset class still missing from that
list, and real estate is arguably the asset class *most* directly exposed
to a Boomer-retirement/K-shape story, more so than generic equities.

Same discipline as every other regression in this project: contemporaneous
and lagged tested as two SEPARATE regressions, swept across lag lengths
1-4, HAC-robust standard errors -- never one combined "current + 4 lags"
model, which is exactly what produced false positives for S&P 500 and GDP
elsewhere in this repo.

Note: reit_basket_quarterly.csv has no composition/source documentation in
the repo beyond its filename and a single `reit_ret` column -- treat this
as "a REIT basket return series" until its exact source/composition is
confirmed.

Run: python reit_predictor_test.py
"""
import sys

import pandas as pd
import statsmodels.api as sm

from _pathutil import find_dir_containing
from dfa_signals import real_estate_rotation

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
K_INDEX_DIR = REPO_ROOT / "k_index_model"
if str(K_INDEX_DIR) not in sys.path:
    sys.path.insert(0, str(K_INDEX_DIR))
from k_index_builder import build_k_index          # noqa: E402
from regressions import contemporaneous_and_lagged_test  # noqa: E402

OUT = "output"


def load_reit_returns() -> pd.Series:
    df = pd.read_csv(REPO_ROOT / "reit_basket_quarterly.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["reit_ret"].sort_index().rename("reit_qtr_return")


def outlier_robustness_check(target: pd.Series, pred: pd.Series, lines: list):
    """Same discipline as the BEDI->LQD-SPY outlier check elsewhere in this
    project, but with a pre-specified, objective outlier rule (quarters
    where |target| exceeds 2 standard deviations of the FULL reit_ret
    series -- not cherry-picked dates) so this can't be read as tuning the
    exclusion set to kill an inconvenient result."""
    thresh = 2 * target.std()
    df = pd.concat([target, pred.rename("K")], axis=1, sort=True).dropna()
    outliers = df[df[target.name].abs() > thresh]
    df_ex = df.drop(index=outliers.index)

    X = sm.add_constant(df[["K"]])
    full = sm.OLS(df[target.name], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    X_ex = sm.add_constant(df_ex[["K"]])
    ex = sm.OLS(df_ex[target.name], X_ex).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    lines.append(f"  outlier-robustness check ({len(outliers)} quarters with |return| > 2 std dropped, "
                 f"objective rule, not hand-picked): {list(outliers.index.strftime('%Y-%m-%d'))}")
    lines.append(f"    full sample:  coef={full.params['K']:.4f}, p={full.pvalues['K']:.4f} (N={len(df)})")
    lines.append(f"    ex-outliers:  coef={ex.params['K']:.4f}, p={ex.pvalues['K']:.4f} (N={len(df_ex)})")
    return full, ex


def main():
    target = load_reit_returns()

    predictors = {
        "real_estate_rotation (Boomer real estate share of assets)": real_estate_rotation("BabyBoom")["real_estate_share_pct"],
        "K-Index": build_k_index()[0]["K"],
    }

    lines = []
    lines.append("Does the Boomer real-estate rotation signal, or K, predict REIT basket returns?")
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
        for n in [1, 2, 3, 4]:
            _, lagged = contemporaneous_and_lagged_test(target, pred, n_lags=n)
            lines.append(f"  n_lags={n}: joint F={lagged.fvalue:.3f}, p={lagged.f_pvalue:.4f}")
            summary_rows.append({"predictor": name, "test": "lagged_joint_F", "n_lags": n,
                                  "coef_or_F": lagged.fvalue, "p_value": lagged.f_pvalue})

        if contemp_p < 0.10:
            _, ex = outlier_robustness_check(target, pred, lines)
            summary_rows.append({"predictor": name, "test": "contemporaneous_ex_outliers", "n_lags": 0,
                                  "coef_or_F": ex.params["K"], "p_value": ex.pvalues["K"]})
        lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/reit_predictor_test_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/reit_predictor_test_summary.csv", index=False)
    print(f"Saved to {OUT}/reit_predictor_test_results.txt and {OUT}/reit_predictor_test_summary.csv")


if __name__ == "__main__":
    main()
