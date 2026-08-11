"""Does the Boomer real-estate rotation signal predict REIT returns?

The most direct pairing available in this project: real_estate_rotation()
(dfa_signals.py) is a real-estate-selling proxy built from DFA data that
was never tradeable on its own -- reit_basket_quarterly.csv (added to the
repo root) is the first real, investable real-estate price series in this
project, making it the natural target for that signal instead of the
strategy's own (non-real-estate) basket returns.

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

from _pathutil import find_dir_containing
from dfa_signals import real_estate_rotation

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
K_INDEX_DIR = REPO_ROOT / "k_index_model"
if str(K_INDEX_DIR) not in sys.path:
    sys.path.insert(0, str(K_INDEX_DIR))
from regressions import contemporaneous_and_lagged_test  # noqa: E402

OUT = "output"


def load_reit_returns() -> pd.Series:
    df = pd.read_csv(REPO_ROOT / "reit_basket_quarterly.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["reit_ret"].sort_index().rename("reit_qtr_return")


def main():
    target = load_reit_returns()

    predictors = {
        "real_estate_rotation (Boomer real estate share of assets)": real_estate_rotation("BabyBoom")["real_estate_share_pct"],
    }

    lines = []
    lines.append("Does the Boomer real-estate rotation signal predict REIT basket returns?")
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
        lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/reit_predictor_test_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/reit_predictor_test_summary.csv", index=False)
    print(f"Saved to {OUT}/reit_predictor_test_results.txt and {OUT}/reit_predictor_test_summary.csv")


if __name__ == "__main__":
    main()
