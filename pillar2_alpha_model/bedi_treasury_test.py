"""Does BEDI -- or the isolated Boomer rotation signal -- predict the 10Y
Treasury return?

BEDI has been tested against LQD-SPY and XLV-XLY forward returns
(bedi_forward_return_test.py) and the isolated rotation signal has been
tested against the long/short strategy and REIT returns
(strategy_predictor_test.py, reit_predictor_test.py) -- but neither has
ever been tested against the 10Y Treasury directly, which is the one
instrument K itself has a robust, lag-1 relationship with
(k_index_model/README.md). This fills that gap: if Boomers actually
rotating into fixed income is part of *why* K predicts Treasury returns,
BEDI or the rotation signal should show something similar.

Uses BEDI's point-in-time/expanding version (bedi_index.build_bedi_expanding)
for the same no-look-ahead reason build_k_index_expanding() was built for
the K-timed Treasury backtest -- not the full-sample version used in the
original LQD-SPY/XLV-XLY test.

Same discipline as every other regression in this project: contemporaneous
and lagged tested as two SEPARATE regressions, swept across lag lengths
1-4.

Run: python bedi_treasury_test.py
"""
import sys

import pandas as pd
import statsmodels.api as sm

from _pathutil import find_dir_containing
from bedi_index import build_bedi_expanding
from dfa_signals import rotation_signal

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
K_INDEX_DIR = REPO_ROOT / "k_index_model"
if str(K_INDEX_DIR) not in sys.path:
    sys.path.insert(0, str(K_INDEX_DIR))
from regressions import contemporaneous_and_lagged_test  # noqa: E402

OUT = "output"


def load_treasury_return() -> pd.Series:
    """Reads k_index_model/data_cache/treasury_10y_total_return.csv
    directly, bypassing target_data.py's own load_pct_change(). That
    module resolves its data_cache path via a marker-file search
    (find_dir_containing("k_index_builder.py")) that assumes it's being
    run from inside k_index_model -- calling it from here hits a real
    cross-directory bug: both k_index_model and pillar2_alpha_model have
    their own same-named _pathutil.py module, and Python's import cache
    keeps whichever one this process loaded first (pillar2_alpha_model's,
    since it's imported at the top of this file), so target_data.py's
    internal path search runs with the wrong module's search root and
    fails. K_INDEX_DIR is already known here, so just read the file."""
    df = pd.read_csv(K_INDEX_DIR / "data_cache" / "treasury_10y_total_return.csv",
                      parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Value"].resample("QE").last().pct_change().rename("treasury_10y_total_return")


def outlier_robustness_check(target: pd.Series, pred: pd.Series, lines: list):
    """Same objective, pre-specified outlier rule as reit_predictor_test.py:
    drop quarters where the TARGET (Treasury return) itself exceeds 2
    standard deviations, re-fit the single-lag (n_lags=1) specification,
    and see if the result survives. Not hand-picked dates."""
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
                 f"|Treasury return| > 2 std dropped): {list(outliers.index.strftime('%Y-%m-%d'))}")
    lines.append(f"    full sample:  coef={full.params['pred_lag1']:.4f}, p={full.pvalues['pred_lag1']:.4f} (N={len(df)})")
    lines.append(f"    ex-outliers:  coef={ex.params['pred_lag1']:.4f}, p={ex.pvalues['pred_lag1']:.4f} (N={len(df_ex)})")
    return full, ex


def main():
    target = load_treasury_return()

    predictors = {
        "BEDI (expanding, equal-weight)": build_bedi_expanding()["BEDI_equal_weight"],
        "rotation_signal (Boomer equity - safe-asset share)": rotation_signal("BabyBoom")["rotation_spread"],
    }

    lines = []
    lines.append("Does BEDI, or the isolated rotation signal, predict the 10Y Treasury return?")
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

        _, ex = outlier_robustness_check(target, pred, lines)
        summary_rows.append({"predictor": name, "test": "lag1_ex_outliers", "n_lags": 1,
                              "coef_or_F": ex.params["pred_lag1"], "p_value": ex.pvalues["pred_lag1"]})
        lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/bedi_treasury_test_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/bedi_treasury_test_summary.csv", index=False)
    print(f"Saved to {OUT}/bedi_treasury_test_results.txt and {OUT}/bedi_treasury_test_summary.csv")


if __name__ == "__main__":
    main()
