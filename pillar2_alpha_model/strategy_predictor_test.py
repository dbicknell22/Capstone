"""Does the Pillar 2 long/short strategy's OWN return correlate with the
Boomer-generation data and indices already built elsewhere in this project?

The static backtest in run_backtest.py tests one thing: is this specific
basket (long defensives, short growth) profitable, full stop, unconditional
on any signal. It says nothing about whether the Boomer thesis it's *based
on* actually shows up in the data. This module closes that gap by testing
the strategy's own quarterly return against three real, already-validated
predictors:

  1. Rotation signal (dfa_signals.rotation_signal) -- Boomer equity share
     minus safe-asset share. A raw DFA level, never z-scored over any
     window, so there's no look-ahead-bias question for this one.
  2. K-Index (k_index_model.k_index_builder.build_k_index) -- the same K
     tested against the 10Y Treasury. Uses K's only existing form, the
     full-sample z-score -- consistent with every other K regression in
     this project, but carrying the same mild look-ahead caveat they all
     do (no point-in-time version of K itself has been built).
  3. Boomer real-estate rotation (dfa_signals.real_estate_rotation, new) --
     real estate's share of Boomer assets, the most literal "are they
     selling homes" proxy. Nothing in this project has tested this before.

Same discipline as every other regression in this project:
contemporaneous and lagged tested as two SEPARATE regressions (never one
combined "current + 4 lags" model, which is exactly what produced false
positives for S&P 500 and GDP elsewhere in this repo), swept across lag
lengths 1-4 to catch a result that's only "significant" at one arbitrary
lag choice.

Run: python strategy_predictor_test.py
"""
import sys

import pandas as pd

from _pathutil import find_dir_containing
from dfa_signals import rotation_signal, real_estate_rotation
from factor_construction import build_long_short

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
K_INDEX_DIR = REPO_ROOT / "k_index_model"
if str(K_INDEX_DIR) not in sys.path:
    sys.path.insert(0, str(K_INDEX_DIR))
from k_index_builder import build_k_index          # noqa: E402
from regressions import contemporaneous_and_lagged_test  # noqa: E402

OUT = "output"


def quarterly_long_short_return(start="1999-01-01", end=None) -> pd.Series:
    """Compounds the strategy's monthly long_short return (long defensives
    minus short growth, from factor_construction.build_long_short) into a
    quarterly return, to match the quarterly frequency every DFA/K
    predictor is measured at."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    ls = build_long_short(start=start, end=end)["long_short"]
    q = ls.resample("QE").apply(lambda s: (1 + s).prod() - 1)
    return q.rename("long_short_qtr_return")


def main():
    target = quarterly_long_short_return()

    predictors = {
        "rotation_signal (Boomer equity - safe-asset share)": rotation_signal("BabyBoom")["rotation_spread"],
        "K-Index": build_k_index()[0]["K"],
        "real_estate_rotation (Boomer real estate share of assets)": real_estate_rotation("BabyBoom")["real_estate_share_pct"],
    }

    lines = []
    lines.append("Does the Pillar 2 long/short strategy's own quarterly return")
    lines.append("correlate with Boomer-generation data or the K-Index?")
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
    with open(f"{OUT}/strategy_predictor_test_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/strategy_predictor_test_summary.csv", index=False)
    print(f"Saved to {OUT}/strategy_predictor_test_results.txt and {OUT}/strategy_predictor_test_summary.csv")


if __name__ == "__main__":
    main()
