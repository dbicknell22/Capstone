"""Does the DIRECTION of K, BEDI, or the rotation signal (rising vs.
falling last period, not the level) predict equity returns?

Every equity test run so far (S&P 500, the long/short strategy, BEDI vs
XLV-XLY) used the LEVEL of K/BEDI/rotation signal. k_specification_test.py
found that, for Treasuries specifically, DIRECTION has a completely
different (contemporaneous, not lagged) relationship than level does.
Nobody has tested direction against any equity target yet -- this closes
that gap, against the two most relevant equity targets: the S&P 500
itself, and the long/short strategy's own return.

Same discipline as every other regression in this project: contemporaneous
and lagged tested as two SEPARATE regressions, swept across lag lengths 1-4.

Run: python direction_equity_test.py
"""
import sys

import pandas as pd

from _pathutil import find_dir_containing
from bedi_index import build_bedi_expanding
from dfa_signals import rotation_signal
from strategy_predictor_test import quarterly_long_short_return

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
K_INDEX_DIR = REPO_ROOT / "k_index_model"
if str(K_INDEX_DIR) not in sys.path:
    sys.path.insert(0, str(K_INDEX_DIR))
from k_index_builder import build_k_index          # noqa: E402
from regressions import contemporaneous_and_lagged_test  # noqa: E402

OUT = "output"


def direction(level: pd.Series) -> pd.Series:
    return (level.diff() > 0).astype(float)


def load_sp500_return() -> pd.Series:
    """Reads k_index_model/data_cache/sp500.csv directly, bypassing
    target_data.py's own load_pct_change() -- same cross-directory
    _pathutil collision as bedi_treasury_test.py's load_treasury_return()."""
    df = pd.read_csv(K_INDEX_DIR / "data_cache" / "sp500.csv",
                      parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Value"].resample("QE").last().pct_change().rename("sp500")


def main():
    K = build_k_index()[0]["K"]
    bedi = build_bedi_expanding()["BEDI_equal_weight"]
    rot = rotation_signal("BabyBoom")["rotation_spread"]

    signals = {
        "Direction of K": direction(K),
        "Direction of BEDI": direction(bedi),
        "Direction of rotation signal": direction(rot),
    }
    targets = {
        "S&P 500 (pct chg)": load_sp500_return(),
        "Long/short strategy (own return)": quarterly_long_short_return(),
    }

    lines = []
    lines.append("Does the DIRECTION (rising vs. falling) of K, BEDI, or the rotation")
    lines.append("signal predict equity returns -- the S&P 500, or the long/short")
    lines.append("strategy's own return? Methodology: contemporaneous_and_lagged_test(),")
    lines.append("swept across lag lengths 1-4, same discipline as every other")
    lines.append("regression in this project.")
    lines.append("")

    summary_rows = []
    for tname, target in targets.items():
        for sname, sig in signals.items():
            df = pd.concat([target, sig], axis=1, sort=True).dropna()
            lines.append(f"=== {sname} -> {tname} ===")
            lines.append(f"  N = {len(df)} quarters ({df.index.min().date()} -> {df.index.max().date()})")
            contemp, _ = contemporaneous_and_lagged_test(target, sig, n_lags=1)
            contemp_coef, contemp_p = contemp.params["K"], contemp.pvalues["K"]
            lines.append(f"  contemporaneous: coef={contemp_coef:.4f}, p={contemp_p:.4f}")
            summary_rows.append({"signal": sname, "target": tname, "test": "contemporaneous",
                                  "n_lags": 0, "coef_or_F": contemp_coef, "p_value": contemp_p})
            for n in [1, 2, 3, 4]:
                _, lagged = contemporaneous_and_lagged_test(target, sig, n_lags=n)
                lines.append(f"  n_lags={n}: joint F={lagged.fvalue:.3f}, p={lagged.f_pvalue:.4f}")
                summary_rows.append({"signal": sname, "target": tname, "test": "lagged_joint_F",
                                      "n_lags": n, "coef_or_F": lagged.fvalue, "p_value": lagged.f_pvalue})
            lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/direction_equity_test_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/direction_equity_test_summary.csv", index=False)
    print(f"Saved to {OUT}/direction_equity_test_results.txt and {OUT}/direction_equity_test_summary.csv")


if __name__ == "__main__":
    main()
