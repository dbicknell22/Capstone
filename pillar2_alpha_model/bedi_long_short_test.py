"""Does BEDI (the combined index, not just the isolated rotation signal)
predict the Pillar 2 long/short strategy's own returns?

strategy_predictor_test.py already tested the isolated rotation signal,
K-Index, and real-estate rotation against this strategy's returns -- all
three came back null. BEDI itself (rotation signal + k-shape gap,
combined) was never tested against this specific strategy. This fills
that gap before deciding whether a BEDI-timed version of the long/short
trade is worth building.

Same discipline as every other regression in this project: contemporaneous
and lagged tested as two SEPARATE regressions, swept across lag lengths
1-4.

Run: python bedi_long_short_test.py
"""
import sys

import pandas as pd

from _pathutil import find_dir_containing
from bedi_index import build_bedi_expanding
from strategy_predictor_test import quarterly_long_short_return

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
K_INDEX_DIR = REPO_ROOT / "k_index_model"
if str(K_INDEX_DIR) not in sys.path:
    sys.path.insert(0, str(K_INDEX_DIR))
from regressions import contemporaneous_and_lagged_test  # noqa: E402

OUT = "output"


def main():
    target = quarterly_long_short_return()
    bedi = build_bedi_expanding()["BEDI_equal_weight"]

    df = pd.concat([target, bedi.rename("BEDI")], axis=1, sort=True).dropna()

    lines = []
    lines.append("Does BEDI (combined) predict the long/short strategy's own quarterly return?")
    lines.append(f"N = {len(df)} quarters ({df.index.min().date()} -> {df.index.max().date()})")
    lines.append("")

    summary_rows = []
    contemp, _ = contemporaneous_and_lagged_test(target, bedi, n_lags=1)
    contemp_coef, contemp_p = contemp.params["K"], contemp.pvalues["K"]
    lines.append(f"contemporaneous: coef={contemp_coef:.4f}, p={contemp_p:.4f}")
    summary_rows.append({"test": "contemporaneous", "n_lags": 0, "coef_or_F": contemp_coef, "p_value": contemp_p})
    for n in [1, 2, 3, 4]:
        _, lagged = contemporaneous_and_lagged_test(target, bedi, n_lags=n)
        lines.append(f"n_lags={n}: joint F={lagged.fvalue:.3f}, p={lagged.f_pvalue:.4f}")
        summary_rows.append({"test": "lagged_joint_F", "n_lags": n, "coef_or_F": lagged.fvalue, "p_value": lagged.f_pvalue})

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/bedi_long_short_test_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/bedi_long_short_test_summary.csv", index=False)
    print(f"\nSaved to {OUT}/bedi_long_short_test_results.txt and {OUT}/bedi_long_short_test_summary.csv")


if __name__ == "__main__":
    main()
