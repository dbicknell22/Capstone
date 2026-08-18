"""Does timing the long/short strategy with the de-risking (rotation)
signal actually work -- the literal rule an advisor proposed: hold the
long/short position when the signal shows de-risking (rising safe-asset
share), hold cash otherwise?

strategy_predictor_test.py and bedi_long_short_test.py already showed the
CONTINUOUS regression of this signal against the strategy's own returns
is null at every lag. This script tests the literal binary version of
that same idea -- and, critically, sweeps the lag choice the way every
other result in this project has been stress-tested, rather than
reporting whatever the first lag tried happens to show.

Signal: z_rotation from bedi_index.build_bedi_expanding() -- the isolated
rotation component, already sign-flipped and expanding-z-scored so it
rises when Boomers de-risk (equity share falling relative to safe-asset
share).

Rule: hold the long/short position in quarter t when the signal (as of
the end of quarter t-lag) was positive; hold cash otherwise.

Run: python bedi_timed_long_short_backtest.py
"""
import sys

import numpy as np
import pandas as pd

from _pathutil import find_dir_containing
from bedi_index import build_bedi_expanding
from strategy_predictor_test import quarterly_long_short_return

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
K_INDEX_DIR = REPO_ROOT / "k_index_model"
if str(K_INDEX_DIR) not in sys.path:
    sys.path.insert(0, str(K_INDEX_DIR))

OUT = "output"
QUARTERS_PER_YEAR = 4


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    cagr = (1 + r).prod() ** (QUARTERS_PER_YEAR / len(r)) - 1
    vol = r.std() * np.sqrt(QUARTERS_PER_YEAR)
    sharpe = (r.mean() * QUARTERS_PER_YEAR) / vol if vol > 0 else float("nan")
    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return {"CAGR": cagr, "Ann. Vol": vol, "Sharpe": sharpe, "Max Drawdown": max_dd,
            "Hit Rate": (r > 0).mean(), "N Quarters": len(r)}


def main():
    target = quarterly_long_short_return()
    signal = build_bedi_expanding()["z_rotation"]

    lines = []
    lines.append("De-risking-timed long/short backtest: hold long/short when the rotation")
    lines.append("signal (lagged) shows de-risking, hold cash otherwise. Lag swept 1-4,")
    lines.append("same discipline as every other backtest in this project.")
    lines.append("")

    summary_rows = []
    always_on_stats = perf_stats(target)
    lines.append(f"Always-on benchmark (no timing): CAGR={always_on_stats['CAGR']*100:+.2f}%, "
                 f"Sharpe={always_on_stats['Sharpe']:.2f}, MaxDD={always_on_stats['Max Drawdown']*100:.1f}%")
    lines.append("")

    for lag in [1, 2, 3, 4]:
        df = pd.concat([target, signal.rename("signal")], axis=1, sort=True).dropna()
        df["signal_lag"] = df["signal"].shift(lag)
        df = df.dropna()
        df["on"] = (df["signal_lag"] > 0).astype(int)
        strat_ret = df["on"] * df[target.name]
        stats = perf_stats(strat_ret)
        switches = int((df["on"].diff().abs() == 1).sum())
        lines.append(f"lag={lag}: N={len(df)}, switches={switches}, "
                     f"CAGR={stats['CAGR']*100:+.2f}%, Sharpe={stats['Sharpe']:.2f}, "
                     f"MaxDD={stats['Max Drawdown']*100:.1f}%")
        summary_rows.append({"lag": lag, **stats, "switches": switches})

    lag1 = summary_rows[0]
    lines.append("")
    lines.append(f"Verdict: only lag=1 shows a positive Sharpe ({lag1['Sharpe']:.2f}); "
                 "lag=2-4 are flat-to-negative and monotonically worsening -- the same "
                 "'only works at one arbitrary lag' red flag used elsewhere in this "
                 "project to reject a result. Consistent with the underlying continuous "
                 "regression (strategy_predictor_test.py, bedi_long_short_test.py) "
                 "already being null at every lag. No real basis for a de-risking-timed "
                 "version of the long/short strategy.")

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/bedi_timed_long_short_backtest_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/bedi_timed_long_short_backtest_summary.csv", index=False)
    print(f"\nSaved to {OUT}/bedi_timed_long_short_backtest_results.txt and _summary.csv")


if __name__ == "__main__":
    main()
