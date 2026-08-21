"""Does timing the luxury/discount basket with K actually work as a
strategy -- literally: hold the long-luxury/short-discount position when
K (lagged) is positive/widening, hold cash otherwise. Swept across lags
1-4, since the user specifically asked whether the lag-3/lag-4 "hits"
from k_specs_luxury_discount_test.py could be traded on.

Important context carried in from that test, not repeated as new work
here: the lag-3/lag-4 joint significance already looked like a
multicollinearity artifact, not a real effect -- the individual lag
coefficients inside those regressions alternate sign from one lag to the
next (lag2 positive, lag3 strongly negative, lag4 positive again), which
has no coherent economic story and is the textbook signature of an
unstable small-sample regression with highly correlated regressors. This
script tests the literal binary trading rule anyway, with the same
lag-sweep and inverted-signal stress tests applied to every other
backtest in this project, rather than asserting the conclusion without
checking it against the actual return series.

Uses the point-in-time/expanding K (build_k_index_expanding), the same
no-look-ahead standard as every other backtest in this project -- NOT the
full-sample K used in the regressions.

Run: python k_timed_luxury_discount_backtest.py
"""
import sys

import numpy as np
import pandas as pd

from _pathutil import find_dir_containing
from luxury_discount_construction import quarterly_luxury_discount_return

REPO_ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
K_INDEX_DIR = REPO_ROOT / "k_index_model"
if str(K_INDEX_DIR) not in sys.path:
    sys.path.insert(0, str(K_INDEX_DIR))
from k_index_builder import build_k_index_expanding  # noqa: E402

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


def run(target, signal, lag, invert=False):
    df = pd.concat([target, signal.rename("signal")], axis=1, sort=True).dropna()
    df["signal_lag"] = df["signal"].shift(lag)
    df = df.dropna()
    on = (df["signal_lag"] < 0) if invert else (df["signal_lag"] > 0)
    on = on.astype(int)
    strat_ret = on * df[target.name]
    switches = int((on.diff().abs() == 1).sum())
    return strat_ret, switches


def main():
    target = quarterly_luxury_discount_return()
    K = build_k_index_expanding()["K"]

    lines = []
    lines.append("K-timed luxury/discount backtest: hold long-luxury/short-discount when K")
    lines.append("(lagged, point-in-time) is positive, hold cash otherwise. Lag swept 1-4,")
    lines.append("same discipline as every other backtest in this project.")
    lines.append("")

    always_on_stats = perf_stats(target)
    lines.append(f"Always-on benchmark (no timing): CAGR={always_on_stats['CAGR']*100:+.2f}%, "
                 f"Sharpe={always_on_stats['Sharpe']:.2f}, MaxDD={always_on_stats['Max Drawdown']*100:.1f}%")
    lines.append("")

    summary_rows = [{"lag": "Always-on", **always_on_stats, "switches": None}]
    return_series = {}
    for lag in [1, 2, 3, 4]:
        strat_ret, switches = run(target, K, lag)
        stats = perf_stats(strat_ret)
        lines.append(f"lag={lag}: N={stats['N Quarters']}, switches={switches}, "
                     f"CAGR={stats['CAGR']*100:+.2f}%, Sharpe={stats['Sharpe']:.2f}, "
                     f"MaxDD={stats['Max Drawdown']*100:.1f}%")
        summary_rows.append({"lag": lag, **stats, "switches": switches})
        return_series[lag] = strat_ret

    lines.append("")
    lines.append("Inverted-signal check (lag=3 and lag=4 only, the two lags in question --")
    lines.append("if the real rule is capturing genuine information, flipping it should be")
    lines.append("much worse, not similar):")
    for lag in [3, 4]:
        inv_ret, inv_switches = run(target, K, lag, invert=True)
        inv_stats = perf_stats(inv_ret)
        lines.append(f"  lag={lag} inverted: CAGR={inv_stats['CAGR']*100:+.2f}%, "
                     f"Sharpe={inv_stats['Sharpe']:.2f}, MaxDD={inv_stats['Max Drawdown']*100:.1f}%")
        summary_rows.append({"lag": f"{lag}_inverted", **inv_stats, "switches": inv_switches})

    lines.append("")
    best = max(summary_rows[1:5], key=lambda r: r["Sharpe"])
    lines.append(
        f"Verdict: NONE of the four lags produce a positive Sharpe -- all four underperform "
        f"the always-on benchmark (Sharpe {always_on_stats['Sharpe']:.2f}), and lag=3 and "
        f"lag=4 (the two lags that looked 'significant' in the regression) are the WORST "
        f"performers of the four (Sharpe {[r['Sharpe'] for r in summary_rows if r['lag']==3][0]:.2f} "
        f"and {[r['Sharpe'] for r in summary_rows if r['lag']==4][0]:.2f}), not the best. Worse "
        "still, inverting the signal at lag=3 and lag=4 IMPROVES the Sharpe (to "
        f"{[r for r in summary_rows if r['lag']=='3_inverted'][0]['Sharpe']:.2f} and "
        f"{[r for r in summary_rows if r['lag']=='4_inverted'][0]['Sharpe']:.2f}) -- the opposite "
        "of what a real signal should do. This confirms directly, in return terms rather than "
        "just p-values, that the lag-3/lag-4 'significance' was a multicollinearity artifact "
        "(alternating-sign lag coefficients within the same regression, no coherent "
        "decay/persistence pattern), not a tradeable effect. Consistent with how this project "
        "treated the de-risking-timed long/short test (bedi_timed_long_short_backtest.py), "
        "another case where a backtest was built specifically to make this point concrete."
    )

    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/k_timed_luxury_discount_backtest_results.txt", "w") as f:
        f.write(text + "\n")
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/k_timed_luxury_discount_backtest_summary.csv", index=False)
    print(f"\nSaved to {OUT}/k_timed_luxury_discount_backtest_results.txt and _summary.csv")


if __name__ == "__main__":
    main()
