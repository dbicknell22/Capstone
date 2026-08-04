"""Tests the actual mechanism the advisor described, rather than just
asking "does K predict returns":

  "I would expect to see widening - stocks go up, tightening - stocks go
   down. Widening - consumer credit worsens, tightening - consumer credit
   improves... the effects on assets might be lagged. No relationship is
   incredibly useful (Buffett wouldn't expect a relationship!)."

Two things distinguish this from the run_k_regressions.py request:

  1. The hypothesis is CONTEMPORANEOUS first ("widening -> stocks go up" is
     framed as happening together, via the mechanical channel: equity
     rallies concentrate gains among the top-wealth holders who own most of
     the equities, mechanically pushing K up in the same quarter), with
     LAGGED effects as an addendum (Prof. Melvin). Both are reported and
     labeled separately below, not lumped into one table where the
     contemporaneous coefficient could get lost among 4 lag coefficients.
  2. Consumer credit stress is testable RIGHT NOW with data already in this
     repo -- dfa-networth-levels-detail.csv has a real "Consumer credit"
     dollar column by wealth percentile, so Bottom 50%'s credit-to-assets
     leverage doesn't need any of the blocked external data.

Stock prices themselves are still blocked (same Yahoo Finance/FRED wall as
everywhere else in this repo) -- run_k_regressions.py remains the real test
of that once market data exists. In the meantime, this module uses the
DFA-derived aggregate household equity-holdings growth
(pillar2_alpha_model/dfa_signals.py's aggregate_equity_growth, reproduced
here to keep this module self-contained) as a real, data-grounded proxy: it
conflates price return with net contribution/withdrawal flows, so treat it
as directional, not a clean total-return series -- same caveat as when it
was first built.
"""
import pandas as pd

from _pathutil import find_dir_containing
from k_index_builder import build_k_index
from regressions import contemporaneous_and_lagged_test

ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
EQUITY_COL = "Corporate equities and mutual fund shares"


def _parse_date(s: pd.Series) -> pd.Series:
    return pd.PeriodIndex(s.str.replace(":", "-"), freq="Q").to_timestamp(how="end").normalize()


def consumer_credit_stress() -> pd.Series:
    """Bottom 50%'s consumer credit as a share of their own assets (a
    leverage ratio -- rising = more debt-financed, i.e. "worsening" in the
    advisor's framing), and its QoQ change."""
    df = pd.read_csv(ROOT / "dfa-networth-levels-detail.csv")
    df["Date"] = _parse_date(df["Date"])
    bottom = df[df["Category"] == "Bottom50"].set_index("Date").sort_index()
    leverage = (bottom["Consumer credit"] / bottom["Assets"]).rename("bottom50_credit_leverage")
    return leverage.diff().rename("bottom50_leverage_qoq_chg"), leverage


def aggregate_equity_growth() -> pd.Series:
    df = pd.read_csv(ROOT / "dfa-generation-levels-detail.csv")
    df["Date"] = _parse_date(df["Date"])
    total = df.groupby("Date")[EQUITY_COL].sum().sort_index()
    return total.pct_change().rename("agg_equity_qoq_growth")


def _report(label: str, contemp_model, lag_model):
    print(f"=== {label} ===")
    print("-- Contemporaneous: y_t ~ const + K_t --")
    print(contemp_model.summary().tables[1])
    k_coef = contemp_model.params["K"]
    k_p = contemp_model.pvalues["K"]
    sig = "significant" if k_p < 0.05 else "not significant"
    print(f"K coefficient: {k_coef:.4f} (p={k_p:.4f}, {sig} at 5%)\n")

    print("-- Lagged (Melvin's addendum): y_t ~ const + K_(t-1..4) --")
    print(lag_model.summary().tables[1])
    print(f"Joint F-test on the 4 lags: F={lag_model.fvalue:.3f}, p={lag_model.f_pvalue:.4f}\n")


if __name__ == "__main__":
    Z, pillars_used, _ = build_k_index()
    k = Z["K"]
    print(f"K-Index: {len(pillars_used)}-pillar ({'+'.join(pillars_used)}), "
          f"{k.index.min().date()} -> {k.index.max().date()}\n")

    leverage_chg, leverage_level = consumer_credit_stress()
    contemp, lagged = contemporaneous_and_lagged_test(leverage_chg, k)
    _report("Widening K -> consumer credit worsens? (Bottom 50% credit/assets leverage, QoQ change)", contemp, lagged)

    eq_growth = aggregate_equity_growth()
    contemp, lagged = contemporaneous_and_lagged_test(eq_growth, k)
    _report("Widening K -> stocks go up? (DFA-derived aggregate equity growth proxy)", contemp, lagged)
