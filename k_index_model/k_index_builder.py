"""Rebuilds Pillar 1's K-Index from the original source files, port of
KIndex_Complete_1.ipynb (Team Zeta, Nathan Garderes) into the repo so it can
be regressed against market/macro data.

Three pillars, each a top-minus-bottom spread, z-scored and averaged:
  - Wealth:   log(top-10% / bottom-50% net worth per household), Fed DFA
  - Income:   top-quartile - bottom-quartile wage growth, Atlanta Fed Wage
              Growth Tracker
  - Consumer: top-tercile - bottom-tercile consumer sentiment, U. Michigan
              Surveys of Consumers

Wealth and income are validated here against the notebook's own printed
output and match exactly:
  wealth  2025:Q3/Q4/2026:Q1 -> 139.0827 / 139.305421 / 138.860484 (match)
  income  2025:Q4/2026:Q1/Q2 -> 0.600000 / 0.366667 / 0.200000 (match)

The consumer pillar needs `bluebk02n.xls` (U. Michigan sentiment by income
tercile), which isn't in the repo yet. If it's absent, `build_k_index()`
builds a clearly-labeled 2-pillar (wealth + income) version instead of
silently guessing at the third — every output that depends on it says
"2-pillar" rather than presenting a partial index as the real thing.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from _pathutil import find_dir_containing

ROOT = find_dir_containing("dfa-networth-levels-detail.csv")
MICH_FILE = ROOT / "bluebk02n.xls"


def _dfa_quarter(q):
    y, qq = str(q).split(":Q")
    return pd.Period(f"{y}Q{qq}", freq="Q")


def _to_qend(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period("Q").to_timestamp("Q")
    return s.groupby(s.index).mean()


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def build_wealth_pillar() -> pd.Series:
    det = pd.read_csv(ROOT / "dfa-networth-levels-detail.csv")
    det["t"] = det["Date"].map(_dfa_quarter)
    nw = det.pivot_table(index="t", columns="Category", values="Net worth", aggfunc="sum")
    hh = det.pivot_table(index="t", columns="Category", values="Household count", aggfunc="sum")
    top, bot = ["TopPt1", "RemainingTop1", "Next9"], ["Bottom50"]
    wealth = (nw[top].sum(axis=1) / hh[top].sum(axis=1)) / (nw[bot].sum(axis=1) / hh[bot].sum(axis=1))
    wealth.index = wealth.index.to_timestamp(how="end").normalize()
    return _to_qend(wealth.rename("wealth"))


def build_income_pillar() -> pd.Series:
    w = pd.read_excel(ROOT / "wage-growth-data.xlsx", sheet_name="Average Wage Quartile", header=None).iloc[3:].copy()
    w.columns = ["date", "q1", "q2", "q3", "q4", "overall", "low_half", "up_half"]
    w["date"] = pd.to_datetime(w["date"])
    for c in w.columns[1:]:
        w[c] = pd.to_numeric(w[c].replace(".", np.nan), errors="coerce")
    w = w.set_index("date").dropna(subset=["q1", "q4"], how="all")
    return _to_qend((w["q4"] - w["q1"]).resample("QE").mean().rename("income"))


def build_consumer_pillar() -> pd.Series:
    """Returns None if bluebk02n.xls hasn't been added to the repo yet."""
    if not MICH_FILE.exists():
        return None
    m = pd.read_excel(MICH_FILE, sheet_name=0, header=None).iloc[8:].copy()
    m.columns = ["qlabel", "year", "sent_bot", "sent_mid", "sent_top",
                 "cur_bot", "cur_mid", "cur_top", "exp_bot", "exp_mid", "exp_top"]
    m = m[m["year"].notna()].copy()
    qmap = {"Jan.-Mar.": 1, "Apr.-Jun.": 2, "Jul.-Sep.": 3, "Oct.-Dec.": 4}
    m["q"] = m["qlabel"].astype(str).str.strip().map(qmap)
    m = m[m["q"].notna()].copy()
    m["t"] = [pd.Period(f"{int(y)}Q{int(q)}", freq="Q").to_timestamp("Q")
              for y, q in zip(m["year"], m["q"])]
    m = m.set_index("t")
    m["sent_bot"] = pd.to_numeric(m["sent_bot"], errors="coerce")
    m["sent_top"] = pd.to_numeric(m["sent_top"], errors="coerce")
    return _to_qend((m["sent_top"] - m["sent_bot"]).rename("consumer"))


def _first_pc(X: np.ndarray):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, 0] * S[0]
    loadings = Vt[0]
    if loadings[0] < 0:  # sign-align so the wealth pillar loads positive, like the notebook
        scores, loadings = -scores, -loadings
    return scores, loadings


def build_k_index():
    """Returns (Z, pillars_used) where Z has z_<pillar> columns, K
    (equal-weight), K_pca, and K_invvar. pillars_used is ["wealth","income"]
    or ["wealth","income","consumer"] depending on whether the Michigan file
    is present."""
    wealth = build_wealth_pillar()
    income = build_income_pillar()
    consumer = build_consumer_pillar()

    series = {"wealth": wealth, "income": income}
    if consumer is not None:
        series["consumer"] = consumer
    pillars_used = list(series.keys())

    common = pd.concat(series.values(), axis=1, sort=True)
    common.columns = pillars_used
    common = common.dropna().copy()
    common["wealth"] = np.log(common["wealth"])

    Z = common.apply(_zscore)
    Z.columns = [f"z_{p}" for p in pillars_used]
    Z["K"] = Z[[f"z_{p}" for p in pillars_used]].mean(axis=1)

    scores, loadings = _first_pc(Z[[f"z_{p}" for p in pillars_used]].values)
    # the notebook z-scores the raw SVD projection before using it as K_pca
    # (`Z["K_pca"] = zscore(pd.Series(M @ pc1, ...))`) -- match that exactly,
    # otherwise K_pca ends up on the SVD's own scale instead of a comparable
    # z-score (caught by diffing this module's kindex.csv output against a
    # notebook rebuild of the same recipe -- corr(K, K_pca) is scale-invariant
    # so it validated fine even with this missing, but the raw values didn't).
    Z["K_pca"] = _zscore(pd.Series(scores, index=Z.index))
    if Z["K_pca"].corr(Z["K"]) < 0:
        Z["K_pca"] = -Z["K_pca"]

    v = common.var(ddof=0)
    iv = (1 / v) / (1 / v).sum()
    Z["K_invvar"] = sum(iv[p] * Z[f"z_{p}"] for p in pillars_used)

    return Z, pillars_used, dict(zip(pillars_used, loadings))


def build_k_index_expanding(min_periods: int = 20) -> pd.DataFrame:
    """Point-in-time version of K (equal-weight only): each quarter's
    z-score uses only the mean/std of quarters up to and including it
    (expanding window), the same discipline as
    bedi_index.build_bedi_expanding() in pillar2_alpha_model. Every K
    regression elsewhere in this project uses build_k_index()'s full-sample
    z-score instead and carries a documented mild look-ahead caveat because
    of it -- fine for a regression coefficient's statistical significance,
    but a real backtest is making capital-allocation decisions with the
    signal, so it needs the no-look-ahead version. `min_periods=20` (5
    years of quarterly data) avoids unstable z-scores on the first few
    quarters' near-zero sample variance."""
    wealth = build_wealth_pillar()
    income = build_income_pillar()
    consumer = build_consumer_pillar()

    series = {"wealth": wealth, "income": income}
    if consumer is not None:
        series["consumer"] = consumer
    pillars_used = list(series.keys())

    common = pd.concat(series.values(), axis=1, sort=True)
    common.columns = pillars_used
    common = common.dropna().copy()
    common["wealth"] = np.log(common["wealth"])

    Z = pd.DataFrame(index=common.index)
    for p in pillars_used:
        exp_mean = common[p].expanding(min_periods=min_periods).mean()
        exp_std = common[p].expanding(min_periods=min_periods).std()
        Z[f"z_{p}"] = (common[p] - exp_mean) / exp_std
    Z = Z.dropna()
    Z["K"] = Z[[f"z_{p}" for p in pillars_used]].mean(axis=1)
    return Z


if __name__ == "__main__":
    Z, pillars_used, loadings = build_k_index()
    print(f"Pillars used: {pillars_used}" + ("" if len(pillars_used) == 3 else
          " -- 2-PILLAR VERSION, missing consumer (bluebk02n.xls not in repo yet)"))
    print(f"Common window: {Z.index.min().date()} -> {Z.index.max().date()} ({len(Z)} quarters)\n")
    print(Z.tail(5))
    print(f"\nPCA loadings: {loadings}")
    print(f"corr(K, K_pca) = {Z['K'].corr(Z['K_pca']):.3f}, "
          f"corr(K, K_invvar) = {Z['K'].corr(Z['K_invvar']):.3f}")
