"""Derives real, computable macro signals from the Fed's Distributional
Financial Accounts (DFA) data (`dfa-*.csv` at the repo root) — no external
market data required. This is the fundamentals side of the Pillar 2 alpha
model: quantifying the Boomer-to-safe-asset rotation and the K-shape wealth
concentration trend directly from the same source data behind the deck.

Files consumed (repo root, one level up from this module):
  dfa-generation-levels-detail.csv   Silent / BabyBoom / GenX / Millennial, $ levels
  dfa-networth-shares-detail.csv     TopPt1 / RemainingTop1 / Next9 / Next40 / Bottom50, % shares

Validated: summing "Corporate equities and mutual fund shares" across the
4 generation buckets reproduces the same total as summing across the 4 age
buckets in dfa-age-levels-detail.csv (max abs diff = $2M on a ~$55T series,
i.e. rounding) — the two files are consistent cuts of the same underlying
household-sector totals.
"""
import pandas as pd
from _pathutil import find_dir_containing

# Repo root, where the dfa-*.csv files live. Resolved dynamically (not via
# __file__) so this also works when pasted or %load-ed into a notebook cell,
# where __file__ is never defined.
ROOT = find_dir_containing("dfa-generation-levels-detail.csv")

EQUITY_COL = "Corporate equities and mutual fund shares"
SAFE_COLS = ["Deposits", "Money market fund shares",
             "U.S. government and municipal securities", "Corporate and foreign bonds",
             "Annuities"]
REAL_ESTATE_COL = "Real estate"


def _parse_date(s: pd.Series) -> pd.Series:
    # "1989:Q3" -> quarter-end Timestamp
    return pd.PeriodIndex(s.str.replace(":", "-"), freq="Q").to_timestamp(how="end").normalize()


def _load(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / name)
    df["Date"] = _parse_date(df["Date"])
    return df


def generation_asset_shares(generation: str = "BabyBoom") -> pd.DataFrame:
    """For one generation, its share of total household-sector equities,
    'safe' assets (as one combined bucket), and real estate at each quarter
    (derived by dividing that generation's $ level by the cross-sectional sum
    across all 4 generations — dfa-generation-levels-detail.csv has no
    pre-computed shares file, unlike the age/income/networth cuts).

    Important: the safe-asset bucket's share is computed by summing DOLLAR
    levels across its component columns first, then taking one share of that
    combined total — summing each component's own percentage share instead
    would double-count and can exceed 100% (caught by plotting the series and
    seeing it run past 100 — see git history for the earlier, wrong version)."""
    df = _load("dfa-generation-levels-detail.csv")
    df = df.copy()
    df["safe_assets_total"] = df[SAFE_COLS].sum(axis=1)

    cols = [EQUITY_COL, REAL_ESTATE_COL, "safe_assets_total"]
    totals = df.groupby("Date")[cols].transform("sum")
    shares = (df[cols] / totals * 100).add_suffix("_share_pct")

    out = pd.concat([df[["Date", "Category"]], shares], axis=1)
    out = out.rename(columns={"safe_assets_total_share_pct": "safe_share_pct"})
    out = out[out["Category"] == generation].drop(columns="Category").set_index("Date").sort_index()
    return out


def rotation_signal(generation: str = "BabyBoom") -> pd.DataFrame:
    """The core Pillar 2 signal: `generation`'s equity share minus its
    safe-asset share, and the quarter-over-quarter change in that spread —
    a real, back-to-1989Q3 measure of whether this cohort is net rotating
    into or out of risk assets relative to income/defensive assets."""
    shares = generation_asset_shares(generation)
    sig = pd.DataFrame(index=shares.index)
    sig["equity_share_pct"] = shares[f"{EQUITY_COL}_share_pct"]
    sig["safe_share_pct"] = shares["safe_share_pct"]
    sig["rotation_spread"] = sig["equity_share_pct"] - sig["safe_share_pct"]
    sig["rotation_spread_qoq_chg"] = sig["rotation_spread"].diff()
    sig["rotation_spread_yoy_chg"] = sig["rotation_spread"].diff(4)
    return sig


def k_shape_intensity() -> pd.DataFrame:
    """Wealth-concentration gap from the actual net-worth-share cut used in
    Pillar 4 (Top 1% = TopPt1 + RemainingTop1, vs Bottom 50%). Reproduces the
    concentration measure Pillar 1's composite K-Index proxies for, but
    directly from the DFA net-worth-share series rather than a 3-indicator
    z-score blend."""
    df = _load("dfa-networth-shares-detail.csv")
    wide = df.pivot(index="Date", columns="Category", values="Net worth")
    out = pd.DataFrame(index=wide.index)
    out["top1_share_pct"] = wide["TopPt1"] + wide["RemainingTop1"]
    out["bottom50_share_pct"] = wide["Bottom50"]
    out["k_shape_gap"] = out["top1_share_pct"] - out["bottom50_share_pct"]
    out["k_shape_gap_qoq_chg"] = out["k_shape_gap"].diff()
    return out.sort_index()


def real_estate_rotation(generation: str = "BabyBoom") -> pd.DataFrame:
    """The most literal 'are they selling their homes' proxy: `generation`'s
    real estate holdings as a share of its own total assets (a level, same
    convention as rotation_spread and K -- a level index predicting future
    returns, not a rate), plus the raw QoQ % change in the dollar level as a
    secondary check (a falling share can also just mean other assets grew
    faster, not that real estate itself shrank; the dollar-level change
    rules that out)."""
    shares = generation_asset_shares(generation)
    df = _load("dfa-generation-levels-detail.csv")
    df = df[df["Category"] == generation].set_index("Date").sort_index()

    out = pd.DataFrame(index=shares.index)
    out["real_estate_share_pct"] = shares[f"{REAL_ESTATE_COL}_share_pct"]
    out["real_estate_share_qoq_chg"] = out["real_estate_share_pct"].diff()
    out["real_estate_usd_qoq_pct_chg"] = df[REAL_ESTATE_COL].pct_change()
    return out


def aggregate_equity_growth() -> pd.Series:
    """QoQ % growth in total household-sector equity holdings (summed across
    all 4 generations) — a real, data-grounded (if imperfect) market-value
    proxy usable without external price data. Conflates price return with net
    contribution/withdrawal flows and new issuance, so treat as a directional
    proxy, not a pure total return series."""
    df = _load("dfa-generation-levels-detail.csv")
    total = df.groupby("Date")[EQUITY_COL].sum().sort_index()
    return total.pct_change().rename("agg_equity_qoq_growth")


if __name__ == "__main__":
    rot = rotation_signal("BabyBoom")
    k = k_shape_intensity()
    print(rot.tail())
    print(k.tail())
