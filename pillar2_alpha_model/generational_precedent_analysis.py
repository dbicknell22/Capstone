"""Tests Artisan's "are we entering uncharted territory" question directly:
is Baby Boomers' wealth-concentration-to-population ratio, as they retire,
actually different from what the pre-Boomer generation's ratio looked like
when THEY were the dominant older cohort? Uses only data already in this
repo -- dfa-generation-levels-detail.csv -- no new sourcing needed.

Methodology: for each generation g at each quarter t,
  wealth_share(g,t)        = NetWorth(g,t) / sum_g NetWorth(g,t) * 100
  household_share(g,t)     = HouseholdCount(g,t) / sum_g HouseholdCount(g,t) * 100
  concentration_ratio(g,t) = wealth_share(g,t) / household_share(g,t)

A ratio of 1.0 means a generation holds exactly its proportional share of
wealth for its number of households; above 1.0 = overrepresented in wealth.

Important caveat on "Silent": the Fed's DFA "Silent" category is really
"everyone born before 1946" -- a fixed, only-shrinking cohort over the
sample (no new members ever enter it), not the narrower 1928-1945 academic
definition. In 1989 it spans ages ~44-100+; by 2026 only the oldest
survivors (~81+) remain. That's actually useful here: it traces one full
cohort's concentration ratio from late-career through deep retirement and
mortality-driven wealth transfer -- exactly the arc Boomers are now
entering, just observed a generation earlier.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _pathutil import find_dir_containing

ROOT = find_dir_containing("dfa-generation-levels-detail.csv")
OUT = "output"
GENERATIONS = ["Silent", "BabyBoom", "GenX", "Millennial"]
COLORS = {"Silent": "#6C757D", "BabyBoom": "#0B1F3A", "GenX": "#2E7D32", "Millennial": "#C8102E"}


def _parse_date(s: pd.Series) -> pd.Series:
    return pd.PeriodIndex(s.str.replace(":", "-"), freq="Q").to_timestamp(how="end").normalize()


def build_concentration_ratios() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "dfa-generation-levels-detail.csv")
    df["Date"] = _parse_date(df["Date"])

    wealth = df.pivot(index="Date", columns="Category", values="Net worth")
    hh = df.pivot(index="Date", columns="Category", values="Household count")

    wealth_share = wealth.div(wealth.sum(axis=1), axis=0) * 100
    hh_share = hh.div(hh.sum(axis=1), axis=0) * 100
    ratio = wealth_share / hh_share

    out = pd.DataFrame(index=ratio.index)
    for g in GENERATIONS:
        out[f"{g}_wealth_share_pct"] = wealth_share[g]
        out[f"{g}_household_share_pct"] = hh_share[g]
        out[f"{g}_concentration_ratio"] = ratio[g]
    return out.sort_index()


def build_age_concentration_ratios() -> pd.DataFrame:
    """Same computation on the age cut instead -- a generation-boundary-free
    cross-check, since age brackets are a fixed 'slot' different
    generations pass through over time rather than a fixed birth cohort."""
    df = pd.read_csv(ROOT / "dfa-age-levels-detail.csv")
    df["Date"] = _parse_date(df["Date"])
    wealth = df.pivot(index="Date", columns="Category", values="Net worth")
    hh = df.pivot(index="Date", columns="Category", values="Household count")
    wealth_share = wealth.div(wealth.sum(axis=1), axis=0) * 100
    hh_share = hh.div(hh.sum(axis=1), axis=0) * 100
    ratio = (wealth_share / hh_share)["age70plus"].rename("age70plus_concentration_ratio")
    return ratio.to_frame()


def main():
    gen = build_concentration_ratios()
    age = build_age_concentration_ratios()

    print("=== Peak concentration ratio reached by each generation, and when ===")
    summary_rows = []
    for g in GENERATIONS:
        col = f"{g}_concentration_ratio"
        peak_val = gen[col].max()
        peak_date = gen[col].idxmax()
        latest_val = gen[col].dropna().iloc[-1]
        latest_date = gen[col].dropna().index[-1]
        print(f"{g:10s}  peak={peak_val:.2f}x on {peak_date.date()}   "
              f"latest={latest_val:.2f}x on {latest_date.date()}")
        summary_rows.append({"generation": g, "peak_ratio": peak_val, "peak_date": peak_date,
                              "latest_ratio": latest_val, "latest_date": latest_date})
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/generational_precedent_summary.csv", index=False)

    print("\n=== Direct comparison: Silent's peak vs. BabyBoom's current reading ===")
    silent_peak = gen["Silent_concentration_ratio"].max()
    boomer_latest = gen["BabyBoom_concentration_ratio"].dropna().iloc[-1]
    print(f"Silent Generation's peak concentration ratio (any point in 1989-2026): {silent_peak:.2f}x")
    print(f"BabyBoom's concentration ratio as of {gen.index[-1].date()}: {boomer_latest:.2f}x")
    print(f"BabyBoom is currently at {boomer_latest/silent_peak:.2f}x Silent's historical peak.")
    print("\nCaveat: Boomers (born 1946-1964) have only partly aged into retirement as of "
          "2026 -- the youngest won't turn 65 until 2029. This reading captures an early-to-"
          "mid stage of their transition, not its full arc, unlike Silent's fully-observed one.")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(1.0, color="#888", lw=0.8, ls="--", label="Proportional (1.0x)")
    for g in GENERATIONS:
        ax.plot(gen.index, gen[f"{g}_concentration_ratio"], color=COLORS[g], lw=2, label=g)
    ax.set_title("Wealth concentration ratio by generation (wealth share ÷ household share)",
                  color="#0B1F3A", weight="bold", fontsize=13)
    ax.set_ylabel("Concentration ratio (1.0 = proportional to household count)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.15)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/generational_precedent_ratio.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(1.0, color="#888", lw=0.8, ls="--")
    ax.plot(age.index, age["age70plus_concentration_ratio"], color="#0B1F3A", lw=2)
    ax.set_title("Age-70-plus households' concentration ratio over time (generation-agnostic cross-check)",
                  color="#0B1F3A", weight="bold", fontsize=13)
    ax.set_ylabel("Concentration ratio")
    ax.grid(alpha=0.15)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/generational_precedent_age70plus.png", dpi=150)
    plt.close()

    gen.to_csv(f"{OUT}/generational_precedent_ratios.csv")
    print(f"\nSaved charts and CSVs to {OUT}/")


if __name__ == "__main__":
    main()
