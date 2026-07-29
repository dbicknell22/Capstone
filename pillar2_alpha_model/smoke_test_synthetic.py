"""SYNTHETIC DATA ONLY.

This validates that the pipeline (basket construction, performance stats,
HAC-robust factor regression) runs end-to-end and can recover a known,
deliberately-injected alpha from fabricated returns. It produces NO
information about real markets, real ETFs, or the real strategy, and must
never be cited as a backtest result — it is a unit test for the code.

Run: python smoke_test_synthetic.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import perf_stats

RNG = np.random.default_rng(7)
N_MONTHS = 300               # 25 years of fabricated monthly data
TRUE_ALPHA_ANNUAL = 0.02     # inject a known 2%/yr alpha to check recovery


def fake_factors(n):
    idx = pd.date_range("2001-01-31", periods=n, freq="ME")
    return pd.DataFrame({
        "Mkt-RF": RNG.normal(0.006, 0.045, n),
        "SMB":    RNG.normal(0.001, 0.020, n),
        "HML":    RNG.normal(0.000, 0.020, n),
        "RMW":    RNG.normal(0.002, 0.015, n),
        "CMA":    RNG.normal(0.000, 0.015, n),
        "UMD":    RNG.normal(0.004, 0.030, n),
        "RF":     np.full(n, 0.0015),
    }, index=idx)


def fake_long_short(factors: pd.DataFrame) -> pd.Series:
    betas = {"Mkt-RF": 0.1, "SMB": -0.2, "HML": 0.3, "RMW": 0.2, "CMA": 0.1, "UMD": -0.15}
    exposure = sum(factors[k] * b for k, b in betas.items())
    noise = RNG.normal(0, 0.02, len(factors))
    monthly_alpha = TRUE_ALPHA_ANNUAL / 12
    return exposure + monthly_alpha + noise


def main():
    factors = fake_factors(N_MONTHS)
    ls = fake_long_short(factors)

    print("=== SYNTHETIC SMOKE TEST — NOT REAL MARKET DATA ===\n")
    print("Performance stats on fabricated long-short series:")
    print(pd.Series(perf_stats(ls, factors["RF"])), "\n")

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    X = sm.add_constant(factors[factor_cols])
    model = sm.OLS(ls, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    recovered_annual_alpha = model.params["const"] * 12

    print(f"True injected annual alpha:      {TRUE_ALPHA_ANNUAL:.2%}")
    print(f"Recovered annual alpha (OLS):    {recovered_annual_alpha:.2%}")
    print(f"Alpha t-stat:                    {model.tvalues['const']:.2f}")
    print("\nIf recovered alpha is close to the injected 2% with a significant "
          "t-stat, the regression/backtest pipeline is working correctly.")

    cum = (1 + ls).cumprod()
    plt.figure(figsize=(9, 5))
    plt.plot(cum.index, cum.values)
    plt.title("SYNTHETIC DATA — pipeline smoke test only (not a real backtest)")
    plt.ylabel("Growth of $1 (fabricated)")
    plt.tight_layout()
    plt.savefig("output/SYNTHETIC_smoke_test.png", dpi=150)


if __name__ == "__main__":
    main()
