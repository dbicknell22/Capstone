"""The regressions the advisor asked for:

  1. Asset prices ~ K:
       Pct_change(stock index)_t   ~ const + K_t + K_{t-1} + ... + K_{t-p}
       Pct_change(10Y Treasury TR)_t ~ const + K_t + ... + K_{t-p}
       Pct_change(USD/JPY, USD/EUR, USD/GBP)_t ~ const + K_t + ... + K_{t-p}
       Pct_change(gold)_t ~ const + K_t + ... + K_{t-p}

  2. Econ growth ~ K:
       Diff(unemployment rate)_t ~ const + K_t + ... + K_{t-p}
       Pct_change(GDP)_t         ~ const + K_t + ... + K_{t-p}
       Pct_change(IP)_t          ~ const + K_t + ... + K_{t-p}

Every regression includes the CURRENT value of K plus `n_lags` lagged
values (default 4 quarters), with Newey-West (HAC) standard errors for
autocorrelation. This directly tests whether K is contemporaneously
associated with these variables AND whether past K levels lead them --
the leading-indicator claim is the more interesting one for "does K predict
markets," and is carried entirely by the lagged K coefficients.
"""
import pandas as pd
import statsmodels.api as sm

N_LAGS_DEFAULT = 4


def _with_k_lags(k: pd.Series, n_lags: int) -> pd.DataFrame:
    df = pd.DataFrame({"K": k})
    cols = ["K"]
    for lag in range(1, n_lags + 1):
        col = f"K_lag{lag}"
        df[col] = df["K"].shift(lag)
        cols.append(col)
    return df[cols]


def run_k_regression(target: pd.Series, k: pd.Series, n_lags: int = N_LAGS_DEFAULT, hac_lags: int = 3):
    """target and k must already be the transformed (pct-change or diff)
    series and the raw K level respectively, both quarterly and indexed by
    quarter-end date. Current K and all lags in one combined regression --
    literal reading of the original brief ("include...the current value of
    K as well as lags"). See `contemporaneous_and_lagged_test` for the
    separated version, which keeps the contemporaneous ("widening -> stocks
    go up") and lagged (Melvin's addendum) hypotheses from competing for
    significance in the same table."""
    X = _with_k_lags(k, n_lags)
    df = pd.concat([target.rename("y"), X], axis=1).dropna()
    Xc = sm.add_constant(df.drop(columns="y"))
    model = sm.OLS(df["y"], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return model, df


def contemporaneous_and_lagged_test(target: pd.Series, k: pd.Series, n_lags: int = N_LAGS_DEFAULT, hac_lags: int = 3):
    """Two separate regressions instead of one combined table:
      contemporaneous: target_t ~ const + K_t
      lagged:          target_t ~ const + K_{t-1} + ... + K_{t-n_lags}
    Splitting them out matters here specifically: the advisor's hypothesis
    is contemporaneous first ("widening -> stocks go up", the mechanical
    wealth-effect channel), with a lagged effect as Prof. Melvin's addendum
    -- lumping both into one regression lets multicollinearity between K_t
    and its own lags obscure which one (if either) is actually doing the
    work, the same failure mode the BEDI structural-break test hit earlier
    in this project."""
    df = pd.DataFrame({"y": target, "K": k})
    for lag in range(1, n_lags + 1):
        df[f"K_lag{lag}"] = df["K"].shift(lag)

    contemp_df = df[["y", "K"]].dropna()
    contemp_X = sm.add_constant(contemp_df[["K"]])
    contemp_model = sm.OLS(contemp_df["y"], contemp_X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    lag_cols = [f"K_lag{lag}" for lag in range(1, n_lags + 1)]
    lag_df = df[["y"] + lag_cols].dropna()
    lag_X = sm.add_constant(lag_df[lag_cols])
    lag_model = sm.OLS(lag_df["y"], lag_X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    return contemp_model, lag_model


ASSET_PRICE_TARGETS = {
    "sp500_pct_chg": "pct_change",
    "treasury_10y_total_return_pct_chg": "pct_change",
    "usdjpy_pct_chg": "pct_change",
    "usdeur_pct_chg": "pct_change",
    "usdgbp_pct_chg": "pct_change",
    "gold_pct_chg": "pct_change",
}

ECON_GROWTH_TARGETS = {
    "unemployment_rate_diff": "diff",
    "gdp_pct_chg": "pct_change",
    "industrial_production_pct_chg": "pct_change",
}
