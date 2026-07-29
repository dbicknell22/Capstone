"""Builds the Boomer Equity Displacement Index (BEDI): a single composite
combining the BabyBoom rotation signal (dfa_signals.rotation_signal) and the
K-shape wealth-concentration gap (dfa_signals.k_shape_intensity), z-scored
onto a common scale so they're comparable and combinable.

BEDI rises when Boomers are de-risking (falling rotation_spread) AND
concentration is widening (rising k_shape_gap) at the same time — the
combined-signal thesis is that this joint condition is more meaningful than
either series alone.

Two versions of BEDI are built, for two different purposes:

  - `build_bedi_full_sample()` — z-scores against the FULL 1989:Q3-2026:Q1
    mean/std. Fine for a descriptive chart of the whole history, but DO NOT
    use this for the predictive regression: at any historical quarter, a
    full-sample z-score uses information from quarters that hadn't happened
    yet, which would make any "predictive" result partly look-ahead-biased.
  - `build_bedi_expanding()` — z-scores (and, for the PCA composite, even the
    PCA loadings themselves) using only data up to and including each
    quarter (point-in-time, no look-ahead). This is the version
    bedi_forward_return_test.py regresses against forward returns.

Both equal-weight and PCA-weight composites are built for each, mirroring
Pillar 1's equal-weight-vs-PCA robustness check (which reported a 0.96
correlation between the two methods there). PCA is implemented directly via
SVD (a couple of lines, for 2 series) rather than pulling in scikit-learn as
a new dependency for one principal component.
"""
import numpy as np
import pandas as pd

from dfa_signals import rotation_signal, k_shape_intensity

# rotation_spread falls as Boomers de-risk; BEDI should RISE on de-risking,
# so the rotation component is sign-flipped before combining.
COMPONENT_SIGNS = {"rotation_spread": -1.0, "k_shape_gap": 1.0}


def _raw_components() -> pd.DataFrame:
    rot = rotation_signal("BabyBoom")[["rotation_spread"]]
    k = k_shape_intensity()[["k_shape_gap"]]
    return rot.join(k, how="inner").sort_index()


def _first_pc(X: np.ndarray):
    """First principal component of an (n_obs, n_features) matrix via SVD.
    Returns (scores, loadings, explained_variance_ratio)."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, 0] * S[0]
    loadings = Vt[0]
    evr = (S[0] ** 2) / (S ** 2).sum()
    return scores, loadings, evr


def build_bedi_full_sample():
    raw = _raw_components()
    z = pd.DataFrame({
        col: COMPONENT_SIGNS[col] * (raw[col] - raw[col].mean()) / raw[col].std()
        for col in raw.columns
    })

    out = pd.DataFrame(index=raw.index)
    out["z_rotation"] = z["rotation_spread"]
    out["z_k_shape"] = z["k_shape_gap"]
    out["BEDI_equal_weight"] = z.mean(axis=1)

    scores, loadings, evr = _first_pc(z[["rotation_spread", "k_shape_gap"]].values)
    scores = pd.Series(scores, index=z.index)
    if scores.corr(out["BEDI_equal_weight"]) < 0:  # PCA sign is arbitrary; align to equal-weight
        scores, loadings = -scores, -loadings
    out["BEDI_pca"] = scores
    loadings = pd.Series(loadings, index=["rotation_spread", "k_shape_gap"])
    return out, loadings, evr


def build_bedi_expanding(min_periods: int = 20) -> pd.DataFrame:
    """Point-in-time version: each quarter's z-score uses only the mean/std
    of quarters up to and including it, and each quarter's PCA loadings are
    refit using only data up to and including it — nothing here uses future
    information, so it's safe to use as a predictor. `min_periods=20` (5
    years of quarterly data) avoids unstable z-scores/loadings on the first
    few quarters' near-zero sample variance."""
    raw = _raw_components()
    z = pd.DataFrame(index=raw.index)
    for col in raw.columns:
        exp_mean = raw[col].expanding(min_periods=min_periods).mean()
        exp_std = raw[col].expanding(min_periods=min_periods).std()
        z[col] = COMPONENT_SIGNS[col] * (raw[col] - exp_mean) / exp_std
    z = z.dropna()

    out = pd.DataFrame(index=z.index)
    out["z_rotation"] = z["rotation_spread"]
    out["z_k_shape"] = z["k_shape_gap"]
    out["BEDI_equal_weight"] = z.mean(axis=1)

    pca_vals = []
    for i in range(min_periods, len(z) + 1):
        window = z.iloc[:i][["rotation_spread", "k_shape_gap"]].values
        scores, _, _ = _first_pc(window)
        val = scores[-1]
        eq_window = out["BEDI_equal_weight"].iloc[:i]
        if pd.Series(scores, index=z.index[:i]).corr(eq_window) < 0:
            val = -val
        pca_vals.append(val)
    out["BEDI_pca"] = pd.Series(pca_vals, index=z.index[min_periods - 1:])

    return out.dropna()


if __name__ == "__main__":
    full, loadings, evr = build_bedi_full_sample()
    print("=== BEDI, full-sample z-score (descriptive only — has look-ahead) ===")
    print(full.tail(8))
    print(f"\nPCA loadings: {loadings.to_dict()}")
    print(f"PCA explained variance ratio: {evr:.3f}")
    print(f"corr(BEDI_equal_weight, BEDI_pca) = {full['BEDI_equal_weight'].corr(full['BEDI_pca']):.3f}")

    exp = build_bedi_expanding()
    print(f"\n=== BEDI, expanding/point-in-time (no look-ahead, {len(exp)} quarters) ===")
    print(exp.tail(8))
    print(f"corr(BEDI_equal_weight, BEDI_pca), expanding = "
          f"{exp['BEDI_equal_weight'].corr(exp['BEDI_pca']):.3f}")
