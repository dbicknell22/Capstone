"""Entry point for the advisor's two requests:
  1. Does K explain asset prices? (stock index, 10Y Treasury total return,
     USD/JPY, USD/EUR, USD/GBP, gold)
  2. Does K explain economic growth? (unemployment rate, GDP, industrial
     production)

Reports two views for each target, per a later round of advisor feedback
that frames the hypothesis as contemporaneous first ("widening -> stocks go
up", the mechanical wealth-effect channel) with lagged effects as an
addendum (Prof. Melvin) -- see mechanism_tests.py for the fuller writeup of
why these are kept separate rather than one combined table:
  - the combined regression:  y_t ~ const + K_t + K_{t-1..4}
  - contemporaneous only:     y_t ~ const + K_t
  - lagged only:              y_t ~ const + K_{t-1..4}

Run: python run_k_regressions.py

Requires:
  - The K-Index itself (k_index_builder.py) -- complete 3-pillar version
    (wealth + income + consumer), validated against the original notebook.
  - Every target series as a CSV in data_cache/ (see target_data.py) -- none
    of these can be fetched live from this sandbox. Fails with an
    instructive per-series error rather than a fabricated result if a file
    is missing.
"""
import pandas as pd

from k_index_builder import build_k_index
from target_data import load_pct_change, load_diff
from regressions import run_k_regression, contemporaneous_and_lagged_test, ASSET_PRICE_TARGETS, ECON_GROWTH_TARGETS

OUT = "output"


def _run_group(group_name, targets, k, results, log_lines):
    print(f"=== {group_name} ===\n")
    for name, transform in targets.items():
        series_name = name.replace("_pct_chg", "").replace("_diff", "")
        try:
            target = load_pct_change(series_name) if transform == "pct_change" else load_diff(series_name)
        except (RuntimeError, ValueError) as e:
            print(f"[SKIPPED] {name}: {e}\n")
            continue

        combined, df = run_k_regression(target, k)
        contemp, lagged = contemporaneous_and_lagged_test(target, k)
        results[name] = {"combined": combined, "contemporaneous": contemp, "lagged": lagged}

        print(f"--- {name} (n={len(df)}) ---")
        print("Combined: y_t ~ const + K_t + K_(t-1..4)")
        print(combined.summary().tables[1])
        print(f"Contemporaneous only: K coef={contemp.params['K']:.4f}, p={contemp.pvalues['K']:.4f}")
        print(f"Lagged only: joint F={lagged.fvalue:.3f}, p={lagged.f_pvalue:.4f}")
        print()

        log_lines.append(f"=== {name} ===\n--- Combined ---\n{combined.summary()}\n\n"
                          f"--- Contemporaneous only ---\n{contemp.summary()}\n\n"
                          f"--- Lagged only ---\n{lagged.summary()}\n\n")


def main():
    Z, pillars_used, _ = build_k_index()
    k = Z["K"]
    label = "3-pillar (wealth+income+consumer)" if len(pillars_used) == 3 else \
            f"{len(pillars_used)}-pillar ({'+'.join(pillars_used)}) -- consumer pillar not yet available"
    print(f"K-Index: {label}, {k.index.min().date()} -> {k.index.max().date()} ({len(k)} quarters)\n")

    results, log_lines = {}, []
    _run_group("1. Does K explain asset prices?", ASSET_PRICE_TARGETS, k, results, log_lines)
    _run_group("2. Does K explain economic growth?", ECON_GROWTH_TARGETS, k, results, log_lines)

    if not results:
        print("No target data available yet -- every regression was skipped. "
              "Add CSVs to data_cache/ per target_data.py and re-run.")
        return

    with open(f"{OUT}/k_regressions_summary.txt", "w") as f:
        f.write("\n".join(log_lines))
    print(f"Saved full regression output to {OUT}/k_regressions_summary.txt")


if __name__ == "__main__":
    main()
