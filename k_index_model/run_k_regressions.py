"""Entry point for the advisor's two requests:
  1. Does K explain asset prices? (stock index, 10Y Treasury total return,
     USD/JPY, USD/EUR, USD/GBP, gold)
  2. Does K explain economic growth? (unemployment rate, GDP, industrial
     production)

Both as: variable_t ~ const + K_t + K_{t-1..4}, HAC standard errors.

Run: python run_k_regressions.py

Requires:
  - The K-Index itself (k_index_builder.py) -- currently a 2-pillar (wealth +
    income) version; upgrades automatically to the full 3-pillar version the
    moment bluebk02n.xls is added to the repo root, no code changes needed.
  - Every target series as a CSV in data_cache/ (see target_data.py) -- none
    of these can be fetched live from this sandbox. Fails with an
    instructive per-series error rather than a fabricated result if a file
    is missing.
"""
import pandas as pd

from k_index_builder import build_k_index
from target_data import load_pct_change, load_diff
from regressions import run_k_regression, ASSET_PRICE_TARGETS, ECON_GROWTH_TARGETS

OUT = "output"


def main():
    Z, pillars_used, _ = build_k_index()
    k = Z["K"]
    label = "3-pillar (wealth+income+consumer)" if len(pillars_used) == 3 else \
            f"{len(pillars_used)}-pillar ({'+'.join(pillars_used)}) -- consumer pillar not yet available"
    print(f"K-Index: {label}, {k.index.min().date()} -> {k.index.max().date()} ({len(k)} quarters)\n")

    results = {}
    print("=== 1. Does K explain asset prices? ===\n")
    for name, transform in ASSET_PRICE_TARGETS.items():
        series_name = name.replace("_pct_chg", "").replace("_diff", "")
        try:
            target = load_pct_change(series_name) if transform == "pct_change" else load_diff(series_name)
        except (RuntimeError, ValueError) as e:
            print(f"[SKIPPED] {name}: {e}\n")
            continue
        model, df = run_k_regression(target, k)
        results[name] = model
        print(f"--- {name} ~ K + 4 lags (n={len(df)}) ---")
        print(model.summary().tables[1])
        print()

    print("=== 2. Does K explain economic growth? ===\n")
    for name, transform in ECON_GROWTH_TARGETS.items():
        series_name = name.replace("_pct_chg", "").replace("_diff", "")
        try:
            target = load_pct_change(series_name) if transform == "pct_change" else load_diff(series_name)
        except (RuntimeError, ValueError) as e:
            print(f"[SKIPPED] {name}: {e}\n")
            continue
        model, df = run_k_regression(target, k)
        results[name] = model
        print(f"--- {name} ~ K + 4 lags (n={len(df)}) ---")
        print(model.summary().tables[1])
        print()

    if not results:
        print("No target data available yet -- every regression was skipped. "
              "Add CSVs to data_cache/ per target_data.py and re-run.")
        return

    with open(f"{OUT}/k_regressions_summary.txt", "w") as f:
        for name, model in results.items():
            f.write(f"=== {name} ===\n{model.summary()}\n\n")
    print(f"Saved full regression output to {OUT}/k_regressions_summary.txt")


if __name__ == "__main__":
    main()
