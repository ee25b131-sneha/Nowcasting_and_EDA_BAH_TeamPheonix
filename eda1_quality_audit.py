"""
EDA Phase 1: Data Quality & Coverage Audit
============================================
Turns phase1_aligned.csv into PPT-ready assets:
  - coverage_stats.csv     : the completeness numbers as a table
  - dynamic_range.csv      : median/percentile/max per channel
  - challenges.csv         : the documented failures side-table
  - daily_coverage.png     : WHERE the gaps fall across 117 days
  - soft_counts_hist.png   : log-scale histogram (baseline + flare tail)

Run:
    cd D:\Team_Pheonix_PS15
    python scripts\eda1_quality_audit.py

All outputs land in D:\Team_Pheonix_PS15\eda\
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save plots to file, no popup window needed
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(r"D:\Team_Pheonix_PS15")
ALIGNED = BASE_DIR / "outputs" / "phase1_aligned.csv"
EDA_DIR = BASE_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ---------------------------------------------------------------
    # Load. This is ~10M rows so it takes a few seconds. We parse
    # 'time' as a real datetime so we can group by calendar day later.
    # ---------------------------------------------------------------
    print("Loading phase1_aligned.csv (~10M rows, give it a moment)...")
    df = pd.read_csv(ALIGNED, parse_dates=["time"])
    n = len(df)
    print(f"  {n:,} rows loaded")

    # ---------------------------------------------------------------
    # 1. COVERAGE STATS TABLE
    # Just the headline completeness numbers, saved as a CSV you can
    # drop into the PPT or screenshot. mean() of a True/False column
    # gives the fraction that are True, so soft_missing.mean() is the
    # missing fraction directly.
    # ---------------------------------------------------------------
    soft_missing = df["soft_missing"].mean()
    hard_missing = df["hard_missing"].mean()
    both_missing = (df["soft_missing"] & df["hard_missing"]).mean()

    coverage = pd.DataFrame({
        "metric": [
            "Total seconds", "Date range start", "Date range end",
            "Soft (SoLEXS) completeness %", "Hard (HEL1OS) completeness %",
            "Both-blind %", "Either-instrument coverage %",
        ],
        "value": [
            f"{n:,}", str(df['time'].min()), str(df['time'].max()),
            f"{100*(1-soft_missing):.2f}", f"{100*(1-hard_missing):.2f}",
            f"{100*both_missing:.2f}", f"{100*(1-both_missing):.2f}",
        ],
    })
    coverage.to_csv(EDA_DIR / "coverage_stats.csv", index=False)
    print("\nCoverage stats:")
    print(coverage.to_string(index=False))

    # ---------------------------------------------------------------
    # 2. DYNAMIC RANGE TABLE
    # Shows the quiet baseline vs flare peak for each channel. The gap
    # between the median and the max is the evidence that real flares
    # are in the data. We use .dropna() so missing seconds don't count.
    # Percentiles (95th, 99th) show how rare the high values are.
    # ---------------------------------------------------------------
    rows = []
    for col in ["counts", "hardness", "cdte_broad", "cdte_hardness",
                "czt_broad", "czt_hardness"]:
        s = df[col].dropna()
        rows.append({
            "channel": col,
            "median": round(s.median(), 2),
            "mean": round(s.mean(), 2),
            "p95": round(s.quantile(0.95), 2),
            "p99": round(s.quantile(0.99), 2),
            "max": round(s.max(), 2),
        })
    dyn = pd.DataFrame(rows)
    dyn.to_csv(EDA_DIR / "dynamic_range.csv", index=False)
    print("\nDynamic range:")
    print(dyn.to_string(index=False))

    # ---------------------------------------------------------------
    # 3. CHALLENGES TABLE (the side-table for your PPT)
    # ---------------------------------------------------------------
    challenges = pd.DataFrame([
        {"challenge": "HEL1OS cadence ~1.6 Hz not 1 Hz",
         "impact": "Naive rounding averages photon counts, suppresses flare spikes",
         "resolution": "Resample to 1 Hz: SUM counts, MEAN hardness"},
        {"challenge": "259,041 duplicate rows in HEL1OS",
         "impact": "Inflated data, double-count risk",
         "resolution": "Verified identical, dropped safely"},
        {"challenge": "SoLEXS gaps as absent rows not flags",
         "impact": "Row-order merge misaligns data after each gap",
         "resolution": "Reindex onto synthetic grid + missing flags"},
        {"challenge": "Empty resample bins read as 0 counts",
         "impact": "Fake zeros bias background estimate low",
         "resolution": "Flag zero-reading bins as missing"},
    ])
    challenges.to_csv(EDA_DIR / "challenges.csv", index=False)
    print("\nChallenges table saved.")

    # ---------------------------------------------------------------
    # 4. DAILY COVERAGE TIMELINE PLOT
    # We can't plot 10M points. So we group by calendar day and compute
    # each day's missing fraction. That's 117 points - plottable, and
    # it shows WHERE gaps cluster (e.g. the multi-day SoLEXS dropouts
    # will show as days hitting 100% missing).
    # ---------------------------------------------------------------
    print("\nBuilding daily coverage plot...")
    df["date"] = df["time"].dt.date
    daily = df.groupby("date").agg(
        soft_missing=("soft_missing", "mean"),
        hard_missing=("hard_missing", "mean"),
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, 100 * daily["soft_missing"],
            label="SoLEXS (soft) missing %", linewidth=1.2)
    ax.plot(daily.index, 100 * daily["hard_missing"],
            label="HEL1OS (hard) missing %", linewidth=1.2)
    ax.set_ylabel("% of day missing")
    ax.set_xlabel("Date")
    ax.set_title("Daily data completeness across the mission")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(EDA_DIR / "daily_coverage.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 5. SOFT COUNTS HISTOGRAM (log scale)
    # A log-y histogram of soft counts shows two things at once: a big
    # cluster at low values (quiet Sun baseline) and a long thin tail
    # stretching to ~15,000 (flares). This single plot visually proves
    # "most of the time quiet, occasionally huge" - the flare signature.
    # We sample 1M points to keep it fast; the shape is identical.
    # ---------------------------------------------------------------
    print("Building soft-counts histogram...")
    s = df["counts"].dropna()
    if len(s) > 1_000_000:
        s = s.sample(1_000_000, random_state=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins=200, log=True)      # log y-axis: see the rare tail
    ax.set_xlabel("Soft X-ray counts (per second)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title("Soft counts distribution: quiet baseline + flare tail")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "soft_counts_hist.png", dpi=150)
    plt.close(fig)

    print(f"\nAll EDA Phase 1 assets saved to: {EDA_DIR}")


if __name__ == "__main__":
    main()
