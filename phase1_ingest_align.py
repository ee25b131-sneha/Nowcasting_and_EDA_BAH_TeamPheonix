"""
PS15 Nowcasting - Phase 1: Ingest & Align
============================================
Reads combined_solexs_dataset.csv and combined_hel1os_dataset_full.csv,
fixes known issues (HEL1OS full-row duplicates, sub-second offset),
builds one continuous 1-second master time grid spanning both
instruments, reindexes both onto it, and writes out phase1_aligned.csv
with explicit per-instrument missing-data flags.

Run from PowerShell:
    cd D:\Team_Pheonix_PS15
    python scripts\phase1_ingest_align.py

Known data issues this script handles (validated against real data,
see conversation log for the diagnostic checks that found these):
  - SoLEXS: ~674,977 NaN seconds (GTI gaps) + 6 multi-day missing
    chunks (rows entirely absent, not NaN). Handled by reindexing
    onto a synthetic full-second grid.
  - HEL1OS: 518,082 full-row exact duplicates (confirmed
    full_dupe_count == time_dupe_count). Handled by drop_duplicates().
  - HEL1OS timestamps carry a fixed sub-second offset (e.g. .733,
    .534). Rounded to nearest second to align with SoLEXS's
    whole-second grid. This is a deliberate precision tradeoff -
    flagged in the sanity report below.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------
# Config - adjust paths here if your filenames differ
# ---------------------------------------------------------------
BASE_DIR = Path(r"D:\Team_Pheonix_PS15")
SOLEXS_PATH = BASE_DIR / "combined_solexs_dataset.csv"
HEL1OS_PATH = BASE_DIR / "combined_hel1os_dataset_full.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "phase1_aligned.csv"


def load_and_clean():
    print("Loading SoLEXS...")
    solexs = pd.read_csv(SOLEXS_PATH)
    solexs["time"] = pd.to_datetime(solexs["time"])
    print(f"  {len(solexs):,} rows loaded")

    print("Loading HEL1OS...")
    hel1os = pd.read_csv(HEL1OS_PATH)
    hel1os["time"] = pd.to_datetime(hel1os["time"])
    print(f"  {len(hel1os):,} rows loaded")

    # --- Dedup HEL1OS (confirmed: full-row dupes only, safe to drop) ---
    before = len(hel1os)
    hel1os = hel1os.drop_duplicates()
    print(f"  Dropped {before - len(hel1os):,} exact duplicate rows "
          f"({len(hel1os):,} remain)")

    # --- Round HEL1OS sub-second timestamps to nearest whole second ---
    # so it aligns with SoLEXS's whole-second grid. This nudges
    # timestamps by up to 0.5s - acceptable for second-cadence
    # nowcasting, but flag if sub-second Neupert-effect timing is
    # ever needed downstream.
    hel1os["time"] = hel1os["time"].dt.round("1s")

    dupe_after_round = hel1os.duplicated("time", keep=False).sum()
    if dupe_after_round > 0:
        print(f"  WARNING: rounding created {dupe_after_round:,} new "
              f"duplicate timestamps - averaging these rows")
        numeric_cols = hel1os.select_dtypes(include=[np.number]).columns
        hel1os = hel1os.groupby("time", as_index=False)[numeric_cols].mean()
    else:
        print("  No new duplicates introduced by rounding")

    return solexs, hel1os


def build_master_grid(solexs, hel1os):
    start = min(solexs["time"].min(), hel1os["time"].min())
    end = max(solexs["time"].max(), hel1os["time"].max())
    grid = pd.date_range(start=start, end=end, freq="1s")
    print(f"\nMaster grid: {start} to {end}")
    print(f"  {len(grid):,} seconds total")
    return grid


def align(solexs, hel1os, grid):
    solexs_idx = solexs.set_index("time").reindex(grid)
    hel1os_idx = hel1os.set_index("time").reindex(grid)

    combined = pd.DataFrame(index=grid)
    combined["counts"] = solexs_idx["counts"]
    combined["hardness"] = solexs_idx["hardness"]
    combined["cdte_broad"] = hel1os_idx["cdte_broad"]
    combined["cdte_hardness"] = hel1os_idx["cdte_hardness"]
    combined["czt_broad"] = hel1os_idx["czt_broad"]
    combined["czt_hardness"] = hel1os_idx["czt_hardness"]

    # Explicit gap flags - Phase 2's rolling background MUST skip
    # these rather than treat missing data as zero-count silence.
    combined["soft_missing"] = combined["counts"].isna()
    combined["hard_missing"] = combined["cdte_broad"].isna()

    combined.index.name = "time"
    combined = combined.reset_index()
    return combined


def sanity_report(combined):
    print("\n" + "=" * 50)
    print("PHASE 1 SANITY REPORT")
    print("=" * 50)
    n = len(combined)
    soft_missing = combined["soft_missing"].sum()
    hard_missing = combined["hard_missing"].sum()
    both_missing = (combined["soft_missing"] & combined["hard_missing"]).sum()

    print(f"Total rows (seconds):      {n:,}")
    print(f"Soft (SoLEXS) missing:     {soft_missing:,} "
          f"({100 * soft_missing / n:.2f}%)")
    print(f"Hard (HEL1OS) missing:     {hard_missing:,} "
          f"({100 * hard_missing / n:.2f}%)")
    print(f"Both missing at once:      {both_missing:,} "
          f"({100 * both_missing / n:.2f}%)")
    print(f"\nDate range: {combined['time'].min()} to "
          f"{combined['time'].max()}")
    print("=" * 50)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    solexs, hel1os = load_and_clean()
    grid = build_master_grid(solexs, hel1os)
    combined = align(solexs, hel1os, grid)
    sanity_report(combined)

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
