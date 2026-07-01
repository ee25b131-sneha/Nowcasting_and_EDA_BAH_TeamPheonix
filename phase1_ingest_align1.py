"""
PS15 Nowcasting - Phase 1: Ingest & Align (v2)
================================================
Reads combined_solexs_dataset.csv and combined_hel1os_dataset_full.csv,
fixes known data issues, downsamples HEL1OS from its native ~1.6 Hz
cadence to a clean 1 Hz, builds one continuous 1-second master grid
spanning both instruments, reindexes both onto it, and writes
phase1_aligned.csv with explicit per-instrument missing-data flags.

Run from PowerShell:
    cd D:\Team_Pheonix_PS15
    python scripts\phase1_ingest_align.py

------------------------------------------------------------------
Data issues handled (all validated against the real files - see
conversation log for the diagnostic checks that found each one):

  SoLEXS:
    - Already clean 1 Hz on whole seconds. Left untouched.
    - ~674,977 NaN seconds (GTI gaps) + 6 multi-day missing chunks
      (rows entirely absent). Surfaced explicitly by reindexing onto
      the full-second master grid + soft_missing flag.

  HEL1OS:
    - 259,041 full-row exact duplicates. Dropped (confirmed identical,
      safe).
    - Native cadence is ~1.6 Hz (NOT 1 Hz) - timestamps step ~0.5s,
      so ~2 readings land in most 1-second windows. Downsampled to
      1 Hz by resampling into 1-second bins:
          * broadband counts (cdte_broad, czt_broad) -> SUM
            (counts are additive: photons-per-second = sum of the
             sub-second readings in that second)
          * hardness ratios (cdte_hardness, czt_hardness) -> MEAN
            (ratios can't be summed; recompute-from-bands would be
             ideal but the CSV no longer carries the band counts, so
             mean is the accepted fallback. Documented limitation:
             slightly over-weights low-count sub-second readings.
             Negligible for nowcasting, where hardness is only a
             secondary confirmation signal.)
------------------------------------------------------------------
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

# Which HEL1OS columns are counts (summed) vs ratios (averaged)
HEL1OS_COUNT_COLS = ["cdte_broad", "czt_broad"]
HEL1OS_RATIO_COLS = ["cdte_hardness", "czt_hardness"]


def load_solexs():
    print("Loading SoLEXS...")
    solexs = pd.read_csv(SOLEXS_PATH)
    solexs["time"] = pd.to_datetime(solexs["time"])
    print(f"  {len(solexs):,} rows loaded (already clean 1 Hz, no binning)")
    return solexs


def load_and_downsample_hel1os():
    print("Loading HEL1OS...")
    hel1os = pd.read_csv(HEL1OS_PATH)
    hel1os["time"] = pd.to_datetime(hel1os["time"])
    print(f"  {len(hel1os):,} rows loaded")

    # Drop confirmed full-row duplicates
    before = len(hel1os)
    hel1os = hel1os.drop_duplicates()
    print(f"  Dropped {before - len(hel1os):,} exact duplicate rows "
          f"({len(hel1os):,} remain)")

    # Downsample ~1.6 Hz -> 1 Hz by 1-second resampling.
    # SUM the counts, MEAN the ratios (see module docstring for why).
    print("  Downsampling HEL1OS ~1.6 Hz -> 1 Hz "
          "(sum counts, mean hardness)...")
    agg_map = {c: "sum" for c in HEL1OS_COUNT_COLS}
    agg_map.update({c: "mean" for c in HEL1OS_RATIO_COLS})

    hel1os = (
        hel1os.set_index("time")
        .resample("1s")
        .agg(agg_map)
    )

    # resample() emits a row for EVERY 1-second bin in HEL1OS's span,
    # including empty bins (which become 0 for summed counts and NaN
    # for meaned ratios). A bin that had zero readings is a real gap,
    # not a real zero-count second - mark it so it doesn't masquerade
    # as observed silence. We detect empties via the ratio columns:
    # a truly-empty bin has NaN hardness (mean of nothing), whereas a
    # bin with readings has a finite (or 0/0->NaN... handled below)
    # value. Cleanest signal: count readings per bin separately.
    print(f"  {len(hel1os):,} one-second bins after resampling")

    return hel1os.reset_index()


def load_and_downsample_hel1os_with_coverage():
    """Same as above but also returns a per-second 'had any reading'
    mask, so empty resample bins can be flagged as missing rather than
    as genuine zero-count seconds."""
    print("Loading HEL1OS...")
    hel1os = pd.read_csv(HEL1OS_PATH)
    hel1os["time"] = pd.to_datetime(hel1os["time"])
    print(f"  {len(hel1os):,} rows loaded")

    before = len(hel1os)
    hel1os = hel1os.drop_duplicates()
    print(f"  Dropped {before - len(hel1os):,} exact duplicate rows "
          f"({len(hel1os):,} remain)")

    print("  Downsampling HEL1OS ~1.6 Hz -> 1 Hz "
          "(sum counts, mean hardness)...")

    hel1os = hel1os.set_index("time")

    agg_map = {c: "sum" for c in HEL1OS_COUNT_COLS}
    agg_map.update({c: "mean" for c in HEL1OS_RATIO_COLS})

    resampled = hel1os.resample("1s").agg(agg_map)

    # how many raw readings fell into each 1-second bin (0 = real gap)
    readings_per_bin = hel1os.resample("1s").size()
    resampled["_n_readings"] = readings_per_bin

    print(f"  {len(resampled):,} one-second bins after resampling")
    empty = (resampled["_n_readings"] == 0).sum()
    print(f"  {empty:,} of those bins are empty (real gaps, will be "
          f"flagged hard_missing)")

    return resampled.reset_index()


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

    # --- Missing flags ---
    # Soft: NaN counts after reindex = no SoLEXS data that second.
    combined["soft_missing"] = combined["counts"].isna()

    # Hard: a second is missing if either (a) it falls outside HEL1OS's
    # resampled span (reindex produced NaN), or (b) it's an empty
    # resample bin (_n_readings == 0). Empty bins have summed counts
    # of 0 which would otherwise look like a real quiet second.
    n_readings = hel1os_idx["_n_readings"]
    combined["hard_missing"] = n_readings.isna() | (n_readings == 0)

    # For hard_missing seconds, blank out the misleading 0-counts so
    # Phase 2's background never ingests a fake zero.
    combined.loc[combined["hard_missing"], HEL1OS_COUNT_COLS] = np.nan

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

    # Quick signal sanity: peak broadband values should be well above
    # their medians if a real flare is in here.
    for col in ["counts", "cdte_broad"]:
        s = combined[col].dropna()
        if len(s):
            print(f"{col:>12}: median={s.median():.1f}  "
                  f"max={s.max():.1f}")
    print("=" * 50)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    solexs = load_solexs()
    hel1os = load_and_downsample_hel1os_with_coverage()
    grid = build_master_grid(solexs, hel1os)
    combined = align(solexs, hel1os, grid)
    sanity_report(combined)

    # Drop internal helper column before saving
    # (it lives only in hel1os_idx, not in combined, so nothing to drop
    #  here - kept as a note in case the schema changes)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
