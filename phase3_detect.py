"""
PS15 Nowcasting - Phase 3: Independent Detection (3a soft + 3b hard)
=====================================================================
Reads phase2_background.parquet. For every second, flags a detection
where the measured value exceeds its Phase-2 threshold. Soft and hard
are detected INDEPENDENTLY -> two separate detection flags. Fusion
(matching the two) is Phase 4.

Run:
    cd D:\Team_Pheonix_PS15
    python scripts\phase3_detect.py

Output: outputs\phase3_detections.parquet

------------------------------------------------------------------
A detection is simply: value > threshold, on a second that is neither
missing nor lacking a valid (warmed-up) threshold. No math beyond a
comparison - all the statistics lived in Phase 2.

Note: these are detected SECONDS, not flares. A single flare spans many
consecutive detected seconds. Phase 5 groups them into events.
------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"D:\Team_Pheonix_PS15")
INPUT_PATH = BASE_DIR / "outputs" / "phase2_background.parquet"
OUTPUT_PATH = BASE_DIR / "outputs" / "phase3_detections.parquet"

# A known real flare we expect the detector to recover (validation).
KNOWN_FLARE_DATE = pd.Timestamp("2026-06-23").date()


def main():
    print("Loading phase2_background.parquet ...")
    df = pd.read_parquet(INPUT_PATH)
    df["time"] = pd.to_datetime(df["time"])
    print(f"  {len(df):,} rows")

    # ----- 3a: SOFT detection -----
    # True only where: value exceeds threshold, the second has soft data,
    # and the threshold is valid (not NaN from the warm-up period).
    df["soft_detect"] = (
        (df["counts"] > df["soft_threshold"])
        & (~df["soft_missing"])
        & (df["soft_threshold"].notna())
    )

    # ----- 3b: HARD detection (CdTe) -----
    df["hard_detect"] = (
        (df["cdte_broad"] > df["hard_threshold"])
        & (~df["hard_missing"])
        & (df["hard_threshold"].notna())
    )

    # ----------------- Report -----------------
    n = len(df)
    n_soft = int(df["soft_detect"].sum())
    n_hard = int(df["hard_detect"].sum())
    soft_usable = int(df["soft_threshold"].notna().sum())
    hard_usable = int(df["hard_threshold"].notna().sum())

    print("\n" + "=" * 55)
    print("PHASE 3 REPORT")
    print("=" * 55)
    print(f"Soft detected seconds: {n_soft:,} "
          f"({100*n_soft/soft_usable:.3f}% of usable soft seconds)")
    print(f"Hard detected seconds: {n_hard:,} "
          f"({100*n_hard/hard_usable:.3f}% of usable hard seconds)")

    # When do detections cluster? Top days by soft-detected seconds.
    df["date"] = df["time"].dt.date
    top_soft = (df.groupby("date")["soft_detect"].sum()
                .sort_values(ascending=False).head(10))
    print("\nTop 10 days by soft detections (should include real flares):")
    for d, c in top_soft.items():
        mark = "  <- known June 23 flare" if d == KNOWN_FLARE_DATE else ""
        print(f"   {d}: {int(c):,} detected seconds{mark}")

    # Explicit validation on the known flare day.
    jun23 = df[df["date"] == KNOWN_FLARE_DATE]
    if len(jun23):
        js = int(jun23["soft_detect"].sum())
        jh = int(jun23["hard_detect"].sum())
        print(f"\nJune 23 validation: {js:,} soft + {jh:,} hard detected seconds")
        if js > 0:
            peak = jun23.loc[jun23["counts"].idxmax()]
            print(f"   Peak soft counts that day: {peak['counts']:.0f} "
                  f"at {peak['time']}")
    print("=" * 55)

    # Save (drop helper 'date' col before writing)
    df.drop(columns=["date"]).to_parquet(OUTPUT_PATH, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"\nSaved: {OUTPUT_PATH}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
