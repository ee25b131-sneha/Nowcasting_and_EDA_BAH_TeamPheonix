"""
PS15 Nowcasting - Phase 2: Adaptive Rolling Background
========================================================
Reads phase1_aligned.csv and computes, for SoLEXS (soft) and HEL1OS
CdTe (hard), a *trailing* rolling background: a local median and a
local spread (MAD), plus a detection threshold = median + N*MAD-sigma.

This produces NO detections yet - it only defines "what is normal right
now" at every second. Phase 3 does the actual flagging.

Run:
    cd D:\Team_Pheonix_PS15
    python scripts\phase2_background.py

Output: outputs\phase2_background.csv

------------------------------------------------------------------
Design (see conversation log for full reasoning):
  * TRAILING window (closed='left' so the current second doesn't sit
    in its own background) -> causal, the defining property of
    nowcasting. No future data is ever used.
  * ROBUST stats: rolling MEDIAN + MAD, not mean + std. Flares are
    outliers; median/MAD barely move when a flare is in the window,
    so the threshold stays honest. MAD is scaled by 1.4826 to be a
    std-equivalent (so "N sigma" means the usual thing).
  * WINDOW = 2h by default. Much longer than a flare (~minutes), much
    shorter than the quiet-Sun drift (~weeks). Tunable below.
  * SOFT and HARD get independent backgrounds.
  * MISSING seconds are excluded from the background entirely, never
    treated as zero.
  * HARD detection uses CdTe only. CZT is carried through untouched as
    a secondary corroboration channel (high noise -> not a detector).
------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"D:\Team_Pheonix_PS15")
ALIGNED = BASE_DIR / "outputs" / "phase1_aligned.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "phase2_background.csv"

# --- Tunable parameters ---
WINDOW = "2h"          # trailing window length
N_SIGMA = 5.0          # threshold = median + N_SIGMA * (MAD-as-sigma)
MAD_TO_SIGMA = 1.4826  # converts MAD to std-equivalent for Gaussian noise
MIN_PERIODS = 600      # need >=600 real samples in the window to trust it
                       # (10 min of data); else background is NaN (no detect)


def rolling_background(series, missing_mask):
    """Compute trailing rolling median + MAD-sigma + threshold for one
    channel. `series` is the per-second values (NaN where missing);
    `missing_mask` is True where data is absent.

    Returns a DataFrame with columns: median, sigma, threshold.
    """
    # Keep only real observations - drop missing seconds so they can't
    # pull the background down. We compute on the valid points using a
    # TIME-BASED window, which correctly handles the irregular spacing
    # that results from dropping gaps.
    valid = series[~missing_mask].dropna()

    # Trailing window. closed='left' excludes the current timestamp from
    # its own window, so a value is compared against its PAST only.
    roll = valid.rolling(WINDOW, closed="left", min_periods=MIN_PERIODS)

    med = roll.median()
    # MAD = median(|x - median|). Need the rolling median first, then a
    # second rolling median of absolute deviations.
    abs_dev = (valid - med).abs()
    mad = abs_dev.rolling(WINDOW, closed="left", min_periods=MIN_PERIODS).median()

    sigma = mad * MAD_TO_SIGMA
    threshold = med + N_SIGMA * sigma

    out = pd.DataFrame({
        "median": med,
        "sigma": sigma,
        "threshold": threshold,
    })
    return out


def main():
    print("Loading phase1_aligned.csv ...")
    df = pd.read_csv(ALIGNED, parse_dates=["time"]).set_index("time")
    print(f"  {len(df):,} rows")

    # ----- SOFT (SoLEXS counts) -----
    print(f"Computing soft background (median+MAD, {WINDOW} trailing)...")
    soft = rolling_background(df["counts"], df["soft_missing"])
    soft = soft.reindex(df.index)   # back onto the full per-second grid
    df["soft_median"] = soft["median"]
    df["soft_sigma"] = soft["sigma"]
    df["soft_threshold"] = soft["threshold"]

    # ----- HARD (HEL1OS CdTe broadband) -----
    print(f"Computing hard background (CdTe, median+MAD, {WINDOW})...")
    hard = rolling_background(df["cdte_broad"], df["hard_missing"])
    hard = hard.reindex(df.index)
    df["hard_median"] = hard["median"]
    df["hard_sigma"] = hard["sigma"]
    df["hard_threshold"] = hard["threshold"]

    # ----- Quick report -----
    print("\n" + "=" * 50)
    print("PHASE 2 REPORT")
    print("=" * 50)
    for name, med_col, thr_col in [
        ("SOFT", "soft_median", "soft_threshold"),
        ("HARD", "hard_median", "hard_threshold"),
    ]:
        m = df[med_col].dropna()
        t = df[thr_col].dropna()
        print(f"{name}: background median ranges "
              f"{m.min():.1f}..{m.max():.1f} (typical {m.median():.1f})")
        print(f"      threshold ranges {t.min():.1f}..{t.max():.1f} "
              f"(typical {t.median():.1f})")
        # how many seconds have a usable (non-NaN) threshold?
        usable = df[thr_col].notna().mean()
        print(f"      usable threshold on {100*usable:.1f}% of seconds")
    print("=" * 50)

    df.reset_index().to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
