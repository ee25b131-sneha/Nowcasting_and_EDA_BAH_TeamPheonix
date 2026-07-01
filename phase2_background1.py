"""
PS15 Nowcasting - Phase 2: Adaptive Rolling Background (v2)
============================================================
Reads phase1_aligned.csv, computes a TRAILING adaptive background and
detection threshold for both channels, writes phase2_background.parquet.

Produces NO detections - only defines "what is normal right now".
Phase 3 does the flagging.

Run:
    cd D:\Team_Pheonix_PS15
    python scripts\phase2_background.py

(needs the parquet engine once:  pip install pyarrow )

------------------------------------------------------------------
TWO channels, TWO methods, because they have different statistics:

  SOFT (SoLEXS counts): continuously variable, non-zero baseline.
    -> robust rolling MEDIAN + MAD. Resists flare contamination in
       the window (median barely moves for brief outliers).

  HARD (HEL1OS CdTe): zero-inflated (0 counts most of the time).
    -> median+MAD fails here (median=0, MAD=0 -> threshold=0). Use
       POISSON statistics instead: local mean rate + N*sqrt(mean),
       which is the correct noise model for photon counting. A small
       absolute floor stops single stray photons from triggering.

Both use a TRAILING (causal) window - no future data. CZT carried
through untouched (secondary corroboration channel, not a detector).
------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"D:\Team_Pheonix_PS15")
ALIGNED = BASE_DIR / "outputs" / "phase1_aligned.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "phase2_background.parquet"

# --- Tunables ---
WINDOW = "2h"
N_SIGMA = 5.0
MAD_TO_SIGMA = 1.4826    # makes MAD a std-equivalent (soft channel)
MIN_PERIODS = 600        # >=10 min of real data before trusting background
HARD_FLOOR = 8           # hard detection also needs >= this many counts/s,
                         # so single stray photons can't trigger it


def soft_background(series, missing_mask):
    """Robust median + MAD background for the continuously-varying soft
    channel."""
    valid = series[~missing_mask].dropna()
    roll = valid.rolling(WINDOW, closed="left", min_periods=MIN_PERIODS)
    med = roll.median()
    mad = (valid - med).abs().rolling(
        WINDOW, closed="left", min_periods=MIN_PERIODS).median()
    sigma = mad * MAD_TO_SIGMA
    threshold = med + N_SIGMA * sigma
    return pd.DataFrame({"median": med, "sigma": sigma, "threshold": threshold})


def hard_background(series, missing_mask):
    """Poisson background for the zero-inflated hard channel.
    Local rate = rolling mean; spread = sqrt(rate) (Poisson);
    threshold = max(rate + N*sqrt(rate), absolute floor)."""
    valid = series[~missing_mask].dropna()
    roll = valid.rolling(WINDOW, closed="left", min_periods=MIN_PERIODS)
    rate = roll.mean()                    # mean is non-zero even if median is 0
    sigma = np.sqrt(rate)                 # Poisson: spread = sqrt(mean)
    threshold = rate + N_SIGMA * sigma
    threshold = threshold.clip(lower=HARD_FLOOR)   # never below the floor
    return pd.DataFrame({"median": rate, "sigma": sigma, "threshold": threshold})


def main():
    print("Loading phase1_aligned.csv ...")
    df = pd.read_csv(ALIGNED, parse_dates=["time"]).set_index("time")
    print(f"  {len(df):,} rows")

    print(f"Computing SOFT background (median+MAD, {WINDOW} trailing)...")
    soft = soft_background(df["counts"], df["soft_missing"]).reindex(df.index)
    df["soft_median"] = soft["median"]
    df["soft_sigma"] = soft["sigma"]
    df["soft_threshold"] = soft["threshold"]

    print(f"Computing HARD background (CdTe, Poisson mean+sqrt, {WINDOW})...")
    hard = hard_background(df["cdte_broad"], df["hard_missing"]).reindex(df.index)
    df["hard_median"] = hard["median"]
    df["hard_sigma"] = hard["sigma"]
    df["hard_threshold"] = hard["threshold"]

    print("\n" + "=" * 50)
    print("PHASE 2 REPORT")
    print("=" * 50)
    for name, med_col, thr_col in [
        ("SOFT", "soft_median", "soft_threshold"),
        ("HARD", "hard_median", "hard_threshold"),
    ]:
        m = df[med_col].dropna()
        t = df[thr_col].dropna()
        print(f"{name}: background level {m.min():.1f}..{m.max():.1f} "
              f"(typical {m.median():.1f})")
        print(f"      threshold       {t.min():.1f}..{t.max():.1f} "
              f"(typical {t.median():.1f})")
        print(f"      usable on {100*df[thr_col].notna().mean():.1f}% of seconds")
    print("=" * 50)

    print(f"\nWriting parquet (compact)...")
    try:
        df.reset_index().to_parquet(OUTPUT_PATH, index=False)
    except Exception as e:
        print("  parquet engine missing. Run:  pip install pyarrow")
        print(f"  ({e})")
        return
    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"Saved: {OUTPUT_PATH}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
