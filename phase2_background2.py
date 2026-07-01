"""
PS15 Nowcasting - Phase 2: Adaptive Rolling Background (v3, two-pass)
=====================================================================
Reads phase1_aligned.csv, computes a TRAILING adaptive background and
detection threshold for both channels, writes phase2_background.parquet.

v3 change: TWO-PASS (iterative) background to fix flare self-masking.
A long sustained flare was pulling its own trailing background up until
the background overtook the signal (observed on the real June 23 soft
flare: counts ~440-500 while threshold crept to ~550, so the flare was
never detected). The fix: estimate the background, find rough
detections, then RE-estimate the background with those detected seconds
excluded (treated like missing), so the background reflects quiet Sun
only and a flare can't mask itself.

Run:
    cd D:\Team_Pheonix_PS15
    python scripts\phase2_background.py
    (needs: pip install pyarrow)

------------------------------------------------------------------
Methods unchanged from v2:
  SOFT: robust rolling MEDIAN + MAD (continuously variable channel)
  HARD: Poisson rolling MEAN + N*sqrt(mean), floored (zero-inflated)
Both TRAILING (causal). CZT carried through untouched.

New in v3:
  * PASS 1 background -> provisional threshold -> provisional detect.
  * PASS 2 background recomputed with provisional-detect seconds masked
    out (unioned with the real missing mask). Final threshold from pass 2.
  * One extra rolling pass per channel (~20-40s more). Standard
    "iterative background estimation excluding flare intervals".
------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"D:\Team_Pheonix_PS15")
ALIGNED = BASE_DIR / "outputs" / "phase1_aligned.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "phase2_background.parquet"

# --- Tunables ---
WINDOW = "12h"
N_SIGMA = 5.0
MAD_TO_SIGMA = 1.4826
MIN_PERIODS = 600
HARD_FLOOR = 8


def _soft_stats(series, exclude_mask):
    """Rolling median + MAD-sigma + threshold on the soft channel,
    computed only over seconds NOT in exclude_mask."""
    valid = series[~exclude_mask].dropna()
    roll = valid.rolling(WINDOW, closed="left", min_periods=MIN_PERIODS)
    med = roll.median()
    mad = (valid - med).abs().rolling(
        WINDOW, closed="left", min_periods=MIN_PERIODS).median()
    sigma = mad * MAD_TO_SIGMA
    threshold = med + N_SIGMA * sigma
    return med, sigma, threshold


def _hard_stats(series, exclude_mask):
    """Poisson rolling mean + N*sqrt(mean), floored, on the hard channel,
    over seconds NOT in exclude_mask."""
    valid = series[~exclude_mask].dropna()
    roll = valid.rolling(WINDOW, closed="left", min_periods=MIN_PERIODS)
    rate = roll.mean()
    sigma = np.sqrt(rate)
    threshold = (rate + N_SIGMA * sigma).clip(lower=HARD_FLOOR)
    return rate, sigma, threshold


def two_pass(series, missing_mask, stats_fn, full_index, label):
    """Generic two-pass background.
    Pass 1: background over (not missing) -> provisional detections.
    Pass 2: background over (not missing AND not provisionally detected)
            -> final median/sigma/threshold.
    """
    # --- Pass 1 ---
    _, _, thr1 = stats_fn(series, missing_mask)
    thr1 = thr1.reindex(full_index)
    prov_detect = (series > thr1) & (~missing_mask) & thr1.notna()
    n1 = int(prov_detect.sum())
    print(f"    {label} pass 1: {n1:,} provisional detected seconds")

    # --- Pass 2: INTERPOLATE across provisional detections (replace
    # flare seconds with a quiet-level estimate) rather than deleting
    # them. Deleting created data holes that starved the trailing
    # window (-> NaN thresholds). Interpolation keeps the window full
    # of quiet-Sun-level data so the background stays low through a
    # flare and cannot mask it.
    prov = prov_detect.fillna(False)
    series_clean = series.copy()
    series_clean[prov] = np.nan
    series_clean = series_clean.interpolate(method="time").bfill().ffill()
    # Real missing seconds stay excluded; interpolated flare seconds are
    # now quiet-level and safe to include.
    med, sigma, thr2 = stats_fn(series_clean, missing_mask)
    med = med.reindex(full_index)
    sigma = sigma.reindex(full_index)
    thr2 = thr2.reindex(full_index)
    n2 = int(((series > thr2) & (~missing_mask) & thr2.notna()).sum())
    print(f"    {label} pass 2: {n2:,} detected seconds after re-estimation "
          f"(was {n1:,})")
    return med, sigma, thr2


def main():
    print("Loading phase1_aligned.csv ...")
    df = pd.read_csv(ALIGNED, parse_dates=["time"]).set_index("time")
    print(f"  {len(df):,} rows")
    idx = df.index

    print(f"\nSOFT background (median+MAD, {WINDOW}, two-pass)...")
    s_med, s_sig, s_thr = two_pass(
        df["counts"], df["soft_missing"], _soft_stats, idx, "SOFT")
    df["soft_median"] = s_med
    df["soft_sigma"] = s_sig
    df["soft_threshold"] = s_thr

    print(f"\nHARD background (CdTe Poisson, {WINDOW}, two-pass)...")
    h_med, h_sig, h_thr = two_pass(
        df["cdte_broad"], df["hard_missing"], _hard_stats, idx, "HARD")
    df["hard_median"] = h_med
    df["hard_sigma"] = h_sig
    df["hard_threshold"] = h_thr

    print("\n" + "=" * 55)
    print("PHASE 2 REPORT (v3 two-pass)")
    print("=" * 55)
    for name, med_col, thr_col in [
        ("SOFT", "soft_median", "soft_threshold"),
        ("HARD", "hard_median", "hard_threshold"),
    ]:
        m = df[med_col].dropna()
        t = df[thr_col].dropna()
        print(f"{name}: background {m.min():.1f}..{m.max():.1f} "
              f"(typical {m.median():.1f}) | threshold {t.min():.1f}.."
              f"{t.max():.1f} (typical {t.median():.1f})")

    # Direct check on the June 23 masking case
    probe = df.loc["2026-06-23 23:25:30"] if \
        pd.Timestamp("2026-06-23 23:25:30") in df.index else None
    if probe is not None:
        print(f"\nJune 23 23:25:30 check: counts={probe['counts']:.0f}  "
              f"soft_threshold={probe['soft_threshold']:.1f}  "
              f"-> {'DETECT' if probe['counts']>probe['soft_threshold'] else 'still masked'}")
    print("=" * 55)

    print("\nWriting parquet...")
    try:
        df.reset_index().to_parquet(OUTPUT_PATH, index=False)
    except Exception as e:
        print("  parquet engine missing. Run:  pip install pyarrow")
        print(f"  ({e})")
        return
    print(f"Saved: {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
