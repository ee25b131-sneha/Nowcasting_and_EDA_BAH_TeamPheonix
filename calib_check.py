"""
Calibration diagnostic: does our catalogue independently recover NOAA's
4 known X-flares (Mar 30, Apr 23, Apr 24, Jun 3, 2026)?

This does NOT change anything - it only reports, so we can anchor the
class boundaries to real ground-truth events.

Run:
    cd D:\Team_Pheonix_PS15
    python scripts\calib_check.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"D:\Team_Pheonix_PS15")
fused = pd.read_csv(BASE / "outputs" / "events_fused.csv",
                    parse_dates=["start", "end", "soft_peak_time", "hard_peak_time"])

# NOAA's 4 known X-flares in the window (date -> official magnitude)
KNOWN_X = {
    "2026-03-30": "X1.5",
    "2026-04-23": "X2.4",
    "2026-04-24": "X2.5",
    "2026-06-03": "X1.0",
}

# Use soft peak as the intensity measure (GOES class is soft X-ray flux).
# For HARD-only events soft_peak is NaN; fill with hard for ranking only.
fused["rank_peak"] = fused["soft_peak"].fillna(fused["hard_peak"])
fused["peak_date"] = fused["soft_peak_time"].fillna(fused["hard_peak_time"]).dt.date

print("=" * 60)
print("PART 1: Events on NOAA's 4 known X-flare dates")
print("=" * 60)
xflare_peaks = []
for d, mag in KNOWN_X.items():
    dd = pd.Timestamp(d).date()
    ev = fused[fused["peak_date"] == dd].sort_values("rank_peak", ascending=False)
    print(f"\n{d} (NOAA {mag}): {len(ev)} events on this day")
    if len(ev):
        top = ev.iloc[0]
        xflare_peaks.append(top["rank_peak"])
        print(f"   strongest: soft_peak={top['soft_peak']}, "
              f"hard_peak={top['hard_peak']}, type={top['type']}, "
              f"at {top['soft_peak_time'] if pd.notna(top['soft_peak_time']) else top['hard_peak_time']}")

print("\n" + "=" * 60)
print("PART 2: Our 15 strongest events overall (by soft peak)")
print("=" * 60)
top15 = fused.sort_values("rank_peak", ascending=False).head(15)
for _, e in top15.iterrows():
    known = "  <-- NOAA X-FLARE DATE" if str(e["peak_date"]) in KNOWN_X else ""
    print(f"  {e['peak_date']}  soft={e['soft_peak']}  hard={e['hard_peak']}  "
          f"type={e['type']}{known}")

print("\n" + "=" * 60)
print("PART 3: If we anchor X threshold to the known X-flares")
print("=" * 60)
if xflare_peaks:
    xmin = min(xflare_peaks)
    print(f"Weakest known X-flare peak in our data: {xmin:.0f} counts")
    print(f"So X threshold ~ {xmin:.0f}, and by decade structure:")
    for name, lo in [("X", xmin), ("M", xmin/10), ("C", xmin/100), ("B", xmin/1000)]:
        n = ((fused["rank_peak"] >= lo) &
             (fused["rank_peak"] < (lo*10 if name!="X" else np.inf))).sum()
        print(f"   {name}: >= {lo:.0f} counts")
else:
    print("None of the 4 known X-flares found in catalogue - need to check gaps")
