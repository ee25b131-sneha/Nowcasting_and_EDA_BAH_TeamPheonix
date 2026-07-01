"""
PS15 Nowcasting - Phase 5: Fusion (soft + hard) -- THE USP
============================================================
Reads events_soft.csv and events_hard.csv. Matches soft events against
hard events that overlap (or nearly overlap) in time, producing a fused
catalogue where each flare is tagged by how many instruments saw it:

  BOTH  - matched soft & hard event  -> highest confidence
  SOFT  - soft-only (no hard match)  -> thermal-dominated / smaller flare
  HARD  - hard-only (no soft match)  -> impulsive / non-thermal burst

Run:
    cd D:\Team_Pheonix_PS15
    python scripts\phase5_fuse.py

Output: outputs\events_fused.csv

------------------------------------------------------------------
Matching rule:
  Two events are the SAME flare if their time spans overlap, OR their
  edges fall within MATCH_TOLERANCE of each other. Tolerance accounts
  for the physical soft/hard timing offset (Neupert effect: hard can
  lead soft by minutes; sometimes soft preheating leads hard by ~3 min).
  We match in BOTH directions, so lead order doesn't matter.

  MATCH_TOLERANCE = 5 min (300 s). Comfortably covers the few-minute
  soft/hard lead-lag reported in the literature.

Confidence:
  BOTH-instrument events are high confidence (two independent detectors
  agree). This is the fusion USP: it suppresses single-instrument false
  positives and confirms genuine flares.
------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"D:\Team_Pheonix_PS15")
SOFT_PATH = BASE_DIR / "outputs" / "events_soft.csv"
HARD_PATH = BASE_DIR / "outputs" / "events_hard.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "events_fused.csv"

MATCH_TOLERANCE = pd.Timedelta(minutes=5)


def overlaps_or_near(a_start, a_end, b_start, b_end, tol):
    """True if intervals [a_start,a_end] and [b_start,b_end] overlap, or
    their nearest edges are within `tol`. Works regardless of which one
    starts first (so hard-leads-soft and soft-leads-hard both match)."""
    # Expand each interval by tol on both sides, then test overlap.
    return (a_start - tol <= b_end) and (b_start - tol <= a_end)


def main():
    print("Loading soft & hard event catalogues...")
    soft = pd.read_csv(SOFT_PATH, parse_dates=["start", "peak_time", "end"])
    hard = pd.read_csv(HARD_PATH, parse_dates=["start", "peak_time", "end"])
    print(f"  soft: {len(soft):,} events | hard: {len(hard):,} events")

    # Sort by start so we can walk efficiently.
    soft = soft.sort_values("start").reset_index(drop=True)
    hard = hard.sort_values("start").reset_index(drop=True)

    hard_used = np.zeros(len(hard), dtype=bool)
    fused_rows = []

    # ---- Match each soft event to overlapping/near hard event(s) ----
    for _, s in soft.iterrows():
        matches = []
        for hi, h in hard.iterrows():
            # Hard events are sorted; once a hard event starts well after
            # the soft event ends (+tol), no later hard can match -> break.
            if h["start"] - MATCH_TOLERANCE > s["end"]:
                break
            if h["end"] + MATCH_TOLERANCE < s["start"]:
                continue
            if overlaps_or_near(s["start"], s["end"],
                                h["start"], h["end"], MATCH_TOLERANCE):
                matches.append(hi)

        if matches:
            # Fuse soft with the strongest matching hard (highest peak).
            hbest = hard.loc[matches].sort_values("peak_value").iloc[-1]
            for hi in matches:
                hard_used[hi] = True
            # combined span
            start = min(s["start"], hbest["start"])
            end = max(s["end"], hbest["end"])
            # hard-vs-soft peak timing (Neupert direction)
            lag_s = (s["peak_time"] - hbest["peak_time"]).total_seconds()
            fused_rows.append({
                "type": "BOTH",
                "start": start, "end": end,
                "duration_s": int((end - start).total_seconds()) + 1,
                "soft_peak": s["peak_value"], "soft_peak_time": s["peak_time"],
                "hard_peak": hbest["peak_value"], "hard_peak_time": hbest["peak_time"],
                "hard_minus_soft_peak_lag_s": lag_s,
                "n_hard_matched": len(matches),
            })
        else:
            # Soft-only event.
            fused_rows.append({
                "type": "SOFT",
                "start": s["start"], "end": s["end"],
                "duration_s": s["duration_s"],
                "soft_peak": s["peak_value"], "soft_peak_time": s["peak_time"],
                "hard_peak": np.nan, "hard_peak_time": pd.NaT,
                "hard_minus_soft_peak_lag_s": np.nan,
                "n_hard_matched": 0,
            })

    # ---- Remaining hard events = hard-only ----
    for hi, h in hard.iterrows():
        if not hard_used[hi]:
            fused_rows.append({
                "type": "HARD",
                "start": h["start"], "end": h["end"],
                "duration_s": h["duration_s"],
                "soft_peak": np.nan, "soft_peak_time": pd.NaT,
                "hard_peak": h["peak_value"], "hard_peak_time": h["peak_time"],
                "hard_minus_soft_peak_lag_s": np.nan,
                "n_hard_matched": 0,
            })

    fused = pd.DataFrame(fused_rows).sort_values("start").reset_index(drop=True)

    # ---- Report ----
    counts = fused["type"].value_counts()
    print("\n" + "=" * 55)
    print("PHASE 5 REPORT (fusion)")
    print("=" * 55)
    print(f"Total fused events: {len(fused):,}")
    for t in ["BOTH", "SOFT", "HARD"]:
        print(f"  {t:5}: {int(counts.get(t,0)):,}")

    both = fused[fused["type"] == "BOTH"]
    if len(both):
        lags = both["hard_minus_soft_peak_lag_s"].dropna() / 60.0
        # positive lag = soft peaks AFTER hard = classic Neupert
        neupert = (lags > 0).mean()
        print(f"\nOf BOTH events, soft peak comes AFTER hard "
              f"(Neupert-consistent) in {100*neupert:.0f}%")
        print(f"  median hard->soft peak lag: {lags.median():.1f} min")

    # June 23 check
    j = fused[(fused["start"].dt.date == pd.Timestamp("2026-06-23").date()) |
              (fused["end"].dt.date == pd.Timestamp("2026-06-23").date())]
    jb = j[j["type"] == "BOTH"]
    if len(jb):
        big = jb.loc[jb["soft_peak"].idxmax()]
        print(f"\nJune 23 fused (BOTH): soft_peak={big['soft_peak']:.0f} "
              f"hard_peak={big['hard_peak']:.0f} "
              f"lag={big['hard_minus_soft_peak_lag_s']/60:.1f} min "
              f"({big['start']} -> {big['end']})")
    print("=" * 55)

    fused.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
