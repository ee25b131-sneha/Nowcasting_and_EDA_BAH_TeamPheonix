"""
PS15 Nowcasting - Phase 4: Group detections into events
========================================================
Reads phase3_detections.parquet. Collapses the per-second detection
flags into discrete EVENTS, separately for soft and hard, each with a
start / peak / end time and summary stats. Writes two event catalogues.

(Pipeline order note: grouping is Phase 4, fusion is Phase 5 - swapped
from the original plan so fusion operates on clean events, not raw
seconds.)

Run:
    cd D:\Team_Pheonix_PS15
    python scripts\phase4_group.py

Outputs:
    outputs\events_soft.csv
    outputs\events_hard.csv

------------------------------------------------------------------
Grouping rules (literature-backed - see GOES flare-duration standards):
  * BRIDGE_GAP = 60 s : detections within 60 s of each other belong to
    the SAME event. A sub-minute dip below threshold mid-flare is noise
    within one flare, not a gap between two flares.
  * MIN_DURATION = 60 s : events shorter than 60 s are discarded as
    likely noise (matches the GOES minimum-flare-duration floor).
  * No maximum duration - a long-duration flare can run for hours, and
    stays one event as long as detections keep coming within 60 s.
------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"D:\Team_Pheonix_PS15")
INPUT_PATH = BASE_DIR / "outputs" / "phase3_detections.parquet"
OUT_SOFT = BASE_DIR / "outputs" / "events_soft.csv"
OUT_HARD = BASE_DIR / "outputs" / "events_hard.csv"

BRIDGE_GAP = pd.Timedelta(seconds=60)   # merge detections within 60 s
MIN_DURATION = pd.Timedelta(seconds=60)  # discard events shorter than 60 s


def group_channel(df, detect_col, value_col, label):
    """Turn per-second detections in one channel into events.

    Steps:
      1. Keep only the seconds flagged as detections.
      2. Walk them in time order. Whenever the gap to the previous
         detection exceeds BRIDGE_GAP, a new event starts.
      3. For each event, record start/peak/end time, duration, peak
         value, and how many seconds it covered.
      4. Drop events shorter than MIN_DURATION.
    """
    det = df.loc[df[detect_col], ["time", value_col]].copy()
    det = det.sort_values("time").reset_index(drop=True)
    print(f"  {label}: {len(det):,} detected seconds to group")

    if len(det) == 0:
        return pd.DataFrame()

    # Time since the previous detection. Where it exceeds the bridge
    # gap, we're starting a new event -> cumulative-sum those breaks
    # into an event id.
    gap = det["time"].diff()
    new_event = (gap > BRIDGE_GAP).fillna(True)  # first row starts event 0
    det["event_id"] = new_event.cumsum()

    # Summarise each event.
    events = []
    for eid, g in det.groupby("event_id"):
        start = g["time"].iloc[0]
        end = g["time"].iloc[-1]
        duration = end - start
        peak_idx = g[value_col].idxmax()
        events.append({
            "event_id": int(eid),
            "start": start,
            "peak_time": g.loc[peak_idx, "time"],
            "end": end,
            "duration_s": int(duration.total_seconds()) + 1,  # inclusive
            "peak_value": float(g[value_col].max()),
            "detected_seconds": len(g),
        })

    ev = pd.DataFrame(events)

    # Noise filter: drop sub-60s events.
    before = len(ev)
    ev = ev[ev["duration_s"] >= MIN_DURATION.total_seconds()].reset_index(drop=True)
    print(f"  {label}: {before:,} raw events -> {len(ev):,} after "
          f"dropping <60 s")
    return ev


def report(ev, label):
    if len(ev) == 0:
        print(f"  {label}: no events")
        return
    dur = ev["duration_s"] / 60.0  # minutes
    print(f"  {label}: {len(ev):,} events | "
          f"duration min/median/max = "
          f"{dur.min():.1f}/{dur.median():.1f}/{dur.max():.1f} min")
    # Does June 23 show up?
    jun23 = ev[ev["peak_time"].dt.date == pd.Timestamp("2026-06-23").date()]
    if len(jun23):
        big = jun23.loc[jun23["peak_value"].idxmax()]
        print(f"       June 23 event: {big['start']} -> {big['end']} "
              f"(peak {big['peak_value']:.0f}, {big['duration_s']/60:.1f} min)")


def main():
    print("Loading phase3_detections.parquet ...")
    df = pd.read_parquet(INPUT_PATH)
    df["time"] = pd.to_datetime(df["time"])
    print(f"  {len(df):,} rows")

    print("\nGrouping SOFT...")
    soft_ev = group_channel(df, "soft_detect", "counts", "soft")

    print("\nGrouping HARD...")
    hard_ev = group_channel(df, "hard_detect", "cdte_broad", "hard")

    print("\n" + "=" * 55)
    print("PHASE 4 REPORT")
    print("=" * 55)
    report(soft_ev, "SOFT")
    report(hard_ev, "HARD")
    print("=" * 55)

    soft_ev.to_csv(OUT_SOFT, index=False)
    hard_ev.to_csv(OUT_HARD, index=False)
    print(f"\nSaved: {OUT_SOFT}")
    print(f"Saved: {OUT_HARD}")


if __name__ == "__main__":
    main()
