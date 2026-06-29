from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import (
    HOURLY_ROW_STATES_PATH,
    NETWORK_OUTAGE_WINDOWS_PATH,
)

OUTPUT_DIR = Path("docs/availability_investigation_checks")
OUTPUT_TXT = OUTPUT_DIR / "window_durations_hourly.txt"
OUTPUT_CSV = OUTPUT_DIR / "window_durations_hourly.csv"

ONLINE_STATES = {"online_complete", "online_partial_missing"}


def compute_window_duration(
    window_id: str,
    window_meta: pd.Series,
    states: pd.DataFrame,
) -> dict:
    result: dict = {
        "window_id": window_id,
        "window_start_utc": window_meta["window_start_utc"],
        "station_count": int(window_meta["station_count"]),
        "outage_class": window_meta.get("outage_class", "unknown"),
    }

    station_ids = window_meta["station_ids"].split(";")
    result["n_stations_in_window"] = len(station_ids)

    window_start = pd.Timestamp(window_meta["window_start_utc"], tz="UTC")
    backfill_start = pd.Timestamp(window_meta["backfill_start_utc"], tz="UTC")

    per_station_recovery: list[dict] = []
    for sid in station_ids:
        sid_states = states[states["station_id"] == sid].sort_values("hour_utc")
        sid_states = sid_states[sid_states["hour_utc"] >= backfill_start]

        outage_hours = sid_states[
            sid_states["row_state"] == "true_outage_candidate"
        ]
        if outage_hours.empty:
            continue

        outage_start = outage_hours["hour_utc"].iloc[0]

        after = sid_states[
            (sid_states["hour_utc"] > outage_start)
            & (sid_states["row_state"].isin(ONLINE_STATES))
        ]

        if after.empty:
            recovery = None
            duration_hours = None
        else:
            recovery = after["hour_utc"].iloc[0]
            duration_hours = (recovery - outage_start).total_seconds() / 3600.0

        per_station_recovery.append({
            "station_id": sid,
            "outage_start_utc": outage_start,
            "recovery_utc": recovery,
            "duration_hours": duration_hours,
        })

    if not per_station_recovery:
        result["status"] = "no_outages_found"
        return result

    durations = [
        r["duration_hours"]
        for r in per_station_recovery
        if r["duration_hours"] is not None
    ]
    recoveries = [
        r["recovery_utc"]
        for r in per_station_recovery
        if r["recovery_utc"] is not None
    ]
    starts = [r["outage_start_utc"] for r in per_station_recovery]

    result["n_with_duration"] = len(durations)
    result["n_never_recovered"] = len(per_station_recovery) - len(durations)

    if starts:
        starts_series = pd.Series(starts)
        result["earliest_outage_start_utc"] = str(starts_series.min())
        result["latest_outage_start_utc"] = str(starts_series.max())
        result["start_spread_hours"] = (
            starts_series.max() - starts_series.min()
        ).total_seconds() / 3600.0

    if durations:
        d = pd.Series(durations)
        result["min_duration_hours"] = float(d.min())
        result["median_duration_hours"] = float(d.median())
        result["max_duration_hours"] = float(d.max())
        result["mean_duration_hours"] = float(d.mean())

    if recoveries:
        r = pd.Series(recoveries)
        result["earliest_recovery_utc"] = str(r.min())
        result["latest_recovery_utc"] = str(r.max())
        result["recovery_spread_hours"] = (
            r.max() - r.min()
        ).total_seconds() / 3600.0

    result["status"] = "ok"
    return result


def main() -> None:
    states = pd.read_parquet(HOURLY_ROW_STATES_PATH)
    windows = pd.read_csv(NETWORK_OUTAGE_WINDOWS_PATH)
    windows = windows.sort_values("window_start_utc").reset_index(drop=True)

    results = []
    for _, window in windows.iterrows():
        results.append(compute_window_duration(window["window_id"], window, states))

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    midnight_mask = pd.to_datetime(df["window_start_utc"]).dt.hour.isin([22, 23])
    midnight = df[midnight_mask & (df["status"] == "ok")].copy()
    nonmidnight = df[~midnight_mask & (df["status"] == "ok")].copy()

    lines: list[str] = []
    lines.append("=== Window durations from hourly row states ===")
    lines.append(f"Total windows analyzed: {len(df)}")
    lines.append(f"  Midnight (22 or 23 UTC): {len(midnight)}")
    lines.append(f"  Non-midnight: {len(nonmidnight)}")
    lines.append("")

    if not midnight.empty:
        lines.append("=== Midnight events: per-station outage durations ===")
        mids = midnight["median_duration_hours"].dropna()
        if not mids.empty:
            lines.append(
                f"Median of median per-station duration: {mids.median():.1f}h"
            )
            lines.append(
                f"Range across windows: {mids.min():.1f}h to {mids.max():.1f}h"
            )
        never = midnight["n_never_recovered"].sum()
        lines.append(f"Total station-events that never recovered: {never}")
        lines.append("")

    if not nonmidnight.empty:
        lines.append("=== Non-midnight events: per-station outage durations ===")
        nmids = nonmidnight["median_duration_hours"].dropna()
        if not nmids.empty:
            lines.append(
                f"Median of median per-station duration: {nmids.median():.1f}h"
            )
            lines.append(
                f"Range across windows: {nmids.min():.1f}h to {nmids.max():.1f}h"
            )
        lines.append("")

    lines.append("=== Per-window summary ===")
    cols = [
        "window_id", "window_start_utc", "station_count",
        "n_stations_in_window", "n_with_duration", "n_never_recovered",
        "start_spread_hours", "median_duration_hours", "max_duration_hours",
        "recovery_spread_hours", "status",
    ]
    present_cols = [c for c in cols if c in df.columns]
    lines.append(df[present_cols].to_string(index=False))

    report = "\n".join(lines) + "\n"
    OUTPUT_TXT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
