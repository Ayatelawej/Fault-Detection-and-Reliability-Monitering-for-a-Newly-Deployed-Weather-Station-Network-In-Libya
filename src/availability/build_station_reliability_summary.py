from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.paths import (
    AVAILABILITY_EVENTS_PATH,
    DATA_AUDIT_SUMMARY_PATH,
    STATION_REGISTRY_PATH,
)

OUTPUT_PATH = Path("data/processed/station_reliability_summary.csv")

DATASET_FREEZE_END = pd.Timestamp("2026-03-31 23:00:00", tz="UTC")


def main() -> None:
    events = pd.read_parquet(AVAILABILITY_EVENTS_PATH)
    registry = pd.read_csv(STATION_REGISTRY_PATH)
    audit = pd.read_csv(DATA_AUDIT_SUMMARY_PATH)

    events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True)
    events["end_utc"] = pd.to_datetime(events["end_utc"], utc=True)

    rows: list[dict] = []
    for _, station in registry.iterrows():
        sid = station["station_id"]
        sid_events = events[events["station_id"] == sid]
        audit_row = audit[audit["station_id"] == sid].iloc[0] if (
            audit["station_id"] == sid
        ).any() else None

        row = {
            "station_id": sid,
            "city": station.get("city"),
            "region": station.get(
                "region",
                "northwest" if station["longitude"] < 16 else "southeast",
            ),
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "elevation_m": station.get("elevation"),
            "install_date": station.get("install_date"),
        }

        if audit_row is not None:
            row["status_class"] = audit_row.get("status_class")
            row["uptime_pct"] = (
                100.0 * audit_row["present_rows"] / audit_row["total_rows"]
                if audit_row["total_rows"] > 0 else None
            )
            row["total_active_hours"] = int(audit_row["total_rows"])
            row["total_present_hours"] = int(audit_row["present_rows"])
            row["total_outage_hours"] = int(audit_row["true_outage_candidate_rows"])

        row["total_event_count"] = int(len(sid_events))
        row["local_event_count"] = int(
            (sid_events["outage_class"] == "local").sum()
        )
        row["network_midnight_event_count"] = int(
            (sid_events["outage_class"] == "network_midnight").sum()
        )
        row["network_other_event_count"] = int(
            (sid_events["outage_class"] == "network_other").sum()
        )

        if not sid_events.empty:
            durations = sid_events["duration_hours"]
            row["median_event_duration_h"] = float(durations.median())
            row["max_event_duration_h"] = float(durations.max())
            row["last_outage_start_utc"] = str(
                sid_events["start_utc"].max()
            )
            last_event = sid_events.loc[sid_events["start_utc"].idxmax()]
            last_end = last_event["end_utc"]
            days_since = (DATASET_FREEZE_END - last_end).total_seconds() / 86400.0
            row["days_since_last_outage_at_freeze"] = (
                float(max(days_since, 0.0))
            )
        else:
            row["median_event_duration_h"] = None
            row["max_event_duration_h"] = None
            row["last_outage_start_utc"] = None
            row["days_since_last_outage_at_freeze"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["region", "uptime_pct"], ascending=[True, False])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
