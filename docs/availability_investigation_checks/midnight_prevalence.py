from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import (
    AVAILABILITY_EVENTS_PATH,
    HOURLY_ROW_STATES_PATH,
)

OUTPUT_PATH = Path(__file__).parent / "midnight_prevalence.txt"


def main() -> None:
    events = pd.read_parquet(AVAILABILITY_EVENTS_PATH)
    states = pd.read_parquet(HOURLY_ROW_STATES_PATH)

    events = events.copy()
    events["start_hour"] = pd.to_datetime(events["start_utc"]).dt.hour

    hour_counts = events["start_hour"].value_counts().sort_index()
    midnight_count = int(events["start_hour"].isin([22, 23]).sum())
    other = events[~events["start_hour"].isin([22, 23])]
    other_avg = float(other.groupby("start_hour").size().mean())
    ratio = midnight_count / (other_avg * 2) if other_avg > 0 else float("nan")

    participating = [
        "IALWAH18", "IBARAS3", "IJABAL13", "IJABAL15", "IJABAL16",
        "IJANZO2", "IJANZO3", "IMISRA12", "IMURQU5", "INUQAT10",
        "INUQAT8", "ITAHLI1", "ITRIPO33", "IZAWIY7",
    ]
    all_stations = list(states["station_id"].unique())
    non_participating = [s for s in all_stations if s not in participating]

    target_hour = pd.Timestamp("2026-03-08 22:00:00", tz="UTC")
    just_before = states[
        (states["station_id"].isin(non_participating))
        & (states["hour_utc"] == target_hour)
    ][["station_id", "row_state"]].sort_values("station_id")

    state_counts = just_before["row_state"].value_counts().to_dict()

    lines: list[str] = []
    lines.append("=== All 1,670 events by start hour (UTC) ===")
    lines.append(hour_counts.to_string())
    lines.append("")
    lines.append(f"Events starting at 22-23 UTC: {midnight_count}")
    lines.append(f"Average events per other hour: {other_avg:.1f}")
    lines.append(f"Ratio: {ratio:.1f}x baseline")
    lines.append("")
    lines.append(
        "=== Non-participating stations: state at 22:00 UTC March 8 (just before drop) ==="
    )
    lines.append(just_before.to_string(index=False))
    lines.append("")
    lines.append(f"State counts: {state_counts}")

    report = "\n".join(lines) + "\n"
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
