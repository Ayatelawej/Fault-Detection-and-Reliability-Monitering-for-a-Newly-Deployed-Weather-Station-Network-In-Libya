from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import (
    AVAILABILITY_EVENTS_PATH,
    HOURLY_ROW_STATES_PATH,
    STATION_REGISTRY_PATH,
)

OUTPUT_DIR = Path("docs/availability_investigation_checks")
OUTPUT_TXT = OUTPUT_DIR / "local_cooccurrence_event_clean.txt"
OUTPUT_CSV = OUTPUT_DIR / "local_cooccurrence_event_clean.csv"

EARTH_RADIUS_KM = 6371.0

INACTIVE_STATES = {
    "pre_install_padded_absence",
    "pre_install_padded_present",
    "pre_install_invalid_unknown",
    "terminal_padded_absence",
}

NETWORK_OUTAGE_CLASSES = {"network_midnight", "network_other"}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def build_excluded_station_hours(events: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    nw_events = events[events["outage_class"].isin(NETWORK_OUTAGE_CLASSES)].copy()
    excluded: set[tuple[str, pd.Timestamp]] = set()
    for _, ev in nw_events.iterrows():
        sid = ev["station_id"]
        start = pd.to_datetime(ev["start_utc"], utc=True)
        end = pd.to_datetime(ev["end_utc"], utc=True)
        current = start.floor("h")
        end_floor = end.floor("h")
        while current <= end_floor:
            excluded.add((sid, current))
            current = current + pd.Timedelta(hours=1)
    return excluded


def main() -> None:
    states = pd.read_parquet(HOURLY_ROW_STATES_PATH)
    events = pd.read_parquet(AVAILABILITY_EVENTS_PATH)
    registry = pd.read_csv(STATION_REGISTRY_PATH)

    states = states.copy()
    events = events.copy()
    states["hour_utc"] = pd.to_datetime(states["hour_utc"], utc=True)
    events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True)
    events["end_utc"] = pd.to_datetime(events["end_utc"], utc=True)

    excluded = build_excluded_station_hours(events)

    states["is_active"] = ~states["row_state"].isin(INACTIVE_STATES)
    states["is_offline"] = states["row_state"] == "true_outage_candidate"
    states["station_hour_key"] = list(
        zip(states["station_id"], states["hour_utc"])
    )
    states["is_in_nw_event"] = states["station_hour_key"].isin(excluded)

    active_states = states[states["is_active"]].copy()
    clean_states = active_states[~active_states["is_in_nw_event"]].copy()

    offline_pivot = clean_states.pivot_table(
        index="hour_utc",
        columns="station_id",
        values="is_offline",
        aggfunc="first",
    )
    active_pivot = clean_states.pivot_table(
        index="hour_utc",
        columns="station_id",
        values="is_active",
        aggfunc="first",
    )

    coords: dict[str, tuple[float, float]] = {
        row["station_id"]: (row["latitude"], row["longitude"])
        for _, row in registry.iterrows()
    }

    rows: list[dict] = []
    station_ids = sorted(offline_pivot.columns.tolist())
    for sid_a, sid_b in combinations(station_ids, 2):
        if sid_a not in coords or sid_b not in coords:
            continue
        lat_a, lon_a = coords[sid_a]
        lat_b, lon_b = coords[sid_b]
        distance = haversine_km(lat_a, lon_a, lat_b, lon_b)

        both_active = (
            active_pivot[sid_a].fillna(False) & active_pivot[sid_b].fillna(False)
        )
        n_both_active = int(both_active.sum())
        if n_both_active == 0:
            continue

        offline_a = offline_pivot[sid_a].fillna(False) & both_active
        offline_b = offline_pivot[sid_b].fillna(False) & both_active

        n_offline_a = int(offline_a.sum())
        n_offline_b = int(offline_b.sum())
        n_both_offline = int((offline_a & offline_b).sum())

        p_a = n_offline_a / n_both_active
        p_b = n_offline_b / n_both_active
        expected_both = p_a * p_b * n_both_active
        ratio = (
            n_both_offline / expected_both if expected_both > 0 else None
        )

        rows.append({
            "station_a": sid_a,
            "station_b": sid_b,
            "distance_km": distance,
            "n_both_active_hours": n_both_active,
            "n_offline_a": n_offline_a,
            "n_offline_b": n_offline_b,
            "n_both_offline": n_both_offline,
            "p_offline_a": p_a,
            "p_offline_b": p_b,
            "expected_both_offline_if_independent": expected_both,
            "cooccurrence_ratio": ratio,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    bins = [0, 25, 50, 100, 200, 500, 2000]
    labels = [
        "0-25km", "25-50km", "50-100km",
        "100-200km", "200-500km", ">500km",
    ]
    df["distance_bucket"] = pd.cut(
        df["distance_km"], bins=bins, labels=labels, include_lowest=True
    )

    with_ratio = df.dropna(subset=["cooccurrence_ratio"])

    agg = with_ratio.groupby("distance_bucket", observed=False).agg(
        n_pairs=("cooccurrence_ratio", "count"),
        median_ratio=("cooccurrence_ratio", "median"),
        mean_ratio=("cooccurrence_ratio", "mean"),
        min_ratio=("cooccurrence_ratio", "min"),
        max_ratio=("cooccurrence_ratio", "max"),
    ).reindex(labels)

    lines: list[str] = []
    lines.append(
        "=== Local outage co-occurrence (excluding all network outage event hours) ==="
    )
    lines.append(f"Excluded station-hours: {len(excluded)}")
    lines.append(f"Total station pairs analyzed: {len(df)}")
    lines.append(f"Pairs with computable ratio: {len(with_ratio)}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  ratio = 1.0 means outages are independent")
    lines.append("  ratio > 1.0 means stations co-fail more often than chance")
    lines.append("")
    lines.append("=== Co-occurrence ratio by distance bucket ===")
    lines.append(agg.to_string())
    lines.append("")

    top_pairs = with_ratio.nlargest(15, "cooccurrence_ratio")
    lines.append("=== Top 15 pairs by co-occurrence ratio ===")
    cols = [
        "station_a", "station_b", "distance_km",
        "n_both_active_hours", "n_both_offline",
        "expected_both_offline_if_independent", "cooccurrence_ratio",
    ]
    lines.append(top_pairs[cols].to_string(index=False))
    lines.append("")

    nearby = df[df["distance_km"] < 25].dropna(subset=["cooccurrence_ratio"])
    if not nearby.empty:
        lines.append("=== All pairs within 25 km ===")
        lines.append(nearby[cols].to_string(index=False))

    report = "\n".join(lines) + "\n"
    OUTPUT_TXT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
