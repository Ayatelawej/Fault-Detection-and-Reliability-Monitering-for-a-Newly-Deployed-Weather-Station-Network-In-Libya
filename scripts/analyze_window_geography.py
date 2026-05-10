from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import (
    NETWORK_OUTAGE_WINDOWS_PATH,
    STATION_REGISTRY_PATH,
)

OUTPUT_DIR = Path("docs/phase2_investigation_checks")
OUTPUT_TXT = OUTPUT_DIR / "window_geography.txt"
OUTPUT_CSV = OUTPUT_DIR / "window_geography.csv"

EARTH_RADIUS_KM = 6371.0


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


def analyze_window_geography(
    window_id: str,
    window_meta: pd.Series,
    station_coords: dict[str, tuple[float, float]],
) -> dict:
    result: dict = {
        "window_id": window_id,
        "window_start_utc": window_meta["window_start_utc"],
        "station_count": int(window_meta["station_count"]),
        "outage_class": window_meta.get("outage_class", "unknown"),
    }

    sids = window_meta["station_ids"].split(";")
    coords: list[tuple[str, float, float]] = []
    for sid in sids:
        if sid in station_coords:
            lat, lon = station_coords[sid]
            coords.append((sid, lat, lon))
    result["n_stations_resolved"] = len(coords)

    if len(coords) < 2:
        result["max_pairwise_km"] = None
        result["min_pairwise_km"] = None
        result["median_pairwise_km"] = None
        result["lat_min"] = None
        result["lat_max"] = None
        result["lon_min"] = None
        result["lon_max"] = None
        result["lat_range"] = None
        result["lon_range"] = None
        result["status"] = "fewer_than_2_stations"
        return result

    distances: list[float] = []
    for (sid_a, lat_a, lon_a), (sid_b, lat_b, lon_b) in combinations(coords, 2):
        distances.append(haversine_km(lat_a, lon_a, lat_b, lon_b))

    lats = [c[1] for c in coords]
    lons = [c[2] for c in coords]
    result["max_pairwise_km"] = float(max(distances))
    result["min_pairwise_km"] = float(min(distances))
    result["median_pairwise_km"] = float(np.median(distances))
    result["lat_min"] = float(min(lats))
    result["lat_max"] = float(max(lats))
    result["lon_min"] = float(min(lons))
    result["lon_max"] = float(max(lons))
    result["lat_range"] = result["lat_max"] - result["lat_min"]
    result["lon_range"] = result["lon_max"] - result["lon_min"]
    result["status"] = "ok"
    return result


def main() -> None:
    windows = pd.read_csv(NETWORK_OUTAGE_WINDOWS_PATH)
    windows["window_start_utc"] = pd.to_datetime(
        windows["window_start_utc"], utc=True
    )
    windows["start_hour"] = windows["window_start_utc"].dt.hour
    windows = windows.sort_values("window_start_utc").reset_index(drop=True)

    registry = pd.read_csv(STATION_REGISTRY_PATH)
    station_coords = {
        row["station_id"]: (row["latitude"], row["longitude"])
        for _, row in registry.iterrows()
    }

    results = []
    for _, window in windows.iterrows():
        results.append(
            analyze_window_geography(
                window["window_id"], window, station_coords
            )
        )

    df = pd.DataFrame(results)
    df["window_start_utc"] = pd.to_datetime(df["window_start_utc"], utc=True)
    df["start_hour"] = df["window_start_utc"].dt.hour
    df["is_midnight"] = df["start_hour"].isin([22, 23])
    df.to_csv(OUTPUT_CSV, index=False)

    midnight = df[df["is_midnight"] & (df["status"] == "ok")].copy()
    nonmidnight = df[~df["is_midnight"] & (df["status"] == "ok")].copy()

    lines: list[str] = []
    lines.append("=== Window geography analysis ===")
    lines.append(f"Total windows analyzed: {len(df)}")
    lines.append(f"  Midnight (22-23 UTC): {len(midnight)}")
    lines.append(f"  Non-midnight: {len(nonmidnight)}")
    lines.append("")

    if not midnight.empty:
        lines.append("=== Midnight events: max pairwise distance ===")
        mx = midnight["max_pairwise_km"].dropna()
        lines.append(f"Median max distance: {mx.median():.1f} km")
        lines.append(f"Mean max distance: {mx.mean():.1f} km")
        lines.append(f"Range: {mx.min():.1f} km to {mx.max():.1f} km")
        lines.append("")

    if not nonmidnight.empty:
        lines.append("=== Non-midnight events: max pairwise distance ===")
        nx = nonmidnight["max_pairwise_km"].dropna()
        lines.append(f"Median max distance: {nx.median():.1f} km")
        lines.append(f"Mean max distance: {nx.mean():.1f} km")
        lines.append(f"Range: {nx.min():.1f} km to {nx.max():.1f} km")
        lines.append("")

        lines.append("=== Non-midnight events: distance distribution ===")
        bins = [0, 25, 50, 100, 200, 500, 2000]
        labels = [
            "0-25km", "25-50km", "50-100km",
            "100-200km", "200-500km", ">500km",
        ]
        binned = pd.cut(nx, bins=bins, labels=labels, include_lowest=True)
        lines.append(
            binned.value_counts().reindex(labels).fillna(0).astype(int).to_string()
        )
        lines.append("")

    lines.append("=== Per-window geography detail ===")
    cols = [
        "window_id", "window_start_utc", "is_midnight", "station_count",
        "n_stations_resolved", "max_pairwise_km", "median_pairwise_km",
        "lat_range", "lon_range", "status",
    ]
    present_cols = [c for c in cols if c in df.columns]
    lines.append(df[present_cols].to_string(index=False))

    report = "\n".join(lines) + "\n"
    OUTPUT_TXT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
