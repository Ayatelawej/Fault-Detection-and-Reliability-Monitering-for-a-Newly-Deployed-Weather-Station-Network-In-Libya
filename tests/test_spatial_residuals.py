from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rules.spatial_residuals import (
    build_all,
    build_neighbor_graph,
    haversine_km,
    neighbor_median,
)


def test_haversine_known_equator_degree_distance() -> None:
    result = haversine_km(0.0, 0.0, 0.0, 1.0)

    assert result == pytest.approx(111.195, abs=0.2)


def test_radius_filter_keeps_in_range_and_drops_out_of_range_pairs() -> None:
    registry = pd.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "latitude": [0.0, 0.0, 0.0],
            "longitude": [0.0, 0.5, 5.0],
        }
    )

    result = build_neighbor_graph(registry, radius_km=100.0)
    pairs = set(zip(result["station_id"], result["neighbor_id"]))

    assert ("A", "B") in pairs
    assert ("B", "A") in pairs
    assert ("A", "C") not in pairs


def test_graph_is_symmetric() -> None:
    registry = pd.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "latitude": [0.0, 0.0, 1.0],
            "longitude": [0.0, 0.5, 0.5],
        }
    )

    result = build_neighbor_graph(registry, radius_km=200.0)
    pairs = set(zip(result["station_id"], result["neighbor_id"]))

    for station_id, neighbor_id in pairs:
        assert (neighbor_id, station_id) in pairs


def test_isolated_station_yields_all_nan_spatial_columns() -> None:
    hours = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    registry = pd.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "latitude": [0.0, 10.0, 20.0],
            "longitude": [0.0, 10.0, 20.0],
        }
    )
    merged = pd.DataFrame(
        {
            "station_id": ["A"] * 4,
            "hour_utc": hours,
            "pressure_max_hpa": [1.0, 2.0, 3.0, 4.0],
            "temp_avg_c": [1.0, 2.0, 3.0, 4.0],
            "dewpoint_avg_c": [1.0, 2.0, 3.0, 4.0],
        }
    )
    graph = build_neighbor_graph(registry, radius_km=50.0)

    result = build_all(
        merged,
        registry,
        graph,
        hours,
        baseline_window_hours=3,
        baseline_min_hours=3,
    )
    isolated = result.loc[result["station_id"].eq("A")]

    assert isolated["r_spatial_pressure"].isna().all()
    assert isolated["z_spatial_pressure"].isna().all()


def test_neighbor_median_requires_present_neighbors_and_returns_median() -> None:
    hours = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    merged = pd.DataFrame(
        {
            "station_id": ["A", "A", "B", "B", "C", "C"],
            "hour_utc": [hours[0], hours[1]] * 3,
            "pressure_max_hpa": [10.0, 10.0, 20.0, np.nan, 30.0, 40.0],
        }
    )
    graph = pd.DataFrame(
        {
            "station_id": ["A", "A", "B", "B", "C", "C"],
            "neighbor_id": ["B", "C", "A", "C", "A", "B"],
            "distance_km": [10.0] * 6,
        }
    )

    result = neighbor_median(merged, "A", "pressure", graph, hours)

    assert result.iloc[0] == pytest.approx(25.0)
    assert pd.isna(result.iloc[1])


def test_spatial_residual_baseline_and_z_are_hand_computed() -> None:
    hours = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    rows = []
    values = {
        "A": [11.0, 12.0, 13.0, 14.0, 20.0],
        "B": [10.0] * 5,
        "C": [10.0] * 5,
    }
    for station_id, series in values.items():
        for time, value in zip(hours, series):
            rows.append(
                {
                    "station_id": station_id,
                    "hour_utc": time,
                    "pressure_max_hpa": value,
                    "temp_avg_c": value,
                    "dewpoint_avg_c": value,
                }
            )
    merged = pd.DataFrame(rows)
    registry = pd.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "latitude": [0.0, 0.0, 0.0],
            "longitude": [0.0, 0.1, 0.2],
        }
    )
    graph = build_neighbor_graph(registry, radius_km=50.0)

    result = build_all(
        merged,
        registry,
        graph,
        hours,
        baseline_window_hours=3,
        baseline_min_hours=3,
    )
    station = result.loc[result["station_id"].eq("A")].reset_index(drop=True)

    assert station.loc[2, "r_spatial_pressure"] == pytest.approx(3.0)
    assert station.loc[2, "base_spatial_pressure"] == pytest.approx(2.0)
    assert station.loc[2, "bmad_spatial_pressure"] == pytest.approx(1.0)
    assert station.loc[4, "base_spatial_pressure"] == pytest.approx(4.0)
    assert station.loc[4, "z_spatial_pressure"] == pytest.approx(6.0)
