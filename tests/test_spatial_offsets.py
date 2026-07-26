from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rules.review_queue import OUTPUT_COLUMNS
from src.rules.spatial_offsets import (
    add_hour_flags,
    apply_external_agreement_gate,
    build_spatial_features,
    build_spatial_queue,
    feature_has_label_leak,
    offset_episodes_for_station,
    spatial_status,
)
from src.rules.spatial_residuals import neighbor_median


def _spatial_frame(periods: int, station_id: str = "STA") -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "station_id": station_id,
            "time_utc": times,
            "r_spatial_pressure": [0.0] * periods,
            "base_spatial_pressure": [0.0] * periods,
            "bmad_spatial_pressure": [1.0] * periods,
            "z_spatial_pressure": [0.0] * periods,
            "z_spatial_temp": [0.0] * periods,
            "z_spatial_dewpoint": [0.0] * periods,
        }
    )


def test_physical_floor_blocks_small_level() -> None:
    frame = _spatial_frame(1)
    frame["base_spatial_pressure"] = 3.9

    result = add_hour_flags(frame)

    assert not result.loc[0, "spatial_offset_flag"]


def test_stability_ceiling_blocks_noisy_large_level() -> None:
    frame = _spatial_frame(1)
    frame["base_spatial_pressure"] = 10.0
    frame["bmad_spatial_pressure"] = 4.1

    result = add_hour_flags(frame)

    assert not result.loc[0, "spatial_offset_flag"]


def test_density_and_min_days_rules() -> None:
    frame = _spatial_frame(200)
    frame = add_hour_flags(frame.assign(base_spatial_pressure=5.0))
    dense = offset_episodes_for_station(frame, "STA")
    sparse = offset_episodes_for_station(frame.iloc[[0, 199]].copy(), "STA")
    short = offset_episodes_for_station(frame.iloc[:24].copy(), "STA")

    assert len(dense) == 1
    assert sparse.empty
    assert short.empty


def test_isolated_station_status_and_no_episode() -> None:
    frame = _spatial_frame(200)
    frame["r_spatial_pressure"] = np.nan
    frame["base_spatial_pressure"] = np.nan
    frame["bmad_spatial_pressure"] = np.nan
    episodes = offset_episodes_for_station(add_hour_flags(frame), "STA")

    assert spatial_status(frame, episodes) == "isolated"
    assert episodes.empty


def test_insufficient_status_yields_no_episode_status() -> None:
    frame = _spatial_frame(200)
    frame["base_spatial_pressure"] = 5.0
    episodes = offset_episodes_for_station(add_hour_flags(frame), "STA")

    assert spatial_status(frame, episodes) == "insufficient"


def test_queue_writer_matches_output_columns() -> None:
    episodes = pd.DataFrame(
        [
            {
                "station_id": "STA",
                "channel": "pressure",
                "start_hour": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "end_hour": pd.Timestamp("2026-01-08 00:00", tz="UTC"),
                "duration_hours": 169.0,
                "n_flagged_hours": 169,
                "tier": "HIGH",
                "level_p10": -7.0,
                "level_p50": -6.0,
                "level_p90": -5.0,
                "peak_abs_level": 8.0,
                "median_bmad": 1.0,
            }
        ]
    )

    result = build_spatial_queue(episodes)

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result.loc[0, "label"] == "spatial_anomaly"
    assert result.loc[0, "dominant_detector"] == "spatial_offset"
    assert "channel=pressure" in result.loc[0, "reasons"]


def test_feature_frame_expected_columns_and_no_label_verdict() -> None:
    frame = _spatial_frame(2)
    counts = pd.Series([2, 3], index=frame.index)

    result = build_spatial_features(frame, counts)

    assert result.columns.tolist() == [
        "station_id",
        "time_utc",
        "z_spatial_pressure",
        "z_spatial_temp",
        "z_spatial_dewpoint",
        "spatial_offset_level_pressure",
        "n_neighbors_present_pressure",
        "spatial_isolated",
    ]
    assert result["n_neighbors_present_pressure"].tolist() == [2, 3]
    assert not feature_has_label_leak(result)


def test_excluded_station_does_not_contribute_to_neighbor_median() -> None:
    hours = pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")
    merged = pd.DataFrame(
        {
            "station_id": ["A", "IBIRAL3", "GOOD"],
            "hour_utc": [hours[0], hours[0], hours[0]],
            "pressure_max_hpa": [10.0, 0.0, 20.0],
        }
    )
    graph = pd.DataFrame(
        {
            "station_id": ["A", "A"],
            "neighbor_id": ["IBIRAL3", "GOOD"],
            "distance_km": [10.0, 20.0],
        }
    )

    result = neighbor_median(
        merged,
        "A",
        "pressure",
        graph,
        hours,
        min_neighbors_present=1,
        min_neighbors=1,
    )

    assert result.iloc[0] == pytest.approx(20.0)


def test_external_gate_drops_positive_episode_not_confirmed_by_era5() -> None:
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    episodes = pd.DataFrame(
        [
            {
                "station_id": "STA",
                "channel": "pressure",
                "start_hour": start,
                "end_hour": start + pd.Timedelta(hours=2),
                "duration_hours": 3.0,
                "n_flagged_hours": 3,
                "tier": "MED",
                "level_p10": 4.0,
                "level_p50": 5.0,
                "level_p90": 6.0,
                "peak_abs_level": 6.0,
                "median_bmad": 1.0,
            }
        ]
    )
    external = pd.DataFrame(
        {
            "station_id": ["STA"] * 3,
            "time_utc": pd.date_range(start, periods=3, freq="h"),
            "r_pressure": [-3.0, -2.5, -3.5],
        }
    )

    kept, gated = apply_external_agreement_gate(episodes, external)

    assert kept.empty
    assert len(gated) == 1
    assert gated.loc[0, "gate_reason"] == "positive-not-confirmed"


def test_external_gate_keeps_negative_episode_confirmed_by_era5() -> None:
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    episodes = pd.DataFrame(
        [
            {
                "station_id": "STA",
                "channel": "pressure",
                "start_hour": start,
                "end_hour": start + pd.Timedelta(hours=2),
                "duration_hours": 3.0,
                "n_flagged_hours": 3,
                "tier": "HIGH",
                "level_p10": -8.0,
                "level_p50": -7.0,
                "level_p90": -6.0,
                "peak_abs_level": 8.0,
                "median_bmad": 1.0,
            }
        ]
    )
    external = pd.DataFrame(
        {
            "station_id": ["STA"] * 3,
            "time_utc": pd.date_range(start, periods=3, freq="h"),
            "r_pressure": [-4.0, -3.0, -5.0],
        }
    )

    kept, gated = apply_external_agreement_gate(episodes, external)

    assert len(kept) == 1
    assert gated.empty
    assert kept.loc[0, "external_r_pressure_median"] == pytest.approx(-4.0)
