from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rules.feature_matrix import (
    build_feature_matrix,
    feature_has_forbidden_columns,
    materialize_statistical_features,
)


def _external() -> pd.DataFrame:
    hours = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "station_id": ["A", "A", "B"],
            "time_utc": hours,
            "z_pressure": [1.0, 2.0, 3.0],
            "r_pressure": [-1.0, -2.0, -3.0],
        }
    )


def _spatial() -> pd.DataFrame:
    hours = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "station_id": ["A", "B"],
            "time_utc": [hours[0], hours[1]],
            "z_spatial_pressure": [4.0, 5.0],
        }
    )


def _registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["A", "B"],
            "elevation": [10.0, 20.0],
        }
    )


def _neighbors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["A", "A"],
            "neighbor_id": ["B", "C"],
            "distance_km": [1.0, 2.0],
        }
    )


def test_join_preserves_spine_row_count_exactly() -> None:
    statistical = pd.DataFrame(
        {
            "station_id": ["A"],
            "time_utc": [pd.Timestamp("2026-01-01", tz="UTC")],
            "z_pressure": [9.0],
        }
    )

    result = build_feature_matrix(
        external_features=_external(),
        statistical_features=statistical,
        spatial_features=_spatial(),
        registry=_registry(),
        neighbors=_neighbors(),
    )

    assert len(result) == len(_external())


def test_statistical_columns_get_prefix_and_external_spatial_do_not_collide() -> None:
    statistical = pd.DataFrame(
        {
            "station_id": ["A"],
            "time_utc": [pd.Timestamp("2026-01-01", tz="UTC")],
            "z_pressure": [9.0],
        }
    )

    result = build_feature_matrix(
        external_features=_external(),
        statistical_features=statistical,
        spatial_features=_spatial(),
        registry=_registry(),
        neighbors=_neighbors(),
    )

    assert "stat_z_pressure" in result.columns
    assert "z_pressure" in result.columns
    assert "z_spatial_pressure" in result.columns
    assert result.loc[0, "stat_z_pressure"] == pytest.approx(9.0)


def test_hand_built_join_values_and_context() -> None:
    statistical = pd.DataFrame(
        {
            "station_id": ["A"],
            "time_utc": [pd.Timestamp("2026-01-01 01:00", tz="UTC")],
            "flag": [1.0],
        }
    )

    result = build_feature_matrix(
        external_features=_external(),
        statistical_features=statistical,
        spatial_features=_spatial(),
        registry=_registry(),
        neighbors=_neighbors(),
    )

    row = result.loc[result["time_utc"].eq(pd.Timestamp("2026-01-01 01:00", tz="UTC"))].iloc[0]
    assert row["r_pressure"] == pytest.approx(-2.0)
    assert row["stat_flag"] == pytest.approx(1.0)
    assert np.isnan(row["z_spatial_pressure"])
    assert row["ctx_elevation"] == pytest.approx(10.0)
    assert row["ctx_n_neighbors"] == 2


def test_no_forbidden_label_verdict_episode_columns() -> None:
    statistical = pd.DataFrame(
        {
            "station_id": ["A"],
            "time_utc": [pd.Timestamp("2026-01-01", tz="UTC")],
            "flag": [1.0],
        }
    )

    result = build_feature_matrix(
        external_features=_external(),
        statistical_features=statistical,
        spatial_features=_spatial(),
        registry=_registry(),
        neighbors=_neighbors(),
    )

    assert not feature_has_forbidden_columns(result)
    assert feature_has_forbidden_columns(result.assign(label="fault"))


def test_materialize_statistical_features_widens_scores_and_sensor_groups() -> None:
    merged = pd.DataFrame(
        {
            "station_id": ["A", "A"],
            "hour_utc": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        }
    )
    scores = pd.DataFrame(
        {
            "station_id": ["A", "A"],
            "hour_utc": [merged.loc[0, "hour_utc"], merged.loc[0, "hour_utc"]],
            "channel": ["temp_avg_c", "windspeed_avg_kmh"],
            "zscore": [3.0, 4.0],
            "rolling_variance": [0.5, 0.0],
            "iforest_score": [0.1, 0.2],
            "flag_zscore": [True, False],
            "flag_stuck": [False, True],
            "flag_iforest": [False, False],
            "flag_physical": [False, False],
            "flag_physical_suspect": [False, False],
            "flag": [True, True],
        }
    )

    result = materialize_statistical_features(
        scores=scores,
        merged=merged,
        output_path=None,
    )

    assert len(result) == 2
    assert result.loc[0, "zscore_temp_avg_c"] == pytest.approx(3.0)
    assert result.loc[0, "flag_stuck_windspeed_avg_kmh"] == pytest.approx(1.0)
    assert result.loc[0, "sensor_group_flag_thermo_hygrometer"] == pytest.approx(1.0)
    assert result.loc[0, "sensor_group_flag_anemometer"] == pytest.approx(1.0)
    assert np.isnan(result.loc[1, "zscore_temp_avg_c"])
