from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rules.external_features import (
    FEATURE_COLUMNS,
    build_external_features,
    feature_has_label_leak,
    solar_daily_relative_ratio,
    systemic_external_evidence,
)


def _residual_rows(stations: list[str], hours: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for station_id in stations:
        for time in hours:
            rows.append({"station_id": station_id, "time_utc": time})
    frame = pd.DataFrame(rows)
    for channel in ["pressure", "temp", "dewpoint", "wind", "solar"]:
        frame[f"z_{channel}"] = 0.0
        frame[f"r_{channel}"] = 0.0
        frame[f"base_{channel}"] = 0.0
        frame[f"bmad_{channel}"] = 1.0
    frame["clear_sky_ratio"] = 1.0
    return frame


def test_feature_frame_has_expected_columns_and_no_label_verdict_column() -> None:
    hours = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    residuals = _residual_rows(["A"], hours)

    result = build_external_features(residuals)

    assert result.columns.tolist() == FEATURE_COLUMNS
    assert not feature_has_label_leak(result)


def test_array_mean_and_valid_count_ignore_missing_channels() -> None:
    hours = pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")
    residuals = _residual_rows(["A"], hours)
    residuals.loc[0, "z_temp"] = 2.0
    residuals.loc[0, "z_dewpoint"] = np.nan
    residuals.loc[0, "z_wind"] = -4.0
    residuals.loc[0, "z_solar"] = np.nan

    result = build_external_features(residuals)

    assert result.loc[0, "ext_abs_z_array_mean"] == pytest.approx(3.0)
    assert result.loc[0, "ext_n_valid_array"] == 2


def test_rel_ratio_broadcasts_daily_value_to_each_hour() -> None:
    hours = pd.date_range("2026-01-01 06:00", periods=4, freq="h", tz="UTC")
    residuals = _residual_rows([f"S{i}" for i in range(8)], hours)
    residuals["clear_sky_ratio"] = np.where(residuals["station_id"].eq("S0"), 2.0, 1.0)

    result = build_external_features(residuals)
    s0 = result.loc[result["station_id"].eq("S0"), "rel_ratio_solar"]

    assert s0.notna().all()
    assert s0.nunique() == 1
    assert s0.iloc[0] == pytest.approx(2.0)


def test_solar_daily_relative_ratio_needs_fleet_support() -> None:
    hours = pd.date_range("2026-01-01 06:00", periods=4, freq="h", tz="UTC")
    residuals = _residual_rows(["A", "B"], hours)

    result = solar_daily_relative_ratio(residuals, min_fleet_stations=3)

    assert result["rel_ratio_solar"].isna().all()


def test_systemic_evidence_classifies_supported_single_benign_and_inconclusive() -> None:
    hours = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    residuals = _residual_rows(["A", "B", "C", "D"], hours)
    residuals.loc[residuals["station_id"].eq("A"), ["z_temp", "z_wind"]] = [4.0, 3.5]
    residuals.loc[residuals["station_id"].eq("B"), "z_solar"] = 3.2
    residuals.loc[residuals["station_id"].eq("C"), ["z_temp", "z_dewpoint", "z_wind", "z_solar"]] = 0.5
    residuals.loc[residuals["station_id"].eq("D"), ["z_temp", "z_dewpoint", "z_wind", "z_solar"]] = np.nan
    labeled = pd.DataFrame(
        {
            "station_id": ["A", "B", "C", "D"],
            "start_hour": [hours[0]] * 4,
            "end_hour": [hours[-1]] * 4,
            "duration_hours": [4.0] * 4,
            "affected_sensor_groups": ["x"] * 4,
            "label": ["systemic_array"] * 4,
            "tier": ["B"] * 4,
        }
    )

    result = systemic_external_evidence(
        residuals,
        labeled,
        expected_count_range=None,
    )
    tiers = dict(zip(result["station_id"], result["support_tier"]))

    assert tiers["A"] == "supported"
    assert tiers["B"] == "single_channel"
    assert tiers["C"] == "benign"
    assert tiers["D"] == "inconclusive"
