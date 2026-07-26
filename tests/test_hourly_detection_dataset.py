from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model.feature_spec import CONTINUOUS_FEATURES, RULE_EVIDENCE_FLAGS, STATIC_FEATURES
from src.model.hourly_detection import (
    MASK_MODE_PER_FEATURE,
    MASK_MODE_PER_HOUR,
    build_hourly_examples,
    build_hourly_labels,
)


def _hourly_frame() -> pd.DataFrame:
    hours = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    rows = []
    for index, hour in enumerate(hours):
        row: dict[str, object] = {
            "station_id": "S1",
            "hour": hour,
            "data_present": 1,
            "n_raw_records": 1,
        }
        for feature_index, column in enumerate(CONTINUOUS_FEATURES):
            row[column] = float(index + feature_index / 100)
        for feature_index, column in enumerate(STATIC_FEATURES):
            row[column] = float(feature_index + 1)
        for column in RULE_EVIDENCE_FLAGS:
            row[column] = 0.0
        row["stat_flag_zscore_temp_avg_c"] = float(index == 0)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.loc[3, "r_pressure"] = 99999.0
    return frame


def _labelled_episodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode_id": "fault_1",
                "station_id": "S1",
                "start_hour": "2026-01-01 01:00:00+00:00",
                "end_hour": "2026-01-01 02:00:00+00:00",
                "binary_fault": 1,
                "label_state": "fault",
                "mechanisms": "spike_impossible|statistical_anomaly",
                "components": "barometer|rain_gauge",
            },
            {
                "episode_id": "review_1",
                "station_id": "S1",
                "start_hour": "2026-01-01 03:00:00+00:00",
                "end_hour": "2026-01-01 03:00:00+00:00",
                "binary_fault": 0,
                "label_state": "borderline_review",
                "mechanisms": "",
                "components": "",
            },
        ]
    )


def test_hourly_labels_assign_fault_review_benign_and_clean() -> None:
    labels = build_hourly_labels(_hourly_frame(), _labelled_episodes())
    by_hour = labels.set_index("hour")
    hour_0 = pd.Timestamp("2026-01-01 00:00:00+00:00")
    hour_1 = pd.Timestamp("2026-01-01 01:00:00+00:00")
    hour_2 = pd.Timestamp("2026-01-01 02:00:00+00:00")
    hour_3 = pd.Timestamp("2026-01-01 03:00:00+00:00")
    hour_4 = pd.Timestamp("2026-01-01 04:00:00+00:00")

    assert by_hour.loc[hour_0, "display_state"] == "benign"
    assert by_hour.loc[hour_0, "fault_hour"] == 0
    assert by_hour.loc[hour_0, "detectors_fired"] == "robust_zscore"
    assert by_hour.loc[hour_1, "display_state"] == "fault"
    assert by_hour.loc[hour_1, "fault_hour"] == 1
    assert by_hour.loc[hour_2, "fault_hour"] == 1
    assert by_hour.loc[hour_1, "mechanisms"] == "spike_impossible|statistical_anomaly"
    assert by_hour.loc[hour_1, "components"] == "barometer|rain_gauge"
    assert by_hour.loc[hour_3, "display_state"] == "excluded"
    assert pd.isna(by_hour.loc[hour_3, "fault_hour"])
    assert not by_hour.loc[hour_3, "training_eligible"]
    assert by_hour.loc[hour_4, "display_state"] == "clean"
    assert by_hour.loc[hour_4, "fault_hour"] == 0


def test_hourly_windows_are_past_only_and_left_padded() -> None:
    hourly = _hourly_frame()
    labels = build_hourly_labels(hourly, _labelled_episodes())
    examples_short = build_hourly_examples(hourly, labels, window_hours=7)
    examples_long = build_hourly_examples(hourly, labels, window_hours=49)
    target_hour = "2026-01-01 02:00:00+00:00"
    short_index = examples_short["hour"].tolist().index(target_hour)
    long_index = examples_long["hour"].tolist().index(target_hour)
    pressure_index = CONTINUOUS_FEATURES.index("r_pressure")
    first_index = examples_short["hour"].tolist().index("2026-01-01 00:00:00+00:00")

    assert examples_short["X_cont"].shape == (5, 7, len(CONTINUOUS_FEATURES))
    assert examples_long["X_cont"].shape == (5, 49, len(CONTINUOUS_FEATURES))
    assert examples_short["X_cont"][short_index, -1, pressure_index] == 2.0
    assert not np.any(examples_short["X_cont"][short_index, :, pressure_index] == 99999.0)
    assert examples_long["X_cont"][long_index, -1, pressure_index] == 2.0
    assert not np.any(examples_long["X_cont"][long_index, :, pressure_index] == 99999.0)
    assert examples_short["mask"][first_index, :-1, 0].sum() == 0.0
    assert examples_short["mask"][first_index, -1, 0] == 1.0
    assert examples_long["mask"][long_index, :46, 0].sum() == 0.0
    assert examples_long["mask"][long_index, 46:, 0].sum() == 3.0
    assert "2026-01-01 03:00:00+00:00" not in examples_short["hour"].tolist()


def test_hours_without_measurements_are_excluded_from_tensors() -> None:
    hourly = _hourly_frame()
    hourly.loc[5, "data_present"] = 0
    labels = build_hourly_labels(hourly, _labelled_episodes())
    missing_hour = pd.Timestamp("2026-01-01 05:00:00+00:00")
    by_hour = labels.set_index("hour")
    examples = build_hourly_examples(hourly, labels, window_hours=7)

    assert by_hour.loc[missing_hour, "display_state"] == "excluded"
    assert pd.isna(by_hour.loc[missing_hour, "fault_hour"])
    assert str(missing_hour) not in examples["hour"].tolist()


def test_hourly_examples_support_both_mask_modes_without_changing_default() -> None:
    hourly = _hourly_frame()
    labels = build_hourly_labels(hourly, _labelled_episodes())
    default_examples = build_hourly_examples(hourly, labels, window_hours=7)
    explicit_per_hour = build_hourly_examples(
        hourly,
        labels,
        window_hours=7,
        mask_mode=MASK_MODE_PER_HOUR,
    )
    per_feature = build_hourly_examples(
        hourly,
        labels,
        window_hours=7,
        mask_mode=MASK_MODE_PER_FEATURE,
    )

    assert set(default_examples) == set(explicit_per_hour)
    for key in default_examples:
        left = default_examples[key]
        right = explicit_per_hour[key]
        if np.issubdtype(np.asarray(left).dtype, np.inexact):
            assert np.array_equal(left, right, equal_nan=True)
        else:
            assert np.array_equal(left, right)
    assert default_examples["mask"].shape == (5, 7, 1)
    assert "mask_per_hour" not in default_examples

    assert per_feature["mask"].shape == per_feature["X_cont"].shape
    assert np.array_equal(
        per_feature["mask"],
        (~np.isnan(per_feature["X_cont"])).astype(np.float32),
    )
    assert np.array_equal(per_feature["mask_per_hour"], default_examples["mask"])
    assert per_feature["mask"][0, :-1].sum() == 0.0
    assert list(per_feature["mask_feature_names"]) == list(CONTINUOUS_FEATURES)
    assert list(per_feature["mask_per_hour_feature_names"]) == ["row_present"]

    with pytest.raises(ValueError, match="unknown hourly mask mode"):
        build_hourly_examples(hourly, labels, window_hours=7, mask_mode="invalid")
