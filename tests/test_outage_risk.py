from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.availability.risk_dataset import FEATURE_COLUMNS, build_risk_dataset
from src.availability.risk_eval import event_recall
from src.availability.risk_model import flicker_predict
from src.features.row_state import (
    ROW_STATE_COMPLETE,
    ROW_STATE_PARTIAL,
    ROW_STATE_TERMINAL_PADDED,
    ROW_STATE_TRUE_OUTAGE,
    ROW_STATE_WARMUP,
)


def _grid(states: list[str], station_id: str = "S1", start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": station_id,
            "hour_utc": pd.date_range(start, periods=len(states), freq="h", tz="UTC"),
            "row_state": states,
        }
    )


def test_synthetic_grid_label_correctness_per_horizon() -> None:
    states = [ROW_STATE_COMPLETE] * 4 + [ROW_STATE_TRUE_OUTAGE] + [ROW_STATE_COMPLETE] * 25
    dataset = build_risk_dataset(_grid(states), horizons=(6, 12, 24))
    frame_6 = dataset.for_horizon(6)
    frame_12 = dataset.for_horizon(12)
    first = pd.Timestamp("2026-01-01T00:00:00Z")

    assert int(frame_6.loc[frame_6["hour_utc"].eq(first), "y"].iloc[0]) == 1
    assert int(frame_6.loc[frame_6["hour_utc"].eq(first + pd.Timedelta(hours=5)), "y"].iloc[0]) == 0
    assert int(frame_12.loc[frame_12["hour_utc"].eq(first), "y"].iloc[0]) == 1


def test_eligibility_excludes_outages_warmup_and_terminal_padding() -> None:
    states = [
        ROW_STATE_WARMUP,
        ROW_STATE_COMPLETE,
        ROW_STATE_TRUE_OUTAGE,
        ROW_STATE_PARTIAL,
        ROW_STATE_TERMINAL_PADDED,
        ROW_STATE_COMPLETE,
        ROW_STATE_COMPLETE,
        ROW_STATE_COMPLETE,
        ROW_STATE_COMPLETE,
    ]
    dataset = build_risk_dataset(_grid(states), horizons=(2,))
    frame = dataset.for_horizon(2)

    assert set(frame["hour_utc"]) == {
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T03:00:00Z"),
        pd.Timestamp("2026-01-01T05:00:00Z"),
        pd.Timestamp("2026-01-01T06:00:00Z"),
    }


def test_right_censoring_drops_per_horizon() -> None:
    states = [ROW_STATE_COMPLETE] * 10
    dataset = build_risk_dataset(_grid(states), horizons=(6,))
    frame = dataset.for_horizon(6)

    assert len(frame) == 4
    assert frame["hour_utc"].max() == pd.Timestamp("2026-01-01T03:00:00Z")


def test_future_randomization_does_not_change_features_at_t() -> None:
    states = [ROW_STATE_COMPLETE] * 10 + [ROW_STATE_TRUE_OUTAGE] * 3 + [ROW_STATE_COMPLETE] * 20
    original = _grid(states)
    randomized = original.copy()
    cutoff = pd.Timestamp("2026-01-01T08:00:00Z")
    mask = randomized["hour_utc"].gt(cutoff)
    shuffled = randomized.loc[mask, "row_state"].sample(frac=1.0, random_state=5).to_numpy()
    randomized.loc[mask, "row_state"] = shuffled
    first = build_risk_dataset(original, horizons=(6,)).frame
    second = build_risk_dataset(randomized, horizons=(6,)).frame
    first_row = first.loc[first["hour_utc"].eq(cutoff), FEATURE_COLUMNS].iloc[0]
    second_row = second.loc[second["hour_utc"].eq(cutoff), FEATURE_COLUMNS].iloc[0]

    pd.testing.assert_series_equal(first_row, second_row, check_names=False)


def test_flicker_baseline_correctness() -> None:
    frame = pd.DataFrame({"trailing_missing_frac_24h": [0.0, 0.1, 1.0]})
    prediction = flicker_predict(frame)

    np.testing.assert_array_equal(prediction.pred, np.asarray([0, 1, 1]))


def test_event_recall_correctness_on_synthetic_event() -> None:
    test_frame = pd.DataFrame(
        {
            "station_id": ["S1"] * 5,
            "hour_utc": pd.date_range("2026-03-16T00:00:00Z", periods=5, freq="h", tz="UTC"),
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "station_id": ["S1"],
            "start_utc": [pd.Timestamp("2026-03-16T04:00:00Z")],
            "end_utc": [pd.Timestamp("2026-03-16T05:00:00Z")],
            "duration_hours": [2],
            "outage_class": ["local"],
        }
    )
    pred = np.asarray([0, 1, 0, 0, 0], dtype=int)
    result = event_recall(test_frame, events, 4, pred)

    assert result["n_test_events"] == 1
    assert result["event_recall"] == pytest.approx(1.0)
    assert result["median_lead_time_h"] == pytest.approx(3.0)
