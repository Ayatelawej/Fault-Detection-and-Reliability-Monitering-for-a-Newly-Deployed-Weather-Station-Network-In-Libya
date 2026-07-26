from __future__ import annotations

import pandas as pd
import pytest

from src.config.paths import MERGED_DATASET_PATH
from src.rules.config import EXTERNAL_CHANNEL_MAP


def test_external_channel_map_keys_exist_in_merged_header() -> None:
    header = pd.read_csv(MERGED_DATASET_PATH, nrows=0).columns

    assert set(EXTERNAL_CHANNEL_MAP) == {
        "pressure_max_hpa",
        "temp_avg_c",
        "dewpoint_avg_c",
        "windspeed_avg_kmh",
        "solar_radiation_high_wm2",
    }
    assert set(EXTERNAL_CHANNEL_MAP) <= set(header)


def test_external_channel_map_conversions() -> None:
    expected = {
        "pressure_max_hpa": 1010.0,
        "temp_avg_c": 25.0,
        "dewpoint_avg_c": 10.0,
        "windspeed_avg_kmh": 10.0,
        "solar_radiation_high_wm2": 500.0,
    }
    raw = {
        "pressure_max_hpa": 1010.0,
        "temp_avg_c": 25.0,
        "dewpoint_avg_c": 10.0,
        "windspeed_avg_kmh": 36.0,
        "solar_radiation_high_wm2": 500.0,
    }

    for channel, value in raw.items():
        converted = EXTERNAL_CHANNEL_MAP[channel]["conversion"](value)
        assert converted == pytest.approx(expected[channel])
