from __future__ import annotations

import pandas as pd
import pytest

from src.config.paths import (
    CANONICAL_COLUMN_ORDER,
    EXPECTED_FROZEN_N_COLS,
    EXPECTED_FROZEN_N_ROWS,
    EXPECTED_STATION_COUNT,
    FIGURES_DIR,
    MERGED_DATASET_PATH,
    PROCESSED_DIR,
    ensure_directories,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(MERGED_DATASET_PATH)


def test_required_directories_exist() -> None:
    ensure_directories()
    assert MERGED_DATASET_PATH.parent.exists()
    assert PROCESSED_DIR.exists()
    assert FIGURES_DIR.exists()


def test_merged_dataset_exists() -> None:
    assert MERGED_DATASET_PATH.exists(), (
        f"Missing merged dataset at {MERGED_DATASET_PATH}. "
        "Rename the current file to data/merged/station_hourly_merged.csv."
    )


def test_frozen_shape_and_schema(df: pd.DataFrame) -> None:
    assert df.shape == (EXPECTED_FROZEN_N_ROWS, EXPECTED_FROZEN_N_COLS)
    assert list(df.columns) == CANONICAL_COLUMN_ORDER


def test_station_hour_key_is_unique(df: pd.DataFrame) -> None:
    assert not df.duplicated(subset=["station_id", "hour_utc"]).any()
    assert df["station_id"].nunique() == EXPECTED_STATION_COUNT


def test_hour_utc_is_parseable_and_utc(df: pd.DataFrame) -> None:
    parsed = pd.to_datetime(df["hour_utc"], utc=True, errors="raise")
    assert parsed.notna().all()
    assert df["hour_utc"].str.endswith("+00:00").all()


def test_utc_columns_are_consistent(df: pd.DataFrame) -> None:
    assert (df["hour_utc"] == df["timestamp_utc_dt"]).all()

    hour_utc_without_offset = (
        pd.to_datetime(df["hour_utc"], utc=True)
        .dt.tz_convert("UTC")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    assert (hour_utc_without_offset == df["timestamp_utc"]).all()


def test_data_present_is_binary(df: pd.DataFrame) -> None:
    assert set(df["data_present"].dropna().unique()) <= {0, 1}


def test_timestamp_local_exists_for_live_rows(df: pd.DataFrame) -> None:
    live_rows = df["data_present"] == 1
    assert df.loc[live_rows, "timestamp_local"].notna().all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))