from __future__ import annotations

import pytest

from src.workflows.rebuild_detection_features import rebuild_detection_features


def test_rebuild_requires_reference_inputs_before_writing_outputs(tmp_path) -> None:
    merged_path = tmp_path / "station_hourly_merged.csv"
    registry_path = tmp_path / "station_registry.csv"
    output_path = tmp_path / "outputs" / "scores.parquet"
    merged_path.write_text("station_id,hour_utc\n", encoding="utf-8")
    registry_path.write_text("station_id\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="public reference parquet"):
        rebuild_detection_features(
            merged_path=merged_path,
            registry_path=registry_path,
            reference_dir=tmp_path / "missing_reference",
            five_min_dir=tmp_path / "missing_five_minute",
            outputs={"statistical_scores": output_path},
        )

    assert not output_path.exists()
