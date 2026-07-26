from __future__ import annotations

import pandas as pd
import pytest

from scripts import build_labels


def test_rain_summary_is_false_when_no_rain_gauge_episode_exists() -> None:
    labels = pd.DataFrame(
        {
            "components": ["anemometer", "barometer"],
            "mechanisms": ["spike_impossible", "stuck_flatline"],
        },
    )

    assert build_labels._rain_summary(labels) == (0, False)


def test_missing_frozen_labels_still_writes_live_labels(tmp_path, monkeypatch) -> None:
    labels = pd.DataFrame({"episode_id": ["episode_001"], "label_state": ["benign"]})
    label_path = tmp_path / "episode_labels.csv"
    crosswalk_path = tmp_path / "label_crosswalk.csv"

    monkeypatch.setattr(build_labels, "LABELS_DIR", tmp_path)
    monkeypatch.setattr(build_labels, "LABEL_PATH", label_path)
    monkeypatch.setattr(build_labels, "CROSSWALK_PATH", crosswalk_path)
    monkeypatch.setattr(build_labels, "FROZEN_LABELS_PATH", tmp_path / "missing_frozen_labels.csv")

    crosswalk, summary = build_labels._write_labels_and_optional_crosswalk(labels)

    assert label_path.exists()
    assert pd.read_csv(label_path).to_dict("records") == labels.to_dict("records")
    assert crosswalk is None
    assert summary is None
    assert not crosswalk_path.exists()


def test_label_inputs_fail_before_the_label_build_when_evidence_is_missing(tmp_path, monkeypatch) -> None:
    merged_path = tmp_path / "station_hourly_merged.csv"
    merged_path.write_text("station_id,hour_utc\n", encoding="utf-8")
    missing_external = tmp_path / "external_residuals.parquet"
    missing_spatial = tmp_path / "spatial_residuals.parquet"
    monkeypatch.setattr(build_labels, "MERGED_DATASET_PATH", merged_path)
    monkeypatch.setattr(build_labels, "EXTERNAL_RESIDUALS_PATH", missing_external)
    monkeypatch.setattr(build_labels, "SPATIAL_RESIDUALS_PATH", missing_spatial)

    with pytest.raises(FileNotFoundError) as error:
        build_labels._require_label_inputs()

    message = str(error.value)
    assert "external residual evidence" in message
    assert "spatial residual evidence" in message
    assert "rebuild_detection_features.py" in message
