from __future__ import annotations

import pytest

from scripts import build_hourly_dataset, generate_report_assets
from scripts import evaluate_outage_risk
from scripts.evaluate_outage_risk import parse_args as parse_outage_args
from scripts.generate_report_assets import parse_args as parse_report_args
from scripts.train_hourly_detection import parse_args as parse_training_args
from scripts.tune_hourly_detection import parse_args as parse_tuning_args
from src.workflows import train_hourly_baseline, tune_hourly_detection


def test_training_front_door_routes_modes_without_consuming_runner_options() -> None:
    mode, remaining = parse_training_args(["baseline", "--seed", "7"])

    assert mode == "baseline"
    assert remaining == ["--seed", "7"]


def test_tuning_front_door_routes_resume_mode_without_consuming_runner_options() -> None:
    mode, remaining = parse_tuning_args(["resume", "--phase", "report"])

    assert mode == "resume"
    assert remaining == ["--phase", "report"]


def test_report_front_door_selects_one_figure_set() -> None:
    assert parse_report_args(["--set", "results"]) == "results"
    assert parse_report_args([]) == "methodology"


def test_report_all_preflights_result_inputs_before_writing_methodology_figures(monkeypatch) -> None:
    built: list[str] = []

    monkeypatch.setattr(generate_report_assets, "require_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        generate_report_assets,
        "require_result_inputs",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing result evidence")),
    )
    monkeypatch.setattr(generate_report_assets, "build_methodology", lambda: built.append("methodology"))
    monkeypatch.setattr(generate_report_assets, "build_results", lambda: built.append("results"))

    with pytest.raises(FileNotFoundError, match="missing result evidence"):
        generate_report_assets.main(["--set", "all"])

    assert built == []


def test_hourly_dataset_checks_all_required_inputs_before_creating_outputs(tmp_path) -> None:
    output = tmp_path / "hourly_labels.csv"

    with pytest.raises(FileNotFoundError) as error:
        build_hourly_dataset.main(
            [
                "--source",
                str(tmp_path / "missing_source.csv"),
                "--features",
                str(tmp_path / "missing_features.parquet"),
                "--labels",
                str(tmp_path / "missing_labels.csv"),
                "--labels-output",
                str(output),
            ],
        )

    message = str(error.value)
    assert "canonical merged dataset" in message
    assert "feature matrix" in message
    assert "live episode labels" in message
    assert not output.exists()


def test_baseline_training_reports_missing_tensors_before_creating_output_directory(tmp_path) -> None:
    output_dir = tmp_path / "baseline_output"

    with pytest.raises(FileNotFoundError) as error:
        train_hourly_baseline.main(
            [
                "--short-tensor",
                str(tmp_path / "missing_short.npz"),
                "--long-tensor",
                str(tmp_path / "missing_long.npz"),
                "--output-dir",
                str(output_dir),
            ],
        )

    assert "short hourly tensor" in str(error.value)
    assert "long hourly tensor" in str(error.value)
    assert not output_dir.exists()


def test_tuning_reports_all_core_prerequisites_before_creating_output_directory(tmp_path) -> None:
    output_dir = tmp_path / "tuning_output"

    with pytest.raises(FileNotFoundError) as error:
        tune_hourly_detection.main(
            [
                "--tensor",
                str(tmp_path / "missing_short.npz"),
                "--manifest",
                str(tmp_path / "missing_manifest.csv"),
                "--boosted-metrics",
                str(tmp_path / "missing_calibration.json"),
                "--prior-rgfn-metrics",
                str(tmp_path / "missing_rgfn.json"),
                "--output-dir",
                str(output_dir),
            ],
        )

    message = str(error.value)
    assert "short hourly tensor" in message
    assert "baseline split manifest" in message
    assert "calibrated baseline metrics" in message
    assert "prior RGFN metrics" in message
    assert not output_dir.exists()


def test_outage_front_door_accepts_an_output_directory() -> None:
    args = parse_outage_args(["--output-dir", "temporary-evaluation"])

    assert str(args.output_dir) == "temporary-evaluation"


def test_outage_evaluation_checks_inputs_before_creating_output_directory(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "evaluation_output"
    monkeypatch.setattr(evaluate_outage_risk, "HOURLY_ROW_STATES_PATH", tmp_path / "missing_states.parquet")

    with pytest.raises(FileNotFoundError) as error:
        evaluate_outage_risk.main(
            [
                "--events",
                str(tmp_path / "missing_events.parquet"),
                "--output-dir",
                str(output_dir),
            ],
        )

    assert "hourly availability states" in str(error.value)
    assert "availability events" in str(error.value)
    assert not output_dir.exists()
