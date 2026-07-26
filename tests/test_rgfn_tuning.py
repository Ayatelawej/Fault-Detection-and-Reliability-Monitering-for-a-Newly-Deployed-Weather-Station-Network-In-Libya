from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import src.workflows.tune_hourly_detection as tuning_workflow
from src.model.feature_spec import CONTINUOUS_FEATURES, RULE_EVIDENCE_FLAGS, STATIC_FEATURES, rule_evidence_feature_names
from src.model.hourly_detection import MASK_MODE_PER_FEATURE, MASK_MODE_PER_HOUR
from src.model.hourly_rgfn_tuning_features import (
    CAUSAL_RULE_EVIDENCE_FEATURE_NAMES,
    CAUSAL_STATIC_FEATURE_NAMES,
    augment_hourly_rgfn_examples,
)
from src.model.hourly_rgfn_tuning import (
    HourlyRgfnTuningConfig,
    build_tuned_hourly_rgfn,
    final_evaluate_hourly_rgfn_arm,
    finalize_architecture_search,
    master_tuning_comparison_frame,
    prepare_tuning_split,
    run_feature_only_arm,
    screen_architecture_search,
)


def _examples() -> dict[str, np.ndarray]:
    x_cont = np.full((1, 7, len(CONTINUOUS_FEATURES)), np.nan, dtype=np.float32)
    mask = np.zeros((1, 7, 1), dtype=np.float32)
    mask[0, 4:, 0] = 1.0
    source_values = {
        "r_pressure": [-1.0, -2.0, -3.0],
        "offset_level_pressure": [2.0, 3.0, 4.0],
        "z_spatial_pressure": [3.0, 4.0, 5.0],
        "ext_abs_z_array_mean": [4.0, 5.0, 6.0],
        "spatial_offset_level_pressure": [-2.0, -3.0, -4.0],
    }
    for name, values in source_values.items():
        x_cont[0, 4:, CONTINUOUS_FEATURES.index(name)] = values
    return {
        "X_cont": x_cont,
        "mask": mask,
        "time_since_last": np.zeros((1, 7, 1), dtype=np.float32),
        "static": np.asarray([[10.0, 2.0, 0.0]], dtype=np.float32),
        "rule_evidence": np.arange(len(RULE_EVIDENCE_FLAGS) * 2, dtype=np.float32).reshape(1, -1),
        "station_id": np.asarray(["S1"], dtype=object),
        "hour": np.asarray(["2026-01-01 02:00:00+00:00"], dtype=object),
        "static_feature_names": np.asarray(STATIC_FEATURES, dtype=object),
        "rule_evidence_feature_names": np.asarray(rule_evidence_feature_names(), dtype=object),
    }


def _raw_hourly() -> pd.DataFrame:
    hours = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "station_id": ["S1"] * len(hours),
            "hour": hours,
            "data_present": [1] * len(hours),
            "stat_flag_zscore_pressure_max_hpa": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "stat_sensor_group_flag_barometer": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "stat_sensor_group_flag_light_uv": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    return frame


def test_causal_augmentation_keeps_base_inputs_and_ignores_later_raw_values() -> None:
    examples = _examples()
    raw = _raw_hourly()
    original = {key: value.copy() for key, value in examples.items() if isinstance(value, np.ndarray)}

    first = augment_hourly_rgfn_examples(examples, raw)
    changed = raw.copy()
    later = changed["hour"].gt(pd.Timestamp("2026-01-01 02:00:00+00:00"))
    changed.loc[later, "stat_flag_zscore_pressure_max_hpa"] = 999.0
    changed.loc[later, "stat_sensor_group_flag_barometer"] = 999.0
    changed.loc[later, "stat_sensor_group_flag_light_uv"] = 999.0
    second = augment_hourly_rgfn_examples(examples, changed)

    assert first["static"].shape == (1, 9)
    assert first["rule_evidence"].shape == (1, 54)
    assert np.array_equal(first["X_cont"], original["X_cont"], equal_nan=True)
    assert np.array_equal(first["mask"], original["mask"], equal_nan=True)
    assert np.array_equal(first["time_since_last"], original["time_since_last"], equal_nan=True)
    assert np.array_equal(first["static"][:, :3], original["static"])
    assert np.array_equal(first["rule_evidence"][:, :38], original["rule_evidence"])
    assert tuple(first["static_feature_names"][-6:]) == CAUSAL_STATIC_FEATURE_NAMES
    assert tuple(first["rule_evidence_feature_names"][-16:]) == CAUSAL_RULE_EVIDENCE_FEATURE_NAMES
    np.testing.assert_allclose(first["static"], second["static"], equal_nan=True)
    np.testing.assert_allclose(first["rule_evidence"], second["rule_evidence"], equal_nan=True)
    np.testing.assert_allclose(first["static"][0, 3:], [2.0, 1.0 / 3.0, 2.0, 2.0, 1.0, 2.0])
    np.testing.assert_allclose(first["rule_evidence"][0, 38:42], [-2.0, 3.0, 4.0, 5.0])


def _training_examples(
    static_width: int,
    rule_width: int,
    mask_mode: str = MASK_MODE_PER_HOUR,
) -> dict[str, np.ndarray]:
    count = 30
    labels = np.asarray([index % 2 for index in range(count)], dtype=np.int64)
    generator = np.random.default_rng(17)
    continuous = generator.normal(size=(count, 7, len(CONTINUOUS_FEATURES))).astype(np.float32)
    continuous[:, :, 0] += labels[:, None] * 1.5
    rules = generator.normal(size=(count, rule_width)).astype(np.float32)
    rules[:, 0] += labels * 2.0
    return {
        "X_cont": continuous,
        "mask": np.ones(
            (count, 7, 1 if mask_mode == MASK_MODE_PER_HOUR else len(CONTINUOUS_FEATURES)),
            dtype=np.float32,
        ),
        "time_since_last": np.zeros((count, 7, 1), dtype=np.float32),
        "static": generator.normal(size=(count, static_width)).astype(np.float32),
        "rule_evidence": rules,
        "y_binary": labels,
    }


def _training_splits() -> dict[str, np.ndarray]:
    return {
        "train": np.arange(0, 18, dtype=np.int64),
        "validation": np.arange(18, 24, dtype=np.int64),
        "test": np.arange(24, 30, dtype=np.int64),
    }


def test_tuned_rgfn_supports_per_feature_mask() -> None:
    examples = _training_examples(3, 38, mask_mode=MASK_MODE_PER_FEATURE)
    prepared = prepare_tuning_split(examples, _training_splits())
    config = HourlyRgfnTuningConfig(
        sensor_hidden_size=8,
        evidence_hidden_size=4,
        gate_hidden_size=4,
        evidence_embed_size=4,
        dropout=0.0,
        mask_mode=MASK_MODE_PER_FEATURE,
    )

    model = build_tuned_hourly_rgfn(prepared, config)
    output = model(*prepared.train.features)

    assert model.mask_mode == MASK_MODE_PER_FEATURE
    assert model.input_width == len(CONTINUOUS_FEATURES) * 2 + 1
    assert output["binary_prob"].shape == (len(_training_splits()["train"]),)


def test_arm1_and_arm2_select_on_validation_before_single_test_evaluation(tmp_path) -> None:
    splits = _training_splits()
    feature_config = HourlyRgfnTuningConfig(max_epochs=2, patience=1, batch_size=64)
    arm1 = run_feature_only_arm(
        _training_examples(9, 54),
        splits,
        "random",
        "arm1_synthetic",
        model_dir=tmp_path,
        seeds=(0,),
        weights=(1.0,),
        thresholds=(0.5,),
        base_config=feature_config,
    )

    assert arm1["selection_source"] == "validation"
    assert arm1["test_evaluation_count"] == 1
    assert arm1["test_evaluation_count_per_seed"] == 1
    assert len(arm1["candidate_model_paths"]) == 1
    assert len(arm1["model_paths"]) == 1

    tuning_config = replace(feature_config, sensor_hidden_size=32, evidence_hidden_size=16, gate_hidden_size=8)
    screen = screen_architecture_search(
        _training_examples(3, 38),
        splits,
        "random",
        (tuning_config,),
        thresholds=(0.5,),
        model_dir=tmp_path,
    )
    finalized = finalize_architecture_search(
        screen,
        "arm2_synthetic",
        top_count=1,
        seeds=(0,),
        thresholds=(0.5,),
        model_dir=tmp_path,
    )

    assert finalized.selection_source == "validation"
    assert finalized.test_ledger.count == 0
    arm2 = final_evaluate_hourly_rgfn_arm(
        finalized,
        _training_examples(3, 38),
        splits,
        model_dir=tmp_path,
    )
    assert arm2["test_evaluation_count"] == 1
    with pytest.raises(RuntimeError, match="already"):
        final_evaluate_hourly_rgfn_arm(
            finalized,
            _training_examples(3, 38),
            splits,
            model_dir=tmp_path,
        )


def _comparison_source(value: float) -> dict[str, dict[str, object]]:
    return {
        split: {
            "test_summary": {
                "precision": value,
                "recall": value - 0.01,
                "f1": value - 0.02,
                "precision_std": 0.01,
                "recall_std": 0.01,
                "f1_std": 0.01,
            },
            "parameter_count": 10,
            "selection_source": "validation",
            "test_evaluation_count": 1,
        }
        for split in ("random", "spaced")
    }


def test_master_comparison_has_all_six_models_for_both_splits() -> None:
    comparison = master_tuning_comparison_frame(
        _comparison_source(0.80),
        _comparison_source(0.81),
        _comparison_source(0.82),
        _comparison_source(0.83),
        _comparison_source(0.84),
        _comparison_source(0.85),
    )

    assert len(comparison) == 12
    assert comparison.groupby(["model", "split"]).size().eq(1).all()
    assert set(comparison["model"]) == {
        "Logistic regression",
        "Gradient boosted",
        "Prior RGFN-GRU",
        "RGFN Arm 1 features",
        "RGFN Arm 2 search",
        "RGFN Arm 3 combined",
    }


def test_tuning_report_keeps_all_configurations_without_a_test_selected_arm() -> None:
    comparison = master_tuning_comparison_frame(
        _comparison_source(0.80),
        _comparison_source(0.81),
        _comparison_source(0.82),
        _comparison_source(0.83),
        _comparison_source(0.84),
        _comparison_source(0.85),
    )
    report = tuning_workflow.tuning_report(
        {
            "device": "cpu",
            "eligible_hourly_examples": 12,
            "excluded_examples_removed": 0,
        },
        {"future_values_used": False},
        comparison,
        pd.DataFrame([{"split": "random", "arm1_minus_prior": 0.0}]),
        pd.DataFrame([{"arm": "rgfn_arm1_features_only", "split": "random"}]),
        {"screened_combined_configurations_per_split": 1},
    )

    assert "FULL PREDEFINED CONFIGURATION COMPARISON" in report
    assert "Prior RGFN-GRU" in report
    assert "RGFN Arm 3 combined" in report
    assert "best_rgfn_arm" not in report
    assert "KEY BASELINE ANSWER" not in report
    assert "key_baseline_answer" not in report
    assert not hasattr(tuning_workflow, "_key_answer_frame")
