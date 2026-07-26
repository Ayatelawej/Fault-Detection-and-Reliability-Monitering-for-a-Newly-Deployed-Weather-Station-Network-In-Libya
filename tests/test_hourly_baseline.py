from __future__ import annotations

import numpy as np

from src.model.hourly_baseline import (
    HourlyBaselineConfig,
    SPLIT_FRACTIONS_80_20,
    baseline_report,
    filter_eligible_examples,
    flatten_hourly_features,
    random_split_indices,
    resolve_fault_class_weight,
    run_baseline_matrix,
    save_model_bundle,
    spaced_split_indices,
    validation_importance_reference,
    write_metrics_json,
)
from src.model.hourly_detection import MASK_MODE_PER_FEATURE, MASK_MODE_PER_HOUR


def _examples(window_hours: int) -> dict[str, np.ndarray]:
    count = 90
    hours = np.asarray(
        [f"2026-{1 + index // 30:02d}-{1 + index % 28:02d} 00:00:00+00:00" for index in range(count)],
        dtype=object,
    )
    labels = np.asarray([int(index % 6 == 0) for index in range(count)], dtype=np.int64)
    display = np.where(labels == 1, "fault", "clean").astype(object)
    display[5] = "excluded"
    labels[5] = -1
    continuous = np.zeros((count, window_hours, 2), dtype=np.float32)
    for index in range(count):
        continuous[index, :, 0] = float(index % 7)
        continuous[index, :, 1] = float(labels[index] == 1)
    source_ids = np.full(count, "", dtype=object)
    for index in np.flatnonzero(labels == 1):
        source_ids[index] = f"episode_{index // 2:03d}"
    return {
        "X_cont": continuous,
        "mask": np.ones((count, window_hours, 1), dtype=np.float32),
        "time_since_last": np.zeros((count, window_hours, 1), dtype=np.float32),
        "static": np.ones((count, 3), dtype=np.float32),
        "rule_evidence": np.zeros((count, 4), dtype=np.float32),
        "y_binary": labels,
        "station_id": np.asarray([f"S{index % 3}" for index in range(count)], dtype=object),
        "hour": hours,
        "display_state": display,
        "source_episode_ids": source_ids,
        "continuous_feature_names": np.asarray(["signal", "fault_signal"], dtype=object),
        "static_feature_names": np.asarray(["ctx_a", "ctx_b", "ctx_c"], dtype=object),
        "rule_evidence_feature_names": np.asarray(["rule_a", "rule_b", "rule_c", "rule_d"], dtype=object),
    }


def test_flattened_feature_contract_includes_every_numeric_input() -> None:
    short, _ = filter_eligible_examples(_examples(7))
    long, _ = filter_eligible_examples(_examples(49))
    short_values, short_names, _ = flatten_hourly_features(short)
    long_values, long_names, _ = flatten_hourly_features(long)

    assert short_values.shape[1] == 7 * (2 + 2) + 4 + 3
    assert long_values.shape[1] == 49 * (2 + 2) + 4 + 3
    assert len(short_names) == short_values.shape[1]
    assert len(long_names) == long_values.shape[1]
    assert "row_present_t0" in short_names
    assert "time_since_last_t0" in short_names
    assert "rule_a" in short_names
    assert "ctx_a" in short_names


def test_flattening_supports_per_feature_masks_without_changing_per_hour_values() -> None:
    examples, _ = filter_eligible_examples(_examples(7))
    default_values, default_names, default_groups = flatten_hourly_features(examples)
    explicit_values, explicit_names, explicit_groups = flatten_hourly_features(
        examples,
        mask_mode=MASK_MODE_PER_HOUR,
    )

    assert np.array_equal(default_values, explicit_values, equal_nan=True)
    assert default_names == explicit_names
    assert set(default_groups) == set(explicit_groups)
    for name in default_groups:
        assert np.array_equal(default_groups[name], explicit_groups[name])

    feature_examples = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in examples.items()}
    feature_examples["X_cont"][0, -1, 1] = np.nan
    feature_examples["mask"] = (~np.isnan(feature_examples["X_cont"])).astype(np.float32)
    feature_values, feature_names, feature_groups = flatten_hourly_features(
        feature_examples,
        mask_mode=MASK_MODE_PER_FEATURE,
    )

    assert feature_values.shape[1] == 7 * (2 * 2 + 1) + 4 + 3
    assert "signal_present_t0" in feature_names
    assert "fault_signal_present_t0" in feature_names
    assert "feature_mask:signal" in feature_groups
    assert "feature_mask:fault_signal" in feature_groups
    missing_index = feature_names.index("fault_signal_present_t0")
    present_index = feature_names.index("signal_present_t0")
    assert feature_values[0, missing_index] == 0.0
    assert feature_values[0, present_index] == 1.0


def test_four_runs_apply_weight_and_exclude_ineligible_rows() -> None:
    examples, excluded = filter_eligible_examples(_examples(7))
    values, _, _ = flatten_hourly_features(examples)
    config = HourlyBaselineConfig(
        seed=17,
        fault_class_weight=4.0,
        max_iter=3,
        max_leaf_nodes=4,
        min_samples_leaf=2,
    )
    rows, models, split_maps, _ = run_baseline_matrix(
        values,
        examples["y_binary"],
        examples["station_id"],
        examples["hour"],
        examples["display_state"],
        examples["source_episode_ids"],
        "short",
        config,
    )

    assert excluded == 1
    assert len(rows) == 2
    assert set(models) == {"short-random", "short-spaced"}
    assert all(model.get_params()["class_weight"] == {0: 1.0, 1: 4.0} for model in models.values())
    for row in rows:
        assert set(row["validation"]) >= {"precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"}
        assert set(row["test"]) >= {"precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"}
        assert sum(row["test"][key] for key in ("tp", "fp", "fn", "tn")) == len(split_maps[row["split_scheme"]]["test"])
    assert not np.equal(examples["display_state"], "excluded").any()


def test_two_windows_produce_four_metric_sets_and_model_artifacts(tmp_path) -> None:
    config = HourlyBaselineConfig(
        seed=19,
        fault_class_weight=3.0,
        max_iter=3,
        max_leaf_nodes=4,
        min_samples_leaf=2,
    )
    all_rows = []
    paths = []
    for window_name, window_hours in (("short", 7), ("long", 49)):
        examples, _ = filter_eligible_examples(_examples(window_hours))
        values, names, _ = flatten_hourly_features(examples)
        rows, models, _, _ = run_baseline_matrix(
            values,
            examples["y_binary"],
            examples["station_id"],
            examples["hour"],
            examples["display_state"],
            examples["source_episode_ids"],
            window_name,
            config,
        )
        all_rows.extend(rows)
        for row in rows:
            path = save_model_bundle(
                models[row["run"]],
                tmp_path / f"{row['run']}.joblib",
                names,
                window_hours,
                config,
                row["split_scheme"],
            )
            paths.append(path)
    metrics_path = write_metrics_json({"runs": all_rows}, tmp_path / "metrics.json")

    assert {row["run"] for row in all_rows} == {
        "short-random",
        "short-spaced",
        "long-random",
        "long-spaced",
    }
    assert all(path.exists() for path in paths)
    assert metrics_path.exists()


def test_split_membership_is_deterministic_complete_and_spaced_by_fault_groups() -> None:
    examples, _ = filter_eligible_examples(_examples(7))
    labels = examples["y_binary"]
    random_one = random_split_indices(labels, seed=11)
    random_two = random_split_indices(labels, seed=11)
    spaced_one, detail_one = spaced_split_indices(
        labels,
        examples["station_id"],
        examples["hour"],
        examples["display_state"],
        examples["source_episode_ids"],
    )
    spaced_two, detail_two = spaced_split_indices(
        labels,
        examples["station_id"],
        examples["hour"],
        examples["display_state"],
        examples["source_episode_ids"],
    )

    assert all(np.array_equal(random_one[name], random_two[name]) for name in random_one)
    assert all(np.array_equal(spaced_one[name], spaced_two[name]) for name in spaced_one)
    assert detail_one["fault_group_count"] == detail_two["fault_group_count"]
    partitions = np.full(len(labels), "", dtype=object)
    for name, indices in spaced_one.items():
        partitions[indices] = name
    for episode_id in np.unique(examples["source_episode_ids"]):
        if not episode_id:
            continue
        member = np.equal(examples["source_episode_ids"], episode_id)
        assert len(set(partitions[member])) == 1


def test_fault_weight_resolves_to_the_not_fault_ratio() -> None:
    labels = np.asarray([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64)

    assert resolve_fault_class_weight(labels) == 3.0
    assert resolve_fault_class_weight(labels, 5.0) == 5.0


def test_true_80_20_runs_without_a_validation_partition() -> None:
    examples, _ = filter_eligible_examples(_examples(7))
    values, _, _ = flatten_hourly_features(examples)
    config = HourlyBaselineConfig(
        seed=23,
        fault_class_weight=4.0,
        max_iter=3,
        max_leaf_nodes=4,
        min_samples_leaf=2,
    )
    rows, _, split_maps, _ = run_baseline_matrix(
        values,
        examples["y_binary"],
        examples["station_id"],
        examples["hour"],
        examples["display_state"],
        examples["source_episode_ids"],
        "short",
        config,
        SPLIT_FRACTIONS_80_20,
        "80_20",
    )

    assert set(split_maps["random"]) == {"train", "test"}
    assert set(split_maps["spaced"]) == {"train", "test"}
    assert {row["run"] for row in rows} == {"short-random-80_20", "short-spaced-80_20"}
    assert all(row["validation"] is None for row in rows)
    assert all(row["split_configuration"] == "80_20" for row in rows)
    assert all(len(split_maps[name]["train"]) > len(split_maps[name]["test"]) for name in split_maps)


def _metrics(f1: float) -> dict[str, float | int]:
    return {
        "precision": f1 - 0.01,
        "recall": f1 - 0.02,
        "f1": f1,
        "accuracy": f1 + 0.01,
        "tp": 3,
        "fp": 1,
        "fn": 1,
        "tn": 10,
    }


def test_importance_reference_uses_validation_not_test_metrics() -> None:
    rows = [
        {
            "run": "validation_reference",
            "window": "short",
            "split_scheme": "spaced",
            "feature_dimension": 8,
            "model_iterations": 3,
            "validation": _metrics(0.90),
            "test": _metrics(0.50),
        },
        {
            "run": "test_favorite",
            "window": "long",
            "split_scheme": "random",
            "feature_dimension": 8,
            "model_iterations": 3,
            "validation": _metrics(0.80),
            "test": _metrics(0.99),
        },
    ]

    reference = validation_importance_reference(rows)
    report = baseline_report(
        rows,
        [{"feature_group": "rule_evidence", "importance_f1_drop": 0.10}],
        HourlyBaselineConfig(),
        {"fault_group_count": 2},
        reference,
    )

    assert reference["run"] == "validation_reference"
    assert "importance_reference_run=validation_reference" in report
    assert "test_favorite" in report
    assert "best_run" not in report
    assert "BEST-RUN" not in report
    assert "Held-out test metrics are descriptive comparisons only" in report
