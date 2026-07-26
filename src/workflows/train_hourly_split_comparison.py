from __future__ import annotations

import gc
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.hourly_baseline import (
    HourlyBaselineConfig,
    SPLIT_FRACTIONS,
    SPLIT_FRACTIONS_80_20,
    assert_matching_metadata,
    filter_eligible_examples,
    flatten_hourly_features,
    grouped_permutation_importance,
    load_hourly_metadata,
    load_hourly_tensor,
    make_split_manifest,
    resolve_fault_class_weight,
    run_baseline_matrix,
    save_model_bundle,
    write_metrics_json,
)
from src.model.hourly_detection import LONG_TENSOR_PATH, MASK_MODE_PER_HOUR, MASK_MODES, SHORT_TENSOR_PATH
from src.workflows.prerequisites import require_files


OUTPUT_DIR = PROJECT_ROOT / "data" / "hourly_detection"
MODEL_DIR_NAME = "split_comparison"
OUTPUT_STEM = "hourly_baseline_70_15_15_and_80_20"
SPLIT_CONFIGURATIONS = {
    "70_15_15": SPLIT_FRACTIONS,
    "80_20": SPLIT_FRACTIONS_80_20,
}


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--short-tensor", type=Path, default=SHORT_TENSOR_PATH)
    parser.add_argument("--long-tensor", type=Path, default=LONG_TENSOR_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--fault-class-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mask-mode", choices=MASK_MODES, default=MASK_MODE_PER_HOUR)
    return parser


def _eligible_metadata(short_path: Path, long_path: Path) -> tuple[dict[str, np.ndarray], int]:
    short = load_hourly_metadata(short_path)
    long = load_hourly_metadata(long_path)
    assert_matching_metadata(short, long)
    states = np.asarray(short["display_state"], dtype=object).astype(str)
    labels = np.asarray(short["y_binary"])
    eligible = np.not_equal(states, "excluded") & np.isin(labels, [0, 1])
    result = {
        key: np.asarray(value)[eligible]
        for key, value in short.items()
        if key != "window_hours"
    }
    if not len(result["y_binary"]):
        raise ValueError("no eligible metadata rows remain")
    return result, int((~eligible).sum())


def _load_filtered_tensor(path: Path) -> tuple[dict[str, np.ndarray], int]:
    return filter_eligible_examples(load_hourly_tensor(path))


def _metric_table(metric: dict[str, object]) -> str:
    return pd.DataFrame(
        [
            {
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "accuracy": metric["accuracy"],
            }
        ]
    ).to_string(index=False)


def _validation_reference(rows: list[dict[str, object]]) -> dict[str, object]:
    candidates = [row for row in rows if row["validation"] is not None]
    if not candidates:
        raise ValueError("a validation-bearing split is required for importance selection")
    return max(
        candidates,
        key=lambda row: (
            float(row["validation"]["f1"]),
            float(row["validation"]["recall"]),
            float(row["validation"]["precision"]),
        ),
    )


def comparison_report(
    rows: list[dict[str, object]],
    importance: list[dict[str, object]],
    config: HourlyBaselineConfig,
    spaced_details: dict[str, dict[str, object]],
    importance_reference: dict[str, object],
) -> str:
    parts = [
        "HOURLY BINARY GRADIENT-BOOSTED SPLIT COMPARISON",
        "",
        "CONFIGURATION",
        "estimator=HistGradientBoostingClassifier",
        f"fault_class_weight={config.fault_class_weight:.6f}",
        f"decision_threshold={config.threshold:.3f}",
        f"max_iter={config.max_iter}",
        f"learning_rate={config.learning_rate:.3f}",
        f"max_leaf_nodes={config.max_leaf_nodes}",
        f"min_samples_leaf={config.min_samples_leaf}",
        "",
    ]
    for row in rows:
        test = row["test"]
        parts.extend(
            [
                f"RUN {row['run']}",
                f"split_configuration={row['split_configuration']}",
                f"feature_dimension={row['feature_dimension']}",
                f"model_iterations={row['model_iterations']}",
                "TEST METRICS",
                _metric_table(test),
                "TEST CONFUSION MATRIX",
                pd.DataFrame(
                    [{"tp": test["tp"], "fp": test["fp"], "fn": test["fn"], "tn": test["tn"]}]
                ).to_string(index=False),
                "SPLIT COMPOSITION",
                pd.DataFrame(row["split_summary"]).to_string(index=False),
                "VALIDATION METRICS",
            ]
        )
        if row["validation"] is None:
            parts.append("not_applicable_for_true_80_20")
        else:
            parts.append(_metric_table(row["validation"]))
        parts.append("")
    summary = pd.DataFrame(
        [
            {
                "split_configuration": row["split_configuration"],
                "run": row["run"],
                "precision": row["test"]["precision"],
                "recall": row["test"]["recall"],
                "f1": row["test"]["f1"],
            }
            for row in rows
        ]
    )
    positive_total = sum(max(0.0, float(row["importance_f1_drop"])) for row in importance)
    dominant = bool(importance) and positive_total > 0.0 and float(importance[0]["importance_f1_drop"]) / positive_total >= 0.50
    parts.extend(
        [
            "SUMMARY COMPARISON",
            summary.to_string(index=False),
            "",
            "SPLIT CONSTRUCTION",
            "70_15_15 uses 70 percent train, 15 percent validation, and 15 percent test.",
            "80_20 uses 80 percent train and 20 percent test, with no validation partition.",
            "Random membership is deterministic and binary-stratified.",
            "Spaced membership keeps connected source fault episodes together, distributes them through chronological strata, and assigns benign and clean hours within month and display-state strata.",
            f"spaced_fault_group_count_70_15_15={spaced_details['70_15_15']['fault_group_count']}",
            f"spaced_fault_group_count_80_20={spaced_details['80_20']['fault_group_count']}",
            "",
            "GROUPED PERMUTATION IMPORTANCE",
            f"importance_reference_run={importance_reference['run']}",
            "The reference is selected from 70/15/15 validation metrics so importance can be measured without using a test set.",
            pd.DataFrame(importance[:20]).to_string(index=False),
            f"single_feature_dominance_flag={dominant}",
            "",
            "EVALUATION NOTE",
            "The windows end at the scored hour, but the existing feature snapshot includes retrospective detector calculations. These are offline comparison metrics, not future-facing deployment metrics.",
        ]
    )
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    args = parse_args().parse_args(argv)
    require_files(
        "Hourly split comparison",
        {
            "short hourly tensor": args.short_tensor,
            "long hourly tensor": args.long_tensor,
        },
        "Run scripts/build_hourly_dataset.py before training.",
    )
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models" / MODEL_DIR_NAME
    metadata, metadata_excluded = _eligible_metadata(args.short_tensor, args.long_tensor)
    resolved_weight = resolve_fault_class_weight(metadata["y_binary"], args.fault_class_weight)
    config = HourlyBaselineConfig(seed=int(args.seed), fault_class_weight=resolved_weight)
    tensors = (("short", args.short_tensor), ("long", args.long_tensor))
    all_rows: list[dict[str, object]] = []
    split_maps_by_configuration: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    spaced_details: dict[str, dict[str, object]] = {}
    feature_names_by_window: dict[str, list[str]] = {}
    model_paths: dict[str, str] = {}
    for window_name, tensor_path in tensors:
        examples, excluded = _load_filtered_tensor(tensor_path)
        if excluded != metadata_excluded:
            raise ValueError("hourly tensors disagree on eligible example count")
        assert_matching_metadata(metadata, examples)
        values, feature_names, _ = flatten_hourly_features(examples, mask_mode=args.mask_mode)
        for configuration_name, fractions in SPLIT_CONFIGURATIONS.items():
            rows, models, split_maps, detail = run_baseline_matrix(
                values,
                examples["y_binary"],
                examples["station_id"],
                examples["hour"],
                examples["display_state"],
                examples["source_episode_ids"],
                window_name,
                config,
                fractions,
                configuration_name,
            )
            if configuration_name not in split_maps_by_configuration:
                split_maps_by_configuration[configuration_name] = split_maps
                spaced_details[configuration_name] = detail
            else:
                expected_maps = split_maps_by_configuration[configuration_name]
                for scheme, expected_splits in expected_maps.items():
                    for partition, expected_indices in expected_splits.items():
                        if not np.array_equal(expected_indices, split_maps[scheme][partition]):
                            raise ValueError("short and long tensors received different split membership")
            for row in rows:
                run_name = str(row["run"])
                path = model_dir / f"{run_name.replace('-', '_')}.joblib"
                save_model_bundle(
                    models[run_name],
                    path,
                    feature_names,
                    int(examples["X_cont"].shape[1]),
                    config,
                    str(row["split_scheme"]),
                    configuration_name,
                    fractions,
                )
                model_paths[run_name] = str(path)
            all_rows.extend(rows)
            del models
            gc.collect()
        feature_names_by_window[window_name] = feature_names
        del examples
        del values
        gc.collect()
    importance_reference = _validation_reference(all_rows)
    selected_window = str(importance_reference["window"])
    selected_path = args.short_tensor if selected_window == "short" else args.long_tensor
    selected_examples, _ = _load_filtered_tensor(selected_path)
    selected_values, _, selected_groups = flatten_hourly_features(
        selected_examples,
        mask_mode=args.mask_mode,
    )
    selected_maps = split_maps_by_configuration[str(importance_reference["split_configuration"])]
    selected_splits = selected_maps[str(importance_reference["split_scheme"])]
    selected_model = joblib.load(model_paths[str(importance_reference["run"])])["estimator"]
    importance = grouped_permutation_importance(
        selected_model,
        selected_values[selected_splits["validation"]],
        selected_examples["y_binary"][selected_splits["validation"]],
        selected_groups,
        config.threshold,
        config.seed,
    )
    manifests = [
        make_split_manifest(
            split_maps_by_configuration[configuration_name],
            metadata["y_binary"],
            metadata["station_id"],
            metadata["hour"],
            metadata["display_state"],
            metadata["source_episode_ids"],
            configuration_name,
        )
        for configuration_name in SPLIT_CONFIGURATIONS
    ]
    manifest = pd.concat(manifests, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{OUTPUT_STEM}_metrics.json"
    report_path = output_dir / f"{OUTPUT_STEM}_report.txt"
    importance_path = output_dir / f"{OUTPUT_STEM}_feature_importance.csv"
    manifest_path = output_dir / f"{OUTPUT_STEM}_split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    pd.DataFrame(importance).to_csv(importance_path, index=False)
    report = comparison_report(all_rows, importance, config, spaced_details, importance_reference)
    report_path.write_text(report + "\n", encoding="utf-8")
    payload = {
        "configuration": asdict(config),
        "mask_mode": args.mask_mode,
        "split_configurations": SPLIT_CONFIGURATIONS,
        "excluded_examples_removed": int(metadata_excluded),
        "runs": all_rows,
        "importance_reference_run": importance_reference,
        "grouped_permutation_importance": importance,
        "spaced_splits": spaced_details,
        "model_paths": model_paths,
        "manifest_path": str(manifest_path),
        "feature_dimensions": {name: len(names) for name, names in feature_names_by_window.items()},
    }
    write_metrics_json(payload, metrics_path)
    print(report)
    print()
    print("OUTPUTS")
    print(f"metrics_json={metrics_path}")
    print(f"report={report_path}")
    print(f"feature_importance={importance_path}")
    print(f"split_manifest={manifest_path}")
    for run_name in sorted(model_paths):
        print(f"model_{run_name}={model_paths[run_name]}")


if __name__ == "__main__":
    main()
