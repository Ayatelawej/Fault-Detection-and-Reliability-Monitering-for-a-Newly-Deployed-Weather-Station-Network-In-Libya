from __future__ import annotations

import gc
from argparse import ArgumentParser
from dataclasses import asdict, replace
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.hourly_baseline import (
    HourlyBaselineConfig,
    assert_matching_metadata,
    baseline_report,
    filter_eligible_examples,
    flatten_hourly_features,
    grouped_permutation_importance,
    load_hourly_metadata,
    load_hourly_tensor,
    make_split_manifest,
    resolve_fault_class_weight,
    run_baseline_matrix,
    save_model_bundle,
    spaced_split_indices,
    validation_importance_reference,
    write_metrics_json,
)
from src.model.hourly_detection import LONG_TENSOR_PATH, MASK_MODE_PER_HOUR, MASK_MODES, SHORT_TENSOR_PATH
from src.workflows.prerequisites import require_files


OUTPUT_DIR = PROJECT_ROOT / "data" / "hourly_detection"
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_PATH = OUTPUT_DIR / "hourly_baseline_metrics.json"
REPORT_PATH = OUTPUT_DIR / "hourly_baseline_report.txt"
IMPORTANCE_PATH = OUTPUT_DIR / "hourly_baseline_feature_importance.csv"
MANIFEST_PATH = OUTPUT_DIR / "hourly_baseline_split_manifest.csv"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--short-tensor", type=Path, default=SHORT_TENSOR_PATH)
    parser.add_argument("--long-tensor", type=Path, default=LONG_TENSOR_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--fault-class-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mask-mode", choices=MASK_MODES, default=MASK_MODE_PER_HOUR)
    return parser


def _matching_eligible_metadata(short_path: Path, long_path: Path) -> tuple[dict[str, object], int]:
    short = load_hourly_metadata(short_path)
    long = load_hourly_metadata(long_path)
    assert_matching_metadata(short, long)
    short_examples = {
        "X_cont": short["y_binary"].reshape(-1, 1, 1).astype("float32"),
        "mask": short["y_binary"].reshape(-1, 1, 1).astype("float32"),
        "time_since_last": short["y_binary"].reshape(-1, 1, 1).astype("float32"),
        "static": short["y_binary"].reshape(-1, 1).astype("float32"),
        "rule_evidence": short["y_binary"].reshape(-1, 1).astype("float32"),
        "y_binary": short["y_binary"],
        "station_id": short["station_id"],
        "hour": short["hour"],
        "display_state": short["display_state"],
        "source_episode_ids": short["source_episode_ids"],
        "continuous_feature_names": ["metadata"],
        "static_feature_names": ["metadata"],
        "rule_evidence_feature_names": ["metadata"],
    }
    eligible, excluded = filter_eligible_examples(short_examples)
    return eligible, excluded


def _load_filtered_tensor(path: Path) -> tuple[dict[str, object], int]:
    examples = load_hourly_tensor(path)
    return filter_eligible_examples(examples)


def main(argv: list[str] | None = None) -> None:
    args = parse_args().parse_args(argv)
    require_files(
        "Hourly baseline training",
        {
            "short hourly tensor": args.short_tensor,
            "long hourly tensor": args.long_tensor,
        },
        "Run scripts/build_hourly_dataset.py before training.",
    )
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    metadata, metadata_excluded = _matching_eligible_metadata(args.short_tensor, args.long_tensor)
    labels = metadata["y_binary"]
    resolved_weight = resolve_fault_class_weight(labels, args.fault_class_weight)
    config = HourlyBaselineConfig(seed=int(args.seed), fault_class_weight=resolved_weight)
    tensors = (("short", args.short_tensor), ("long", args.long_tensor))
    all_rows = []
    all_split_maps = None
    spaced_detail = None
    feature_names_by_window: dict[str, list[str]] = {}
    model_paths: dict[str, str] = {}
    for window_name, tensor_path in tensors:
        examples, excluded = _load_filtered_tensor(tensor_path)
        if excluded != metadata_excluded:
            raise ValueError("hourly tensors disagree on eligible example count")
        assert_matching_metadata(metadata, examples)
        values, feature_names, _ = flatten_hourly_features(examples, mask_mode=args.mask_mode)
        rows, models, split_maps, current_spaced_detail = run_baseline_matrix(
            values,
            examples["y_binary"],
            examples["station_id"],
            examples["hour"],
            examples["display_state"],
            examples["source_episode_ids"],
            window_name,
            config,
        )
        if all_split_maps is None:
            all_split_maps = split_maps
            spaced_detail = current_spaced_detail
        else:
            for split_name in ("random", "spaced"):
                for partition in ("train", "validation", "test"):
                    if not (all_split_maps[split_name][partition] == split_maps[split_name][partition]).all():
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
            )
            model_paths[run_name] = str(path)
        feature_names_by_window[window_name] = feature_names
        all_rows.extend(rows)
        del examples
        del values
        del models
        gc.collect()
    if all_split_maps is None or spaced_detail is None:
        raise RuntimeError("no baseline runs were completed")
    importance_reference = validation_importance_reference(all_rows)
    reference_window = str(importance_reference["window"])
    reference_path = args.short_tensor if reference_window == "short" else args.long_tensor
    reference_examples, _ = _load_filtered_tensor(reference_path)
    reference_values, _, reference_groups = flatten_hourly_features(
        reference_examples,
        mask_mode=args.mask_mode,
    )
    reference_splits = all_split_maps[str(importance_reference["split_scheme"])]
    reference_model = joblib.load(model_paths[str(importance_reference["run"])])["estimator"]
    importance = grouped_permutation_importance(
        reference_model,
        reference_values[reference_splits["validation"]],
        reference_examples["y_binary"][reference_splits["validation"]],
        reference_groups,
        config.threshold,
        config.seed,
    )
    manifest = make_split_manifest(
        all_split_maps,
        metadata["y_binary"],
        metadata["station_id"],
        metadata["hour"],
        metadata["display_state"],
        metadata["source_episode_ids"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_PATH.name
    metrics_path = output_dir / METRICS_PATH.name
    report_path = output_dir / REPORT_PATH.name
    importance_path = output_dir / IMPORTANCE_PATH.name
    manifest.to_csv(manifest_path, index=False)
    pd.DataFrame(importance).to_csv(importance_path, index=False)
    report = baseline_report(
        all_rows,
        importance,
        config,
        spaced_detail,
        importance_reference,
    )
    report_path.write_text(report + "\n", encoding="utf-8")
    payload = {
        "configuration": asdict(config),
        "mask_mode": args.mask_mode,
        "excluded_examples_removed": int(metadata_excluded),
        "runs": all_rows,
        "importance_reference": importance_reference,
        "importance_reference_selection": "validation_f1_then_recall_then_precision",
        "grouped_permutation_importance": importance,
        "spaced_split": spaced_detail,
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
