from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.hourly_baseline import filter_eligible_examples, load_hourly_tensor, write_metrics_json
from src.model.hourly_detection import FEATURE_PATH, SHORT_TENSOR_PATH, SOURCE_PATH, load_hourly_frame
from src.model.hourly_rgfn_training import DEVICE, load_calibrated_baseline, load_manifest_splits
from src.model.hourly_rgfn_tuning import (
    TUNING_SEEDS,
    arm_result_payload,
    configuration_payload,
    final_evaluate_hourly_rgfn_arm,
    finalize_architecture_search,
    make_coarse_search_configurations,
    run_combined_arm,
    run_feature_only_arm,
    screen_architecture_search,
    search_configuration_coverage,
    select_top_screened_configurations,
)
from src.model.hourly_rgfn_tuning_features import (
    CAUSAL_RULE_EVIDENCE_FEATURE_NAMES,
    CAUSAL_STATIC_FEATURE_NAMES,
    augment_hourly_rgfn_examples,
)
from src.model.hourly_rgfn_tuning_logistic import train_hourly_logistic_variant
from src.workflows.prerequisites import require_files


OUTPUT_DIR = PROJECT_ROOT / "data" / "hourly_detection"
MANIFEST_PATH = OUTPUT_DIR / "hourly_baseline_split_manifest.csv"
BOOSTED_METRICS_PATH = OUTPUT_DIR / "hourly_short_calibration_metrics.json"
PRIOR_RGFN_METRICS_PATH = OUTPUT_DIR / "hourly_rgfn_comparison_metrics.json"
METRICS_PATH = OUTPUT_DIR / "hourly_rgfn_tuning_metrics.json"
REPORT_PATH = OUTPUT_DIR / "hourly_rgfn_tuning_report.txt"
SWEEP_PATH = OUTPUT_DIR / "hourly_rgfn_tuning_validation_sweep.csv"
SCREENING_PATH = OUTPUT_DIR / "hourly_rgfn_tuning_architecture_screen.csv"

MODEL_ORDER = (
    "logistic_regression",
    "gradient_boosted",
    "rgfn_gru_prior",
    "rgfn_arm1_features_only",
    "rgfn_arm2_tuned_only",
    "rgfn_arm3_combined",
)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--tensor", type=Path, default=SHORT_TENSOR_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--boosted-metrics", type=Path, default=BOOSTED_METRICS_PATH)
    parser.add_argument("--prior-rgfn-metrics", type=Path, default=PRIOR_RGFN_METRICS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--search-count", type=int, default=24)
    parser.add_argument("--top-count", type=int, default=3)
    return parser


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(name): _jsonable(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _load_prior_rgfn(path: Path) -> dict[str, dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = payload.get("rgfn_gru")
    if not isinstance(result, dict):
        raise KeyError("prior hourly RGFN metrics lack rgfn_gru results")
    for split in ("random", "spaced"):
        if split not in result:
            raise KeyError(f"prior hourly RGFN metrics lack {split} results")
    return result


def _metric(source: dict[str, object], split: str) -> dict[str, float]:
    item = source[split]
    if not isinstance(item, dict):
        raise TypeError(f"metric source for {split} is invalid")
    values = item.get("test_summary", item)
    if not isinstance(values, dict):
        raise TypeError(f"test metrics for {split} are invalid")
    result: dict[str, float] = {}
    for name in ("precision", "recall", "f1"):
        if name not in values:
            raise KeyError(f"test metrics for {split} lack {name}")
        result[name] = float(values[name])
        std_value = values.get(f"{name}_std", np.nan)
        result[f"{name}_std"] = float(np.nan if std_value is None else std_value)
    return result


def master_tuning_comparison_frame(
    logistic: dict[str, dict[str, object]],
    boosted: dict[str, dict[str, object]],
    prior: dict[str, dict[str, object]],
    arm1: dict[str, dict[str, object]],
    arm2: dict[str, dict[str, object]],
    arm3: dict[str, dict[str, object]],
) -> pd.DataFrame:
    sources = (
        ("logistic_regression", "new current-manifest run", logistic),
        ("gradient_boosted", "carried current-manifest calibration", boosted),
        ("rgfn_gru_prior", "carried current-manifest prior run", prior),
        ("rgfn_arm1_features_only", "new current-manifest run", arm1),
        ("rgfn_arm2_tuned_only", "new current-manifest run", arm2),
        ("rgfn_arm3_combined", "new current-manifest run", arm3),
    )
    rows: list[dict[str, object]] = []
    for split in ("random", "spaced"):
        for model, origin, source in sources:
            values = _metric(source, split)
            rows.append({"split": split, "model": model, "metric_origin": origin, **values})
    return pd.DataFrame(rows).sort_values(
        ["split", "model"],
        key=lambda values: values.map({name: index for index, name in enumerate(MODEL_ORDER)}).fillna(len(MODEL_ORDER)) if values.name == "model" else values,
        kind="stable",
    ).reset_index(drop=True)


def _verify_augmentation(base: dict[str, np.ndarray], augmented: dict[str, np.ndarray]) -> dict[str, object]:
    for name in ("X_cont", "mask", "time_since_last"):
        if not np.array_equal(np.asarray(base[name]), np.asarray(augmented[name]), equal_nan=True):
            raise RuntimeError(f"augmentation changed the base {name} values")
    if not np.array_equal(np.asarray(base["static"]), np.asarray(augmented["static"])[:, :3], equal_nan=True):
        raise RuntimeError("augmentation changed the three base static values")
    if not np.array_equal(np.asarray(base["rule_evidence"]), np.asarray(augmented["rule_evidence"])[:, :38], equal_nan=True):
        raise RuntimeError("augmentation changed the 38 base evidence values")
    if np.asarray(augmented["static"]).shape[1] != 9:
        raise RuntimeError("augmented static width is not nine")
    if np.asarray(augmented["rule_evidence"]).shape[1] != 54:
        raise RuntimeError("augmented evidence width is not 54")
    names = [str(value) for value in np.asarray(augmented["static_feature_names"], dtype=object)]
    forbidden = [name for name in names if "duration" in name.lower() or "episode" in name.lower()]
    if forbidden:
        raise RuntimeError(f"augmented static names contain disallowed values: {forbidden}")
    return {
        "base_static_width": 3,
        "augmented_static_width": 9,
        "base_rule_evidence_width": 38,
        "augmented_rule_evidence_width": 54,
        "added_static_features": list(CAUSAL_STATIC_FEATURE_NAMES),
        "added_rule_evidence_features": list(CAUSAL_RULE_EVIDENCE_FEATURE_NAMES),
        "excluded_historical_duration_features": [
            "episode_log_duration_hours",
            "episode_duration_ge_7h",
            "episode_duration_ge_24h",
            "episode_duration_ge_72h",
        ],
        "past_only": True,
        "future_values_used": False,
        "integrity_check": "base arrays unchanged; all additions use the current hour and preceding six hours",
    }


def _progress(item: dict[str, object]) -> None:
    arm = str(item.get("arm", "candidate"))
    split = str(item.get("split", ""))
    seed = int(item.get("seed", 0))
    epochs = int(item.get("epochs_completed", 0))
    best_epoch = int(item.get("best_epoch", 0))
    loss = float(item.get("best_validation_loss", np.nan))
    weight = item.get("fault_class_weight")
    suffix = "" if weight is None else f":weight={float(weight):.1f}"
    print(
        f"candidate_complete={arm}:{split}:seed={seed}{suffix}:"
        f"epochs={epochs}:best_epoch={best_epoch}:validation_loss={loss:.6f}",
        flush=True,
    )


def _validation_frame(result: dict[str, dict[str, object]], arm_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, item in result.items():
        selected = item.get("best_balanced", {})
        selected_config = selected.get("configuration", item.get("selected_config", {})) if isinstance(selected, dict) else {}
        selected_threshold = float(item.get("selected_threshold", selected.get("threshold", np.nan))) if isinstance(selected, dict) else np.nan
        for row in item.get("validation_seed_rows", []):
            if not isinstance(row, dict):
                continue
            metric = dict(row.get("validation", {}))
            config = dict(row.get("configuration", {}))
            rows.append(
                {
                    "arm": arm_name,
                    "split": split,
                    "row_type": "seed",
                    "seed": row.get("seed"),
                    "configuration_id": row.get("configuration_id"),
                    "fault_class_weight": row.get("fault_class_weight"),
                    "threshold": row.get("threshold"),
                    "precision": metric.get("precision"),
                    "recall": metric.get("recall"),
                    "f1": metric.get("f1"),
                    "accuracy": metric.get("accuracy"),
                    "minimum_metric": row.get("validation_minimum_metric"),
                    "best_epoch": row.get("best_epoch"),
                    "best_validation_loss": row.get("best_validation_loss"),
                    "selected": bool(
                        float(row.get("threshold", np.nan)) == selected_threshold
                        and config == selected_config
                    ),
                    **{f"config_{name}": value for name, value in config.items()},
                }
            )
        for row in item.get("validation_grid", []):
            if not isinstance(row, dict):
                continue
            metric = dict(row.get("validation", {}))
            deviation = dict(row.get("validation_std", {}))
            config = dict(row.get("configuration", {}))
            rows.append(
                {
                    "arm": arm_name,
                    "split": split,
                    "row_type": "aggregate",
                    "seed": None,
                    "configuration_id": row.get("configuration_id"),
                    "fault_class_weight": row.get("fault_class_weight"),
                    "threshold": row.get("threshold"),
                    "precision": metric.get("precision"),
                    "recall": metric.get("recall"),
                    "f1": metric.get("f1"),
                    "accuracy": metric.get("accuracy"),
                    "minimum_metric": row.get("validation_minimum_metric"),
                    "precision_std": deviation.get("precision"),
                    "recall_std": deviation.get("recall"),
                    "f1_std": deviation.get("f1"),
                    "best_epoch": None,
                    "best_validation_loss": None,
                    "selected": bool(
                        float(row.get("threshold", np.nan)) == selected_threshold
                        and config == selected_config
                    ),
                    **{f"config_{name}": value for name, value in config.items()},
                }
            )
    return pd.DataFrame(rows)


def _screening_frame(screen_details: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, detail in screen_details.items():
        top = {json.dumps(value, sort_keys=True) for value in detail["top_configurations"]}
        for row in detail["validation_grid"]:
            configuration = dict(row["configuration"])
            rows.append(
                {
                    "split": split,
                    "configuration_id": row["configuration_id"],
                    "fault_class_weight": row["fault_class_weight"],
                    "threshold": row["threshold"],
                    "precision": row["validation"]["precision"],
                    "recall": row["validation"]["recall"],
                    "f1": row["validation"]["f1"],
                    "accuracy": row["validation"]["accuracy"],
                    "minimum_metric": row["validation_minimum_metric"],
                    "seed_count": row["seed_count"],
                    "top_configuration": json.dumps(configuration, sort_keys=True) in top,
                    **{f"config_{name}": value for name, value in configuration.items()},
                }
            )
    return pd.DataFrame(rows)


def _attribution_frame(
    prior: dict[str, dict[str, object]],
    arm1: dict[str, dict[str, object]],
    arm2: dict[str, dict[str, object]],
    arm3: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for split in ("random", "spaced"):
        prior_f1 = _metric(prior, split)["f1"]
        arm1_f1 = _metric(arm1, split)["f1"]
        arm2_f1 = _metric(arm2, split)["f1"]
        arm3_f1 = _metric(arm3, split)["f1"]
        rows.append(
            {
                "split": split,
                "prior_rgfn_f1": prior_f1,
                "arm1_features_f1": arm1_f1,
                "arm1_minus_prior": arm1_f1 - prior_f1,
                "arm2_tuned_f1": arm2_f1,
                "arm2_minus_prior": arm2_f1 - prior_f1,
                "arm3_combined_f1": arm3_f1,
                "arm3_minus_arm1": arm3_f1 - arm1_f1,
                "arm3_minus_arm2": arm3_f1 - arm2_f1,
                "arm3_beats_either_alone": bool(arm3_f1 > arm1_f1 and arm3_f1 > arm2_f1),
            }
        )
    return pd.DataFrame(rows)


def _selected_config_frame(
    arm1: dict[str, dict[str, object]],
    arm2: dict[str, dict[str, object]],
    arm3: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm_name, source in (
        ("rgfn_arm1_features_only", arm1),
        ("rgfn_arm2_tuned_only", arm2),
        ("rgfn_arm3_combined", arm3),
    ):
        for split in ("random", "spaced"):
            item = source[split]
            row = {
                "arm": arm_name,
                "split": split,
                "threshold": item["selected_threshold"],
                "parameter_count": item["parameter_count"],
                "best_epoch_mean": item["selected_best_epoch_mean"],
                "best_epoch_std": item["selected_best_epoch_std"],
                "test_evaluation_count": item["test_evaluation_count"],
                "candidate_checkpoints": len(item.get("candidate_model_paths", [])),
                "final_checkpoints": len(item.get("model_paths", [])),
                **{f"config_{name}": value for name, value in item["selected_config"].items()},
            }
            rows.append(row)
    return pd.DataFrame(rows)


def tuning_report(
    inputs: dict[str, object],
    integrity: dict[str, object],
    comparison: pd.DataFrame,
    attribution: pd.DataFrame,
    configurations: pd.DataFrame,
    search_details: dict[str, object],
) -> str:
    parts = [
        "HOUR-LEVEL RGFN-GRU TUNING",
        "",
        "EXPERIMENT INTEGRITY",
        f"device={inputs['device']}",
        f"eligible_hourly_examples={inputs['eligible_hourly_examples']}",
        f"excluded_examples_removed={inputs['excluded_examples_removed']}",
        "The literal existing manifest supplies both random and spaced train, validation, and test memberships.",
        "All new RGFN selection uses validation metrics only; every final seed model receives one test evaluation after selection.",
        "Logistic regression is a fresh current-manifest run because no saved logistic result matched these labels and partitions.",
        "",
        "CAUSAL FEATURE INTEGRITY",
        pd.DataFrame([integrity]).to_string(index=False),
        "",
        "FULL PREDEFINED CONFIGURATION COMPARISON",
        "All predefined configurations are reported. Held-out test metrics are descriptive comparisons only and do not select or name a winning arm.",
        comparison.to_string(index=False),
        "",
        "ATTRIBUTION",
        attribution.to_string(index=False),
        "",
        "SELECTED CONFIGURATIONS AND CHECKPOINT COUNTS",
        configurations.to_string(index=False),
        "",
        "ARM 2 COARSE SEARCH",
        pd.DataFrame([search_details]).to_string(index=False),
        "",
        "FEATURE ACCOUNTING",
        "Arm 1 uses the original hourly inputs plus the six causal trailing static values and sixteen trailing evidence summaries listed above.",
        "Historical episode-duration values are excluded because their endpoint is unavailable at the scored hour.",
        "Arm 2 retains the original hourly feature set. Arm 3 uses Arm 1 inputs with Arm 2's selected architecture, optimizer, class weight, and threshold.",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    args = parse_args().parse_args(argv)
    require_files(
        "Hourly RGFN tuning",
        {
            "short hourly tensor": args.tensor,
            "baseline split manifest": args.manifest,
            "calibrated baseline metrics": args.boosted_metrics,
            "prior RGFN metrics": args.prior_rgfn_metrics,
            "canonical merged dataset": SOURCE_PATH,
            "feature matrix": FEATURE_PATH,
        },
        "Build hourly data, then run baseline, calibration, and RGFN training before tuning.",
    )
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models" / "rgfn_hourly_tuning"
    examples, excluded_removed = filter_eligible_examples(load_hourly_tensor(Path(args.tensor)))
    splits = load_manifest_splits(examples, Path(args.manifest))
    raw_hourly = load_hourly_frame()
    augmented = augment_hourly_rgfn_examples(examples, raw_hourly)
    integrity = _verify_augmentation(examples, augmented)
    boosted = load_calibrated_baseline(Path(args.boosted_metrics))
    prior = _load_prior_rgfn(Path(args.prior_rgfn_metrics))
    search_configs = make_coarse_search_configurations(count=int(args.search_count))
    coverage = search_configuration_coverage(search_configs)
    print(f"device={DEVICE}", flush=True)
    print(f"eligible_hourly_examples={len(examples['y_binary'])}", flush=True)
    print(f"arm2_coarse_configurations={len(search_configs)}", flush=True)

    logistic: dict[str, dict[str, object]] = {}
    arm1: dict[str, dict[str, object]] = {}
    arm2: dict[str, dict[str, object]] = {}
    arm3: dict[str, dict[str, object]] = {}
    screen_details: dict[str, dict[str, object]] = {}
    for split_name in ("random", "spaced"):
        print(f"training=logistic_regression split={split_name}", flush=True)
        logistic[split_name] = train_hourly_logistic_variant(
            examples,
            splits[split_name],
            split_name,
            model_dir=model_dir,
        )
        print(f"training=rgfn_arm1_features_only split={split_name}", flush=True)
        arm1[split_name] = run_feature_only_arm(
            augmented,
            splits[split_name],
            split_name,
            "rgfn_arm1_features_only",
            model_dir=model_dir,
            progress=_progress,
        )
        print(f"training=rgfn_arm2_screen split={split_name}", flush=True)
        screen = screen_architecture_search(
            examples,
            splits[split_name],
            split_name,
            search_configs,
            model_dir=model_dir,
            arm_name="rgfn_arm2_screen",
            progress=_progress,
        )
        top_configs = select_top_screened_configurations(screen, top_count=int(args.top_count))
        screen_details[split_name] = {
            "coverage": screen.coverage,
            "screening_seed": screen.screening_seed,
            "candidate_model_paths": screen.candidate_model_paths,
            "validation_rows": screen.validation_rows,
            "validation_grid": screen.validation_grid,
            "top_configurations": [configuration_payload(config) for config in top_configs],
        }
        print(f"training=rgfn_arm2_tuned_only split={split_name}:top_configs={len(top_configs)}", flush=True)
        tuned = finalize_architecture_search(
            screen,
            "rgfn_arm2_tuned_only",
            top_configs=top_configs,
            model_dir=model_dir,
            progress=_progress,
        )
        arm2[split_name] = final_evaluate_hourly_rgfn_arm(
            tuned,
            examples,
            splits[split_name],
            model_dir=model_dir,
        )
        print(f"training=rgfn_arm3_combined split={split_name}", flush=True)
        arm3[split_name] = run_combined_arm(
            augmented,
            splits[split_name],
            split_name,
            "rgfn_arm3_combined",
            tuned.selected_config,
            tuned.selected_threshold,
            model_dir=model_dir,
            progress=_progress,
        )

    comparison = master_tuning_comparison_frame(logistic, boosted, prior, arm1, arm2, arm3)
    attribution = _attribution_frame(prior, arm1, arm2, arm3)
    configurations = _selected_config_frame(arm1, arm2, arm3)
    validation = pd.concat(
        [
            _validation_frame(arm1, "rgfn_arm1_features_only"),
            _validation_frame(arm2, "rgfn_arm2_tuned_only"),
            _validation_frame(arm3, "rgfn_arm3_combined"),
        ],
        ignore_index=True,
    )
    screening = _screening_frame(screen_details)
    search_details = {
        "screened_combined_configurations_per_split": len(search_configs),
        "screening_seed_per_configuration": 1,
        "finalist_configurations_per_split": int(args.top_count),
        "additional_finalist_seeds_per_configuration": len(TUNING_SEEDS) - 1,
        "thresholds_per_prediction": 11,
        "coverage": coverage,
    }
    inputs = {
        "device": str(DEVICE),
        "tensor": str(Path(args.tensor)),
        "split_manifest": str(Path(args.manifest)),
        "boosted_metrics": str(Path(args.boosted_metrics)),
        "prior_rgfn_metrics": str(Path(args.prior_rgfn_metrics)),
        "eligible_hourly_examples": int(len(examples["y_binary"])),
        "excluded_examples_removed": int(excluded_removed),
    }
    payload = {
        "configuration": {
            "task": "hour-level binary fault detection",
            "window_hours": 7,
            "device": str(DEVICE),
            "seeds": list(TUNING_SEEDS),
            "selection": "aggregate validation maximum of minimum precision, recall, and f1",
            "test_evaluation": "one evaluation per selected seed model",
            "arm1": {
                "feature_set": "causal augmentation with prior defaults",
                "training": {
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "batch_size": 64,
                    "max_epochs": 100,
                    "patience": 10,
                },
            },
            "arm2": {
                "feature_set": "original hourly features only",
                "search": search_details,
            },
            "arm3": {
                "feature_set": "causal augmentation",
                "configuration_source": "Arm 2 validation selection",
            },
        },
        "inputs": inputs,
        "causal_feature_integrity": integrity,
        "logistic_regression": logistic,
        "gradient_boosted": boosted,
        "rgfn_gru_prior": prior,
        "rgfn_arm1_features_only": arm1,
        "rgfn_arm2_tuned_only": arm2,
        "rgfn_arm3_combined": arm3,
        "arm2_screen": screen_details,
        "master_test_comparison": comparison.to_dict(orient="records"),
        "attribution": attribution.to_dict(orient="records"),
        "selected_configurations": configurations.to_dict(orient="records"),
    }
    report = tuning_report(inputs, integrity, comparison, attribution, configurations, search_details)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(_jsonable(payload), output_dir / METRICS_PATH.name)
    (output_dir / REPORT_PATH.name).write_text(report + "\n", encoding="utf-8")
    validation.to_csv(output_dir / SWEEP_PATH.name, index=False)
    screening.to_csv(output_dir / SCREENING_PATH.name, index=False)
    print(report, flush=True)
    print(f"metrics={output_dir / METRICS_PATH.name}", flush=True)
    print(f"report={output_dir / REPORT_PATH.name}", flush=True)
    print(f"validation_sweep={output_dir / SWEEP_PATH.name}", flush=True)
    print(f"architecture_screen={output_dir / SCREENING_PATH.name}", flush=True)
    print(f"models={model_dir}", flush=True)


if __name__ == "__main__":
    main()
