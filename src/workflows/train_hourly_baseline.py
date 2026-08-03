from __future__ import annotations

import gc
from argparse import ArgumentParser
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.hourly_baseline import (
    HourlyBaselineConfig,
    HourlyReasonCodeConfig,
    HourlyReasonCodeLogisticConfig,
    HourlyLogisticScaler,
    aggregate_reason_code_prediction_rows,
    assert_matching_metadata,
    baseline_report,
    evaluate_reason_code_models,
    fit_reason_code_logistic_models,
    filter_eligible_examples,
    flatten_hourly_features,
    grouped_permutation_importance,
    load_hourly_metadata,
    load_hourly_tensor,
    load_reason_code_manifest_splits,
    make_split_manifest,
    prepare_reason_code_population,
    reason_code_aggregated_metric_rows,
    reason_code_connected_group_metadata,
    select_reason_code_event_configuration,
    reason_code_prediction_frame,
    reason_code_comparison_rows,
    reason_code_method_comparison_report,
    reason_code_threshold_robustness_rows,
    reason_code_report,
    reason_code_summary_rows,
    resolve_fault_class_weight,
    run_baseline_matrix,
    save_model_bundle,
    save_reason_code_model,
    spaced_split_indices,
    fit_reason_code_models,
    validation_importance_reference,
    write_metrics_json,
)
from src.model.hourly_detection import FEATURE_PATH, LONG_TENSOR_PATH, MASK_MODE_PER_HOUR, MASK_MODES, SHORT_TENSOR_PATH
from src.model.hourly_rgfn import HourlyRgfnConfig, build_hourly_rgfn
from src.model.hourly_rgfn_training import (
    DEVICE,
    HourlyReasonCodeRgfnConfig,
    HourlyRgfnScaler,
    evaluate_reason_code_rgfn_models,
    fit_reason_code_rgfn_models,
    reason_code_rgfn_config_from_arm2,
    save_reason_code_rgfn_model,
)
from src.workflows.prerequisites import require_files


OUTPUT_DIR = PROJECT_ROOT / "data" / "hourly_detection"
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_PATH = OUTPUT_DIR / "hourly_baseline_metrics.json"
REPORT_PATH = OUTPUT_DIR / "hourly_baseline_report.txt"
IMPORTANCE_PATH = OUTPUT_DIR / "hourly_baseline_feature_importance.csv"
MANIFEST_PATH = OUTPUT_DIR / "hourly_baseline_split_manifest.csv"
REASON_CODE_METRICS_PATH = OUTPUT_DIR / "reason_code_metrics.json"
REASON_CODE_REPORT_PATH = OUTPUT_DIR / "reason_code_report.txt"
REASON_CODE_PREDICTIONS_PATH = OUTPUT_DIR / "reason_code_predictions.csv"
REASON_CODE_COMPARISON_METRICS_PATH = OUTPUT_DIR / "reason_code_method_comparison_metrics.json"
REASON_CODE_COMPARISON_REPORT_PATH = OUTPUT_DIR / "reason_code_method_comparison_report.txt"
REASON_CODE_COMPARISON_PREDICTIONS_PATH = OUTPUT_DIR / "reason_code_method_comparison_predictions.csv"
REASON_CODE_THRESHOLD_ROBUSTNESS_PATH = OUTPUT_DIR / "reason_code_threshold_robustness.csv"
REASON_CODE_EPISODE_AGGREGATION_PATH = OUTPUT_DIR / "reason_code_episode_aggregation.csv"
REASON_CODE_EPISODE_METRICS_PATH = OUTPUT_DIR / "reason_code_episode_aggregation_metrics.csv"
REASON_CODE_VALIDATION_EVENT_AGGREGATION_PATH = OUTPUT_DIR / "reason_code_validation_event_aggregation.csv"
REASON_CODE_VALIDATION_EVENT_METRICS_PATH = OUTPUT_DIR / "reason_code_validation_event_metrics.csv"
REASON_CODE_MECHANISM_SELECTION_PATH = OUTPUT_DIR / "reason_code_mechanism_selection.csv"
REASON_CODE_MECHANISM_SELECTION_TRACE_PATH = OUTPUT_DIR / "reason_code_mechanism_selection.json"
REASON_CODE_SELECTED_MECHANISM_TEST_METRICS_PATH = OUTPUT_DIR / "reason_code_selected_mechanism_test_metrics.csv"
REASON_CODE_HOUR_VS_EPISODE_PATH = OUTPUT_DIR / "reason_code_hour_vs_episode_f1.csv"
REASON_CODE_GROUP_DIAGNOSTICS_PATH = OUTPUT_DIR / "reason_code_event_group_diagnostics.csv"
REASON_CODE_POSTPROCESS_METRICS_PATH = OUTPUT_DIR / "reason_code_postprocess_metrics.json"
REASON_CODE_POSTPROCESS_REPORT_PATH = OUTPUT_DIR / "reason_code_postprocess_report.txt"
RGFN_TUNING_METRICS_PATH = OUTPUT_DIR / "hourly_rgfn_tuning_metrics.json"


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


def parse_reason_code_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Train conditional reason-code models; mechanism event diagnosis is the delivered reporting scope.",
    )
    parser.add_argument("--short-tensor", type=Path, default=SHORT_TENSOR_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--features", type=Path, default=FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rgfn-seed", type=int, default=0)
    parser.add_argument("--rgfn-arm2-metrics", type=Path, default=RGFN_TUNING_METRICS_PATH)
    parser.add_argument("--reason-code-timebox-minutes", type=float, default=60.0)
    parser.add_argument(
        "--rgfn-timebox-minutes",
        type=float,
        default=None,
        help="Optional smaller cap inside the 60-minute reason-code workflow timebox.",
    )
    parser.add_argument(
        "--postprocess-existing",
        action="store_true",
        help="Aggregate existing reason-code predictions as a retrospective event-level analysis without fitting models.",
    )
    parser.add_argument(
        "--postprocess-timebox-minutes",
        type=float,
        default=30.0,
        help="Maximum allowed time for no-refit reason-code post-processing.",
    )
    parser.add_argument(
        "--comparison-predictions",
        type=Path,
        default=REASON_CODE_COMPARISON_PREDICTIONS_PATH,
    )
    parser.add_argument(
        "--comparison-metrics",
        type=Path,
        default=REASON_CODE_COMPARISON_METRICS_PATH,
    )
    parser.add_argument(
        "--reason-code-model-dir",
        type=Path,
        default=MODEL_DIR / "reason_codes",
    )
    parser.add_argument(
        "--source-reason-code-commit",
        default="a162cf6",
        help="Recorded provenance for the existing fitted artifacts; it is not a model-embedded claim.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _binary_artifact_snapshot(short_tensor: Path, manifest: Path) -> dict[str, str]:
    candidates = {
        Path(short_tensor),
        Path(manifest),
        OUTPUT_DIR / "hourly_baseline_metrics.json",
        OUTPUT_DIR / "hourly_baseline_report.txt",
        OUTPUT_DIR / "hourly_baseline_split_manifest.csv",
        OUTPUT_DIR / "hourly_baseline_70_15_15_and_80_20_metrics.json",
        OUTPUT_DIR / "hourly_baseline_70_15_15_and_80_20_report.txt",
        OUTPUT_DIR / "hourly_baseline_70_15_15_and_80_20_split_manifest.csv",
    }
    model_dir = OUTPUT_DIR / "models"
    if OUTPUT_DIR.exists():
        candidates.update(
            path
            for path in OUTPUT_DIR.iterdir()
            if path.is_file()
            and not path.name.startswith("reason_code")
            and path.suffix.lower() in {".csv", ".json", ".txt"}
        )
    if model_dir.exists():
        candidates.update(
            path
            for path in model_dir.rglob("*")
            if path.is_file() and "reason_codes" not in path.relative_to(model_dir).parts
        )
    result: dict[str, str] = {}
    for path in sorted(candidates):
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            name = str(resolved.relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            name = str(resolved)
        result[name] = _sha256(path)
    return result


def _reason_code_arm2_configs(
    path: Path,
    seed: int,
) -> tuple[dict[str, HourlyReasonCodeRgfnConfig], dict[str, dict[str, object]]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    arm = payload.get("rgfn_arm2_tuned_only")
    if not isinstance(arm, dict):
        raise KeyError("RGFN tuning metrics lack rgfn_arm2_tuned_only")
    configurations: dict[str, HourlyReasonCodeRgfnConfig] = {}
    source: dict[str, dict[str, object]] = {}
    for scheme in ("random", "spaced"):
        result = arm.get(scheme)
        if not isinstance(result, dict):
            raise KeyError(f"RGFN tuning metrics lack Arm 2 {scheme} results")
        selected = result.get("selected_config")
        if not isinstance(selected, dict):
            raise KeyError(f"RGFN tuning metrics lack Arm 2 {scheme} selected_config")
        source[scheme] = dict(selected)
        configurations[scheme] = reason_code_rgfn_config_from_arm2(selected, seed=seed)
    return configurations, source


def _reason_code_axes(examples: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for axis, target_key, names_key in (
        ("mechanism", "y_mechanism", "mechanism_label_names"),
        ("component", "y_component", "component_label_names"),
    ):
        if target_key not in examples or names_key not in examples:
            raise KeyError(f"reason-code tensor lacks {axis} targets")
        targets = np.asarray(examples[target_key], dtype=int)
        names = np.asarray(examples[names_key], dtype=object).astype(str)
        if targets.ndim != 2 or targets.shape != (len(examples["y_binary"]), len(names)):
            raise ValueError(f"reason-code {axis} targets do not match their names")
        result[axis] = (targets, names)
    return result


def _reason_code_station_hour_keys(stations: np.ndarray, hours: np.ndarray) -> np.ndarray:
    station_values = np.asarray(stations, dtype=object).astype(str)
    parsed = pd.to_datetime(pd.Series(hours), utc=True, format="mixed")
    ticks = parsed.astype("int64").to_numpy(dtype=np.int64)
    return np.asarray(
        [f"{station}\x1f{int(tick)}" for station, tick in zip(station_values, ticks)],
        dtype=object,
    )


def _reason_code_long_test_rows(
    predictions: pd.DataFrame,
    examples: dict[str, np.ndarray],
    group_ids: np.ndarray,
) -> pd.DataFrame:
    required = {
        "method",
        "split_scheme",
        "split",
        "station_id",
        "hour",
        "source_episode_ids",
        "true_mechanisms",
        "true_components",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise KeyError(f"existing reason-code predictions lack fields: {missing}")
    if not predictions["split"].eq("test").all():
        raise ValueError("existing reason-code post-processing input must contain held-out test rows only")
    fault_keys = _reason_code_station_hour_keys(examples["station_id"], examples["hour"])
    if len(np.unique(fault_keys)) != len(fault_keys):
        raise ValueError("fault-only reason-code tensor has duplicate station-hour keys")
    group_lookup = dict(zip(fault_keys.astype(str), np.asarray(group_ids, dtype=object).astype(str)))
    source = predictions.copy()
    source["_sample_key"] = _reason_code_station_hour_keys(
        source["station_id"].to_numpy(), source["hour"].to_numpy()
    )
    source["connected_episode_group_id"] = source["_sample_key"].map(group_lookup)
    if source["connected_episode_group_id"].isna().any():
        raise ValueError("existing prediction rows are not all present in the fault-only tensor")
    base_columns = [
        "method",
        "split_scheme",
        "split",
        "station_id",
        "hour",
        "source_episode_ids",
        "connected_episode_group_id",
        "tensor_detector_groups",
        "detector_groups_fired",
        "detector_channels_fired",
        "detector_component_evidence",
    ]
    available_base = [column for column in base_columns if column in source.columns]
    output: list[pd.DataFrame] = []
    for axis, (_, names) in _reason_code_axes(examples).items():
        true_column = f"true_{axis}s"
        for name in names:
            probability_column = f"{axis}_probability_{name}"
            threshold_column = f"{axis}_threshold_{name}"
            prediction_column = f"{axis}_prediction_{name}"
            required_columns = [probability_column, threshold_column, prediction_column]
            missing_columns = [column for column in required_columns if column not in source.columns]
            if missing_columns:
                raise KeyError(f"existing prediction rows lack {missing_columns}")
            frame = source.loc[:, available_base].copy()
            frame["axis"] = axis
            frame["label"] = str(name)
            text = source[true_column].fillna("").astype(str)
            frame["truth"] = text.map(
                lambda value: int(str(name) in {part for part in value.split("|") if part})
            )
            frame["probability"] = pd.to_numeric(source[probability_column], errors="raise")
            frame["threshold"] = pd.to_numeric(source[threshold_column], errors="raise")
            frame["hourly_prediction"] = pd.to_numeric(source[prediction_column], errors="raise")
            output.append(frame)
    return pd.concat(output, ignore_index=True)


def _reason_code_long_inference_rows(
    method: str,
    split_scheme: str,
    split: str,
    indices: np.ndarray,
    examples: dict[str, np.ndarray],
    group_ids: np.ndarray,
    probabilities_by_axis: dict[str, np.ndarray],
    thresholds_by_axis: dict[str, np.ndarray],
) -> pd.DataFrame:
    selected = np.asarray(indices, dtype=np.int64)
    axes = _reason_code_axes(examples)
    output: list[pd.DataFrame] = []
    for axis, (targets, names) in axes.items():
        probabilities = np.asarray(probabilities_by_axis[axis], dtype=float)
        thresholds = np.asarray(thresholds_by_axis[axis], dtype=float)
        if probabilities.shape != (len(selected), len(names)) or thresholds.shape != (len(names),):
            raise ValueError(f"{method} {split_scheme} {axis} inference has an invalid shape")
        base = pd.DataFrame(
            {
                "method": str(method),
                "split_scheme": str(split_scheme),
                "split": str(split),
                "station_id": np.asarray(examples["station_id"], dtype=object)[selected].astype(str),
                "hour": np.asarray(examples["hour"], dtype=object)[selected].astype(str),
                "source_episode_ids": np.asarray(examples["source_episode_ids"], dtype=object)[selected].astype(str),
                "connected_episode_group_id": np.asarray(group_ids, dtype=object)[selected].astype(str),
                "axis": axis,
            }
        )
        for index, name in enumerate(names):
            frame = base.copy()
            frame["label"] = str(name)
            frame["truth"] = np.asarray(targets[selected, index], dtype=int)
            frame["probability"] = probabilities[:, index]
            frame["threshold"] = float(thresholds[index])
            frame["hourly_prediction"] = np.asarray(
                probabilities[:, index] >= float(thresholds[index]), dtype=int
            )
            output.append(frame)
    return pd.concat(output, ignore_index=True)


def _validate_existing_test_membership(
    predictions: pd.DataFrame,
    examples: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
) -> None:
    required = {
        "method",
        "split_scheme",
        "split",
        "station_id",
        "hour",
        "source_episode_ids",
        "true_mechanisms",
        "true_components",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise KeyError(f"existing reason-code predictions lack membership fields: {missing}")
    if not predictions["split"].eq("test").all():
        raise ValueError("existing reason-code predictions include a non-test row")
    axes = _reason_code_axes(examples)
    for scheme in ("random", "spaced"):
        expected_indices = np.asarray(splits[scheme]["test"], dtype=np.int64)
        expected_keys = _reason_code_station_hour_keys(
            np.asarray(examples["station_id"], dtype=object)[expected_indices],
            np.asarray(examples["hour"], dtype=object)[expected_indices],
        )
        expected_lookup = {str(key): int(index) for key, index in zip(expected_keys, expected_indices)}
        for method in ("logistic", "gradient_boosted", "rgfn"):
            frame = predictions.loc[
                predictions["method"].eq(method) & predictions["split_scheme"].eq(scheme)
            ].copy()
            keys = _reason_code_station_hour_keys(
                frame["station_id"].to_numpy(), frame["hour"].to_numpy()
            )
            if len(frame) != len(expected_indices) or len(np.unique(keys)) != len(keys):
                raise ValueError(
                    f"existing {method} {scheme} predictions do not contain one row per frozen test key"
                )
            if set(keys.astype(str).tolist()) != set(expected_lookup):
                raise ValueError(
                    f"existing {method} {scheme} predictions do not match the frozen manifest test keys"
                )
            mapped = np.asarray([expected_lookup[str(key)] for key in keys], dtype=np.int64)
            expected_source = np.asarray(examples["source_episode_ids"], dtype=object)[mapped].astype(str)
            if not np.array_equal(frame["source_episode_ids"].astype(str).to_numpy(), expected_source):
                raise ValueError(f"existing {method} {scheme} prediction episode identifiers disagree with the tensor")
            for axis, (targets, names) in axes.items():
                expected_text = np.asarray(
                    [
                        "|".join(str(name) for name, active in zip(names, row) if int(active) == 1)
                        for row in np.asarray(targets[mapped], dtype=int)
                    ],
                    dtype=object,
                )
                actual_text = frame[f"true_{axis}s"].fillna("").astype(str).to_numpy(dtype=object)
                if not np.array_equal(actual_text, expected_text):
                    raise ValueError(
                        f"existing {method} {scheme} {axis} truth strings disagree with the tensor"
                    )


def _load_tabular_partition_probabilities(
    method: str,
    model_dir: Path,
    values: np.ndarray,
    examples: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
    partition: str,
) -> dict[str, dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]]:
    axes = _reason_code_axes(examples)
    result: dict[str, dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]] = {}
    for scheme in ("random", "spaced"):
        indices = np.asarray(splits[scheme][partition], dtype=np.int64)
        by_axis: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
        for axis, (_, names) in axes.items():
            probabilities = np.zeros((len(indices), len(names)), dtype=float)
            thresholds = np.zeros(len(names), dtype=float)
            for index, name in enumerate(names):
                path = model_dir / method / scheme / f"{axis}_{name}.joblib"
                payload = joblib.load(path)
                if (
                    payload.get("mode") != "reason_codes"
                    or payload.get("method") != method
                    or payload.get("split_scheme") != scheme
                    or payload.get("axis") != axis
                    or payload.get("label") != str(name)
                ):
                    raise ValueError(f"reason-code artifact metadata does not match {path}")
                feature_names = payload.get("feature_names")
                flattened_names = flatten_hourly_features(examples, mask_mode=MASK_MODE_PER_HOUR)[1]
                if list(feature_names) != flattened_names:
                    raise ValueError(f"reason-code artifact feature contract does not match {path}")
                selected = np.asarray(values, dtype=np.float32)[indices]
                scaler_payload = payload.get("scaler_payload")
                if scaler_payload is not None:
                    selected = HourlyLogisticScaler(**scaler_payload).transform(selected)
                probabilities[:, index] = np.asarray(
                    payload["estimator"].predict_proba(selected)[:, 1], dtype=float
                )
                thresholds[index] = float(payload["selected_threshold"])
            by_axis[axis] = ({"probabilities": probabilities}, {"thresholds": thresholds})
        result[scheme] = by_axis
    return result


def _load_rgfn_partition_probabilities(
    model_dir: Path,
    examples: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
    partition: str,
) -> dict[str, dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]]:
    axes = _reason_code_axes(examples)
    mechanism_names = axes["mechanism"][1]
    component_names = axes["component"][1]
    expected_labels = np.concatenate((mechanism_names, component_names)).astype(str).tolist()
    result: dict[str, dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]] = {}
    for scheme in ("random", "spaced"):
        path = model_dir / "rgfn" / f"hourly_reason_codes_{scheme}.pt"
        payload = torch.load(path, map_location=DEVICE, weights_only=False)
        if (
            payload.get("mode") != "reason_codes"
            or payload.get("method") != "rgfn"
            or payload.get("split_scheme") != scheme
            or list(payload.get("label_names", [])) != expected_labels
        ):
            raise ValueError(f"RGFN artifact metadata does not match {path}")
        configuration = HourlyReasonCodeRgfnConfig(**dict(payload["configuration"]))
        architecture = HourlyRgfnConfig(
            n_continuous=int(payload["n_continuous"]),
            n_static=int(payload["n_static"]),
            n_rule_evidence=int(payload["n_rule_evidence"]),
            window_hours=int(payload["window_hours"]),
            sensor_hidden_size=int(configuration.sensor_hidden_size),
            evidence_hidden_size=int(configuration.evidence_hidden_size),
            evidence_embed_size=int(configuration.evidence_embed_size),
            fusion_hidden_size=int(configuration.gate_hidden_size),
            dropout=float(configuration.dropout),
            mask_mode=str(payload["mask_mode"]),
        )
        model = build_hourly_rgfn(
            str(payload["encoder"]),
            config=architecture,
            output_dim=len(expected_labels),
            mechanism_count=int(payload["mechanism_count"]),
        ).to(DEVICE)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        scaler = HourlyRgfnScaler(**dict(payload["scaler"]))
        values = scaler.transform(examples)
        indices = np.asarray(splits[scheme][partition], dtype=np.int64)
        output: list[np.ndarray] = []
        batch_size = max(1, int(configuration.batch_size))
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                features = (
                    torch.as_tensor(np.ascontiguousarray(values["X_cont"][batch]), dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(np.ascontiguousarray(values["mask"][batch]), dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(np.ascontiguousarray(values["time_since_last"][batch]), dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(np.ascontiguousarray(values["static"][batch]), dtype=torch.float32, device=DEVICE),
                    torch.as_tensor(np.ascontiguousarray(values["rule_evidence"][batch]), dtype=torch.float32, device=DEVICE),
                )
                output.append(
                    model(*features)["reason_code_probabilities"].detach().cpu().numpy().astype(float)
                )
        probabilities = np.concatenate(output, axis=0) if output else np.empty((0, len(expected_labels)))
        thresholds = np.asarray(payload["selected_thresholds"], dtype=float)
        result[scheme] = {
            "mechanism": (
                {"probabilities": probabilities[:, : len(mechanism_names)]},
                {"thresholds": thresholds[: len(mechanism_names)]},
            ),
            "component": (
                {"probabilities": probabilities[:, len(mechanism_names) :]},
                {"thresholds": thresholds[len(mechanism_names) :]},
            ),
        }
    return result


def _reason_code_saved_model_inference_rows(
    model_dir: Path,
    examples: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
    group_ids: np.ndarray,
    partition: str,
) -> pd.DataFrame:
    if partition not in {"validation", "test"}:
        raise ValueError("saved reason-code inference is limited to validation or test partitions")
    values, _, _ = flatten_hourly_features(examples, mask_mode=MASK_MODE_PER_HOUR)
    model_values = {
        "logistic": _load_tabular_partition_probabilities(
            "logistic", model_dir, values, examples, splits, partition
        ),
        "gradient_boosted": _load_tabular_partition_probabilities(
            "gradient_boosted", model_dir, values, examples, splits, partition
        ),
        "rgfn": _load_rgfn_partition_probabilities(model_dir, examples, splits, partition),
    }
    rows: list[pd.DataFrame] = []
    for method, schemes in model_values.items():
        for scheme, axes in schemes.items():
            indices = np.asarray(splits[scheme][partition], dtype=np.int64)
            rows.append(
                _reason_code_long_inference_rows(
                    method,
                    scheme,
                    partition,
                    indices,
                    examples,
                    group_ids,
                    {
                        axis: np.asarray(payload[0]["probabilities"], dtype=float)
                        for axis, payload in axes.items()
                    },
                    {
                        axis: np.asarray(payload[1]["thresholds"], dtype=float)
                        for axis, payload in axes.items()
                    },
                )
            )
    return pd.concat(rows, ignore_index=True)


def _verify_saved_prediction_artifact(
    saved_rows: pd.DataFrame,
    inferred_rows: pd.DataFrame,
    probability_tolerance: float = 2e-6,
    threshold_tolerance: float = 1e-10,
) -> dict[str, dict[str, object]]:
    identity = ["method", "split_scheme", "axis", "label"]
    required = {
        *identity,
        "station_id",
        "hour",
        "source_episode_ids",
        "truth",
        "probability",
        "threshold",
        "hourly_prediction",
    }
    missing_saved = sorted(required.difference(saved_rows.columns))
    missing_inferred = sorted(required.difference(inferred_rows.columns))
    if missing_saved or missing_inferred:
        raise KeyError(
            f"saved/inferred reason-code rows lack fields: saved={missing_saved}, inferred={missing_inferred}"
        )
    left = saved_rows.loc[:, sorted(required)].copy()
    right = inferred_rows.loc[:, sorted(required)].copy()
    for frame in (left, right):
        frame["_sample_key"] = _reason_code_station_hour_keys(
            frame["station_id"].to_numpy(), frame["hour"].to_numpy()
        ).astype(str)
    keys = [*identity, "_sample_key"]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("saved/inferred reason-code prediction rows duplicate a label-level station-hour key")
    compared = left.merge(right, on=keys, how="outer", suffixes=("_saved", "_inferred"), indicator=True)
    if not compared["_merge"].eq("both").all():
        raise ValueError("saved reason-code predictions and loaded-model inference have different row membership")
    details: dict[str, dict[str, object]] = {}
    for method, frame in compared.groupby("method", sort=True):
        same_source = frame["source_episode_ids_saved"].astype(str).eq(
            frame["source_episode_ids_inferred"].astype(str)
        )
        same_truth = pd.to_numeric(frame["truth_saved"], errors="raise").eq(
            pd.to_numeric(frame["truth_inferred"], errors="raise")
        )
        same_prediction = pd.to_numeric(frame["hourly_prediction_saved"], errors="raise").eq(
            pd.to_numeric(frame["hourly_prediction_inferred"], errors="raise")
        )
        probability_delta = np.abs(
            pd.to_numeric(frame["probability_saved"], errors="raise").to_numpy(dtype=float)
            - pd.to_numeric(frame["probability_inferred"], errors="raise").to_numpy(dtype=float)
        )
        threshold_delta = np.abs(
            pd.to_numeric(frame["threshold_saved"], errors="raise").to_numpy(dtype=float)
            - pd.to_numeric(frame["threshold_inferred"], errors="raise").to_numpy(dtype=float)
        )
        max_probability_delta = float(np.max(probability_delta)) if len(probability_delta) else 0.0
        max_threshold_delta = float(np.max(threshold_delta)) if len(threshold_delta) else 0.0
        if (
            not same_source.all()
            or not same_truth.all()
            or not same_prediction.all()
            or max_probability_delta > float(probability_tolerance)
            or max_threshold_delta > float(threshold_tolerance)
        ):
            raise ValueError(
                f"saved {method} reason-code predictions do not match inference from the loaded artifact"
            )
        details[str(method)] = {
            "rows": int(len(frame)),
            "source_episode_ids_match": True,
            "truth_match": True,
            "binary_prediction_match": True,
            "max_abs_probability_delta": max_probability_delta,
            "probability_tolerance": float(probability_tolerance),
            "max_abs_threshold_delta": max_threshold_delta,
            "threshold_tolerance": float(threshold_tolerance),
        }
    return details


def _reason_code_model_hashes(model_dir: Path) -> dict[str, str]:
    paths = sorted(path for path in Path(model_dir).rglob("*") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"reason-code model directory is empty: {model_dir}")
    return {
        str(path.resolve().relative_to(PROJECT_ROOT.resolve())): _sha256(path)
        for path in paths
    }


def _require_complete_reason_code_model_set(model_dir: Path, examples: dict[str, np.ndarray]) -> None:
    required: list[Path] = []
    for method in ("logistic", "gradient_boosted"):
        for scheme in ("random", "spaced"):
            for axis, (_, names) in _reason_code_axes(examples).items():
                required.extend(
                    model_dir / method / scheme / f"{axis}_{name}.joblib"
                    for name in names
                )
    required.extend(
        model_dir / "rgfn" / f"hourly_reason_codes_{scheme}.pt"
        for scheme in ("random", "spaced")
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        lines = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"saved reason-code artifacts are incomplete:\n{lines}")


def _frozen_selection_test_metrics(
    aggregated_test_rows: pd.DataFrame,
    mechanism_selection: dict[str, object],
) -> pd.DataFrame:
    required = {
        "selected_method",
        "selected_aggregation_rule",
        "selection_split_scheme",
        "axis",
        "selection_labels",
    }
    missing = sorted(required.difference(mechanism_selection))
    if missing:
        raise KeyError(f"reason-code selection trace lacks fields: {missing}")
    if str(mechanism_selection["selection_split_scheme"]) != "spaced":
        raise ValueError("frozen reason-code test evaluation requires the spaced selection split")
    if str(mechanism_selection["axis"]) != "mechanism":
        raise ValueError("frozen reason-code test evaluation is limited to mechanism labels")
    labels = [str(value) for value in mechanism_selection["selection_labels"]]
    if not labels:
        raise ValueError("frozen reason-code test evaluation has no selected labels")
    selected = aggregated_test_rows.loc[
        aggregated_test_rows["method"].eq(str(mechanism_selection["selected_method"]))
        & aggregated_test_rows["split_scheme"].eq("spaced")
        & aggregated_test_rows["aggregation_rule"].eq(
            str(mechanism_selection["selected_aggregation_rule"])
        )
        & aggregated_test_rows["axis"].eq("mechanism")
        & aggregated_test_rows["label"].isin(labels)
    ].copy()
    if set(selected["label"].astype(str)) != set(labels):
        raise ValueError("frozen reason-code test evaluation is missing a selected mechanism label")
    metrics = pd.DataFrame(reason_code_aggregated_metric_rows(selected))
    per_label = metrics.loc[metrics["average"].eq("per_label")]
    supported_labels = sorted(
        per_label.loc[per_label["estimated"].astype(bool), "label"].astype(str).tolist()
    )
    metrics["selection_evaluation_scope"] = (
        "descriptive_heldout_test_of_validation_frozen_configuration"
    )
    metrics["selected_method"] = str(mechanism_selection["selected_method"])
    metrics["selected_aggregation_rule"] = str(
        mechanism_selection["selected_aggregation_rule"]
    )
    metrics["frozen_selection_labels"] = "|".join(labels)
    metrics["frozen_selection_label_count"] = int(len(labels))
    metrics["frozen_selection_labels_with_test_support"] = int(len(supported_labels))
    metrics["frozen_selection_labels_without_test_support"] = "|".join(
        label for label in labels if label not in set(supported_labels)
    )
    metrics["test_metrics_used_for_selection"] = False
    return metrics


def _hour_vs_episode_f1_rows(
    comparison_payload: dict[str, object],
    episode_metrics: pd.DataFrame,
) -> pd.DataFrame:
    raw = comparison_payload.get("per_label_test_metrics")
    if not isinstance(raw, dict):
        raise KeyError("reason-code comparison metrics lack per_label_test_metrics")
    hourly_rows = [row for values in raw.values() if isinstance(values, list) for row in values]
    hourly = pd.DataFrame(hourly_rows)
    required = {"method", "split_scheme", "axis", "label", "f1", "support"}
    missing = sorted(required.difference(hourly.columns))
    if missing:
        raise KeyError(f"reason-code comparison metrics lack fields: {missing}")
    hourly = hourly.loc[:, ["method", "split_scheme", "axis", "label", "f1", "support"]].rename(
        columns={"f1": "hour_level_f1", "support": "hour_level_positive_hours"}
    )
    episode = episode_metrics.loc[
        episode_metrics["average"].eq("per_label"),
        [
            "method",
            "split_scheme",
            "aggregation_rule",
            "axis",
            "label",
            "f1",
            "support",
            "episode_groups",
            "complete_evaluation_groups",
            "fragmented_evaluation_groups",
            "evaluation_partition",
            "evaluation_unit",
        ],
    ].rename(
        columns={
            "f1": "episode_level_f1",
            "support": "episode_level_positive_groups",
        }
    )
    output = episode.merge(hourly, on=["method", "split_scheme", "axis", "label"], how="left", validate="many_to_one")
    output["f1_change_episode_minus_hour"] = output["episode_level_f1"] - output["hour_level_f1"]
    return output.sort_values(
        ["method", "split_scheme", "axis", "label", "aggregation_rule"]
    ).reset_index(drop=True)


def _display_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    shown = frame.copy()
    for column in shown.columns:
        if shown[column].dtype.kind in {"f", "i"}:
            shown[column] = shown[column].map(lambda value: "N/A" if pd.isna(value) else value)
    return shown.to_string(index=False)


def _reason_code_postprocess_report(
    group_diagnostics: pd.DataFrame,
    validation_event_metrics: pd.DataFrame,
    mechanism_selection_candidates: pd.DataFrame,
    mechanism_selection: dict[str, object],
    selected_mechanism_test_metrics: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    hour_vs_episode: pd.DataFrame,
    prediction_artifact_match: dict[str, dict[str, object]],
) -> str:
    random_groups = group_diagnostics.loc[
        group_diagnostics["split_scheme"].eq("random") & group_diagnostics["test_fault_hours"].gt(0)
    ]
    spaced_groups = group_diagnostics.loc[
        group_diagnostics["split_scheme"].eq("spaced") & group_diagnostics["test_fault_hours"].gt(0)
    ]
    macro_micro = episode_metrics.loc[episode_metrics["average"].isin(["macro", "micro"])]
    per_label = episode_metrics.loc[episode_metrics["average"].eq("per_label")]
    validation_macro_micro = validation_event_metrics.loc[
        validation_event_metrics["average"].isin(["macro", "micro"])
        & validation_event_metrics["split_scheme"].eq("spaced")
        & validation_event_metrics["axis"].eq("mechanism")
    ]
    parts = [
        "RETROSPECTIVE REASON-CODE CONNECTED-EVENT-GROUP ANALYSIS",
        "",
        "SCOPE",
        "This is a post-processing experiment over the existing fitted reason-code artifacts. It performs no fitting, tuning sweep, split change, or binary-detector change.",
        "The saved held-out prediction artifact was checked for exact key/truth/prediction agreement and probability agreement within the recorded tolerance against a fresh inference pass from the same unchanged saved artifacts.",
        "",
        "EVALUATION-UNIT CAVEAT",
        "A connected event group joins original source episode IDs that overlap in any station-hour. It is therefore not always one raw source episode; its truth is the union of its constituent labels.",
        f"Random test aggregation contains {len(random_groups)} observed connected-event groups; {int(random_groups['is_complete_within_test'].sum())} are wholly held out and {int((~random_groups['is_complete_within_test']).sum())} are fragments spanning more than one partition.",
        f"Spaced test aggregation contains {len(spaced_groups)} connected-event groups; {int(spaced_groups['is_complete_within_test'].sum())} are wholly held out. Only the spaced rows are complete held-out connected-event evaluation.",
        "Random rows are retained as test-fragment aggregation for transparency, not as an independent episode-level generalisation result.",
        "",
        "AGGREGATION RULES",
        "any: assert a connected event if any constituent held-out hour passed its pre-existing frozen hourly threshold.",
        "majority: assert only if strictly more than half of constituent held-out hours passed that threshold; ties are negative.",
        "mean_probability: assert if the mean held-out hourly probability meets the same pre-existing frozen hourly threshold.",
        "All three predefined rules are reported. No rule is selected from held-out test results.",
        "",
        "VALIDATION-ONLY MECHANISM CONFIGURATION SELECTION",
        "The selector receives only complete spaced validation connected-event groups for mechanism reason codes. Labels with fewer than five positive validation connected events are excluded from selection. It ranks all predefined method/rule configurations by support-qualified validation macro F1, then validation macro precision, accuracy, recall, and fixed lexical names. No held-out reason-code prediction, metric, or report is supplied to, parsed by, or used by the selector.",
        f"selected_method={mechanism_selection['selected_method']}",
        f"selected_aggregation_rule={mechanism_selection['selected_aggregation_rule']}",
        f"selection_labels={'|'.join(str(value) for value in mechanism_selection['selection_labels'])}",
        f"excluded_low_support_validation_labels={'|'.join(str(value) for value in mechanism_selection['excluded_low_support_labels']) or 'none'}",
        "VALIDATION CANDIDATES",
        _display_table(mechanism_selection_candidates),
        "VALIDATION MACRO AND MICRO TABLE",
        _display_table(validation_macro_micro),
        "",
        "FROZEN VALIDATION-SELECTED CONFIGURATION: HELD-OUT TEST",
        "This is the descriptive held-out evaluation of the exact method, aggregation rule, and support-qualified label set frozen from validation. It is produced after selection and does not feed back into any choice.",
        _display_table(selected_mechanism_test_metrics),
        "",
        "HELD-OUT TEST CONNECTED-EVENT-GROUP MACRO AND MICRO TABLE",
        _display_table(macro_micro),
        "",
        "HELD-OUT TEST CONNECTED-EVENT-GROUP PER-LABEL TABLE",
        _display_table(per_label),
        "",
        "HOUR-LEVEL VERSUS CONNECTED-EVENT-GROUP F1",
        _display_table(hour_vs_episode),
        "",
        "REPORTING NOTE",
        "Reason-code labels recover the project rule-and-evidence taxonomy. These results are retrospective and are not independent technician-verified root-cause accuracy.",
        "The selected configuration is a retrospective, ground-truth-connected-event diagnosis choice. It is not a source-ID-free live dashboard deployment result; predicted-event segmentation remains a separate future dashboard step.",
        "SAVED PREDICTION / ARTIFACT CONSISTENCY",
        _display_table(pd.DataFrame.from_dict(prediction_artifact_match, orient="index").reset_index(names="method")),
    ]
    return "\n".join(parts)


def reason_codes_postprocess_main(args: object) -> None:
    if not 0.0 < float(args.postprocess_timebox_minutes) <= 30.0:
        raise ValueError("--postprocess-timebox-minutes must be greater than zero and no more than 30")
    output_dir = Path(args.output_dir)
    model_dir = Path(args.reason_code_model_dir)
    require_files(
        "Reason-code no-refit post-processing",
        {
            "short hourly tensor": args.short_tensor,
            "binary split manifest": args.manifest,
            "existing comparison predictions": args.comparison_predictions,
            "existing comparison metrics": args.comparison_metrics,
            "saved logistic reason-code artifact": model_dir / "logistic" / "random" / "mechanism_spike_impossible.joblib",
            "saved RGFN reason-code artifact": model_dir / "rgfn" / "hourly_reason_codes_random.pt",
        },
        "Run the frozen reason-code comparison before no-refit post-processing.",
    )
    started = time.monotonic()
    binary_before = _binary_artifact_snapshot(args.short_tensor, args.manifest)
    full_examples, excluded = _load_filtered_tensor(args.short_tensor)
    full_splits = load_reason_code_manifest_splits(full_examples, args.manifest)
    fault_examples, fault_splits, partition_trace = prepare_reason_code_population(full_examples, full_splits)
    group_ids, group_diagnostics = reason_code_connected_group_metadata(
        fault_examples["source_episode_ids"], fault_splits
    )
    _require_complete_reason_code_model_set(model_dir, fault_examples)
    model_before = _reason_code_model_hashes(model_dir)
    validation_rows = _reason_code_saved_model_inference_rows(
        model_dir, fault_examples, fault_splits, group_ids, "validation"
    )
    validation_aggregated = aggregate_reason_code_prediction_rows(
        validation_rows,
        group_diagnostics,
        evaluation_partition="validation",
    )
    validation_event_metrics = pd.DataFrame(
        reason_code_aggregated_metric_rows(validation_aggregated)
    )
    mechanism_selection, mechanism_selection_candidates = (
        select_reason_code_event_configuration(validation_event_metrics)
    )
    elapsed = time.monotonic() - started
    if elapsed > float(args.postprocess_timebox_minutes) * 60.0:
        raise TimeoutError(
            "reason-code post-processing reached its timebox before validation-only selection"
        )
    test_wide = pd.read_csv(args.comparison_predictions, keep_default_na=False)
    test_rows = _reason_code_long_test_rows(test_wide, fault_examples, group_ids)
    _validate_existing_test_membership(test_wide, fault_examples, fault_splits)
    inferred_test_rows = _reason_code_saved_model_inference_rows(
        model_dir, fault_examples, fault_splits, group_ids, "test"
    )
    prediction_artifact_match = _verify_saved_prediction_artifact(test_rows, inferred_test_rows)
    aggregated = aggregate_reason_code_prediction_rows(test_rows, group_diagnostics)
    episode_metrics = pd.DataFrame(reason_code_aggregated_metric_rows(aggregated))
    selected_mechanism_test_metrics = _frozen_selection_test_metrics(
        aggregated,
        mechanism_selection,
    )
    with Path(args.comparison_metrics).open("r", encoding="utf-8") as handle:
        comparison_payload = json.load(handle)
    hour_vs_episode = _hour_vs_episode_f1_rows(comparison_payload, episode_metrics)
    binary_after = _binary_artifact_snapshot(args.short_tensor, args.manifest)
    model_after = _reason_code_model_hashes(model_dir)
    if binary_before != binary_after:
        raise RuntimeError("no-refit reason-code post-processing altered a frozen binary artifact")
    if model_before != model_after:
        raise RuntimeError("no-refit reason-code post-processing altered an existing fitted model artifact")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "event_group_diagnostics": output_dir / REASON_CODE_GROUP_DIAGNOSTICS_PATH.name,
        "validation_event_aggregation": output_dir / REASON_CODE_VALIDATION_EVENT_AGGREGATION_PATH.name,
        "validation_event_metrics": output_dir / REASON_CODE_VALIDATION_EVENT_METRICS_PATH.name,
        "mechanism_selection": output_dir / REASON_CODE_MECHANISM_SELECTION_PATH.name,
        "mechanism_selection_trace": output_dir / REASON_CODE_MECHANISM_SELECTION_TRACE_PATH.name,
        "selected_mechanism_test_metrics": output_dir / REASON_CODE_SELECTED_MECHANISM_TEST_METRICS_PATH.name,
        "episode_aggregation": output_dir / REASON_CODE_EPISODE_AGGREGATION_PATH.name,
        "episode_metrics": output_dir / REASON_CODE_EPISODE_METRICS_PATH.name,
        "hour_vs_episode": output_dir / REASON_CODE_HOUR_VS_EPISODE_PATH.name,
        "metrics": output_dir / REASON_CODE_POSTPROCESS_METRICS_PATH.name,
        "report": output_dir / REASON_CODE_POSTPROCESS_REPORT_PATH.name,
    }
    group_diagnostics.to_csv(paths["event_group_diagnostics"], index=False)
    validation_aggregated.to_csv(paths["validation_event_aggregation"], index=False)
    validation_event_metrics.to_csv(paths["validation_event_metrics"], index=False)
    mechanism_selection_candidates.to_csv(paths["mechanism_selection"], index=False)
    write_metrics_json(mechanism_selection, paths["mechanism_selection_trace"])
    selected_mechanism_test_metrics.to_csv(paths["selected_mechanism_test_metrics"], index=False)
    aggregated.to_csv(paths["episode_aggregation"], index=False)
    episode_metrics.to_csv(paths["episode_metrics"], index=False)
    hour_vs_episode.to_csv(paths["hour_vs_episode"], index=False)
    report = _reason_code_postprocess_report(
        group_diagnostics,
        validation_event_metrics,
        mechanism_selection_candidates,
        mechanism_selection,
        selected_mechanism_test_metrics,
        episode_metrics,
        hour_vs_episode,
        prediction_artifact_match,
    )
    paths["report"].write_text(report + "\n", encoding="utf-8")
    payload = {
        "mode": "retrospective_reason_code_event_aggregation_postprocess",
        "postprocess_only": True,
        "no_model_fit_called": True,
        "source_reason_code_commit_recorded": str(args.source_reason_code_commit),
        "source_commit_is_not_embedded_in_legacy_model_artifacts": True,
        "comparison_predictions_path": str(args.comparison_predictions),
        "comparison_predictions_sha256": _sha256(args.comparison_predictions),
        "comparison_metrics_path": str(args.comparison_metrics),
        "comparison_metrics_sha256": _sha256(args.comparison_metrics),
        "manifest_path": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "partition_trace": partition_trace,
        "fault_hours": int(len(fault_examples["y_binary"])),
        "excluded_examples_removed_before_reason_code_filter": int(excluded),
        "event_group_count": int(len(np.unique(group_ids))),
        "random_complete_test_event_groups": int(
            group_diagnostics.loc[
                group_diagnostics["split_scheme"].eq("random")
                & group_diagnostics["is_complete_within_test"],
                "connected_episode_group_id",
            ].nunique()
        ),
        "spaced_complete_test_event_groups": int(
            group_diagnostics.loc[
                group_diagnostics["split_scheme"].eq("spaced")
                & group_diagnostics["is_complete_within_test"],
                "connected_episode_group_id",
            ].nunique()
        ),
        "spaced_complete_validation_event_groups": int(
            group_diagnostics.loc[
                group_diagnostics["split_scheme"].eq("spaced")
                & group_diagnostics["is_complete_within_validation"],
                "connected_episode_group_id",
            ].nunique()
        ),
        "mechanism_configuration_selection": mechanism_selection,
        "configuration_selection_inputs": "fresh_saved_artifact_validation_inference_only",
        "configuration_selection_claim_scope": "retrospective_ground_truth_connected_event_diagnosis_only",
        "test_metrics_read_during_configuration_selection": bool(
            mechanism_selection["test_metrics_read_during_configuration_selection"]
        ),
        "frozen_selection_test_evaluation": {
            "method": str(mechanism_selection["selected_method"]),
            "aggregation_rule": str(mechanism_selection["selected_aggregation_rule"]),
            "labels": [str(value) for value in mechanism_selection["selection_labels"]],
            "test_metrics_used_for_selection": False,
        },
        "saved_test_predictions_match_loaded_artifacts": prediction_artifact_match,
        "binary_artifacts_before": binary_before,
        "binary_artifacts_after": binary_after,
        "binary_artifacts_unchanged": True,
        "reason_code_model_artifacts_before": model_before,
        "reason_code_model_artifacts_after": model_after,
        "reason_code_model_artifacts_unchanged": True,
        "workflow_elapsed_seconds": float(time.monotonic() - started),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    write_metrics_json(payload, paths["metrics"])
    print(report)
    print()
    print("OUTPUTS")
    for name, path in paths.items():
        print(f"{name}={path}")
    print("no_model_fit_called=True")
    print(f"binary_artifacts_unchanged={binary_before == binary_after}")
    print(f"reason_code_model_artifacts_unchanged={model_before == model_after}")


def reason_codes_main(argv: list[str] | None = None) -> None:
    args = parse_reason_code_args().parse_args(argv)
    if bool(args.postprocess_existing):
        reason_codes_postprocess_main(args)
        return
    if not 0.0 < float(args.reason_code_timebox_minutes) <= 60.0:
        raise ValueError("--reason-code-timebox-minutes must be greater than zero and no more than 60")
    if args.rgfn_timebox_minutes is not None and not 0.0 < float(args.rgfn_timebox_minutes) <= 60.0:
        raise ValueError("--rgfn-timebox-minutes must be greater than zero and no more than 60")
    require_files(
        "Hourly reason-code training",
        {
            "short hourly tensor": args.short_tensor,
            "binary split manifest": args.manifest,
            "feature matrix for localisation evidence": args.features,
            "Arm 2 RGFN tuning metrics": args.rgfn_arm2_metrics,
        },
        "Run scripts/build_hourly_dataset.py and the frozen binary baseline before reason-code training.",
    )
    workflow_started = time.monotonic()
    workflow_deadline = workflow_started + float(args.reason_code_timebox_minutes) * 60.0
    binary_before = _binary_artifact_snapshot(args.short_tensor, args.manifest)
    full_examples, excluded = _load_filtered_tensor(args.short_tensor)
    full_splits = load_reason_code_manifest_splits(full_examples, args.manifest)
    fault_examples, fault_splits, partition_trace = prepare_reason_code_population(
        full_examples,
        full_splits,
    )
    if not np.equal(np.asarray(fault_examples["y_binary"], dtype=int), 1).all():
        raise RuntimeError("reason-code training population contains a non-fault hour")
    if not np.equal(
        np.asarray(fault_examples["display_state"], dtype=object).astype(str),
        "fault",
    ).all():
        raise RuntimeError("reason-code training population contains an excluded or non-fault state")
    values, feature_names, _ = flatten_hourly_features(
        fault_examples,
        mask_mode=MASK_MODE_PER_HOUR,
    )
    logistic_config = HourlyReasonCodeLogisticConfig(seed=int(args.seed))
    logistic_fitted, logistic_selection_trace = fit_reason_code_logistic_models(
        values,
        fault_examples,
        fault_splits,
        logistic_config,
    )
    boosted_config = HourlyReasonCodeConfig(seed=int(args.seed))
    boosted_fitted, boosted_selection_trace = fit_reason_code_models(
        values,
        fault_examples,
        fault_splits,
        boosted_config,
    )
    rgfn_configs, arm2_source = _reason_code_arm2_configs(
        args.rgfn_arm2_metrics,
        int(args.rgfn_seed),
    )
    remaining_seconds = max(0.0, workflow_deadline - time.monotonic())
    if args.rgfn_timebox_minutes is not None:
        remaining_seconds = min(remaining_seconds, float(args.rgfn_timebox_minutes) * 60.0)
    if remaining_seconds <= 0.0:
        rgfn_fitted = {}
        rgfn_selection_trace = []
        rgfn_status = {
            scheme: {
                "split_scheme": scheme,
                "status": "not_started_global_timebox",
                "elapsed_seconds": float(time.monotonic() - workflow_started),
                "epochs_completed": 0,
                "best_epoch": 0,
                "best_validation_loss": None,
                "test_evaluated": False,
            }
            for scheme in ("random", "spaced")
        }
    else:
        rgfn_fitted, rgfn_selection_trace, rgfn_status = fit_reason_code_rgfn_models(
            fault_examples,
            fault_splits,
            rgfn_configs,
            timebox_seconds=remaining_seconds,
            cv_seed=int(args.seed),
        )
    all_selection_trace = [
        *logistic_selection_trace,
        *boosted_selection_trace,
        *rgfn_selection_trace,
    ]
    if any(bool(row["test_metrics_read_during_selection"]) for row in all_selection_trace):
        raise RuntimeError("a reason-code method read held-out test metrics during selection")
    logistic_evaluations, logistic_metric_rows = evaluate_reason_code_models(
        logistic_fitted,
        values,
        fault_examples,
        fault_splits,
    )
    boosted_evaluations, boosted_metric_rows = evaluate_reason_code_models(
        boosted_fitted,
        values,
        fault_examples,
        fault_splits,
    )
    rgfn_evaluations, rgfn_metric_rows = evaluate_reason_code_rgfn_models(
        rgfn_fitted,
        fault_examples,
        fault_splits,
    )
    method_evaluations = {
        "logistic": logistic_evaluations,
        "gradient_boosted": boosted_evaluations,
        "rgfn": rgfn_evaluations,
    }
    method_metric_rows = {
        "logistic": logistic_metric_rows,
        "gradient_boosted": boosted_metric_rows,
        "rgfn": rgfn_metric_rows,
    }
    comparison_rows = reason_code_comparison_rows(
        method_evaluations,
        {
            "logistic": {"random": "completed", "spaced": "completed"},
            "gradient_boosted": {"random": "completed", "spaced": "completed"},
            "rgfn": {
                scheme: str(detail["status"])
                for scheme, detail in rgfn_status.items()
            },
        },
    )
    threshold_robustness = reason_code_threshold_robustness_rows(
        method_evaluations,
        method_metric_rows,
        all_selection_trace,
    )
    feature_frame = pd.read_parquet(args.features)
    predictions = reason_code_prediction_frame(
        fault_examples,
        boosted_evaluations,
        feature_frame,
    )
    expected_prediction_rows = sum(
        len(fault_splits[scheme]["test"])
        for scheme in ("random", "spaced")
    )
    if len(predictions) != expected_prediction_rows:
        raise RuntimeError("reason-code localisation output does not contain exactly the held-out fault hours")
    if not predictions["tensor_detector_groups"].eq(predictions["detector_groups_fired"]).all():
        raise RuntimeError("localisation detector-group evidence disagrees with the hourly tensor")
    comparison_prediction_frames: list[pd.DataFrame] = []
    for method, evaluations in method_evaluations.items():
        if not evaluations:
            continue
        frame = reason_code_prediction_frame(fault_examples, evaluations, feature_frame)
        frame.insert(0, "method", method)
        if not frame["tensor_detector_groups"].eq(frame["detector_groups_fired"]).all():
            raise RuntimeError(f"{method} localisation evidence disagrees with the hourly tensor")
        comparison_prediction_frames.append(frame)
    comparison_predictions = (
        pd.concat(comparison_prediction_frames, ignore_index=True)
        .sort_values(["method", "split_scheme", "station_id", "hour"])
        .reset_index(drop=True)
        if comparison_prediction_frames
        else pd.DataFrame()
    )
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models" / "reason_codes"
    output_dir.mkdir(parents=True, exist_ok=True)
    boosted_model_paths: dict[str, str] = {}
    comparison_model_paths: dict[str, str] = {}
    for item in boosted_fitted:
        destination = model_dir / "gradient_boosted" / item.split_scheme / f"{item.axis}_{item.label_name}.joblib"
        saved = save_reason_code_model(
            item,
            destination,
            feature_names,
            int(fault_examples["X_cont"].shape[1]),
            boosted_config,
            args.manifest,
        )
        boosted_model_paths[f"{item.split_scheme}:{item.axis}:{item.label_name}"] = str(saved)
        comparison_model_paths[f"gradient_boosted:{item.split_scheme}:{item.axis}:{item.label_name}"] = str(saved)
    for item in logistic_fitted:
        destination = model_dir / "logistic" / item.split_scheme / f"{item.axis}_{item.label_name}.joblib"
        saved = save_reason_code_model(
            item,
            destination,
            feature_names,
            int(fault_examples["X_cont"].shape[1]),
            logistic_config,
            args.manifest,
        )
        comparison_model_paths[f"logistic:{item.split_scheme}:{item.axis}:{item.label_name}"] = str(saved)
    for scheme, item in rgfn_fitted.items():
        saved = save_reason_code_rgfn_model(
            item,
            model_dir / "rgfn" / f"hourly_reason_codes_{scheme}.pt",
            args.manifest,
            arm2_source[scheme],
        )
        comparison_model_paths[f"rgfn:{scheme}"] = str(saved)
    boosted_summary_rows = reason_code_summary_rows(boosted_evaluations)
    boosted_report = reason_code_report(
        partition_trace,
        boosted_selection_trace,
        boosted_metric_rows,
        boosted_summary_rows,
    )
    comparison_report = reason_code_method_comparison_report(
        partition_trace,
        all_selection_trace,
        method_metric_rows,
        comparison_rows,
        rgfn_status,
        threshold_robustness,
    )
    metrics_path = output_dir / REASON_CODE_METRICS_PATH.name
    report_path = output_dir / REASON_CODE_REPORT_PATH.name
    predictions_path = output_dir / REASON_CODE_PREDICTIONS_PATH.name
    comparison_metrics_path = output_dir / REASON_CODE_COMPARISON_METRICS_PATH.name
    comparison_report_path = output_dir / REASON_CODE_COMPARISON_REPORT_PATH.name
    comparison_predictions_path = output_dir / REASON_CODE_COMPARISON_PREDICTIONS_PATH.name
    threshold_robustness_path = output_dir / REASON_CODE_THRESHOLD_ROBUSTNESS_PATH.name
    report_path.write_text(boosted_report + "\n", encoding="utf-8")
    comparison_report_path.write_text(comparison_report + "\n", encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)
    comparison_predictions.to_csv(comparison_predictions_path, index=False)
    pd.DataFrame(threshold_robustness).to_csv(threshold_robustness_path, index=False)
    binary_after = _binary_artifact_snapshot(args.short_tensor, args.manifest)
    if binary_before != binary_after:
        raise RuntimeError("reason-code training altered a frozen binary artifact")
    payload = {
        "mode": "reason_codes",
        "configuration": asdict(boosted_config),
        "tensor_path": str(args.short_tensor),
        "manifest_path": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "feature_matrix_path": str(args.features),
        "mask_mode": MASK_MODE_PER_HOUR,
        "feature_dimension": int(values.shape[1]),
        "fault_hours": int(len(fault_examples["y_binary"])),
        "excluded_examples_removed_before_reason_code_filter": int(excluded),
        "partition_trace": partition_trace,
        "selection_trace": boosted_selection_trace,
        "test_metrics_read_before_threshold_selection": False,
        "all_thresholds_selected_before_test_evaluation": True,
        "per_label_test_metrics": boosted_metric_rows,
        "macro_micro_test_metrics": boosted_summary_rows,
        "localisation_rows": int(len(predictions)),
        "localisation_detector_groups_match_tensor": True,
        "model_paths": boosted_model_paths,
        "binary_artifacts_before": binary_before,
        "binary_artifacts_after": binary_after,
        "binary_artifacts_unchanged": True,
    }
    write_metrics_json(payload, metrics_path)
    comparison_payload = {
        "mode": "reason_code_method_comparison",
        "tensor_path": str(args.short_tensor),
        "manifest_path": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "mask_mode": MASK_MODE_PER_HOUR,
        "feature_dimension": int(values.shape[1]),
        "fault_hours": int(len(fault_examples["y_binary"])),
        "excluded_examples_removed_before_reason_code_filter": int(excluded),
        "partition_trace": partition_trace,
        "configurations": {
            "logistic": asdict(logistic_config),
            "gradient_boosted": asdict(boosted_config),
            "rgfn": {scheme: asdict(config) for scheme, config in rgfn_configs.items()},
        },
        "rgfn_arm2_source_configurations": arm2_source,
        "historical_binary_rgfn_weights_and_thresholds_not_reused": True,
        "reason_code_workflow_timebox_seconds": float(args.reason_code_timebox_minutes) * 60.0,
        "rgfn_timebox_seconds": float(remaining_seconds),
        "workflow_elapsed_seconds": float(time.monotonic() - workflow_started),
        "selection_trace": all_selection_trace,
        "test_metrics_read_before_final_evaluation": False,
        "all_completed_thresholds_selected_before_test_evaluation": True,
        "per_label_test_metrics": method_metric_rows,
        "comparison_table": comparison_rows,
        "threshold_robustness": threshold_robustness,
        "rgfn_status": rgfn_status,
        "comparison_localisation_rows": int(len(comparison_predictions)),
        "comparison_localisation_detector_groups_match_tensor": True,
        "model_paths": comparison_model_paths,
        "binary_artifacts_before": binary_before,
        "binary_artifacts_after": binary_after,
        "binary_artifacts_unchanged": True,
    }
    write_metrics_json(comparison_payload, comparison_metrics_path)
    print(comparison_report)
    print()
    print("OUTPUTS")
    print(f"metrics_json={metrics_path}")
    print(f"report={report_path}")
    print(f"localisation_predictions={predictions_path}")
    print(f"comparison_metrics_json={comparison_metrics_path}")
    print(f"comparison_report={comparison_report_path}")
    print(f"comparison_localisation_predictions={comparison_predictions_path}")
    print(f"threshold_robustness={threshold_robustness_path}")
    print(f"models={model_dir}")
    print(f"binary_artifacts_unchanged={binary_before == binary_after}")


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
