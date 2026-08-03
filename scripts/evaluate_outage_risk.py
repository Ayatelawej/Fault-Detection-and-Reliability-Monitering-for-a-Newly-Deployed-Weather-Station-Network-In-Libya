from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.availability.risk_dataset import (
    CAUSAL_FORECAST_FEATURE_COLUMNS,
    FAULT_WINDOW_POLICY_DESCRIPTION,
    HORIZONS,
    ONSET_RECOVERY_EXCLUSION_HOURS,
    build_backward_event_history_features,
    build_causal_forecast_features,
    build_confirmed_incident_recurrence_history,
    build_discrete_hazard_features,
    discrete_hazard_causal_source_columns,
    build_event_history_feature_audit,
    build_fault_risk_dataset,
    build_incident_hazard_dataset,
    build_retrospective_persistence_history,
    build_risk_dataset,
    event_history_feature_columns,
    load_causal_forecast_sources,
    summarize_delete_future_validation,
    summarize_fault_risk_construction,
    summarize_fault_station_support,
    validate_delete_future_event_history_features,
    validate_delete_future_features,
)
from src.availability.risk_eval import (
    build_label_split_characteristics,
    evaluate_horizon,
    event_recall,
    summarize_station_positive_support,
)
from src.availability.risk_model import (
    DISCRETE_HAZARD_METHODS,
    FORECAST_MODEL_NAME,
    FORECAST_THRESHOLDS,
    ForecastTrainingTimeboxReached,
    calibrate_hazard_probability,
    cumulate_stationary_hazard,
    discrete_hazard_probability,
    fit_discrete_hazard_model,
    fit_hazard_probability_calibrator,
    fit_forecast_hist_gradient_boosting,
    forecast_base_rate_prediction,
    forecast_classification_metrics,
    forecast_model_probability,
    forecast_persistence_prediction,
    forecast_recurrence_prediction,
    select_discrete_hazard_threshold,
    select_validation_threshold_rule,
    validation_permutation_importance,
)
from src.config.paths import (
    AVAILABILITY_CLASSIFICATION_PATH,
    AVAILABILITY_EVENTS_PATH,
    MERGED_DATASET_PATH,
    NETWORK_OUTAGE_WINDOWS_PATH,
    PARTIAL_OUTAGE_EVENTS_PATH,
    STATION_REGISTRY_PATH,
)
from src.rules.config import EXTERNAL_CACHE_DIR
from src.model.hourly_detection import (
    FEATURE_PATH as HOURLY_FEATURE_PATH,
    HOURLY_LABEL_PATH,
    HOURLY_DATA_DIR,
    LABEL_PATH as EPISODE_LABEL_PATH,
    LONG_TENSOR_PATH,
    SHORT_TENSOR_PATH,
    SOURCE_PATH as HOURLY_SOURCE_PATH,
)
from src.workflows.prerequisites import require_files

OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"
HOUR_METRICS_PATH = OUTPUT_DIR / "outage_risk_hour_metrics.csv"
EVENT_METRICS_PATH = OUTPUT_DIR / "outage_risk_event_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "outage_risk_predictions.parquet"
LABEL_SPLIT_SUMMARY_PATH = OUTPUT_DIR / "outage_risk_label_split_summary.csv"
LABEL_SPLIT_PURGE_PATH = OUTPUT_DIR / "outage_risk_label_split_purge.csv"
LABEL_CHANGE_PATH = OUTPUT_DIR / "outage_risk_label_changes.csv"
LABEL_SPLIT_REPORT_PATH = OUTPUT_DIR / "outage_risk_label_split_report.txt"
FAULT_LABEL_SPLIT_SUMMARY_PATH = OUTPUT_DIR / "fault_risk_label_split_summary.csv"
FAULT_LABEL_SPLIT_PURGE_PATH = OUTPUT_DIR / "fault_risk_label_split_purge.csv"
FAULT_CONSTRUCTION_PATH = OUTPUT_DIR / "fault_risk_label_construction.csv"
FAULT_COMPARISON_PATH = OUTPUT_DIR / "fault_risk_vs_outage_positive_rates.csv"
FAULT_STATION_SUPPORT_PATH = OUTPUT_DIR / "fault_risk_station_positive_support.csv"
FAULT_DIRECT_STATION_SUPPORT_PATH = (
    OUTPUT_DIR / "fault_risk_station_direct_fault_support.csv"
)
FAULT_INVARIANT_HASHES_PATH = OUTPUT_DIR / "fault_risk_invariant_hashes.json"
FAULT_LABEL_SPLIT_REPORT_PATH = OUTPUT_DIR / "fault_risk_label_split_report.txt"
PREVIOUS_FORECAST_OUTPUT_DIR = OUTPUT_DIR / "forecast_risk_corrected_scope"
FORECAST_OUTPUT_DIR = OUTPUT_DIR / "forecast_risk_event_history"
ONSET_FORECAST_OUTPUT_DIR = OUTPUT_DIR / "forecast_risk_onset"
DISCRETE_HAZARD_OUTPUT_DIR = OUTPUT_DIR / "forecast_risk_discrete_hazard"
THRESHOLD_RESELECTION_OUTPUT_DIR = OUTPUT_DIR / "forecast_risk_threshold_reselection"
THRESHOLD_RESELECTION_SOURCE_COMMIT = "954e8e0"
THRESHOLD_RESELECTION_PRECISION_FLOOR = 0.55
FORECAST_METRICS_PATH = FORECAST_OUTPUT_DIR / "forecast_risk_metrics.csv"
FORECAST_CONFUSION_PATH = FORECAST_OUTPUT_DIR / "forecast_risk_confusion_matrices.csv"
FORECAST_SELECTION_PATH = FORECAST_OUTPUT_DIR / "forecast_risk_selection_trace.csv"
FORECAST_AUDIT_PATH = FORECAST_OUTPUT_DIR / "forecast_risk_causality_audit.csv"
FORECAST_FEATURE_SET_PATH = FORECAST_OUTPUT_DIR / "forecast_risk_feature_set.csv"
FORECAST_IMPORTANCE_PATH = FORECAST_OUTPUT_DIR / "forecast_risk_feature_importance.csv"
FORECAST_PREDICTIONS_PATH = FORECAST_OUTPUT_DIR / "forecast_risk_predictions.parquet"
FORECAST_DIGESTS_PATH = FORECAST_OUTPUT_DIR / "forecast_risk_split_digests.json"
FORECAST_INVARIANTS_PATH = FORECAST_OUTPUT_DIR / "forecast_risk_invariant_hashes.json"
FORECAST_REPORT_PATH = FORECAST_OUTPUT_DIR / "forecast_risk_report.txt"
FORECAST_MODELS_DIR = FORECAST_OUTPUT_DIR / "models"
FORECAST_FUTURE_VALIDATION_PATH = FORECAST_OUTPUT_DIR / "forecast_risk_delete_future_validation.csv"
FORECAST_FUTURE_VALIDATION_SUMMARY_PATH = (
    FORECAST_OUTPUT_DIR / "forecast_risk_delete_future_summary.csv"
)
FORECAST_PREVIOUS_COMPARISON_PATH = (
    FORECAST_OUTPUT_DIR / "previous_1014_vs_event_history.csv"
)
FORECAST_FEATURE_COUNTS_PATH = FORECAST_OUTPUT_DIR / "forecast_risk_feature_counts.csv"
FORECAST_DEPLOYMENT_VARIANT_PATH = (
    FORECAST_OUTPUT_DIR / "forecast_risk_deployment_realism.csv"
)
FORECAST_PERSISTENCE_COMPARISON_PATH = (
    FORECAST_OUTPUT_DIR / "forecast_risk_model_vs_persistence.csv"
)
THRESHOLD_RESELECTION_METRICS_NAME = "forecast_risk_threshold_reselection_metrics.csv"
THRESHOLD_RESELECTION_TRADEOFF_NAME = "forecast_risk_validation_threshold_tradeoff.csv"
THRESHOLD_RESELECTION_PREDICTIONS_NAME = "forecast_risk_threshold_reselection_predictions.parquet"
THRESHOLD_RESELECTION_BASELINES_NAME = "forecast_risk_threshold_reselection_baselines.csv"
THRESHOLD_RESELECTION_SOURCE_LOCK_NAME = "forecast_risk_threshold_reselection_source_lock.json"
THRESHOLD_RESELECTION_REPORT_NAME = "forecast_risk_threshold_reselection_report.txt"
FORECAST_TIMEBOX_SECONDS = 60 * 60
ONSET_FORECAST_TIMEBOX_SECONDS = 90 * 60
DISCRETE_HAZARD_TIMEBOX_SECONDS = 90 * 60

def evaluate_all_with_predictions(events_path: Path = AVAILABILITY_EVENTS_PATH) -> dict[str, pd.DataFrame]:
    dataset = build_risk_dataset()
    events = pd.read_parquet(events_path)
    metric_frames = []
    event_rows = []
    prediction_rows = []
    for horizon in HORIZONS:
        horizon_frame = dataset.for_horizon(horizon)
        result = evaluate_horizon(horizon_frame, horizon)
        metric_frames.append(result["metrics"])
        metadata = result["split"]["metadata"]
        if not isinstance(metadata, dict):
            raise TypeError("risk split metadata must be a dictionary")
        for model_name, prediction in result["predictions"].items():
            row = event_recall(
                result["split"]["test"],
                events,
                horizon,
                prediction.pred,
                test_start_utc=metadata["test_start_utc"],
            )
            event_rows.append({"horizon_h": horizon, "model": model_name, **row})
            frame = result["split"]["test"].loc[:, ["station_id", "hour_utc", "y"]].copy()
            frame["horizon_h"] = int(horizon)
            frame["model"] = model_name
            frame["prob"] = prediction.prob.astype(float)
            frame["pred"] = prediction.pred.astype(int)
            frame["threshold"] = float(prediction.threshold)
            frame["split"] = "test"
            prediction_rows.append(frame)
    return {
        "hour_metrics": pd.concat(metric_frames, ignore_index=True),
        "event_metrics": pd.DataFrame(event_rows),
        "predictions": pd.concat(prediction_rows, ignore_index=True),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build corrected continuous-hour outage or fault-risk labels and "
            "purged timestamp splits, or run the causal six-configuration "
            "forecast-risk experiment."
        )
    )
    parser.add_argument(
        "--target",
        choices=("outage", "fault", "all"),
        default="outage",
    )
    parser.add_argument("--events", type=Path, default=AVAILABILITY_EVENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-models",
        action="store_true",
        help=(
            "Explicitly fit and evaluate provisional outage-risk models after "
            "writing label/split diagnostics. This is disabled by default."
        ),
    )
    parser.add_argument(
        "--train-risk-models",
        action="store_true",
        help=(
            "Train the six causal HistGradientBoosting forecast-risk models "
            "for --target all. Selection uses training/validation only."
        ),
    )
    parser.add_argument(
        "--train-discrete-hazard",
        action="store_true",
        help=(
            "Fit the logistic and shallow-boosted discrete-time incident-hazard "
            "comparison for --target all. Selection uses validation only."
        ),
    )
    parser.add_argument(
        "--reselect-forecast-thresholds",
        action="store_true",
        help=(
            "Reuse the frozen continuation-risk validation trace and test "
            "probabilities to compare validation-selected threshold rules without "
            "training a model."
        ),
    )
    parser.add_argument(
        "--risk-definition",
        choices=("continuation", "onset"),
        default="continuation",
        help=(
            "Use the existing future-event-presence target or the clean-state, "
            "sustained-incident onset target."
        ),
    )
    return parser.parse_args(argv)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_hour_reference_cache() -> Path:
    reference_dir = PROJECT_ROOT / EXTERNAL_CACHE_DIR
    if not reference_dir.is_dir() or not any(reference_dir.glob("*.parquet")):
        raise FileNotFoundError(
            "Causal fault- and outage-risk model training requires exact-hour "
            f"ERA5 reference parquet files under {reference_dir}"
        )
    return reference_dir


def _full_outage_duration_summary(events: pd.DataFrame) -> dict[str, object]:
    required = ["event_id", "start_utc", "end_utc", "duration_hours"]
    missing = sorted(set(required).difference(events.columns))
    if missing:
        raise KeyError(missing)
    durations = pd.to_numeric(events["duration_hours"], errors="coerce")
    if durations.isna().any():
        raise ValueError("availability events contain invalid duration_hours")
    fingerprint = events.loc[:, required].copy()
    fingerprint["start_utc"] = pd.to_datetime(
        fingerprint["start_utc"], utc=True, errors="coerce"
    )
    fingerprint["end_utc"] = pd.to_datetime(
        fingerprint["end_utc"], utc=True, errors="coerce"
    )
    if fingerprint[["start_utc", "end_utc"]].isna().any().any():
        raise ValueError("availability events contain invalid event timestamps")
    fingerprint = fingerprint.sort_values("event_id", kind="mergesort")
    payload = fingerprint.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return {
        "total_full_outage_hours": float(durations.sum()),
        "mean_full_outage_duration_hours": float(durations.mean()),
        "median_full_outage_duration_hours": float(durations.median()),
        "max_full_outage_duration_hours": float(durations.max()),
        "duration_series_sha256": sha256(payload).hexdigest(),
    }


def _availability_invariant_summary(events_path: Path) -> dict[str, object]:
    events = pd.read_parquet(events_path)
    windows = pd.read_csv(NETWORK_OUTAGE_WINDOWS_PATH)
    partial_events = pd.read_parquet(PARTIAL_OUTAGE_EVENTS_PATH)
    event_count = int(len(events))
    window_count = int(len(windows))
    partial_event_count = int(len(partial_events))
    return {
        "full_outage_events": event_count,
        "network_outage_windows": window_count,
        "partial_outage_events": partial_event_count,
        "full_outage_duration_summary": _full_outage_duration_summary(events),
        "artifact_sha256": {
            "hourly_availability_classification": _file_sha256(
                AVAILABILITY_CLASSIFICATION_PATH
            ),
            "availability_events": _file_sha256(events_path),
            "network_outage_windows": _file_sha256(NETWORK_OUTAGE_WINDOWS_PATH),
            "partial_outage_events": _file_sha256(PARTIAL_OUTAGE_EVENTS_PATH),
        },
    }


def _hourly_fault_artifact_hashes() -> dict[str, str]:
    paths = {
        "live_episode_labels": EPISODE_LABEL_PATH,
        "canonical_hourly_source": HOURLY_SOURCE_PATH,
        "feature_matrix": HOURLY_FEATURE_PATH,
        "hourly_label_export": HOURLY_LABEL_PATH,
        "short_hourly_tensor": SHORT_TENSOR_PATH,
        "long_hourly_tensor": LONG_TENSOR_PATH,
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def _logical_risk_label_hashes(dataset) -> dict[str, str]:
    hashes = {}
    for horizon in dataset.horizons:
        frame = dataset.for_horizon(horizon).loc[
            :, ["station_id", "hour_utc", "label_end_utc", "y"]
        ]
        payload = (
            frame.sort_values(["station_id", "hour_utc"], kind="mergesort")
            .to_csv(index=False, lineterminator="\n")
            .encode("utf-8")
        )
        hashes[f"{int(horizon)}h"] = sha256(payload).hexdigest()
    return hashes


def _format_frame(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False) if not frame.empty else "(none)"


def _format_duration_summary(summary: object) -> str:
    if not isinstance(summary, dict):
        raise TypeError("full outage duration summary must be a dictionary")
    return (
        "full_outage_duration_hours="
        f"total={float(summary['total_full_outage_hours']):.0f}, "
        f"mean={float(summary['mean_full_outage_duration_hours']):.2f}, "
        f"median={float(summary['median_full_outage_duration_hours']):.2f}, "
        f"max={float(summary['max_full_outage_duration_hours']):.0f}"
    )


def _label_split_report(
    result: dict[str, object],
    invariant_summary: dict[str, object],
) -> str:
    partition_summary = result["partition_summary"]
    purge_summary = result["purge_summary"]
    label_changes = result["label_changes"]
    structural_gaps = result["structural_gaps"]
    if not all(
        isinstance(frame, pd.DataFrame)
        for frame in [partition_summary, purge_summary, label_changes, structural_gaps]
    ):
        raise TypeError("label/split characteristics must be DataFrames")
    imbalance_lines = []
    for row in partition_summary.itertuples(index=False):
        if int(row.n_positive) == 0:
            ratio = "undefined (no positive examples)"
        else:
            ratio = f"{float(row.negative_to_positive_ratio):.2f}:1"
        imbalance_lines.append(
            f"H={int(row.horizon_h)}h, {row.partition}: "
            f"not-outage : outage = {int(row.n_negative)} : {int(row.n_positive)} "
            f"({ratio})"
        )
    total_materialized_gap_hours = int(
        structural_gaps["materialized_gap_hours"].sum()
    )
    no_label_changes = bool(label_changes["changed_label_rows"].eq(0).all())
    lines = [
        "OUTAGE-RISK LABEL AND SPLIT CHARACTERISTICS",
        "",
        "No outage-risk model was fitted by this run.",
        "Labels use continuous station-hour grids; the horizon interval is strictly after",
        "the scored hour through the next H clock-hours. Timestamp partitions are purged",
        "for the relevant horizon before each later partition.",
        "",
        "FROZEN AVAILABILITY INVARIANTS",
        f"full_outage_events={invariant_summary['full_outage_events']}",
        f"network_outage_windows={invariant_summary['network_outage_windows']}",
        f"partial_outage_events={invariant_summary['partial_outage_events']}",
        "full_outage_duration_series_unchanged=true",
        _format_duration_summary(invariant_summary["full_outage_duration_summary"]),
        "availability_artifact_hashes_unchanged=true",
        "",
        "MATERIALISED STRUCTURAL GAPS",
        (
            f"stations_with_structural_gaps={len(structural_gaps)}; "
            f"materialized_gap_hours={total_materialized_gap_hours}"
        ),
        (
            "Elapsed gap span includes the two observed endpoints; materialized "
            "gap hours are the omitted station-hours classified as full outages."
        ),
        _format_frame(structural_gaps),
        "",
        "LABEL CHANGES: CONTINUOUS HOURS VS LEGACY ROW OFFSETS",
        _format_frame(label_changes),
        *(
            [
                "Current-snapshot note: zero aligned labels changed because every "
                "structural gap follows an already observed full-outage hour. This "
                "does not make row-offset arithmetic valid for future data."
            ]
            if no_label_changes
            else []
        ),
        "",
        "PARTITION CHARACTERISTICS AFTER HORIZON-SPECIFIC PURGE",
        _format_frame(partition_summary),
        "",
        "CLASS IMBALANCE (NOT-OUTAGE : OUTAGE)",
        *imbalance_lines,
        "",
        "PURGE REMOVALS",
        _format_frame(purge_summary),
    ]
    return "\n".join(lines) + "\n"


def build_label_split_report(events_path: Path = AVAILABILITY_EVENTS_PATH) -> dict[str, object]:
    before = _availability_invariant_summary(events_path)
    dataset = build_risk_dataset()
    result = build_label_split_characteristics(dataset)
    after = _availability_invariant_summary(events_path)
    if before != after:
        raise RuntimeError("outage-risk label construction changed availability artifacts")
    structural_gaps = (
        dataset.frame.loc[
            dataset.frame["is_materialized_gap"],
            ["station_id", "hour_utc"],
        ]
        .groupby("station_id", as_index=False)
        .agg(
            first_hour_utc=("hour_utc", "min"),
            last_hour_utc=("hour_utc", "max"),
            materialized_gap_hours=("hour_utc", "size"),
        )
        .sort_values("station_id", kind="mergesort")
        .reset_index(drop=True)
    )
    structural_gaps["elapsed_gap_span_hours"] = (
        structural_gaps["materialized_gap_hours"] + 1
    )
    result["structural_gaps"] = structural_gaps
    result["availability_invariants"] = after
    return result


def write_label_split_outputs(
    result: dict[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "partition_summary": output_dir / LABEL_SPLIT_SUMMARY_PATH.name,
        "purge_summary": output_dir / LABEL_SPLIT_PURGE_PATH.name,
        "label_changes": output_dir / LABEL_CHANGE_PATH.name,
        "report": output_dir / LABEL_SPLIT_REPORT_PATH.name,
    }
    for result_key, path_key in [
        ("partition_summary", "partition_summary"),
        ("purge_summary", "purge_summary"),
        ("label_changes", "label_changes"),
    ]:
        frame = result[result_key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{result_key} must be a DataFrame")
        frame.to_csv(output_paths[path_key], index=False)
    invariants = result["availability_invariants"]
    if not isinstance(invariants, dict):
        raise TypeError("availability invariants must be a dictionary")
    output_paths["report"].write_text(
        _label_split_report(result, invariants),
        encoding="utf-8",
    )
    return output_paths


def _fault_imbalance_lines(partition_summary: pd.DataFrame) -> list[str]:
    lines = []
    for row in partition_summary.itertuples(index=False):
        ratio = (
            "undefined (no positive examples)"
            if int(row.n_positive) == 0
            else f"{float(row.negative_to_positive_ratio):.2f}:1"
        )
        lines.append(
            f"H={int(row.horizon_h)}h, {row.partition}: "
            f"not-fault : fault = {int(row.n_negative)} : {int(row.n_positive)} "
            f"({ratio})"
        )
    return lines


def _fault_label_split_report(result: dict[str, object]) -> str:
    partition_summary = result["partition_summary"]
    purge_summary = result["purge_summary"]
    construction = result["construction"]
    comparison = result["outage_comparison"]
    station_support = result["station_support"]
    direct_station_support = result["direct_station_support"]
    if not all(
        isinstance(frame, pd.DataFrame)
        for frame in [
            partition_summary,
            purge_summary,
            construction,
            comparison,
            station_support,
            direct_station_support,
        ]
    ):
        raise TypeError("fault-risk report inputs must be DataFrames")
    near_zero = station_support.loc[
        station_support["near_zero_positive_support"]
    ].sort_values(["horizon_h", "partition", "station_id"], kind="mergesort")
    near_zero_direct = direct_station_support.loc[
        direct_station_support["near_zero_direct_fault_support"]
    ]
    lines = [
        "FAULT-RISK LABEL AND SPLIT CHARACTERISTICS",
        "",
        "No fault-risk model was fitted by this run.",
        "Target: a confirmed labelled fault occurs strictly after the scored hour ",
        "through the next H continuous clock-hours on the same station.",
        "",
        "EXCLUDED-HOUR POLICY",
        FAULT_WINDOW_POLICY_DESCRIPTION,
        "Warmup hours remain on the clock and may be future fault occurrences, but ",
        "they are not scoreable current prediction rows.",
        "",
        "FROZEN-ARTIFACT INVARIANTS",
        "fault_labels_hourly_dataset_availability_and_outage_labels_unchanged=true",
        "Detailed before/after SHA-256 evidence is saved in "
        "fault_risk_invariant_hashes.json.",
        "",
        "CONTINUOUS-GRID CONSTRUCTION",
        _format_frame(construction),
        "",
        "PARTITION CHARACTERISTICS AFTER HORIZON-SPECIFIC PURGE",
        _format_frame(partition_summary),
        "",
        "CLASS IMBALANCE (NOT-FAULT : FAULT)",
        *_fault_imbalance_lines(partition_summary),
        "",
        "FAULT VS OUTAGE POSITIVE-RATE COMPARISON",
        _format_frame(comparison),
        "",
        "NEAR-ZERO DIRECT FAULT-HOUR SUPPORT (AT MOST FIVE FAULT HOURS)",
        _format_frame(near_zero_direct),
        "",
        "NEAR-ZERO STATION POSITIVE SUPPORT (AT MOST FIVE POSITIVES)",
        _format_frame(near_zero),
        "",
        "PURGE REMOVALS",
        _format_frame(purge_summary),
        "",
        "PARTITION INVARIANT",
        "all timestamps are assigned as whole groups, and every retained train or ",
        "validation label ends before the next partition boundary=true",
    ]
    return "\n".join(lines) + "\n"


def build_fault_label_split_report(
    events_path: Path = AVAILABILITY_EVENTS_PATH,
) -> dict[str, object]:
    before_hourly_artifacts = _hourly_fault_artifact_hashes()
    before_availability = _availability_invariant_summary(events_path)
    outage_before = build_risk_dataset()
    before_outage_labels = _logical_risk_label_hashes(outage_before)
    outage_characteristics = build_label_split_characteristics(outage_before)
    outage_metadata = {
        horizon: split["metadata"]
        for horizon, split in outage_characteristics["splits"].items()
    }

    fault_dataset = build_fault_risk_dataset()
    construction = summarize_fault_risk_construction(fault_dataset)
    fault_characteristics = build_label_split_characteristics(
        fault_dataset,
        boundary_metadata_by_horizon=outage_metadata,
        label_changes=construction,
    )
    station_support = summarize_station_positive_support(
        fault_characteristics["splits"]
    )
    direct_station_support = summarize_fault_station_support(fault_dataset)
    fault_partition_summary = fault_characteristics["partition_summary"].rename(
        columns={
            "n_rows": "fault_n_rows",
            "n_positive": "fault_n_positive",
            "positive_rate": "fault_positive_rate",
            "negative_to_positive_ratio": "fault_negative_to_positive_ratio",
        }
    )
    outage_partition_summary = outage_characteristics["partition_summary"].rename(
        columns={
            "n_rows": "outage_n_rows",
            "n_positive": "outage_n_positive",
            "positive_rate": "outage_positive_rate",
            "negative_to_positive_ratio": "outage_negative_to_positive_ratio",
        }
    )
    comparison = fault_partition_summary.loc[
        :, [
            "horizon_h",
            "partition",
            "fault_n_rows",
            "fault_n_positive",
            "fault_positive_rate",
            "fault_negative_to_positive_ratio",
        ]
    ].merge(
        outage_partition_summary.loc[
            :, [
                "horizon_h",
                "partition",
                "outage_n_rows",
                "outage_n_positive",
                "outage_positive_rate",
                "outage_negative_to_positive_ratio",
            ]
        ],
        on=["horizon_h", "partition"],
        how="inner",
        validate="one_to_one",
    )
    comparison["fault_minus_outage_rate"] = (
        comparison["fault_positive_rate"] - comparison["outage_positive_rate"]
    )

    after_hourly_artifacts = _hourly_fault_artifact_hashes()
    after_availability = _availability_invariant_summary(events_path)
    outage_after = build_risk_dataset()
    after_outage_labels = _logical_risk_label_hashes(outage_after)
    if before_hourly_artifacts != after_hourly_artifacts:
        raise RuntimeError("fault-risk construction changed a fault-label or hourly artifact")
    if before_availability != after_availability:
        raise RuntimeError("fault-risk construction changed an availability artifact")
    if before_outage_labels != after_outage_labels:
        raise RuntimeError("fault-risk construction changed outage-risk labels")
    return {
        "partition_summary": fault_characteristics["partition_summary"],
        "purge_summary": fault_characteristics["purge_summary"],
        "construction": construction,
        "outage_comparison": comparison,
        "station_support": station_support,
        "direct_station_support": direct_station_support,
        "availability_invariants": after_availability,
        "hourly_artifact_hashes": after_hourly_artifacts,
        "outage_label_hashes": after_outage_labels,
        "invariant_hashes": {
            "before": {
                "hourly_artifacts": before_hourly_artifacts,
                "availability_artifacts": before_availability["artifact_sha256"],
                "outage_logical_labels": before_outage_labels,
            },
            "after": {
                "hourly_artifacts": after_hourly_artifacts,
                "availability_artifacts": after_availability["artifact_sha256"],
                "outage_logical_labels": after_outage_labels,
            },
        },
    }


def write_fault_label_split_outputs(
    result: dict[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "partition_summary": output_dir / FAULT_LABEL_SPLIT_SUMMARY_PATH.name,
        "purge_summary": output_dir / FAULT_LABEL_SPLIT_PURGE_PATH.name,
        "construction": output_dir / FAULT_CONSTRUCTION_PATH.name,
        "outage_comparison": output_dir / FAULT_COMPARISON_PATH.name,
        "station_support": output_dir / FAULT_STATION_SUPPORT_PATH.name,
        "direct_station_support": output_dir / FAULT_DIRECT_STATION_SUPPORT_PATH.name,
        "invariant_hashes": output_dir / FAULT_INVARIANT_HASHES_PATH.name,
        "report": output_dir / FAULT_LABEL_SPLIT_REPORT_PATH.name,
    }
    for result_key in [
        "partition_summary",
        "purge_summary",
        "construction",
        "outage_comparison",
        "station_support",
        "direct_station_support",
    ]:
        frame = result[result_key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{result_key} must be a DataFrame")
        frame.to_csv(output_paths[result_key], index=False)
    invariant_hashes = result["invariant_hashes"]
    if not isinstance(invariant_hashes, dict):
        raise TypeError("fault-risk invariant hashes must be a dictionary")
    output_paths["invariant_hashes"].write_text(
        json.dumps(invariant_hashes, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_paths["report"].write_text(
        _fault_label_split_report(result),
        encoding="utf-8",
    )
    return output_paths


def _directory_sha256(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        str(candidate.relative_to(path)).replace("\\", "/"): _file_sha256(candidate)
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
    }


def _risk_split_digest(split: dict[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for partition in ["train", "validation", "test", "purged"]:
        frame = split[partition]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{partition} split must be a DataFrame")
        columns = ["station_id", "hour_utc", "label_end_utc", "y"]
        payload = (
            frame.loc[:, columns]
            .sort_values(["station_id", "hour_utc"], kind="mergesort")
            .to_csv(index=False, lineterminator="\n")
            .encode("utf-8")
        )
        result[partition] = sha256(payload).hexdigest()
    return result


def _attach_forecast_features(
    split: dict[str, object],
    feature_frame: pd.DataFrame,
    event_history_frame: pd.DataFrame,
    persistence_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    event_history_columns: tuple[str, ...],
    horizon: int,
    recurrence_frame: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    feature_lookup = feature_frame.loc[:, ["station_id", "hour_utc", *feature_columns]]
    event_history_lookup = event_history_frame.loc[:, [
        "station_id",
        "hour_utc",
        *event_history_columns,
    ]]
    persistence_column = f"persistence_event_count_{int(horizon)}h"
    persistence_lookup = persistence_frame.loc[:, [
        "station_id",
        "hour_utc",
        persistence_column,
    ]]
    recurrence_lookup: pd.DataFrame | None = None
    if recurrence_frame is not None:
        recurrence_columns = [
            column
            for column in recurrence_frame.columns
            if column not in {"station_id", "hour_utc"}
        ]
        if len(recurrence_columns) != 1:
            raise ValueError("recurrence history must contain exactly one feature column")
        recurrence_lookup = recurrence_frame.loc[:, [
            "station_id",
            "hour_utc",
            recurrence_columns[0],
        ]]
    attached: dict[str, pd.DataFrame] = {}
    for partition in ["train", "validation", "test", "purged"]:
        frame = split[partition]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{partition} split must be a DataFrame")
        joined = frame.merge(
            feature_lookup,
            on=["station_id", "hour_utc"],
            how="left",
            validate="one_to_one",
            indicator="_feature_merge",
        ).merge(
            event_history_lookup,
            on=["station_id", "hour_utc"],
            how="left",
            validate="one_to_one",
            indicator="_event_history_merge",
        ).merge(
            persistence_lookup,
            on=["station_id", "hour_utc"],
            how="left",
            validate="one_to_one",
            indicator="_persistence_merge",
        )
        if recurrence_lookup is not None:
            joined = joined.merge(
                recurrence_lookup,
                on=["station_id", "hour_utc"],
                how="left",
                validate="one_to_one",
                indicator="_recurrence_merge",
            )
        if len(joined) != len(frame):
            raise RuntimeError("forecast feature attachment changed split row count")
        if not joined["_feature_merge"].eq("both").all():
            raise RuntimeError("risk split has rows without causal feature values")
        if not joined["_event_history_merge"].eq("both").all():
            raise RuntimeError("risk split has rows without event-history feature values")
        if not joined["_persistence_merge"].eq("both").all():
            raise RuntimeError("risk split has rows without persistence history")
        if recurrence_lookup is not None and not joined["_recurrence_merge"].eq("both").all():
            raise RuntimeError("risk split has rows without recurrence history")
        indicators = ["_feature_merge", "_event_history_merge", "_persistence_merge"]
        if recurrence_lookup is not None:
            indicators.append("_recurrence_merge")
        attached[partition] = joined.drop(
            columns=indicators
        ).reset_index(drop=True)
    return attached


def _forecast_invariant_snapshot(
    events_path: Path,
    outage_dataset,
    fault_dataset,
) -> dict[str, object]:
    return {
        "hourly_artifacts": _hourly_fault_artifact_hashes(),
        "availability_artifacts": _availability_invariant_summary(events_path)[
            "artifact_sha256"
        ],
        "outage_logical_labels": _logical_risk_label_hashes(outage_dataset),
        "fault_logical_labels": _logical_risk_label_hashes(fault_dataset),
        "detection_and_reason_code_artifacts": _directory_sha256(HOURLY_DATA_DIR),
        "causal_feature_source_artifacts": {
            "canonical_merged_dataset": _file_sha256(MERGED_DATASET_PATH),
            "station_registry": _file_sha256(STATION_REGISTRY_PATH),
            "exact_hour_reference_cache": _directory_sha256(
                PROJECT_ROOT / EXTERNAL_CACHE_DIR
            ),
        },
    }


def _build_forecast_target_splits() -> tuple[dict[str, object], dict[str, object]]:
    outage_dataset = build_risk_dataset()
    outage_characteristics = build_label_split_characteristics(outage_dataset)
    outage_boundaries = {
        horizon: split["metadata"]
        for horizon, split in outage_characteristics["splits"].items()
    }
    fault_dataset = build_fault_risk_dataset()
    fault_characteristics = build_label_split_characteristics(
        fault_dataset,
        boundary_metadata_by_horizon=outage_boundaries,
        label_changes=summarize_fault_risk_construction(fault_dataset),
    )
    return (
        {
            "dataset": outage_dataset,
            "characteristics": outage_characteristics,
            "event_column": "is_outage",
            "history_source": "outage",
        },
        {
            "dataset": fault_dataset,
            "characteristics": fault_characteristics,
            "event_column": "fault_target",
            "history_source": "fault",
        },
    )


def _build_onset_target_splits(
    continuation_outage_info: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    continuation_characteristics = continuation_outage_info["characteristics"]
    if not isinstance(continuation_characteristics, dict):
        raise TypeError("continuation outage characteristics must be a dictionary")
    continuation_splits = continuation_characteristics["splits"]
    if not isinstance(continuation_splits, dict):
        raise TypeError("continuation outage characteristics lack split definitions")
    boundaries = {
        int(horizon): split["metadata"]
        for horizon, split in continuation_splits.items()
        if isinstance(split, dict)
    }
    if set(boundaries) != {int(horizon) for horizon in HORIZONS}:
        raise RuntimeError("continuation outage splits lack all onset comparison boundaries")

    outage_dataset = build_incident_hazard_dataset(
        "outage",
        recovery_exclusion_hours=ONSET_RECOVERY_EXCLUSION_HOURS,
    )
    fault_dataset = build_incident_hazard_dataset(
        "fault",
        recovery_exclusion_hours=ONSET_RECOVERY_EXCLUSION_HOURS,
    )
    outage_characteristics = build_label_split_characteristics(
        outage_dataset,
        boundary_metadata_by_horizon=boundaries,
    )
    fault_characteristics = build_label_split_characteristics(
        fault_dataset,
        boundary_metadata_by_horizon=boundaries,
    )
    return (
        {
            "dataset": outage_dataset,
            "characteristics": outage_characteristics,
            "event_column": "is_outage",
            "history_source": "outage",
        },
        {
            "dataset": fault_dataset,
            "characteristics": fault_characteristics,
            "event_column": "fault_target",
            "history_source": "fault",
        },
    )


def _onset_risk_set_comparison(
    continuation_info: dict[str, dict[str, object]],
    onset_info: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for target in ["fault", "outage"]:
        continuation_characteristics = continuation_info[target]["characteristics"]
        onset_characteristics = onset_info[target]["characteristics"]
        if not isinstance(continuation_characteristics, dict) or not isinstance(
            onset_characteristics,
            dict,
        ):
            raise TypeError("risk-set characteristics must be dictionaries")
        continuation = continuation_characteristics["partition_summary"].copy()
        onset = onset_characteristics["partition_summary"].copy()
        if not isinstance(continuation, pd.DataFrame) or not isinstance(onset, pd.DataFrame):
            raise TypeError("risk-set partition summaries must be DataFrames")
        keys = ["horizon_h", "partition"]
        continuation = continuation.rename(
            columns={
                column: f"continuation_{column}"
                for column in continuation.columns
                if column not in keys
            }
        )
        onset = onset.rename(
            columns={
                column: f"onset_{column}"
                for column in onset.columns
                if column not in keys
            }
        )
        combined = continuation.merge(onset, on=keys, how="outer", validate="one_to_one")
        combined.insert(0, "target", target)
        for column in ["n_rows", "n_positive"]:
            combined[f"delta_{column}_onset_minus_continuation"] = (
                combined[f"onset_{column}"] - combined[f"continuation_{column}"]
            )
        combined["delta_positive_rate_onset_minus_continuation"] = (
            combined["onset_positive_rate"] - combined["continuation_positive_rate"]
        )
        rows.append(combined)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["target", "horizon_h", "partition"],
        kind="mergesort",
    ).reset_index(drop=True)


def _independent_onset_support(
    onset_info: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in ["fault", "outage"]:
        info = onset_info[target]
        dataset = info["dataset"]
        characteristics = info["characteristics"]
        if not hasattr(dataset, "frame") or not isinstance(characteristics, dict):
            raise TypeError("onset target construction returned an invalid dataset")
        splits = characteristics["splits"]
        if not isinstance(splits, dict):
            raise TypeError("onset characteristics lack split definitions")
        qualifying = dataset.frame["incident_qualifying_start"].astype(bool)
        total_qualifying = int(qualifying.sum())
        for horizon in HORIZONS:
            split = splits[int(horizon)]
            if not isinstance(split, dict):
                raise TypeError("onset split must be a dictionary")
            for partition in ["train", "validation", "test"]:
                frame = split[partition]
                if not isinstance(frame, pd.DataFrame):
                    raise TypeError(f"onset {partition} split must be a DataFrame")
                positive = frame.loc[frame["y"].eq(1)]
                event_ids = positive["future_event_id"].fillna("").astype(str)
                rows.append(
                    {
                        "target": target,
                        "horizon_h": int(horizon),
                        "partition": partition,
                        "score_rows": int(len(frame)),
                        "positive_score_rows": int(len(positive)),
                        "independent_future_incident_onsets": int(
                            event_ids.loc[event_ids.ne("")].nunique()
                        ),
                        "total_qualifying_incidents_global": total_qualifying,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["target", "horizon_h", "partition"],
        kind="mergesort",
    ).reset_index(drop=True)


def _onset_eligibility_audit(
    onset_info: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in ["fault", "outage"]:
        dataset = onset_info[target]["dataset"]
        if not hasattr(dataset, "frame"):
            raise TypeError("onset dataset lacks its continuous frame")
        frame = dataset.frame
        active = (
            frame["any_fault_target"].astype(bool)
            if target == "fault"
            else frame["is_outage"].astype(bool)
        )
        scoreable_while_active = int((frame["incident_scoreable"].astype(bool) & active).sum())
        scoreable_during_recovery = int(
            (
                frame["incident_scoreable"].astype(bool)
                & frame["incident_post_event_recovery_excluded"].astype(bool)
            ).sum()
        )
        if scoreable_while_active or scoreable_during_recovery:
            raise RuntimeError("onset risk set contains an active or recovery-excluded row")
        unobserved_qualifying_starts = int(
            (
                frame["incident_qualifying_start"].astype(bool)
                & ~frame["incident_observed_start"].astype(bool)
            ).sum()
        )
        if unobserved_qualifying_starts:
            raise RuntimeError("an unobservable incident start entered the onset target")
        for horizon in HORIZONS:
            eligible = frame[f"incident_eligible_{int(horizon)}h"].astype(bool)
            eligible_while_active = int((eligible & active).sum())
            if eligible_while_active:
                raise RuntimeError("an active incident entered an onset risk split")
            rows.append(
                {
                    "target": target,
                    "horizon_h": int(horizon),
                    "active_scoreable_rows": scoreable_while_active,
                    "recovery_excluded_scoreable_rows": scoreable_during_recovery,
                    "active_eligible_rows": eligible_while_active,
                    "unobserved_qualifying_starts": unobserved_qualifying_starts,
                    "recovery_exclusion_hours": int(dataset.recovery_exclusion_hours),
                    "passed": True,
                }
            )
    return pd.DataFrame(rows)


def _onset_network_event_policy(onset_outage_dataset) -> pd.DataFrame:
    frame = onset_outage_dataset.frame
    associated = frame["incident_network_associated"].astype(bool)
    sustained_associated = associated & frame["incident_duration_hours"].ge(
        int(onset_outage_dataset.minimum_duration_hours)
    )
    return pd.DataFrame(
        [
            {
                "target": "outage",
                "policy": "censor_network_associated_station_starts",
                "network_windows": int(pd.read_csv(NETWORK_OUTAGE_WINDOWS_PATH).shape[0]),
                "station_level_network_associated_starts": int(associated.sum()),
                "sustained_network_associated_starts": int(sustained_associated.sum()),
                "qualifying_non_network_station_starts": int(
                    frame["incident_qualifying_start"].astype(bool).sum()
                ),
                "interpretation": (
                    "Known network-associated starts are censored from the station-level "
                    "onset target; they are neither independent positives nor a separate "
                    "network-level prediction target."
                ),
            }
        ]
    )


def _attach_discrete_hazard_features(
    split: dict[str, object],
    *,
    one_step_labels: pd.DataFrame,
    hazard_features: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    _require = {"station_id", "hour_utc", "label_end_utc", "y"}
    if not _require.issubset(one_step_labels.columns):
        raise KeyError(sorted(_require.difference(one_step_labels.columns)))
    hazard_lookup = one_step_labels.loc[:, [
        "station_id",
        "hour_utc",
        "label_end_utc",
        "y",
    ]].rename(
        columns={"label_end_utc": "hazard_label_end_utc", "y": "hazard_y_1h"}
    )
    feature_lookup = hazard_features.loc[:, ["station_id", "hour_utc", *feature_columns]]
    attached: dict[str, pd.DataFrame] = {}
    for partition in ["train", "validation", "test", "purged"]:
        frame = split[partition]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{partition} split must be a DataFrame")
        joined = frame.merge(
            hazard_lookup,
            on=["station_id", "hour_utc"],
            how="left",
            validate="one_to_one",
            indicator="_hazard_label_merge",
        ).merge(
            feature_lookup,
            on=["station_id", "hour_utc"],
            how="left",
            validate="one_to_one",
            indicator="_hazard_feature_merge",
        )
        if len(joined) != len(frame):
            raise RuntimeError("discrete-hazard attachment changed a split row count")
        if not joined["_hazard_label_merge"].eq("both").all():
            raise RuntimeError("onset split has rows without one-hour hazard labels")
        if not joined["_hazard_feature_merge"].eq("both").all():
            raise RuntimeError("onset split has rows without compact hazard features")
        attached[partition] = joined.drop(
            columns=["_hazard_label_merge", "_hazard_feature_merge"]
        ).reset_index(drop=True)
    return attached


def _hazard_partition_digest(frame: pd.DataFrame, label_column: str) -> str:
    required = ["station_id", "hour_utc", label_column]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(missing)
    payload = (
        frame.loc[:, required]
        .sort_values(["station_id", "hour_utc"], kind="mergesort")
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    return sha256(payload).hexdigest()


def _verify_saved_onset_split_manifest(
    rebuilt: dict[str, dict[str, str]],
    *,
    manifest_path: Path = ONSET_FORECAST_OUTPUT_DIR / FORECAST_DIGESTS_PATH.name,
) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Discrete-hazard evaluation requires the saved onset split manifest: "
            f"{manifest_path}"
        )
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(expected, dict):
        raise ValueError("saved onset split manifest is not a JSON object")
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for key, actual_partitions in sorted(rebuilt.items()):
        expected_partitions = expected.get(key)
        if not isinstance(expected_partitions, dict):
            failures.append(f"missing {key}")
            continue
        for partition in ["train", "validation", "test", "purged"]:
            expected_digest = expected_partitions.get(partition)
            actual_digest = actual_partitions.get(partition)
            passed = isinstance(expected_digest, str) and expected_digest == actual_digest
            rows.append(
                {
                    "target_horizon": key,
                    "partition": partition,
                    "saved_digest": expected_digest,
                    "rebuilt_digest": actual_digest,
                    "passed": passed,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": _file_sha256(manifest_path),
                }
            )
            if not passed:
                failures.append(f"{key}:{partition}")
    if failures:
        raise RuntimeError(
            "rebuilt onset split membership differs from the saved manifest: "
            f"{failures}"
        )
    return pd.DataFrame(rows)


def _hazard_compact_feature_validation(
    audit: pd.DataFrame,
    causal_detail: pd.DataFrame,
    history_detail: pd.DataFrame,
) -> pd.DataFrame:
    causal_passed = causal_detail.groupby("feature", sort=False)["passed"].all().to_dict()
    history_passed = history_detail.groupby("feature", sort=False)["passed"].all().to_dict()
    rows: list[dict[str, object]] = []
    for row in audit.itertuples(index=False):
        feature = str(getattr(row, "feature"))
        if feature.startswith("history_"):
            source_features = [feature]
            passed = bool(history_passed.get(feature, False))
            method = "strictly_prior_history_delete_future_validation"
        elif feature.startswith("hazard_log1p_hours_since_last_"):
            source = str(getattr(row, "source"))
            source_features = [source]
            passed = bool(history_passed.get(source, False))
            method = "deterministic_transform_of_validated_history"
        elif feature.startswith("hazard_detector_count_24h_"):
            source_features = [
                value.strip() for value in str(getattr(row, "source")).split(";") if value.strip()
            ]
            passed = bool(source_features) and all(
                bool(causal_passed.get(source, False)) for source in source_features
            )
            method = "sum_of_validated_causal_detector_features"
        elif feature in {
            "hazard_day_of_week_sin",
            "hazard_day_of_week_cos",
            "hazard_month_sin",
            "hazard_month_cos",
            "hazard_station_age_hours",
        } or feature.startswith("hazard_station_indicator_"):
            source_features = [str(getattr(row, "source"))]
            passed = True
            method = "deterministic_timestamp_or_static_registry_feature"
        else:
            source_features = [feature]
            passed = bool(causal_passed.get(feature, False))
            method = "causal_measurement_delete_future_validation"
        rows.append(
            {
                "feature": feature,
                "source_features": "; ".join(source_features),
                "validation_method": method,
                "passed": passed,
            }
        )
    result = pd.DataFrame(rows).sort_values("feature", kind="mergesort").reset_index(drop=True)
    if result.empty or not result["passed"].all():
        failed = result.loc[~result["passed"], "feature"].tolist()
        raise RuntimeError(f"discrete-hazard compact feature future validation failed: {failed}")
    return result


def _hazard_calibration_rows(
    *,
    target: str,
    method: str,
    horizon: int,
    truth: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    values = pd.DataFrame(
        {
            "truth": np.asarray(truth, dtype=int),
            "probability": np.asarray(probability, dtype=float),
        }
    )
    if values.empty:
        return pd.DataFrame()
    values["decile"] = pd.qcut(
        values["probability"].rank(method="first"),
        q=min(10, len(values)),
        labels=False,
        duplicates="drop",
    ).astype(int) + 1
    brier = float(np.mean((values["probability"] - values["truth"]) ** 2))
    result = values.groupby("decile", sort=True).agg(
        n_rows=("truth", "size"),
        n_positive=("truth", "sum"),
        observed_onset_rate=("truth", "mean"),
        mean_predicted_horizon_risk=("probability", "mean"),
    ).reset_index()
    result.insert(0, "target", target)
    result.insert(1, "method", method)
    result.insert(2, "horizon_h", int(horizon))
    result["test_brier_score"] = brier
    return result


def _hazard_vs_direct_onset(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target",
        "horizon_h",
        "method",
        "hazard_test_precision",
        "direct_onset_test_precision",
        "delta_test_precision",
        "hazard_test_recall",
        "direct_onset_test_recall",
        "delta_test_recall",
        "hazard_test_f1",
        "direct_onset_test_f1",
        "delta_test_f1",
        "hazard_test_accuracy",
        "direct_onset_test_accuracy",
        "delta_test_accuracy",
        "direct_onset_metrics_sha256",
        "comparison_created_after_test_evaluation",
    ]
    direct_path = ONSET_FORECAST_OUTPUT_DIR / FORECAST_METRICS_PATH.name
    required = {
        "target",
        "horizon_h",
        "predictor",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_accuracy",
    }
    if metrics.empty or not required.issubset(metrics.columns) or not direct_path.exists():
        return pd.DataFrame(columns=columns)
    direct = pd.read_csv(direct_path)
    if not required.issubset(direct.columns):
        return pd.DataFrame(columns=columns)
    direct = direct.loc[
        direct["predictor"].eq(FORECAST_MODEL_NAME),
        ["target", "horizon_h", "test_precision", "test_recall", "test_f1", "test_accuracy"],
    ]
    hazard = metrics.loc[
        metrics["predictor"].isin(DISCRETE_HAZARD_METHODS),
        ["target", "horizon_h", "predictor", "test_precision", "test_recall", "test_f1", "test_accuracy"],
    ].rename(columns={"predictor": "method"})
    comparison = hazard.merge(
        direct,
        on=["target", "horizon_h"],
        how="inner",
        suffixes=("_hazard", "_direct"),
        validate="many_to_one",
    )
    result = comparison.loc[:, ["target", "horizon_h", "method"]].copy()
    for metric in ["precision", "recall", "f1", "accuracy"]:
        result[f"hazard_test_{metric}"] = comparison[f"test_{metric}_hazard"]
        result[f"direct_onset_test_{metric}"] = comparison[f"test_{metric}_direct"]
        result[f"delta_test_{metric}"] = (
            result[f"hazard_test_{metric}"] - result[f"direct_onset_test_{metric}"]
        )
    result["direct_onset_metrics_sha256"] = _file_sha256(direct_path)
    result["comparison_created_after_test_evaluation"] = True
    return result.loc[:, columns].sort_values(
        ["target", "horizon_h", "method"], kind="mergesort"
    ).reset_index(drop=True)


def _build_event_history_grid(
    outage_dataset,
    fault_dataset,
) -> pd.DataFrame:
    outage = outage_dataset.frame.loc[:, ["station_id", "hour_utc", "is_outage"]].copy()
    fault = fault_dataset.frame.loc[:, [
        "station_id",
        "hour_utc",
        "fault_target",
        "fault_scoreable",
    ]].copy()
    history = outage.merge(
        fault,
        on=["station_id", "hour_utc"],
        how="left",
        validate="one_to_one",
        indicator="_fault_history_merge",
    )
    if not history["_fault_history_merge"].eq("both").all():
        raise RuntimeError("continuous outage grid has rows without fault-history state")
    return history.drop(columns="_fault_history_merge").sort_values(
        ["station_id", "hour_utc"], kind="mergesort"
    ).reset_index(drop=True)


def _forecast_master_row(
    *,
    target: str,
    horizon: int,
    predictor: str,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    train_positive: int,
    validation_positive: int,
    test_positive: int,
    selected_positive_class_weight: float | None = None,
    selected_weight_multiplier: float | None = None,
    selected_threshold: float | None = None,
    test_evaluation_count: int = 1,
) -> dict[str, object]:
    row: dict[str, object] = {
        "target": target,
        "horizon_h": int(horizon),
        "predictor": predictor,
        "train_rows": int(train_rows),
        "validation_rows": int(validation_rows),
        "test_rows": int(test_rows),
        "train_positive": int(train_positive),
        "validation_positive": int(validation_positive),
        "test_positive": int(test_positive),
        "selected_positive_class_weight": selected_positive_class_weight,
        "selected_weight_multiplier": selected_weight_multiplier,
        "selected_threshold": selected_threshold,
        "test_evaluation_count": int(test_evaluation_count),
        "test_evaluated_once": int(test_evaluation_count) == 1,
    }
    for name in ["precision", "recall", "f1", "accuracy", "maximin_prf"]:
        row[f"validation_{name}"] = float(validation_metrics[name])
        row[f"test_{name}"] = float(test_metrics[name])
    row["test_all_precision_recall_f1_ge_080"] = bool(
        min(
            float(test_metrics["precision"]),
            float(test_metrics["recall"]),
            float(test_metrics["f1"]),
        )
        >= 0.80
    )
    return row


def _forecast_prediction_rows(
    frame: pd.DataFrame,
    *,
    target: str,
    horizon: int,
    predictor: str,
    probability: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    result = frame.loc[:, ["station_id", "hour_utc", "label_end_utc", "y"]].copy()
    result["target"] = target
    result["horizon_h"] = int(horizon)
    result["predictor"] = predictor
    result["probability"] = np.asarray(probability, dtype=float)
    result["threshold"] = float(threshold)
    result["prediction"] = (result["probability"] >= float(threshold)).astype(int)
    return result


def _forecast_confusion_row(
    *,
    target: str,
    horizon: int,
    predictor: str,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "target": target,
        "horizon_h": int(horizon),
        "predictor": predictor,
        "true_negative": int(metrics["true_negative"]),
        "false_positive": int(metrics["false_positive"]),
        "false_negative": int(metrics["false_negative"]),
        "true_positive": int(metrics["true_positive"]),
    }


def _previous_scope_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    previous_path = PREVIOUS_FORECAST_OUTPUT_DIR / FORECAST_METRICS_PATH.name
    columns = [
        "target",
        "horizon_h",
        "previous_1014_feature_test_precision",
        "event_history_test_precision",
        "delta_test_precision",
        "previous_1014_feature_test_recall",
        "event_history_test_recall",
        "delta_test_recall",
        "previous_1014_feature_test_f1",
        "event_history_test_f1",
        "delta_test_f1",
        "previous_1014_feature_test_accuracy",
        "event_history_test_accuracy",
        "delta_test_accuracy",
        "previous_metrics_sha256",
        "comparison_created_after_test_evaluation",
    ]
    if not previous_path.exists() or metrics.empty:
        return pd.DataFrame(columns=columns)
    previous = pd.read_csv(previous_path)
    required = {"target", "horizon_h", "predictor", "test_precision", "test_recall", "test_f1", "test_accuracy"}
    if not required.issubset(previous.columns):
        return pd.DataFrame(columns=columns)
    current = metrics.loc[
        metrics["predictor"].eq(FORECAST_MODEL_NAME),
        ["target", "horizon_h", "test_precision", "test_recall", "test_f1", "test_accuracy"],
    ].copy()
    prior = previous.loc[
        previous["predictor"].eq(FORECAST_MODEL_NAME),
        ["target", "horizon_h", "test_precision", "test_recall", "test_f1", "test_accuracy"],
    ].copy()
    comparison = prior.merge(
        current,
        on=["target", "horizon_h"],
        how="inner",
        suffixes=("_previous", "_corrected"),
        validate="one_to_one",
    )
    result = comparison.loc[:, ["target", "horizon_h"]].copy()
    for metric in ["precision", "recall", "f1", "accuracy"]:
        result[f"previous_1014_feature_test_{metric}"] = comparison[
            f"test_{metric}_previous"
        ]
        result[f"event_history_test_{metric}"] = comparison[
            f"test_{metric}_corrected"
        ]
        result[f"delta_test_{metric}"] = (
            result[f"event_history_test_{metric}"]
            - result[f"previous_1014_feature_test_{metric}"]
        )
    result["previous_metrics_sha256"] = _file_sha256(previous_path)
    result["comparison_created_after_test_evaluation"] = True
    return result.loc[:, columns].sort_values(
        ["target", "horizon_h"], kind="mergesort"
    ).reset_index(drop=True)


def _model_vs_persistence_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target",
        "horizon_h",
        "model_test_precision",
        "persistence_test_precision",
        "delta_test_precision",
        "model_test_recall",
        "persistence_test_recall",
        "delta_test_recall",
        "model_test_f1",
        "persistence_test_f1",
        "delta_test_f1",
        "model_test_accuracy",
        "persistence_test_accuracy",
        "delta_test_accuracy",
        "model_beats_persistence_test_f1",
        "comparison_created_after_test_evaluation",
    ]
    required = {
        "target",
        "horizon_h",
        "predictor",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_accuracy",
    }
    if metrics.empty or not required.issubset(metrics.columns):
        return pd.DataFrame(columns=columns)
    model = metrics.loc[
        metrics["predictor"].eq(FORECAST_MODEL_NAME),
        ["target", "horizon_h", "test_precision", "test_recall", "test_f1", "test_accuracy"],
    ]
    persistence = metrics.loc[
        metrics["predictor"].eq("persistence"),
        ["target", "horizon_h", "test_precision", "test_recall", "test_f1", "test_accuracy"],
    ]
    comparison = model.merge(
        persistence,
        on=["target", "horizon_h"],
        how="inner",
        suffixes=("_model", "_persistence"),
        validate="one_to_one",
    )
    result = comparison.loc[:, ["target", "horizon_h"]].copy()
    for metric in ["precision", "recall", "f1", "accuracy"]:
        result[f"model_test_{metric}"] = comparison[f"test_{metric}_model"]
        result[f"persistence_test_{metric}"] = comparison[
            f"test_{metric}_persistence"
        ]
        result[f"delta_test_{metric}"] = (
            result[f"model_test_{metric}"] - result[f"persistence_test_{metric}"]
        )
    result["model_beats_persistence_test_f1"] = result["delta_test_f1"].gt(0.0)
    result["comparison_created_after_test_evaluation"] = True
    return result.loc[:, columns].sort_values(
        ["target", "horizon_h"], kind="mergesort"
    ).reset_index(drop=True)


def _model_vs_recurrence_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target",
        "horizon_h",
        "model_test_precision",
        "recurrence_test_precision",
        "delta_test_precision",
        "model_test_recall",
        "recurrence_test_recall",
        "delta_test_recall",
        "model_test_f1",
        "recurrence_test_f1",
        "delta_test_f1",
        "model_test_accuracy",
        "recurrence_test_accuracy",
        "delta_test_accuracy",
        "model_beats_recurrence_test_f1",
        "comparison_created_after_test_evaluation",
    ]
    required = {
        "target",
        "horizon_h",
        "predictor",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_accuracy",
    }
    if metrics.empty or not required.issubset(metrics.columns):
        return pd.DataFrame(columns=columns)
    model = metrics.loc[
        metrics["predictor"].eq(FORECAST_MODEL_NAME),
        ["target", "horizon_h", "test_precision", "test_recall", "test_f1", "test_accuracy"],
    ]
    recurrence = metrics.loc[
        metrics["predictor"].eq("recurrence_168h"),
        ["target", "horizon_h", "test_precision", "test_recall", "test_f1", "test_accuracy"],
    ]
    comparison = model.merge(
        recurrence,
        on=["target", "horizon_h"],
        how="inner",
        suffixes=("_model", "_recurrence"),
        validate="one_to_one",
    )
    result = comparison.loc[:, ["target", "horizon_h"]].copy()
    for metric in ["precision", "recall", "f1", "accuracy"]:
        result[f"model_test_{metric}"] = comparison[f"test_{metric}_model"]
        result[f"recurrence_test_{metric}"] = comparison[
            f"test_{metric}_recurrence"
        ]
        result[f"delta_test_{metric}"] = (
            result[f"model_test_{metric}"] - result[f"recurrence_test_{metric}"]
        )
    result["model_beats_recurrence_test_f1"] = result["delta_test_f1"].gt(0.0)
    result["comparison_created_after_test_evaluation"] = True
    return result.loc[:, columns].sort_values(
        ["target", "horizon_h"], kind="mergesort"
    ).reset_index(drop=True)


def _continuation_vs_onset_comparison(onset_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target",
        "horizon_h",
        "continuation_test_precision",
        "onset_test_precision",
        "delta_test_precision_onset_minus_continuation",
        "continuation_test_recall",
        "onset_test_recall",
        "delta_test_recall_onset_minus_continuation",
        "continuation_test_f1",
        "onset_test_f1",
        "delta_test_f1_onset_minus_continuation",
        "continuation_test_accuracy",
        "onset_test_accuracy",
        "delta_test_accuracy_onset_minus_continuation",
        "continuation_metrics_sha256",
        "comparison_created_after_test_evaluation",
    ]
    previous_path = FORECAST_OUTPUT_DIR / FORECAST_METRICS_PATH.name
    required = {
        "target",
        "horizon_h",
        "predictor",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_accuracy",
    }
    if (
        onset_metrics.empty
        or not required.issubset(onset_metrics.columns)
        or not previous_path.exists()
    ):
        return pd.DataFrame(columns=columns)
    continuation = pd.read_csv(previous_path)
    if not required.issubset(continuation.columns):
        return pd.DataFrame(columns=columns)
    left = continuation.loc[
        continuation["predictor"].eq(FORECAST_MODEL_NAME),
        ["target", "horizon_h", "test_precision", "test_recall", "test_f1", "test_accuracy"],
    ]
    right = onset_metrics.loc[
        onset_metrics["predictor"].eq(FORECAST_MODEL_NAME),
        ["target", "horizon_h", "test_precision", "test_recall", "test_f1", "test_accuracy"],
    ]
    comparison = left.merge(
        right,
        on=["target", "horizon_h"],
        how="inner",
        suffixes=("_continuation", "_onset"),
        validate="one_to_one",
    )
    result = comparison.loc[:, ["target", "horizon_h"]].copy()
    for metric in ["precision", "recall", "f1", "accuracy"]:
        result[f"continuation_test_{metric}"] = comparison[
            f"test_{metric}_continuation"
        ]
        result[f"onset_test_{metric}"] = comparison[f"test_{metric}_onset"]
        result[f"delta_test_{metric}_onset_minus_continuation"] = (
            result[f"onset_test_{metric}"] - result[f"continuation_test_{metric}"]
        )
    result["continuation_metrics_sha256"] = _file_sha256(previous_path)
    result["comparison_created_after_test_evaluation"] = True
    return result.loc[:, columns].sort_values(
        ["target", "horizon_h"], kind="mergesort"
    ).reset_index(drop=True)


def _onset_forecast_report(result: dict[str, object]) -> str:
    required_frames = [
        "metrics",
        "confusion",
        "importance",
        "future_validation_summary",
        "feature_counts",
        "risk_set_comparison",
        "independent_onset_support",
        "onset_construction",
        "onset_eligibility",
        "network_event_policy",
        "continuation_comparison",
        "recurrence_comparison",
    ]
    for name in required_frames:
        if not isinstance(result.get(name), pd.DataFrame):
            raise TypeError(f"onset report requires DataFrame {name}")
    metrics = result["metrics"]
    model_columns = [
        "target",
        "horizon_h",
        "predictor",
        "validation_f1",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_all_precision_recall_f1_ge_080",
    ]
    model_rows = (
        metrics.loc[metrics["predictor"].eq(FORECAST_MODEL_NAME)].copy()
        if "predictor" in metrics.columns
        else pd.DataFrame(columns=model_columns)
    )
    model_rows["test_minus_validation_f1"] = (
        model_rows["test_f1"] - model_rows["validation_f1"]
    )
    passing = model_rows.loc[
        model_rows["test_all_precision_recall_f1_ge_080"].eq(True),
        ["target", "horizon_h"],
    ]
    failing = model_rows.loc[
        model_rows["test_all_precision_recall_f1_ge_080"].eq(False),
        ["target", "horizon_h", "test_precision", "test_recall", "test_f1"],
    ]
    lines = [
        "ONSET-ONLY INCIDENT-HAZARD FORECAST EXPERIMENT",
        "",
        "This is a separate, like-for-like rerun of the causal HistGradientBoosting route.",
        "It does not overwrite or select against the earlier continuation-target result.",
        "",
        "TARGET DEFINITION",
        "A scored row must be currently clean and at risk: it cannot be inside an active",
        "fault or full outage, and it must be outside the fixed post-incident recovery exclusion.",
        "The label is one when a new sustained qualifying incident starts strictly after",
        "the scored hour and within the next H clock-hours on the same station.",
        "A start is qualifying only when the immediately preceding hour was observable",
        "and clean; starts first seen after an unobservable interval are excluded because",
        "their true onset time cannot be established.",
        "Fault incidents require at least 3 continuous labelled fault-hours; outage incidents",
        "require at least 6 continuous full-outage hours. Confirmation extends the label span",
        "and therefore the timestamp-boundary purge to H + duration - 1 hours.",
        f"The recovery exclusion is fixed a priori at {ONSET_RECOVERY_EXCLUSION_HOURS} hours",
        "for every target and horizon: a full clean day is required before a new risk score.",
        "",
        "NETWORK-OUTAGE POLICY",
        _format_frame(result["network_event_policy"]),
        "",
        "RISK-SET CONSTRUCTION AND COMPARISON WITH THE CONTINUATION TARGET",
        _format_frame(result["onset_construction"]),
        _format_frame(result["risk_set_comparison"]),
        "",
        "INDEPENDENT INCIDENT-ONSET SUPPORT",
        "Positive score rows are not independent incidents. The table counts distinct future",
        "qualifying incident IDs reached by each partition's score rows.",
        _format_frame(result["independent_onset_support"]),
        "",
        "ONSET ELIGIBILITY INVARIANT",
        _format_frame(result["onset_eligibility"]),
        "",
        "FEATURE SET AND CAUSALITY",
        "The model retains the 1,014 causal measurement features plus the existing strictly",
        "prior event-history family. Fault-history fields use past confirmed ground-truth labels",
        "and are an optimistic retrospective input, not a deployment-ready dashboard feature.",
        _format_frame(result["feature_counts"]),
        "",
        "DELETE-THE-FUTURE VALIDATION",
        _format_frame(result["future_validation_summary"]),
        "",
        "VALIDATION-SELECTED OPERATING POINTS AND HELD-OUT TEST RESULTS",
        "Class weights are derived from training labels only. The weight/threshold sweep selects",
        "the maximum validation minimum of precision, recall, and F1. Test is evaluated once",
        "after that choice is frozen. No test metric selects a configuration.",
        _format_frame(metrics),
        "",
        "MODEL VERSUS FIXED SEVEN-DAY RECURRENCE BASELINE",
        "The recurrence baseline predicts a new onset when a qualifying incident had already",
        "been confirmed during the strictly preceding seven days. It is a fixed comparison,",
        "not a selected operating point.",
        _format_frame(result["recurrence_comparison"]),
        "",
        "ONSET VERSUS CONTINUATION-TARGET MODEL RESULTS",
        "This post-test table is descriptive only and was not available to the selection process.",
        _format_frame(result["continuation_comparison"]),
        "",
        "VALIDATION-TO-TEST F1 GAP FOR THE ONSET MODELS",
        _format_frame(model_rows.loc[:, [
            "target",
            "horizon_h",
            "validation_f1",
            "test_f1",
            "test_minus_validation_f1",
        ]]),
        "",
        "CONFIGURATIONS WITH TEST PRECISION, RECALL, AND F1 ALL AT LEAST 0.80",
        _format_frame(passing),
        "",
        "CONFIGURATIONS FALLING SHORT OF THE 0.80 THREE-METRIC CRITERION",
        _format_frame(failing),
        "",
        "HELD-OUT CONFUSION MATRICES",
        _format_frame(result["confusion"]),
        "",
        "VALIDATION-ONLY H24 PERMUTATION IMPORTANCE",
        _format_frame(result["importance"]),
        "",
        f"completed_configurations={result['completed_configurations']}",
        f"timebox_reached={bool(result['timebox_reached'])}",
        f"elapsed_seconds={float(result['elapsed_seconds']):.3f}",
    ]
    return "\n".join(lines) + "\n"


def _forecast_report(result: dict[str, object]) -> str:
    if result.get("risk_definition") == "onset":
        return _onset_forecast_report(result)
    metrics = result["metrics"]
    audit = result["causality_audit"]
    confusion = result["confusion"]
    importance = result["importance"]
    future_validation = result["future_validation"]
    future_validation_summary = result["future_validation_summary"]
    previous_comparison = result["previous_comparison"]
    persistence_comparison = result["persistence_comparison"]
    feature_counts = result["feature_counts"]
    deployment_realistic_variant = result["deployment_realistic_variant"]
    if not all(
        isinstance(frame, pd.DataFrame)
        for frame in [
            metrics,
            audit,
            confusion,
            importance,
            future_validation,
            future_validation_summary,
            previous_comparison,
            persistence_comparison,
            feature_counts,
            deployment_realistic_variant,
        ]
    ):
        raise TypeError("forecast report inputs must be DataFrames")
    model_columns = [
        "target",
        "horizon_h",
        "predictor",
        "validation_f1",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_all_precision_recall_f1_ge_080",
    ]
    model_rows = (
        metrics.loc[metrics["predictor"].eq(FORECAST_MODEL_NAME)].copy()
        if "predictor" in metrics.columns
        else pd.DataFrame(columns=model_columns)
    )
    model_rows["test_minus_validation_f1"] = (
        model_rows["test_f1"] - model_rows["validation_f1"]
    )
    passing = model_rows.loc[
        model_rows["test_all_precision_recall_f1_ge_080"].eq(True),
        ["target", "horizon_h"],
    ]
    failing = model_rows.loc[
        model_rows["test_all_precision_recall_f1_ge_080"].eq(False),
        ["target", "horizon_h", "test_precision", "test_recall", "test_f1"],
    ]
    included = audit.loc[
        audit["included"],
        ["feature", "source", "time_contract", "scope_change"],
    ]
    excluded = audit.loc[~audit["included"]]
    reinstated = audit.loc[
        audit["scope_change"].eq("reinstated_with_as_of_time_reconstruction")
    ]
    completed = result["completed_configurations"]
    lines = [
        "CAUSAL FORECAST-RISK EVENT-HISTORY EXPERIMENT",
        "",
        "Six predeclared target/horizon configurations use HistGradientBoosting.",
        "The corrected scope permits raw station readings, exact-hour ERA5/reference values,",
        "same-hour spatial comparisons, and causally reconstructed detector evidence at or before t.",
        "All rolling measurement baselines are explicitly shifted; historical detector-matrix artifacts, labels,",
        "episodes, forward five-minute snapshots, whole-day solar ratios, and retrospective baselines remain excluded.",
        "This experiment additionally supplies strictly prior target-specific event-history features.",
        "Fault-history values use past confirmed ground-truth labels and are therefore an optimistic",
        "retrospective feature source, not a deployment-ready fault-history input.",
        f"base_causal_measurement_feature_count={len(CAUSAL_FORECAST_FEATURE_COLUMNS)}",
        f"reinstated_feature_count={len(reinstated)}",
        f"completed_configurations={completed}",
        f"timebox_reached={bool(result['timebox_reached'])}",
        f"elapsed_seconds={float(result['elapsed_seconds']):.3f}",
        "",
        "MODEL FEATURE COUNTS BY CONFIGURATION",
        _format_frame(feature_counts),
        "",
        "CAUSALITY AUDIT",
        "Included features:",
        _format_frame(included),
        "",
        "Excluded feature count by source:",
        _format_frame(
            excluded.groupby("source", as_index=False).size().rename(columns={"size": "n_excluded"})
        ),
        "The full per-feature audit is saved in forecast_risk_causality_audit.csv.",
        "",
        "DELETE-THE-FUTURE VALIDATION",
        _format_frame(future_validation_summary),
        "Every value comparison is saved in forecast_risk_delete_future_validation.csv.",
        "",
        "DEPLOYMENT-REALISM VARIANT",
        _format_frame(deployment_realistic_variant),
        "",
        "VALIDATION-SELECTED OPERATING POINTS AND HELD-OUT TEST RESULTS",
        _format_frame(metrics),
        "",
        "VALIDATION-TO-TEST F1 GAP FOR THE MODEL ROWS",
        _format_frame(model_rows.loc[:, [
            "target",
            "horizon_h",
            "validation_f1",
            "test_f1",
            "test_minus_validation_f1",
        ]]),
        "",
        "CONFIGURATIONS WITH TEST PRECISION, RECALL, AND F1 ALL AT LEAST 0.80",
        _format_frame(passing),
        "",
        "CONFIGURATIONS FALLING SHORT OF THE 0.80 THREE-METRIC CRITERION",
        _format_frame(failing),
        "",
        "BASELINE INTERPRETATION",
        "Base-rate and persistence rows are reported beside each model row. Persistence remains a comparison baseline. The fitted model receives the expanded target-specific history family; fault-history inputs are explicitly marked as retrospective ground-truth history rather than a live dashboard input.",
        "",
        "PREVIOUS 1,014-FEATURE VERSUS EVENT-HISTORY COMPARISON",
        _format_frame(previous_comparison),
        "The previous test results are read only after all event-history test evaluations complete and never influence selection.",
        "",
        "MODEL VERSUS PERSISTENCE ON THE HELD-OUT TEST SET",
        _format_frame(persistence_comparison),
        "This is a post-selection comparison only; it does not choose a model, threshold, or class weight.",
        "",
        "PER-STATION LIMITATION",
        "IJANZO4 has two direct fault-hours and some station/partition combinations have zero positives; per-station performance is not meaningful for those cases.",
        "",
        "TEST-DATA DISCIPLINE",
        "Class weights are derived from training labels. Weight/threshold selection uses validation only. Each selected model is evaluated on test once after selection is frozen; no test result chooses a configuration.",
        "",
        "CONFUSION MATRICES",
        _format_frame(confusion),
        "",
        "VALIDATION-ONLY H24 PERMUTATION IMPORTANCE",
        _format_frame(importance),
    ]
    return "\n".join(lines) + "\n"


def train_forecast_risk_models(
    output_dir: Path,
    *,
    events_path: Path = AVAILABILITY_EVENTS_PATH,
    timebox_seconds: int = FORECAST_TIMEBOX_SECONDS,
    risk_definition: str = "continuation",
) -> dict[str, object]:
    if risk_definition not in {"continuation", "onset"}:
        raise ValueError("risk definition must be continuation or onset")
    started = time.monotonic()
    _require_exact_hour_reference_cache()
    continuation_outage_info, continuation_fault_info = _build_forecast_target_splits()
    continuation_outage_dataset = continuation_outage_info["dataset"]
    continuation_fault_dataset = continuation_fault_info["dataset"]
    if risk_definition == "continuation":
        outage_info, fault_info = continuation_outage_info, continuation_fault_info
    else:
        outage_info, fault_info = _build_onset_target_splits(continuation_outage_info)
    outage_dataset = outage_info["dataset"]
    fault_dataset = fault_info["dataset"]
    if not hasattr(continuation_outage_dataset, "frame") or not hasattr(
        continuation_fault_dataset,
        "frame",
    ):
        raise TypeError("continuation target construction returned an invalid dataset")
    before = _forecast_invariant_snapshot(
        events_path,
        continuation_outage_dataset,
        continuation_fault_dataset,
    )
    feature_matrix_columns = ParquetFile(HOURLY_FEATURE_PATH).schema.names
    causal_sources = load_causal_forecast_sources()
    feature_bundle = build_causal_forecast_features(
        continuation_outage_dataset.frame,
        **causal_sources,
        feature_matrix_columns=feature_matrix_columns,
    )
    event_history_grid = _build_event_history_grid(
        continuation_outage_dataset,
        continuation_fault_dataset,
    )
    event_history = build_backward_event_history_features(
        event_history_grid,
        horizons=HORIZONS,
    )
    causal_future_validation = validate_delete_future_features(
        continuation_outage_dataset.frame,
        **causal_sources,
        feature_matrix_columns=feature_matrix_columns,
        full_bundle=feature_bundle,
    )
    event_history_future_validation = validate_delete_future_event_history_features(
        event_history_grid,
        horizons=HORIZONS,
        full_history=event_history,
    )
    future_validation = pd.concat(
        [
            causal_future_validation.assign(feature_family="causal_measurement"),
            event_history_future_validation.assign(
                feature_family="strictly_prior_event_history"
            ),
        ],
        ignore_index=True,
    )
    future_validation_summary = summarize_delete_future_validation(future_validation)
    if not bool(future_validation_summary.loc[0, "all_passed"]):
        failures = future_validation.loc[~future_validation["passed"].astype(bool)]
        raise RuntimeError(
            f"delete-the-future validation failed for {len(failures)} feature values"
        )
    feature_keys = feature_bundle.frame.loc[:, ["station_id", "hour_utc"]].sort_values(
        ["station_id", "hour_utc"],
        kind="mergesort",
    ).reset_index(drop=True)
    for name, dataset in {"fault": fault_dataset, "outage": outage_dataset}.items():
        if not hasattr(dataset, "frame"):
            raise TypeError(f"{name} target construction returned an invalid dataset")
        target_keys = dataset.frame.loc[:, ["station_id", "hour_utc"]].sort_values(
            ["station_id", "hour_utc"],
            kind="mergesort",
        ).reset_index(drop=True)
        if not feature_keys.equals(target_keys):
            raise RuntimeError(f"{name} target grid differs from causal feature attachment keys")
    if not feature_bundle.frame.loc[:, ["station_id", "hour_utc"]].equals(
        event_history.loc[:, ["station_id", "hour_utc"]]
    ):
        raise RuntimeError("causal measurements and event history have different station-hour keys")

    target_info = {"outage": outage_info, "fault": fault_info}
    model_directory = Path(output_dir) / FORECAST_MODELS_DIR.name
    model_directory.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    selection_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    history_audit_frames: list[pd.DataFrame] = []
    feature_count_rows: list[dict[str, object]] = []
    split_digests: dict[str, object] = {}
    completed_configurations: list[str] = []
    timebox_reached = False

    for target in ["fault", "outage"]:
        info = target_info[target]
        dataset = info["dataset"]
        characteristics = info["characteristics"]
        event_column = str(info["event_column"])
        if not hasattr(dataset, "frame") or not isinstance(characteristics, dict):
            raise TypeError("forecast target construction returned an invalid dataset")
        persistence = build_retrospective_persistence_history(
            dataset.frame,
            event_column=event_column,
        )
        recurrence_history = (
            build_confirmed_incident_recurrence_history(dataset)
            if risk_definition == "onset"
            else None
        )
        recurrence_column = "confirmed_incident_start_count_trailing_168h"
        splits = characteristics["splits"]
        if not isinstance(splits, dict):
            raise TypeError("forecast target characteristics lack split definitions")
        for horizon in HORIZONS:
            if time.monotonic() - started > int(timebox_seconds):
                timebox_reached = True
                break
            split = splits[int(horizon)]
            if not isinstance(split, dict):
                raise TypeError("forecast split must be a dictionary")
            split_digests[f"{target}_{int(horizon)}h"] = _risk_split_digest(split)
            history_columns = event_history_feature_columns(int(horizon))
            model_feature_columns = (
                *feature_bundle.feature_columns,
                *history_columns,
            )
            attached = _attach_forecast_features(
                split,
                feature_bundle.frame,
                event_history,
                persistence,
                feature_bundle.feature_columns,
                history_columns,
                int(horizon),
                recurrence_frame=recurrence_history,
            )
            train = attached["train"]
            validation = attached["validation"]
            test = attached["test"]
            try:
                selection = fit_forecast_hist_gradient_boosting(
                    train,
                    validation,
                    model_feature_columns,
                    deadline_monotonic=started + float(timebox_seconds),
                )
            except ForecastTrainingTimeboxReached:
                timebox_reached = True
                break
            trace = selection.selection_trace.copy()
            trace["target"] = target
            trace["horizon_h"] = int(horizon)
            trace["risk_definition"] = risk_definition
            trace["selection_rule"] = "maximize_validation_min_precision_recall_f1"
            trace["test_metrics_accessed_during_selection"] = False
            selection_frames.append(trace)

            if int(horizon) == 24:
                importance = validation_permutation_importance(
                    selection.model,
                    validation,
                    model_feature_columns,
                    selection.threshold,
                ).head(20)
                importance["target"] = target
                importance["horizon_h"] = int(horizon)
                importance_frames.append(importance)

            model_test_probability = forecast_model_probability(
                selection.model,
                test,
                model_feature_columns,
            )
            model_test_metrics = forecast_classification_metrics(
                test["y"].to_numpy(dtype=int),
                model_test_probability,
                selection.threshold,
            )
            model_row = _forecast_master_row(
                target=target,
                horizon=int(horizon),
                predictor=FORECAST_MODEL_NAME,
                validation_metrics=selection.validation_metrics,
                test_metrics=model_test_metrics,
                train_rows=len(train),
                validation_rows=len(validation),
                test_rows=len(test),
                train_positive=int(train["y"].sum()),
                validation_positive=int(validation["y"].sum()),
                test_positive=int(test["y"].sum()),
                selected_positive_class_weight=selection.positive_class_weight,
                selected_weight_multiplier=selection.weight_multiplier,
                selected_threshold=selection.threshold,
            )
            metric_rows.append(model_row)
            metric_rows[-1]["risk_definition"] = risk_definition
            metric_rows[-1]["model_feature_count"] = len(model_feature_columns)
            metric_rows[-1]["history_feature_count"] = len(history_columns)
            confusion_rows.append(
                _forecast_confusion_row(
                    target=target,
                    horizon=int(horizon),
                    predictor=FORECAST_MODEL_NAME,
                    metrics=model_test_metrics,
                )
            )
            confusion_rows[-1]["risk_definition"] = risk_definition
            model_prediction_rows = _forecast_prediction_rows(
                test,
                target=target,
                horizon=int(horizon),
                predictor=FORECAST_MODEL_NAME,
                probability=model_test_probability,
                threshold=selection.threshold,
            )
            model_prediction_rows["risk_definition"] = risk_definition
            prediction_frames.append(model_prediction_rows)

            baselines = [
                (
                    "base_rate",
                    forecast_base_rate_prediction(train["y"].to_numpy(dtype=int), validation),
                    forecast_base_rate_prediction(train["y"].to_numpy(dtype=int), test),
                )
            ]
            if risk_definition == "continuation":
                baselines.append(
                    (
                        "persistence",
                        forecast_persistence_prediction(validation, int(horizon)),
                        forecast_persistence_prediction(test, int(horizon)),
                    )
                )
            else:
                baselines.append(
                    (
                        "recurrence_168h",
                        forecast_recurrence_prediction(
                            validation,
                            recurrence_column=recurrence_column,
                        ),
                        forecast_recurrence_prediction(
                            test,
                            recurrence_column=recurrence_column,
                        ),
                    )
                )
            for predictor, validation_prediction, test_prediction in baselines:
                validation_metrics = forecast_classification_metrics(
                    validation["y"].to_numpy(dtype=int),
                    validation_prediction.prob,
                    validation_prediction.threshold,
                )
                test_metrics = forecast_classification_metrics(
                    test["y"].to_numpy(dtype=int),
                    test_prediction.prob,
                    test_prediction.threshold,
                )
                metric_rows.append(
                    _forecast_master_row(
                        target=target,
                        horizon=int(horizon),
                        predictor=predictor,
                        validation_metrics=validation_metrics,
                        test_metrics=test_metrics,
                        train_rows=len(train),
                        validation_rows=len(validation),
                        test_rows=len(test),
                        train_positive=int(train["y"].sum()),
                        validation_positive=int(validation["y"].sum()),
                        test_positive=int(test["y"].sum()),
                        selected_threshold=validation_prediction.threshold,
                    )
                )
                metric_rows[-1]["risk_definition"] = risk_definition
                confusion_rows.append(
                    _forecast_confusion_row(
                        target=target,
                        horizon=int(horizon),
                        predictor=predictor,
                        metrics=test_metrics,
                    )
                )
                confusion_rows[-1]["risk_definition"] = risk_definition
                baseline_prediction_rows = _forecast_prediction_rows(
                    test,
                    target=target,
                    horizon=int(horizon),
                    predictor=predictor,
                    probability=test_prediction.prob,
                    threshold=test_prediction.threshold,
                )
                baseline_prediction_rows["risk_definition"] = risk_definition
                prediction_frames.append(baseline_prediction_rows)

            artifact_suffix = "risk" if risk_definition == "continuation" else "onset"
            model_path = model_directory / f"{target}_{artifact_suffix}_{int(horizon)}h.joblib"
            joblib.dump(
                {
                    "model": selection.model,
                    "model_name": FORECAST_MODEL_NAME,
                    "target": target,
                    "risk_definition": risk_definition,
                    "horizon_h": int(horizon),
                    "feature_columns": list(model_feature_columns),
                    "event_history_source": "retrospective_ground_truth_fault_labels_and_observed_outages",
                    "positive_class_weight": selection.positive_class_weight,
                    "weight_multiplier": selection.weight_multiplier,
                    "threshold": selection.threshold,
                    "selection_rule": "maximize_validation_min_precision_recall_f1",
                    "split_key_digests": split_digests[f"{target}_{int(horizon)}h"],
                    "validation_metrics": selection.validation_metrics,
                },
                model_path,
            )
            history_audit = build_event_history_feature_audit(int(horizon))
            history_audit["target"] = target
            history_audit["horizon_h"] = int(horizon)
            history_audit["risk_definition"] = risk_definition
            history_audit_frames.append(history_audit)
            feature_count_rows.append(
                {
                    "target": target,
                    "horizon_h": int(horizon),
                    "risk_definition": risk_definition,
                    "causal_measurement_feature_count": len(feature_bundle.feature_columns),
                    "event_history_feature_count": len(history_columns),
                    "model_feature_count": len(model_feature_columns),
                }
            )
            completed_configurations.append(f"{target}_{int(horizon)}h")
        if timebox_reached:
            break

    after_outage = build_risk_dataset()
    after_fault = build_fault_risk_dataset()
    after = _forecast_invariant_snapshot(events_path, after_outage, after_fault)
    if before != after:
        raise RuntimeError("forecast-risk training changed an upstream artifact")
    metrics = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(confusion_rows)
    selection_trace = (
        pd.concat(selection_frames, ignore_index=True)
        if selection_frames
        else pd.DataFrame()
    )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    importance = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame()
    )
    continuation_info = {
        "outage": continuation_outage_info,
        "fault": continuation_fault_info,
    }
    if risk_definition == "continuation":
        previous_comparison = _previous_scope_comparison(metrics)
        persistence_comparison = _model_vs_persistence_comparison(metrics)
        onset_construction = pd.DataFrame()
        risk_set_comparison = pd.DataFrame()
        independent_onset_support = pd.DataFrame()
        onset_eligibility = pd.DataFrame()
        network_event_policy = pd.DataFrame()
        continuation_comparison = pd.DataFrame()
        recurrence_comparison = pd.DataFrame()
    else:
        previous_comparison = pd.DataFrame()
        persistence_comparison = pd.DataFrame()
        onset_construction = pd.concat(
            [
                target_info["fault"]["dataset"].construction_summary(),
                target_info["outage"]["dataset"].construction_summary(),
            ],
            ignore_index=True,
        ).sort_values(["target", "horizon_h"], kind="mergesort").reset_index(
            drop=True
        )
        risk_set_comparison = _onset_risk_set_comparison(continuation_info, target_info)
        independent_onset_support = _independent_onset_support(target_info)
        onset_eligibility = _onset_eligibility_audit(target_info)
        network_event_policy = _onset_network_event_policy(target_info["outage"]["dataset"])
        continuation_comparison = _continuation_vs_onset_comparison(metrics)
        recurrence_comparison = _model_vs_recurrence_comparison(metrics)
    history_audit = (
        pd.concat(history_audit_frames, ignore_index=True)
        if history_audit_frames
        else pd.DataFrame()
    )
    causality_audit = pd.concat(
        [
            feature_bundle.causality_audit.assign(
                deployment_status="causal_measurement_or_static_context"
            ),
            history_audit,
        ],
        ignore_index=True,
        sort=False,
    )
    feature_set = causality_audit.loc[
        causality_audit["included"],
        [
            "feature",
            "source",
            "time_contract",
            "scope_change",
            "reason",
            "deployment_status",
        ],
    ].copy()
    feature_set.insert(0, "feature_index", np.arange(1, len(feature_set) + 1, dtype=int))
    elapsed_seconds = float(time.monotonic() - started)
    return {
        "metrics": metrics,
        "confusion": confusion,
        "selection_trace": selection_trace,
        "predictions": predictions,
        "importance": importance,
        "future_validation": future_validation,
        "future_validation_summary": future_validation_summary,
        "risk_definition": risk_definition,
        "previous_comparison": previous_comparison,
        "persistence_comparison": persistence_comparison,
        "onset_construction": onset_construction,
        "risk_set_comparison": risk_set_comparison,
        "independent_onset_support": independent_onset_support,
        "onset_eligibility": onset_eligibility,
        "network_event_policy": network_event_policy,
        "continuation_comparison": continuation_comparison,
        "recurrence_comparison": recurrence_comparison,
        "causality_audit": causality_audit,
        "feature_set": feature_set,
        "feature_counts": pd.DataFrame(feature_count_rows),
        "deployment_realistic_variant": pd.DataFrame(
            [
                {
                    "status": "not_run",
                    "reason": (
                        "No causal all-hour detector-prediction ledger exists. Saved binary "
                        "detector models use non-temporal splits, so replaying them over the "
                        "history would leak later labelled training information."
                    ),
                }
            ]
        ),
        "split_digests": split_digests,
        "invariants": {"before": before, "after": after},
        "completed_configurations": completed_configurations,
        "timebox_reached": bool(timebox_reached or elapsed_seconds >= timebox_seconds),
        "elapsed_seconds": elapsed_seconds,
    }


def _discrete_hazard_report(result: dict[str, object]) -> str:
    required = [
        "metrics",
        "selection_trace",
        "calibration",
        "feature_counts",
        "feature_audit",
        "feature_future_validation",
        "hazard_support",
        "independent_onset_support",
        "onset_construction",
        "onset_eligibility",
        "network_event_policy",
        "direct_onset_comparison",
        "manifest_validation",
        "deployment_scope",
        "run_status",
    ]
    for name in required:
        if not isinstance(result.get(name), pd.DataFrame):
            raise TypeError(f"discrete-hazard report requires DataFrame {name}")
    metrics = result["metrics"]
    criterion_columns = [
        "target",
        "horizon_h",
        "predictor",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    if metrics.empty or "test_all_precision_recall_f1_ge_080" not in metrics.columns:
        failing = pd.DataFrame(columns=criterion_columns)
        passing = pd.DataFrame(columns=["target", "horizon_h", "predictor"])
    else:
        failing = metrics.loc[
            ~metrics["test_all_precision_recall_f1_ge_080"].astype(bool),
            criterion_columns,
        ]
        passing = metrics.loc[
            metrics["test_all_precision_recall_f1_ge_080"].astype(bool),
            ["target", "horizon_h", "predictor"],
        ]
    lines = [
        "DISCRETE-TIME RECURRENT-INCIDENT HAZARD EXPERIMENT",
        "",
        "This experiment fits one hourly onset-hazard model per target and method, then",
        "derives 6, 12, and 24-hour risk by holding all score-time covariates fixed and",
        "applying P(event within H) = 1 - (1 - h_t)^H. It does not overwrite the direct",
        "onset-classifier experiment.",
        "",
        "RUN STATUS",
        _format_frame(result["run_status"]),
        "",
        "RISK SET AND TARGET",
        "Rows are currently clean, observable, and outside the fixed 24-hour recovery",
        "exclusion. The one-hour outcome is a new observed, sustained qualifying onset in",
        "the next clock hour; fault and outage duration confirmation remain part of the",
        "label span. Network-associated outage starts remain censored.",
        _format_frame(result["hazard_support"]),
        "",
        "SHARED HAZARD FIT AND HORIZON EVALUATION",
        "Each target/method is fitted once on the strictest existing H24-safe training",
        "partition. Its probability calibrator is fitted only on the matching H24 validation",
        "partition. Each H6/H12/H24 operating threshold is then frozen from that horizon's",
        "existing validation partition; no cross-method winner is named and no test metric",
        "selects a method, threshold, or model parameter.",
        _format_frame(result["manifest_validation"]),
        "",
        "COMPACT PREDECLARED FEATURE SCHEMA",
        "The schema has 45 numeric fields plus regularised one-hot station indicators. It",
        "contains calendar/static context, prior availability, group-level detector evidence,",
        "pressure/temperature/wind summaries, and target-specific strictly prior recurrence.",
        "Fault-history fields remain an optimistic retrospective input rather than a live",
        "dashboard feature until a causal all-hour detector ledger exists.",
        _format_frame(result["feature_counts"]),
        _format_frame(result["feature_future_validation"]),
        _format_frame(result["deployment_scope"]),
        "",
        "INDEPENDENT ONSET SUPPORT",
        "Positive hourly rows are not independent events. The direct-horizon table below",
        "reports distinct future incident IDs per partition.",
        _format_frame(result["independent_onset_support"]),
        "",
        "VALIDATION-SELECTED THRESHOLDS AND HELD-OUT TEST RESULTS",
        _format_frame(metrics),
        "",
        "TEST CALIBRATION",
        "Rows are probability deciles on the held-out test set. Brier score is reported for",
        "the H-hour cumulated probability. Sparse deciles must not be over-interpreted.",
        _format_frame(result["calibration"]),
        "",
        "DIRECT ONSET-CLASSIFIER COMPARISON",
        "This is calculated only after all discrete-hazard test evaluations complete and has",
        "no role in model selection. A timeboxed partial run leaves this table empty.",
        _format_frame(result["direct_onset_comparison"]),
        "",
        "ONSET CONSTRUCTION AND ELIGIBILITY INVARIANTS",
        _format_frame(result["onset_construction"]),
        _format_frame(result["onset_eligibility"]),
        _format_frame(result["network_event_policy"]),
        "",
        "0.80 THREE-METRIC CRITERION",
        "Configurations meeting precision, recall, and F1 >= 0.80:",
        _format_frame(passing),
        "Configurations below that criterion:",
        _format_frame(failing),
        "",
        "TEST-DATA DISCIPLINE",
        "Training class weights use the one-hour training labels only. Probability calibration",
        "uses the fixed H24 validation partition only. Horizon thresholds use validation only.",
        "Each test configuration is evaluated once after all selections are frozen.",
    ]
    return "\n".join(lines) + "\n"


def train_discrete_hazard_models(
    output_dir: Path,
    *,
    events_path: Path = AVAILABILITY_EVENTS_PATH,
    timebox_seconds: int = DISCRETE_HAZARD_TIMEBOX_SECONDS,
) -> dict[str, object]:
    started = time.monotonic()
    deadline = started + float(timebox_seconds)
    _require_exact_hour_reference_cache()
    continuation_outage_info, continuation_fault_info = _build_forecast_target_splits()
    onset_outage_info, onset_fault_info = _build_onset_target_splits(
        continuation_outage_info
    )
    continuation_outage_dataset = continuation_outage_info["dataset"]
    continuation_fault_dataset = continuation_fault_info["dataset"]
    if not hasattr(continuation_outage_dataset, "frame") or not hasattr(
        continuation_fault_dataset, "frame"
    ):
        raise TypeError("continuation target construction returned an invalid dataset")
    before = _forecast_invariant_snapshot(
        events_path,
        continuation_outage_dataset,
        continuation_fault_dataset,
    )
    feature_matrix_columns = ParquetFile(HOURLY_FEATURE_PATH).schema.names
    causal_sources = load_causal_forecast_sources()
    compact_causal_columns = discrete_hazard_causal_source_columns("fault")
    feature_bundle = build_causal_forecast_features(
        continuation_outage_dataset.frame,
        **causal_sources,
        feature_matrix_columns=feature_matrix_columns,
        feature_columns=compact_causal_columns,
    )
    event_history_grid = _build_event_history_grid(
        continuation_outage_dataset,
        continuation_fault_dataset,
    )
    event_history = build_backward_event_history_features(
        event_history_grid,
        horizons=HORIZONS,
    )
    causal_future_validation = validate_delete_future_features(
        continuation_outage_dataset.frame,
        **causal_sources,
        feature_matrix_columns=feature_matrix_columns,
        feature_columns=compact_causal_columns,
        full_bundle=feature_bundle,
    )
    history_future_validation = validate_delete_future_event_history_features(
        event_history_grid,
        horizons=HORIZONS,
        full_history=event_history,
    )
    future_validation = pd.concat(
        [
            causal_future_validation.assign(feature_family="causal_measurement"),
            history_future_validation.assign(feature_family="strictly_prior_event_history"),
        ],
        ignore_index=True,
    )
    future_validation_summary = summarize_delete_future_validation(future_validation)
    if not bool(future_validation_summary.loc[0, "all_passed"]):
        failures = future_validation.loc[~future_validation["passed"].astype(bool)]
        raise RuntimeError(
            f"delete-the-future validation failed for {len(failures)} feature values"
        )
    feature_keys = feature_bundle.frame.loc[:, ["station_id", "hour_utc"]].reset_index(
        drop=True
    )
    if not feature_keys.equals(event_history.loc[:, ["station_id", "hour_utc"]]):
        raise RuntimeError("causal measurements and event history have different station-hour keys")
    target_info = {"fault": onset_fault_info, "outage": onset_outage_info}
    metric_rows: list[dict[str, object]] = []
    selection_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    feature_count_rows: list[dict[str, object]] = []
    feature_audit_frames: list[pd.DataFrame] = []
    compact_validation_frames: list[pd.DataFrame] = []
    manifest_validation_frames: list[pd.DataFrame] = []
    support_rows: list[dict[str, object]] = []
    split_digests: dict[str, object] = {}
    pending_tests: list[dict[str, object]] = []
    completed_configurations: list[str] = []
    model_directory = Path(output_dir) / "models"
    model_directory.mkdir(parents=True, exist_ok=True)
    timebox_reached = False

    for target in ["fault", "outage"]:
        if time.monotonic() >= deadline:
            timebox_reached = True
            break
        info = target_info[target]
        onset_dataset = info["dataset"]
        characteristics = info["characteristics"]
        if not hasattr(onset_dataset, "frame") or not isinstance(characteristics, dict):
            raise TypeError("onset target construction returned an invalid dataset")
        one_step_dataset = build_incident_hazard_dataset(
            target,
            horizons=(1,),
            recovery_exclusion_hours=ONSET_RECOVERY_EXCLUSION_HOURS,
        )
        if not onset_dataset.frame.loc[:, ["station_id", "hour_utc"]].equals(
            one_step_dataset.frame.loc[:, ["station_id", "hour_utc"]]
        ):
            raise RuntimeError("one-hour hazard target grid differs from onset target grid")
        if not feature_keys.equals(one_step_dataset.frame.loc[:, ["station_id", "hour_utc"]]):
            raise RuntimeError("one-hour hazard target grid differs from causal feature keys")
        compact_bundle = build_discrete_hazard_features(
            feature_bundle,
            event_history,
            target=target,
            registry=causal_sources["registry"],
        )
        if not feature_keys.equals(compact_bundle.frame.loc[:, ["station_id", "hour_utc"]]):
            raise RuntimeError("compact hazard feature keys differ from causal feature keys")
        feature_audit = compact_bundle.causality_audit.copy()
        feature_audit["target"] = target
        feature_audit_frames.append(feature_audit)
        compact_validation = _hazard_compact_feature_validation(
            compact_bundle.causality_audit,
            causal_future_validation,
            history_future_validation,
        )
        compact_validation["target"] = target
        compact_validation_frames.append(compact_validation)
        feature_count_rows.append(
            {
                "target": target,
                "numeric_feature_count": len(compact_bundle.numeric_feature_columns),
                "station_indicator_count": len(compact_bundle.station_indicator_columns),
                "model_feature_count": len(compact_bundle.model_feature_columns),
            }
        )
        one_step_labels = one_step_dataset.for_horizon(1)
        full_hazard_base_rate = float(one_step_labels["y"].mean())
        splits = characteristics["splits"]
        if not isinstance(splits, dict):
            raise TypeError("onset target characteristics lack split definitions")
        attached_by_horizon: dict[int, dict[str, pd.DataFrame]] = {}
        target_split_digests: dict[str, dict[str, str]] = {}
        for horizon in HORIZONS:
            split = splits[int(horizon)]
            if not isinstance(split, dict):
                raise TypeError("onset target split must be a dictionary")
            manifest_key = f"{target}_{int(horizon)}h"
            target_split_digests[manifest_key] = _risk_split_digest(split)
            split_digests[manifest_key] = target_split_digests[manifest_key]
            attached_by_horizon[int(horizon)] = _attach_discrete_hazard_features(
                split,
                one_step_labels=one_step_labels,
                hazard_features=compact_bundle.frame,
                feature_columns=compact_bundle.model_feature_columns,
            )
        manifest_validation_frames.append(
            _verify_saved_onset_split_manifest(target_split_digests)
        )
        strict_train = attached_by_horizon[24]["train"].copy()
        strict_validation = attached_by_horizon[24]["validation"].copy()
        for partition, frame in {"train": strict_train, "validation": strict_validation}.items():
            support_rows.append(
                {
                    "target": target,
                    "partition": f"h24_safe_{partition}",
                    "horizon_h": 1,
                    "n_rows": int(len(frame)),
                    "n_positive": int(frame["hazard_y_1h"].sum()),
                    "hourly_hazard_base_rate": float(frame["hazard_y_1h"].mean()),
                    "independent_qualifying_incidents": int(frame.loc[
                        frame["hazard_y_1h"].eq(1), "future_event_id"
                    ].fillna("").astype(str).replace("", np.nan).nunique()),
                    "role": "shared_model_fit" if partition == "train" else "shared_probability_calibration",
                }
            )
        support_rows.append(
            {
                "target": target,
                "partition": "all_one_hour_eligible",
                "horizon_h": 1,
                "n_rows": int(len(one_step_labels)),
                "n_positive": int(one_step_labels["y"].sum()),
                "hourly_hazard_base_rate": full_hazard_base_rate,
                "independent_qualifying_incidents": int(one_step_labels.loc[
                    one_step_labels["y"].eq(1), "future_event_id"
                ].fillna("").astype(str).replace("", np.nan).nunique()),
                "role": "full_onset_risk_set",
            }
        )
        fit_train = strict_train.copy()
        fit_train["y"] = fit_train["hazard_y_1h"].astype(int)
        calibration = strict_validation.copy()
        calibration["y"] = calibration["hazard_y_1h"].astype(int)
        split_digests[f"{target}_one_hour_h24_train"] = _hazard_partition_digest(
            fit_train, "y"
        )
        split_digests[f"{target}_one_hour_h24_validation"] = _hazard_partition_digest(
            calibration, "y"
        )
        for method in DISCRETE_HAZARD_METHODS:
            if time.monotonic() >= deadline:
                timebox_reached = True
                break
            try:
                fit = fit_discrete_hazard_model(
                    method,
                    fit_train,
                    compact_bundle.model_feature_columns,
                    deadline_monotonic=deadline,
                )
            except ForecastTrainingTimeboxReached:
                timebox_reached = True
                break
            if time.monotonic() >= deadline:
                timebox_reached = True
                break
            raw_calibration_probability = discrete_hazard_probability(fit, calibration)
            calibrator = fit_hazard_probability_calibrator(
                calibration["y"].to_numpy(dtype=int),
                raw_calibration_probability,
            )
            thresholds: dict[int, float] = {}
            validation_metrics_by_horizon: dict[int, dict[str, float]] = {}
            for horizon in HORIZONS:
                validation = attached_by_horizon[int(horizon)]["validation"]
                calibrated_hourly = calibrate_hazard_probability(
                    calibrator,
                    discrete_hazard_probability(fit, validation),
                )
                horizon_probability = cumulate_stationary_hazard(
                    calibrated_hourly,
                    int(horizon),
                )
                threshold, validation_metrics, trace = select_discrete_hazard_threshold(
                    validation["y"].to_numpy(dtype=int),
                    horizon_probability,
                )
                trace["target"] = target
                trace["method"] = method
                trace["horizon_h"] = int(horizon)
                trace["base_hazard_horizon_h"] = 1
                trace["cumulation_assumption"] = "hold_score_time_covariates_fixed"
                trace["calibration_method"] = calibrator.method
                trace["selection_rule"] = "maximize_validation_min_precision_recall_f1"
                trace["test_metrics_accessed_during_selection"] = False
                selection_frames.append(trace)
                thresholds[int(horizon)] = threshold
                validation_metrics_by_horizon[int(horizon)] = validation_metrics
                pending_tests.append(
                    {
                        "target": target,
                        "method": method,
                        "horizon": int(horizon),
                        "fit": fit,
                        "calibrator": calibrator,
                        "threshold": threshold,
                        "validation_metrics": validation_metrics,
                        "test": attached_by_horizon[int(horizon)]["test"].copy(),
                        "threshold_validation_rows": len(validation),
                        "threshold_validation_positive": int(validation["y"].sum()),
                        "fit_train_rows": len(fit_train),
                        "fit_train_positive": int(fit_train["y"].sum()),
                        "calibration_rows": len(calibration),
                        "calibration_positive": int(calibration["y"].sum()),
                        "model_feature_count": len(compact_bundle.model_feature_columns),
                    }
                )
            joblib.dump(
                {
                    "model": fit.model,
                    "method": fit.method,
                    "target": target,
                    "model_family": "discrete_time_hazard",
                    "base_hazard_horizon_h": 1,
                    "cumulation_assumption": "hold_score_time_covariates_fixed",
                    "feature_columns": list(fit.feature_columns),
                    "positive_class_weight": fit.positive_class_weight,
                    "calibrator": calibrator,
                    "calibration_method": calibrator.method,
                    "validation_selected_thresholds": thresholds,
                    "validation_metrics_by_horizon": validation_metrics_by_horizon,
                    "split_key_digests": {
                        key: value
                        for key, value in split_digests.items()
                        if key.startswith(f"{target}_")
                    },
                },
                model_directory / f"{target}_{method}.joblib",
            )
            completed_configurations.append(f"{target}_{method}")
        if timebox_reached:
            break

    for pending in pending_tests:
        if time.monotonic() >= deadline:
            timebox_reached = True
            break
        test = pending["test"]
        fit = pending["fit"]
        calibrator = pending["calibrator"]
        horizon = int(pending["horizon"])
        calibrated_hourly = calibrate_hazard_probability(
            calibrator,
            discrete_hazard_probability(fit, test),
        )
        horizon_probability = cumulate_stationary_hazard(calibrated_hourly, horizon)
        threshold = float(pending["threshold"])
        test_metrics = forecast_classification_metrics(
            test["y"].to_numpy(dtype=int), horizon_probability, threshold
        )
        row = _forecast_master_row(
            target=str(pending["target"]),
            horizon=horizon,
            predictor=str(pending["method"]),
            validation_metrics=pending["validation_metrics"],
            test_metrics=test_metrics,
            train_rows=int(pending["fit_train_rows"]),
            validation_rows=int(pending["threshold_validation_rows"]),
            test_rows=int(len(test)),
            train_positive=int(pending["fit_train_positive"]),
            validation_positive=int(pending["threshold_validation_positive"]),
            test_positive=int(test["y"].sum()),
            selected_positive_class_weight=float(fit.positive_class_weight),
            selected_threshold=threshold,
        )
        row.update(
            {
                "model_family": "discrete_time_hazard",
                "base_hazard_horizon_h": 1,
                "cumulation_assumption": "hold_score_time_covariates_fixed",
                "calibration_method": calibrator.method,
                "calibration_validation_rows": int(pending["calibration_rows"]),
                "calibration_validation_positive": int(pending["calibration_positive"]),
                "model_feature_count": int(pending["model_feature_count"]),
                "test_brier_score": float(
                    np.mean((horizon_probability - test["y"].to_numpy(dtype=float)) ** 2)
                ),
                "test_metrics_accessed_during_selection": False,
                "deployment_status": (
                    "retrospective_only_ground_truth_fault_history"
                    if str(pending["target"]) == "fault"
                    else "retrospective_outage_history_observable_after_event"
                ),
            }
        )
        metric_rows.append(row)
        predictions = _forecast_prediction_rows(
            test,
            target=str(pending["target"]),
            horizon=horizon,
            predictor=str(pending["method"]),
            probability=horizon_probability,
            threshold=threshold,
        )
        predictions["model_family"] = "discrete_time_hazard"
        predictions["hazard_probability_1h"] = calibrated_hourly
        predictions["horizon_probability"] = horizon_probability
        predictions["cumulation_assumption"] = "hold_score_time_covariates_fixed"
        prediction_frames.append(predictions)
        calibration_frames.append(
            _hazard_calibration_rows(
                target=str(pending["target"]),
                method=str(pending["method"]),
                horizon=horizon,
                truth=test["y"].to_numpy(dtype=int),
                probability=horizon_probability,
            )
        )

    after_outage = build_risk_dataset()
    after_fault = build_fault_risk_dataset()
    after = _forecast_invariant_snapshot(events_path, after_outage, after_fault)
    if before != after:
        raise RuntimeError("discrete-hazard training changed an upstream artifact")
    metrics = pd.DataFrame(metric_rows)
    elapsed_seconds = float(time.monotonic() - started)
    planned_test_configurations = len(DISCRETE_HAZARD_METHODS) * 2 * len(HORIZONS)
    completed_test_configurations = len(metric_rows)
    run_status = pd.DataFrame(
        [
            {
                "timebox_seconds": int(timebox_seconds),
                "elapsed_seconds": elapsed_seconds,
                "timebox_reached": bool(timebox_reached or elapsed_seconds >= timebox_seconds),
                "model_fits_completed": len(completed_configurations),
                "model_fits_planned": len(DISCRETE_HAZARD_METHODS) * 2,
                "test_configurations_completed": completed_test_configurations,
                "test_configurations_planned": planned_test_configurations,
                "run_status": (
                    "complete"
                    if completed_test_configurations == planned_test_configurations
                    else "timeboxed_partial"
                ),
                "timebox_contract": (
                    "best_effort_cap_checked before and after each fit and before each "
                    "test evaluation; completed work is retained without rerunning tests"
                ),
            }
        ]
    )
    deployment_scope = pd.DataFrame(
        [
            {
                "target": "fault",
                "deployment_status": "retrospective_only_ground_truth_fault_history",
                "interpretation": (
                    "Prior fault-label history is chronologically prior but is not a live "
                    "dashboard input until an all-hour causal detector ledger exists."
                ),
            },
            {
                "target": "outage",
                "deployment_status": "retrospective_outage_history_observable_after_event",
                "interpretation": (
                    "Prior outage history is observable after an outage has been detected and "
                    "closed, but this experiment remains retrospective."
                ),
            },
        ]
    )
    direct_onset_comparison = (
        _hazard_vs_direct_onset(metrics)
        if completed_test_configurations == planned_test_configurations
        else pd.DataFrame()
    )
    return {
        "metrics": metrics,
        "selection_trace": pd.concat(selection_frames, ignore_index=True)
        if selection_frames
        else pd.DataFrame(),
        "predictions": pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(),
        "calibration": pd.concat(calibration_frames, ignore_index=True)
        if calibration_frames
        else pd.DataFrame(),
        "feature_counts": pd.DataFrame(feature_count_rows),
        "feature_audit": pd.concat(feature_audit_frames, ignore_index=True)
        if feature_audit_frames
        else pd.DataFrame(),
        "feature_future_validation": pd.concat(compact_validation_frames, ignore_index=True)
        if compact_validation_frames
        else pd.DataFrame(),
        "manifest_validation": pd.concat(manifest_validation_frames, ignore_index=True)
        if manifest_validation_frames
        else pd.DataFrame(),
        "hazard_support": pd.DataFrame(support_rows),
        "independent_onset_support": _independent_onset_support(target_info),
        "onset_construction": pd.concat(
            [
                onset_fault_info["dataset"].construction_summary(),
                onset_outage_info["dataset"].construction_summary(),
            ],
            ignore_index=True,
        ).sort_values(["target", "horizon_h"], kind="mergesort").reset_index(drop=True),
        "onset_eligibility": _onset_eligibility_audit(target_info),
        "network_event_policy": _onset_network_event_policy(onset_outage_info["dataset"]),
        "direct_onset_comparison": direct_onset_comparison,
        "deployment_scope": deployment_scope,
        "run_status": run_status,
        "future_validation": future_validation,
        "future_validation_summary": future_validation_summary,
        "split_digests": split_digests,
        "invariants": {"before": before, "after": after},
        "completed_configurations": completed_configurations,
        "timebox_reached": bool(timebox_reached or elapsed_seconds >= timebox_seconds),
        "elapsed_seconds": elapsed_seconds,
    }


def write_discrete_hazard_outputs(
    result: dict[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": destination / "discrete_hazard_metrics.csv",
        "selection_trace": destination / "discrete_hazard_selection_trace.csv",
        "predictions": destination / "discrete_hazard_predictions.parquet",
        "calibration": destination / "discrete_hazard_test_calibration.csv",
        "feature_counts": destination / "discrete_hazard_feature_counts.csv",
        "feature_audit": destination / "discrete_hazard_feature_audit.csv",
        "feature_future_validation": destination / "discrete_hazard_feature_future_validation.csv",
        "manifest_validation": destination / "discrete_hazard_manifest_validation.csv",
        "hazard_support": destination / "discrete_hazard_one_hour_support.csv",
        "independent_onset_support": destination / "discrete_hazard_independent_onset_support.csv",
        "onset_construction": destination / "discrete_hazard_onset_construction.csv",
        "onset_eligibility": destination / "discrete_hazard_onset_eligibility.csv",
        "network_event_policy": destination / "discrete_hazard_network_event_policy.csv",
        "direct_onset_comparison": destination / "discrete_hazard_vs_direct_onset.csv",
        "deployment_scope": destination / "discrete_hazard_deployment_scope.csv",
        "run_status": destination / "discrete_hazard_run_status.csv",
        "future_validation": destination / "discrete_hazard_source_delete_future_validation.csv",
        "future_validation_summary": destination / "discrete_hazard_source_delete_future_summary.csv",
        "split_digests": destination / "discrete_hazard_split_digests.json",
        "invariants": destination / "discrete_hazard_invariant_hashes.json",
        "report": destination / "discrete_hazard_report.txt",
    }
    for name in [
        "metrics",
        "selection_trace",
        "calibration",
        "feature_counts",
        "feature_audit",
        "feature_future_validation",
        "manifest_validation",
        "hazard_support",
        "independent_onset_support",
        "onset_construction",
        "onset_eligibility",
        "network_event_policy",
        "direct_onset_comparison",
        "deployment_scope",
        "run_status",
        "future_validation",
        "future_validation_summary",
    ]:
        frame = result[name]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"discrete-hazard {name} must be a DataFrame")
        frame.to_csv(paths[name], index=False)
    predictions = result["predictions"]
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("discrete-hazard predictions must be a DataFrame")
    predictions.to_parquet(paths["predictions"], index=False)
    paths["split_digests"].write_text(
        json.dumps(result["split_digests"], indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    paths["invariants"].write_text(
        json.dumps(result["invariants"], indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    paths["report"].write_text(_discrete_hazard_report(result), encoding="utf-8")
    return paths


def _forecast_threshold_reselection_source_paths(source_dir: Path) -> dict[str, Path]:
    source = Path(source_dir)
    return {
        "metrics": source / FORECAST_METRICS_PATH.name,
        "selection_trace": source / FORECAST_SELECTION_PATH.name,
        "test_predictions": source / FORECAST_PREDICTIONS_PATH.name,
        "split_digests": source / FORECAST_DIGESTS_PATH.name,
        "invariants": source / FORECAST_INVARIANTS_PATH.name,
    }


def _forecast_threshold_reselection_model_path(
    source_dir: Path,
    target: str,
    horizon: int,
) -> Path:
    return Path(source_dir) / FORECAST_MODELS_DIR.name / f"{target}_risk_{int(horizon)}h.joblib"


def _forecast_threshold_reselection_source_lock(source_dir: Path) -> dict[str, object]:
    paths = _forecast_threshold_reselection_source_paths(source_dir)
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    model_paths = {
        f"{target}_{int(horizon)}h": _forecast_threshold_reselection_model_path(
            source_dir,
            target,
            int(horizon),
        )
        for target in ("fault", "outage")
        for horizon in HORIZONS
    }
    missing.extend(
        f"model {name}: {path}"
        for name, path in model_paths.items()
        if not path.is_file()
    )
    if missing:
        raise FileNotFoundError(
            "threshold reselection requires frozen Experiment 1 artifacts:\n"
            + "\n".join(missing)
        )
    return {
        "source_experiment_commit": THRESHOLD_RESELECTION_SOURCE_COMMIT,
        "source_dir": str(Path(source_dir)),
        "source_artifact_sha256": {
            name: _file_sha256(path) for name, path in paths.items()
        },
        "model_artifact_sha256": {
            name: _file_sha256(path) for name, path in model_paths.items()
        },
    }


def _forecast_threshold_selection_metadata(
    source_dir: Path,
    selection_metrics: pd.DataFrame,
    selection_trace: pd.DataFrame,
    split_digests: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    required_metrics = {
        "target",
        "horizon_h",
        "predictor",
        "train_rows",
        "validation_rows",
        "test_rows",
        "train_positive",
        "validation_positive",
        "test_positive",
        "selected_positive_class_weight",
        "selected_weight_multiplier",
        "selected_threshold",
    }
    required_trace = {
        "target",
        "horizon_h",
        "weight_multiplier",
        "positive_class_weight",
        "threshold",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_accuracy",
        "validation_maximin_prf",
        "selected",
    }
    missing_metrics = sorted(required_metrics.difference(selection_metrics.columns))
    missing_trace = sorted(required_trace.difference(selection_trace.columns))
    if missing_metrics:
        raise KeyError(f"frozen forecast metrics are missing columns: {missing_metrics}")
    if missing_trace:
        raise KeyError(f"frozen forecast selection trace is missing columns: {missing_trace}")

    models = selection_metrics.loc[
        selection_metrics["predictor"].eq(FORECAST_MODEL_NAME)
    ].copy()
    expected_configurations = {
        (target, int(horizon)) for target in ("fault", "outage") for horizon in HORIZONS
    }
    observed_configurations = {
        (str(row.target), int(row.horizon_h)) for row in models.itertuples(index=False)
    }
    if observed_configurations != expected_configurations or len(models) != len(expected_configurations):
        raise RuntimeError("frozen forecast metrics do not contain exactly six model configurations")

    decision_rows: list[dict[str, object]] = []
    tradeoff_frames: list[pd.DataFrame] = []
    model_metadata: list[dict[str, object]] = []
    rule_descriptions = {
        "maximin": "maximise validation minimum of precision, recall, and F1",
        "max_f1": "maximise validation F1",
        "max_recall_precision_floor": (
            f"maximise validation recall subject to validation precision >= "
            f"{THRESHOLD_RESELECTION_PRECISION_FLOOR:.2f}"
        ),
    }

    for source_row in models.sort_values(["target", "horizon_h"], kind="mergesort").itertuples(index=False):
        target = str(source_row.target)
        horizon = int(source_row.horizon_h)
        configuration_key = f"{target}_{horizon}h"
        frozen_weight = float(source_row.selected_weight_multiplier)
        trace = selection_trace.loc[
            selection_trace["target"].eq(target)
            & selection_trace["horizon_h"].astype(int).eq(horizon)
            & np.isclose(selection_trace["weight_multiplier"].astype(float), frozen_weight)
        ].copy()
        trace = trace.sort_values("threshold", kind="mergesort").reset_index(drop=True)
        observed_thresholds = tuple(trace["threshold"].astype(float).round(2))
        if observed_thresholds != FORECAST_THRESHOLDS:
            raise RuntimeError(
                f"{configuration_key} does not contain the frozen 19-threshold validation grid"
            )
        original = trace.loc[trace["selected"].astype(bool)].copy()
        if len(original) != 1:
            raise RuntimeError(
                f"{configuration_key} does not have one selected frozen validation threshold"
            )
        original = original.iloc[0]
        if not np.isclose(float(original["threshold"]), float(source_row.selected_threshold)):
            raise RuntimeError(
                f"{configuration_key} selected threshold differs between frozen metrics and trace"
            )
        maximin = select_validation_threshold_rule(trace, "maximin")
        if maximin is None or not np.isclose(
            float(maximin["threshold"]), float(original["threshold"])
        ):
            raise RuntimeError(
                f"{configuration_key} cannot reproduce its frozen maximin threshold"
            )

        model_path = _forecast_threshold_reselection_model_path(source_dir, target, horizon)
        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict):
            raise TypeError(f"{model_path.name} is not a forecast model bundle")
        expected_metadata = {
            "model_name": FORECAST_MODEL_NAME,
            "target": target,
            "horizon_h": horizon,
        }
        for key, expected in expected_metadata.items():
            if bundle.get(key) != expected:
                raise RuntimeError(
                    f"{model_path.name} has unexpected {key}: {bundle.get(key)!r}"
                )
        if not np.isclose(float(bundle["weight_multiplier"]), frozen_weight):
            raise RuntimeError(f"{model_path.name} has a different frozen class-weight multiplier")
        if not np.isclose(float(bundle["threshold"]), float(source_row.selected_threshold)):
            raise RuntimeError(f"{model_path.name} has a different frozen threshold")
        bundle_digests = bundle.get("split_key_digests")
        if bundle_digests != split_digests.get(configuration_key):
            raise RuntimeError(f"{model_path.name} has different split-key digests")
        model_metadata.append(
            {
                "configuration": configuration_key,
                "model_path": str(model_path),
                "model_name": str(bundle["model_name"]),
                "frozen_weight_multiplier": float(bundle["weight_multiplier"]),
                "frozen_threshold": float(bundle["threshold"]),
                "split_key_digests_match": True,
            }
        )

        selected_by_rule: dict[str, pd.Series | None] = {}
        for rule in rule_descriptions:
            selected = select_validation_threshold_rule(
                trace,
                rule,
                precision_floor=THRESHOLD_RESELECTION_PRECISION_FLOOR,
            )
            selected_by_rule[rule] = selected
            row: dict[str, object] = {
                "target": target,
                "horizon_h": horizon,
                "predictor": FORECAST_MODEL_NAME,
                "selection_rule": rule,
                "selection_rule_description": rule_descriptions[rule],
                "rule_feasible": selected is not None,
                "precision_floor": THRESHOLD_RESELECTION_PRECISION_FLOOR,
                "source_experiment_commit": THRESHOLD_RESELECTION_SOURCE_COMMIT,
                "frozen_weight_multiplier": frozen_weight,
                "frozen_positive_class_weight": float(
                    source_row.selected_positive_class_weight
                ),
                "source_maximin_threshold": float(source_row.selected_threshold),
                "train_rows": int(source_row.train_rows),
                "validation_rows": int(source_row.validation_rows),
                "test_rows": int(source_row.test_rows),
                "train_positive": int(source_row.train_positive),
                "validation_positive": int(source_row.validation_positive),
                "test_positive": int(source_row.test_positive),
                "test_metrics_accessed_during_selection": False,
                "test_evaluation_count": 0,
                "test_evaluated_once": False,
            }
            if selected is None:
                row["selection_status"] = "precision_floor_infeasible"
                for prefix in ("validation", "test"):
                    for metric in ("precision", "recall", "f1", "accuracy", "maximin_prf"):
                        row[f"{prefix}_{metric}"] = np.nan
                row["selected_threshold"] = np.nan
                row["test_all_precision_recall_f1_ge_080"] = False
            else:
                row["selection_status"] = "selected_from_validation"
                row["selected_threshold"] = float(selected["threshold"])
                for metric in ("precision", "recall", "f1", "accuracy", "maximin_prf"):
                    row[f"validation_{metric}"] = float(selected[f"validation_{metric}"])
            decision_rows.append(row)

        tradeoff = trace.loc[
            :,
            [
                "threshold",
                "validation_precision",
                "validation_recall",
                "validation_f1",
                "validation_accuracy",
                "validation_maximin_prf",
            ],
        ].copy()
        tradeoff.insert(0, "horizon_h", horizon)
        tradeoff.insert(0, "target", target)
        tradeoff.insert(2, "frozen_weight_multiplier", frozen_weight)
        for rule, selected in selected_by_rule.items():
            tradeoff[f"selected_by_{rule}"] = (
                False
                if selected is None
                else np.isclose(
                    tradeoff["threshold"].astype(float), float(selected["threshold"])
                )
            )
        tradeoff_frames.append(tradeoff)

    decisions = pd.DataFrame(decision_rows).sort_values(
        ["target", "horizon_h", "selection_rule"], kind="mergesort"
    ).reset_index(drop=True)
    tradeoffs = pd.concat(tradeoff_frames, ignore_index=True).sort_values(
        ["target", "horizon_h", "threshold"], kind="mergesort"
    ).reset_index(drop=True)
    return decisions, tradeoffs, model_metadata


def reselect_forecast_risk_thresholds(
    source_dir: Path = FORECAST_OUTPUT_DIR,
) -> dict[str, object]:
    source = Path(source_dir)
    before = _forecast_threshold_reselection_source_lock(source)
    selection_columns = {
        "target",
        "horizon_h",
        "predictor",
        "train_rows",
        "validation_rows",
        "test_rows",
        "train_positive",
        "validation_positive",
        "test_positive",
        "selected_positive_class_weight",
        "selected_weight_multiplier",
        "selected_threshold",
    }
    source_paths = _forecast_threshold_reselection_source_paths(source)
    validation_metrics = pd.read_csv(
        source_paths["metrics"],
        usecols=lambda column: column in selection_columns,
    )
    validation_trace = pd.read_csv(source_paths["selection_trace"])
    split_digests = json.loads(source_paths["split_digests"].read_text(encoding="utf-8"))
    decisions, tradeoffs, model_metadata = _forecast_threshold_selection_metadata(
        source,
        validation_metrics,
        validation_trace,
        split_digests,
    )

    source_predictions = pd.read_parquet(source_paths["test_predictions"])
    required_predictions = {
        "target",
        "horizon_h",
        "predictor",
        "probability",
        "threshold",
        "y",
        "station_id",
        "hour_utc",
        "label_end_utc",
    }
    missing_predictions = sorted(required_predictions.difference(source_predictions.columns))
    if missing_predictions:
        raise KeyError(f"frozen forecast predictions are missing columns: {missing_predictions}")

    source_metrics = pd.read_csv(source_paths["metrics"])
    output_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for decision in decisions.to_dict(orient="records"):
        target = str(decision["target"])
        horizon = int(decision["horizon_h"])
        source_model_row = source_metrics.loc[
            source_metrics["target"].eq(target)
            & source_metrics["horizon_h"].astype(int).eq(horizon)
            & source_metrics["predictor"].eq(FORECAST_MODEL_NAME)
        ]
        if len(source_model_row) != 1:
            raise RuntimeError(f"frozen metrics lack one model row for {target}_{horizon}h")
        source_model_row = source_model_row.iloc[0]
        test = source_predictions.loc[
            source_predictions["target"].eq(target)
            & source_predictions["horizon_h"].astype(int).eq(horizon)
            & source_predictions["predictor"].eq(FORECAST_MODEL_NAME)
        ].copy()
        if len(test) != int(decision["test_rows"]):
            raise RuntimeError(f"frozen test-prediction row count differs for {target}_{horizon}h")
        if not np.isclose(
            float(test["threshold"].iloc[0]), float(decision["source_maximin_threshold"])
        ) or not test["threshold"].astype(float).eq(float(test["threshold"].iloc[0])).all():
            raise RuntimeError(f"frozen test predictions have an inconsistent threshold for {target}_{horizon}h")

        row = dict(decision)
        if bool(row["rule_feasible"]):
            threshold = float(row["selected_threshold"])
            test_metrics = forecast_classification_metrics(
                test["y"].to_numpy(dtype=int),
                test["probability"].to_numpy(dtype=float),
                threshold,
            )
            for metric, value in test_metrics.items():
                row[f"test_{metric}"] = float(value)
            row["test_evaluation_count"] = 1
            row["test_evaluated_once"] = True
            row["test_all_precision_recall_f1_ge_080"] = bool(
                min(
                    float(test_metrics["precision"]),
                    float(test_metrics["recall"]),
                    float(test_metrics["f1"]),
                )
                >= 0.80
            )
            output = test.loc[
                :, ["station_id", "hour_utc", "label_end_utc", "y", "target", "horizon_h", "probability"]
            ].copy()
            output["predictor"] = FORECAST_MODEL_NAME
            output["selection_rule"] = str(row["selection_rule"])
            output["selected_threshold"] = threshold
            output["prediction"] = output["probability"].ge(threshold).astype(int)
            prediction_frames.append(output)
            if str(row["selection_rule"]) == "maximin":
                for metric in ("precision", "recall", "f1", "accuracy"):
                    if not np.isclose(
                        float(row[f"test_{metric}"]),
                        float(source_model_row[f"test_{metric}"]),
                    ):
                        raise RuntimeError(
                            f"maximin reselection does not reproduce frozen {metric} for "
                            f"{target}_{horizon}h"
                        )
        output_rows.append(row)

    metrics = pd.DataFrame(output_rows).sort_values(
        ["target", "horizon_h", "selection_rule"], kind="mergesort"
    ).reset_index(drop=True)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(
            columns=[
                "station_id",
                "hour_utc",
                "label_end_utc",
                "y",
                "target",
                "horizon_h",
                "probability",
                "predictor",
                "selection_rule",
                "selected_threshold",
                "prediction",
            ]
        )
    )
    baseline_snapshot = source_metrics.loc[
        source_metrics["predictor"].isin(["base_rate", "persistence"])
    ].copy()
    expected_baselines = len(HORIZONS) * 2 * 2
    if len(baseline_snapshot) != expected_baselines:
        raise RuntimeError("frozen continuation-risk baseline rows are incomplete")
    baseline_snapshot["source_metrics_sha256"] = before["source_artifact_sha256"]["metrics"]
    baseline_snapshot["copied_without_reevaluation"] = True

    after = _forecast_threshold_reselection_source_lock(source)
    if before != after:
        raise RuntimeError("threshold reselection changed a frozen Experiment 1 artifact")
    comparison = metrics.loc[
        metrics["rule_feasible"].astype(bool),
        ["target", "horizon_h", "selection_rule", "test_f1"],
    ].pivot(
        index=["target", "horizon_h"],
        columns="selection_rule",
        values="test_f1",
    ).reset_index()
    comparison.columns.name = None
    infeasible = metrics.loc[
        ~metrics["rule_feasible"].astype(bool),
        ["target", "horizon_h", "selection_rule", "selection_status"],
    ].copy()
    return {
        "metrics": metrics,
        "validation_tradeoff": tradeoffs,
        "predictions": predictions,
        "baselines": baseline_snapshot.sort_values(
            ["target", "horizon_h", "predictor"], kind="mergesort"
        ).reset_index(drop=True),
        "test_f1_comparison": comparison.sort_values(
            ["target", "horizon_h"], kind="mergesort"
        ).reset_index(drop=True),
        "precision_floor_infeasible": infeasible,
        "source_lock": {
            "before": before,
            "after": after,
            "unchanged": before == after,
            "models_loaded_read_only": True,
            "model_metadata": model_metadata,
            "validation_selection_complete_before_test_prediction_load": True,
            "class_weight_sweep": "skipped_requires_retraining",
            "source_validation_prediction_artifact": (
                "not saved; the frozen validation selection trace supplies the "
                "19-threshold metrics for each frozen selected model"
            ),
        },
    }


def _forecast_threshold_reselection_report(result: dict[str, object]) -> str:
    metrics = result["metrics"]
    tradeoffs = result["validation_tradeoff"]
    baselines = result["baselines"]
    comparison = result["test_f1_comparison"]
    infeasible = result["precision_floor_infeasible"]
    source_lock = result["source_lock"]
    if not all(
        isinstance(frame, pd.DataFrame)
        for frame in [metrics, tradeoffs, baselines, comparison, infeasible]
    ) or not isinstance(source_lock, dict):
        raise TypeError("threshold reselection report inputs are invalid")
    passing = metrics.loc[
        metrics["rule_feasible"].astype(bool)
        & metrics["test_all_precision_recall_f1_ge_080"].astype(bool),
        ["target", "horizon_h", "selection_rule"],
    ]
    failing = metrics.loc[
        metrics["rule_feasible"].astype(bool)
        & ~metrics["test_all_precision_recall_f1_ge_080"].astype(bool),
        ["target", "horizon_h", "selection_rule", "test_precision", "test_recall", "test_f1"],
    ]
    lines = [
        "FORECAST-RISK THRESHOLD RESELECTION EXPERIMENT",
        "",
        f"source_experiment_commit={THRESHOLD_RESELECTION_SOURCE_COMMIT}",
        "Scope: threshold-only post-processing of the saved continuation-risk experiment.",
        "Each configuration keeps its originally saved class-weight model fixed.",
        "Class-weight sweeping is skipped because it would require retraining and the non-selected model objects were not saved.",
        "All threshold choices are derived from the frozen validation trace before test probabilities are opened.",
        "No rule is named a winner from held-out test results.",
        "",
        "SOURCE LOCK AND NO-RETRAINING VERIFICATION",
        json.dumps(source_lock, indent=2, sort_keys=True, default=str),
        "",
        "VALIDATION-SELECTED THRESHOLDS AND ONE-TIME HELD-OUT TEST RESULTS",
        _format_frame(metrics),
        "",
        "TEST F1 SIDE-BY-SIDE BY PREDECLARED VALIDATION RULE",
        _format_frame(comparison),
        "",
        "UNCHANGED BASELINES COPIED FROM THE FROZEN EXPERIMENT",
        _format_frame(baselines),
        "",
        "VALIDATION PRECISION-RECALL-F1 TRADEOFFS AT THE FIXED 0.05 TO 0.95 GRID",
        _format_frame(tradeoffs),
        "",
        "PRECISION-FLOOR RULES WITH NO VALIDATION OPERATING POINT",
        _format_frame(infeasible),
        "",
        "CONFIGURATIONS WITH HELD-OUT PRECISION, RECALL, AND F1 ALL AT LEAST 0.80",
        _format_frame(passing),
        "",
        "CONFIGURATIONS BELOW THE 0.80 THREE-METRIC CRITERION",
        _format_frame(failing),
        "",
        "INTERPRETATION",
        "The test table is descriptive. It does not select a threshold rule, a class weight, or a deployment configuration.",
        "The maximin row must reproduce the frozen Experiment 1 HGB result; other rows are alternative validation-only operating points for the same frozen model.",
    ]
    return "\n".join(lines) + "\n"


def write_forecast_threshold_reselection_outputs(
    result: dict[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": destination / THRESHOLD_RESELECTION_METRICS_NAME,
        "validation_tradeoff": destination / THRESHOLD_RESELECTION_TRADEOFF_NAME,
        "predictions": destination / THRESHOLD_RESELECTION_PREDICTIONS_NAME,
        "baselines": destination / THRESHOLD_RESELECTION_BASELINES_NAME,
        "source_lock": destination / THRESHOLD_RESELECTION_SOURCE_LOCK_NAME,
        "report": destination / THRESHOLD_RESELECTION_REPORT_NAME,
    }
    for name in ["metrics", "validation_tradeoff", "baselines"]:
        frame = result[name]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"threshold reselection {name} must be a DataFrame")
        frame.to_csv(paths[name], index=False)
    predictions = result["predictions"]
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("threshold reselection predictions must be a DataFrame")
    predictions.to_parquet(paths["predictions"], index=False)
    source_lock = result["source_lock"]
    if not isinstance(source_lock, dict):
        raise TypeError("threshold reselection source lock must be a dictionary")
    paths["source_lock"].write_text(
        json.dumps(source_lock, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    paths["report"].write_text(
        _forecast_threshold_reselection_report(result), encoding="utf-8"
    )
    return paths


def write_forecast_risk_outputs(
    result: dict[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "metrics": destination / FORECAST_METRICS_PATH.name,
        "confusion": destination / FORECAST_CONFUSION_PATH.name,
        "selection_trace": destination / FORECAST_SELECTION_PATH.name,
        "causality_audit": destination / FORECAST_AUDIT_PATH.name,
        "feature_set": destination / FORECAST_FEATURE_SET_PATH.name,
        "importance": destination / FORECAST_IMPORTANCE_PATH.name,
        "predictions": destination / FORECAST_PREDICTIONS_PATH.name,
        "split_digests": destination / FORECAST_DIGESTS_PATH.name,
        "invariants": destination / FORECAST_INVARIANTS_PATH.name,
        "report": destination / FORECAST_REPORT_PATH.name,
        "future_validation": destination / FORECAST_FUTURE_VALIDATION_PATH.name,
        "future_validation_summary": destination
        / FORECAST_FUTURE_VALIDATION_SUMMARY_PATH.name,
        "previous_comparison": destination / FORECAST_PREVIOUS_COMPARISON_PATH.name,
        "feature_counts": destination / FORECAST_FEATURE_COUNTS_PATH.name,
        "deployment_realistic_variant": destination
        / FORECAST_DEPLOYMENT_VARIANT_PATH.name,
        "persistence_comparison": destination
        / FORECAST_PERSISTENCE_COMPARISON_PATH.name,
    }
    csv_names = [
        "metrics",
        "confusion",
        "selection_trace",
        "causality_audit",
        "feature_set",
        "importance",
        "future_validation",
        "future_validation_summary",
        "previous_comparison",
        "feature_counts",
        "deployment_realistic_variant",
        "persistence_comparison",
    ]
    if result.get("risk_definition") == "onset":
        output_paths.update(
            {
                "onset_construction": destination / "onset_risk_construction.csv",
                "risk_set_comparison": destination / "onset_risk_set_comparison.csv",
                "independent_onset_support": destination
                / "onset_independent_incident_support.csv",
                "onset_eligibility": destination / "onset_eligibility_audit.csv",
                "network_event_policy": destination / "onset_network_event_policy.csv",
                "continuation_comparison": destination
                / "continuation_vs_onset.csv",
                "recurrence_comparison": destination
                / "onset_model_vs_recurrence.csv",
            }
        )
        csv_names.extend(
            [
                "onset_construction",
                "risk_set_comparison",
                "independent_onset_support",
                "onset_eligibility",
                "network_event_policy",
                "continuation_comparison",
                "recurrence_comparison",
            ]
        )
    for name in csv_names:
        frame = result[name]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"forecast {name} must be a DataFrame")
        frame.to_csv(output_paths[name], index=False)
    predictions = result["predictions"]
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("forecast predictions must be a DataFrame")
    predictions.to_parquet(output_paths["predictions"], index=False)
    output_paths["split_digests"].write_text(
        json.dumps(result["split_digests"], indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    output_paths["invariants"].write_text(
        json.dumps(result["invariants"], indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    output_paths["report"].write_text(_forecast_report(result), encoding="utf-8")
    return output_paths


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.reselect_forecast_thresholds:
        if args.run_models or args.train_risk_models or args.train_discrete_hazard:
            raise ValueError(
                "--reselect-forecast-thresholds cannot be combined with model training"
            )
        if args.target != "all":
            raise ValueError("--reselect-forecast-thresholds requires --target all")
        if args.risk_definition != "continuation":
            raise ValueError(
                "--reselect-forecast-thresholds only applies to the saved continuation-risk experiment"
            )
        source_paths = _forecast_threshold_reselection_source_paths(FORECAST_OUTPUT_DIR)
        require_files(
            "Frozen Experiment 1 threshold-reselection artifacts",
            {
                "forecast metrics": source_paths["metrics"],
                "forecast validation trace": source_paths["selection_trace"],
                "forecast test predictions": source_paths["test_predictions"],
                "forecast split digests": source_paths["split_digests"],
                "forecast invariant hashes": source_paths["invariants"],
            },
        )
        destination = (
            THRESHOLD_RESELECTION_OUTPUT_DIR
            if Path(args.output_dir) == OUTPUT_DIR
            else Path(args.output_dir)
        )
        result = reselect_forecast_risk_thresholds(FORECAST_OUTPUT_DIR)
        output_paths = write_forecast_threshold_reselection_outputs(result, destination)
        print(output_paths["report"].read_text(encoding="utf-8"), end="")
        for name, path in output_paths.items():
            print(f"{name}={path}")
        return

    if args.train_discrete_hazard:
        if args.run_models or args.train_risk_models:
            raise ValueError(
                "--train-discrete-hazard cannot be combined with --run-models or "
                "--train-risk-models"
            )
        if args.target != "all":
            raise ValueError("--train-discrete-hazard requires --target all")
        require_files(
            "Discrete-time fault- and outage-hazard comparison",
            {
                "hourly availability classification": AVAILABILITY_CLASSIFICATION_PATH,
                "availability events": args.events,
                "network outage windows": NETWORK_OUTAGE_WINDOWS_PATH,
                "partial outage events": PARTIAL_OUTAGE_EVENTS_PATH,
                "canonical merged dataset": MERGED_DATASET_PATH,
                "station registry": STATION_REGISTRY_PATH,
                "feature matrix for the audited causal schema": HOURLY_FEATURE_PATH,
                "live episode labels": EPISODE_LABEL_PATH,
                "hourly label export": HOURLY_LABEL_PATH,
                "short hourly tensor": SHORT_TENSOR_PATH,
                "long hourly tensor": LONG_TENSOR_PATH,
            },
        )
        destination = (
            DISCRETE_HAZARD_OUTPUT_DIR
            if Path(args.output_dir) == OUTPUT_DIR
            else Path(args.output_dir)
        )
        result = train_discrete_hazard_models(
            destination,
            events_path=Path(args.events),
        )
        output_paths = write_discrete_hazard_outputs(result, destination)
        report_path = output_paths["report"]
        print(report_path.read_text(encoding="utf-8"), end="")
        for name, path in output_paths.items():
            print(f"{name}={path}")
        return

    if args.train_risk_models:
        if args.run_models:
            raise ValueError("--train-risk-models cannot be combined with --run-models")
        if args.target != "all":
            raise ValueError("--train-risk-models requires --target all")
        require_files(
            "Causal fault- and outage-risk model training",
            {
                "hourly availability classification": AVAILABILITY_CLASSIFICATION_PATH,
                "availability events": args.events,
                "network outage windows": NETWORK_OUTAGE_WINDOWS_PATH,
                "partial outage events": PARTIAL_OUTAGE_EVENTS_PATH,
                "canonical merged dataset": MERGED_DATASET_PATH,
                "station registry": STATION_REGISTRY_PATH,
                "feature matrix for the audited legacy schema": HOURLY_FEATURE_PATH,
                "live episode labels": EPISODE_LABEL_PATH,
                "hourly label export": HOURLY_LABEL_PATH,
                "short hourly tensor": SHORT_TENSOR_PATH,
                "long hourly tensor": LONG_TENSOR_PATH,
            },
        )
        destination = (
            (
                ONSET_FORECAST_OUTPUT_DIR
                if args.risk_definition == "onset"
                else FORECAST_OUTPUT_DIR
            )
            if Path(args.output_dir) == OUTPUT_DIR
            else Path(args.output_dir)
        )
        result = train_forecast_risk_models(
            destination,
            events_path=Path(args.events),
            timebox_seconds=(
                ONSET_FORECAST_TIMEBOX_SECONDS
                if args.risk_definition == "onset"
                else FORECAST_TIMEBOX_SECONDS
            ),
            risk_definition=args.risk_definition,
        )
        output_paths = write_forecast_risk_outputs(result, destination)
        report_path = output_paths["report"]
        print(report_path.read_text(encoding="utf-8"), end="")
        for name, path in output_paths.items():
            print(f"{name}={path}")
        return

    if args.target == "all":
        raise ValueError(
            "--target all is only valid with --train-risk-models or "
            "--train-discrete-hazard or --reselect-forecast-thresholds"
        )
    if args.target == "fault":
        if args.run_models:
            raise ValueError("fault-risk model fitting is not part of this label-only run")
        require_files(
            "Fault-risk label and split construction",
            {
                "live episode labels": EPISODE_LABEL_PATH,
                "canonical hourly source": HOURLY_SOURCE_PATH,
                "feature matrix": HOURLY_FEATURE_PATH,
                "hourly label export": HOURLY_LABEL_PATH,
                "short hourly tensor": SHORT_TENSOR_PATH,
                "long hourly tensor": LONG_TENSOR_PATH,
                "hourly availability classification": AVAILABILITY_CLASSIFICATION_PATH,
                "availability events": args.events,
                "network outage windows": NETWORK_OUTAGE_WINDOWS_PATH,
                "partial outage events": PARTIAL_OUTAGE_EVENTS_PATH,
            },
        )
        fault_result = build_fault_label_split_report(Path(args.events))
        output_paths = write_fault_label_split_outputs(fault_result, args.output_dir)
        report_path = output_paths["report"]
        print(report_path.read_text(encoding="utf-8"), end="")
        for name, path in output_paths.items():
            print(f"{name}={path}")
        return

    require_files(
        "Outage-risk label and split construction",
        {
            "hourly availability classification": AVAILABILITY_CLASSIFICATION_PATH,
            "availability events": args.events,
            "network outage windows": NETWORK_OUTAGE_WINDOWS_PATH,
            "partial outage events": PARTIAL_OUTAGE_EVENTS_PATH,
        },
    )
    output_dir = Path(args.output_dir)
    labels = build_label_split_report(Path(args.events))
    label_paths = write_label_split_outputs(labels, output_dir)
    report_path = label_paths["report"]
    print(report_path.read_text(encoding="utf-8"), end="")
    print(f"partition_summary={label_paths['partition_summary']}")
    print(f"purge_summary={label_paths['purge_summary']}")
    print(f"label_changes={label_paths['label_changes']}")
    print(f"report={report_path}")

    if not args.run_models:
        return

    result = evaluate_all_with_predictions(Path(args.events))
    hour_metrics_path = output_dir / HOUR_METRICS_PATH.name
    event_metrics_path = output_dir / EVENT_METRICS_PATH.name
    predictions_path = output_dir / PREDICTIONS_PATH.name
    result["hour_metrics"].to_csv(hour_metrics_path, index=False)
    result["event_metrics"].to_csv(event_metrics_path, index=False)
    result["predictions"].to_parquet(predictions_path, index=False)
    print()
    print("OUTAGE RISK HOUR-LEVEL METRICS (PROVISIONAL)")
    print(result["hour_metrics"].to_string(index=False))
    print()
    print("OUTAGE RISK EVENT-LEVEL METRICS (PROVISIONAL)")
    print(result["event_metrics"].to_string(index=False))
    print()
    print(f"hour_metrics={hour_metrics_path}")
    print(f"event_metrics={event_metrics_path}")
    print(f"predictions={predictions_path}")


if __name__ == "__main__":
    main()
