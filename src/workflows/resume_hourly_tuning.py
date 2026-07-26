from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.workflows.tune_hourly_detection import (
    BOOSTED_METRICS_PATH,
    MANIFEST_PATH,
    METRICS_PATH,
    OUTPUT_DIR,
    PRIOR_RGFN_METRICS_PATH,
    REPORT_PATH,
    SCREENING_PATH,
    SHORT_TENSOR_PATH,
    SWEEP_PATH,
    _attribution_frame,
    _jsonable,
    _load_prior_rgfn,
    _progress,
    _screening_frame,
    _selected_config_frame,
    _validation_frame,
    _verify_augmentation,
    master_tuning_comparison_frame,
    tuning_report,
)
from src.model.hourly_baseline import filter_eligible_examples, load_hourly_tensor, write_metrics_json
from src.model.hourly_detection import FEATURE_PATH, SOURCE_PATH, load_hourly_frame
from src.model.hourly_rgfn import HourlyRgfnConfig, build_hourly_rgfn
from src.model.hourly_rgfn_training import (
    DEVICE,
    METRIC_NAMES,
    HourlyRgfnScaler,
    load_calibrated_baseline,
    load_manifest_splits,
    metric_summary,
)
from src.model.hourly_rgfn_tuning import (
    ArchitectureScreen,
    HourlyRgfnTuningConfig,
    TUNING_SEEDS,
    _candidate_path,
    _finalize_from_models,
    _sample_std,
    _save_candidate_model,
    _selected_validation_rows,
    aggregate_validation_rows,
    configuration_id,
    configuration_payload,
    final_evaluate_hourly_rgfn_arm,
    make_coarse_search_configurations,
    prepare_tuning_split,
    run_combined_arm,
    run_feature_only_arm,
    screen_architecture_search,
    search_configuration_coverage,
    select_top_screened_configurations,
    select_validation_configuration,
    selected_config_from_row,
    train_tuned_candidate,
    validation_rows_for_candidate,
)
from src.workflows.prerequisites import require_files
from src.model.hourly_rgfn_tuning_features import augment_hourly_rgfn_examples
from src.model.hourly_rgfn_tuning_logistic import train_hourly_logistic_variant
from src.model.hourly_calibration import target_check


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--tensor", type=Path, default=SHORT_TENSOR_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--boosted-metrics", type=Path, default=BOOSTED_METRICS_PATH)
    parser.add_argument("--prior-rgfn-metrics", type=Path, default=PRIOR_RGFN_METRICS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--search-count", type=int, default=24)
    parser.add_argument("--top-count", type=int, default=3)
    parser.add_argument(
        "--phase",
        choices=("all", "random", "spaced_arm1", "spaced_screen", "spaced_finalize", "report"),
        default="all",
    )
    return parser


def _payload(path: Path) -> dict[str, object]:
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint is invalid: {path}")
    return value


def _assert_scaler(payload: dict[str, object], scaler: HourlyRgfnScaler) -> None:
    saved = payload.get("scaler")
    if not isinstance(saved, dict):
        raise KeyError("checkpoint lacks train-derived scaler")
    current = scaler.payload()
    if set(saved) != set(current):
        raise ValueError("checkpoint scaler fields do not match current split")
    for name in current:
        if not np.allclose(
            np.asarray(saved[name], dtype=np.float32),
            np.asarray(current[name], dtype=np.float32),
            equal_nan=True,
        ):
            raise ValueError(f"checkpoint scaler does not match current manifest training values: {name}")


def _load_model(
    path: Path,
    prepared: object,
    expected_static: int,
    expected_rules: int,
) -> tuple[torch.nn.Module, HourlyRgfnTuningConfig, dict[str, object], dict[str, object]]:
    value = _payload(path)
    config_values = value.get("tuning_config")
    if not isinstance(config_values, dict):
        raise KeyError(f"checkpoint lacks tuning configuration: {path}")
    config = HourlyRgfnTuningConfig(**config_values)
    if int(value.get("window_hours", 0)) != 7:
        raise ValueError(f"checkpoint does not have a seven-hour input: {path}")
    if int(value.get("n_continuous", 0)) != 24:
        raise ValueError(f"checkpoint continuous width is incompatible: {path}")
    if int(value.get("n_static", 0)) != int(expected_static):
        raise ValueError(f"checkpoint static width is incompatible: {path}")
    if int(value.get("n_rule_evidence", 0)) != int(expected_rules):
        raise ValueError(f"checkpoint rule width is incompatible: {path}")
    _assert_scaler(value, prepared.scaler)
    model_config = HourlyRgfnConfig(
        n_continuous=int(value["n_continuous"]),
        n_static=int(value["n_static"]),
        n_rule_evidence=int(value["n_rule_evidence"]),
        window_hours=int(value["window_hours"]),
        sensor_hidden_size=int(config.sensor_hidden_size),
        evidence_hidden_size=int(config.evidence_hidden_size),
        evidence_embed_size=int(config.evidence_embed_size),
        fusion_hidden_size=int(config.gate_hidden_size),
        dropout=float(config.dropout),
    )
    model = build_hourly_rgfn(str(config.encoder), config=model_config).to(DEVICE)
    state = value.get("state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"checkpoint lacks model state: {path}")
    model.load_state_dict(state)
    selection = value.get("selection")
    if not isinstance(selection, dict):
        raise KeyError(f"checkpoint lacks selection metadata: {path}")
    training = selection.get("training", {})
    if not isinstance(training, dict):
        raise TypeError(f"checkpoint training metadata is invalid: {path}")
    return model, config, dict(training), value


def _candidate_data(
    pattern: str,
    prepared: object,
    expected_static: int,
    expected_rules: int,
    retain_models: bool,
) -> tuple[
    dict[tuple[str, int], torch.nn.Module],
    dict[tuple[str, int], dict[str, object]],
    list[dict[str, object]],
    dict[str, HourlyRgfnTuningConfig],
    dict[str, dict[str, torch.Tensor]],
    list[str],
]:
    paths = sorted(Path(pattern).parent.glob(Path(pattern).name))
    if not paths:
        raise FileNotFoundError(f"no saved candidates match {pattern}")
    models: dict[tuple[str, int], torch.nn.Module] = {}
    infos: dict[tuple[str, int], dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    configs: dict[str, HourlyRgfnTuningConfig] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    names: list[str] = []
    for path in paths:
        model, config, info, value = _load_model(path, prepared, expected_static, expected_rules)
        identifier = configuration_id(config)
        key = (identifier, int(config.seed))
        if key in infos:
            raise ValueError(f"duplicate checkpoint configuration and seed: {path}")
        configs[identifier] = config
        infos[key] = info
        rows.extend(validation_rows_for_candidate(model, prepared, config, info=info))
        names.append(str(path))
        if retain_models:
            models[key] = model
        else:
            states[identifier] = {
                name: tensor.detach().cpu().clone()
                for name, tensor in value["state_dict"].items()
            }
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    return models, infos, rows, configs, states, names


def _best_payload(
    selected_config: HourlyRgfnTuningConfig,
    selected_threshold: float,
    selected_validation_summary: dict[str, object],
) -> dict[str, object]:
    return {
        "configuration": configuration_payload(selected_config),
        "fault_class_weight": float(selected_config.fault_class_weight),
        "threshold": float(selected_threshold),
        "validation": {
            name: float(selected_validation_summary[name])
            for name in ("precision", "recall", "f1", "accuracy")
        },
        "validation_std": {
            name: float(selected_validation_summary[f"{name}_std"])
            for name in ("precision", "recall", "f1", "accuracy")
        },
        "validation_minimum_metric": float(
            min(float(selected_validation_summary[name]) for name in ("precision", "recall", "f1"))
        ),
    }


def _restore_completed_arm(
    arm_name: str,
    split_name: str,
    candidate_pattern: str,
    final_pattern: str,
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    expected_static: int,
    expected_rules: int,
) -> dict[str, object]:
    prepared = prepare_tuning_split(examples, splits)
    _, infos, rows, _, _, candidate_paths = _candidate_data(
        candidate_pattern,
        prepared,
        expected_static,
        expected_rules,
        retain_models=False,
    )
    grid = aggregate_validation_rows(rows)
    selected = select_validation_configuration(grid, "balanced")
    selected_config = selected_config_from_row(selected, seed=0)
    selected_rows = _selected_validation_rows(rows, selected)
    selected_summary = metric_summary([{name: row[name] for name in METRIC_NAMES} for row in selected_rows])
    final_paths = sorted(Path(final_pattern).parent.glob(Path(final_pattern).name))
    if len(final_paths) != len(TUNING_SEEDS):
        raise ValueError(f"completed {arm_name} must have five final test checkpoints")
    test_rows: list[dict[str, object]] = []
    selected_epochs: list[float] = []
    for path in final_paths:
        value = _payload(path)
        config_values = value.get("tuning_config")
        selection = value.get("selection")
        if not isinstance(config_values, dict) or not isinstance(selection, dict):
            raise ValueError(f"completed final checkpoint is malformed: {path}")
        config = HourlyRgfnTuningConfig(**config_values)
        if configuration_id(config) != configuration_id(selected_config):
            raise ValueError(f"completed final checkpoint does not match reconstructed validation selection: {path}")
        if float(selection.get("threshold", np.nan)) != float(selected["threshold"]):
            raise ValueError(f"completed final checkpoint threshold does not match validation selection: {path}")
        test = selection.get("test")
        if not isinstance(test, dict):
            raise ValueError(f"completed final checkpoint lacks saved test result: {path}")
        test_rows.append({"seed": int(config.seed), **test})
        info = infos.get((configuration_id(config), int(config.seed)))
        if info is None:
            raise ValueError(f"completed final checkpoint lacks its candidate metadata: {path}")
        selected_epochs.append(float(info["best_epoch"]))
    test_rows = sorted(test_rows, key=lambda row: int(row["seed"]))
    test_summary = metric_summary(test_rows)
    return {
        "arm": arm_name,
        "split": split_name,
        "parameter_count": int(infos[(configuration_id(selected_config), 0)]["parameter_count"]),
        "selected_config": configuration_payload(selected_config),
        "selected_threshold": float(selected["threshold"]),
        "validation_seed_rows": rows,
        "validation_grid": grid,
        "best_balanced": _best_payload(selected_config, float(selected["threshold"]), selected_summary),
        "selected_validation_seed_rows": selected_rows,
        "selected_validation_summary": selected_summary,
        "test_seed_rows": test_rows,
        "test_summary": test_summary,
        "final_test_target_check": target_check(test_summary),
        "selected_best_epoch_mean": float(np.mean(selected_epochs)),
        "selected_best_epoch_std": _sample_std(selected_epochs),
        "test_evaluation_count": len(test_rows),
        "test_evaluation_count_per_seed": 1,
        "selection_source": "reconstructed_validation_with_saved_final_test",
        "model_paths": [str(path) for path in final_paths],
        "candidate_model_paths": candidate_paths,
        "search_metadata": {"mode": "restored_completed_feature_arm"},
    }


def _restore_completed_architecture_arm(
    arm_name: str,
    split_name: str,
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    model_dir: Path,
) -> dict[str, object]:
    prepared = prepare_tuning_split(examples, splits[split_name])
    screen, top_configs = _screen_from_saved(examples, splits, split_name, model_dir)
    finalist_ids = {configuration_id(config) for config in top_configs}
    rows = [
        dict(row)
        for row in screen.validation_rows
        if str(row["configuration_id"]) in finalist_ids
    ]
    infos: dict[tuple[str, int], dict[str, object]] = {
        (identifier, int(screen.screening_seed)): dict(screen.candidate_info[identifier])
        for identifier in finalist_ids
    }
    candidate_paths = list(screen.candidate_model_paths)
    for config in top_configs:
        identifier = configuration_id(config)
        for seed in TUNING_SEEDS:
            if int(seed) == int(screen.screening_seed):
                continue
            path = _candidate_path(
                model_dir,
                arm_name,
                split_name,
                replace(config, seed=int(seed)),
            )
            if not path.exists():
                raise FileNotFoundError(f"completed finalist checkpoint is missing: {path}")
            model, loaded_config, info, _ = _load_model(path, prepared, 3, 38)
            if configuration_id(loaded_config) != identifier or int(loaded_config.seed) != int(seed):
                raise ValueError(f"completed finalist checkpoint configuration is inconsistent: {path}")
            infos[(identifier, int(seed))] = dict(info)
            rows.extend(validation_rows_for_candidate(model, prepared, loaded_config, info=info))
            candidate_paths.append(str(path))
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    grid = aggregate_validation_rows(rows)
    selected = select_validation_configuration(grid, "balanced")
    selected_config = selected_config_from_row(selected, seed=0)
    selected_rows = _selected_validation_rows(rows, selected)
    selected_summary = metric_summary([{name: row[name] for name in METRIC_NAMES} for row in selected_rows])
    final_paths = _final_paths(model_dir, arm_name, split_name)
    if len(final_paths) != len(TUNING_SEEDS):
        raise ValueError(f"completed {arm_name} must have five final test checkpoints")
    test_rows: list[dict[str, object]] = []
    selected_epochs: list[float] = []
    for path in final_paths:
        value = _payload(path)
        config_values = value.get("tuning_config")
        selection = value.get("selection")
        if not isinstance(config_values, dict) or not isinstance(selection, dict):
            raise ValueError(f"completed final checkpoint is malformed: {path}")
        config = HourlyRgfnTuningConfig(**config_values)
        if configuration_id(config) != configuration_id(selected_config):
            raise ValueError(f"completed final checkpoint does not match reconstructed validation selection: {path}")
        if float(selection.get("threshold", np.nan)) != float(selected["threshold"]):
            raise ValueError(f"completed final checkpoint threshold does not match validation selection: {path}")
        test = selection.get("test")
        if not isinstance(test, dict):
            raise ValueError(f"completed final checkpoint lacks saved test result: {path}")
        key = (configuration_id(config), int(config.seed))
        info = infos.get(key)
        if info is None:
            raise ValueError(f"completed final checkpoint lacks its finalist metadata: {path}")
        test_rows.append({"seed": int(config.seed), **test})
        selected_epochs.append(float(info["best_epoch"]))
    test_rows = sorted(test_rows, key=lambda row: int(row["seed"]))
    test_summary = metric_summary(test_rows)
    return {
        "arm": arm_name,
        "split": split_name,
        "parameter_count": int(infos[(configuration_id(selected_config), 0)]["parameter_count"]),
        "selected_config": configuration_payload(selected_config),
        "selected_threshold": float(selected["threshold"]),
        "validation_seed_rows": rows,
        "validation_grid": grid,
        "best_balanced": _best_payload(selected_config, float(selected["threshold"]), selected_summary),
        "selected_validation_seed_rows": selected_rows,
        "selected_validation_summary": selected_summary,
        "test_seed_rows": test_rows,
        "test_summary": test_summary,
        "final_test_target_check": target_check(test_summary),
        "selected_best_epoch_mean": float(np.mean(selected_epochs)),
        "selected_best_epoch_std": _sample_std(selected_epochs),
        "test_evaluation_count": len(test_rows),
        "test_evaluation_count_per_seed": 1,
        "selection_source": "reconstructed_finalist_validation_with_saved_final_test",
        "model_paths": [str(path) for path in final_paths],
        "candidate_model_paths": candidate_paths,
        "search_metadata": {
            "mode": "restored_completed_architecture_search",
            "screening_seed": int(screen.screening_seed),
            "screening_candidate_count": int(len(screen.candidate_configs)),
            "finalist_count": int(len(top_configs)),
            "finalist_configuration_ids": sorted(finalist_ids),
        },
    }


def _restore_logistic(path: Path) -> dict[str, object]:
    value = joblib.load(path)
    selection = value.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("saved logistic model lacks selection metadata")
    test = selection.get("test")
    validation = selection.get("validation")
    if not isinstance(test, dict) or not isinstance(validation, dict):
        raise ValueError("saved logistic model lacks validation or test metrics")
    summary = metric_summary([test])
    return {
        "model": "logistic_regression",
        "split": str(selection["split"]),
        "window_hours": int(value["window_hours"]),
        "feature_dimension": int(value["feature_dimension"]),
        "parameter_count": int(value["parameter_count"]),
        "configuration": dict(value["configuration"]),
        "best_balanced": {
            "fault_class_weight": float(selection["fault_class_weight"]),
            "threshold": float(selection["threshold"]),
            "validation": validation,
        },
        "selected_validation_seed_rows": [{"seed": int(value["configuration"]["seed"]), **validation}],
        "selected_validation_summary": metric_summary([validation]),
        "test_seed_rows": [{"seed": int(value["configuration"]["seed"]), **test}],
        "test_summary": summary,
        "final_test_target_check": target_check(test),
        "test_evaluation_count": 1,
        "test_evaluation_count_per_seed": 1,
        "selection_source": "saved_validation_selection",
        "model_path": str(path),
    }


def _resume_arm2_random(
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    model_dir: Path,
) -> tuple[dict[str, object], dict[str, object], object]:
    prepared = prepare_tuning_split(examples, splits)
    _, screen_infos, screen_rows, screen_configs, screen_states, screen_paths = _candidate_data(
        str(model_dir / "candidates" / "rgfn_arm2_screen_random_*.pt"),
        prepared,
        3,
        38,
        retain_models=False,
    )
    screen = ArchitectureScreen(
        prepared=prepared,
        split_name="random",
        candidate_configs=screen_configs,
        candidate_states=screen_states,
        candidate_info={identifier: screen_infos[(identifier, 0)] for identifier in screen_configs},
        validation_rows=screen_rows,
        validation_grid=aggregate_validation_rows(screen_rows),
        screening_seed=0,
        coverage=search_configuration_coverage(tuple(screen_configs.values())),
        candidate_model_paths=screen_paths,
    )
    top_configs = select_top_screened_configurations(screen, top_count=3)
    expected_hashes = {configuration_id(config) for config in top_configs}
    partial_models, partial_infos, partial_rows, partial_configs, _, partial_paths = _candidate_data(
        str(model_dir / "candidates" / "rgfn_arm2_tuned_only_random_*.pt"),
        prepared,
        3,
        38,
        retain_models=True,
    )
    unexpected = set(partial_configs).difference(expected_hashes)
    if unexpected:
        raise ValueError("saved Arm 2 finalists do not match reconstructed validation shortlist")
    models: dict[tuple[str, int], torch.nn.Module] = dict(partial_models)
    infos: dict[tuple[str, int], dict[str, object]] = dict(partial_infos)
    rows = [dict(row) for row in screen_rows if str(row["configuration_id"]) in expected_hashes]
    rows.extend(partial_rows)
    candidate_paths = list(screen_paths) + list(partial_paths)
    for candidate in top_configs:
        identifier = configuration_id(candidate)
        key = (identifier, 0)
        if key not in models:
            state = screen_states[identifier]
            model_config = HourlyRgfnConfig(
                n_continuous=24,
                n_static=3,
                n_rule_evidence=38,
                window_hours=7,
                sensor_hidden_size=int(candidate.sensor_hidden_size),
                evidence_hidden_size=int(candidate.evidence_hidden_size),
                evidence_embed_size=int(candidate.evidence_embed_size),
                fusion_hidden_size=int(candidate.gate_hidden_size),
                dropout=float(candidate.dropout),
            )
            model = build_hourly_rgfn(str(candidate.encoder), config=model_config).to(DEVICE)
            model.load_state_dict(state)
            models[key] = model
            infos[key] = dict(screen.candidate_info[identifier])
        for seed in TUNING_SEEDS:
            key = (identifier, int(seed))
            if key in models:
                continue
            config = replace(candidate, seed=int(seed))
            model, info = train_tuned_candidate(prepared, config)
            models[key] = model
            infos[key] = info
            rows.extend(validation_rows_for_candidate(model, prepared, config, info=info))
            saved = _save_candidate_model(
                model_dir,
                "rgfn_arm2_tuned_only",
                "random",
                model,
                prepared.scaler,
                config,
                info,
            )
            if saved is not None:
                candidate_paths.append(saved)
            _progress(
                {
                    "arm": "rgfn_arm2_tuned_only_resume",
                    "split": "random",
                    "seed": int(seed),
                    "fault_class_weight": float(config.fault_class_weight),
                    **info,
                }
            )
    grid = aggregate_validation_rows(rows)
    finalized = _finalize_from_models(
        "rgfn_arm2_tuned_only",
        "random",
        prepared,
        models,
        infos,
        rows,
        grid,
        candidate_paths,
    )
    result = final_evaluate_hourly_rgfn_arm(finalized, examples, splits, model_dir=model_dir)
    detail = {
        "coverage": screen.coverage,
        "screening_seed": 0,
        "candidate_model_paths": screen_paths,
        "validation_rows": screen_rows,
        "validation_grid": screen.validation_grid,
        "top_configurations": [configuration_payload(config) for config in top_configs],
        "resumed_missing_finalist_fits": 3,
    }
    return result, detail, finalized


def _run_spaced(
    examples: dict[str, np.ndarray],
    augmented: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    model_dir: Path,
    search_configs: tuple[HourlyRgfnTuningConfig, ...],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    logistic = train_hourly_logistic_variant(examples, splits["spaced"], "spaced", model_dir=model_dir)
    arm1 = run_feature_only_arm(
        augmented,
        splits["spaced"],
        "spaced",
        "rgfn_arm1_features_only",
        model_dir=model_dir,
        progress=_progress,
    )
    screen = screen_architecture_search(
        examples,
        splits["spaced"],
        "spaced",
        search_configs,
        model_dir=model_dir,
        arm_name="rgfn_arm2_screen",
        progress=_progress,
    )
    top_configs = select_top_screened_configurations(screen, top_count=3)
    from src.model.hourly_rgfn_tuning import finalize_architecture_search

    finalized = finalize_architecture_search(
        screen,
        "rgfn_arm2_tuned_only",
        top_configs=top_configs,
        model_dir=model_dir,
        progress=_progress,
    )
    arm2 = final_evaluate_hourly_rgfn_arm(finalized, examples, splits["spaced"], model_dir=model_dir)
    arm3 = run_combined_arm(
        augmented,
        splits["spaced"],
        "spaced",
        "rgfn_arm3_combined",
        finalized.selected_config,
        finalized.selected_threshold,
        model_dir=model_dir,
        progress=_progress,
    )
    detail = {
        "coverage": screen.coverage,
        "screening_seed": screen.screening_seed,
        "candidate_model_paths": screen.candidate_model_paths,
        "validation_rows": screen.validation_rows,
        "validation_grid": screen.validation_grid,
        "top_configurations": [configuration_payload(config) for config in top_configs],
    }
    return logistic, arm1, arm2, arm3, detail


def _load_inputs(args: object) -> tuple[
    Path,
    Path,
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    tuple[HourlyRgfnTuningConfig, ...],
    dict[str, object],
]:
    require_files(
        "Hourly RGFN tuning resume",
        {
            "short hourly tensor": Path(args.tensor),
            "baseline split manifest": Path(args.manifest),
            "calibrated baseline metrics": Path(args.boosted_metrics),
            "prior RGFN metrics": Path(args.prior_rgfn_metrics),
            "canonical merged dataset": SOURCE_PATH,
            "feature matrix": FEATURE_PATH,
        },
        "Build hourly data, then run baseline, calibration, and RGFN training before resuming tuning.",
    )
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models" / "rgfn_hourly_tuning"
    examples, excluded_removed = filter_eligible_examples(load_hourly_tensor(Path(args.tensor)))
    splits = load_manifest_splits(examples, Path(args.manifest))
    augmented = augment_hourly_rgfn_examples(examples, load_hourly_frame())
    integrity = _verify_augmentation(examples, augmented)
    boosted = load_calibrated_baseline(Path(args.boosted_metrics))
    prior = _load_prior_rgfn(Path(args.prior_rgfn_metrics))
    search_configs = make_coarse_search_configurations(count=int(args.search_count))
    inputs = {
        "device": str(DEVICE),
        "tensor": str(Path(args.tensor)),
        "split_manifest": str(Path(args.manifest)),
        "boosted_metrics": str(Path(args.boosted_metrics)),
        "prior_rgfn_metrics": str(Path(args.prior_rgfn_metrics)),
        "eligible_hourly_examples": int(len(examples["y_binary"])),
        "excluded_examples_removed": int(excluded_removed),
    }
    return output_dir, model_dir, examples, splits, augmented, integrity, boosted, prior, search_configs, inputs


def _final_paths(model_dir: Path, arm_name: str, split_name: str) -> list[Path]:
    return sorted(model_dir.glob(f"{arm_name}_{split_name}_seed_*.pt"))


def _screen_from_saved(
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    split_name: str,
    model_dir: Path,
) -> tuple[ArchitectureScreen, tuple[HourlyRgfnTuningConfig, ...]]:
    prepared = prepare_tuning_split(examples, splits[split_name])
    _, infos, rows, configs, states, paths = _candidate_data(
        str(model_dir / "candidates" / f"rgfn_arm2_screen_{split_name}_*.pt"),
        prepared,
        3,
        38,
        retain_models=False,
    )
    if len(configs) != 24:
        raise ValueError(f"{split_name} Arm 2 screen must contain 24 saved configurations")
    screen = ArchitectureScreen(
        prepared=prepared,
        split_name=split_name,
        candidate_configs=configs,
        candidate_states=states,
        candidate_info={identifier: infos[(identifier, 0)] for identifier in configs},
        validation_rows=rows,
        validation_grid=aggregate_validation_rows(rows),
        screening_seed=0,
        coverage=search_configuration_coverage(tuple(configs.values())),
        candidate_model_paths=paths,
    )
    return screen, select_top_screened_configurations(screen, top_count=3)


def _screen_detail(screen: ArchitectureScreen, top_configs: tuple[HourlyRgfnTuningConfig, ...]) -> dict[str, object]:
    return {
        "coverage": screen.coverage,
        "screening_seed": screen.screening_seed,
        "candidate_model_paths": screen.candidate_model_paths,
        "validation_rows": screen.validation_rows,
        "validation_grid": screen.validation_grid,
        "top_configurations": [configuration_payload(config) for config in top_configs],
    }


def _run_random_phase(args: object) -> None:
    _, model_dir, examples, splits, augmented, _, _, _, _, _ = _load_inputs(args)
    arm2_paths = _final_paths(model_dir, "rgfn_arm2_tuned_only", "random")
    if len(arm2_paths) == len(TUNING_SEEDS):
        arm2 = _restore_completed_arm(
            "rgfn_arm2_tuned_only",
            "random",
            str(model_dir / "candidates" / "rgfn_arm2_tuned_only_random_*.pt"),
            str(model_dir / "rgfn_arm2_tuned_only_random_seed_*.pt"),
            examples,
            splits["random"],
            3,
            38,
        )
        selected_config = HourlyRgfnTuningConfig(**{**arm2["selected_config"], "seed": 0})
        selected_threshold = float(arm2["selected_threshold"])
        print("resume=random Arm 2 final artifacts already present", flush=True)
    else:
        arm2, _, finalized = _resume_arm2_random(examples, splits["random"], model_dir)
        selected_config = finalized.selected_config
        selected_threshold = finalized.selected_threshold
    arm3_paths = _final_paths(model_dir, "rgfn_arm3_combined", "random")
    if len(arm3_paths) == len(TUNING_SEEDS):
        print("resume=random Arm 3 final artifacts already present", flush=True)
        return
    print("training=rgfn_arm3_combined split=random", flush=True)
    run_combined_arm(
        augmented,
        splits["random"],
        "random",
        "rgfn_arm3_combined",
        selected_config,
        selected_threshold,
        model_dir=model_dir,
        progress=_progress,
    )


def _run_spaced_arm1_phase(args: object) -> None:
    _, model_dir, examples, splits, augmented, _, _, _, _, _ = _load_inputs(args)
    logistic_path = model_dir / "hourly_logistic_spaced.joblib"
    if logistic_path.exists():
        print("resume=spaced logistic final artifact already present", flush=True)
    else:
        print("training=logistic_regression split=spaced", flush=True)
        train_hourly_logistic_variant(examples, splits["spaced"], "spaced", model_dir=model_dir)
    paths = _final_paths(model_dir, "rgfn_arm1_features_only", "spaced")
    if len(paths) == len(TUNING_SEEDS):
        print("resume=spaced Arm 1 final artifacts already present", flush=True)
        return
    print("training=rgfn_arm1_features_only split=spaced", flush=True)
    run_feature_only_arm(
        augmented,
        splits["spaced"],
        "spaced",
        "rgfn_arm1_features_only",
        model_dir=model_dir,
        progress=_progress,
    )


def _run_spaced_screen_phase(args: object) -> None:
    _, model_dir, examples, splits, _, _, _, _, search_configs, _ = _load_inputs(args)
    candidates = list((model_dir / "candidates").glob("rgfn_arm2_screen_spaced_*.pt"))
    if len(candidates) == len(search_configs):
        print("resume=spaced Arm 2 screen artifacts already present", flush=True)
        return
    if candidates:
        raise RuntimeError("spaced Arm 2 screen is partial; remove no files and rerun this phase only after review")
    print("training=rgfn_arm2_screen split=spaced", flush=True)
    screen_architecture_search(
        examples,
        splits["spaced"],
        "spaced",
        search_configs,
        model_dir=model_dir,
        arm_name="rgfn_arm2_screen",
        progress=_progress,
    )


def _run_spaced_finalize_phase(args: object) -> None:
    _, model_dir, examples, splits, augmented, _, _, _, _, _ = _load_inputs(args)
    arm2_paths = _final_paths(model_dir, "rgfn_arm2_tuned_only", "spaced")
    if len(arm2_paths) == len(TUNING_SEEDS):
        arm2 = _restore_completed_arm(
            "rgfn_arm2_tuned_only",
            "spaced",
            str(model_dir / "candidates" / "rgfn_arm2_tuned_only_spaced_*.pt"),
            str(model_dir / "rgfn_arm2_tuned_only_spaced_seed_*.pt"),
            examples,
            splits["spaced"],
            3,
            38,
        )
        selected_config = HourlyRgfnTuningConfig(**{**arm2["selected_config"], "seed": 0})
        selected_threshold = float(arm2["selected_threshold"])
        print("resume=spaced Arm 2 final artifacts already present", flush=True)
    else:
        screen, top_configs = _screen_from_saved(examples, splits, "spaced", model_dir)
        from src.model.hourly_rgfn_tuning import finalize_architecture_search

        finalized = finalize_architecture_search(
            screen,
            "rgfn_arm2_tuned_only",
            top_configs=top_configs,
            model_dir=model_dir,
            progress=_progress,
        )
        final_evaluate_hourly_rgfn_arm(finalized, examples, splits["spaced"], model_dir=model_dir)
        selected_config = finalized.selected_config
        selected_threshold = finalized.selected_threshold
    arm3_paths = _final_paths(model_dir, "rgfn_arm3_combined", "spaced")
    if len(arm3_paths) == len(TUNING_SEEDS):
        print("resume=spaced Arm 3 final artifacts already present", flush=True)
        return
    print("training=rgfn_arm3_combined split=spaced", flush=True)
    run_combined_arm(
        augmented,
        splits["spaced"],
        "spaced",
        "rgfn_arm3_combined",
        selected_config,
        selected_threshold,
        model_dir=model_dir,
        progress=_progress,
    )


def _write_report_from_saved(args: object) -> None:
    output_dir, model_dir, examples, splits, augmented, integrity, boosted, prior, search_configs, inputs = _load_inputs(args)
    logistic = {
        split: _restore_logistic(model_dir / f"hourly_logistic_{split}.joblib")
        for split in ("random", "spaced")
    }
    arm1 = {
        split: _restore_completed_arm(
            "rgfn_arm1_features_only",
            split,
            str(model_dir / "candidates" / f"rgfn_arm1_features_only_{split}_*.pt"),
            str(model_dir / f"rgfn_arm1_features_only_{split}_seed_*.pt"),
            augmented,
            splits[split],
            9,
            54,
        )
        for split in ("random", "spaced")
    }
    arm2 = {
        split: _restore_completed_architecture_arm(
            "rgfn_arm2_tuned_only",
            split,
            examples,
            splits,
            model_dir,
        )
        for split in ("random", "spaced")
    }
    arm3 = {
        split: _restore_completed_arm(
            "rgfn_arm3_combined",
            split,
            str(model_dir / "candidates" / f"rgfn_arm3_combined_{split}_*.pt"),
            str(model_dir / f"rgfn_arm3_combined_{split}_seed_*.pt"),
            augmented,
            splits[split],
            9,
            54,
        )
        for split in ("random", "spaced")
    }
    screen_details: dict[str, dict[str, object]] = {}
    for split in ("random", "spaced"):
        screen, top_configs = _screen_from_saved(examples, splits, split, model_dir)
        screen_details[split] = _screen_detail(screen, top_configs)
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
        "coverage": search_configuration_coverage(search_configs),
        "resume": "completed test results were restored from saved final artifacts without a second test evaluation",
    }
    payload = {
        "configuration": {
            "task": "hour-level binary fault detection",
            "window_hours": 7,
            "device": str(DEVICE),
            "seeds": list(TUNING_SEEDS),
            "selection": "aggregate validation maximum of minimum precision, recall, and f1",
            "test_evaluation": "one evaluation per selected seed model",
            "resume": search_details["resume"],
            "arm2": {"search": search_details},
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


def _run_phase(args: object) -> None:
    phase = str(args.phase)
    if phase == "random":
        _run_random_phase(args)
        return
    if phase == "spaced_arm1":
        _run_spaced_arm1_phase(args)
        return
    if phase == "spaced_screen":
        _run_spaced_screen_phase(args)
        return
    if phase == "spaced_finalize":
        _run_spaced_finalize_phase(args)
        return
    if phase == "report":
        _write_report_from_saved(args)
        return
    raise ValueError(f"unsupported phase: {phase}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args().parse_args(argv)
    if str(args.phase) != "all":
        _run_phase(args)
        return
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
    print("resume=random completed Arm 1/logistic", flush=True)
    logistic_random = _restore_logistic(model_dir / "hourly_logistic_random.joblib")
    arm1_random = _restore_completed_arm(
        "rgfn_arm1_features_only",
        "random",
        str(model_dir / "candidates" / "rgfn_arm1_features_only_random_*.pt"),
        str(model_dir / "rgfn_arm1_features_only_random_seed_*.pt"),
        augmented,
        splits["random"],
        9,
        54,
    )
    print("resume=random Arm 2", flush=True)
    arm2_random, random_screen, random_finalized = _resume_arm2_random(
        examples,
        splits["random"],
        model_dir,
    )
    print("training=rgfn_arm3_combined split=random", flush=True)
    arm3_random = run_combined_arm(
        augmented,
        splits["random"],
        "random",
        "rgfn_arm3_combined",
        random_finalized.selected_config,
        random_finalized.selected_threshold,
        model_dir=model_dir,
        progress=_progress,
    )
    print("training=all spaced arms", flush=True)
    logistic_spaced, arm1_spaced, arm2_spaced, arm3_spaced, spaced_screen = _run_spaced(
        examples,
        augmented,
        splits,
        model_dir,
        search_configs,
    )
    logistic = {"random": logistic_random, "spaced": logistic_spaced}
    arm1 = {"random": arm1_random, "spaced": arm1_spaced}
    arm2 = {"random": arm2_random, "spaced": arm2_spaced}
    arm3 = {"random": arm3_random, "spaced": arm3_spaced}
    screen_details = {"random": random_screen, "spaced": spaced_screen}
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
        "resume": "completed random Arm 1 and logistic test records restored from their saved final artifacts",
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
            "resume": search_details["resume"],
            "arm2": {"search": search_details},
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
