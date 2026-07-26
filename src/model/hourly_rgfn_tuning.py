from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.model.hourly_baseline import binary_metrics
from src.model.hourly_calibration import select_operating_point, target_check
from src.model.hourly_rgfn import ENCODER_GRU, MASK_MODE_PER_HOUR, HourlyRgfnConfig, build_hourly_rgfn, set_hourly_rgfn_seed
from src.model.hourly_rgfn_training import (
    DEVICE,
    FEATURE_KEYS,
    METRIC_NAMES,
    HourlyRgfnScaler,
    TensorPartition,
    metric_summary,
)


TUNING_SEEDS = (0, 1, 2, 3, 4)
TUNING_WEIGHTS = (1.0, 2.0, 4.0, 6.0, 8.0)
TUNING_THRESHOLDS = tuple(float(value) for value in np.round(np.arange(0.30, 0.8001, 0.05), 2))
SEARCH_SENSOR_WIDTHS = (32, 48, 64, 96)
SEARCH_EVIDENCE_WIDTHS = (16, 32, 64)
SEARCH_GATE_WIDTHS = (8, 16, 32)
SEARCH_GATE_L1 = (0.0, 0.01, 0.03)
SEARCH_LEARNING_RATES = (3e-4, 1e-3, 3e-3)
SEARCH_WEIGHT_DECAYS = (1e-4, 1e-3)


@dataclass(frozen=True)
class HourlyRgfnTuningConfig:
    seed: int = 0
    fault_class_weight: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 15
    min_delta: float = 1e-4
    sensor_hidden_size: int = 48
    evidence_hidden_size: int = 32
    gate_hidden_size: int = 32
    evidence_embed_size: int = 16
    dropout: float = 0.3
    gate_l1: float = 0.0
    encoder: str = ENCODER_GRU
    mask_mode: str = MASK_MODE_PER_HOUR


@dataclass
class PreparedHourlyRgfnTuningSplit:
    train: TensorPartition
    validation: TensorPartition
    scaler: HourlyRgfnScaler
    sample_count: int


@dataclass
class TestEvaluationLedger:
    count: int = 0
    seeds: set[int] = field(default_factory=set)

    def record(self, seed: int) -> None:
        value = int(seed)
        if value in self.seeds:
            raise RuntimeError(f"test evaluation was already recorded for seed {value}")
        self.seeds.add(value)
        self.count += 1


@dataclass
class ArchitectureScreen:
    prepared: PreparedHourlyRgfnTuningSplit
    split_name: str
    candidate_configs: dict[str, HourlyRgfnTuningConfig]
    candidate_states: dict[str, dict[str, torch.Tensor]]
    candidate_info: dict[str, dict[str, object]]
    validation_rows: list[dict[str, object]]
    validation_grid: list[dict[str, object]]
    screening_seed: int
    coverage: dict[str, object]
    candidate_model_paths: list[str]


@dataclass
class FinalizedHourlyRgfnArm:
    arm_name: str
    split_name: str
    prepared: PreparedHourlyRgfnTuningSplit
    selected_config: HourlyRgfnTuningConfig
    selected_threshold: float
    models: dict[int, nn.Module]
    model_info: dict[int, dict[str, object]]
    validation_rows: list[dict[str, object]]
    validation_grid: list[dict[str, object]]
    selected_validation_rows: list[dict[str, object]]
    selected_validation_summary: dict[str, object]
    candidate_model_paths: list[str] = field(default_factory=list)
    search_metadata: dict[str, object] = field(default_factory=dict)
    selection_source: str = "validation"
    test_ledger: TestEvaluationLedger = field(default_factory=TestEvaluationLedger)


def default_feature_arm_config() -> HourlyRgfnTuningConfig:
    return HourlyRgfnTuningConfig(max_epochs=100, patience=10)


def default_search_config() -> HourlyRgfnTuningConfig:
    return HourlyRgfnTuningConfig(max_epochs=200, patience=15)


def configuration_payload(config: HourlyRgfnTuningConfig, include_seed: bool = False) -> dict[str, object]:
    payload = asdict(config)
    if not include_seed:
        payload.pop("seed")
    return payload


def configuration_id(config: HourlyRgfnTuningConfig) -> str:
    return json.dumps(configuration_payload(config), sort_keys=True, separators=(",", ":"))


def _batch_ranges(count: int, batch_size: int):
    for start in range(0, int(count), int(batch_size)):
        yield start, min(start + int(batch_size), int(count))


def _make_partition(
    values: dict[str, np.ndarray],
    labels: np.ndarray,
    indices: np.ndarray,
) -> TensorPartition:
    selected = np.asarray(indices, dtype=np.int64)
    features = tuple(
        torch.as_tensor(
            np.ascontiguousarray(np.asarray(values[name], dtype=np.float32)[selected]),
            dtype=torch.float32,
            device=DEVICE,
        )
        for name in FEATURE_KEYS
    )
    target = torch.as_tensor(
        np.ascontiguousarray(np.asarray(labels, dtype=np.float32)[selected]),
        dtype=torch.float32,
        device=DEVICE,
    )
    return TensorPartition(features=features, labels=target)


def _validate_split_indices(splits: dict[str, np.ndarray], count: int) -> None:
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError("tuning requires train, validation, and test index sets")
    pieces = [np.asarray(splits[name], dtype=np.int64) for name in ("train", "validation", "test")]
    combined = np.concatenate(pieces)
    if len(combined) != int(count) or len(np.unique(combined)) != int(count):
        raise ValueError("tuning index sets must be disjoint and complete")
    if np.any(combined < 0) or np.any(combined >= int(count)):
        raise ValueError("tuning index sets contain an invalid example index")


def prepare_tuning_split(
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
) -> PreparedHourlyRgfnTuningSplit:
    labels = np.asarray(examples["y_binary"], dtype=int)
    x_cont = np.asarray(examples["X_cont"], dtype=np.float32)
    if x_cont.ndim != 3 or x_cont.shape[1] != 7:
        raise ValueError("hourly RGFN tuning requires a seven-hour causal input")
    _validate_split_indices(splits, len(labels))
    for name in ("train", "validation"):
        values = labels[np.asarray(splits[name], dtype=np.int64)]
        if not np.equal(values, 0).any() or not np.equal(values, 1).any():
            raise ValueError(f"{name} data must contain both binary classes")
    scaler = HourlyRgfnScaler.fit(examples, np.asarray(splits["train"], dtype=np.int64))
    scaled = scaler.transform(examples)
    return PreparedHourlyRgfnTuningSplit(
        train=_make_partition(scaled, labels, np.asarray(splits["train"], dtype=np.int64)),
        validation=_make_partition(scaled, labels, np.asarray(splits["validation"], dtype=np.int64)),
        scaler=scaler,
        sample_count=int(len(labels)),
    )


def _make_test_partition(
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    scaler: HourlyRgfnScaler,
) -> TensorPartition:
    labels = np.asarray(examples["y_binary"], dtype=int)
    _validate_split_indices(splits, len(labels))
    scaled = scaler.transform(examples)
    return _make_partition(scaled, labels, np.asarray(splits["test"], dtype=np.int64))


def _loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    criterion: nn.Module,
    gate_l1: float,
) -> torch.Tensor:
    value = criterion(output["final_fault_logit"], labels)
    if float(gate_l1) > 0.0:
        value = value + float(gate_l1) * torch.mean(torch.abs(output["alpha"]))
    return value


def _validation_loss(
    model: nn.Module,
    partition: TensorPartition,
    criterion: nn.Module,
    gate_l1: float,
    batch_size: int,
) -> float:
    if partition.labels is None:
        raise ValueError("validation labels are required")
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start, stop in _batch_ranges(partition.count, batch_size):
            output = model(*[value[start:stop] for value in partition.features])
            value = _loss(output, partition.labels[start:stop], criterion, gate_l1)
            total += float(value.detach().cpu()) * (stop - start)
    return float(total / max(partition.count, 1))


def _copy_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _model_device_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _restore_model_device_state(model: nn.Module, values: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    with torch.no_grad():
        for name, value in current.items():
            value.copy_(values[name])


def _zero_adam_state(optimizer: torch.optim.Optimizer) -> None:
    with torch.no_grad():
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()


def _make_adam(model: nn.Module, config: HourlyRgfnTuningConfig, capturable: bool) -> torch.optim.Adam:
    kwargs: dict[str, object] = {
        "lr": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
    }
    if capturable:
        kwargs["capturable"] = True
    return torch.optim.Adam(model.parameters(), **kwargs)


def _eager_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    gate_l1: float,
    set_to_none: bool,
) -> None:
    optimizer.zero_grad(set_to_none=set_to_none)
    output = model(*features)
    value = _loss(output, labels, criterion, gate_l1)
    value.backward()
    optimizer.step()


@dataclass
class _CudaGraphStep:
    graph: torch.cuda.CUDAGraph
    static_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    static_labels: torch.Tensor
    batch_size: int

    def stage(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        if int(indices.numel()) != self.batch_size:
            raise ValueError("CUDA graph staging received an unexpected batch size")
        for target, source in zip(self.static_features, features):
            torch.index_select(source, 0, indices, out=target)
        torch.index_select(labels, 0, indices, out=self.static_labels)

    def replay(self) -> None:
        self.graph.replay()


def _prepare_cuda_graph_step(
    model: nn.Module,
    optimizer: torch.optim.Adam,
    criterion: nn.Module,
    partition: TensorPartition,
    config: HourlyRgfnTuningConfig,
) -> _CudaGraphStep | None:
    if DEVICE.type != "cuda" or not torch.cuda.is_available() or partition.labels is None:
        return None
    if partition.count < int(config.batch_size):
        return None
    static_features = tuple(
        torch.zeros(
            (int(config.batch_size), *value.shape[1:]),
            dtype=value.dtype,
            device=value.device,
        )
        for value in partition.features
    )
    static_labels = torch.zeros(
        int(config.batch_size),
        dtype=partition.labels.dtype,
        device=partition.labels.device,
    )
    initial_model = _model_device_state(model)
    initial_rng = torch.cuda.get_rng_state(device=DEVICE)
    warmup_stream = torch.cuda.Stream(device=DEVICE)
    graph = torch.cuda.CUDAGraph()
    try:
        warmup_stream.wait_stream(torch.cuda.current_stream(device=DEVICE))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                _eager_step(
                    model,
                    optimizer,
                    criterion,
                    static_features,
                    static_labels,
                    float(config.gate_l1),
                    set_to_none=False,
                )
        torch.cuda.current_stream(device=DEVICE).wait_stream(warmup_stream)
        with torch.cuda.graph(graph):
            _eager_step(
                model,
                optimizer,
                criterion,
                static_features,
                static_labels,
                float(config.gate_l1),
                set_to_none=False,
            )
        _restore_model_device_state(model, initial_model)
        _zero_adam_state(optimizer)
        optimizer.zero_grad(set_to_none=False)
        torch.cuda.set_rng_state(initial_rng, device=DEVICE)
        return _CudaGraphStep(
            graph=graph,
            static_features=static_features,
            static_labels=static_labels,
            batch_size=int(config.batch_size),
        )
    except RuntimeError:
        _restore_model_device_state(model, initial_model)
        torch.cuda.set_rng_state(initial_rng, device=DEVICE)
        return None


def build_tuned_hourly_rgfn(
    prepared: PreparedHourlyRgfnTuningSplit,
    config: HourlyRgfnTuningConfig,
) -> nn.Module:
    features = prepared.train.features
    model_config = HourlyRgfnConfig(
        n_continuous=int(features[0].shape[-1]),
        n_static=int(features[3].shape[-1]),
        n_rule_evidence=int(features[4].shape[-1]),
        window_hours=int(features[0].shape[1]),
        sensor_hidden_size=int(config.sensor_hidden_size),
        evidence_hidden_size=int(config.evidence_hidden_size),
        evidence_embed_size=int(config.evidence_embed_size),
        fusion_hidden_size=int(config.gate_hidden_size),
        dropout=float(config.dropout),
        mask_mode=str(config.mask_mode),
    )
    return build_hourly_rgfn(str(config.encoder), config=model_config).to(DEVICE)


def train_tuned_candidate(
    prepared: PreparedHourlyRgfnTuningSplit,
    config: HourlyRgfnTuningConfig,
) -> tuple[nn.Module, dict[str, object]]:
    if prepared.train.labels is None or prepared.validation.labels is None:
        raise ValueError("tuning needs train and validation labels")
    if config.batch_size < 1 or config.max_epochs < 1 or config.patience < 1:
        raise ValueError("training sizes and durations must be positive")
    if config.min_delta < 0.0 or config.fault_class_weight <= 0.0:
        raise ValueError("training values must be non-negative with a positive class weight")
    if config.gate_l1 < 0.0:
        raise ValueError("gate L1 strength must be non-negative")
    set_hourly_rgfn_seed(int(config.seed))
    model = build_tuned_hourly_rgfn(prepared, config)
    optimizer = _make_adam(model, config, capturable=DEVICE.type == "cuda")
    positive_weight = torch.tensor([float(config.fault_class_weight)], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    graph_step = _prepare_cuda_graph_step(model, optimizer, criterion, prepared.train, config)
    if graph_step is None and DEVICE.type == "cuda":
        optimizer = _make_adam(model, config, capturable=False)
    generator = torch.Generator(device=DEVICE.type)
    generator.manual_seed(int(config.seed))
    best_state = _copy_state(model)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    completed = 0
    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        permutation = torch.randperm(prepared.train.count, generator=generator, device=DEVICE)
        for start, stop in _batch_ranges(prepared.train.count, int(config.batch_size)):
            indices = permutation[start:stop]
            if graph_step is not None and int(stop - start) == graph_step.batch_size:
                graph_step.stage(prepared.train.features, prepared.train.labels, indices)
                graph_step.replay()
            else:
                features = tuple(value.index_select(0, indices) for value in prepared.train.features)
                labels = prepared.train.labels.index_select(0, indices)
                _eager_step(
                    model,
                    optimizer,
                    criterion,
                    features,
                    labels,
                    float(config.gate_l1),
                    set_to_none=graph_step is None,
                )
        validation_loss = _validation_loss(
            model,
            prepared.validation,
            criterion,
            float(config.gate_l1),
            max(int(config.batch_size), 512),
        )
        completed = int(epoch)
        if validation_loss < best_loss - float(config.min_delta):
            best_loss = float(validation_loss)
            best_epoch = int(epoch)
            best_state = _copy_state(model)
            stale = 0
        else:
            stale += 1
        if stale >= int(config.patience):
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "epochs_completed": int(completed),
        "best_validation_loss": float(best_loss),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "cuda_graph_used": bool(graph_step is not None),
        "cuda_graph_batch_size": int(config.batch_size) if graph_step is not None else None,
        "config": configuration_payload(config, include_seed=True),
    }


def predict_tuned_candidate(
    model: nn.Module,
    partition: TensorPartition,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start, stop in _batch_ranges(partition.count, int(batch_size)):
            output = model(*[value[start:stop] for value in partition.features])
            values.append(output["binary_prob"].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(values, axis=0) if values else np.empty(0, dtype=np.float32)


def validation_rows_for_candidate(
    model: nn.Module,
    prepared: PreparedHourlyRgfnTuningSplit,
    config: HourlyRgfnTuningConfig,
    thresholds: Sequence[float] = TUNING_THRESHOLDS,
    info: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    if prepared.validation.labels is None:
        raise ValueError("validation labels are required")
    probabilities = predict_tuned_candidate(model, prepared.validation)
    labels = prepared.validation.labels.detach().cpu().numpy().astype(int)
    metadata = {} if info is None else dict(info)
    rows = []
    for threshold in thresholds:
        value = float(threshold)
        if not 0.0 < value < 1.0:
            raise ValueError("thresholds must be strictly between zero and one")
        metric = binary_metrics(labels, probabilities, value)
        rows.append(
            {
                "seed": int(config.seed),
                "configuration_id": configuration_id(config),
                "configuration": configuration_payload(config),
                "fault_class_weight": float(config.fault_class_weight),
                "threshold": value,
                "validation": metric,
                "validation_minimum_metric": float(
                    min(float(metric[name]) for name in ("precision", "recall", "f1"))
                ),
                "best_epoch": int(metadata.get("best_epoch", 0)),
                "best_validation_loss": float(metadata.get("best_validation_loss", np.nan)),
            }
        )
    return rows


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def aggregate_validation_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("validation aggregation requires at least one row")
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["configuration_id"]), float(row["threshold"]))
        grouped.setdefault(key, []).append(dict(row))
    result: list[dict[str, object]] = []
    for (_, threshold), values in sorted(grouped.items(), key=lambda item: item[0]):
        metric_rows = [dict(value["validation"]) for value in values]
        summary = metric_summary(metric_rows)
        validation = {
            name: float(summary[name])
            for name in ("precision", "recall", "f1", "accuracy")
        }
        validation_std = {
            name: float(summary[f"{name}_std"])
            for name in ("precision", "recall", "f1", "accuracy")
        }
        exemplar = values[0]
        result.append(
            {
                "configuration_id": str(exemplar["configuration_id"]),
                "configuration": dict(exemplar["configuration"]),
                "fault_class_weight": float(exemplar["fault_class_weight"]),
                "threshold": float(threshold),
                "validation": validation,
                "validation_std": validation_std,
                "validation_minimum_metric": float(
                    min(validation[name] for name in ("precision", "recall", "f1"))
                ),
                "seed_count": int(len(values)),
            }
        )
    return result


def select_validation_configuration(
    validation_grid: Sequence[dict[str, object]],
    criterion: str = "balanced",
) -> dict[str, object]:
    if not validation_grid:
        raise ValueError("validation selection requires at least one operating point")
    prepared = [dict(row) for row in validation_grid]
    return select_operating_point(prepared, criterion)


def selected_config_from_row(row: dict[str, object], seed: int = 0) -> HourlyRgfnTuningConfig:
    values = dict(row["configuration"])
    values["seed"] = int(seed)
    return HourlyRgfnTuningConfig(**values)


def save_tuned_hourly_rgfn_model(
    path: Path,
    model: nn.Module,
    scaler: HourlyRgfnScaler,
    config: HourlyRgfnTuningConfig,
    selection: dict[str, object],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": str(config.encoder),
            "state_dict": _copy_state(model),
            "scaler": scaler.payload(),
            "tuning_config": configuration_payload(config, include_seed=True),
            "selection": selection,
            "window_hours": int(model.window_hours),
            "n_continuous": int(model.n_continuous),
            "n_static": int(model.n_static),
            "n_rule_evidence": int(model.n_rule_evidence),
            "mask_mode": str(model.mask_mode),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
        destination,
    )
    return destination


def _candidate_path(
    model_dir: Path,
    arm_name: str,
    split_name: str,
    config: HourlyRgfnTuningConfig,
) -> Path:
    digest = hashlib.sha256(configuration_id(config).encode("utf-8")).hexdigest()[:12]
    return Path(model_dir) / "candidates" / (
        f"{arm_name}_{split_name}_{digest}_seed_{int(config.seed)}.pt"
    )


def _save_candidate_model(
    model_dir: Path | None,
    arm_name: str,
    split_name: str,
    model: nn.Module,
    scaler: HourlyRgfnScaler,
    config: HourlyRgfnTuningConfig,
    info: dict[str, object],
) -> str | None:
    if model_dir is None:
        return None
    saved = save_tuned_hourly_rgfn_model(
        _candidate_path(Path(model_dir), arm_name, split_name, config),
        model,
        scaler,
        config,
        {
            "arm": str(arm_name),
            "split": str(split_name),
            "stage": "validation_candidate",
            "configuration": configuration_payload(config),
            "training": dict(info),
        },
    )
    return str(saved)


def _selected_validation_rows(
    rows: Sequence[dict[str, object]],
    selected: dict[str, object],
) -> list[dict[str, object]]:
    matched = [
        dict(row)
        for row in rows
        if str(row["configuration_id"]) == str(selected["configuration_id"])
        and float(row["threshold"]) == float(selected["threshold"])
    ]
    if not matched:
        raise RuntimeError("selected validation operating point has no seed rows")
    result = []
    for row in matched:
        result.append(
            {
                "seed": int(row["seed"]),
                **dict(row["validation"]),
                "best_epoch": int(row["best_epoch"]),
                "best_validation_loss": float(row["best_validation_loss"]),
            }
        )
    return sorted(result, key=lambda row: int(row["seed"]))


def _finalize_from_models(
    arm_name: str,
    split_name: str,
    prepared: PreparedHourlyRgfnTuningSplit,
    models: dict[tuple[str, int], nn.Module],
    infos: dict[tuple[str, int], dict[str, object]],
    validation_rows: list[dict[str, object]],
    validation_grid: list[dict[str, object]],
    candidate_model_paths: Sequence[str] = (),
    search_metadata: dict[str, object] | None = None,
) -> FinalizedHourlyRgfnArm:
    selected = select_validation_configuration(validation_grid, "balanced")
    selected_id = str(selected["configuration_id"])
    selected_models = {
        int(seed): model
        for (identifier, seed), model in models.items()
        if identifier == selected_id
    }
    selected_info = {
        int(seed): dict(info)
        for (identifier, seed), info in infos.items()
        if identifier == selected_id
    }
    if not selected_models:
        raise RuntimeError("selected configuration has no retained model")
    selected_rows = _selected_validation_rows(validation_rows, selected)
    metrics = [{name: row[name] for name in METRIC_NAMES} for row in selected_rows]
    config = selected_config_from_row(selected, seed=min(selected_models))
    return FinalizedHourlyRgfnArm(
        arm_name=str(arm_name),
        split_name=str(split_name),
        prepared=prepared,
        selected_config=config,
        selected_threshold=float(selected["threshold"]),
        models=selected_models,
        model_info=selected_info,
        validation_rows=list(validation_rows),
        validation_grid=list(validation_grid),
        selected_validation_rows=selected_rows,
        selected_validation_summary=metric_summary(metrics),
        candidate_model_paths=list(candidate_model_paths),
        search_metadata={} if search_metadata is None else dict(search_metadata),
    )


def final_evaluate_hourly_rgfn_arm(
    finalized: FinalizedHourlyRgfnArm,
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    model_dir: Path | None = None,
) -> dict[str, object]:
    if finalized.test_ledger.count:
        raise RuntimeError("final test metrics have already been evaluated for this arm")
    test_partition = _make_test_partition(examples, splits, finalized.prepared.scaler)
    if test_partition.labels is None:
        raise RuntimeError("final test labels are required")
    labels = test_partition.labels.detach().cpu().numpy().astype(int)
    test_rows: list[dict[str, object]] = []
    paths: list[str] = []
    for seed, model in sorted(finalized.models.items()):
        probabilities = predict_tuned_candidate(model, test_partition)
        metric = binary_metrics(labels, probabilities, float(finalized.selected_threshold))
        finalized.test_ledger.record(int(seed))
        test_rows.append({"seed": int(seed), **metric})
        if model_dir is not None:
            config = replace(finalized.selected_config, seed=int(seed))
            destination = Path(model_dir) / (
                f"{finalized.arm_name}_{finalized.split_name}_seed_{int(seed)}.pt"
            )
            saved = save_tuned_hourly_rgfn_model(
                destination,
                model,
                finalized.prepared.scaler,
                config,
                {
                    "arm": finalized.arm_name,
                    "split": finalized.split_name,
                    "threshold": float(finalized.selected_threshold),
                    "validation_selection": {
                        "configuration": configuration_payload(finalized.selected_config),
                        "threshold": float(finalized.selected_threshold),
                    },
                    "test": metric,
                },
            )
            paths.append(str(saved))
    summary = metric_summary(test_rows)
    epochs = [float(info["best_epoch"]) for _, info in sorted(finalized.model_info.items())]
    result = {
        "arm": finalized.arm_name,
        "split": finalized.split_name,
        "parameter_count": int(sum(parameter.numel() for parameter in next(iter(finalized.models.values())).parameters())),
        "selected_config": configuration_payload(finalized.selected_config),
        "selected_threshold": float(finalized.selected_threshold),
        "validation_seed_rows": finalized.validation_rows,
        "validation_grid": finalized.validation_grid,
        "best_balanced": {
            "configuration": configuration_payload(finalized.selected_config),
            "fault_class_weight": float(finalized.selected_config.fault_class_weight),
            "threshold": float(finalized.selected_threshold),
            "validation": {
                name: float(finalized.selected_validation_summary[name])
                for name in ("precision", "recall", "f1", "accuracy")
            },
            "validation_std": {
                name: float(finalized.selected_validation_summary[f"{name}_std"])
                for name in ("precision", "recall", "f1", "accuracy")
            },
            "validation_minimum_metric": float(
                min(
                    float(finalized.selected_validation_summary[name])
                    for name in ("precision", "recall", "f1")
                )
            ),
        },
        "selected_validation_seed_rows": finalized.selected_validation_rows,
        "selected_validation_summary": finalized.selected_validation_summary,
        "test_seed_rows": test_rows,
        "test_summary": summary,
        "final_test_target_check": target_check(summary),
        "selected_best_epoch_mean": float(np.mean(epochs)),
        "selected_best_epoch_std": _sample_std(epochs),
        "test_evaluation_count": int(finalized.test_ledger.count),
        "test_evaluation_count_per_seed": 1,
        "selection_source": finalized.selection_source,
        "model_paths": paths,
        "candidate_model_paths": list(finalized.candidate_model_paths),
        "search_metadata": dict(finalized.search_metadata),
    }
    return result


def run_feature_only_arm(
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    split_name: str,
    arm_name: str,
    model_dir: Path | None = None,
    seeds: Sequence[int] = TUNING_SEEDS,
    weights: Sequence[float] = TUNING_WEIGHTS,
    thresholds: Sequence[float] = TUNING_THRESHOLDS,
    base_config: HourlyRgfnTuningConfig | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    if not seeds or not weights or not thresholds:
        raise ValueError("feature arm requires seeds, class weights, and thresholds")
    prepared = prepare_tuning_split(examples, splits)
    base = default_feature_arm_config() if base_config is None else base_config
    models: dict[tuple[str, int], nn.Module] = {}
    infos: dict[tuple[str, int], dict[str, object]] = {}
    validation_rows: list[dict[str, object]] = []
    candidate_paths: list[str] = []
    for weight in weights:
        for seed in seeds:
            config = replace(base, seed=int(seed), fault_class_weight=float(weight))
            model, info = train_tuned_candidate(prepared, config)
            identifier = configuration_id(config)
            models[(identifier, int(seed))] = model
            infos[(identifier, int(seed))] = info
            validation_rows.extend(validation_rows_for_candidate(model, prepared, config, thresholds, info))
            saved = _save_candidate_model(
                model_dir,
                arm_name,
                split_name,
                model,
                prepared.scaler,
                config,
                info,
            )
            if saved is not None:
                candidate_paths.append(saved)
            if progress is not None:
                progress(
                    {
                        "arm": str(arm_name),
                        "split": str(split_name),
                        "seed": int(seed),
                        "fault_class_weight": float(weight),
                        **info,
                    }
                )
    validation_grid = aggregate_validation_rows(validation_rows)
    finalized = _finalize_from_models(
        arm_name,
        split_name,
        prepared,
        models,
        infos,
        validation_rows,
        validation_grid,
        candidate_paths,
        {
            "mode": "feature_weight_sweep",
            "candidate_training_count": int(len(seeds) * len(weights)),
            "seed_count": int(len(seeds)),
            "weight_count": int(len(weights)),
            "threshold_count": int(len(thresholds)),
        },
    )
    result = final_evaluate_hourly_rgfn_arm(finalized, examples, splits, model_dir)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def make_coarse_search_configurations(
    count: int = 24,
    random_seed: int = 20260723,
    base_config: HourlyRgfnTuningConfig | None = None,
) -> tuple[HourlyRgfnTuningConfig, ...]:
    if int(count) < 10:
        raise ValueError("coarse search needs at least ten configurations for coverage")
    base = default_search_config() if base_config is None else base_config
    coverage = [
        (32, 16, 8, 0.0, 3e-4, 1e-4, 1.0),
        (48, 32, 16, 0.01, 1e-3, 1e-3, 2.0),
        (64, 64, 32, 0.03, 3e-3, 1e-4, 4.0),
        (96, 16, 8, 0.0, 1e-3, 1e-3, 6.0),
        (32, 32, 16, 0.01, 3e-3, 1e-4, 8.0),
        (48, 64, 32, 0.03, 3e-4, 1e-3, 1.0),
        (64, 16, 8, 0.01, 1e-3, 1e-4, 2.0),
        (96, 32, 16, 0.03, 3e-3, 1e-3, 4.0),
        (32, 64, 32, 0.0, 3e-4, 1e-4, 6.0),
        (48, 16, 8, 0.01, 1e-3, 1e-3, 8.0),
    ]
    candidates: list[tuple[int, int, int, float, float, float, float]] = list(coverage)
    universe = [
        (sensor, evidence, gate, gate_l1, learning_rate, weight_decay, weight)
        for sensor in SEARCH_SENSOR_WIDTHS
        for evidence in SEARCH_EVIDENCE_WIDTHS
        for gate in SEARCH_GATE_WIDTHS
        for gate_l1 in SEARCH_GATE_L1
        for learning_rate in SEARCH_LEARNING_RATES
        for weight_decay in SEARCH_WEIGHT_DECAYS
        for weight in TUNING_WEIGHTS
    ]
    generator = np.random.default_rng(int(random_seed))
    order = generator.permutation(len(universe))
    for index in order:
        candidate = universe[int(index)]
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) >= int(count):
            break
    result = []
    for sensor, evidence, gate, gate_l1, learning_rate, weight_decay, weight in candidates[: int(count)]:
        result.append(
            replace(
                base,
                sensor_hidden_size=int(sensor),
                evidence_hidden_size=int(evidence),
                gate_hidden_size=int(gate),
                gate_l1=float(gate_l1),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                fault_class_weight=float(weight),
            )
        )
    return tuple(result)


def search_configuration_coverage(
    configs: Sequence[HourlyRgfnTuningConfig],
) -> dict[str, object]:
    if not configs:
        raise ValueError("configuration coverage requires at least one configuration")
    fields = (
        "sensor_hidden_size",
        "evidence_hidden_size",
        "gate_hidden_size",
        "gate_l1",
        "learning_rate",
        "weight_decay",
        "fault_class_weight",
    )
    return {
        "candidate_count": int(len(configs)),
        "sensor_hidden_size": sorted({int(config.sensor_hidden_size) for config in configs}),
        "evidence_hidden_size": sorted({int(config.evidence_hidden_size) for config in configs}),
        "gate_hidden_size": sorted({int(config.gate_hidden_size) for config in configs}),
        "gate_l1": sorted({float(config.gate_l1) for config in configs}),
        "learning_rate": sorted({float(config.learning_rate) for config in configs}),
        "weight_decay": sorted({float(config.weight_decay) for config in configs}),
        "fault_class_weight": sorted({float(config.fault_class_weight) for config in configs}),
        "fields": list(fields),
    }


def screen_architecture_search(
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    split_name: str,
    candidate_configs: Sequence[HourlyRgfnTuningConfig],
    thresholds: Sequence[float] = TUNING_THRESHOLDS,
    screening_seed: int = 0,
    model_dir: Path | None = None,
    arm_name: str = "arm2_screen",
    progress: Callable[[dict[str, object]], None] | None = None,
) -> ArchitectureScreen:
    if not candidate_configs:
        raise ValueError("architecture screening needs candidate configurations")
    prepared = prepare_tuning_split(examples, splits)
    rows: list[dict[str, object]] = []
    configs: dict[str, HourlyRgfnTuningConfig] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    infos: dict[str, dict[str, object]] = {}
    candidate_paths: list[str] = []
    for candidate in candidate_configs:
        config = replace(candidate, seed=int(screening_seed))
        identifier = configuration_id(config)
        if identifier in configs:
            continue
        model, info = train_tuned_candidate(prepared, config)
        configs[identifier] = config
        states[identifier] = _copy_state(model)
        infos[identifier] = dict(info)
        rows.extend(validation_rows_for_candidate(model, prepared, config, thresholds, info))
        saved = _save_candidate_model(
            model_dir,
            arm_name,
            split_name,
            model,
            prepared.scaler,
            config,
            info,
        )
        if saved is not None:
            candidate_paths.append(saved)
        if progress is not None:
            progress(
                {
                    "arm": "architecture_screen",
                    "split": str(split_name),
                    "seed": int(screening_seed),
                    "candidate": int(len(configs)),
                    "candidate_count": int(len(candidate_configs)),
                    **info,
                }
            )
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return ArchitectureScreen(
        prepared=prepared,
        split_name=str(split_name),
        candidate_configs=configs,
        candidate_states=states,
        candidate_info=infos,
        validation_rows=rows,
        validation_grid=aggregate_validation_rows(rows),
        screening_seed=int(screening_seed),
        coverage=search_configuration_coverage(tuple(configs.values())),
        candidate_model_paths=candidate_paths,
    )


def select_top_screened_configurations(
    screen: ArchitectureScreen,
    top_count: int = 3,
    criterion: str = "balanced",
) -> tuple[HourlyRgfnTuningConfig, ...]:
    if int(top_count) < 1:
        raise ValueError("top configuration count must be positive")
    per_config: dict[str, dict[str, object]] = {}
    for row in screen.validation_grid:
        identifier = str(row["configuration_id"])
        current = per_config.get(identifier)
        if current is None:
            per_config[identifier] = dict(row)
            continue
        selected = select_validation_configuration([current, dict(row)], criterion)
        per_config[identifier] = selected
    ordered = sorted(
        per_config.values(),
        key=lambda row: (
            float(row["validation_minimum_metric"]) if criterion == "balanced" else float(row["validation"]["f1"]),
            float(row["validation"]["f1"]),
            float(row["validation"]["precision"]),
            float(row["validation"]["recall"]),
            str(row["configuration_id"]),
        ),
        reverse=True,
    )
    result = []
    for row in ordered[: int(top_count)]:
        result.append(selected_config_from_row(row, seed=0))
    return tuple(result)


def finalize_architecture_search(
    screen: ArchitectureScreen,
    arm_name: str,
    top_configs: Sequence[HourlyRgfnTuningConfig] | None = None,
    top_count: int = 3,
    seeds: Sequence[int] = TUNING_SEEDS,
    thresholds: Sequence[float] = TUNING_THRESHOLDS,
    model_dir: Path | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> FinalizedHourlyRgfnArm:
    selected_configs = (
        tuple(top_configs)
        if top_configs is not None
        else select_top_screened_configurations(screen, top_count=top_count)
    )
    if not selected_configs or not seeds:
        raise ValueError("architecture finalization requires selected configurations and seeds")
    if int(screen.screening_seed) not in {int(seed) for seed in seeds}:
        raise ValueError("finalization seeds must include the screening seed")
    unique_configs: dict[str, HourlyRgfnTuningConfig] = {}
    for candidate in selected_configs:
        identifier = configuration_id(candidate)
        if identifier not in screen.candidate_configs:
            raise ValueError("architecture finalization received a configuration absent from screening")
        unique_configs[identifier] = replace(candidate, seed=int(screen.screening_seed))
    models: dict[tuple[str, int], nn.Module] = {}
    infos: dict[tuple[str, int], dict[str, object]] = {}
    rows: list[dict[str, object]] = [
        dict(row)
        for row in screen.validation_rows
        if str(row["configuration_id"]) in unique_configs
    ]
    candidate_paths = list(screen.candidate_model_paths)
    for identifier, candidate in unique_configs.items():
        if int(screen.screening_seed) in {int(seed) for seed in seeds}:
            model = build_tuned_hourly_rgfn(screen.prepared, candidate)
            model.load_state_dict(screen.candidate_states[identifier])
            models[(identifier, int(screen.screening_seed))] = model
            infos[(identifier, int(screen.screening_seed))] = dict(screen.candidate_info[identifier])
        for seed in seeds:
            if int(seed) == int(screen.screening_seed):
                continue
            config = replace(candidate, seed=int(seed))
            model, info = train_tuned_candidate(screen.prepared, config)
            models[(identifier, int(seed))] = model
            infos[(identifier, int(seed))] = info
            rows.extend(validation_rows_for_candidate(model, screen.prepared, config, thresholds, info))
            saved = _save_candidate_model(
                model_dir,
                arm_name,
                screen.split_name,
                model,
                screen.prepared.scaler,
                config,
                info,
            )
            if saved is not None:
                candidate_paths.append(saved)
            if progress is not None:
                progress(
                    {
                        "arm": str(arm_name),
                        "split": screen.split_name,
                        "seed": int(seed),
                        "configuration_id": identifier,
                        **info,
                    }
                )
    validation_grid = aggregate_validation_rows(rows)
    return _finalize_from_models(
        arm_name,
        screen.split_name,
        screen.prepared,
        models,
        infos,
        rows,
        validation_grid,
        candidate_paths,
        {
            "mode": "coarse_architecture_search",
            "screening_seed": int(screen.screening_seed),
            "screening_coverage": dict(screen.coverage),
            "screening_candidate_count": int(len(screen.candidate_configs)),
            "finalist_count": int(len(unique_configs)),
            "finalist_configuration_ids": sorted(unique_configs),
            "seed_count": int(len(seeds)),
            "threshold_count": int(len(thresholds)),
        },
    )


def run_combined_arm(
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    split_name: str,
    arm_name: str,
    architecture_config: HourlyRgfnTuningConfig,
    selected_threshold: float,
    model_dir: Path | None = None,
    seeds: Sequence[int] = TUNING_SEEDS,
    resweep_operating_point: bool = False,
    weights: Sequence[float] = TUNING_WEIGHTS,
    thresholds: Sequence[float] = TUNING_THRESHOLDS,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    threshold = float(selected_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("combined arm threshold must be strictly between zero and one")
    result = run_feature_only_arm(
        examples=examples,
        splits=splits,
        split_name=split_name,
        arm_name=arm_name,
        model_dir=model_dir,
        seeds=seeds,
        weights=weights if resweep_operating_point else (float(architecture_config.fault_class_weight),),
        thresholds=thresholds if resweep_operating_point else (threshold,),
        base_config=architecture_config,
        progress=progress,
    )
    result["combined_operating_point_source"] = (
        "augmented_validation_resweep" if resweep_operating_point else "transferred_arm2_validation_choice"
    )
    result["transferred_arm2_config"] = configuration_payload(architecture_config)
    result["transferred_arm2_threshold"] = threshold
    result["search_metadata"] = {
        **dict(result["search_metadata"]),
        "mode": "combined_augmentation",
        "operating_point_source": result["combined_operating_point_source"],
    }
    return result


def _comparison_metrics(payload: dict[str, object]) -> dict[str, object]:
    if "test_summary" in payload:
        source = dict(payload["test_summary"])
    elif "final_test" in payload:
        source = dict(payload["final_test"])
    else:
        source = dict(payload)
    required = ("precision", "recall", "f1")
    missing = [name for name in required if name not in source]
    if missing:
        raise KeyError(f"comparison metrics are missing: {missing}")
    return source


def master_tuning_comparison_frame(
    logistic: dict[str, dict[str, object]],
    gradient_boosted: dict[str, dict[str, object]],
    prior_rgfn_gru: dict[str, dict[str, object]],
    arm1: dict[str, dict[str, object]],
    arm2: dict[str, dict[str, object]],
    arm3: dict[str, dict[str, object]],
) -> pd.DataFrame:
    sources = (
        ("Logistic regression", logistic),
        ("Gradient boosted", gradient_boosted),
        ("Prior RGFN-GRU", prior_rgfn_gru),
        ("RGFN Arm 1 features", arm1),
        ("RGFN Arm 2 search", arm2),
        ("RGFN Arm 3 combined", arm3),
    )
    rows: list[dict[str, object]] = []
    for model_name, source in sources:
        for split in ("random", "spaced"):
            if split not in source:
                raise KeyError(f"comparison source lacks {model_name} {split}")
            payload = dict(source[split])
            metric = _comparison_metrics(payload)
            row: dict[str, object] = {
                "model": model_name,
                "split": split,
                "precision": float(metric["precision"]),
                "recall": float(metric["recall"]),
                "f1": float(metric["f1"]),
                "precision_std": float(metric.get("precision_std", np.nan)),
                "recall_std": float(metric.get("recall_std", np.nan)),
                "f1_std": float(metric.get("f1_std", np.nan)),
                "parameter_count": payload.get("parameter_count"),
                "selection_source": payload.get("selection_source", "saved_metric"),
                "test_evaluation_count": payload.get("test_evaluation_count"),
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "model"]).reset_index(drop=True)


def arm_result_payload(result: dict[str, object]) -> dict[str, object]:
    return {
        key: result[key]
        for key in (
            "arm",
            "split",
            "parameter_count",
            "selected_config",
            "selected_threshold",
            "best_balanced",
            "selected_validation_summary",
            "test_summary",
            "final_test_target_check",
            "test_evaluation_count",
            "test_evaluation_count_per_seed",
            "selection_source",
            "model_paths",
            "candidate_model_paths",
            "search_metadata",
        )
    }
