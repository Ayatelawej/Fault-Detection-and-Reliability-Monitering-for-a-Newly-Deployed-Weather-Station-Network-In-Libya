from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.model.hourly_detection import MASK_MODE_PER_FEATURE, MASK_MODE_PER_HOUR, MASK_MODES


SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
SPLIT_FRACTIONS_80_20 = {
    "train": 0.80,
    "test": 0.20,
}
SPACED_SPLIT_VERSION = "hourly-spaced-v1"
REQUIRED_TENSOR_KEYS = (
    "X_cont",
    "mask",
    "time_since_last",
    "static",
    "rule_evidence",
    "y_binary",
    "station_id",
    "hour",
    "display_state",
    "source_episode_ids",
    "continuous_feature_names",
    "static_feature_names",
    "rule_evidence_feature_names",
)


@dataclass(frozen=True)
class HourlyBaselineConfig:
    seed: int = 2026
    fault_class_weight: float = 13.0
    threshold: float = 0.5
    learning_rate: float = 0.05
    max_iter: int = 100
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0


def load_hourly_tensor(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_hourly_metadata(path: Path) -> dict[str, np.ndarray]:
    keys = (
        "y_binary",
        "station_id",
        "hour",
        "display_state",
        "source_episode_ids",
        "window_hours",
    )
    with np.load(path, allow_pickle=True) as data:
        missing = sorted(set(keys).difference(data.files))
        if missing:
            raise KeyError(f"hourly tensor lacks metadata fields: {missing}")
        return {key: data[key] for key in keys}


def validate_tensor(examples: dict[str, np.ndarray]) -> None:
    missing = sorted(set(REQUIRED_TENSOR_KEYS).difference(examples))
    if missing:
        raise KeyError(f"hourly tensor fields missing: {missing}")
    count = len(examples["y_binary"])
    if count == 0:
        raise ValueError("hourly tensor has no examples")
    sample_keys = (
        "X_cont",
        "mask",
        "time_since_last",
        "static",
        "rule_evidence",
        "station_id",
        "hour",
        "display_state",
        "source_episode_ids",
    )
    for key in sample_keys:
        value = np.asarray(examples[key])
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError(f"hourly tensor field has an unexpected sample dimension: {key}")
    labels = np.asarray(examples["y_binary"])
    valid = np.isin(labels, [0, 1])
    states = np.asarray(examples["display_state"], dtype=object).astype(str)
    invalid = ~valid & ~np.equal(states, "excluded")
    if invalid.any():
        raise ValueError("hourly tensor has non-binary labels outside excluded rows")


def filter_eligible_examples(examples: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    validate_tensor(examples)
    count = len(examples["y_binary"])
    labels = np.asarray(examples["y_binary"])
    states = np.asarray(examples["display_state"], dtype=object).astype(str)
    eligible = np.not_equal(states, "excluded") & np.isin(labels, [0, 1])
    result: dict[str, np.ndarray] = {}
    for key, value in examples.items():
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == count:
            result[key] = value[eligible]
        else:
            result[key] = value
    if not len(result["y_binary"]):
        raise ValueError("no eligible hourly examples remain")
    result_states = np.asarray(result["display_state"], dtype=object).astype(str)
    if np.equal(result_states, "excluded").any():
        raise RuntimeError("excluded examples remained after eligibility filtering")
    if not np.isin(result["y_binary"], [0, 1]).all():
        raise RuntimeError("eligible examples have non-binary labels")
    return result, int((~eligible).sum())


def assert_matching_metadata(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> None:
    keys = ("y_binary", "station_id", "hour", "display_state", "source_episode_ids")
    for key in keys:
        if key not in left or key not in right:
            raise KeyError(f"metadata field missing for tensor comparison: {key}")
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"hourly tensors disagree on {key}")


def resolve_fault_class_weight(
    labels: np.ndarray,
    requested_weight: float | None = None,
) -> float:
    if requested_weight is not None:
        value = float(requested_weight)
        if value <= 0.0:
            raise ValueError("fault class weight must be positive")
        return value
    labels = np.asarray(labels, dtype=int)
    fault_count = int(np.equal(labels, 1).sum())
    not_fault_count = int(np.equal(labels, 0).sum())
    if fault_count == 0 or not_fault_count == 0:
        raise ValueError("both binary classes are required to resolve fault class weight")
    return float(not_fault_count / fault_count)


def _feature_names(examples: dict[str, np.ndarray]) -> tuple[list[str], dict[str, np.ndarray]]:
    continuous = [str(value) for value in examples["continuous_feature_names"]]
    static = [str(value) for value in examples["static_feature_names"]]
    rules = [str(value) for value in examples["rule_evidence_feature_names"]]
    window_hours = int(np.asarray(examples["X_cont"]).shape[1])
    per_hour_width = len(continuous) + 2
    names: list[str] = []
    groups: dict[str, list[int]] = {f"continuous:{name}": [] for name in continuous}
    groups["mask:all_lags"] = []
    groups["time_since_last:all_lags"] = []
    position = 0
    for hour_index in range(window_hours):
        hours_ago = window_hours - hour_index - 1
        suffix = "t0" if hours_ago == 0 else f"tminus_{hours_ago}h"
        for feature_index, name in enumerate(continuous):
            names.append(f"{name}_{suffix}")
            groups[f"continuous:{name}"].append(position + feature_index)
        names.append(f"row_present_{suffix}")
        groups["mask:all_lags"].append(position + len(continuous))
        names.append(f"time_since_last_{suffix}")
        groups["time_since_last:all_lags"].append(position + len(continuous) + 1)
        position += per_hour_width
    for name in rules:
        names.append(name)
        groups[f"rule:{name}"] = [position]
        position += 1
    for name in static:
        names.append(name)
        groups[f"static:{name}"] = [position]
        position += 1
    return names, {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in groups.items()
    }


def _flatten_hourly_features_per_hour(
    examples: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    validate_tensor(examples)
    continuous = np.asarray(examples["X_cont"], dtype=np.float32)
    mask = np.asarray(examples["mask"], dtype=np.float32)
    elapsed = np.asarray(examples["time_since_last"], dtype=np.float32)
    static = np.asarray(examples["static"], dtype=np.float32)
    rules = np.asarray(examples["rule_evidence"], dtype=np.float32)
    if continuous.ndim != 3 or mask.shape[:2] != continuous.shape[:2] or elapsed.shape[:2] != continuous.shape[:2]:
        raise ValueError("hourly temporal arrays do not share a sample and window shape")
    if mask.shape[2] != 1 or elapsed.shape[2] != 1:
        raise ValueError("hourly mask and elapsed arrays must each have one channel")
    count, window_hours, continuous_width = continuous.shape
    dimension = window_hours * (continuous_width + 2) + rules.shape[1] + static.shape[1]
    values = np.empty((count, dimension), dtype=np.float32)
    cursor = 0
    for hour_index in range(window_hours):
        values[:, cursor:cursor + continuous_width] = continuous[:, hour_index, :]
        cursor += continuous_width
        values[:, cursor] = mask[:, hour_index, 0]
        cursor += 1
        values[:, cursor] = elapsed[:, hour_index, 0]
        cursor += 1
    values[:, cursor:cursor + rules.shape[1]] = rules
    cursor += rules.shape[1]
    values[:, cursor:cursor + static.shape[1]] = static
    cursor += static.shape[1]
    if cursor != dimension:
        raise RuntimeError("hourly feature flattening ended at an unexpected dimension")
    values[~np.isfinite(values)] = np.nan
    names, groups = _feature_names(examples)
    if len(names) != dimension:
        raise RuntimeError("hourly feature names do not match flattened dimension")
    return values, names, groups


def _feature_names_per_feature(
    examples: dict[str, np.ndarray],
) -> tuple[list[str], dict[str, np.ndarray]]:
    continuous = [str(value) for value in examples["continuous_feature_names"]]
    static = [str(value) for value in examples["static_feature_names"]]
    rules = [str(value) for value in examples["rule_evidence_feature_names"]]
    window_hours = int(np.asarray(examples["X_cont"]).shape[1])
    n_continuous = len(continuous)
    per_hour_width = n_continuous * 2 + 1
    names: list[str] = []
    groups: dict[str, list[int]] = {f"continuous:{name}": [] for name in continuous}
    groups.update({f"feature_mask:{name}": [] for name in continuous})
    groups["time_since_last:all_lags"] = []
    position = 0
    for hour_index in range(window_hours):
        hours_ago = window_hours - hour_index - 1
        suffix = "t0" if hours_ago == 0 else f"tminus_{hours_ago}h"
        for feature_index, name in enumerate(continuous):
            names.append(f"{name}_{suffix}")
            groups[f"continuous:{name}"].append(position + feature_index)
        for feature_index, name in enumerate(continuous):
            names.append(f"{name}_present_{suffix}")
            groups[f"feature_mask:{name}"].append(position + n_continuous + feature_index)
        names.append(f"time_since_last_{suffix}")
        groups["time_since_last:all_lags"].append(position + n_continuous * 2)
        position += per_hour_width
    for name in rules:
        names.append(name)
        groups[f"rule:{name}"] = [position]
        position += 1
    for name in static:
        names.append(name)
        groups[f"static:{name}"] = [position]
        position += 1
    return names, {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in groups.items()
    }


def _flatten_hourly_features_per_feature(
    examples: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    validate_tensor(examples)
    continuous = np.asarray(examples["X_cont"], dtype=np.float32)
    mask = np.asarray(examples["mask"], dtype=np.float32)
    elapsed = np.asarray(examples["time_since_last"], dtype=np.float32)
    static = np.asarray(examples["static"], dtype=np.float32)
    rules = np.asarray(examples["rule_evidence"], dtype=np.float32)
    if continuous.ndim != 3 or mask.shape[:2] != continuous.shape[:2] or elapsed.shape[:2] != continuous.shape[:2]:
        raise ValueError("hourly temporal arrays do not share a sample and window shape")
    if mask.shape[2] != continuous.shape[2] or elapsed.shape[2] != 1:
        raise ValueError("hourly per-feature mask and elapsed arrays must match the expected channel widths")
    count, window_hours, continuous_width = continuous.shape
    dimension = window_hours * (continuous_width * 2 + 1) + rules.shape[1] + static.shape[1]
    values = np.empty((count, dimension), dtype=np.float32)
    cursor = 0
    for hour_index in range(window_hours):
        values[:, cursor:cursor + continuous_width] = continuous[:, hour_index, :]
        cursor += continuous_width
        values[:, cursor:cursor + continuous_width] = mask[:, hour_index, :]
        cursor += continuous_width
        values[:, cursor] = elapsed[:, hour_index, 0]
        cursor += 1
    values[:, cursor:cursor + rules.shape[1]] = rules
    cursor += rules.shape[1]
    values[:, cursor:cursor + static.shape[1]] = static
    cursor += static.shape[1]
    if cursor != dimension:
        raise RuntimeError("hourly feature flattening ended at an unexpected dimension")
    values[~np.isfinite(values)] = np.nan
    names, groups = _feature_names_per_feature(examples)
    if len(names) != dimension:
        raise RuntimeError("hourly feature names do not match flattened dimension")
    return values, names, groups


def flatten_hourly_features(
    examples: dict[str, np.ndarray],
    mask_mode: str = MASK_MODE_PER_HOUR,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    resolved_mask_mode = str(mask_mode)
    if resolved_mask_mode == MASK_MODE_PER_HOUR:
        return _flatten_hourly_features_per_hour(examples)
    if resolved_mask_mode == MASK_MODE_PER_FEATURE:
        return _flatten_hourly_features_per_feature(examples)
    raise ValueError(f"unknown hourly mask mode: {mask_mode}; expected one of {MASK_MODES}")


def allocation_counts(
    total: int,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> dict[str, int]:
    if total < 0:
        raise ValueError("allocation total must be non-negative")
    names = tuple(fractions)
    raw = {name: total * fraction for name, fraction in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    ranked = sorted(
        names,
        key=lambda name: (raw[name] - counts[name], -names.index(name)),
        reverse=True,
    )
    for name in ranked[:remaining]:
        counts[name] += 1
    return counts


def _split_indices_from_assignments(
    assignments: np.ndarray,
    names: tuple[str, ...] = SPLIT_NAMES,
) -> dict[str, np.ndarray]:
    return {
        name: np.flatnonzero(assignments == name).astype(np.int64)
        for name in names
    }


def validate_splits(
    splits: dict[str, np.ndarray],
    labels: np.ndarray,
) -> None:
    labels = np.asarray(labels, dtype=int)
    count = len(labels)
    names = tuple(splits)
    pieces = [np.asarray(splits[name], dtype=np.int64) for name in names]
    combined = np.concatenate(pieces)
    if len(combined) != count or len(np.unique(combined)) != count or set(combined.tolist()) != set(range(count)):
        raise ValueError("split membership is not disjoint and complete")
    for name, indices in zip(names, pieces):
        if not len(indices):
            raise ValueError(f"split is empty: {name}")
        values = labels[indices]
        if not np.isin(values, [0, 1]).all() or not np.equal(values, 0).any() or not np.equal(values, 1).any():
            raise ValueError(f"split does not contain both binary classes: {name}")


def random_split_indices(
    labels: np.ndarray,
    seed: int,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    source = np.arange(len(labels), dtype=np.int64)
    names = tuple(fractions)
    if names == ("train", "test"):
        train, test = train_test_split(
            source,
            test_size=float(fractions["test"]),
            random_state=int(seed),
            stratify=labels,
        )
        result = {
            "train": np.sort(train.astype(np.int64)),
            "test": np.sort(test.astype(np.int64)),
        }
    elif names == ("train", "validation", "test"):
        train, remainder = train_test_split(
            source,
            test_size=float(fractions["validation"] + fractions["test"]),
            random_state=int(seed),
            stratify=labels,
        )
        validation, test = train_test_split(
            remainder,
            test_size=float(fractions["test"] / (fractions["validation"] + fractions["test"])),
            random_state=int(seed) + 1,
            stratify=labels[remainder],
        )
        result = {
            "train": np.sort(train.astype(np.int64)),
            "validation": np.sort(validation.astype(np.int64)),
            "test": np.sort(test.astype(np.int64)),
        }
    else:
        raise ValueError(f"unsupported random split names: {names}")
    validate_splits(result, labels)
    return result


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fault_groups(
    labels: np.ndarray,
    source_episode_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    positive = np.flatnonzero(np.equal(labels, 1)).astype(np.int64)
    parent: dict[str, str] = {}

    def find(token: str) -> str:
        parent.setdefault(token, token)
        if parent[token] != token:
            parent[token] = find(parent[token])
        return parent[token]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    row_tokens: dict[int, list[str]] = {}
    for index in positive:
        tokens = [token for token in str(source_episode_ids[index]).split("|") if token]
        if not tokens:
            tokens = [f"unlinked_{index}"]
        row_tokens[int(index)] = tokens
        for token in tokens[1:]:
            union(tokens[0], token)
        find(tokens[0])
    groups: dict[str, list[int]] = {}
    for index in positive:
        root = find(row_tokens[int(index)][0])
        groups.setdefault(root, []).append(int(index))
    return {
        "|".join(sorted({token for index in indices for token in row_tokens[index]})): np.asarray(indices, dtype=np.int64)
        for _, indices in groups.items()
    }


def _fault_strata(
    groups: dict[str, np.ndarray],
    hours: np.ndarray,
    partition_count: int = len(SPLIT_NAMES),
) -> list[tuple[str, list[tuple[str, np.ndarray]]]]:
    parsed = pd.to_datetime(pd.Series(hours), utc=True).dt.tz_localize(None)
    by_month: dict[pd.Period, list[tuple[str, np.ndarray]]] = {}
    for group_id, indices in groups.items():
        month = parsed.iloc[indices].min().to_period("M")
        by_month.setdefault(month, []).append((group_id, indices))
    if not by_month:
        return []
    months = pd.period_range(min(by_month), max(by_month), freq="M")
    result: list[tuple[str, list[tuple[str, np.ndarray]]]] = []
    pending: list[tuple[str, np.ndarray]] = []
    pending_start: pd.Period | None = None
    for month in months:
        if pending_start is None:
            pending_start = month
        pending.extend(by_month.get(month, []))
        if len(pending) >= partition_count:
            result.append((f"{pending_start.strftime('%Y-%m')}_to_{month.strftime('%Y-%m')}", pending))
            pending = []
            pending_start = None
    if pending:
        if result:
            previous_name, previous_groups = result[-1]
            result[-1] = (f"{previous_name}_plus_tail", previous_groups + pending)
        else:
            end = months[-1]
            result.append((f"{pending_start.strftime('%Y-%m')}_to_{end.strftime('%Y-%m')}", pending))
    return result


def _assign_fault_stratum(
    groups: list[tuple[str, np.ndarray]],
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> dict[str, np.ndarray]:
    names = tuple(fractions)
    quotas = {name: 0.0 for name in names}
    total = float(sum(len(indices) for _, indices in groups))
    for name, fraction in fractions.items():
        quotas[name] = total * fraction
    weights = {name: 0.0 for name in names}
    assigned_groups = {name: 0 for name in names}
    assignment: dict[str, np.ndarray] = {}
    ordered = sorted(groups, key=lambda item: (-len(item[1]), _digest(f"{SPACED_SPLIT_VERSION}|{item[0]}")))
    for position, (group_id, indices) in enumerate(ordered):
        remaining_groups = len(ordered) - position
        empty = [name for name in names if assigned_groups[name] == 0]
        candidates = empty if empty and remaining_groups == len(empty) else list(names)
        scores: dict[str, float] = {}
        for name in candidates:
            proposed = dict(weights)
            proposed[name] += len(indices)
            scores[name] = float(
                sum(
                    ((proposed[split_name] - quotas[split_name]) / max(quotas[split_name], 1.0)) ** 2
                    for split_name in names
                )
            )
        selected = min(names, key=lambda name: (scores.get(name, np.inf), names.index(name)))
        weights[selected] += len(indices)
        assigned_groups[selected] += 1
        assignment[group_id] = np.full(len(indices), selected, dtype=object)
    return assignment


def _assign_non_fault_rows(
    assignments: np.ndarray,
    labels: np.ndarray,
    station_ids: np.ndarray,
    hours: np.ndarray,
    display_states: np.ndarray,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> None:
    parsed = pd.to_datetime(pd.Series(hours), utc=True).dt.tz_localize(None)
    strata: dict[tuple[str, str], list[int]] = {}
    for index in np.flatnonzero(np.equal(labels, 0)):
        key = (str(parsed.iloc[index].to_period("M")), str(display_states[index]))
        strata.setdefault(key, []).append(int(index))
    for key, indices in strata.items():
        ordered = sorted(
            indices,
            key=lambda index: _digest(
                f"{SPACED_SPLIT_VERSION}|{key[0]}|{key[1]}|{station_ids[index]}|{hours[index]}"
            ),
        )
        counts = allocation_counts(len(ordered), fractions)
        start = 0
        for name in fractions:
            end = start + counts[name]
            assignments[np.asarray(ordered[start:end], dtype=np.int64)] = name
            start = end


def spaced_split_indices(
    labels: np.ndarray,
    station_ids: np.ndarray,
    hours: np.ndarray,
    display_states: np.ndarray,
    source_episode_ids: np.ndarray,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    labels = np.asarray(labels, dtype=int)
    station_ids = np.asarray(station_ids, dtype=object).astype(str)
    hours = np.asarray(hours, dtype=object).astype(str)
    display_states = np.asarray(display_states, dtype=object).astype(str)
    source_episode_ids = np.asarray(source_episode_ids, dtype=object).astype(str)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("spaced split requires only eligible binary labels")
    assignments = np.full(len(labels), "", dtype=object)
    fault_groups = _fault_groups(labels, source_episode_ids)
    names = tuple(fractions)
    strata = _fault_strata(fault_groups, hours, partition_count=len(names))
    fault_group_assignment: dict[str, str] = {}
    stratum_rows = []
    for stratum_name, groups in strata:
        local = _assign_fault_stratum(groups, fractions)
        counts = {name: 0 for name in names}
        groups_by_split = {name: 0 for name in names}
        ordered = sorted(groups, key=lambda item: (-len(item[1]), _digest(f"{SPACED_SPLIT_VERSION}|{item[0]}")))
        for group_id, indices in ordered:
            values = local[group_id]
            split_name = str(values[0])
            assignments[indices] = split_name
            fault_group_assignment[group_id] = split_name
            counts[split_name] += len(indices)
            groups_by_split[split_name] += 1
        stratum_rows.append(
            {
                "stratum": stratum_name,
                **{f"{name}_fault_hours": int(counts[name]) for name in names},
                **{f"{name}_fault_groups": int(groups_by_split[name]) for name in names},
            }
        )
    _assign_non_fault_rows(assignments, labels, station_ids, hours, display_states, fractions)
    if np.equal(assignments, "").any():
        raise RuntimeError("spaced split left examples without a partition")
    result = _split_indices_from_assignments(assignments, names)
    validate_splits(result, labels)
    detail = {
        "version": SPACED_SPLIT_VERSION,
        "fault_group_count": int(len(fault_groups)),
        "fault_strata": stratum_rows,
        "fault_group_assignment": fault_group_assignment,
    }
    return result, detail


def split_summary(
    splits: dict[str, np.ndarray],
    labels: np.ndarray,
    hours: np.ndarray,
    display_states: np.ndarray,
) -> list[dict[str, object]]:
    parsed = pd.to_datetime(pd.Series(hours), utc=True)
    rows = []
    for name in splits:
        indices = np.asarray(splits[name], dtype=np.int64)
        subset_labels = np.asarray(labels, dtype=int)[indices]
        subset_states = np.asarray(display_states, dtype=object).astype(str)[indices]
        faults = indices[np.equal(subset_labels, 1)]
        rows.append(
            {
                "split": name,
                "hours": int(len(indices)),
                "fault_hours": int(np.equal(subset_labels, 1).sum()),
                "not_fault_hours": int(np.equal(subset_labels, 0).sum()),
                "benign_hours": int(np.equal(subset_states, "benign").sum()),
                "clean_hours": int(np.equal(subset_states, "clean").sum()),
                "first_fault_hour": "" if not len(faults) else str(parsed.iloc[faults].min()),
                "last_fault_hour": "" if not len(faults) else str(parsed.iloc[faults].max()),
            }
        )
    return rows


def make_classifier(config: HourlyBaselineConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(config.learning_rate),
        max_iter=int(config.max_iter),
        max_leaf_nodes=int(config.max_leaf_nodes),
        min_samples_leaf=int(config.min_samples_leaf),
        l2_regularization=float(config.l2_regularization),
        early_stopping=False,
        random_state=int(config.seed),
        class_weight={0: 1.0, 1: float(config.fault_class_weight)},
    )


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(probabilities >= float(threshold), dtype=int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "tp": int(matrix[1, 1]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tn": int(matrix[0, 0]),
    }


def train_baseline_run(
    values: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    window_name: str,
    split_name: str,
    config: HourlyBaselineConfig,
    split_configuration: str | None = None,
) -> tuple[dict[str, object], HistGradientBoostingClassifier]:
    labels = np.asarray(labels, dtype=int)
    validate_splits(splits, labels)
    model = make_classifier(config)
    model.fit(values[splits["train"]], labels[splits["train"]])
    test_probabilities = model.predict_proba(values[splits["test"]])[:, 1]
    validation = None
    if "validation" in splits:
        validation_probabilities = model.predict_proba(values[splits["validation"]])[:, 1]
        validation = binary_metrics(labels[splits["validation"]], validation_probabilities, config.threshold)
    suffix = "" if split_configuration is None else f"-{split_configuration}"
    return {
        "run": f"{window_name}-{split_name}{suffix}",
        "window": window_name,
        "split_scheme": split_name,
        "split_configuration": "70_15_15" if split_configuration is None else split_configuration,
        "feature_dimension": int(values.shape[1]),
        "fault_class_weight": float(config.fault_class_weight),
        "threshold": float(config.threshold),
        "model_iterations": int(model.n_iter_),
        "validation": validation,
        "test": binary_metrics(labels[splits["test"]], test_probabilities, config.threshold),
    }, model


def run_baseline_matrix(
    values: np.ndarray,
    labels: np.ndarray,
    station_ids: np.ndarray,
    hours: np.ndarray,
    display_states: np.ndarray,
    source_episode_ids: np.ndarray,
    window_name: str,
    config: HourlyBaselineConfig,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
    split_configuration: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, HistGradientBoostingClassifier], dict[str, dict[str, np.ndarray]], dict[str, object]]:
    labels = np.asarray(labels, dtype=int)
    random_splits = random_split_indices(labels, config.seed, fractions)
    spaced_splits, spaced_detail = spaced_split_indices(
        labels,
        station_ids,
        hours,
        display_states,
        source_episode_ids,
        fractions,
    )
    split_map = {
        "random": random_splits,
        "spaced": spaced_splits,
    }
    rows = []
    models: dict[str, HistGradientBoostingClassifier] = {}
    for split_name, splits in split_map.items():
        row, model = train_baseline_run(
            values,
            labels,
            splits,
            window_name,
            split_name,
            config,
            split_configuration,
        )
        row["split_summary"] = split_summary(splits, labels, hours, display_states)
        rows.append(row)
        models[str(row["run"])] = model
    return rows, models, split_map, spaced_detail


def make_split_manifest(
    splits_by_scheme: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
    station_ids: np.ndarray,
    hours: np.ndarray,
    display_states: np.ndarray,
    source_episode_ids: np.ndarray,
    split_configuration: str = "70_15_15",
) -> pd.DataFrame:
    rows = []
    for scheme, splits in splits_by_scheme.items():
        for split_name, indices in splits.items():
            rows.append(
                pd.DataFrame(
                    {
                        "split_configuration": split_configuration,
                        "split_scheme": scheme,
                        "split": split_name,
                        "station_id": np.asarray(station_ids, dtype=object)[indices].astype(str),
                        "hour": np.asarray(hours, dtype=object)[indices].astype(str),
                        "fault_hour": np.asarray(labels, dtype=int)[indices],
                        "display_state": np.asarray(display_states, dtype=object)[indices].astype(str),
                        "source_episode_ids": np.asarray(source_episode_ids, dtype=object)[indices].astype(str),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["split_configuration", "split_scheme", "split", "station_id", "hour"],
    ).reset_index(drop=True)


def save_model_bundle(
    model: HistGradientBoostingClassifier,
    path: Path,
    feature_names: list[str],
    window_hours: int,
    config: HourlyBaselineConfig,
    split_name: str,
    split_configuration: str = "70_15_15",
    split_fractions: dict[str, float] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "estimator": model,
        "feature_names": feature_names,
        "window_hours": int(window_hours),
        "config": asdict(config),
        "split_scheme": split_name,
        "split_configuration": split_configuration,
        "split_fractions": dict(SPLIT_FRACTIONS if split_fractions is None else split_fractions),
        "class_weight": {0: 1.0, 1: float(config.fault_class_weight)},
    }
    joblib.dump(bundle, destination)
    return destination


def stratified_sample_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if len(labels) <= maximum:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    faults = np.flatnonzero(np.equal(labels, 1))
    not_faults = np.flatnonzero(np.equal(labels, 0))
    fault_count = max(1, min(len(faults), int(round(maximum * len(faults) / len(labels)))))
    not_fault_count = maximum - fault_count
    if not_fault_count > len(not_faults):
        not_fault_count = len(not_faults)
        fault_count = maximum - not_fault_count
    chosen = np.concatenate(
        [
            rng.choice(faults, size=fault_count, replace=False),
            rng.choice(not_faults, size=not_fault_count, replace=False),
        ]
    )
    return np.sort(chosen.astype(np.int64))


def grouped_permutation_importance(
    model: HistGradientBoostingClassifier,
    values: np.ndarray,
    labels: np.ndarray,
    groups: dict[str, np.ndarray],
    threshold: float,
    seed: int,
    maximum_rows: int = 2000,
) -> list[dict[str, object]]:
    selected = stratified_sample_indices(labels, maximum_rows, seed)
    sample_values = values[selected].copy()
    sample_labels = np.asarray(labels, dtype=int)[selected]
    base_probabilities = model.predict_proba(sample_values)[:, 1]
    base_f1 = float(f1_score(sample_labels, base_probabilities >= threshold, zero_division=0))
    rng = np.random.default_rng(seed)
    rows = []
    for name, columns in groups.items():
        permuted = sample_values.copy()
        permutation = rng.permutation(len(sample_values))
        permuted[:, columns] = permuted[permutation][:, columns]
        probabilities = model.predict_proba(permuted)[:, 1]
        score = float(f1_score(sample_labels, probabilities >= threshold, zero_division=0))
        rows.append(
            {
                "feature_group": name,
                "importance_f1_drop": float(base_f1 - score),
                "base_validation_f1": base_f1,
                "permuted_validation_f1": score,
                "sample_rows": int(len(selected)),
            }
        )
    return sorted(rows, key=lambda row: (-float(row["importance_f1_drop"]), str(row["feature_group"])))


def validation_importance_reference(rows: list[dict[str, object]]) -> dict[str, object]:
    candidates = [row for row in rows if isinstance(row.get("validation"), dict)]
    if not candidates:
        raise ValueError("cannot choose an importance reference without validation results")
    return max(
        candidates,
        key=lambda row: (
            float(row["validation"]["f1"]),
            float(row["validation"]["recall"]),
            float(row["validation"]["precision"]),
        ),
    )


def baseline_report(
    rows: list[dict[str, object]],
    top_importance: list[dict[str, object]],
    config: HourlyBaselineConfig,
    spaced_detail: dict[str, object],
    importance_reference: dict[str, object],
) -> str:
    parts = [
        "HOURLY BINARY GRADIENT-BOOSTED BASELINE",
        "",
        "CONFIGURATION",
        f"estimator=HistGradientBoostingClassifier",
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
        validation = row["validation"]
        parts.extend(
            [
                f"RUN {row['run']}",
                f"feature_dimension={row['feature_dimension']}",
                f"model_iterations={row['model_iterations']}",
                "TEST METRICS",
                pd.DataFrame(
                    [
                        {
                            "precision": test["precision"],
                            "recall": test["recall"],
                            "f1": test["f1"],
                            "accuracy": test["accuracy"],
                        }
                    ]
                ).to_string(index=False),
                "TEST CONFUSION MATRIX",
                pd.DataFrame(
                    [{"tp": test["tp"], "fp": test["fp"], "fn": test["fn"], "tn": test["tn"]}]
                ).to_string(index=False),
                "VALIDATION METRICS",
            ]
        )
        if validation is None:
            parts.append("not available for this split configuration")
        else:
            parts.append(
                pd.DataFrame(
                    [
                        {
                            "precision": validation["precision"],
                            "recall": validation["recall"],
                            "f1": validation["f1"],
                            "accuracy": validation["accuracy"],
                        }
                    ]
                ).to_string(index=False)
            )
        parts.append("")
    summary_rows = []
    for row in rows:
        validation = row["validation"]
        test = row["test"]
        summary_rows.append(
            {
                "run": row["run"],
                "validation_precision": validation["precision"] if validation is not None else np.nan,
                "validation_recall": validation["recall"] if validation is not None else np.nan,
                "validation_f1": validation["f1"] if validation is not None else np.nan,
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_f1": test["f1"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    positive_total = sum(max(0.0, float(row["importance_f1_drop"])) for row in top_importance)
    dominant = bool(top_importance) and positive_total > 0.0 and float(top_importance[0]["importance_f1_drop"]) / positive_total >= 0.50
    parts.extend(
        [
            "SUMMARY COMPARISON",
            summary.to_string(index=False),
            "",
            "REPORTING POLICY",
            "All predefined configurations are reported. Held-out test metrics are descriptive comparisons only and do not select or name a winner.",
            f"importance_reference_run={importance_reference['run']}",
            "Feature importance uses the validation-selected reference above, with validation F1, recall, and precision as its selection order.",
            "",
            "RANDOM SPLIT CONSTRUCTION",
            "Rows are assigned with a deterministic seed and binary stratification at 70/15/15.",
            "",
            "SPACED SPLIT CONSTRUCTION",
            "Fault hours are grouped by connected source episode identifiers, placed within chronological strata, and assigned against 70/15/15 positive-hour targets.",
            "Benign and clean hours are assigned separately within calendar-month and display-state strata using deterministic hashes and exact largest-remainder quotas.",
            f"spaced_fault_group_count={spaced_detail.get('fault_group_count', 0)}",
            "",
            "VALIDATION-SELECTED GROUPED PERMUTATION IMPORTANCE",
            pd.DataFrame(top_importance[:20]).to_string(index=False),
            f"single_feature_dominance_flag={dominant}",
            "",
            "EVALUATION NOTE",
            "The windows end at the scored hour, but the existing feature snapshot includes retrospective detector calculations. These are offline comparison metrics, not future-facing deployment metrics.",
        ]
    )
    return "\n".join(parts)


def write_metrics_json(payload: dict[str, object], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return destination
