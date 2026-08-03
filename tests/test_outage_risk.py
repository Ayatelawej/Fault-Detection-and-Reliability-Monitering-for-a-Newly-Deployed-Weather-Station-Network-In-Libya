from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.availability.risk_dataset import (
    CAUSAL_FORECAST_FEATURE_COLUMNS,
    CausalForecastFeatureBundle,
    FEATURE_COLUMNS,
    INCIDENT_HAZARD_CONTEXT_FEATURE_COLUMNS,
    INCIDENT_HAZARD_EVIDENCE_FEATURE_COLUMNS,
    INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS,
    build_backward_event_history_features,
    build_causal_forecast_features,
    build_confirmed_incident_recurrence_history,
    build_discrete_hazard_features,
    discrete_hazard_causal_source_columns,
    discrete_hazard_numeric_feature_columns,
    build_fault_risk_dataset,
    build_incident_hazard_dataset,
    build_risk_dataset,
    build_retrospective_persistence_history,
    event_history_feature_columns,
    summarize_delete_future_validation,
    validate_delete_future_event_history_features,
    validate_delete_future_features,
)
from src.availability.risk_eval import (
    build_label_split_characteristics,
    event_recall,
    regression_error_improvement_percent,
    regression_metrics,
    split_train_validation_test,
    split_timestamp_partitions,
)
from src.availability.risk_model import (
    CausalIncidentHazardRGFN,
    IncidentHazardRgfnConfig,
    attach_incident_hazard_features,
    build_incident_hazard_tensor_bundle,
    cumulate_stationary_hazard,
    flicker_predict,
    fit_incident_hazard_normalizer,
    fit_incident_hazard_rgfn,
    fit_forecast_hist_gradient_boosting,
    forecast_recurrence_prediction,
    forecast_training_positive_class_weight,
    select_discrete_hazard_threshold,
    select_validation_maximin,
    select_validation_threshold_rule,
)
from src.features.row_state import (
    ROW_STATE_COMPLETE,
    ROW_STATE_PARTIAL,
    ROW_STATE_TERMINAL_PADDED,
    ROW_STATE_TRUE_OUTAGE,
    ROW_STATE_WARMUP,
)


def _grid(states: list[str], station_id: str = "S1", start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": station_id,
            "hour_utc": pd.date_range(start, periods=len(states), freq="h", tz="UTC"),
            "row_state": states,
        }
    )


def _availability_grid(
    states: list[str],
    station_id: str = "S1",
    start: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    availability_class = {
        ROW_STATE_COMPLETE: "online",
        ROW_STATE_PARTIAL: "partial_outage",
        ROW_STATE_TRUE_OUTAGE: "full_outage",
        ROW_STATE_WARMUP: "online",
    }
    return pd.DataFrame(
        {
            "station_id": station_id,
            "hour_utc": pd.date_range(start, periods=len(states), freq="h", tz="UTC"),
            "availability_class": [availability_class[state] for state in states],
            "row_state": states,
            "source_kind": ["observed_row"] * len(states),
        }
    )


def _fault_labels(
    states: list[str],
    *,
    fault_indices: set[int] | None = None,
    excluded_indices: set[int] | None = None,
    station_id: str = "S1",
    start: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    fault_indices = set() if fault_indices is None else fault_indices
    excluded_indices = set() if excluded_indices is None else excluded_indices
    count = len(states)
    fault_hour = pd.Series(pd.NA, index=range(count), dtype="Int64")
    display_state = np.full(count, "clean", dtype=object)
    training_eligible = np.ones(count, dtype=bool)
    for index in range(count):
        if index in excluded_indices:
            display_state[index] = "excluded"
            training_eligible[index] = False
            continue
        fault_hour.iloc[index] = int(index in fault_indices)
        if index in fault_indices:
            display_state[index] = "fault"
    return pd.DataFrame(
        {
            "station_id": [station_id] * count,
            "hour": pd.date_range(start, periods=count, freq="h", tz="UTC"),
            "fault_hour": fault_hour,
            "display_state": display_state,
            "training_eligible": training_eligible,
        }
    )


def _incident_feature_frame(
    hours: int = 18,
    station_id: str = "S1",
) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=hours, freq="h", tz="UTC")
    schema = tuple(
        dict.fromkeys(
            [
                *INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS,
                *INCIDENT_HAZARD_EVIDENCE_FEATURE_COLUMNS,
                *INCIDENT_HAZARD_CONTEXT_FEATURE_COLUMNS,
            ]
        )
    )
    data: dict[str, object] = {
        "station_id": station_id,
        "hour_utc": timestamps,
    }
    for index, column in enumerate(schema):
        data[column] = np.arange(hours, dtype=float) + float(index)
    return pd.DataFrame(data)


def _incident_partition(
    feature_frame: pd.DataFrame,
    indices: list[int],
    values: list[int],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": feature_frame.loc[indices, "station_id"].astype(str).to_numpy(),
            "hour_utc": feature_frame.loc[indices, "hour_utc"].to_numpy(),
            "y": np.asarray(values, dtype=int),
        }
    )


def test_synthetic_grid_label_correctness_per_horizon() -> None:
    states = [ROW_STATE_COMPLETE] * 4 + [ROW_STATE_TRUE_OUTAGE] + [ROW_STATE_COMPLETE] * 25
    dataset = build_risk_dataset(_grid(states), horizons=(6, 12, 24))
    frame_6 = dataset.for_horizon(6)
    frame_12 = dataset.for_horizon(12)
    first = pd.Timestamp("2026-01-01T00:00:00Z")

    assert int(frame_6.loc[frame_6["hour_utc"].eq(first), "y"].iloc[0]) == 1
    assert int(frame_6.loc[frame_6["hour_utc"].eq(first + pd.Timedelta(hours=5)), "y"].iloc[0]) == 0
    assert int(frame_12.loc[frame_12["hour_utc"].eq(first), "y"].iloc[0]) == 1


def test_eligibility_excludes_outages_warmup_and_terminal_padding() -> None:
    states = [
        ROW_STATE_WARMUP,
        ROW_STATE_COMPLETE,
        ROW_STATE_TRUE_OUTAGE,
        ROW_STATE_PARTIAL,
        ROW_STATE_TERMINAL_PADDED,
        ROW_STATE_COMPLETE,
        ROW_STATE_COMPLETE,
        ROW_STATE_COMPLETE,
        ROW_STATE_COMPLETE,
    ]
    dataset = build_risk_dataset(_grid(states), horizons=(2,))
    frame = dataset.for_horizon(2)

    assert set(frame["hour_utc"]) == {
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T03:00:00Z"),
        pd.Timestamp("2026-01-01T05:00:00Z"),
        pd.Timestamp("2026-01-01T06:00:00Z"),
    }


def test_right_censoring_drops_per_horizon() -> None:
    states = [ROW_STATE_COMPLETE] * 10
    dataset = build_risk_dataset(_grid(states), horizons=(6,))
    frame = dataset.for_horizon(6)

    assert len(frame) == 4
    assert frame["hour_utc"].max() == pd.Timestamp("2026-01-01T03:00:00Z")


def test_sparse_structural_gap_matches_explicit_outage_hours() -> None:
    states = [ROW_STATE_COMPLETE] * 16
    sparse = _grid(states).drop(index=[2, 3, 4]).reset_index(drop=True)
    explicit_states = states.copy()
    explicit_states[2:5] = [ROW_STATE_TRUE_OUTAGE] * 3
    explicit = _grid(explicit_states)

    sparse_dataset = build_risk_dataset(sparse, horizons=(6,))
    explicit_dataset = build_risk_dataset(explicit, horizons=(6,))
    sparse_labels = sparse_dataset.for_horizon(6).loc[:, ["hour_utc", "y"]]
    explicit_labels = explicit_dataset.for_horizon(6).loc[:, ["hour_utc", "y"]]
    merged = sparse_labels.merge(explicit_labels, on="hour_utc", suffixes=("_sparse", "_explicit"))
    compared_hour = pd.Timestamp("2026-01-01T01:00:00Z")

    assert int(merged.loc[merged["hour_utc"].eq(compared_hour), "y_sparse"].iloc[0]) == 1
    pd.testing.assert_series_equal(
        merged["y_sparse"],
        merged["y_explicit"],
        check_names=False,
    )
    assert int(sparse_dataset.frame["is_materialized_gap"].sum()) == 3
    assert int(sparse_dataset.label_change_summary()["changed_label_rows"].iloc[0]) > 0


def test_warmup_rows_remain_present_but_never_scoreable() -> None:
    source = _grid(
        [ROW_STATE_WARMUP, ROW_STATE_WARMUP] + [ROW_STATE_COMPLETE] * 8
    )
    original = source.copy(deep=True)
    dataset = build_risk_dataset(source, horizons=(2,))
    prepared = dataset.frame

    assert not prepared.loc[prepared["row_state"].eq(ROW_STATE_WARMUP), "is_outage"].any()
    assert not prepared.loc[prepared["row_state"].eq(ROW_STATE_WARMUP), "is_scoreable"].any()
    pd.testing.assert_frame_equal(source, original)


def test_partial_transmission_is_not_a_full_outage_target() -> None:
    states = [ROW_STATE_COMPLETE, ROW_STATE_PARTIAL, ROW_STATE_COMPLETE] + [ROW_STATE_COMPLETE] * 6
    dataset = build_risk_dataset(_grid(states), horizons=(2,))
    first = pd.Timestamp("2026-01-01T00:00:00Z")
    labels = dataset.for_horizon(2)

    assert int(labels.loc[labels["hour_utc"].eq(first), "y"].iloc[0]) == 0


def test_availability_class_source_uses_full_outages_as_targets() -> None:
    hours = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="h", tz="UTC")
    source = pd.DataFrame(
        {
            "station_id": ["S1"] * len(hours),
            "hour_utc": hours,
            "availability_scope": ["active"] * len(hours),
            "availability_class": [
                "online",
                "full_outage",
                "partial_outage",
                "online",
                "online",
                "online",
            ],
            "row_state": [
                ROW_STATE_COMPLETE,
                ROW_STATE_TRUE_OUTAGE,
                ROW_STATE_PARTIAL,
                ROW_STATE_COMPLETE,
                ROW_STATE_COMPLETE,
                ROW_STATE_COMPLETE,
            ],
            "source_kind": ["observed_row"] * len(hours),
        }
    )
    dataset = build_risk_dataset(source, horizons=(1,))
    labels = dataset.for_horizon(1)

    assert int(labels.loc[labels["hour_utc"].eq(hours[0]), "y"].iloc[0]) == 1
    assert int(labels.loc[labels["hour_utc"].eq(hours[2]), "y"].iloc[0]) == 0


def test_fault_risk_keeps_structural_gap_on_the_clock() -> None:
    states = [ROW_STATE_COMPLETE] * 6
    availability = _availability_grid(states).drop(index=[1]).reset_index(drop=True)
    labels = _fault_labels(states, fault_indices={2}).drop(index=[1]).reset_index(drop=True)
    dataset = build_fault_risk_dataset(labels, availability, horizons=(2,))
    first = pd.Timestamp("2026-01-01T00:00:00Z")
    risk_rows = dataset.for_horizon(2)

    assert int(dataset.frame["is_materialized_gap"].sum()) == 1
    assert int(risk_rows.loc[risk_rows["hour_utc"].eq(first), "y"].iloc[0]) == 1


def test_fault_risk_warmup_is_not_scoreable_but_remains_a_future_fault() -> None:
    states = [ROW_STATE_COMPLETE, ROW_STATE_WARMUP] + [ROW_STATE_COMPLETE] * 5
    availability = _availability_grid(states)
    labels = _fault_labels(states, fault_indices={1})
    dataset = build_fault_risk_dataset(labels, availability, horizons=(1,))
    first = pd.Timestamp("2026-01-01T00:00:00Z")
    warmup = first + pd.Timedelta(hours=1)
    risk_rows = dataset.for_horizon(1)

    assert int(risk_rows.loc[risk_rows["hour_utc"].eq(first), "y"].iloc[0]) == 1
    assert not risk_rows["hour_utc"].eq(warmup).any()


def test_fault_risk_excluded_future_hour_is_neutral_not_a_window_blocker() -> None:
    states = [ROW_STATE_COMPLETE] * 6
    availability = _availability_grid(states)
    labels = _fault_labels(states, fault_indices={2}, excluded_indices={1})
    dataset = build_fault_risk_dataset(labels, availability, horizons=(2,))
    first = pd.Timestamp("2026-01-01T00:00:00Z")
    risk_rows = dataset.for_horizon(2)

    assert int(risk_rows.loc[risk_rows["hour_utc"].eq(first), "y"].iloc[0]) == 1
    assert bool(
        dataset.frame.loc[
            dataset.frame["hour_utc"].eq(first), "future_excluded_2h"
        ].iloc[0]
    )


def test_fault_risk_full_outage_hour_is_not_a_current_prediction_row() -> None:
    states = [ROW_STATE_COMPLETE, ROW_STATE_TRUE_OUTAGE] + [ROW_STATE_COMPLETE] * 5
    availability = _availability_grid(states)
    labels = _fault_labels(states, excluded_indices={1})
    dataset = build_fault_risk_dataset(labels, availability, horizons=(1,))
    outage_hour = pd.Timestamp("2026-01-01T01:00:00Z")

    assert not dataset.for_horizon(1)["hour_utc"].eq(outage_hour).any()


def test_fault_incident_hazard_excludes_continuation_and_censors_unobservable_windows() -> None:
    states = [ROW_STATE_COMPLETE] * 14
    labels = _fault_labels(
        states,
        fault_indices={4, 5, 6},
        excluded_indices={10},
    )
    dataset = build_incident_hazard_dataset(
        "fault",
        hourly_labels=labels,
        availability=_availability_grid(states),
        horizons=(2,),
        minimum_duration_hours=3,
    )
    first = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = dataset.frame

    assert frame.loc[
        frame["hour_utc"].eq(first + pd.Timedelta(hours=2)), "incident_eligible_2h"
    ].iloc[0]
    assert int(
        frame.loc[
            frame["hour_utc"].eq(first + pd.Timedelta(hours=2)), "incident_y_2h"
        ].iloc[0]
    ) == 1
    assert not frame.loc[
        frame["hour_utc"].eq(first + pd.Timedelta(hours=4)), "incident_scoreable"
    ].iloc[0]
    assert not frame.loc[
        frame["hour_utc"].eq(first + pd.Timedelta(hours=8)), "incident_eligible_2h"
    ].iloc[0]
    assert frame.loc[
        frame["hour_utc"].eq(first + pd.Timedelta(hours=8)),
        "incident_future_censored_2h",
    ].iloc[0]


def test_one_hour_hazard_targets_the_next_observed_incident_start() -> None:
    states = [ROW_STATE_COMPLETE] * 12
    labels = _fault_labels(states, fault_indices={4, 5, 6})
    dataset = build_incident_hazard_dataset(
        "fault",
        hourly_labels=labels,
        availability=_availability_grid(states),
        horizons=(1,),
        minimum_duration_hours=3,
    )
    first = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = dataset.for_horizon(1)

    assert int(
        rows.loc[rows["hour_utc"].eq(first + pd.Timedelta(hours=3)), "y"].iloc[0]
    ) == 1
    assert int(
        rows.loc[rows["hour_utc"].eq(first + pd.Timedelta(hours=2)), "y"].iloc[0]
    ) == 0
    assert not rows["hour_utc"].eq(first + pd.Timedelta(hours=4)).any()


def test_incident_hazard_applies_a_strictly_prior_recovery_exclusion() -> None:
    states = [ROW_STATE_COMPLETE] * 16
    labels = _fault_labels(states, fault_indices={4, 5, 6})
    dataset = build_incident_hazard_dataset(
        "fault",
        hourly_labels=labels,
        availability=_availability_grid(states),
        horizons=(2,),
        minimum_duration_hours=3,
        recovery_exclusion_hours=3,
    )
    first = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = dataset.frame

    assert not frame.loc[
        frame["hour_utc"].eq(first + pd.Timedelta(hours=4)), "incident_scoreable"
    ].iloc[0]
    for offset in [7, 8, 9]:
        row = frame.loc[frame["hour_utc"].eq(first + pd.Timedelta(hours=offset))]
        assert row["incident_post_event_recovery_excluded"].iloc[0]
        assert not row["incident_scoreable"].iloc[0]
    assert not frame.loc[
        frame["hour_utc"].eq(first + pd.Timedelta(hours=10)),
        "incident_post_event_recovery_excluded",
    ].iloc[0]
    assert frame.loc[
        frame["hour_utc"].eq(first + pd.Timedelta(hours=10)), "incident_scoreable"
    ].iloc[0]


def test_incident_hazard_does_not_promote_a_start_after_an_unobservable_gap() -> None:
    states = [ROW_STATE_COMPLETE] * 12
    labels = _fault_labels(
        states,
        fault_indices={4, 5, 6},
        excluded_indices={3},
    )
    dataset = build_incident_hazard_dataset(
        "fault",
        hourly_labels=labels,
        availability=_availability_grid(states),
        horizons=(2,),
        minimum_duration_hours=3,
    )
    first = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = dataset.frame

    start = frame.loc[frame["hour_utc"].eq(first + pd.Timedelta(hours=4))]
    assert start["incident_start"].iloc[0]
    assert start["incident_unobserved_start"].iloc[0]
    assert not start["incident_observed_start"].iloc[0]
    assert not start["incident_qualifying_start"].iloc[0]
    assert int(
        frame.loc[
            frame["hour_utc"].eq(first + pd.Timedelta(hours=2)), "incident_y_2h"
        ].iloc[0]
    ) == 0


def test_confirmed_incident_recurrence_waits_for_duration_confirmation() -> None:
    states = [ROW_STATE_COMPLETE] * 12
    labels = _fault_labels(states, fault_indices={2, 3, 4})
    dataset = build_incident_hazard_dataset(
        "fault",
        hourly_labels=labels,
        availability=_availability_grid(states),
        horizons=(2,),
        minimum_duration_hours=3,
    )
    history = build_confirmed_incident_recurrence_history(dataset)
    first = pd.Timestamp("2026-01-01T00:00:00Z")
    column = "confirmed_incident_start_count_trailing_168h"

    assert int(
        history.loc[history["hour_utc"].eq(first + pd.Timedelta(hours=4)), column].iloc[0]
    ) == 0
    assert int(
        history.loc[history["hour_utc"].eq(first + pd.Timedelta(hours=5)), column].iloc[0]
    ) == 1


def test_mechanism_incident_hazard_requires_a_fault_free_current_hour() -> None:
    states = [ROW_STATE_COMPLETE] * 16
    labels = _fault_labels(states, fault_indices={1, 5, 6, 7})
    labels["mechanisms"] = ""
    labels.loc[1, "mechanisms"] = "stuck_flatline"
    labels.loc[[5, 6, 7], "mechanisms"] = "statistical_anomaly"
    dataset = build_incident_hazard_dataset(
        "fault_statistical_anomaly",
        hourly_labels=labels,
        availability=_availability_grid(states),
        horizons=(3,),
    )
    first = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = dataset.frame

    assert dataset.minimum_duration_hours == 3
    assert int(frame["incident_qualifying_start"].sum()) == 1
    assert not frame.loc[
        frame["hour_utc"].eq(first + pd.Timedelta(hours=1)), "incident_scoreable"
    ].iloc[0]
    assert int(
        frame.loc[
            frame["hour_utc"].eq(first + pd.Timedelta(hours=2)), "incident_y_3h"
        ].iloc[0]
    ) == 1


def test_outage_incident_hazard_censors_network_events_and_uses_duration_span() -> None:
    states = [ROW_STATE_COMPLETE] * 3 + [ROW_STATE_TRUE_OUTAGE] * 6 + [ROW_STATE_COMPLETE] * 8
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    network_windows = pd.DataFrame(
        {
            "backfill_start_utc": [start + pd.Timedelta(hours=3)],
            "backfill_end_utc": [start + pd.Timedelta(hours=8)],
            "station_ids": ["S1"],
        }
    )
    dataset = build_incident_hazard_dataset(
        "outage",
        availability=_availability_grid(states),
        network_windows=network_windows,
        horizons=(2,),
        minimum_duration_hours=3,
    )
    frame = dataset.frame

    assert int(frame["incident_network_associated"].sum()) == 1
    assert int(frame["incident_qualifying_start"].sum()) == 0
    assert frame.loc[
        frame["hour_utc"].eq(start + pd.Timedelta(hours=1)),
        "incident_future_censored_2h",
    ].iloc[0]
    assert frame.loc[
        frame["hour_utc"].eq(start), "incident_label_end_2h"
    ].iloc[0] == start + pd.Timedelta(hours=4)


def test_incident_hazard_tensor_keeps_clock_time_and_never_uses_future_values() -> None:
    features = _incident_feature_frame(hours=12)
    partition = _incident_partition(features, [2, 4], [0, 1])
    attached = attach_incident_hazard_features(partition, features)
    normalizer = fit_incident_hazard_normalizer(features, attached)
    first = build_incident_hazard_tensor_bundle(
        features,
        attached,
        normalizer,
        window_hours=6,
    )
    changed = features.copy(deep=True)
    future = changed["hour_utc"].gt(pd.Timestamp("2026-01-01T04:00:00Z"))
    changed.loc[future, list(INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS)] = 99999.0
    second = build_incident_hazard_tensor_bundle(
        changed,
        attached,
        normalizer,
        window_hours=6,
    )

    np.testing.assert_array_equal(first.temporal_values[1], second.temporal_values[1])
    np.testing.assert_array_equal(first.temporal_mask[1], second.temporal_mask[1])
    assert first.temporal_mask[0, :3].sum() == 0.0
    assert first.temporal_mask[0, 3:].sum() > 0.0


def test_incident_hazard_normalizer_is_fit_without_later_partition_values() -> None:
    features = _incident_feature_frame(hours=12)
    train = _incident_partition(features, [1, 2, 3, 4], [0, 1, 0, 1])
    attached = attach_incident_hazard_features(train, features)
    first = fit_incident_hazard_normalizer(features, attached)
    changed = features.copy(deep=True)
    changed.loc[changed["hour_utc"].gt(pd.Timestamp("2026-01-01T04:00:00Z")), list(INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS)] = -99999.0
    second = fit_incident_hazard_normalizer(changed, attached)

    np.testing.assert_array_equal(first.temporal_center, second.temporal_center)
    np.testing.assert_array_equal(first.temporal_scale, second.temporal_scale)
    assert first.station_to_code == second.station_to_code


def test_incident_hazard_model_and_selection_use_train_validation_only() -> None:
    features = _incident_feature_frame(hours=18)
    train = _incident_partition(features, list(range(2, 10)), [0, 1, 0, 1, 0, 0, 1, 0])
    validation = _incident_partition(features, list(range(10, 14)), [0, 1, 0, 1])
    attached_train = attach_incident_hazard_features(train, features)
    normalizer = fit_incident_hazard_normalizer(features, attached_train)
    bundle = build_incident_hazard_tensor_bundle(
        features,
        attached_train,
        normalizer,
        window_hours=6,
    )
    config = IncidentHazardRgfnConfig(
        window_hours=6,
        batch_size=4,
        max_epochs=1,
        patience=1,
        dropout=0.0,
    )
    model = CausalIncidentHazardRGFN(config, n_stations=2)
    output = model(
        torch.from_numpy(bundle.temporal_values[:2]),
        torch.from_numpy(bundle.temporal_mask[:2]),
        torch.from_numpy(bundle.evidence_values[:2]),
        torch.from_numpy(bundle.evidence_mask[:2]),
        torch.from_numpy(bundle.context_values[:2]),
        torch.from_numpy(bundle.context_mask[:2]),
        torch.from_numpy(bundle.station_codes[:2]),
    )
    assert output["incident_probability"].shape == (2,)
    assert torch.all(output["incident_probability"].ge(0.0))
    assert torch.all(output["incident_probability"].le(1.0))

    selection = fit_incident_hazard_rgfn(
        features,
        train,
        validation,
        config=config,
        weight_multipliers=(1.0,),
        thresholds=(0.25, 0.50),
    )
    assert len(selection.validation_probability) == len(validation)
    assert selection.selection_trace["test_metrics_accessed_during_selection"].eq(False).all()


def test_incident_hazard_split_purges_the_full_prediction_and_duration_span() -> None:
    states = [ROW_STATE_COMPLETE] * 24
    labels = _fault_labels(states, fault_indices={14, 15, 16})
    dataset = build_incident_hazard_dataset(
        "fault",
        hourly_labels=labels,
        availability=_availability_grid(states),
        horizons=(2,),
        minimum_duration_hours=3,
    )
    split = split_train_validation_test(
        dataset.for_horizon(2),
        2,
        validation_start_utc=pd.Timestamp("2026-01-01T10:00:00Z"),
        test_start_utc=pd.Timestamp("2026-01-01T17:00:00Z"),
    )
    metadata = split["metadata"]

    assert metadata["label_span_hours"] == 4
    assert metadata["train_purge_start_utc"] == pd.Timestamp("2026-01-01T06:00:00Z")
    assert metadata["validation_purge_start_utc"] == pd.Timestamp("2026-01-01T13:00:00Z")
    assert split["train"]["label_end_utc"].lt(metadata["validation_start_utc"]).all()
    assert split["validation"]["label_end_utc"].lt(metadata["test_start_utc"]).all()


def test_fault_risk_split_characteristics_have_a_default_construction_summary() -> None:
    states = [ROW_STATE_COMPLETE] * 18
    availability = _availability_grid(states)
    labels = _fault_labels(states, fault_indices={5, 12})
    dataset = build_fault_risk_dataset(labels, availability, horizons=(2,))

    characteristics = build_label_split_characteristics(dataset)

    construction = characteristics["label_changes"]
    assert isinstance(construction, pd.DataFrame)
    assert int(construction.loc[0, "direct_fault_hours"]) == 2


def test_fault_risk_split_keeps_each_shared_timestamp_in_one_partition() -> None:
    states = [ROW_STATE_COMPLETE] * 20
    availability = pd.concat(
        [
            _availability_grid(states, station_id="S1"),
            _availability_grid(states, station_id="S2"),
        ],
        ignore_index=True,
    )
    labels = pd.concat(
        [
            _fault_labels(states, station_id="S1", fault_indices={6}),
            _fault_labels(states, station_id="S2", fault_indices={12}),
        ],
        ignore_index=True,
    )
    dataset = build_fault_risk_dataset(labels, availability, horizons=(2,))

    characteristics = build_label_split_characteristics(dataset)
    split = characteristics["splits"][2]
    membership = pd.concat(
        [
            split[name].loc[:, ["station_id", "hour_utc"]].assign(partition=name)
            for name in ["train", "validation", "test", "purged"]
        ],
        ignore_index=True,
    )

    assert membership.groupby("hour_utc")["partition"].nunique().eq(1).all()


def test_future_randomization_does_not_change_features_at_t() -> None:
    states = [ROW_STATE_COMPLETE] * 10 + [ROW_STATE_TRUE_OUTAGE] * 3 + [ROW_STATE_COMPLETE] * 20
    original = _grid(states)
    randomized = original.copy()
    cutoff = pd.Timestamp("2026-01-01T08:00:00Z")
    mask = randomized["hour_utc"].gt(cutoff)
    shuffled = randomized.loc[mask, "row_state"].sample(frac=1.0, random_state=5).to_numpy()
    randomized.loc[mask, "row_state"] = shuffled
    first = build_risk_dataset(original, horizons=(6,)).frame
    second = build_risk_dataset(randomized, horizons=(6,)).frame
    first_row = first.loc[first["hour_utc"].eq(cutoff), FEATURE_COLUMNS].iloc[0]
    second_row = second.loc[second["hour_utc"].eq(cutoff), FEATURE_COLUMNS].iloc[0]

    pd.testing.assert_series_equal(first_row, second_row, check_names=False)


def test_flicker_baseline_correctness() -> None:
    frame = pd.DataFrame({"trailing_missing_frac_24h": [0.0, 0.1, 1.0]})
    prediction = flicker_predict(frame)

    np.testing.assert_array_equal(prediction.pred, np.asarray([0, 1, 1]))


def test_event_recall_correctness_on_synthetic_event() -> None:
    test_frame = pd.DataFrame(
        {
            "station_id": ["S1"] * 5,
            "hour_utc": pd.date_range("2026-03-16T00:00:00Z", periods=5, freq="h", tz="UTC"),
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "station_id": ["S1"],
            "start_utc": [pd.Timestamp("2026-03-16T04:00:00Z")],
            "end_utc": [pd.Timestamp("2026-03-16T05:00:00Z")],
            "duration_hours": [2],
            "outage_class": ["local"],
        }
    )
    pred = np.asarray([0, 1, 0, 0, 0], dtype=int)
    result = event_recall(test_frame, events, 4, pred)

    assert result["n_test_events"] == 1
    assert result["event_recall"] == pytest.approx(1.0)
    assert result["median_lead_time_h"] == pytest.approx(3.0)


def test_timestamp_split_keeps_shared_hours_together_and_purges_horizon() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    states = (
        [ROW_STATE_COMPLETE] * 12
        + [ROW_STATE_TRUE_OUTAGE]
        + [ROW_STATE_COMPLETE] * 11
        + [ROW_STATE_TRUE_OUTAGE]
        + [ROW_STATE_COMPLETE] * 11
    )
    source = pd.concat(
        [_grid(states, station_id="S1"), _grid(states, station_id="S2")],
        ignore_index=True,
    ).sample(frac=1.0, random_state=2)
    dataset = build_risk_dataset(source, horizons=(4,))
    split = split_train_validation_test(
        dataset.for_horizon(4),
        4,
        validation_start_utc=start + pd.Timedelta(hours=12),
        test_start_utc=start + pd.Timedelta(hours=24),
    )
    metadata = split["metadata"]
    memberships = pd.concat(
        [
            split[name].assign(partition=name)
            for name in ["train", "validation", "test", "purged"]
        ],
        ignore_index=True,
    )
    boundary_row = dataset.for_horizon(4).loc[
        lambda frame: frame["hour_utc"].eq(start + pd.Timedelta(hours=8))
    ]
    validation_boundary_row = dataset.for_horizon(4).loc[
        lambda frame: frame["hour_utc"].eq(start + pd.Timedelta(hours=20))
    ]

    assert boundary_row["y"].eq(1).all()
    assert validation_boundary_row["y"].eq(1).all()
    assert memberships.groupby("hour_utc")["partition"].nunique().max() == 1
    assert (split["train"]["label_end_utc"] < metadata["validation_start_utc"]).all()
    assert (split["validation"]["label_end_utc"] < metadata["test_start_utc"]).all()
    assert split["purged"]["hour_utc"].eq(
        start + pd.Timedelta(hours=8)
    ).any()
    assert split["purged"]["hour_utc"].eq(
        start + pd.Timedelta(hours=20)
    ).any()
    assert not (
        split["train"]["hour_utc"].eq(start + pd.Timedelta(hours=8)).any()
        or split["validation"]["hour_utc"].eq(start + pd.Timedelta(hours=8)).any()
        or split["test"]["hour_utc"].eq(start + pd.Timedelta(hours=8)).any()
        or split["train"]["hour_utc"].eq(start + pd.Timedelta(hours=20)).any()
        or split["validation"]["hour_utc"].eq(start + pd.Timedelta(hours=20)).any()
        or split["test"]["hour_utc"].eq(start + pd.Timedelta(hours=20)).any()
    )


def test_generic_timestamp_split_preserves_numeric_targets_and_shared_hours() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    hours = pd.date_range(start, periods=20, freq="h", tz="UTC")
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "hour_utc": hours,
                    "label_end_utc": hours + pd.Timedelta(hours=4),
                    "health_future": np.asarray(
                        np.arange(len(hours), dtype=float) + offset,
                        dtype=np.float32,
                    ),
                    "delta_health": np.arange(len(hours), dtype=float) / 10.0,
                }
            )
            for station_id, offset in [("S1", 0.0), ("S2", 100.0)]
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=7)

    split = split_timestamp_partitions(
        frame,
        target_columns=("health_future", "delta_health"),
        validation_start_utc=start + pd.Timedelta(hours=8),
        test_start_utc=start + pd.Timedelta(hours=14),
    )
    combined = pd.concat(
        [split[name] for name in ["train", "validation", "test", "purged"]],
        ignore_index=True,
    ).sort_values(["hour_utc", "station_id"], kind="mergesort")
    expected = frame.sort_values(["hour_utc", "station_id"], kind="mergesort")
    memberships = pd.concat(
        [
            split[name].loc[:, ["station_id", "hour_utc"]].assign(partition=name)
            for name in ["train", "validation", "test", "purged"]
        ],
        ignore_index=True,
    )

    pd.testing.assert_series_equal(
        combined["health_future"].reset_index(drop=True),
        expected["health_future"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        combined["delta_health"].reset_index(drop=True),
        expected["delta_health"].reset_index(drop=True),
        check_names=False,
    )
    assert combined["health_future"].dtype == np.dtype("float32")
    assert memberships.groupby("hour_utc")["partition"].nunique().eq(1).all()


def test_generic_timestamp_split_purges_labels_reaching_later_partition() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    hours = pd.date_range(start, periods=20, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "station_id": ["S1"] * len(hours),
            "hour_utc": hours,
            "label_end_utc": hours + pd.Timedelta(hours=4),
            "health_future": np.linspace(50.0, 70.0, len(hours)),
            "delta_health": np.linspace(-2.0, 2.0, len(hours)),
        }
    )

    split = split_timestamp_partitions(
        frame,
        target_columns=("health_future", "delta_health"),
        validation_start_utc=start + pd.Timedelta(hours=8),
        test_start_utc=start + pd.Timedelta(hours=14),
    )
    metadata = split["metadata"]

    assert split["purged"]["hour_utc"].eq(start + pd.Timedelta(hours=4)).any()
    assert split["purged"]["hour_utc"].eq(start + pd.Timedelta(hours=10)).any()
    assert split["train"]["label_end_utc"].lt(metadata["validation_start_utc"]).all()
    assert split["validation"]["label_end_utc"].lt(metadata["test_start_utc"]).all()


def test_regression_metric_helpers_report_error_and_relative_improvement() -> None:
    metrics = regression_metrics(
        np.asarray([0.0, 2.0, 4.0]),
        np.asarray([1.0, 2.0, 3.0]),
    )

    assert metrics["mae"] == pytest.approx(2.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert metrics["r2"] == pytest.approx(0.75)
    assert regression_error_improvement_percent(2.0, 1.0) == pytest.approx(50.0)
    assert np.isnan(regression_error_improvement_percent(0.0, 1.0))


def test_fault_risk_purge_excludes_labels_reaching_later_partitions() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    states = [ROW_STATE_COMPLETE] * 36
    dataset = build_fault_risk_dataset(
        _fault_labels(states, fault_indices={12, 24}),
        _availability_grid(states),
        horizons=(4,),
    )
    split = split_train_validation_test(
        dataset.for_horizon(4),
        4,
        validation_start_utc=start + pd.Timedelta(hours=12),
        test_start_utc=start + pd.Timedelta(hours=24),
    )

    for hour in [start + pd.Timedelta(hours=8), start + pd.Timedelta(hours=20)]:
        assert split["purged"]["hour_utc"].eq(hour).any()
        assert not any(
            split[partition]["hour_utc"].eq(hour).any()
            for partition in ["train", "validation", "test"]
        )


def test_purge_width_is_horizon_specific() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    dataset = build_risk_dataset(_grid([ROW_STATE_COMPLETE] * 40), horizons=(2, 6))
    split_2 = split_train_validation_test(
        dataset.for_horizon(2),
        2,
        validation_start_utc=start + pd.Timedelta(hours=16),
        test_start_utc=start + pd.Timedelta(hours=28),
    )
    split_6 = split_train_validation_test(
        dataset.for_horizon(6),
        6,
        validation_start_utc=start + pd.Timedelta(hours=16),
        test_start_utc=start + pd.Timedelta(hours=28),
    )

    assert len(split_6["purged"]) > len(split_2["purged"])
    assert split_2["metadata"]["horizon_h"] == 2
    assert split_6["metadata"]["horizon_h"] == 6


def test_event_recall_uses_the_split_test_start_not_a_fixed_date() -> None:
    start = pd.Timestamp("2026-04-01T00:00:00Z")
    test_frame = pd.DataFrame(
        {
            "station_id": ["S1"] * 5,
            "hour_utc": pd.date_range(start, periods=5, freq="h", tz="UTC"),
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["before", "inside"],
            "station_id": ["S1", "S1"],
            "start_utc": [start - pd.Timedelta(hours=1), start + pd.Timedelta(hours=4)],
        }
    )
    result = event_recall(
        test_frame,
        events,
        4,
        np.asarray([0, 1, 0, 0, 0], dtype=int),
        test_start_utc=start,
    )

    assert result["n_test_events"] == 1
    assert result["event_recall"] == pytest.approx(1.0)


def test_causal_forecast_features_ignore_all_future_operational_inputs() -> None:
    states = [ROW_STATE_COMPLETE] * 18
    availability = _availability_grid(states)
    availability["absent_sensor_groups"] = ""
    observations = availability.loc[:, ["station_id", "hour_utc"]].copy()
    observations["n_raw_records"] = 4
    observations["pressure_max_hpa"] = 1000.0
    reference = observations.loc[:, ["station_id", "hour_utc"]].copy()
    reference["pressure_msl"] = 995.0
    registry = pd.DataFrame(
        {
            "station_id": ["S1"],
            "latitude": [32.9],
            "longitude": [13.2],
            "elevation": [20.0],
            "install_date": ["2025-12-01"],
        }
    )
    cutoff = pd.Timestamp("2026-01-01T08:00:00Z")
    original_grid = build_risk_dataset(availability, horizons=(2,)).frame
    original = build_causal_forecast_features(
        original_grid,
        availability=availability,
        observations=observations,
        registry=registry,
        reference=reference,
        feature_matrix_columns=(),
    )

    future_availability = availability.copy(deep=True)
    future_observations = observations.copy(deep=True)
    future = future_availability["hour_utc"].gt(cutoff)
    future_availability.loc[future, "row_state"] = ROW_STATE_TRUE_OUTAGE
    future_availability.loc[future, "availability_class"] = "full_outage"
    future_availability.loc[future, "absent_sensor_groups"] = "anemometer|barometer"
    future_observations.loc[future, "n_raw_records"] = 999
    future_observations.loc[future, "pressure_max_hpa"] = 9999.0
    future_reference = reference.copy(deep=True)
    future_reference.loc[future, "pressure_msl"] = -9999.0
    future_grid = build_risk_dataset(future_availability, horizons=(2,)).frame
    changed = build_causal_forecast_features(
        future_grid,
        availability=future_availability,
        observations=future_observations,
        registry=registry,
        reference=future_reference,
        feature_matrix_columns=(),
    )

    original_row = original.frame.loc[
        original.frame["hour_utc"].eq(cutoff), list(CAUSAL_FORECAST_FEATURE_COLUMNS)
    ].iloc[0]
    changed_row = changed.frame.loc[
        changed.frame["hour_utc"].eq(cutoff), list(CAUSAL_FORECAST_FEATURE_COLUMNS)
    ].iloc[0]
    pd.testing.assert_series_equal(original_row, changed_row, check_names=False)
    assert tuple(original.feature_columns) == CAUSAL_FORECAST_FEATURE_COLUMNS
    assert original.causality_audit.loc[
        original.causality_audit["included"], "causality"
    ].eq("causal").all()
    assert original.causality_audit.loc[
        original.causality_audit["feature"].eq("raw_now_pressure_max_hpa"),
        "included",
    ].eq(True).all()
    assert not original.causality_audit["feature"].str.startswith(
        "current_raw_"
    ).any()
    assert not any(
        token in feature
        for feature in CAUSAL_FORECAST_FEATURE_COLUMNS
        for token in ["fault_hour", "source_episode", "mechanisms", "components"]
    )


def test_delete_future_validation_rebuilds_every_feature_as_of_t() -> None:
    states = [ROW_STATE_COMPLETE] * 20
    availability = pd.concat(
        [
            _availability_grid(states, station_id=station_id)
            for station_id in ["S1", "S2", "S3"]
        ],
        ignore_index=True,
    )
    availability["absent_sensor_groups"] = ""
    observations = availability.loc[:, ["station_id", "hour_utc"]].copy()
    observations["n_raw_records"] = 4
    observations["pressure_max_hpa"] = observations["station_id"].map(
        {"S1": 1000.0, "S2": 990.0, "S3": 990.0}
    )
    observations["temp_avg_c"] = 20.0
    observations["dewpoint_avg_c"] = 8.0
    observations["windspeed_avg_kmh"] = 18.0
    observations["solar_radiation_high_wm2"] = 500.0
    reference = observations.loc[:, ["station_id", "hour_utc"]].copy()
    reference["pressure_msl"] = 995.0
    reference["temperature_2m"] = 19.0
    reference["dew_point_2m"] = 7.0
    reference["wind_speed_10m"] = 4.0
    reference["shortwave_radiation"] = 450.0
    registry = pd.DataFrame(
        {
            "station_id": ["S1", "S2", "S3", "FUTURE"],
            "latitude": [32.9, 32.92, 32.94, 32.95],
            "longitude": [13.2, 13.22, 13.24, 13.25],
            "elevation": [20.0, 25.0, 30.0, 35.0],
            "install_date": [
                "2025-12-01",
                "2025-12-01",
                "2025-12-01",
                "2026-02-01",
            ],
        }
    )
    grid = build_risk_dataset(availability, horizons=(2,)).frame
    cutoff = pd.Timestamp("2026-01-01T12:00:00Z")
    detail = validate_delete_future_features(
        grid,
        availability=availability,
        observations=observations,
        registry=registry,
        reference=reference,
        feature_matrix_columns=(),
        sample_keys=pd.DataFrame({"station_id": ["S1"], "hour_utc": [cutoff]}),
    )
    summary = summarize_delete_future_validation(detail)

    assert len(detail) == len(CAUSAL_FORECAST_FEATURE_COLUMNS)
    assert detail["passed"].all()
    assert int(summary.loc[0, "failed_comparisons"]) == 0
    assert int(summary.loc[0, "features_validated"]) == len(
        CAUSAL_FORECAST_FEATURE_COLUMNS
    )


def test_causal_current_reference_and_spatial_features_use_hour_t() -> None:
    states = [ROW_STATE_COMPLETE] * 5
    availability = pd.concat(
        [
            _availability_grid(states, station_id=station_id)
            for station_id in ["S1", "S2", "S3"]
        ],
        ignore_index=True,
    )
    availability["absent_sensor_groups"] = ""
    observations = availability.loc[:, ["station_id", "hour_utc"]].copy()
    observations["n_raw_records"] = 4
    observations["pressure_max_hpa"] = observations["station_id"].map(
        {"S1": 1000.0, "S2": 990.0, "S3": 990.0}
    )
    observations["temp_avg_c"] = 20.0
    observations["dewpoint_avg_c"] = 8.0
    observations["windspeed_avg_kmh"] = 18.0
    observations["solar_radiation_high_wm2"] = 500.0
    reference = observations.loc[:, ["station_id", "hour_utc"]].copy()
    reference["pressure_msl"] = 995.0
    reference["temperature_2m"] = 19.0
    reference["dew_point_2m"] = 7.0
    reference["wind_speed_10m"] = 4.0
    reference["shortwave_radiation"] = 450.0
    registry = pd.DataFrame(
        {
            "station_id": ["S1", "S2", "S3"],
            "latitude": [32.9, 32.92, 32.94],
            "longitude": [13.2, 13.22, 13.24],
            "elevation": [20.0, 25.0, 30.0],
            "install_date": ["2025-12-01"] * 3,
        }
    )
    frame = build_causal_forecast_features(
        build_risk_dataset(availability, horizons=(2,)).frame,
        availability=availability,
        observations=observations,
        registry=registry,
        reference=reference,
    ).frame
    row = frame.loc[
        frame["station_id"].eq("S1")
        & frame["hour_utc"].eq(pd.Timestamp("2026-01-01T00:00:00Z"))
    ].iloc[0]

    assert float(row["raw_now_pressure_max_hpa"]) == pytest.approx(1000.0)
    assert float(row["external_residual_now_pressure"]) == pytest.approx(5.0)
    assert float(row["spatial_residual_now_pressure"]) == pytest.approx(10.0)
    assert float(row["spatial_neighbor_count_now_pressure"]) == pytest.approx(2.0)


def test_retrospective_persistence_uses_only_strictly_prior_events() -> None:
    hours = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="h", tz="UTC")
    grid = pd.DataFrame(
        {
            "station_id": ["S1"] * len(hours),
            "hour_utc": hours,
            "event": [False, False, True, False, False, False],
        }
    )
    history = build_retrospective_persistence_history(
        grid,
        event_column="event",
        horizons=(2,),
    )

    assert int(history.loc[history["hour_utc"].eq(hours[2]), "persistence_event_count_2h"].iloc[0]) == 0
    assert int(history.loc[history["hour_utc"].eq(hours[3]), "persistence_event_count_2h"].iloc[0]) == 1
    assert int(history.loc[history["hour_utc"].eq(hours[4]), "persistence_event_count_2h"].iloc[0]) == 1


def test_recurrence_baseline_uses_only_strictly_prior_event_history() -> None:
    hours = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="h", tz="UTC")
    grid = pd.DataFrame(
        {
            "station_id": ["S1"] * len(hours),
            "hour_utc": hours,
            "fault_target": [False, False, True, False, False, False],
            "fault_scoreable": [True] * len(hours),
            "is_outage": [False] * len(hours),
        }
    )
    history = build_backward_event_history_features(grid, horizons=(6,))
    prediction = forecast_recurrence_prediction(
        history,
        event_source="fault",
    )

    assert prediction.model == "recurrence_168h"
    assert int(prediction.pred[2]) == 0
    assert int(prediction.pred[3]) == 1


def test_event_history_is_strictly_prior_and_counts_event_hours_separately() -> None:
    hours = pd.date_range("2026-01-01T00:00:00Z", periods=10, freq="h", tz="UTC")
    grid = pd.DataFrame(
        {
            "station_id": ["S1"] * len(hours),
            "hour_utc": hours,
            "fault_target": [False, False, True, True, True, False, False, False, False, False],
            "fault_scoreable": [True] * len(hours),
            "is_outage": [False] * len(hours),
        }
    )
    history = build_backward_event_history_features(grid, horizons=(2,))
    columns = event_history_feature_columns(2)
    assert set(columns).issubset(history.columns)
    at_current_start = history.loc[history["hour_utc"].eq(hours[2])].iloc[0]
    one_hour_later = history.loc[history["hour_utc"].eq(hours[3])].iloc[0]
    after_three_event_hours = history.loc[history["hour_utc"].eq(hours[5])].iloc[0]
    assert at_current_start["history_fault_event_hour_count_trailing_2h"] == 0.0
    assert one_hour_later["history_fault_event_hour_count_trailing_2h"] == 1.0
    assert one_hour_later["history_fault_distinct_event_count_trailing_168h"] == 1.0
    assert after_three_event_hours["history_fault_event_hour_count_trailing_2h"] == 2.0
    assert after_three_event_hours["history_fault_distinct_event_count_trailing_168h"] == 1.0


def test_event_history_keeps_excluded_fault_time_unknown() -> None:
    hours = pd.date_range("2026-01-01T00:00:00Z", periods=9, freq="h", tz="UTC")
    grid = pd.DataFrame(
        {
            "station_id": ["S1"] * len(hours),
            "hour_utc": hours,
            "fault_target": [False, False, True, True, False, True, False, False, False],
            "fault_scoreable": [True, True, True, True, False, True, True, True, True],
            "is_outage": [False] * len(hours),
        }
    )
    history = build_backward_event_history_features(grid, horizons=(2,))
    after_unknown = history.loc[history["hour_utc"].eq(hours[5])].iloc[0]
    later = history.loc[history["hour_utc"].eq(hours[7])].iloc[0]
    assert after_unknown["history_fault_distinct_event_count_trailing_168h"] == 1.0
    assert after_unknown["history_fault_unknown_present_trailing_2h"] == 1.0
    assert later["history_fault_last_event_end_never_seen"] == 0.0
    assert later["history_fault_hours_since_last_event_end"] == 2.0


def test_event_history_delete_future_validation_and_station_isolation() -> None:
    hours = pd.date_range("2026-01-01T00:00:00Z", periods=12, freq="h", tz="UTC")
    grid = pd.concat(
        [
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "hour_utc": hours,
                    "fault_target": events,
                    "fault_scoreable": [True] * len(hours),
                    "is_outage": [False] * len(hours),
                }
            )
            for station_id, events in [
                ("S1", [False] * len(hours)),
                ("S2", [False] * 8 + [True, True, False, False]),
            ]
        ],
        ignore_index=True,
    )
    history = build_backward_event_history_features(grid, horizons=(2,))
    s1 = history.loc[history["station_id"].eq("S1")]
    assert s1["history_fault_event_hour_count_trailing_2h"].eq(0.0).all()
    detail = validate_delete_future_event_history_features(
        grid,
        horizons=(2,),
        sample_keys=pd.DataFrame({"station_id": ["S2"], "hour_utc": [hours[9]]}),
        full_history=history,
    )
    assert len(detail) == len(event_history_feature_columns(2))
    assert detail["passed"].all()


def test_forecast_selection_uses_validation_maximin_and_training_weight_only() -> None:
    trace = pd.DataFrame(
        {
            "weight_multiplier": [0.5, 1.0, 1.0],
            "threshold": [0.45, 0.50, 0.55],
            "validation_precision": [0.80, 0.84, 0.90],
            "validation_recall": [0.80, 0.83, 0.70],
            "validation_f1": [0.80, 0.835, 0.79],
            "validation_maximin_prf": [0.80, 0.83, 0.70],
        }
    )

    selected = select_validation_maximin(trace)

    assert float(selected["weight_multiplier"]) == pytest.approx(1.0)
    assert float(selected["threshold"]) == pytest.approx(0.50)
    assert forecast_training_positive_class_weight(np.asarray([0, 0, 0, 1])) == pytest.approx(3.0)


def test_forecast_threshold_rules_use_validation_metrics_and_preserve_infeasibility() -> None:
    trace = pd.DataFrame(
        {
            "threshold": [0.10, 0.20, 0.30, 0.40],
            "validation_precision": [0.60, 0.80, 0.55, 0.40],
            "validation_recall": [0.60, 0.55, 0.80, 0.95],
            "validation_f1": [0.60, 0.65, 0.66, 0.56],
            "validation_accuracy": [0.80, 0.85, 0.81, 0.70],
            "validation_maximin_prf": [0.60, 0.55, 0.55, 0.40],
        }
    )

    maximin = select_validation_threshold_rule(trace, "maximin")
    max_f1 = select_validation_threshold_rule(trace, "max_f1")
    recall_floor = select_validation_threshold_rule(
        trace,
        "max_recall_precision_floor",
        precision_floor=0.55,
    )
    infeasible = select_validation_threshold_rule(
        trace.assign(validation_precision=0.54),
        "max_recall_precision_floor",
        precision_floor=0.55,
    )

    assert maximin is not None
    assert max_f1 is not None
    assert recall_floor is not None
    assert float(maximin["threshold"]) == pytest.approx(0.10)
    assert float(max_f1["threshold"]) == pytest.approx(0.30)
    assert float(recall_floor["threshold"]) == pytest.approx(0.30)
    assert infeasible is None


def test_discrete_hazard_cumulation_and_validation_only_threshold_selection() -> None:
    hourly_hazard = np.asarray([0.0, 0.1, 0.2, 1.0])

    np.testing.assert_allclose(
        cumulate_stationary_hazard(hourly_hazard, 6),
        np.asarray([0.0, 1.0 - 0.9**6, 1.0 - 0.8**6, 1.0]),
    )
    with pytest.raises(ValueError, match="must be positive"):
        cumulate_stationary_hazard(hourly_hazard, 0)

    threshold, metrics, trace = select_discrete_hazard_threshold(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.10, 0.40, 0.60, 0.90]),
        thresholds=(0.20, 0.50, 0.80),
    )

    assert threshold == pytest.approx(0.50)
    assert metrics["f1"] == pytest.approx(1.0)
    assert int(trace["selected"].sum()) == 1
    assert not any(column.startswith("test_") for column in trace.columns)


def test_discrete_hazard_schema_is_fixed_and_compact() -> None:
    fault_columns = discrete_hazard_numeric_feature_columns("fault")
    outage_columns = discrete_hazard_numeric_feature_columns("outage")

    assert len(fault_columns) == 45
    assert len(outage_columns) == 45
    assert set(fault_columns).difference(outage_columns) == {
        "history_fault_hours_since_last_event_end",
        "hazard_log1p_hours_since_last_fault_event_end",
        "history_fault_distinct_event_count_trailing_168h",
        "history_fault_distinct_event_count_trailing_720h",
    }


def test_discrete_hazard_feature_bundle_has_a_fixed_causal_schema() -> None:
    states = [ROW_STATE_COMPLETE] * 32
    availability = _availability_grid(states)
    availability["absent_sensor_groups"] = ""
    observations = availability.loc[:, ["station_id", "hour_utc"]].copy()
    observations["n_raw_records"] = 4
    for column, start in [
        ("pressure_max_hpa", 1000.0),
        ("pressure_trend_hpa", 0.5),
        ("temp_avg_c", 20.0),
        ("windspeed_avg_kmh", 12.0),
        ("windgust_high_kmh", 18.0),
    ]:
        observations[column] = start + np.arange(len(observations), dtype=float)
    registry = pd.DataFrame(
        {
            "station_id": ["S1"],
            "latitude": [32.9],
            "longitude": [13.2],
            "elevation": [20.0],
            "install_date": ["2025-12-01"],
        }
    )
    grid = build_risk_dataset(availability, horizons=(2,)).frame
    source_columns = discrete_hazard_causal_source_columns("fault")
    causal = build_causal_forecast_features(
        grid,
        availability=availability,
        observations=observations,
        registry=registry,
        feature_columns=source_columns,
    )
    history_grid = grid.loc[:, ["station_id", "hour_utc", "is_outage"]].copy()
    history_grid["fault_target"] = False
    history_grid["fault_scoreable"] = True
    history = build_backward_event_history_features(history_grid, horizons=(6, 12, 24))
    compact = build_discrete_hazard_features(
        causal,
        history,
        target="fault",
        registry=registry,
    )

    assert len(compact.numeric_feature_columns) == 45
    assert compact.station_indicator_columns == ("hazard_station_indicator_S1",)
    assert len(compact.model_feature_columns) == 46
    assert set(compact.causality_audit["feature"]) == set(compact.model_feature_columns)

    cutoff = pd.Timestamp("2026-01-01T12:00:00Z")
    changed_source = causal.frame.copy(deep=True)
    changed_history = history.copy(deep=True)
    future = changed_source["hour_utc"].gt(cutoff)
    changed_source.loc[future, list(causal.feature_columns)] = -9999.0
    changed_history.loc[future, [
        "history_fault_hours_since_last_event_end",
        "history_fault_distinct_event_count_trailing_168h",
        "history_fault_distinct_event_count_trailing_720h",
    ]] = -9999.0
    changed = build_discrete_hazard_features(
        CausalForecastFeatureBundle(
            changed_source,
            causal.feature_columns,
            causal.causality_audit,
        ),
        changed_history,
        target="fault",
        registry=registry,
    )
    original_row = compact.frame.loc[
        compact.frame["hour_utc"].eq(cutoff), list(compact.model_feature_columns)
    ].iloc[0]
    changed_row = changed.frame.loc[
        changed.frame["hour_utc"].eq(cutoff), list(changed.model_feature_columns)
    ].iloc[0]
    pd.testing.assert_series_equal(original_row, changed_row, check_names=False)


def test_causal_neighbor_count_uses_only_already_installed_stations() -> None:
    states = [ROW_STATE_COMPLETE] * 30
    availability = _availability_grid(states)
    availability["absent_sensor_groups"] = ""
    observations = availability.loc[:, ["station_id", "hour_utc"]].copy()
    observations["n_raw_records"] = 2
    registry = pd.DataFrame(
        {
            "station_id": ["S1", "S2"],
            "latitude": [32.9, 32.95],
            "longitude": [13.2, 13.25],
            "elevation": [20.0, 25.0],
            "install_date": ["2025-12-01", "2026-01-02"],
        }
    )
    grid = build_risk_dataset(availability, horizons=(2,)).frame
    features = build_causal_forecast_features(
        grid,
        availability=availability,
        observations=observations,
        registry=registry,
        feature_matrix_columns=(),
    ).frame

    before_install = pd.Timestamp("2026-01-01T12:00:00Z")
    after_install = pd.Timestamp("2026-01-02T00:00:00Z")
    assert int(
        features.loc[
            features["hour_utc"].eq(before_install), "ctx_n_neighbors"
        ].iloc[0]
    ) == 0
    assert int(
        features.loc[
            features["hour_utc"].eq(after_install), "ctx_n_neighbors"
        ].iloc[0]
    ) == 1


def test_forecast_hgb_receives_training_derived_class_weights_only(monkeypatch) -> None:
    captured_weights: list[np.ndarray] = []

    class RecordingModel:
        def fit(self, values, target, sample_weight):
            captured_weights.append(np.asarray(sample_weight, dtype=float).copy())
            return self

        def predict_proba(self, values):
            feature = np.asarray(values, dtype=float)[:, 0]
            probability = np.clip(feature, 0.05, 0.95)
            return np.column_stack([1.0 - probability, probability])

    monkeypatch.setattr(
        "src.availability.risk_model.make_forecast_hist_gradient_boosting",
        lambda seed=2026: RecordingModel(),
    )
    train = pd.DataFrame({"f": [0.1, 0.2, 0.3, 0.9], "y": [0, 0, 0, 1]})
    validation = pd.DataFrame({"f": [0.1, 0.8], "y": [0, 1]})

    selection = fit_forecast_hist_gradient_boosting(
        train,
        validation,
        ("f",),
        weight_multipliers=(1.0,),
        thresholds=(0.5,),
    )

    np.testing.assert_array_equal(captured_weights[0], np.asarray([1.0, 1.0, 1.0, 3.0]))
    assert selection.positive_class_weight == pytest.approx(3.0)
    assert selection.selection_trace["selected"].sum() == 1
