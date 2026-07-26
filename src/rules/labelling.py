from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.rules.channel_handlers import sensor_group_for_channel
from src.rules.config import (
    PHYSICAL_LIMIT_RULES,
    ROLLING_VARIANCE_WINDOW_HOURS,
    STUCK_IGNORE_ZERO_CHANNELS,
    STUCK_SKIP_CHANNELS,
)
from src.rules.detectors.rolling_variance import detect_stuck_values
from src.rules.physical_limits import physical_limit_flags
from src.rules.statistical_gate import statistical_channels_for_window


CALM_WIND_MAX_KMH = 1.0
MECHANISM_ORDER = [
    "spike_impossible",
    "statistical_anomaly",
    "calibration_offset",
    "stuck_flatline",
]
OUTPUT_COLUMNS = [
    "episode_id",
    "station_id",
    "start_hour",
    "end_hour",
    "duration_hours",
    "binary_fault",
    "label_state",
    "mechanisms",
    "components",
    "fired_channels",
    "period",
]
CROSSWALK_COLUMNS = [
    "frozen_row_id",
    "labelled_episode_id",
    "station_id",
    "frozen_start_hour",
    "frozen_end_hour",
    "labelled_start_hour",
    "labelled_end_hour",
    "overlap_hours",
    "overlap_fraction_frozen",
    "overlap_fraction_labelled",
    "interval_iou",
    "frozen_family",
    "labelled_binary_fault",
    "labelled_label_state",
    "labelled_mechanisms",
    "labelled_components",
    "pair_agreement",
    "frozen_aggregate_mechanisms",
    "frozen_aggregate_components",
    "frozen_aggregate_label_states",
    "frozen_agreement",
    "pair_training_state",
    "frozen_match_degree",
    "labelled_match_degree",
]


def normalize_times(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], utc=True, format="mixed")
    return result


def period_for_start(value: object) -> str:
    timestamp = pd.to_datetime(value, utc=True)
    if timestamp < pd.Timestamp("2026-04-01", tz="UTC"):
        return "pre_april"
    if timestamp < pd.Timestamp("2026-06-01", tz="UTC"):
        return "april_may"
    return "june"


def _tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {token for token in str(value).split("|") if token}


def _canonical_channel(channel: str) -> str:
    if channel in {"winddir_sin", "winddir_cos"}:
        return "winddir_avg_deg"
    return channel


def _score_channels(channel: str) -> set[str]:
    if channel == "winddir_avg_deg":
        return {"winddir_sin", "winddir_cos"}
    return {channel}


def _hour_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="h", tz="UTC")


def _fallback_stuck_times(raw_window: pd.DataFrame, channel: str) -> set[pd.Timestamp]:
    if channel in STUCK_SKIP_CHANNELS or channel not in raw_window.columns:
        return set()
    values = raw_window.loc[:, ["hour_utc", channel]].copy()
    values[channel] = pd.to_numeric(values[channel], errors="coerce")
    if values[channel].notna().sum() < ROLLING_VARIANCE_WINDOW_HOURS:
        return set()
    indexed = values.drop_duplicates("hour_utc").set_index("hour_utc")[channel]
    grid = _hour_grid(indexed.index.min(), indexed.index.max())
    series = indexed.reindex(grid)
    detected = detect_stuck_values(
        series,
        ignore_zero=channel in STUCK_IGNORE_ZERO_CHANNELS,
    )
    return set(detected.index[detected["flag"].astype(bool)])


def _score_stuck_times(
    score_window: pd.DataFrame,
    score_channels: set[str],
) -> set[pd.Timestamp]:
    if score_window.empty or "channel" not in score_window.columns:
        return set()
    subset = score_window.loc[score_window["channel"].isin(score_channels)].copy()
    if subset.empty:
        return set()
    flag_stuck = subset.get("flag_stuck", pd.Series(False, index=subset.index)).fillna(False).astype(bool)
    return set(pd.to_datetime(subset.loc[flag_stuck, "hour_utc"], utc=True))


def _stuck_times(
    raw_window: pd.DataFrame,
    score_window: pd.DataFrame,
    channel: str,
) -> set[pd.Timestamp]:
    score_times = _score_stuck_times(score_window, _score_channels(channel))
    fallback_times = _fallback_stuck_times(raw_window, channel)
    return score_times | fallback_times


def _direction_stuck_times(
    raw_window: pd.DataFrame,
    score_window: pd.DataFrame,
) -> set[pd.Timestamp]:
    sin_times = _score_stuck_times(score_window, {"winddir_sin"})
    cos_times = _score_stuck_times(score_window, {"winddir_cos"})
    if sin_times or cos_times:
        return sin_times & cos_times
    return _fallback_stuck_times(raw_window, "winddir_avg_deg")


def _wind_is_calm(raw_window: pd.DataFrame, timestamps: set[pd.Timestamp]) -> bool | None:
    if not timestamps or "windspeed_avg_kmh" not in raw_window.columns:
        return None
    values = raw_window.loc[
        raw_window["hour_utc"].isin(timestamps),
        "windspeed_avg_kmh",
    ]
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return bool(values.abs().le(CALM_WIND_MAX_KMH).all())


def _add_finding(
    findings: dict[str, set[str]],
    mechanism: str,
    channel: str,
) -> None:
    findings[mechanism].add(channel)


def classify_episode(
    episode: pd.Series | dict[str, object],
    raw_window: pd.DataFrame,
    score_window: pd.DataFrame,
    statistical_evidence_window: pd.DataFrame | None = None,
) -> dict[str, object]:
    row = episode if isinstance(episode, pd.Series) else pd.Series(episode)
    raw = normalize_times(raw_window, ["hour_utc"])
    scores = normalize_times(score_window, ["hour_utc"]) if not score_window.empty else score_window.copy()
    statistical = (
        normalize_times(statistical_evidence_window, ["hour_utc"])
        if statistical_evidence_window is not None and not statistical_evidence_window.empty
        else pd.DataFrame()
    )
    findings: dict[str, set[str]] = defaultdict(set)
    channels = [channel for channel in PHYSICAL_LIMIT_RULES if channel in raw.columns]
    direction_times = _direction_stuck_times(raw, scores)
    speed_times = _stuck_times(raw, scores, "windspeed_avg_kmh")
    shared_wind_times = direction_times & speed_times

    for channel in channels:
        values = pd.to_numeric(raw[channel], errors="coerce")
        hard = bool(physical_limit_flags(values, channel).any())
        if hard:
            _add_finding(findings, "spike_impossible", channel)

        if channel == "winddir_avg_deg":
            if shared_wind_times:
                continue
            calm = _wind_is_calm(raw, direction_times)
            if direction_times and calm is False:
                _add_finding(findings, "stuck_flatline", channel)
            continue

        if channel == "windspeed_avg_kmh" and shared_wind_times:
            _add_finding(findings, "stuck_flatline", channel)
            continue

        if _stuck_times(raw, scores, channel):
            _add_finding(findings, "stuck_flatline", channel)

    for channel in statistical_channels_for_window(statistical):
        _add_finding(findings, "statistical_anomaly", channel)

    mechanisms = [mechanism for mechanism in MECHANISM_ORDER if findings.get(mechanism)]
    components = sorted(
        {
            sensor_group_for_channel(channel)
            for channels_for_mechanism in findings.values()
            for channel in channels_for_mechanism
        },
    )
    fired = []
    for mechanism in mechanisms:
        channels_for_mechanism = sorted(findings[mechanism])
        if channels_for_mechanism:
            fired.append(f"{mechanism}=" + "+".join(channels_for_mechanism))
    return {
        "binary_fault": int(bool(mechanisms)),
        "mechanisms": "|".join(mechanisms),
        "components": "|".join(components),
        "fired_channels": "|".join(fired),
    }


def build_episode_labels(
    episodes: pd.DataFrame,
    raw: pd.DataFrame,
    scores: pd.DataFrame,
    statistical_evidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    episode_frame = normalize_times(episodes, ["start_hour", "end_hour"])
    raw_frame = normalize_times(raw, ["hour_utc"])
    score_frame = normalize_times(scores, ["hour_utc"])
    evidence_frame = (
        normalize_times(statistical_evidence, ["hour_utc"])
        if statistical_evidence is not None and not statistical_evidence.empty
        else pd.DataFrame(columns=["station_id", "hour_utc"])
    )
    if raw_frame.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("hourly input has duplicate station-hour rows")
    episode_frame = episode_frame.sort_values(
        ["station_id", "start_hour", "end_hour"],
    ).reset_index(drop=True)
    raw_by_station = {
        str(station_id): frame.sort_values("hour_utc")
        for station_id, frame in raw_frame.groupby("station_id", sort=False)
    }
    scores_by_station = {
        str(station_id): frame.sort_values("hour_utc")
        for station_id, frame in score_frame.groupby("station_id", sort=False)
    }
    evidence_by_station = {
        str(station_id): frame.sort_values("hour_utc")
        for station_id, frame in evidence_frame.groupby("station_id", sort=False)
    }
    rows = []
    for position, episode in episode_frame.iterrows():
        station_id = str(episode["station_id"])
        start = pd.to_datetime(episode["start_hour"], utc=True)
        end = pd.to_datetime(episode["end_hour"], utc=True)
        raw_station = raw_by_station.get(station_id, pd.DataFrame(columns=raw_frame.columns))
        score_station = scores_by_station.get(station_id, pd.DataFrame(columns=score_frame.columns))
        evidence_station = evidence_by_station.get(station_id, pd.DataFrame(columns=evidence_frame.columns))
        raw_window = raw_station.loc[
            raw_station["hour_utc"].between(start, end, inclusive="both"),
        ]
        score_window = score_station.loc[
            score_station["hour_utc"].between(start, end, inclusive="both"),
        ]
        evidence_window = evidence_station.loc[
            evidence_station["hour_utc"].between(start, end, inclusive="both"),
        ]
        classified = classify_episode(
            episode,
            raw_window,
            score_window,
            evidence_window,
        )
        rows.append(
            {
                "episode_id": f"v2_{position + 1:06d}",
                "station_id": station_id,
                "start_hour": start,
                "end_hour": end,
                "duration_hours": int(episode["duration_hours"]),
                **classified,
                "label_state": "fault" if classified["binary_fault"] else "benign",
                "period": period_for_start(start),
            },
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _mechanism_set(value: object) -> set[str]:
    return _tokens(value)


def _component_set(value: object) -> set[str]:
    return _tokens(value)


def _family_agreement(
    family: object,
    binary_fault: bool,
    mechanisms: set[str],
    components: set[str],
) -> bool:
    value = str(family)
    if value == "outlier_benign":
        return not binary_fault
    if value == "spike_impossible":
        return "spike_impossible" in mechanisms
    if value == "stuck_flatline":
        return "stuck_flatline" in mechanisms
    if value == "systemic_array":
        return bool(mechanisms) and len(components) >= 2
    return False


def _interval_metrics(
    frozen_start: pd.Timestamp,
    frozen_end: pd.Timestamp,
    labelled_start: pd.Timestamp,
    labelled_end: pd.Timestamp,
) -> tuple[int, float, float, float]:
    overlap_start = max(frozen_start, labelled_start)
    overlap_end = min(frozen_end, labelled_end)
    overlap_hours = int((overlap_end - overlap_start) / pd.Timedelta(hours=1)) + 1
    frozen_hours = int((frozen_end - frozen_start) / pd.Timedelta(hours=1)) + 1
    labelled_hours = int((labelled_end - labelled_start) / pd.Timedelta(hours=1)) + 1
    union_hours = frozen_hours + labelled_hours - overlap_hours
    return (
        overlap_hours,
        float(overlap_hours / frozen_hours),
        float(overlap_hours / labelled_hours),
        float(overlap_hours / union_hours),
    )


def build_crosswalk(
    frozen: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    frozen_frame = normalize_times(frozen, ["start_hour", "end_hour"]).reset_index(drop=True)
    labelled_frame = normalize_times(labels, ["start_hour", "end_hour"]).reset_index(drop=True)
    frozen_frame["frozen_row_id"] = [f"frozen_{position + 1:06d}" for position in frozen_frame.index]
    edges = []
    for station_id, frozen_station in frozen_frame.groupby("station_id", sort=False):
        labelled_station = labelled_frame.loc[
            labelled_frame["station_id"].astype(str).eq(str(station_id)),
        ]
        if labelled_station.empty:
            continue
        for _, frozen_row in frozen_station.iterrows():
            matches = labelled_station.loc[
                labelled_station["start_hour"].le(frozen_row["end_hour"])
                & labelled_station["end_hour"].ge(frozen_row["start_hour"]),
            ]
            for _, labelled_row in matches.iterrows():
                overlap_hours, frozen_fraction, labelled_fraction, interval_iou = _interval_metrics(
                    frozen_row["start_hour"],
                    frozen_row["end_hour"],
                    labelled_row["start_hour"],
                    labelled_row["end_hour"],
                )
                mechanisms = _mechanism_set(labelled_row["mechanisms"])
                components = _component_set(labelled_row["components"])
                edges.append(
                    {
                        "frozen_row_id": frozen_row["frozen_row_id"],
                        "labelled_episode_id": labelled_row["episode_id"],
                        "station_id": str(station_id),
                        "frozen_start_hour": frozen_row["start_hour"],
                        "frozen_end_hour": frozen_row["end_hour"],
                        "labelled_start_hour": labelled_row["start_hour"],
                        "labelled_end_hour": labelled_row["end_hour"],
                        "overlap_hours": overlap_hours,
                        "overlap_fraction_frozen": frozen_fraction,
                        "overlap_fraction_labelled": labelled_fraction,
                        "interval_iou": interval_iou,
                        "frozen_family": frozen_row["label"],
                        "labelled_binary_fault": int(labelled_row["binary_fault"]),
                        "labelled_label_state": str(labelled_row.get("label_state", "fault" if labelled_row["binary_fault"] else "benign")),
                        "labelled_mechanisms": labelled_row["mechanisms"],
                        "labelled_components": labelled_row["components"],
                        "pair_agreement": _family_agreement(
                            frozen_row["label"],
                            bool(labelled_row["binary_fault"]),
                            mechanisms,
                            components,
                        ),
                    },
                )
    if not edges:
        return pd.DataFrame(columns=CROSSWALK_COLUMNS)
    result = pd.DataFrame(edges)
    frozen_summary = []
    for frozen_row_id, group in result.groupby("frozen_row_id", sort=False):
        mechanisms = set().union(*[_mechanism_set(value) for value in group["labelled_mechanisms"]])
        components = set().union(*[_component_set(value) for value in group["labelled_components"]])
        family = group["frozen_family"].iloc[0]
        frozen_summary.append(
            {
                "frozen_row_id": frozen_row_id,
                "frozen_aggregate_mechanisms": "|".join(
                    [mechanism for mechanism in MECHANISM_ORDER if mechanism in mechanisms],
                ),
                "frozen_aggregate_components": "|".join(sorted(components)),
                "frozen_aggregate_label_states": "|".join(sorted(set(group["labelled_label_state"]))),
                "frozen_agreement": _family_agreement(
                    family,
                    bool(mechanisms),
                    mechanisms,
                    components,
                ),
                "frozen_match_degree": int(len(group)),
            },
        )
    labelled_degree = result.groupby("labelled_episode_id").size().rename("labelled_match_degree").reset_index()
    result = result.merge(pd.DataFrame(frozen_summary), on="frozen_row_id", how="left")
    result = result.merge(labelled_degree, on="labelled_episode_id", how="left")
    result["pair_training_state"] = np.where(
        result["labelled_label_state"].eq("borderline_review"),
        "held_out",
        "eligible",
    )
    return result.loc[:, CROSSWALK_COLUMNS].sort_values(
        ["station_id", "frozen_start_hour", "labelled_start_hour", "labelled_episode_id"],
    ).reset_index(drop=True)


def crosswalk_summary(
    frozen: pd.DataFrame,
    labels: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> dict[str, object]:
    frozen_ids = {f"frozen_{position + 1:06d}" for position in range(len(frozen))}
    labelled_ids = set(labels["episode_id"].astype(str))
    matched_frozen = set(crosswalk.get("frozen_row_id", pd.Series(dtype=object)).astype(str))
    matched_labelled = set(crosswalk.get("labelled_episode_id", pd.Series(dtype=object)).astype(str))
    aggregate = crosswalk.drop_duplicates("frozen_row_id") if not crosswalk.empty else crosswalk
    frozen_agree = int(aggregate.get("frozen_agreement", pd.Series(dtype=bool)).fillna(False).sum())
    frozen_matched = int(len(aggregate))
    return {
        "pair_count": int(len(crosswalk)),
        "pair_agree": int(crosswalk.get("pair_agreement", pd.Series(dtype=bool)).fillna(False).sum()),
        "frozen_matched": frozen_matched,
        "frozen_agree": frozen_agree,
        "frozen_unmatched": int(len(frozen_ids - matched_frozen)),
        "labelled_matched": int(len(matched_labelled)),
        "labelled_unmatched": int(len(labelled_ids - matched_labelled)),
    }


def summary_by_period_and_station(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = (
        labels.groupby(["period", "station_id"], as_index=False)
        .agg(
            episodes=("episode_id", "size"),
            faults=("binary_fault", "sum"),
        )
        .sort_values(["period", "station_id"])
    )
    base["fault_rate"] = base["faults"] / base["episodes"]
    mechanism_rows = labels.assign(mechanism=labels["mechanisms"].str.split("|"))
    mechanism_rows = mechanism_rows.explode("mechanism")
    mechanism_rows = mechanism_rows.loc[mechanism_rows["mechanism"].fillna("").ne("")]
    mechanism = (
        mechanism_rows.groupby(["period", "station_id", "mechanism"]).size()
        .unstack(fill_value=0)
        .reset_index()
        if not mechanism_rows.empty
        else pd.DataFrame(columns=["period", "station_id"])
    )
    component_rows = labels.assign(component=labels["components"].str.split("|"))
    component_rows = component_rows.explode("component")
    component_rows = component_rows.loc[component_rows["component"].fillna("").ne("")]
    component = (
        component_rows.groupby(["period", "station_id", "component"]).size()
        .unstack(fill_value=0)
        .reset_index()
        if not component_rows.empty
        else pd.DataFrame(columns=["period", "station_id"])
    )
    station_summary = base.merge(mechanism, on=["period", "station_id"], how="left")
    station_summary = station_summary.merge(component, on=["period", "station_id"], how="left")
    station_summary = station_summary.fillna(0)
    period_summary = (
        station_summary.drop(columns=["station_id"])
        .groupby("period", as_index=False)
        .sum(numeric_only=True)
        .sort_values("period")
    )
    period_summary["fault_rate"] = period_summary["faults"] / period_summary["episodes"]
    return period_summary, station_summary
