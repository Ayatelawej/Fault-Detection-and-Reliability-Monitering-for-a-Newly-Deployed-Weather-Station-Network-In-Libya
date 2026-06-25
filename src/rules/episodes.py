from __future__ import annotations

import pandas as pd

from src.rules.channel_handlers import sensor_group_for_channel


OUTPUT_COLUMNS = [
    "station_id",
    "start_hour",
    "end_hour",
    "duration_hours",
    "n_events",
    "n_channels",
    "affected_channels",
    "n_sensor_groups",
    "affected_sensor_groups",
    "dominant_detector",
    "detector_concordance",
    "max_abs_zscore",
    "max_iforest_score",
    "min_rolling_variance",
    "reasons",
]
DETECTOR_PRIORITY = ["stuck", "iforest", "zscore"]
REASON_TO_DETECTOR = {
    "mad_high": "zscore",
    "stuck_variance_zero": "stuck",
    "iforest_outlier": "iforest",
}


def _tokens(values: pd.Series) -> set[str]:
    tokens: set[str] = set()

    for value in values.dropna().astype(str):
        tokens.update(token for token in value.split("|") if token)

    return tokens


def _dominant_detector(events: pd.DataFrame) -> str:
    counts = events["dominant_detector"].value_counts().to_dict()
    return max(
        DETECTOR_PRIORITY,
        key=lambda detector: int(counts.get(detector, 0)),
    )


def _detector_concordance(reason_tokens: set[str]) -> int:
    detectors = {
        REASON_TO_DETECTOR[token]
        for token in reason_tokens
        if token in REASON_TO_DETECTOR
    }
    return int(len(detectors))


def _episode_row(events: pd.DataFrame) -> dict[str, object]:
    start_hour = events["start_hour"].min()
    end_hour = events["end_hour"].max()
    channels = sorted(events["channel"].dropna().astype(str).unique())
    sensor_groups = sorted({sensor_group_for_channel(channel) for channel in channels})
    reason_tokens = _tokens(events["reasons"])
    duration_hours = int((end_hour - start_hour) / pd.Timedelta(hours=1)) + 1

    return {
        "station_id": events["station_id"].iloc[0],
        "start_hour": start_hour,
        "end_hour": end_hour,
        "duration_hours": duration_hours,
        "n_events": int(len(events)),
        "n_channels": int(len(channels)),
        "affected_channels": "|".join(channels),
        "n_sensor_groups": int(len(sensor_groups)),
        "affected_sensor_groups": "|".join(sensor_groups),
        "dominant_detector": _dominant_detector(events),
        "detector_concordance": _detector_concordance(reason_tokens),
        "max_abs_zscore": events["max_abs_zscore"].max(skipna=True),
        "max_iforest_score": events["max_iforest_score"].max(skipna=True),
        "min_rolling_variance": events["min_rolling_variance"].min(skipna=True),
        "reasons": "|".join(sorted(reason_tokens)),
    }


def _station_episodes(
    station_events: pd.DataFrame,
    onset_tolerance_hours: int,
) -> list[pd.DataFrame]:
    station_events = station_events.sort_values(
        ["start_hour", "channel"],
    ).reset_index(drop=True)
    unassigned = list(station_events.index)
    episodes: list[pd.DataFrame] = []
    tolerance = pd.Timedelta(hours=onset_tolerance_hours)

    while unassigned:
        anchor_index = unassigned[0]
        anchor = station_events.loc[anchor_index]
        anchor_start = anchor["start_hour"]
        running_end = anchor["end_hour"]
        member_indices = [anchor_index]
        remaining: list[int] = []

        for event_index in unassigned[1:]:
            event = station_events.loc[event_index]
            within_onset = event["start_hour"] - anchor_start <= tolerance
            overlaps = event["start_hour"] <= running_end

            if within_onset and overlaps:
                member_indices.append(event_index)
                running_end = max(running_end, event["end_hour"])
            else:
                remaining.append(event_index)

        episodes.append(station_events.loc[member_indices].copy())
        unassigned = remaining

    return episodes


def build_episodes(
    events_df: pd.DataFrame,
    onset_tolerance_hours: int = 6,
) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frame = events_df.copy()
    frame["start_hour"] = pd.to_datetime(frame["start_hour"], utc=True)
    frame["end_hour"] = pd.to_datetime(frame["end_hour"], utc=True)
    episode_rows: list[dict[str, object]] = []

    for _, station_events in frame.groupby("station_id", sort=False):
        for episode_events in _station_episodes(
            station_events,
            onset_tolerance_hours,
        ):
            episode_rows.append(_episode_row(episode_events))

    result = pd.DataFrame(episode_rows, columns=OUTPUT_COLUMNS)
    return result.sort_values(
        ["station_id", "start_hour"],
    ).reset_index(drop=True)
