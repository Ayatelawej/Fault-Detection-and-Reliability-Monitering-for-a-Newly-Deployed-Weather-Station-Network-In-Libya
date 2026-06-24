from __future__ import annotations

import pandas as pd


OUTPUT_COLUMNS = [
    "station_id",
    "channel",
    "start_hour",
    "end_hour",
    "duration_hours",
    "dominant_detector",
    "detector_concordance",
    "max_abs_zscore",
    "max_iforest_score",
    "min_rolling_variance",
    "reasons",
]


DETECTOR_COLUMNS = {
    "stuck": "flag_stuck",
    "iforest": "flag_iforest",
    "zscore": "flag_zscore",
}


def _dominant_detector(event_rows: pd.DataFrame) -> str:
    counts = {
        detector: int(event_rows[column].fillna(False).astype(bool).sum())
        for detector, column in DETECTOR_COLUMNS.items()
    }
    return max(["stuck", "iforest", "zscore"], key=lambda detector: counts[detector])


def _detector_concordance(event_rows: pd.DataFrame) -> int:
    return int(
        sum(
            bool(event_rows[column].fillna(False).astype(bool).any())
            for column in DETECTOR_COLUMNS.values()
        )
    )


def _reasons(event_rows: pd.DataFrame) -> str:
    tokens: set[str] = set()

    for reason in event_rows["reason"].dropna().astype(str):
        tokens.update(token for token in reason.split("|") if token)

    return "|".join(sorted(tokens))


def _event_row(event_rows: pd.DataFrame) -> dict[str, object]:
    return {
        "station_id": event_rows["station_id"].iloc[0],
        "channel": event_rows["channel"].iloc[0],
        "start_hour": event_rows["hour_utc"].iloc[0],
        "end_hour": event_rows["hour_utc"].iloc[-1],
        "duration_hours": int(len(event_rows)),
        "dominant_detector": _dominant_detector(event_rows),
        "detector_concordance": _detector_concordance(event_rows),
        "max_abs_zscore": event_rows["zscore"].abs().max(skipna=True),
        "max_iforest_score": event_rows["iforest_score"].max(skipna=True),
        "min_rolling_variance": event_rows["rolling_variance"].min(skipna=True),
        "reasons": _reasons(event_rows),
    }


def build_events(scores_df: pd.DataFrame, max_gap_hours: int = 0) -> pd.DataFrame:
    frame = scores_df.copy()
    frame["hour_utc"] = pd.to_datetime(frame["hour_utc"], utc=True)
    frame = frame.loc[frame["flag"].fillna(False).astype(bool)].copy()

    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    allowed_gap = pd.Timedelta(hours=max_gap_hours + 1)
    events: list[dict[str, object]] = []

    for _, group in frame.groupby(["station_id", "channel"], sort=False):
        group = group.sort_values("hour_utc").reset_index(drop=True)
        gaps = group["hour_utc"].diff()
        run_ids = gaps.gt(allowed_gap).fillna(False).cumsum()

        for _, event_rows in group.groupby(run_ids, sort=False):
            events.append(_event_row(event_rows))

    result = pd.DataFrame(events, columns=OUTPUT_COLUMNS)
    return result.sort_values(
        ["station_id", "channel", "start_hour"],
    ).reset_index(drop=True)
