from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import MERGED_DATASET_PATH, PROCESSED_DIR
from src.rules.episodes import build_episodes
from src.rules.events import build_events
from src.rules.score import compute_anomaly_scores

OUTPUT_PATH = PROCESSED_DIR / "statistical_anomaly_scores.parquet"
EVENT_OUTPUT_PATH = PROCESSED_DIR / "fault_events.parquet"
EPISODE_OUTPUT_PATH = PROCESSED_DIR / "fault_episodes.parquet"
METADATA_NUMERIC_COLUMNS = {
    "n_raw_records",
    "latitude",
    "longitude",
    "qc_status",
    "epoch",
    "data_present",
    "elevation",
}
EXPECTED_NETWORK_POOLED_STATIONS = {"IJANZO4", "I90583612", "ITRIPO32", "IDERNA7"}


def _rate(value: float) -> str:
    return f"{value:.4%}"


def _numeric_channels(df: pd.DataFrame) -> list[str]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return [
        column
        for column in numeric_columns
        if column not in METADATA_NUMERIC_COLUMNS
    ]


def _per_channel_table(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("channel", as_index=False)
        .agg(
            n_rows=("flag", "size"),
            flag_rate=("flag", "mean"),
            flag_zscore=("flag_zscore", "mean"),
            flag_stuck=("flag_stuck", "mean"),
            flag_iforest=("flag_iforest", "mean"),
        )
        .sort_values("flag_rate", ascending=False)
    )


def _print_detector_contribution(scores: pd.DataFrame) -> None:
    flagged = scores.loc[scores["flag"]]
    print()
    print("DETECTOR CONTRIBUTION")
    if flagged.empty:
        print("No flagged rows.")
        return

    for reason in ["mad_high", "stuck_variance_zero", "iforest_outlier"]:
        fraction = flagged["reason"].str.contains(reason, regex=False).mean()
        print(f"{reason}: {_rate(float(fraction))}")


def _print_baseline_sources(scores: pd.DataFrame) -> None:
    source_table = (
        scores.assign(value=1)
        .pivot_table(
            index="station_id",
            columns="baseline_source",
            values="value",
            aggfunc="sum",
            fill_value=0,
        )
        .pipe(lambda frame: frame.div(frame.sum(axis=1), axis=0))
    )

    for column in ["network_pooled", "station"]:
        if column not in source_table.columns:
            source_table[column] = 0.0

    source_table = source_table.loc[:, ["network_pooled", "station"]].sort_index()
    printable = source_table.reset_index()
    print()
    print("BASELINE SOURCE")
    print(
        printable.to_string(
            index=False,
            formatters={
                "network_pooled": _rate,
                "station": _rate,
            },
        )
    )

    print()
    print("EXPECTED NETWORK-POOLED CHECK")
    for station_id in sorted(EXPECTED_NETWORK_POOLED_STATIONS):
        if station_id not in source_table.index:
            print(f"{station_id}: MISSING")
            continue

        pooled_fraction = float(source_table.loc[station_id, "network_pooled"])
        outcome = "YES" if pooled_fraction > 0.5 else "NO"
        print(
            f"{station_id}: predominantly network_pooled={outcome} "
            f"({ _rate(pooled_fraction) })"
        )

    other_stations = [
        station_id
        for station_id in source_table.index
        if station_id not in EXPECTED_NETWORK_POOLED_STATIONS
    ]
    station_dominant = source_table.loc[other_stations, "station"].gt(0.5)
    print(
        "Other stations predominantly station="
        f"{'YES' if bool(station_dominant.all()) else 'NO'} "
        f"({int(station_dominant.sum())}/{len(station_dominant)})"
    )


def _reason_counts(frame: pd.DataFrame) -> str:
    flagged = frame.loc[frame["flag"]]
    if flagged.empty:
        return "none"

    counts = flagged["reason"].value_counts().sort_index()
    return ", ".join(f"{reason}={count}" for reason, count in counts.items())


def _print_known_faults(scores: pd.DataFrame) -> None:
    wind_channels = [
        channel
        for channel in scores["channel"].unique()
        if channel in {"winddir_sin", "winddir_cos"}
        or channel.startswith("windspeed_")
        or channel.startswith("windgust_")
    ]
    precip_channels = [
        channel
        for channel in scores["channel"].unique()
        if channel.startswith("precip_")
    ]

    itripo33 = scores.loc[
        scores["station_id"].eq("ITRIPO33")
        & scores["channel"].isin(wind_channels)
    ]
    imurqu7 = scores.loc[
        scores["station_id"].eq("IMURQU7")
        & scores["channel"].isin(precip_channels)
    ]

    print()
    print("KNOWN-FAULT SPOT CHECKS")
    for label, frame in [
        ("ITRIPO33 windspeed/winddir", itripo33),
        ("IMURQU7 precip", imurqu7),
    ]:
        flagged = frame.loc[frame["flag"]]
        print(
            f"{label}: flagged_rows={len(flagged)}, "
            f"flagged_unique_hours={flagged['hour_utc'].nunique()}, "
            f"reasons={_reason_counts(frame)}"
        )


def _event_count_table(events: pd.DataFrame, column: str, limit: int = 8) -> pd.DataFrame:
    return (
        events[column]
        .value_counts()
        .head(limit)
        .rename_axis(column)
        .reset_index(name="n_events")
    )


def _print_known_fault_events(events: pd.DataFrame) -> None:
    wind_channels = [
        channel
        for channel in events["channel"].unique()
        if channel in {"winddir_sin", "winddir_cos"}
        or channel.startswith("windspeed_")
        or channel.startswith("windgust_")
    ]
    precip_channels = [
        channel
        for channel in events["channel"].unique()
        if channel.startswith("precip_")
    ]
    checks = [
        (
            "ITRIPO33 windspeed/winddir",
            events.loc[
                events["station_id"].eq("ITRIPO33")
                & events["channel"].isin(wind_channels)
            ],
        ),
        (
            "IMURQU7 precip",
            events.loc[
                events["station_id"].eq("IMURQU7")
                & events["channel"].isin(precip_channels)
            ],
        ),
    ]

    print()
    print("KNOWN-FAULT EVENTS")
    for label, frame in checks:
        printable = frame.loc[
            :,
            [
                "channel",
                "start_hour",
                "end_hour",
                "duration_hours",
                "dominant_detector",
                "reasons",
            ],
        ].sort_values(["channel", "start_hour"])
        print(f"{label}: n_events={len(printable)}")
        if printable.empty:
            print("none")
        else:
            print(printable.to_string(index=False))


def _print_event_summary(events: pd.DataFrame, scores: pd.DataFrame) -> None:
    flagged_hours = int(scores["flag"].sum())
    event_fraction = len(events) / flagged_hours if flagged_hours else 0.0
    long_events = int(events["duration_hours"].ge(24).sum()) if not events.empty else 0

    print()
    print("EVENT SUMMARY")
    print(f"Total events: {len(events):,}")
    print(f"Flagged station-hours: {flagged_hours:,}")
    print(f"Events per flagged station-hour: {_rate(event_fraction)}")

    if events.empty:
        print("No events.")
        return

    print()
    print("EVENTS BY DOMINANT DETECTOR")
    print(events["dominant_detector"].value_counts().to_string())
    print()
    print("DURATION HOURS")
    print(f"median: {float(events['duration_hours'].median()):.2f}")
    print(f"max: {int(events['duration_hours'].max())}")
    print(f"events_duration_ge_24: {long_events}")
    print()
    print("DETECTOR CONCORDANCE")
    print(events["detector_concordance"].value_counts().sort_index().to_string())
    print()
    print("TOP CHANNELS BY EVENT COUNT")
    print(_event_count_table(events, "channel").to_string(index=False))
    print()
    print("TOP STATIONS BY EVENT COUNT")
    print(_event_count_table(events, "station_id").to_string(index=False))
    _print_known_fault_events(events)


def _episode_count_table(
    episodes: pd.DataFrame,
    column: str,
    limit: int = 10,
) -> pd.DataFrame:
    return (
        episodes[column]
        .value_counts()
        .head(limit)
        .rename_axis(column)
        .reset_index(name="n_episodes")
    )


def _print_known_fault_episodes(episodes: pd.DataFrame) -> None:
    print()
    print("KNOWN-FAULT EPISODES")
    checks = [
        ("ITRIPO33 wind", "ITRIPO33", {"anemometer", "wind_vane"}),
        ("IBIRAL3 pressure", "IBIRAL3", {"barometer"}),
        ("IMURQU7 precip", "IMURQU7", {"rain_gauge"}),
    ]

    for label, station_id, sensor_groups in checks:
        station_frame = episodes.loc[episodes["station_id"].eq(station_id)]
        group_mask = station_frame["affected_sensor_groups"].map(
            lambda value: bool(set(str(value).split("|")) & sensor_groups)
        )
        frame = station_frame.loc[group_mask]
        printable = frame.loc[
            :,
            [
                "start_hour",
                "duration_hours",
                "affected_sensor_groups",
            ],
        ].sort_values("start_hour")
        print(f"{label}: n_episodes={len(printable)}")
        if printable.empty:
            print("none")
        else:
            print(printable.to_string(index=False))


def _print_episode_summary(episodes: pd.DataFrame, events: pd.DataFrame) -> None:
    reduction = 1.0 - (len(episodes) / len(events)) if len(events) else 0.0

    print()
    print("EPISODE SUMMARY")
    print(f"Total events: {len(events):,}")
    print(f"Total episodes: {len(episodes):,}")
    print(f"Event-to-episode reduction: {_rate(float(reduction))}")

    if episodes.empty:
        print("No episodes.")
        return

    print()
    print("EPISODES BY SENSOR-GROUP COUNT")
    print(episodes["n_sensor_groups"].value_counts().sort_index().to_string())
    print()
    print("TOP SENSOR-GROUP SIGNATURES BY EPISODE COUNT")
    print(
        _episode_count_table(
            episodes,
            "affected_sensor_groups",
        ).to_string(index=False)
    )
    print()
    print("EPISODE DURATION HOURS")
    print(f"median: {float(episodes['duration_hours'].median()):.2f}")
    print(f"max: {int(episodes['duration_hours'].max())}")
    print(
        "episodes_duration_ge_24: "
        f"{int(episodes['duration_hours'].ge(24).sum())}"
    )
    _print_known_fault_episodes(episodes)


def main() -> None:
    df = pd.read_csv(MERGED_DATASET_PATH, parse_dates=["hour_utc"])
    channels = _numeric_channels(df)
    scores = compute_anomaly_scores(df, channels=channels)
    events = build_events(scores)
    episodes = build_episodes(events)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(OUTPUT_PATH, index=False)
    events.to_parquet(EVENT_OUTPUT_PATH, index=False)
    episodes.to_parquet(EPISODE_OUTPUT_PATH, index=False)

    scored_channel_count = int(scores["channel"].nunique())
    print("STAGE 3 SCORE DIAGNOSTIC")
    print(f"Input path: {MERGED_DATASET_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Event output path: {EVENT_OUTPUT_PATH}")
    print(f"Episode output path: {EPISODE_OUTPUT_PATH}")
    print(f"Input rows: {len(df):,}")
    print(f"Input scored channels: {len(channels)}")
    print(f"Output scored channels: {scored_channel_count}")
    print(f"Total scored rows: {len(scores):,}")
    print(f"Overall flag rate: {_rate(float(scores['flag'].mean()))}")
    print()
    print("PER-CHANNEL FLAG RATES")
    print(
        _per_channel_table(scores).to_string(
            index=False,
            formatters={
                "flag_rate": _rate,
                "flag_zscore": _rate,
                "flag_stuck": _rate,
                "flag_iforest": _rate,
            },
        )
    )

    _print_detector_contribution(scores)
    _print_baseline_sources(scores)
    _print_known_faults(scores)
    _print_event_summary(events, scores)
    _print_episode_summary(episodes, events)


if __name__ == "__main__":
    main()
