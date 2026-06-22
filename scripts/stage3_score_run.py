from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import MERGED_DATASET_PATH, PROCESSED_DIR
from src.rules.score import compute_anomaly_scores

OUTPUT_PATH = PROCESSED_DIR / "statistical_anomaly_scores.parquet"
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


def main() -> None:
    df = pd.read_csv(MERGED_DATASET_PATH, parse_dates=["hour_utc"])
    channels = _numeric_channels(df)
    scores = compute_anomaly_scores(df, channels=channels)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(OUTPUT_PATH, index=False)

    scored_channel_count = int(scores["channel"].nunique())
    print("STAGE 3 SCORE DIAGNOSTIC")
    print(f"Input path: {MERGED_DATASET_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
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


if __name__ == "__main__":
    main()
