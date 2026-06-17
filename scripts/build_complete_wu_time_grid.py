from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CADENCE = "5min"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete station-time grid from WU observations so "
            "missing/outage periods are represented as explicit blank rows."
        ),
    )
    parser.add_argument(
        "--pooled-csv",
        required=True,
        help="Pooled WU observation CSV.",
    )
    parser.add_argument(
        "--registry-csv",
        required=True,
        help="Public station registry CSV with station_id and start_date.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where complete-grid outputs will be written.",
    )
    parser.add_argument(
        "--cadence",
        default=DEFAULT_CADENCE,
        help="Grid cadence. Defaults to 5min.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive UTC end date in YYYY-MM-DD format. Defaults to max requested/API date.",
    )
    return parser.parse_args()


def read_inputs(
    pooled_csv: Path,
    registry_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = pd.read_csv(pooled_csv, low_memory=False)
    registry = pd.read_csv(registry_csv)

    if "obsTimeUtc" not in observations.columns:
        raise ValueError("pooled_csv must include obsTimeUtc.")
    if "stationID" not in observations.columns:
        raise ValueError("pooled_csv must include stationID.")

    required_registry = [
        "station_id",
        "station_name",
        "city",
        "country",
        "latitude",
        "longitude",
        "elevation_m",
        "start_date",
    ]
    missing_registry = [
        column for column in required_registry
        if column not in registry.columns
    ]
    if missing_registry:
        missing = ", ".join(missing_registry)
        raise ValueError(f"registry_csv is missing required columns: {missing}")

    return observations, registry


def infer_end_date(
    observations: pd.DataFrame,
    explicit_end_date: str | None,
) -> pd.Timestamp:
    if explicit_end_date:
        return pd.Timestamp(explicit_end_date, tz="UTC")

    if "requested_date" in observations.columns:
        requested_dates = pd.to_datetime(
            observations["requested_date"],
            errors="coerce",
            utc=True,
        )
        if requested_dates.notna().any():
            return requested_dates.max().normalize()

    obs_times = pd.to_datetime(
        observations["obsTimeUtc"],
        errors="coerce",
        utc=True,
    )
    if obs_times.notna().any():
        return obs_times.max().normalize()

    raise ValueError("Could not infer end date from observations.")


def normalize_observations(
    observations: pd.DataFrame,
    cadence: str,
) -> pd.DataFrame:
    observations = observations.copy()
    observations["obsTimeUtc"] = pd.to_datetime(
        observations["obsTimeUtc"],
        errors="coerce",
        utc=True,
    )
    observations = observations.dropna(subset=["stationID", "obsTimeUtc"])
    observations["time_bin_utc"] = observations["obsTimeUtc"].dt.floor(cadence)

    if "requested_date" in observations.columns:
        observations = observations.rename(
            columns={"requested_date": "api_request_date"}
        )

    bin_counts = (
        observations.groupby(["stationID", "time_bin_utc"], dropna=False)
        .size()
        .rename("n_observations_in_bin")
        .reset_index()
    )

    observations = observations.sort_values(
        ["stationID", "time_bin_utc", "obsTimeUtc"],
        kind="mergesort",
    )
    observations = observations.drop_duplicates(
        subset=["stationID", "time_bin_utc"],
        keep="last",
    )
    observations = observations.merge(
        bin_counts,
        on=["stationID", "time_bin_utc"],
        how="left",
    )
    observations["observation_present"] = 1
    return observations


def build_grid(
    registry: pd.DataFrame,
    *,
    end_date: pd.Timestamp,
    cadence: str,
) -> pd.DataFrame:
    rows = []
    final_timestamp = end_date + pd.Timedelta(days=1) - pd.Timedelta(cadence)

    for station in registry.itertuples(index=False):
        station_id = str(station.station_id)
        start = pd.Timestamp(station.start_date, tz="UTC").normalize()
        station_times = pd.date_range(
            start=start,
            end=final_timestamp,
            freq=cadence,
            tz="UTC",
        )
        rows.append(
            pd.DataFrame(
                {
                    "stationID": station_id,
                    "time_bin_utc": station_times,
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=["stationID", "time_bin_utc"])
    return pd.concat(rows, ignore_index=True)


def attach_registry_metadata(
    complete: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    metadata = registry[
        ["station_id", "latitude", "longitude", "elevation_m"]
    ].rename(columns={"station_id": "stationID"})
    complete = complete.drop(
        columns=["latitude", "longitude", "elevation_m"],
        errors="ignore",
    )
    return complete.merge(metadata, on="stationID", how="left")


def drop_empty_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep_columns = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            normalized = series.astype("string").str.strip().str.lower()
            has_value = (
                series.notna()
                & ~normalized.isin({"", "nan", "nat", "none", "null", "na", "n/a"})
            ).any()
        else:
            has_value = series.notna().any()
        if has_value:
            keep_columns.append(column)
    return frame[keep_columns]


def build_complete_dataset(
    observations: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    cadence: str,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    normalized = normalize_observations(observations, cadence)
    grid = build_grid(registry, end_date=end_date, cadence=cadence)

    complete = grid.merge(
        normalized,
        on=["stationID", "time_bin_utc"],
        how="left",
        suffixes=("", "_observed"),
    )
    complete["observation_present"] = (
        complete["observation_present"].fillna(0).astype("int64")
    )
    complete["n_observations_in_bin"] = (
        complete["n_observations_in_bin"].fillna(0).astype("int64")
    )
    complete = attach_registry_metadata(complete, registry)
    complete = drop_empty_columns(complete)

    preferred_first = [
        "stationID",
        "latitude",
        "longitude",
        "elevation_m",
        "time_bin_utc",
        "observation_present",
        "n_observations_in_bin",
        "obsTimeUtc",
        "obsTimeLocal",
        "api_request_date",
    ]
    ordered = [
        column for column in preferred_first
        if column in complete.columns
    ]
    ordered.extend(
        column for column in complete.columns
        if column not in ordered
    )
    return complete[ordered]


def write_outputs(
    complete: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    station_dir = output_dir / "per_station_complete"
    station_dir.mkdir(parents=True, exist_ok=True)

    pooled_path = output_dir / "observations_complete.csv"
    complete.to_csv(pooled_path, index=False, encoding="utf-8")

    for station_id, station_frame in complete.groupby("stationID", sort=True):
        station_path = station_dir / f"{station_id}_complete.csv"
        station_frame.to_csv(station_path, index=False, encoding="utf-8")

    missing_rows = int(complete["observation_present"].eq(0).sum())
    present_rows = int(complete["observation_present"].eq(1).sum())
    print(f"Complete pooled path: {pooled_path}")
    print(f"Per-station complete directory: {station_dir}")
    print(f"Total complete rows: {len(complete):,}")
    print(f"Rows with observations: {present_rows:,}")
    print(f"Explicit missing/outage rows: {missing_rows:,}")
    print(f"Columns: {len(complete.columns):,}")


def main() -> None:
    args = parse_args()
    pooled_csv = Path(args.pooled_csv)
    registry_csv = Path(args.registry_csv)
    output_dir = Path(args.output_dir)

    observations, registry = read_inputs(pooled_csv, registry_csv)
    end_date = infer_end_date(observations, args.end_date)
    complete = build_complete_dataset(
        observations,
        registry,
        cadence=args.cadence,
        end_date=end_date,
    )
    write_outputs(complete, output_dir)


if __name__ == "__main__":
    main()
