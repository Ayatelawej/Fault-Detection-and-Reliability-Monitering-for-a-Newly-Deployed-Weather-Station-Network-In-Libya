from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_PUBLIC_DIR = Path(r"C:\Users\m\Desktop\Mozn Data")
DEFAULT_END_DATE = "2026-06-05"
DEFAULT_FREQ = "5min"

DROP_COLUMNS = {
    "api_request_date",
    "registry_city",
    "registry_country",
    "registry_elevation_m",
    "registry_install_date",
    "registry_latitude",
    "registry_longitude",
    "registry_station_id",
    "registry_station_name",
    "stationID",
    "requested_date",
}

COLUMN_RENAMES = {
    "stationID": "station_id",
    "tz": "timezone",
    "obsTimeUtc": "obs_time_utc",
    "obsTimeLocal": "obs_time_local",
    "lat": "latitude",
    "lon": "longitude",
    "solarRadiationHigh": "solar_radiation_high_wm2",
    "uvHigh": "uv_high",
    "winddirAvg": "winddir_avg_deg",
    "humidityHigh": "humidity_high_pct",
    "humidityLow": "humidity_low_pct",
    "humidityAvg": "humidity_avg_pct",
    "qcStatus": "qc_status",
    "metric.tempHigh": "temp_high_c",
    "metric.tempLow": "temp_low_c",
    "metric.tempAvg": "temp_avg_c",
    "metric.windspeedHigh": "windspeed_high_kmh",
    "metric.windspeedLow": "windspeed_low_kmh",
    "metric.windspeedAvg": "windspeed_avg_kmh",
    "metric.windgustHigh": "windgust_high_kmh",
    "metric.windgustLow": "windgust_low_kmh",
    "metric.windgustAvg": "windgust_avg_kmh",
    "metric.dewptHigh": "dewpoint_high_c",
    "metric.dewptLow": "dewpoint_low_c",
    "metric.dewptAvg": "dewpoint_avg_c",
    "metric.windchillHigh": "windchill_high_c",
    "metric.windchillLow": "windchill_low_c",
    "metric.windchillAvg": "windchill_avg_c",
    "metric.heatindexHigh": "heatindex_high_c",
    "metric.heatindexLow": "heatindex_low_c",
    "metric.heatindexAvg": "heatindex_avg_c",
    "metric.pressureMax": "pressure_max_hpa",
    "metric.pressureMin": "pressure_min_hpa",
    "metric.pressureTrend": "pressure_trend_hpa",
    "metric.precipRate": "precip_rate_mmh",
    "metric.precipTotal": "precip_total_mm",
    "requested_date": "api_request_date",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 5-minute gap-filled Weather Underground PWS dataset "
            "from exact observation CSVs without calling the API."
        ),
    )
    parser.add_argument(
        "--public-dir",
        default=str(DEFAULT_PUBLIC_DIR),
        help="Directory containing the public pooled CSV and station registry.",
    )
    parser.add_argument(
        "--per-station-dir",
        default=None,
        help="Directory containing per-station exact observation CSVs.",
    )
    parser.add_argument(
        "--registry-csv",
        default=None,
        help="Public station registry CSV.",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="Inclusive UTC end date for the expected grid, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--freq",
        default=DEFAULT_FREQ,
        help="Expected grid frequency. Default: 5min.",
    )
    parser.add_argument(
        "--pooled-output",
        default=None,
        help="Output path for the pooled gap-filled CSV.",
    )
    parser.add_argument(
        "--station-output-dir",
        default=None,
        help="Output directory for per-station gap-filled CSVs.",
    )
    return parser.parse_args()


def output_defaults(args: argparse.Namespace) -> dict[str, Path]:
    public_dir = Path(args.public_dir)
    return {
        "public_dir": public_dir,
        "per_station_dir": Path(args.per_station_dir)
        if args.per_station_dir
        else public_dir / "per_station_observations",
        "registry_csv": Path(args.registry_csv)
        if args.registry_csv
        else public_dir / "station_registry.csv",
        "pooled_output": Path(args.pooled_output)
        if args.pooled_output
        else public_dir / "observations_complete.csv",
        "station_output_dir": Path(args.station_output_dir)
        if args.station_output_dir
        else public_dir / "per_station_complete",
    }


def find_station_file(per_station_dir: Path, station_id: str) -> Path:
    candidates = [
        per_station_dir / f"{station_id}_observations.csv",
        per_station_dir / f"{station_id}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No observation CSV found for {station_id}")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns=COLUMN_RENAMES)


def iso_utc(series: pd.Series) -> pd.Series:
    return series.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def local_day_boundary_utc(
    value: str,
    timezone: str,
    *,
    end_of_day: bool,
    freq: str,
) -> pd.Timestamp:
    timestamp = pd.to_datetime(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid date: {value}")
    timestamp = pd.Timestamp(timestamp).tz_localize(timezone).tz_convert("UTC")
    if end_of_day:
        timestamp = timestamp + pd.Timedelta(days=1) - pd.Timedelta(freq)
    return timestamp


def build_station_grid(
    obs_path: Path,
    registry_row: pd.Series,
    *,
    end_date: str,
    freq: str,
) -> pd.DataFrame:
    station_id = str(registry_row["station_id"])
    observations = normalize_columns(pd.read_csv(obs_path, low_memory=False))
    observations["obs_time_utc_dt"] = pd.to_datetime(
        observations["obs_time_utc"],
        utc=True,
        errors="coerce",
    )
    observations = observations.dropna(subset=["obs_time_utc_dt"]).copy()
    if observations.empty:
        raise ValueError(f"No valid obs_time_utc values found for {station_id}")

    observations["expected_time_utc_dt"] = observations["obs_time_utc_dt"].dt.floor(
        freq
    )
    counts = (
        observations.groupby("expected_time_utc_dt")
        .size()
        .rename("n_observations_in_bin")
    )
    observations = observations.sort_values(
        "obs_time_utc_dt",
        kind="mergesort",
    ).drop_duplicates("expected_time_utc_dt", keep="last")
    observations = observations.merge(
        counts,
        left_on="expected_time_utc_dt",
        right_index=True,
        how="left",
    )

    timezone = (
        observations["timezone"].dropna().astype(str).iloc[0]
        if observations["timezone"].notna().any()
        else "Africa/Tripoli"
    )
    first_observed_bin = observations["expected_time_utc_dt"].min()
    last_observed_bin = observations["expected_time_utc_dt"].max()
    start_time = min(
        local_day_boundary_utc(
            str(registry_row["start_date"]),
            timezone,
            end_of_day=False,
            freq=freq,
        ),
        first_observed_bin,
    )
    end_time = max(
        local_day_boundary_utc(
            end_date,
            timezone,
            end_of_day=True,
            freq=freq,
        ),
        last_observed_bin,
    )
    grid = pd.DataFrame(
        {
            "expected_time_utc_dt": pd.date_range(
                start=start_time,
                end=end_time,
                freq=freq,
                tz="UTC",
            )
        }
    )
    grid = grid.merge(observations, on="expected_time_utc_dt", how="left")
    grid["data_present"] = grid["obs_time_utc"].notna().astype("int8")
    grid["n_observations_in_bin"] = (
        pd.to_numeric(grid["n_observations_in_bin"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    grid["expected_time_utc"] = iso_utc(grid["expected_time_utc_dt"])
    grid["date_utc"] = grid["expected_time_utc_dt"].dt.strftime("%Y-%m-%d")

    grid["station_id"] = station_id
    grid["elevation_m"] = registry_row["elevation_m"]

    grid["timezone"] = grid["timezone"].fillna("Africa/Tripoli")
    grid["latitude"] = pd.to_numeric(grid["latitude"], errors="coerce").fillna(
        pd.to_numeric(pd.Series([registry_row["latitude"]]), errors="coerce").iloc[0]
    )
    grid["longitude"] = pd.to_numeric(grid["longitude"], errors="coerce").fillna(
        pd.to_numeric(pd.Series([registry_row["longitude"]]), errors="coerce").iloc[0]
    )

    grid = grid.drop(columns=["obs_time_utc_dt", "expected_time_utc_dt"], errors="ignore")
    grid = grid.drop(columns=[column for column in DROP_COLUMNS if column in grid])

    leading_columns = [
        "station_id",
        "latitude",
        "longitude",
        "elevation_m",
        "expected_time_utc",
        "date_utc",
        "data_present",
        "n_observations_in_bin",
    ]
    remaining_columns = [
        column
        for column in grid.columns
        if column not in leading_columns
    ]
    return grid[leading_columns + remaining_columns]


def write_notes(
    public_dir: Path,
    *,
    pooled_output: Path,
    station_output_dir: Path,
    manifest: dict[str, Any],
) -> None:
    notes_path = public_dir / "complete_notes.md"
    notes = (
        "# 5-Minute Gap-Filled Dataset\n\n"
        "This file is derived from the exact Weather Underground observation "
        "exports. It does not call the API and does not invent measurement "
        "values.\n\n"
        "- `expected_time_utc` is the regular 5-minute UTC time grid.\n"
        "- The grid covers each station's local `start_date` through the "
        "requested local end date, converted to UTC. For Libya "
        "(Africa/Tripoli), local midnight appears as the previous UTC "
        "evening.\n"
        "- `data_present` is `1` when a WU observation was present in that "
        "5-minute bin, and `0` when no observation was present.\n"
        "- Rows with `data_present = 0` intentionally keep measurement cells "
        "blank so outages and gaps are visible in the table.\n"
        "- `obs_time_utc` is the actual WU observation timestamp. It can differ "
        "from `expected_time_utc` because WU reports may arrive with seconds "
        "and small timing offsets.\n"
        "- If more than one observation falls in the same 5-minute bin, the "
        "latest observation is kept and `n_observations_in_bin` records the "
        "count.\n"
        "- The exact-observation CSV remains the source table for analyses "
        "that need original timestamps without binning.\n\n"
        f"Pooled file: `{pooled_output.name}`\n\n"
        f"Per-station folder: `{station_output_dir.name}`\n"
    )
    notes_path.write_text(notes, encoding="utf-8")

    manifest_path = public_dir / "complete_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    args = parse_args()
    paths = output_defaults(args)

    registry = pd.read_csv(paths["registry_csv"])
    required_columns = [
        "station_id",
        "station_name",
        "city",
        "country",
        "latitude",
        "longitude",
        "elevation_m",
        "start_date",
    ]
    missing = [column for column in required_columns if column not in registry]
    if missing:
        raise SystemExit(f"Registry is missing columns: {', '.join(missing)}")

    paths["station_output_dir"].mkdir(parents=True, exist_ok=True)
    if paths["pooled_output"].exists():
        paths["pooled_output"].unlink()

    pooled_header_written = False
    station_summaries: list[dict[str, Any]] = []

    for _, row in registry.sort_values("station_id").iterrows():
        station_id = str(row["station_id"])
        obs_path = find_station_file(paths["per_station_dir"], station_id)
        station_grid = build_station_grid(
            obs_path,
            row,
            end_date=args.end_date,
            freq=args.freq,
        )

        station_output = paths["station_output_dir"] / f"{station_id}_complete.csv"
        station_grid.to_csv(station_output, index=False, encoding="utf-8-sig")
        station_grid.to_csv(
            paths["pooled_output"],
            index=False,
            mode="a",
            header=not pooled_header_written,
            encoding="utf-8-sig",
        )
        pooled_header_written = True

        station_summaries.append(
            {
                "station_id": station_id,
                "rows": int(len(station_grid)),
                "present_rows": int(station_grid["data_present"].sum()),
                "missing_rows": int((station_grid["data_present"] == 0).sum()),
                "grid_start_utc": str(station_grid["expected_time_utc"].iloc[0]),
                "grid_end_utc": str(station_grid["expected_time_utc"].iloc[-1]),
                "station_csv": str(station_output),
            }
        )
        print(
            f"{station_id}: rows={len(station_grid):,} "
            f"present={int(station_grid['data_present'].sum()):,} "
            f"missing={int((station_grid['data_present'] == 0).sum()):,}"
        )

    manifest = {
        "frequency": args.freq,
        "end_date": args.end_date,
        "pooled_csv": str(paths["pooled_output"]),
        "station_output_dir": str(paths["station_output_dir"]),
        "station_count": len(station_summaries),
        "total_rows": sum(item["rows"] for item in station_summaries),
        "total_present_rows": sum(item["present_rows"] for item in station_summaries),
        "total_missing_rows": sum(item["missing_rows"] for item in station_summaries),
        "stations": station_summaries,
    }
    write_notes(
        paths["public_dir"],
        pooled_output=paths["pooled_output"],
        station_output_dir=paths["station_output_dir"],
        manifest=manifest,
    )
    print("")
    print(f"Pooled output: {paths['pooled_output']}")
    print(f"Total rows: {manifest['total_rows']:,}")
    print(f"Total present rows: {manifest['total_present_rows']:,}")
    print(f"Total missing rows: {manifest['total_missing_rows']:,}")


if __name__ == "__main__":
    main()
