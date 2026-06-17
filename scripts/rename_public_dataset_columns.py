from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

DEFAULT_PUBLIC_DIR = Path(r"C:\Users\m\Desktop\Mozn Data")

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
        description="Rename public Mozn dataset CSV columns to unit-suffixed names.",
    )
    parser.add_argument(
        "--public-dir",
        default=str(DEFAULT_PUBLIC_DIR),
        help="Public dataset directory to update.",
    )
    return parser.parse_args()


def target_csvs(public_dir: Path) -> list[Path]:
    paths: list[Path] = []
    top_level_names = [
        "observations_pooled.csv",
        "observations_complete.csv",
    ]
    for name in top_level_names:
        path = public_dir / name
        if path.exists():
            paths.append(path)

    for folder_name in [
        "per_station_observations",
        "per_station_complete",
    ]:
        folder = public_dir / folder_name
        if folder.exists():
            paths.extend(sorted(folder.glob("*.csv")))

    return paths


def renamed_header(header: list[str]) -> list[str]:
    return [COLUMN_RENAMES.get(column, column) for column in header]


def rename_csv_header(path: Path) -> bool:
    temp_path = path.with_name(path.name + ".tmp")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        first_line = source.readline()
        if not first_line:
            return False
        header = next(csv.reader([first_line]))

        new_header = renamed_header(header)
        if new_header == header:
            return False

        with temp_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(new_header)
            shutil.copyfileobj(source, target, length=1024 * 1024)

    os.replace(temp_path, path)
    return True


def main() -> None:
    public_dir = Path(parse_args().public_dir)
    paths = target_csvs(public_dir)
    updated = 0
    for path in paths:
        changed = rename_csv_header(path)
        updated += int(changed)
        status = "updated" if changed else "already clean"
        print(f"{status}: {path}")
    print(f"CSV files checked: {len(paths)}")
    print(f"CSV files updated: {updated}")


if __name__ == "__main__":
    main()
