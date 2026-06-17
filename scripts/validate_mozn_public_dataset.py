from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_PUBLIC_DIR = Path(r"C:\Users\m\Desktop\Mozn Data")
EXACT_POOLED_NAME = "observations_pooled.csv"
GAP_POOLED_NAME = "observations_complete.csv"
REGISTRY_NAME = "station_registry.csv"
EXACT_STATION_DIR_NAME = "per_station_observations"
GAP_STATION_DIR_NAME = "per_station_complete"

REGISTRY_COLUMNS = [
    "station_id",
    "station_name",
    "city",
    "country",
    "latitude",
    "longitude",
    "elevation_m",
    "start_date",
]

EXACT_COLUMNS = [
    "station_id",
    "latitude",
    "longitude",
    "elevation_m",
    "timezone",
    "obs_time_utc",
    "obs_time_local",
    "epoch",
    "solar_radiation_high_wm2",
    "uv_high",
    "winddir_avg_deg",
    "humidity_high_pct",
    "humidity_low_pct",
    "humidity_avg_pct",
    "qc_status",
    "temp_high_c",
    "temp_low_c",
    "temp_avg_c",
    "windspeed_high_kmh",
    "windspeed_low_kmh",
    "windspeed_avg_kmh",
    "windgust_high_kmh",
    "windgust_low_kmh",
    "windgust_avg_kmh",
    "dewpoint_high_c",
    "dewpoint_low_c",
    "dewpoint_avg_c",
    "windchill_high_c",
    "windchill_low_c",
    "windchill_avg_c",
    "heatindex_high_c",
    "heatindex_low_c",
    "heatindex_avg_c",
    "pressure_max_hpa",
    "pressure_min_hpa",
    "pressure_trend_hpa",
    "precip_rate_mmh",
    "precip_total_mm",
]

GAP_COLUMNS = [
    "station_id",
    "latitude",
    "longitude",
    "elevation_m",
    "expected_time_utc",
    "date_utc",
    "data_present",
    "n_observations_in_bin",
    *EXACT_COLUMNS[4:],
]

MEASUREMENT_AND_OBS_COLUMNS = [
    "obs_time_utc",
    "obs_time_local",
    "epoch",
    "solar_radiation_high_wm2",
    "uv_high",
    "winddir_avg_deg",
    "humidity_high_pct",
    "humidity_low_pct",
    "humidity_avg_pct",
    "qc_status",
    "temp_high_c",
    "temp_low_c",
    "temp_avg_c",
    "windspeed_high_kmh",
    "windspeed_low_kmh",
    "windspeed_avg_kmh",
    "windgust_high_kmh",
    "windgust_low_kmh",
    "windgust_avg_kmh",
    "dewpoint_high_c",
    "dewpoint_low_c",
    "dewpoint_avg_c",
    "windchill_high_c",
    "windchill_low_c",
    "windchill_avg_c",
    "heatindex_high_c",
    "heatindex_low_c",
    "heatindex_avg_c",
    "pressure_max_hpa",
    "pressure_min_hpa",
    "pressure_trend_hpa",
    "precip_rate_mmh",
    "precip_total_mm",
]

EMPTY_TEXT_VALUES = {"", "nan", "nat", "none", "null", "na", "n/a"}
FORBIDDEN_HEADER_TEXT = ["metric.", "obsTime", "qcStatus", "stationID"]
REDUNDANT_COLUMNS = {
    "registry_station_id",
    "registry_station_name",
    "registry_city",
    "registry_country",
    "registry_latitude",
    "registry_longitude",
    "registry_elevation_m",
    "registry_install_date",
    "api_request_date",
}
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the public Mozn Weather Station dataset files.",
    )
    parser.add_argument(
        "--public-dir",
        default=str(DEFAULT_PUBLIC_DIR),
        help="Public dataset directory to validate.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per chunk for large CSV validation.",
    )
    return parser.parse_args()


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader([handle.readline()]))


def empty_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        normalized = series.astype("string").str.strip().str.lower()
        mask = mask | normalized.isin(EMPTY_TEXT_VALUES)
    return mask


def update_nonempty_columns(nonempty: set[str], frame: pd.DataFrame) -> None:
    for column in frame.columns:
        if (~empty_mask(frame[column])).any():
            nonempty.add(column)


def count_csv_rows(path: Path, chunksize: int) -> int:
    total = 0
    for chunk in pd.read_csv(path, usecols=[0], chunksize=chunksize):
        total += len(chunk)
    return total


def pass_fail(condition: bool, message: str) -> dict[str, Any]:
    return {
        "status": "PASS" if condition else "FAIL",
        "message": message,
    }


def add_check(report: dict[str, Any], name: str, condition: bool, message: str) -> None:
    report["checks"][name] = pass_fail(condition, message)


def validate_registry(path: Path, report: dict[str, Any]) -> pd.DataFrame:
    registry = pd.read_csv(path)
    missing_columns = [column for column in REGISTRY_COLUMNS if column not in registry]
    add_check(
        report,
        "registry_required_columns",
        not missing_columns,
        f"Missing columns: {missing_columns}",
    )
    add_check(
        report,
        "registry_station_count",
        len(registry) == 26,
        f"Registry station count: {len(registry)}",
    )
    add_check(
        report,
        "registry_unique_station_ids",
        registry["station_id"].is_unique,
        "Station IDs are unique.",
    )

    registry_dates = pd.to_datetime(registry["start_date"], errors="coerce")
    latitudes = pd.to_numeric(registry["latitude"], errors="coerce")
    longitudes = pd.to_numeric(registry["longitude"], errors="coerce")
    elevations = pd.to_numeric(registry["elevation_m"], errors="coerce")
    add_check(
        report,
        "registry_valid_start_dates",
        registry_dates.notna().all(),
        "All registry start dates parse.",
    )
    add_check(
        report,
        "registry_valid_latitudes",
        latitudes.between(-90, 90).all(),
        "All latitudes are in [-90, 90].",
    )
    add_check(
        report,
        "registry_valid_longitudes",
        longitudes.between(-180, 180).all(),
        "All longitudes are in [-180, 180].",
    )
    add_check(
        report,
        "registry_valid_elevations",
        elevations.notna().all(),
        "All elevations parse as numeric meters.",
    )
    report["registry"] = {
        "path": str(path),
        "station_count": int(len(registry)),
        "columns": list(registry.columns),
    }
    return registry


def validate_header(
    path: Path,
    expected_columns: list[str],
    report: dict[str, Any],
    check_prefix: str,
) -> list[str]:
    header = read_header(path)
    forbidden_hits = [
        token for token in FORBIDDEN_HEADER_TEXT
        if any(token in column for column in header)
    ]
    nonsnake_columns = [
        column for column in header
        if not SNAKE_CASE.match(column)
    ]
    add_check(
        report,
        f"{check_prefix}_header_columns_match",
        header == expected_columns,
        f"{path.name} has {len(header)} columns.",
    )
    add_check(
        report,
        f"{check_prefix}_header_no_old_names",
        not forbidden_hits,
        f"Forbidden old-name fragments found: {forbidden_hits}",
    )
    add_check(
        report,
        f"{check_prefix}_header_snake_case",
        not nonsnake_columns,
        f"Non-snake-case columns: {nonsnake_columns[:10]}",
    )
    redundant_hits = [
        column for column in header
        if column in REDUNDANT_COLUMNS
    ]
    add_check(
        report,
        f"{check_prefix}_header_no_redundant_metadata",
        not redundant_hits,
        f"Redundant metadata columns found: {redundant_hits}",
    )
    return header


def count_station_metadata_mismatches(
    chunk: pd.DataFrame,
    registry_by_station: pd.DataFrame,
) -> int:
    mismatches = 0
    for column in ["latitude", "longitude", "elevation_m"]:
        actual = pd.to_numeric(chunk[column], errors="coerce")
        expected = pd.to_numeric(
            chunk["station_id"].map(registry_by_station[column]),
            errors="coerce",
        )
        mismatches += int(actual.ne(expected).sum())
    return mismatches


def validate_exact_pooled(
    path: Path,
    registry: pd.DataFrame,
    report: dict[str, Any],
    chunksize: int,
) -> dict[str, Any]:
    registry_by_station = registry.set_index("station_id")
    station_counts: Counter[str] = Counter()
    qc_counts: Counter[str] = Counter()
    nonempty_columns: set[str] = set()
    invalid_qc = 0
    missing_obs_time = 0
    station_metadata_mismatch = 0
    duplicate_obs_keys = 0
    seen_obs_keys: set[tuple[str, str]] = set()
    total_rows = 0

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        update_nonempty_columns(nonempty_columns, chunk)
        station_counts.update(chunk["station_id"].astype(str).tolist())
        station_metadata_mismatch += count_station_metadata_mismatches(
            chunk,
            registry_by_station,
        )

        qc_values = chunk["qc_status"].astype("string").fillna("<NA>")
        qc_counts.update(qc_values.tolist())
        invalid_qc += (~qc_values.isin(["-1", "0", "1"])).sum()

        obs_times = chunk["obs_time_utc"].astype("string")
        missing_obs_time += empty_mask(obs_times).sum()
        keys = zip(chunk["station_id"].astype(str), obs_times.fillna("").astype(str))
        for key in keys:
            if key in seen_obs_keys:
                duplicate_obs_keys += 1
            else:
                seen_obs_keys.add(key)

    fully_empty_columns = [
        column for column in EXACT_COLUMNS
        if column not in nonempty_columns
    ]
    summary = {
        "path": str(path),
        "total_rows": int(total_rows),
        "station_counts": dict(sorted(station_counts.items())),
        "qc_status_counts": dict(sorted(qc_counts.items())),
        "fully_empty_columns": fully_empty_columns,
        "invalid_qc_rows": int(invalid_qc),
        "missing_obs_time_rows": int(missing_obs_time),
        "station_metadata_mismatch_cells": int(station_metadata_mismatch),
        "duplicate_station_obs_time_rows": int(duplicate_obs_keys),
    }
    report["exact_pooled"] = summary
    add_check(
        report,
        "exact_total_rows_positive",
        total_rows > 0,
        f"Exact pooled rows: {total_rows:,}",
    )
    add_check(
        report,
        "exact_station_coverage",
        set(station_counts) == set(registry["station_id"].astype(str)),
        f"Stations in exact pooled: {len(station_counts)}",
    )
    add_check(
        report,
        "exact_qc_status_values",
        invalid_qc == 0,
        f"Invalid qc_status rows: {invalid_qc}",
    )
    add_check(
        report,
        "exact_obs_time_present",
        missing_obs_time == 0,
        f"Missing obs_time_utc rows: {missing_obs_time}",
    )
    add_check(
        report,
        "exact_station_metadata_matches_registry",
        station_metadata_mismatch == 0,
        f"Station metadata mismatch cells: {station_metadata_mismatch}",
    )
    add_check(
        report,
        "exact_no_duplicate_station_timestamps",
        duplicate_obs_keys == 0,
        f"Duplicate station_id + obs_time_utc rows: {duplicate_obs_keys}",
    )
    return summary


def validate_gap_pooled(
    path: Path,
    registry: pd.DataFrame,
    exact_summary: dict[str, Any],
    report: dict[str, Any],
    chunksize: int,
) -> dict[str, Any]:
    registry_by_station = registry.set_index("station_id")
    station_counts: Counter[str] = Counter()
    present_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    nonempty_columns: set[str] = set()
    data_present_values: Counter[str] = Counter()
    invalid_data_present = 0
    invalid_bin_count = 0
    missing_blank_violations: Counter[str] = Counter()
    station_metadata_mismatch = 0
    total_rows = 0
    total_present = 0
    total_missing = 0

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        update_nonempty_columns(nonempty_columns, chunk)
        station_counts.update(chunk["station_id"].astype(str).tolist())
        station_metadata_mismatch += count_station_metadata_mismatches(
            chunk,
            registry_by_station,
        )

        present_numeric = pd.to_numeric(chunk["data_present"], errors="coerce")
        bin_counts = pd.to_numeric(chunk["n_observations_in_bin"], errors="coerce")
        data_present_values.update(present_numeric.astype("string").fillna("<NA>"))
        invalid_data_present += (~present_numeric.isin([0, 1])).sum()

        invalid_bin_count += (
            present_numeric.eq(0) & bin_counts.ne(0)
        ).sum()
        invalid_bin_count += (
            present_numeric.eq(1) & bin_counts.lt(1)
        ).sum()

        missing_mask = present_numeric.eq(0)
        present_mask = present_numeric.eq(1)
        total_present += int(present_mask.sum())
        total_missing += int(missing_mask.sum())
        present_counts.update(chunk.loc[present_mask, "station_id"].astype(str))
        missing_counts.update(chunk.loc[missing_mask, "station_id"].astype(str))

        missing_rows = chunk.loc[missing_mask]
        for column in MEASUREMENT_AND_OBS_COLUMNS:
            if column in missing_rows:
                count = int((~empty_mask(missing_rows[column])).sum())
                if count:
                    missing_blank_violations[column] += count

    fully_empty_columns = [
        column for column in GAP_COLUMNS
        if column not in nonempty_columns
    ]
    summary = {
        "path": str(path),
        "total_rows": int(total_rows),
        "present_rows": int(total_present),
        "missing_rows": int(total_missing),
        "station_counts": dict(sorted(station_counts.items())),
        "present_counts": dict(sorted(present_counts.items())),
        "missing_counts": dict(sorted(missing_counts.items())),
        "data_present_values": dict(sorted(data_present_values.items())),
        "fully_empty_columns": fully_empty_columns,
        "invalid_data_present_rows": int(invalid_data_present),
        "invalid_n_observations_in_bin_rows": int(invalid_bin_count),
        "missing_blank_violations": dict(sorted(missing_blank_violations.items())),
        "station_metadata_mismatch_cells": int(station_metadata_mismatch),
    }
    report["gap_pooled"] = summary

    exact_total = int(exact_summary["total_rows"])
    exact_station_counts = exact_summary["station_counts"]
    add_check(
        report,
        "gap_total_rows_positive",
        total_rows > 0,
        f"Gap-filled pooled rows: {total_rows:,}",
    )
    add_check(
        report,
        "gap_present_rows_equal_exact_rows",
        total_present == exact_total,
        f"Gap present rows: {total_present:,}; exact rows: {exact_total:,}",
    )
    add_check(
        report,
        "gap_station_coverage",
        set(station_counts) == set(registry["station_id"].astype(str)),
        f"Stations in gap pooled: {len(station_counts)}",
    )
    add_check(
        report,
        "gap_present_counts_equal_exact_counts",
        dict(sorted(present_counts.items())) == exact_station_counts,
        "Per-station present counts match exact observation counts.",
    )
    add_check(
        report,
        "gap_has_missing_rows",
        total_missing > 0,
        f"Gap missing rows: {total_missing:,}",
    )
    add_check(
        report,
        "gap_data_present_values",
        invalid_data_present == 0,
        f"Invalid data_present rows: {invalid_data_present}",
    )
    add_check(
        report,
        "gap_n_observations_in_bin_consistent",
        invalid_bin_count == 0,
        f"Invalid bin count rows: {invalid_bin_count}",
    )
    add_check(
        report,
        "gap_missing_rows_keep_measurements_blank",
        not missing_blank_violations,
        f"Nonblank measurement values in missing rows: {dict(missing_blank_violations)}",
    )
    add_check(
        report,
        "gap_station_metadata_matches_registry",
        station_metadata_mismatch == 0,
        f"Station metadata mismatch cells: {station_metadata_mismatch}",
    )
    return summary


def validate_station_files(
    public_dir: Path,
    exact_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    report: dict[str, Any],
    chunksize: int,
) -> None:
    exact_dir = public_dir / EXACT_STATION_DIR_NAME
    gap_dir = public_dir / GAP_STATION_DIR_NAME
    exact_files = sorted(exact_dir.glob("*.csv"))
    gap_files = sorted(gap_dir.glob("*.csv"))

    exact_header = read_header(public_dir / EXACT_POOLED_NAME)
    gap_header = read_header(public_dir / GAP_POOLED_NAME)
    exact_file_counts: dict[str, int] = {}
    gap_file_counts: dict[str, int] = {}
    gap_file_present_counts: dict[str, int] = {}
    bad_headers: list[str] = []
    bad_station_values: list[str] = []
    bad_gap_frequency: list[str] = []

    for path in exact_files:
        station_id = path.name.replace("_observations.csv", "")
        if read_header(path) != exact_header:
            bad_headers.append(str(path))
        count = 0
        station_values: set[str] = set()
        for chunk in pd.read_csv(
            path,
            usecols=["station_id"],
            chunksize=chunksize,
        ):
            count += len(chunk)
            station_values.update(chunk["station_id"].astype(str).unique())
        exact_file_counts[station_id] = count
        if station_values != {station_id}:
            bad_station_values.append(str(path))

    for path in gap_files:
        station_id = path.name.replace("_complete.csv", "")
        if read_header(path) != gap_header:
            bad_headers.append(str(path))
        frame = pd.read_csv(
            path,
            usecols=[
                "station_id",
                "expected_time_utc",
                "data_present",
            ],
        )
        gap_file_counts[station_id] = len(frame)
        gap_file_present_counts[station_id] = int(
            pd.to_numeric(frame["data_present"], errors="coerce").eq(1).sum()
        )
        if set(frame["station_id"].astype(str).unique()) != {station_id}:
            bad_station_values.append(str(path))
        times = pd.to_datetime(frame["expected_time_utc"], utc=True, errors="coerce")
        if times.isna().any():
            bad_gap_frequency.append(f"{path}: invalid timestamp")
            continue
        diffs = times.sort_values().diff().dropna()
        if not diffs.eq(pd.Timedelta(minutes=5)).all():
            bad_gap_frequency.append(str(path))

    report["station_files"] = {
        "exact_file_count": len(exact_files),
        "gap_file_count": len(gap_files),
        "exact_file_counts": dict(sorted(exact_file_counts.items())),
        "gap_file_counts": dict(sorted(gap_file_counts.items())),
        "gap_file_present_counts": dict(sorted(gap_file_present_counts.items())),
        "bad_headers": bad_headers,
        "bad_station_values": bad_station_values,
        "bad_gap_frequency": bad_gap_frequency,
    }
    add_check(
        report,
        "station_file_counts",
        len(exact_files) == 26 and len(gap_files) == 26,
        f"Exact files: {len(exact_files)}; gap files: {len(gap_files)}",
    )
    add_check(
        report,
        "station_file_headers_match_pooled",
        not bad_headers,
        f"Files with bad headers: {bad_headers[:5]}",
    )
    add_check(
        report,
        "station_file_station_ids_consistent",
        not bad_station_values,
        f"Files with bad station values: {bad_station_values[:5]}",
    )
    add_check(
        report,
        "station_file_exact_counts_match_pooled",
        exact_file_counts == exact_summary["station_counts"],
        "Per-station exact file counts match pooled exact counts.",
    )
    add_check(
        report,
        "station_file_gap_counts_match_pooled",
        gap_file_counts == gap_summary["station_counts"],
        "Per-station gap file counts match pooled gap counts.",
    )
    add_check(
        report,
        "station_file_gap_present_counts_match_exact",
        gap_file_present_counts == exact_summary["station_counts"],
        "Per-station gap present counts match exact counts.",
    )
    add_check(
        report,
        "station_file_gap_frequency_regular_5min",
        not bad_gap_frequency,
        f"Files with irregular 5-minute grids: {bad_gap_frequency[:5]}",
    )


def write_reports(public_dir: Path, report: dict[str, Any]) -> None:
    json_path = public_dir / "validation_report.json"
    md_path = public_dir / "validation_report.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    failed = [
        (name, check)
        for name, check in report["checks"].items()
        if check["status"] != "PASS"
    ]
    lines = [
        "# Mozn Dataset Validation Report",
        "",
        f"Overall status: **{'PASS' if not failed else 'FAIL'}**",
        "",
        "## Summary",
        "",
        f"- Registry stations: {report['registry']['station_count']}",
        f"- Exact pooled rows: {report['exact_pooled']['total_rows']:,}",
        f"- Gap-filled pooled rows: {report['gap_pooled']['total_rows']:,}",
        f"- Gap-filled present rows: {report['gap_pooled']['present_rows']:,}",
        f"- Gap-filled missing rows: {report['gap_pooled']['missing_rows']:,}",
        f"- Per-station exact files: {report['station_files']['exact_file_count']}",
        f"- Per-station gap-filled files: {report['station_files']['gap_file_count']}",
        "",
        "## Checks",
        "",
    ]
    for name, check in report["checks"].items():
        lines.append(f"- **{check['status']}** `{name}`: {check['message']}")
    lines.extend(
        [
            "",
            "## qc_status Counts",
            "",
            "```json",
            json.dumps(report["exact_pooled"]["qc_status_counts"], indent=2),
            "```",
            "",
            "The value `-1` means no quality-control check was performed by the API.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    public_dir = Path(args.public_dir)
    report: dict[str, Any] = {
        "public_dir": str(public_dir),
        "checks": {},
    }

    registry_path = public_dir / REGISTRY_NAME
    exact_pooled_path = public_dir / EXACT_POOLED_NAME
    gap_pooled_path = public_dir / GAP_POOLED_NAME

    for name, path in {
        "registry": registry_path,
        "exact_pooled": exact_pooled_path,
        "gap_pooled": gap_pooled_path,
        "exact_station_dir": public_dir / EXACT_STATION_DIR_NAME,
        "gap_station_dir": public_dir / GAP_STATION_DIR_NAME,
    }.items():
        add_check(
            report,
            f"{name}_exists",
            path.exists(),
            str(path),
        )

    registry = validate_registry(registry_path, report)
    validate_header(exact_pooled_path, EXACT_COLUMNS, report, "exact_pooled")
    validate_header(gap_pooled_path, GAP_COLUMNS, report, "gap_pooled")
    exact_summary = validate_exact_pooled(
        exact_pooled_path,
        registry,
        report,
        args.chunksize,
    )
    gap_summary = validate_gap_pooled(
        gap_pooled_path,
        registry,
        exact_summary,
        report,
        args.chunksize,
    )
    validate_station_files(
        public_dir,
        exact_summary,
        gap_summary,
        report,
        args.chunksize,
    )
    write_reports(public_dir, report)

    failed = [
        name for name, check in report["checks"].items()
        if check["status"] != "PASS"
    ]
    print(f"Checks run: {len(report['checks'])}")
    print(f"Failed checks: {len(failed)}")
    if failed:
        print("Failed:")
        for name in failed:
            print(f"- {name}: {report['checks'][name]['message']}")
    print(f"Exact rows: {exact_summary['total_rows']:,}")
    print(f"Gap rows: {gap_summary['total_rows']:,}")
    print(f"Gap missing rows: {gap_summary['missing_rows']:,}")
    print(f"Report: {public_dir / 'validation_report.md'}")


if __name__ == "__main__":
    main()
