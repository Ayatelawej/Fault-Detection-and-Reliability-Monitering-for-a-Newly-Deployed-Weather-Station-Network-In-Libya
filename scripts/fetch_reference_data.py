from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import STATION_REGISTRY_PATH
from src.rules.config import (
    EXTERNAL_API_URL,
    EXTERNAL_CACHE_DIR,
    EXTERNAL_CELL_SELECTION,
    EXTERNAL_END_DATE,
    EXTERNAL_EXCLUDED_STATION_IDS,
    EXTERNAL_HOURLY_VARS,
    EXTERNAL_MODELS,
    EXTERNAL_START_DATE,
)
from src.workflows.prerequisites import require_files

REFERENCE_DIR = PROJECT_ROOT / EXTERNAL_CACHE_DIR
MANIFEST_PATH = REFERENCE_DIR.parent / "reference_manifest.csv"
REQUEST_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 3
INITIAL_BACKOFF_SECONDS = 2
STATION_SLEEP_SECONDS = 1


class ExternalReferenceError(Exception):
    pass


def build_request_params(station_row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    row = station_row if isinstance(station_row, dict) else station_row.to_dict()
    return {
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "elevation": float(row["elevation"]),
        "start_date": EXTERNAL_START_DATE,
        "end_date": EXTERNAL_END_DATE,
        "hourly": EXTERNAL_HOURLY_VARS,
        "models": EXTERNAL_MODELS,
        "cell_selection": EXTERNAL_CELL_SELECTION,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }


def parse_response(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, object]]:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not hourly:
        raise ExternalReferenceError("response missing hourly data")
    if "time" not in hourly:
        raise ExternalReferenceError("response missing hourly time array")

    times = hourly["time"]
    if not isinstance(times, list) or not times:
        raise ExternalReferenceError("response has empty hourly time array")

    frame = pd.DataFrame({"time_utc": pd.to_datetime(times, utc=True)})
    for variable in EXTERNAL_HOURLY_VARS:
        values = hourly.get(variable)
        if not isinstance(values, list) or len(values) != len(times):
            raise ExternalReferenceError(f"response missing complete {variable}")
        if all(value is None for value in values):
            raise ExternalReferenceError(f"response has null {variable}")
        frame[variable] = values

    frame = frame.set_index("time_utc")
    frame.index.name = "time_utc"
    metadata = {
        "response_latitude": payload.get("latitude"),
        "response_longitude": payload.get("longitude"),
        "model_elevation": payload.get("elevation"),
    }
    return frame, metadata


def expected_reference_index() -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(EXTERNAL_START_DATE, tz="UTC"),
        end=pd.Timestamp(EXTERNAL_END_DATE, tz="UTC") + pd.Timedelta(hours=23),
        freq="h",
        name="time_utc",
    )


def _reference_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    source = frame["time_utc"] if "time_utc" in frame.columns else frame.index
    return pd.DatetimeIndex(pd.to_datetime(source, utc=True, errors="coerce"), name="time_utc")


def validate_reference_index(frame: pd.DataFrame) -> None:
    expected = expected_reference_index()
    if len(frame) != len(expected):
        raise ExternalReferenceError(
            f"expected {len(expected)} rows, got {len(frame)}"
        )
    actual = _reference_index(frame)
    if actual.isna().any() or not actual.equals(expected):
        raise ExternalReferenceError(
            "reference timestamps do not match the configured hourly range "
            f"{expected[0].isoformat()} through {expected[-1].isoformat()}"
        )


def station_cache_path(station_id: str) -> Path:
    return REFERENCE_DIR / f"{station_id}.parquet"


def _hourly_param(params: dict[str, Any]) -> dict[str, Any]:
    request_params = dict(params)
    request_params["hourly"] = ",".join(EXTERNAL_HOURLY_VARS)
    return request_params


def fetch_payload(params: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                EXTERNAL_API_URL,
                params=_hourly_param(params),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429 or response.status_code >= 500:
                raise ExternalReferenceError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
            raise ExternalReferenceError(
                f"HTTP {response.status_code}: {response.text[:300]}"
            )
        except (requests.RequestException, ExternalReferenceError) as exc:
            last_error = exc
            if attempt == MAX_ATTEMPTS:
                break
            time.sleep(INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    raise ExternalReferenceError(str(last_error))


def _manifest_row(
    station_row: pd.Series,
    metadata: dict[str, object],
    rows: int,
    fetched_at_utc: str,
) -> dict[str, object]:
    return {
        "station_id": station_row["station_id"],
        "registry_latitude": station_row["latitude"],
        "registry_longitude": station_row["longitude"],
        "registry_elevation": station_row["elevation"],
        "response_latitude": metadata.get("response_latitude"),
        "response_longitude": metadata.get("response_longitude"),
        "model_elevation": metadata.get("model_elevation"),
        "rows": rows,
        "fetched_at_utc": fetched_at_utc,
    }


def fetch_station(station_row: pd.Series, force: bool) -> dict[str, object]:
    station_id = str(station_row["station_id"])
    path = station_cache_path(station_id)
    fetched_at_utc = datetime.now(timezone.utc).isoformat()

    if path.exists() and not force:
        frame = pd.read_parquet(path)
        validate_reference_index(frame)
        metadata = {
            "response_latitude": pd.NA,
            "response_longitude": pd.NA,
            "model_elevation": pd.NA,
        }
        row = _manifest_row(station_row, metadata, len(frame), "")
        row["status"] = "skipped"
        return row

    payload = fetch_payload(build_request_params(station_row))
    frame, metadata = parse_response(payload)
    validate_reference_index(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    row = _manifest_row(station_row, metadata, len(frame), fetched_at_utc)
    row["status"] = "fetched"
    time.sleep(STATION_SLEEP_SECONDS)
    return row


def solar_average_columns(merged_header: pd.Index) -> list[str]:
    return [
        column
        for column in merged_header
        if "solar" in column.lower()
        and "avg" in column.lower()
        and column != "solar_radiation_high_wm2"
    ]


def _print_summary(manifest: pd.DataFrame) -> None:
    summary = manifest.copy()
    summary["model_minus_registry_elevation"] = (
        pd.to_numeric(summary["model_elevation"], errors="coerce")
        - pd.to_numeric(summary["registry_elevation"], errors="coerce")
    )
    columns = [
        "station_id",
        "rows",
        "model_minus_registry_elevation",
        "status",
    ]
    print("REFERENCE FETCH SUMMARY")
    print(summary.loc[:, columns].to_string(index=False))
    total = pd.DataFrame(
        [
            {
                "station_id": "TOTAL",
                "rows": int(summary["rows"].sum()),
                "model_minus_registry_elevation": pd.NA,
                "status": "|".join(summary["status"].value_counts().sort_index().index),
            }
        ]
    )
    print(total.loc[:, columns].to_string(index=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=STATION_REGISTRY_PATH)
    parser.add_argument("--merged", type=Path, default=PROJECT_ROOT / "data" / "merged" / "station_hourly_merged.csv")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def active_reference_registry(registry: pd.DataFrame) -> pd.DataFrame:
    frame = registry.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    return (
        frame.loc[~frame["station_id"].isin(EXTERNAL_EXCLUDED_STATION_IDS)]
        .sort_values("station_id")
        .reset_index(drop=True)
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    require_files(
        "Public reference fetch",
        {
            "station registry": args.registry,
            "canonical merged dataset": args.merged,
        },
    )
    registered = pd.read_csv(args.registry).sort_values("station_id").reset_index(drop=True)
    registry = active_reference_registry(registered)
    excluded = sorted(set(registered["station_id"].astype(str)).difference(registry["station_id"]))
    if registry.empty:
        raise SystemExit("no active stations are available for reference fetching")
    merged_header = pd.read_csv(args.merged, nrows=0).columns
    solar_avg = solar_average_columns(merged_header)

    print("SOLAR AVERAGE COLUMN CHECK")
    print("solar_avg_columns=" + ("|".join(solar_avg) if solar_avg else "none"))
    if excluded:
        print("excluded_reference_stations=" + "|".join(excluded))

    rows: list[dict[str, object]] = []
    for _, station_row in registry.iterrows():
        rows.append(fetch_station(station_row, args.force))

    manifest = pd.DataFrame(rows)
    expected_rows = len(expected_reference_index())
    if manifest["rows"].ne(expected_rows).any():
        bad = manifest.loc[manifest["rows"].ne(expected_rows)]
        raise SystemExit(f"bad row counts:\n{bad.to_string(index=False)}")

    if manifest["status"].eq("fetched").any():
        output = manifest.drop(columns=["status"])
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(MANIFEST_PATH, index=False)

    _print_summary(manifest)
    if MANIFEST_PATH.exists():
        print(f"manifest={MANIFEST_PATH}")


if __name__ == "__main__":
    main()
