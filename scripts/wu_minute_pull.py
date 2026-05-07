from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_ENDPOINT = "https://api.weather.com/v2/pws/history/all"
SCRIPT_PATH = "scripts/wu_minute_pull.py"

NUMERIC_COLUMNS = [
    "epoch",
    "latitude",
    "longitude",
    "qc_status",
    "humidity_avg_pct",
    "temp_avg_c",
    "windspeed_avg_kmh",
    "windgust_avg_kmh",
    "winddir_avg_deg",
    "pressure_avg_hpa",
    "precip_rate_mmh",
    "precip_total_mm",
    "solar_radiation_high_wm2",
    "uv_high",
]

OBSERVATION_COLUMNS = [
    "station_id",
    "obs_time_utc",
    "obs_time_local",
    "epoch",
    "latitude",
    "longitude",
    "neighborhood",
    "software_type",
    "qc_status",
    "humidity_avg_pct",
    "temp_avg_c",
    "windspeed_avg_kmh",
    "windgust_avg_kmh",
    "winddir_avg_deg",
    "pressure_avg_hpa",
    "precip_rate_mmh",
    "precip_total_mm",
    "solar_radiation_high_wm2",
    "uv_high",
    "city",
    "elevation_m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull minute-level WU PWS history for a detected network outage "
            "window."
        ),
    )
    parser.add_argument("--window-id", default="NW_20260308T23")
    parser.add_argument(
        "--windows-csv",
        default="data/processed/network_outage_windows.csv",
    )
    parser.add_argument(
        "--registry-csv",
        default="data/merged/station_registry.csv",
    )
    parser.add_argument("--padding-hours", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default="data/external/wu_minute_pulls",
    )
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def _require_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    frame_name: str,
) -> None:
    missing_columns = [
        column for column in required_columns
        if column not in frame.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _numeric_value(value: Any) -> Any:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def _iso_utc(timestamp: pd.Timestamp) -> str | None:
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return _iso_utc(value)
    if pd.isna(value):
        return None
    return value


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://www.wunderground.com",
            "Referer": "https://www.wunderground.com/",
        }
    )

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def load_window(
    windows_csv: Path,
    window_id: str,
) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp, list[str]]:
    windows = pd.read_csv(windows_csv)
    _require_columns(
        windows,
        ["window_id", "window_start_utc", "window_end_utc", "station_ids"],
        "windows_csv",
    )
    matches = windows.loc[windows["window_id"].eq(window_id)]
    if matches.empty:
        raise SystemExit(f"Window id not found in {windows_csv}: {window_id}")

    window = matches.iloc[0]
    window_start = pd.to_datetime(
        window["window_start_utc"],
        utc=True,
        errors="coerce",
    )
    window_end = pd.to_datetime(
        window["window_end_utc"],
        utc=True,
        errors="coerce",
    )
    if pd.isna(window_start) or pd.isna(window_end):
        raise SystemExit(f"Window has invalid UTC bounds: {window_id}")

    stations = [
        station_id.strip()
        for station_id in str(window["station_ids"]).split(";")
        if station_id.strip()
    ]
    if not stations:
        raise SystemExit(f"Window has no station_ids: {window_id}")

    return window, window_start, window_end, stations


def utc_dates_between(
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> list[datetime.date]:
    start_date = analysis_start.date()
    end_date = analysis_end.date()
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current = current + timedelta(days=1)
    return dates


def build_station_lookup(registry_csv: Path) -> dict[str, dict[str, Any]]:
    registry = pd.read_csv(registry_csv)
    _require_columns(registry, ["station_id"], "registry_csv")

    lookup: dict[str, dict[str, Any]] = {}
    for row in registry.itertuples(index=False):
        station_id = str(getattr(row, "station_id"))
        lookup[station_id] = {
            "city": getattr(row, "city", pd.NA),
            "latitude": getattr(row, "latitude", pd.NA),
            "longitude": getattr(row, "longitude", pd.NA),
            "elevation_m": getattr(row, "elevation", pd.NA),
            "station_name": getattr(row, "station_name", pd.NA),
        }
    return lookup


def fetch_history_day(
    session: requests.Session,
    *,
    station_id: str,
    day_yyyymmdd: str,
    api_key: str,
    cache_path: Path,
    timeout: int,
) -> tuple[dict[str, Any], bool]:
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle), True

    params = {
        "stationId": station_id,
        "format": "json",
        "units": "m",
        "date": day_yyyymmdd,
        "apiKey": api_key,
    }
    response = session.get(API_ENDPOINT, params=params, timeout=timeout)
    if response.status_code == 204:
        print(f"No data for {station_id} on {day_yyyymmdd} (204).")
        data = {"observations": []}
    elif response.status_code == 200:
        data = response.json()
    else:
        response.raise_for_status()
        data = {"observations": []}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return data, False


def flatten_observation(obs: dict[str, Any], station_id: str) -> dict[str, Any]:
    metric = obs.get("metric", {}) or {}
    pressure_avg = metric.get("pressureAvg")
    if pressure_avg is None:
        pressure_avg = metric.get("pressureMax")

    return {
        "station_id": obs.get("stationID") or station_id,
        "obs_time_utc": obs.get("obsTimeUtc"),
        "obs_time_local": obs.get("obsTimeLocal"),
        "epoch": _numeric_value(obs.get("epoch")),
        "latitude": _numeric_value(obs.get("lat")),
        "longitude": _numeric_value(obs.get("lon")),
        "neighborhood": obs.get("neighborhood"),
        "software_type": obs.get("softwareType"),
        "qc_status": _numeric_value(
            obs.get("qcStatus", obs.get("qualityControlStatus"))
        ),
        "humidity_avg_pct": _numeric_value(obs.get("humidityAvg")),
        "temp_avg_c": _numeric_value(metric.get("tempAvg")),
        "windspeed_avg_kmh": _numeric_value(metric.get("windspeedAvg")),
        "windgust_avg_kmh": _numeric_value(metric.get("windgustAvg")),
        "winddir_avg_deg": _numeric_value(obs.get("winddirAvg")),
        "pressure_avg_hpa": _numeric_value(pressure_avg),
        "precip_rate_mmh": _numeric_value(metric.get("precipRate")),
        "precip_total_mm": _numeric_value(metric.get("precipTotal")),
        "solar_radiation_high_wm2": _numeric_value(
            obs.get("solarRadiationHigh")
        ),
        "uv_high": _numeric_value(obs.get("uvHigh")),
    }


def build_observations_dataframe(
    rows: list[dict[str, Any]],
    *,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    station_lookup: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    observations = pd.DataFrame(rows)
    if observations.empty:
        observations = pd.DataFrame(columns=OBSERVATION_COLUMNS)
        observations["obs_time_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
        return observations

    observations["obs_time_utc"] = pd.to_datetime(
        observations["obs_time_utc"],
        utc=True,
        errors="coerce",
    )
    for column in NUMERIC_COLUMNS:
        if column in observations.columns:
            observations[column] = pd.to_numeric(
                observations[column],
                errors="coerce",
            )

    observations = observations.loc[
        observations["obs_time_utc"].ge(analysis_start)
        & observations["obs_time_utc"].le(analysis_end)
    ].copy()
    observations["city"] = observations["station_id"].map(
        lambda station_id: station_lookup.get(str(station_id), {}).get("city")
    )
    observations["elevation_m"] = observations["station_id"].map(
        lambda station_id: station_lookup.get(str(station_id), {}).get(
            "elevation_m"
        )
    )
    observations["elevation_m"] = pd.to_numeric(
        observations["elevation_m"],
        errors="coerce",
    )

    observations = observations.sort_values(
        ["obs_time_utc", "station_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return observations[OBSERVATION_COLUMNS]


def build_presence_matrix(
    observations: pd.DataFrame,
    *,
    stations: list[str],
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> pd.DataFrame:
    minute_index = pd.date_range(
        analysis_start.floor("min"),
        analysis_end.ceil("min"),
        freq="1min",
        tz="UTC",
    )
    grid = pd.MultiIndex.from_product(
        [minute_index, stations],
        names=["minute_utc", "station_id"],
    ).to_frame(index=False)
    grid["present"] = 0

    if observations.empty:
        return grid

    present = observations[["station_id", "obs_time_utc"]].copy()
    present["minute_utc"] = pd.to_datetime(
        present["obs_time_utc"],
        utc=True,
        errors="coerce",
    ).dt.floor("min")
    present = present.dropna(subset=["minute_utc"])
    present["present"] = 1
    present = present[["minute_utc", "station_id", "present"]].drop_duplicates()

    matrix = grid.merge(
        present,
        on=["minute_utc", "station_id"],
        how="left",
        suffixes=("", "_observed"),
    )
    matrix["present"] = (
        pd.to_numeric(matrix["present_observed"], errors="coerce")
        .fillna(matrix["present"])
        .astype("int64")
    )
    return matrix[["minute_utc", "station_id", "present"]]


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def write_manifest(
    output_path: Path,
    *,
    window_id: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    padding_hours: int,
    stations: list[str],
    dates_fetched: list[datetime.date],
    n_api_calls_attempted: int,
    n_api_calls_from_cache: int,
    n_api_calls_hit_network: int,
    n_observations_total: int,
    n_observations_in_window: int,
) -> None:
    manifest = {
        "pull_date_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "Phase 2 investigation: root cause analysis of network-wide "
            "outage window"
        ),
        "window_id": window_id,
        "window_start_utc": _iso_utc(window_start),
        "window_end_utc": _iso_utc(window_end),
        "analysis_start_utc": _iso_utc(analysis_start),
        "analysis_end_utc": _iso_utc(analysis_end),
        "padding_hours": int(padding_hours),
        "stations": stations,
        "n_stations": len(stations),
        "dates_fetched": [day.isoformat() for day in dates_fetched],
        "n_api_calls_attempted": int(n_api_calls_attempted),
        "n_api_calls_from_cache": int(n_api_calls_from_cache),
        "n_api_calls_hit_network": int(n_api_calls_hit_network),
        "n_observations_total": int(n_observations_total),
        "n_observations_in_window": int(n_observations_in_window),
        "api_endpoint": API_ENDPOINT,
        "script": SCRIPT_PATH,
        "git_commit": git_commit(),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=_json_default)


def write_readme(
    output_path: Path,
    *,
    window_id: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    stations: list[str],
    dates_fetched: list[datetime.date],
) -> None:
    dates = ", ".join(day.isoformat() for day in dates_fetched)
    text = (
        "# WU Minute-Level Pull\n\n"
        "This directory contains Weather Underground PWS history/all "
        "observations pulled for Phase 2 root cause analysis of a detected "
        "network-wide outage window.\n\n"
        f"- window_id: {window_id}\n"
        f"- window_start_utc: {_iso_utc(window_start)}\n"
        f"- window_end_utc: {_iso_utc(window_end)}\n"
        f"- analysis_start_utc: {_iso_utc(analysis_start)}\n"
        f"- analysis_end_utc: {_iso_utc(analysis_end)}\n"
        f"- n_stations: {len(stations)}\n"
        f"- dates: {dates}\n\n"
        "WU_API_KEY was loaded from .env; the key itself is not recorded here.\n\n"
        "Warning: this data was pulled outside the original frozen dataset "
        "(June 2025 - March 2026) and lives in data/external/, not "
        "data/merged/ or data/processed/.\n"
    )
    output_path.write_text(text, encoding="utf-8")


def print_summary(
    *,
    window_id: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    stations: list[str],
    dates_fetched: list[datetime.date],
    n_api_calls_hit_network: int,
    observations: pd.DataFrame,
) -> None:
    print(f"Window id: {window_id}")
    print(f"Window UTC: {_iso_utc(window_start)} to {_iso_utc(window_end)}")
    print(
        "Analysis UTC: "
        f"{_iso_utc(analysis_start)} to {_iso_utc(analysis_end)}"
    )
    print(f"n_stations: {len(stations)}")
    print(f"n_dates_fetched: {len(dates_fetched)}")
    print(f"n_api_calls_hit_network: {n_api_calls_hit_network}")
    print(f"n_observations_in_window: {len(observations)}")

    print("Per-station observation count in analysis window:")
    if observations.empty:
        print("(none)")
    else:
        counts = (
            observations.groupby("station_id")
            .size()
            .rename("observation_count")
            .reset_index()
            .sort_values(["station_id"], kind="mergesort")
        )
        print(counts.to_string(index=False))

    print("Boundary observations by station:")
    boundary_rows = []
    for station_id in stations:
        station_obs = observations.loc[observations["station_id"].eq(station_id)]
        before = station_obs.loc[
            station_obs["obs_time_utc"].lt(window_start),
            "obs_time_utc",
        ].max()
        after = station_obs.loc[
            station_obs["obs_time_utc"].gt(window_end),
            "obs_time_utc",
        ].min()
        boundary_rows.append(
            {
                "station_id": station_id,
                "last_before_window_start_utc": _iso_utc(before),
                "first_after_window_end_utc": _iso_utc(after),
            }
        )
    print(pd.DataFrame(boundary_rows).to_string(index=False))


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.environ.get("WU_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("WU_API_KEY is not set. Add WU_API_KEY to .env.")

    windows_csv = Path(args.windows_csv)
    registry_csv = Path(args.registry_csv)
    output_root = Path(args.output_dir)

    _, window_start, window_end, stations = load_window(
        windows_csv,
        args.window_id,
    )
    padding = timedelta(hours=int(args.padding_hours))
    analysis_start = window_start - padding
    analysis_end = window_end + padding
    dates_fetched = utc_dates_between(analysis_start, analysis_end)

    station_lookup = build_station_lookup(registry_csv)
    output_dir = output_root / args.window_id
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = create_session()
    rows: list[dict[str, Any]] = []
    n_api_calls_from_cache = 0
    n_api_calls_hit_network = 0

    for station_id in stations:
        for day in dates_fetched:
            day_yyyymmdd = day.strftime("%Y%m%d")
            cache_path = raw_dir / f"{station_id}_{day_yyyymmdd}.json"
            data, from_cache = fetch_history_day(
                session,
                station_id=station_id,
                day_yyyymmdd=day_yyyymmdd,
                api_key=api_key,
                cache_path=cache_path,
                timeout=int(args.timeout),
            )
            if from_cache:
                n_api_calls_from_cache += 1
            else:
                n_api_calls_hit_network += 1
                time.sleep(float(args.sleep))

            observations = data.get("observations", []) or []
            for obs in observations:
                if isinstance(obs, dict):
                    rows.append(flatten_observation(obs, station_id))

    n_api_calls_attempted = len(stations) * len(dates_fetched)
    n_observations_total = len(rows)
    observations_df = build_observations_dataframe(
        rows,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        station_lookup=station_lookup,
    )
    presence_df = build_presence_matrix(
        observations_df,
        stations=stations,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
    )

    observations_df.to_csv(output_dir / "observations.csv", index=False)
    presence_df.to_csv(output_dir / "presence_minutely.csv", index=False)
    write_manifest(
        output_dir / "manifest.json",
        window_id=args.window_id,
        window_start=window_start,
        window_end=window_end,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        padding_hours=int(args.padding_hours),
        stations=stations,
        dates_fetched=dates_fetched,
        n_api_calls_attempted=n_api_calls_attempted,
        n_api_calls_from_cache=n_api_calls_from_cache,
        n_api_calls_hit_network=n_api_calls_hit_network,
        n_observations_total=n_observations_total,
        n_observations_in_window=len(observations_df),
    )
    write_readme(
        output_dir / "README.md",
        window_id=args.window_id,
        window_start=window_start,
        window_end=window_end,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        stations=stations,
        dates_fetched=dates_fetched,
    )
    print_summary(
        window_id=args.window_id,
        window_start=window_start,
        window_end=window_end,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        stations=stations,
        dates_fetched=dates_fetched,
        n_api_calls_hit_network=n_api_calls_hit_network,
        observations=observations_df,
    )


if __name__ == "__main__":
    main()
