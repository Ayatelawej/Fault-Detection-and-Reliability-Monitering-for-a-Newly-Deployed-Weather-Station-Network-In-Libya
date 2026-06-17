from __future__ import annotations

# Lifted from Mozn AI on 2026-05-11.
# Original source path: C:\Users\m\Desktop\Mozn AI\mozn-postprocessing-system\src\postprocessing\ingestion\openmeteo.py
# Canonical version is maintained in Mozn; this is a one-time EC499 reference copy.

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.paths import EXTERNAL_DIR

ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT_SECONDS = 60
POLITE_DELAY_SECONDS = 4
OPENMETEO_DIR = EXTERNAL_DIR / "openmeteo"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "surface_pressure",
    "pressure_msl",
    "rain",
    "shortwave_radiation",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "precipitation",
    "weather_code",
    "is_day",
    "sunshine_duration",
    "direct_radiation",
    "diffuse_radiation",
    "et0_fao_evapotranspiration",
    "snowfall",
]

DAILY_VARIABLES = [
    "sunrise",
    "sunset",
    "daylight_duration",
    "sunshine_duration",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
]

HOURLY_RENAME = {
    "temperature_2m": "base_temperature_c",
    "relative_humidity_2m": "base_relative_humidity_pct",
    "dew_point_2m": "base_dew_point_c",
    "wind_speed_10m": "base_wind_speed_kmh",
    "wind_gusts_10m": "base_wind_gust_kmh",
    "wind_direction_10m": "base_wind_direction_deg",
    "surface_pressure": "base_surface_pressure_hpa",
    "pressure_msl": "base_msl_pressure_hpa",
    "rain": "base_rain_total_mm",
    "shortwave_radiation": "base_solar_radiation_wm2",
    "cloud_cover": "base_cloud_cover_pct",
    "cloud_cover_low": "base_cloud_cover_low_pct",
    "cloud_cover_mid": "base_cloud_cover_mid_pct",
    "cloud_cover_high": "base_cloud_cover_high_pct",
    "precipitation": "base_precipitation_mm",
    "weather_code": "base_weather_code",
    "is_day": "base_is_day",
    "sunshine_duration": "base_sunshine_seconds",
    "direct_radiation": "base_direct_radiation_wm2",
    "diffuse_radiation": "base_diffuse_radiation_wm2",
    "et0_fao_evapotranspiration": "base_evapotranspiration_mm",
    "snowfall": "base_snowfall_mm",
}

DAILY_RENAME = {
    "sunrise": "base_sunrise_utc",
    "sunset": "base_sunset_utc",
    "daylight_duration": "base_daylight_seconds",
    "sunshine_duration": "base_sunshine_daily_seconds",
    "temperature_2m_max": "base_temperature_max_c",
    "temperature_2m_min": "base_temperature_min_c",
    "precipitation_sum": "base_precipitation_daily_mm",
}


class OpenMeteoFetchError(Exception):
    pass


def _hourly_path(station_id: str) -> Path:
    return OPENMETEO_DIR / f"{station_id}_hourly.parquet"


def _daily_path(station_id: str) -> Path:
    return OPENMETEO_DIR / f"{station_id}_daily.parquet"


def _format_date(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=16),
    retry=retry_if_exception_type((requests.RequestException, OpenMeteoFetchError)),
    reraise=True,
)
def _api_call(params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(ARCHIVE_BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise OpenMeteoFetchError(
            f"HTTP {response.status_code} from Open-Meteo: {response.text[:300]}"
        )
    return response.json()


# TODO: lift when Stage 4 begins — depends on Station model
# def _build_hourly_dataframe(
#     station: Station, payload: dict[str, Any]
# ) -> pd.DataFrame:
#     hourly = payload.get("hourly")
#     if not hourly or "time" not in hourly:
#         return pd.DataFrame()
#
#     df = pd.DataFrame(hourly)
#     df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
#     df = df.rename(columns={"time": "valid_time_utc"})
#
#     present_renames = {k: v for k, v in HOURLY_RENAME.items() if k in df.columns}
#     df = df.rename(columns=present_renames)
#
#     df.insert(0, "station_id", station.station_id)
#     return df
#
#
# TODO: lift when Stage 4 begins — depends on Station model
# def _build_daily_dataframe(
#     station: Station, payload: dict[str, Any]
# ) -> pd.DataFrame:
#     daily = payload.get("daily")
#     if not daily or "time" not in daily:
#         return pd.DataFrame()
#
#     df = pd.DataFrame(daily)
#     df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
#     df = df.rename(columns={"time": "date_utc"})
#
#     for col in ("sunrise", "sunset"):
#         if col in df.columns:
#             df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
#
#     present_renames = {k: v for k, v in DAILY_RENAME.items() if k in df.columns}
#     df = df.rename(columns=present_renames)
#
#     df.insert(0, "station_id", station.station_id)
#     return df


def _has_complete_cache(
    hourly_path: Path,
    daily_path: Path,
    start: datetime,
    end: datetime,
) -> bool:
    if not hourly_path.is_file() or not daily_path.is_file():
        return False

    try:
        hourly = pd.read_parquet(hourly_path, columns=["valid_time_utc"])
        daily = pd.read_parquet(daily_path, columns=["date_utc"])
    except Exception:
        return False

    if hourly.empty or daily.empty:
        return False

    hourly_start = hourly["valid_time_utc"].min()
    hourly_end = hourly["valid_time_utc"].max()
    return hourly_start <= start and hourly_end >= end


# TODO: lift when Stage 4 begins — depends on Station model
# def fetch_station_baseline(
#     station: Station,
#     start: datetime | None = None,
#     end: datetime | None = None,
#     force_refresh: bool = False,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     start = start or station.fetch_start
#     end = end or station.fetch_end
#
#     hourly_path = _hourly_path(station.station_id)
#     daily_path = _daily_path(station.station_id)
#
#     if not force_refresh and _has_complete_cache(hourly_path, daily_path, start, end):
#         return (
#             pd.read_parquet(hourly_path),
#             pd.read_parquet(daily_path),
#         )
#
#     params = {
#         "latitude": station.latitude,
#         "longitude": station.longitude,
#         "elevation": station.elevation_m,
#         "start_date": _format_date(start),
#         "end_date": _format_date(end),
#         "hourly": ",".join(HOURLY_VARIABLES),
#         "daily": ",".join(DAILY_VARIABLES),
#         "timezone": "UTC",
#         "wind_speed_unit": "kmh",
#     }
#
#     payload = _api_call(params)
#
#     hourly_df = _build_hourly_dataframe(station, payload)
#     daily_df = _build_daily_dataframe(station, payload)
#
#     hourly_path.parent.mkdir(parents=True, exist_ok=True)
#     hourly_df.to_parquet(hourly_path, engine="pyarrow", index=False)
#     daily_df.to_parquet(daily_path, engine="pyarrow", index=False)
#
#     return hourly_df, daily_df


def polite_sleep() -> None:
    time.sleep(POLITE_DELAY_SECONDS)
