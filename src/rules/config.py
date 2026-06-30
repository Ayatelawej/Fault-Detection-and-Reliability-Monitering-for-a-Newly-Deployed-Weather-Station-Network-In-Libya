ROBUST_ZSCORE_FLAG_PERCENTILE = 99.7
ROLLING_VARIANCE_WINDOW_HOURS = 24
ROLLING_VARIANCE_FLAG_THRESHOLD = 1e-6
ISOLATION_FOREST_CONTAMINATION = 0.003
COVERAGE_FLOOR_HOURS = 1500
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES = 10
REVIEW_SUSTAINED_NOISE_HOURS = 24
PHYSICAL_LIMIT_RULES = {
    "temp_avg_c": {"min": -60.0, "max": 60.0, "kind": "temperature"},
    "temp_high_c": {"min": -60.0, "max": 60.0, "kind": "temperature"},
    "temp_low_c": {"min": -60.0, "max": 60.0, "kind": "temperature"},
    "humidity_avg_pct": {"min": 0.0, "max": 100.0, "kind": "humidity"},
    "humidity_high_pct": {"min": 0.0, "max": 100.0, "kind": "humidity"},
    "humidity_low_pct": {"min": 0.0, "max": 100.0, "kind": "humidity"},
    "windspeed_avg_kmh": {"min": 0.0, "max": 250.0, "kind": "wind"},
    "windspeed_high_kmh": {"min": 0.0, "max": 250.0, "kind": "wind"},
    "windspeed_low_kmh": {"min": 0.0, "max": 250.0, "kind": "wind"},
    "windgust_avg_kmh": {"min": 0.0, "max": 300.0, "kind": "wind"},
    "windgust_high_kmh": {"min": 0.0, "max": 300.0, "kind": "wind"},
    "windgust_low_kmh": {"min": 0.0, "max": 300.0, "kind": "wind"},
    "winddir_avg_deg": {"min": 0.0, "max": 360.0, "kind": "wind_direction"},
    "pressure_max_hpa": {"min": 870.0, "max": 1085.0, "kind": "pressure"},
    "pressure_min_hpa": {"min": 870.0, "max": 1085.0, "kind": "pressure"},
    "pressure_trend_hpa": {"max_abs": 20.0, "kind": "pressure_trend"},
    "precip_rate_mmh": {"min": 0.0, "max": 1000.0, "kind": "rain_rate"},
    "precip_total_mm": {"min": 0.0, "max": 1000.0, "kind": "rain_total"},
    "solar_radiation_high_wm2": {"min": 0.0, "max": 1400.0, "kind": "solar"},
    "uv_high": {"min": 0.0, "max": 25.0, "kind": "uv"},
}
PHYSICAL_SUSPECT_RULES = {
    "solar_radiation_high_wm2": {"max": 1100.0, "kind": "solar"},
    "uv_high": {"max": 16.0, "kind": "uv"},
    "windspeed_avg_kmh": {"max": 150.0, "kind": "wind"},
    "windspeed_high_kmh": {"max": 150.0, "kind": "wind"},
    "windspeed_low_kmh": {"max": 150.0, "kind": "wind"},
    "windgust_avg_kmh": {"max": 180.0, "kind": "wind"},
    "windgust_high_kmh": {"max": 180.0, "kind": "wind"},
    "windgust_low_kmh": {"max": 180.0, "kind": "wind"},
    "precip_rate_mmh": {"max": 300.0, "kind": "rain_rate"},
    "precip_total_mm": {"max": 300.0, "kind": "rain_total"},
    "pressure_trend_hpa": {"max_abs": 20.0, "kind": "pressure_trend"},
}

CHANNEL_BASELINE_WINDOWS = {
    "pressure_max_hpa": 30 * 24,
    "pressure_min_hpa": 30 * 24,
    "precip_total_mm": 14 * 24,
    "precip_rate_mmh": 14 * 24,
}

CHANNELS_REQUIRING_CIRCULAR_TRANSFORM = ["winddir_avg_deg"]
CHANNELS_REQUIRING_LOG_TRANSFORM = ["precip_total_mm", "precip_rate_mmh"]
CHANNELS_EXCLUDED_FROM_STATISTICAL_LAYER = [
    "dewpoint_avg_c",
    "dewpoint_high_c",
    "dewpoint_low_c",
    "windchill_avg_c",
    "windchill_high_c",
    "windchill_low_c",
    "heatindex_avg_c",
    "heatindex_high_c",
    "heatindex_low_c",
]
STUCK_IGNORE_ZERO_CHANNELS = [
    "precip_rate_mmh",
    "solar_radiation_high_wm2",
    "uv_high",
    "windspeed_low_kmh",
    "windgust_low_kmh",
]
STUCK_SKIP_CHANNELS = ["precip_total_mm"]
SENSOR_GROUP_PREFIXES = {
    "temp": "thermo_hygrometer",
    "humidity": "thermo_hygrometer",
    "windspeed": "anemometer",
    "windgust": "anemometer",
    "winddir": "wind_vane",
    "precip": "rain_gauge",
    "solar": "light_uv",
    "uv": "light_uv",
    "pressure": "barometer",
}
