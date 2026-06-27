ROBUST_ZSCORE_FLAG_PERCENTILE = 99.7
ROLLING_VARIANCE_WINDOW_HOURS = 24
ROLLING_VARIANCE_FLAG_THRESHOLD = 1e-6
ISOLATION_FOREST_CONTAMINATION = 0.003
COVERAGE_FLOOR_HOURS = 1500
HDBSCAN_MIN_CLUSTER_SIZE = 30
HDBSCAN_MIN_SAMPLES = 10

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
