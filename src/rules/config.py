"""Stage 3 statistical anomaly detection configuration.

This module centralizes detector thresholds, baseline coverage requirements,
and channel-specific handling rules for the Stage 3 rules layer. The 0.3%
flag target ties ROBUST_ZSCORE_FLAG_PERCENTILE and
ISOLATION_FOREST_CONTAMINATION together: both derive from the Stage 5 manual
review budget and should move together if that review budget changes.
"""

ROBUST_ZSCORE_FLAG_PERCENTILE = 99.7
ROLLING_VARIANCE_WINDOW_HOURS = 6
ROLLING_VARIANCE_FLAG_THRESHOLD = 1e-6
ISOLATION_FOREST_CONTAMINATION = 0.003
COVERAGE_FLOOR_HOURS = 1500

CHANNEL_BASELINE_WINDOWS = {
    "pressure_max_hpa": 30 * 24,
    "pressure_min_hpa": 30 * 24,
    "precip_total_mm": 14 * 24,
    "precip_rate_mmh": 14 * 24,
}

CHANNELS_REQUIRING_CIRCULAR_TRANSFORM = ["winddir_avg_deg"]
CHANNELS_REQUIRING_LOG_TRANSFORM = ["precip_total_mm", "precip_rate_mmh"]
CHANNELS_EXCLUDED_FROM_STATISTICAL_LAYER = []
