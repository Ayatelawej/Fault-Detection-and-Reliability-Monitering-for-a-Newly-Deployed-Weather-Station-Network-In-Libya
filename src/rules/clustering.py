from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from src.rules.config import (
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    SENSOR_GROUP_PREFIXES,
)


ID_COLUMNS = ["station_id", "start_hour", "end_hour"]
KNOWN_SENSOR_GROUPS = sorted(set(SENSOR_GROUP_PREFIXES.values()))
DETECTORS = ["iforest", "stuck", "zscore"]
FEATURE_COLUMNS = [
    "sg_anemometer",
    "sg_barometer",
    "sg_light_uv",
    "sg_rain_gauge",
    "sg_thermo_hygrometer",
    "sg_wind_vane",
    "sg_other",
    "det_iforest",
    "det_stuck",
    "det_zscore",
    "n_sensor_groups",
    "detector_concordance",
    "log_duration",
    "max_abs_zscore",
    "max_iforest_score",
    "min_rolling_variance",
]


def _sensor_group_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()

    return {token for token in str(value).split("|") if token}


def build_episode_features(
    episodes_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = episodes_df.loc[:, ID_COLUMNS].copy()
    known_groups = set(SENSOR_GROUP_PREFIXES.values())
    features = pd.DataFrame(index=episodes_df.index)

    sensor_tokens = episodes_df["affected_sensor_groups"].map(_sensor_group_tokens)

    for group in KNOWN_SENSOR_GROUPS:
        features[f"sg_{group}"] = sensor_tokens.map(
            lambda tokens, group=group: float(group in tokens)
        )

    features["sg_other"] = sensor_tokens.map(
        lambda tokens: float(bool(tokens - known_groups))
    )

    for detector in DETECTORS:
        features[f"det_{detector}"] = episodes_df["dominant_detector"].map(
            lambda value, detector=detector: float(value == detector)
        )

    features["n_sensor_groups"] = pd.to_numeric(
        episodes_df["n_sensor_groups"],
        errors="coerce",
    ).fillna(0.0)
    features["detector_concordance"] = pd.to_numeric(
        episodes_df["detector_concordance"],
        errors="coerce",
    ).fillna(0.0)
    features["log_duration"] = np.log1p(
        pd.to_numeric(
            episodes_df["duration_hours"],
            errors="coerce",
        ).fillna(0.0)
    )
    features["max_abs_zscore"] = pd.to_numeric(
        episodes_df["max_abs_zscore"],
        errors="coerce",
    ).fillna(0.0)
    features["max_iforest_score"] = pd.to_numeric(
        episodes_df["max_iforest_score"],
        errors="coerce",
    ).fillna(0.0)

    rolling_variance = pd.to_numeric(
        episodes_df["min_rolling_variance"],
        errors="coerce",
    )
    rolling_fill = rolling_variance.max(skipna=True)
    if pd.isna(rolling_fill):
        rolling_fill = 0.0
    features["min_rolling_variance"] = rolling_variance.fillna(float(rolling_fill))

    return ids, features.loc[:, FEATURE_COLUMNS].astype("float64")


def cluster_features(
    X: pd.DataFrame | np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if len(matrix) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    scaled = StandardScaler().fit_transform(matrix)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    clusterer.fit(scaled)

    labels = np.asarray(clusterer.labels_, dtype=int)
    probabilities = np.clip(
        np.asarray(clusterer.probabilities_, dtype=float),
        0.0,
        1.0,
    )
    return labels, probabilities


def cluster_episodes(
    episodes_df: pd.DataFrame,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
) -> pd.DataFrame:
    _, features = build_episode_features(episodes_df)
    labels, probabilities = cluster_features(
        features,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    result = episodes_df.copy()
    result["cluster_label"] = labels.astype(int)
    result["cluster_probability"] = probabilities.astype(float)
    return result
