from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.paths import FIVE_MINUTE_INPUT_DIR, MEASUREMENT_COLUMNS, MERGED_DATASET_PATH, STATION_REGISTRY_PATH
from src.rules.clustering import cluster_episodes
from src.rules.config import (
    EXTERNAL_CACHE_DIR,
    EXTERNAL_FEATURES_PATH,
    EXTERNAL_RESIDUALS_PATH,
    HDBSCAN_MIN_CLUSTER_SIZE,
    SPATIAL_FEATURES_PATH,
    SPATIAL_NEIGHBORS_PATH,
    SPATIAL_RESIDUALS_PATH,
    STATISTICAL_FEATURES_PATH,
)
from src.rules.episodes import build_episodes
from src.rules.events import build_events
from src.rules.external_features import build_external_features
from src.rules.external_residuals import build_all as build_external_residuals
from src.rules.feature_matrix import build_feature_matrix, materialize_statistical_features, write_feature_matrix
from src.rules.review_queue import build_review_queue
from src.rules.score import compute_anomaly_scores
from src.rules.spatial_offsets import build_spatial_features, neighbor_present_counts
from src.rules.spatial_residuals import (
    build_all as build_spatial_residuals,
    build_neighbor_graph,
    write_neighbor_graph,
    write_spatial_residuals,
)
from src.workflows.prerequisites import require_files


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIVE_MIN_DIR = FIVE_MINUTE_INPUT_DIR
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / EXTERNAL_CACHE_DIR
DEFAULT_OUTPUTS = {
    "statistical_scores": PROJECT_ROOT / "data" / "processed" / "statistical_anomaly_scores.parquet",
    "fault_events": PROJECT_ROOT / "data" / "processed" / "fault_events.parquet",
    "fault_episodes": PROJECT_ROOT / "data" / "processed" / "fault_episodes.parquet",
    "fault_clusters": PROJECT_ROOT / "data" / "processed" / "fault_clusters.parquet",
    "review_queue": PROJECT_ROOT / "data" / "processed" / "review_queue.csv",
    "statistical_features": PROJECT_ROOT / STATISTICAL_FEATURES_PATH,
    "external_residuals": PROJECT_ROOT / EXTERNAL_RESIDUALS_PATH,
    "external_features": PROJECT_ROOT / EXTERNAL_FEATURES_PATH,
    "spatial_neighbors": PROJECT_ROOT / SPATIAL_NEIGHBORS_PATH,
    "spatial_residuals": PROJECT_ROOT / SPATIAL_RESIDUALS_PATH,
    "spatial_features": PROJECT_ROOT / SPATIAL_FEATURES_PATH,
    "feature_matrix": PROJECT_ROOT / "data" / "features" / "feature_matrix.parquet",
}


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".parquet":
        frame.to_parquet(destination, index=False)
    else:
        frame.to_csv(destination, index=False)
    return destination


def _safe_clusters(episodes: pd.DataFrame) -> pd.DataFrame:
    if len(episodes) < HDBSCAN_MIN_CLUSTER_SIZE:
        result = episodes.copy()
        result["cluster_label"] = -1
        result["cluster_probability"] = 0.0
        return result
    return cluster_episodes(episodes)


def _measurement_channels(merged: pd.DataFrame) -> list[str]:
    return [
        column
        for column in MEASUREMENT_COLUMNS
        if column in merged.columns and pd.api.types.is_numeric_dtype(merged[column])
    ]


def _require_rebuild_inputs(
    merged_path: Path,
    registry_path: Path,
    reference_dir: Path,
    five_min_dir: Path,
) -> None:
    require_files(
        "Detection feature rebuild",
        {
            "canonical merged dataset": merged_path,
            "station registry": registry_path,
        },
    )
    reference_paths = sorted(Path(reference_dir).glob("*.parquet"))
    if not reference_paths:
        raise FileNotFoundError(
            "Detection feature rebuild requires public reference parquet files in "
            f"{reference_dir}. Run scripts/fetch_reference_data.py first."
        )
    require_files(
        "Detection feature rebuild",
        {
            f"complete five-minute input for {path.stem}": Path(five_min_dir) / f"{path.stem}_complete.csv"
            for path in reference_paths
        },
        "Supply one <station_id>_complete.csv file for every station with reference data.",
    )


def rebuild_detection_features(
    *,
    merged_path: Path = MERGED_DATASET_PATH,
    registry_path: Path = STATION_REGISTRY_PATH,
    reference_dir: Path = DEFAULT_REFERENCE_DIR,
    five_min_dir: Path = DEFAULT_FIVE_MIN_DIR,
    outputs: dict[str, Path] | None = None,
) -> dict[str, int]:
    _require_rebuild_inputs(
        Path(merged_path),
        Path(registry_path),
        Path(reference_dir),
        Path(five_min_dir),
    )
    paths = DEFAULT_OUTPUTS if outputs is None else {**DEFAULT_OUTPUTS, **outputs}
    merged = pd.read_csv(merged_path, low_memory=False)
    merged["station_id"] = merged["station_id"].astype(str)
    merged["hour_utc"] = pd.to_datetime(merged["hour_utc"], utc=True)
    if merged.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("merged data contains duplicate station-hour keys")
    registry = pd.read_csv(registry_path)
    registry["station_id"] = registry["station_id"].astype(str)

    scores = compute_anomaly_scores(merged, _measurement_channels(merged))
    events = build_events(scores)
    episodes = build_episodes(events)
    clusters = _safe_clusters(episodes)
    review_queue = build_review_queue(clusters)
    _write_frame(scores, paths["statistical_scores"])
    _write_frame(events, paths["fault_events"])
    _write_frame(episodes, paths["fault_episodes"])
    _write_frame(clusters, paths["fault_clusters"])
    _write_frame(review_queue, paths["review_queue"])
    statistical = materialize_statistical_features(
        scores=scores,
        merged=merged.loc[:, ["station_id", "hour_utc"]],
        output_path=paths["statistical_features"],
    )

    external_residuals = build_external_residuals(
        merged_path=Path(merged_path),
        reference_dir=Path(reference_dir),
        five_min_dir=Path(five_min_dir),
    )
    if external_residuals.empty:
        raise ValueError("external residual rebuild produced no rows")
    external_features = build_external_features(external_residuals)
    _write_frame(external_residuals, paths["external_residuals"])
    _write_frame(external_features, paths["external_features"])

    neighbors = build_neighbor_graph(registry)
    time_index = pd.date_range(merged["hour_utc"].min(), merged["hour_utc"].max(), freq="h", tz="UTC")
    spatial_residuals = build_spatial_residuals(
        merged=merged,
        registry=registry,
        graph=neighbors,
        time_index=time_index,
    )
    neighbor_counts = neighbor_present_counts(spatial_residuals, merged, neighbors)
    spatial_features = build_spatial_features(spatial_residuals, neighbor_counts)
    write_neighbor_graph(neighbors, paths["spatial_neighbors"])
    write_spatial_residuals(spatial_residuals, paths["spatial_residuals"])
    _write_frame(spatial_features, paths["spatial_features"])

    feature_matrix = build_feature_matrix(
        external_features=external_features,
        statistical_features=statistical,
        spatial_features=spatial_features,
        registry=registry,
        neighbors=neighbors,
    )
    write_feature_matrix(feature_matrix, paths["feature_matrix"])
    return {
        "merged_rows": int(len(merged)),
        "statistical_scores": int(len(scores)),
        "fault_episodes": int(len(episodes)),
        "external_residuals": int(len(external_residuals)),
        "spatial_residuals": int(len(spatial_residuals)),
        "feature_matrix": int(len(feature_matrix)),
    }
