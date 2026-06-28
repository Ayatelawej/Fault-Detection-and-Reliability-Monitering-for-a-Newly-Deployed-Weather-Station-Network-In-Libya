from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "review_id",
    "cluster_label",
    "cluster_size",
    "role",
    "needs_5min_confirmation",
    "station_id",
    "start_hour",
    "end_hour",
    "duration_hours",
    "affected_sensor_groups",
    "n_sensor_groups",
    "dominant_detector",
    "detector_concordance",
    "max_abs_zscore",
    "max_iforest_score",
    "min_rolling_variance",
    "reasons",
    "cluster_probability",
    "label",
]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLUSTERS_PATH = PROJECT_ROOT / "data" / "processed" / "fault_clusters.parquet"
REVIEW_QUEUE_PATH = PROJECT_ROOT / "data" / "processed" / "review_queue.csv"


def _reason_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()

    return {token for token in str(value).split("|") if token}


def _needs_5min_confirmation(row: pd.Series) -> bool:
    return bool(
        row["dominant_detector"] == "stuck"
        or "stuck_variance_zero" in _reason_tokens(row["reasons"])
    )


def _with_review_fields(
    frame: pd.DataFrame,
    role: str,
    cluster_size: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["cluster_size"] = int(cluster_size)
    result["role"] = role
    result["needs_5min_confirmation"] = result.apply(
        _needs_5min_confirmation,
        axis=1,
    )
    result["label"] = ""
    return result


def _cluster_review_rows(
    cluster: pd.DataFrame,
    reps_per_cluster: int,
    boundary_per_cluster: int,
) -> list[pd.DataFrame]:
    sorted_cluster = cluster.sort_values(
        ["cluster_probability", "start_hour"],
        ascending=[False, True],
    )
    cluster_size = len(sorted_cluster)
    representatives = sorted_cluster.head(min(reps_per_cluster, cluster_size))
    remaining = sorted_cluster.drop(index=representatives.index)
    boundaries = remaining.tail(min(boundary_per_cluster, len(remaining)))

    rows = [
        _with_review_fields(
            representatives,
            "representative",
            cluster_size,
        )
    ]

    if not boundaries.empty:
        rows.append(
            _with_review_fields(
                boundaries,
                "boundary",
                cluster_size,
            )
        )

    return rows


def build_review_queue(
    clusters_df: pd.DataFrame,
    reps_per_cluster: int = 5,
    boundary_per_cluster: int = 2,
    noise_sample: int = 20,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    non_noise = clusters_df.loc[clusters_df["cluster_label"].ge(0)]

    for _, cluster in non_noise.groupby("cluster_label", sort=True):
        rows.extend(
            _cluster_review_rows(
                cluster,
                reps_per_cluster,
                boundary_per_cluster,
            )
        )

    noise = clusters_df.loc[clusters_df["cluster_label"].eq(-1)].sort_values(
        ["max_abs_zscore", "start_hour"],
        ascending=[False, True],
    )
    if not noise.empty:
        rows.append(
            _with_review_fields(
                noise.head(min(noise_sample, len(noise))),
                "noise_check",
                len(noise),
            )
        )

    if rows:
        result = pd.concat(rows, ignore_index=True)
    else:
        result = pd.DataFrame(columns=clusters_df.columns)
        result["cluster_size"] = pd.Series(dtype="int64")
        result["role"] = pd.Series(dtype="object")
        result["needs_5min_confirmation"] = pd.Series(dtype="bool")
        result["label"] = pd.Series(dtype="object")

    role_order = {"representative": 0, "boundary": 1, "noise_check": 2}
    result["_role_order"] = result["role"].map(role_order)
    result["_is_noise"] = result["cluster_label"].eq(-1)
    result = result.sort_values(
        [
            "_is_noise",
            "cluster_label",
            "_role_order",
            "cluster_probability",
            "max_abs_zscore",
            "start_hour",
        ],
        ascending=[True, True, True, False, False, True],
    ).reset_index(drop=True)
    result["review_id"] = range(1, len(result) + 1)

    return result.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)


def _representatives_for_station_clusters(
    clusters: pd.DataFrame,
    queue: pd.DataFrame,
    station_id: str,
) -> pd.DataFrame:
    cluster_labels = sorted(
        label
        for label in clusters.loc[
            clusters["station_id"].eq(station_id)
            & clusters["cluster_label"].ge(0),
            "cluster_label",
        ].unique()
    )
    representatives = queue.loc[
        queue["cluster_label"].isin(cluster_labels)
        & queue["role"].eq("representative")
    ]
    return representatives.loc[
        :,
        [
            "station_id",
            "duration_hours",
            "affected_sensor_groups",
            "dominant_detector",
            "cluster_probability",
        ],
    ]


def _print_representatives(
    clusters: pd.DataFrame,
    queue: pd.DataFrame,
    station_id: str,
) -> None:
    representatives = _representatives_for_station_clusters(
        clusters,
        queue,
        station_id,
    )
    print()
    print(f"{station_id} REPRESENTATIVES")
    if representatives.empty:
        print("none")
    else:
        print(representatives.to_string(index=False))


def main() -> None:
    clusters = pd.read_parquet(CLUSTERS_PATH)
    queue = build_review_queue(clusters)

    REVIEW_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(REVIEW_QUEUE_PATH, index=False)

    role_counts = queue["role"].value_counts()
    needs_5min = queue.loc[queue["needs_5min_confirmation"]]
    represented_clusters = int(
        queue.loc[queue["cluster_label"].ge(0), "cluster_label"].nunique()
    )
    needs_5min_clusters = int(needs_5min["cluster_label"].nunique())

    print("REVIEW QUEUE SUMMARY")
    print(f"Input path: {CLUSTERS_PATH}")
    print(f"Output path: {REVIEW_QUEUE_PATH}")
    print(f"Total review rows: {len(queue):,}")
    print("Counts by role:")
    for role in ["representative", "boundary", "noise_check"]:
        print(f"{role}: {int(role_counts.get(role, 0))}")
    print(f"Distinct non-noise clusters represented: {represented_clusters}")
    print(f"needs_5min_confirmation rows: {len(needs_5min):,}")
    print(f"needs_5min_confirmation clusters: {needs_5min_clusters}")

    _print_representatives(clusters, queue, "ITRIPO33")
    _print_representatives(clusters, queue, "IMURQU7")


if __name__ == "__main__":
    main()
