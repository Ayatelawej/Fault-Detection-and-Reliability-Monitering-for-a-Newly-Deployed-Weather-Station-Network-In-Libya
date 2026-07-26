from __future__ import annotations

import pandas as pd

from src.config.paths import MERGED_DATASET_PATH, STATION_REGISTRY_PATH
from src.features.row_state import (
    ROW_STATE_COMPLETE,
    ROW_STATE_PARTIAL,
    ROW_STATE_TERMINAL_PADDED,
    ROW_STATE_WARMUP,
    classify_row_states,
)

PRESENT_ROW_STATES = [
    ROW_STATE_WARMUP,
    ROW_STATE_PARTIAL,
    ROW_STATE_COMPLETE,
]

ENRICHMENT_COLUMNS = [
    "present_rate",
    "terminal_padded_rows",
    "status_class",
]


def assign_status_class(
    present_rate: float,
    terminal_padded_rows: int,
) -> str:
    if present_rate < 0.55 or terminal_padded_rows >= 168:
        return "outage_dominated"
    if present_rate < 0.70:
        return "weak"
    if present_rate < 0.85:
        return "usable"
    if terminal_padded_rows > 0:
        return "reliable_but_terminal_gap"
    return "reliable_reference_candidate"


def build_enriched_station_registry(
    merged: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    classified = classify_row_states(merged, registry)

    row_state_counts = pd.crosstab(
        classified["station_id"],
        classified["row_state"],
    ).reindex(registry["station_id"], fill_value=0)

    for state in [*PRESENT_ROW_STATES, ROW_STATE_TERMINAL_PADDED]:
        if state not in row_state_counts.columns:
            row_state_counts[state] = 0

    total_rows = row_state_counts.sum(axis=1)
    present_rows = row_state_counts[PRESENT_ROW_STATES].sum(axis=1)
    present_rate = (
        present_rows
        .div(total_rows.where(total_rows.gt(0)))
        .fillna(0.0)
    )

    metrics = pd.DataFrame(
        {
            "station_id": row_state_counts.index,
            "present_rate": present_rate.to_numpy(),
            "terminal_padded_rows": (
                row_state_counts[ROW_STATE_TERMINAL_PADDED]
                .astype("int64")
                .to_numpy()
            ),
        }
    )
    metrics["status_class"] = [
        assign_status_class(row.present_rate, row.terminal_padded_rows)
        for row in metrics.itertuples(index=False)
    ]

    base_registry = registry.drop(
        columns=[
            column for column in ENRICHMENT_COLUMNS
            if column in registry.columns
        ],
    )
    return base_registry.merge(metrics, on="station_id", how="left")


def main() -> None:
    registry = pd.read_csv(STATION_REGISTRY_PATH)
    merged = pd.read_csv(MERGED_DATASET_PATH)

    enriched = build_enriched_station_registry(merged, registry)
    enriched.to_csv(STATION_REGISTRY_PATH, index=False)

    sorted_registry = enriched.sort_values(
        ["present_rate", "station_id"],
        ascending=[True, True],
    )
    status_counts = enriched["status_class"].value_counts().sort_index()

    print("Enriched station registry sorted by present_rate ascending:")
    print(sorted_registry.to_string(index=False))
    print()
    print("status_class counts:")
    print(status_counts.to_string())


if __name__ == "__main__":
    main()
