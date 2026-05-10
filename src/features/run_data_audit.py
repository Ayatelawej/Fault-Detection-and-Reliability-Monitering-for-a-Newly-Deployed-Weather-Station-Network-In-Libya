from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from src.config.paths import (
    DATA_AUDIT_SUMMARY_PATH,
    FIGURES_DIR,
    HOURLY_ROW_STATES_PATH,
    MEASUREMENT_COLUMNS,
    MERGED_DATASET_PATH,
    MISSINGNESS_HEATMAP_PATH,
    PROCESSED_DIR,
    STATION_COVERAGE_FIGURE_PATH,
    STATION_REGISTRY_PATH,
    ensure_directories,
)
from src.features.row_state import (
    ROW_STATE_BEFORE_FIRST,
    ROW_STATE_COMPLETE,
    ROW_STATE_INVALID,
    ROW_STATE_NEVER_ACTIVE,
    ROW_STATE_PARTIAL,
    ROW_STATE_TERMINAL_PADDED,
    ROW_STATE_TRUE_OUTAGE,
    ROW_STATE_WARMUP,
    classify_row_states,
)

MISSINGNESS_BY_VARIABLE_PATH = PROCESSED_DIR / "missingness_by_variable.csv"

SUMMARY_COLUMNS = [
    "station_id",
    "station_name",
    "city",
    "latitude",
    "longitude",
    "elevation",
    "status_class",
    "total_rows",
    "present_rows",
    "absent_rows",
    "present_rate",
    "first_timestamp_utc",
    "last_timestamp_utc",
    "first_present_timestamp",
    "last_present_timestamp",
    "warmup_rows",
    "terminal_padded_absence_rows",
    "true_outage_candidate_rows",
    "online_partial_missing_rows",
    "online_complete_rows",
]

STATUS_CLASS_ORDER = [
    "outage_dominated",
    "weak",
    "usable",
    "reliable_but_terminal_gap",
    "reliable_reference_candidate",
]

TIMELINE_BAND_BY_ROW_STATE = {
    ROW_STATE_COMPLETE: "online",
    ROW_STATE_PARTIAL: "online",
    ROW_STATE_TRUE_OUTAGE: "outage",
    ROW_STATE_TERMINAL_PADDED: "padded",
    ROW_STATE_BEFORE_FIRST: "padded",
    ROW_STATE_NEVER_ACTIVE: "padded",
    ROW_STATE_INVALID: "padded",
    ROW_STATE_WARMUP: "warmup",
}

TIMELINE_COLORS = {
    "online": "#2ca25f",
    "outage": "#de2d26",
    "padded": "#d9d9d9",
    "warmup": "#fdd835",
}

TIMELINE_LABELS = {
    "online": "Online",
    "outage": "True outage candidate",
    "padded": "Padded / outside active window",
    "warmup": "Warmup",
}


def _format_timestamp_series(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    return timestamps.dt.strftime("%Y-%m-%d %H:%M:%S%z")


def _row_state_counts(classified: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(classified["station_id"], classified["row_state"])
    row_states = [
        ROW_STATE_WARMUP,
        ROW_STATE_TERMINAL_PADDED,
        ROW_STATE_TRUE_OUTAGE,
        ROW_STATE_PARTIAL,
        ROW_STATE_COMPLETE,
    ]
    return counts.reindex(columns=row_states, fill_value=0)


def build_audit_summary(
    classified: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    data_present = pd.to_numeric(classified["data_present"], errors="coerce")
    station_group = classified.groupby("station_id", dropna=False)

    summary = station_group.agg(
        total_rows=("station_id", "size"),
        present_rows=("data_present", lambda values: pd.to_numeric(
            values,
            errors="coerce",
        ).eq(1).sum()),
        first_timestamp_utc=("hour_utc", "min"),
        last_timestamp_utc=("hour_utc", "max"),
        first_present_timestamp=("first_present_timestamp", "first"),
        last_present_timestamp=("last_present_timestamp", "first"),
    )
    summary["absent_rows"] = summary["total_rows"] - summary["present_rows"]
    summary["present_rate"] = (
        summary["present_rows"]
        .div(summary["total_rows"].where(summary["total_rows"].gt(0)))
        .fillna(0.0)
    )

    counts = _row_state_counts(classified)
    summary = summary.join(counts, how="left").fillna(0)
    summary = summary.rename(
        columns={
            ROW_STATE_WARMUP: "warmup_rows",
            ROW_STATE_TERMINAL_PADDED: "terminal_padded_absence_rows",
            ROW_STATE_TRUE_OUTAGE: "true_outage_candidate_rows",
            ROW_STATE_PARTIAL: "online_partial_missing_rows",
            ROW_STATE_COMPLETE: "online_complete_rows",
        }
    )

    count_columns = [
        "total_rows",
        "present_rows",
        "absent_rows",
        "warmup_rows",
        "terminal_padded_absence_rows",
        "true_outage_candidate_rows",
        "online_partial_missing_rows",
        "online_complete_rows",
    ]
    summary[count_columns] = summary[count_columns].astype("int64")

    summary = summary.reset_index()
    summary["first_timestamp_utc"] = _format_timestamp_series(
        summary["first_timestamp_utc"],
    )
    summary["last_timestamp_utc"] = _format_timestamp_series(
        summary["last_timestamp_utc"],
    )
    summary["first_present_timestamp"] = _format_timestamp_series(
        summary["first_present_timestamp"],
    )
    summary["last_present_timestamp"] = _format_timestamp_series(
        summary["last_present_timestamp"],
    )

    metadata_columns = [
        "station_id",
        "station_name",
        "city",
        "latitude",
        "longitude",
        "elevation",
        "status_class",
    ]
    summary = registry[metadata_columns].merge(
        summary,
        on="station_id",
        how="left",
    )
    return summary[SUMMARY_COLUMNS]


def build_missingness_by_variable(classified: pd.DataFrame) -> pd.DataFrame:
    present_mask = pd.to_numeric(
        classified["data_present"],
        errors="coerce",
    ).eq(1)
    present_rows = classified.loc[present_mask, MEASUREMENT_COLUMNS]
    present_denominator = max(len(present_rows), 1)
    total_rows = len(classified)

    rows = []
    for variable in MEASUREMENT_COLUMNS:
        missing_count = int(classified[variable].isna().sum())
        missing_when_present = int(present_rows[variable].isna().sum())
        rows.append(
            {
                "variable": variable,
                "total_rows": total_rows,
                "missing_count": missing_count,
                "missing_pct": missing_count / total_rows,
                "missing_pct_when_data_present": (
                    missing_when_present / present_denominator
                ),
            },
        )
    return pd.DataFrame(rows)


def _timeline_segments(station_rows: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    ordered = station_rows.sort_values("hour_utc").copy()
    ordered["timeline_band"] = (
        ordered["row_state"]
        .map(TIMELINE_BAND_BY_ROW_STATE)
        .fillna("padded")
    )
    new_segment = (
        ordered["timeline_band"].ne(ordered["timeline_band"].shift())
        | ordered["hour_utc"].diff().ne(pd.Timedelta(hours=1))
    )
    ordered["segment_id"] = new_segment.cumsum()

    segments = []
    for _, segment in ordered.groupby("segment_id", sort=False):
        band = segment["timeline_band"].iloc[0]
        start = segment["hour_utc"].iloc[0]
        end = segment["hour_utc"].iloc[-1] + pd.Timedelta(hours=1)
        segments.append((band, start, end))
    return segments


def plot_station_coverage_timeline(
    classified: pd.DataFrame,
    registry: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_registry = registry.copy()
    plot_registry["install_date_sort"] = pd.to_datetime(
        plot_registry["install_date"],
        errors="coerce",
    )
    plot_registry = plot_registry.sort_values(
        ["install_date_sort", "station_id"],
        na_position="last",
    )

    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for row_number, station_id in enumerate(plot_registry["station_id"]):
        station_rows = classified.loc[classified["station_id"] == station_id]
        segments_by_band = {band: [] for band in TIMELINE_COLORS}
        for band, start, end in _timeline_segments(station_rows):
            start_num = mdates.date2num(start.to_pydatetime())
            width = (end - start) / pd.Timedelta(days=1)
            segments_by_band[band].append((start_num, width))

        for band, segments in segments_by_band.items():
            if not segments:
                continue
            ax.broken_barh(
                segments,
                (row_number - 0.34, 0.68),
                facecolors=TIMELINE_COLORS[band],
                edgecolors="none",
            )

    ax.set_yticks(range(len(plot_registry)))
    ax.set_yticklabels(plot_registry["station_id"], fontsize=7)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.7)
    ax.set_title(
        "Station coverage timeline (June 2025 – March 2026)",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("Station")
    ax.legend(
        handles=[
            Patch(color=TIMELINE_COLORS[band], label=TIMELINE_LABELS[band])
            for band in ["online", "outage", "padded", "warmup"]
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_station_variable_missingness(
    classified: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    station_order = registry.copy()
    station_order["status_class_order"] = pd.Categorical(
        station_order["status_class"],
        categories=STATUS_CLASS_ORDER,
        ordered=True,
    )
    station_order = station_order.sort_values(
        ["status_class_order", "station_id"],
        na_position="last",
    )

    rows = []
    for station_id in station_order["station_id"]:
        station_present = classified.loc[
            (classified["station_id"] == station_id)
            & pd.to_numeric(classified["data_present"], errors="coerce").eq(1),
            MEASUREMENT_COLUMNS,
        ]
        if station_present.empty:
            missing_pct = pd.Series(0.0, index=MEASUREMENT_COLUMNS)
        else:
            missing_pct = station_present.isna().mean()
        rows.append(missing_pct.rename(station_id))

    return pd.DataFrame(rows)


def plot_missingness_heatmap(
    station_variable_missingness: pd.DataFrame,
    output_path: Path,
) -> None:
    matrix = station_variable_missingness.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    fig.patch.set_facecolor("white")
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(np.arange(len(MEASUREMENT_COLUMNS)))
    ax.set_xticklabels(MEASUREMENT_COLUMNS, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(station_variable_missingness.index)))
    ax.set_yticklabels(station_variable_missingness.index, fontsize=7)
    ax.set_title(
        "Missingness per variable, conditional on data_present == 1",
        fontsize=13,
        pad=12,
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if value >= 0.05:
                text_color = "white" if value >= 0.55 else "black"
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=text_color,
                )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    colorbar.set_label("Missing when data_present == 1")

    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_data_audit() -> dict[str, Path]:
    ensure_directories()

    merged = pd.read_csv(MERGED_DATASET_PATH)
    registry = pd.read_csv(STATION_REGISTRY_PATH)
    classified = classify_row_states(merged, registry)

    audit_summary = build_audit_summary(classified, registry)
    missingness_by_variable = build_missingness_by_variable(classified)
    station_variable_missingness = build_station_variable_missingness(
        classified,
        registry,
    )

    classified.to_parquet(HOURLY_ROW_STATES_PATH, index=False)
    audit_summary.to_csv(DATA_AUDIT_SUMMARY_PATH, index=False)
    missingness_by_variable.to_csv(MISSINGNESS_BY_VARIABLE_PATH, index=False)
    plot_station_coverage_timeline(
        classified,
        registry,
        STATION_COVERAGE_FIGURE_PATH,
    )
    plot_missingness_heatmap(
        station_variable_missingness,
        MISSINGNESS_HEATMAP_PATH,
    )

    return {
        "hourly_row_states": HOURLY_ROW_STATES_PATH,
        "data_audit_summary": DATA_AUDIT_SUMMARY_PATH,
        "missingness_by_variable": MISSINGNESS_BY_VARIABLE_PATH,
        "station_coverage_timeline": STATION_COVERAGE_FIGURE_PATH,
        "missingness_heatmap": MISSINGNESS_HEATMAP_PATH,
    }


def main() -> None:
    written_files = run_data_audit()
    summary = pd.read_csv(DATA_AUDIT_SUMMARY_PATH)

    print("Phase 1 data audit complete.")
    print(f"Rows processed: {summary['total_rows'].sum():,}")
    print("Files written:")
    for label, path in written_files.items():
        print(f"  {label}: {path}")
    print("status_class counts:")
    print(summary["status_class"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
