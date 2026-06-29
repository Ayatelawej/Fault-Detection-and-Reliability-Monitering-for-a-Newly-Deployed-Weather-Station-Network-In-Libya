from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import NETWORK_OUTAGE_WINDOWS_PATH

OUTPUT_DIR = Path("docs/availability_investigation_checks")
OUTPUT_TXT = OUTPUT_DIR / "midnight_cadence.txt"
OUTPUT_CSV = OUTPUT_DIR / "midnight_cadence.csv"


def main() -> None:
    windows = pd.read_csv(NETWORK_OUTAGE_WINDOWS_PATH)
    windows["window_start_utc"] = pd.to_datetime(
        windows["window_start_utc"], utc=True
    )
    windows["start_hour"] = windows["window_start_utc"].dt.hour
    midnight = windows[windows["start_hour"].isin([22, 23])].copy()
    midnight = midnight.sort_values("window_start_utc").reset_index(drop=True)

    midnight["day_of_week"] = midnight["window_start_utc"].dt.day_name()
    midnight["day_of_month"] = midnight["window_start_utc"].dt.day
    midnight["year_month"] = midnight["window_start_utc"].dt.strftime("%Y-%m")
    midnight["iso_week"] = midnight["window_start_utc"].dt.strftime("%G-W%V")

    midnight["gap_hours"] = (
        midnight["window_start_utc"]
        .diff()
        .dt.total_seconds()
        / 3600.0
    )
    midnight["gap_days"] = midnight["gap_hours"] / 24.0

    dow_counts = midnight["day_of_week"].value_counts().reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]).fillna(0).astype(int)

    dom_counts = midnight["day_of_month"].value_counts().sort_index()

    ym_counts = midnight["year_month"].value_counts().sort_index()

    gaps = midnight["gap_days"].dropna()

    midnight.to_csv(OUTPUT_CSV, index=False)

    lines: list[str] = []
    lines.append("=== Midnight window cadence analysis ===")
    lines.append(f"Total midnight windows (start hour 22 or 23 UTC): {len(midnight)}")
    if len(midnight) > 0:
        lines.append(
            f"Date range: {midnight['window_start_utc'].min()} "
            f"to {midnight['window_start_utc'].max()}"
        )
    lines.append("")

    lines.append("=== Day of week distribution ===")
    lines.append(dow_counts.to_string())
    lines.append("")

    lines.append("=== Day of month distribution ===")
    lines.append(dom_counts.to_string())
    lines.append("")

    lines.append("=== Year-month distribution ===")
    lines.append(ym_counts.to_string())
    lines.append("")

    lines.append("=== Gap between consecutive midnight events (days) ===")
    if not gaps.empty:
        lines.append(f"Number of gaps: {len(gaps)}")
        lines.append(f"Median gap: {gaps.median():.2f} days")
        lines.append(f"Mean gap: {gaps.mean():.2f} days")
        lines.append(f"Min gap: {gaps.min():.2f} days")
        lines.append(f"Max gap: {gaps.max():.2f} days")
        lines.append("")
        lines.append("Gap distribution:")
        bins = [0, 1, 2, 3, 5, 7, 14, 30, 365]
        labels = [
            "0-1d", "1-2d", "2-3d", "3-5d",
            "5-7d", "7-14d", "14-30d", ">30d",
        ]
        binned = pd.cut(gaps, bins=bins, labels=labels, include_lowest=True)
        lines.append(binned.value_counts().reindex(labels).fillna(0).astype(int).to_string())
        lines.append("")

    lines.append("=== Per-window cadence detail ===")
    cols = [
        "window_id", "window_start_utc", "day_of_week",
        "day_of_month", "year_month", "iso_week",
        "gap_days", "station_count",
    ]
    lines.append(midnight[cols].to_string(index=False))

    report = "\n".join(lines) + "\n"
    OUTPUT_TXT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
