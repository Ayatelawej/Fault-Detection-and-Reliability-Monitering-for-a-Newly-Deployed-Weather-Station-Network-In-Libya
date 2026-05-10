from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.paths import HOURLY_ROW_STATES_PATH

OUTPUT_PATH = Path(__file__).parent / "concurrent_offline.txt"

PRE_INSTALL_STATES = {
    "pre_install_padded_absence",
    "pre_install_padded_present",
    "pre_install_invalid_unknown",
}
INACTIVE_STATES = PRE_INSTALL_STATES | {"terminal_padded_absence"}


def main() -> None:
    states = pd.read_parquet(HOURLY_ROW_STATES_PATH)

    states = states.copy()
    states["is_active"] = ~states["row_state"].isin(INACTIVE_STATES)
    states["is_offline"] = states["row_state"] == "true_outage_candidate"

    per_hour = states.groupby("hour_utc").agg(
        active_count=("is_active", "sum"),
        offline_count=("is_offline", "sum"),
    )
    per_hour = per_hour[per_hour["active_count"] > 0].copy()
    per_hour["offline_fraction"] = (
        per_hour["offline_count"] / per_hour["active_count"]
    )
    per_hour["hour_of_day"] = per_hour.index.hour

    total_hours = len(per_hour)

    thresholds = [0.5, 0.7, 0.8, 0.9, 1.0]
    threshold_counts = {
        t: int((per_hour["offline_fraction"] >= t).sum()) for t in thresholds
    }

    top_blackout = per_hour.nlargest(20, "offline_fraction").reset_index()

    high_blackout = per_hour[per_hour["offline_fraction"] >= 0.8]
    hod_dist_high = high_blackout["hour_of_day"].value_counts().sort_index()

    full_blackout = per_hour[per_hour["offline_fraction"] >= 1.0]
    hod_dist_full = full_blackout["hour_of_day"].value_counts().sort_index()

    target = pd.Timestamp("2026-03-08 23:00:00", tz="UTC")
    target_row = (
        per_hour.loc[target] if target in per_hour.index else None
    )

    lines: list[str] = []
    lines.append("=== Concurrent offline analysis ===")
    lines.append(f"Total hours analyzed (active_count > 0): {total_hours}")
    lines.append("")
    lines.append("Hours by offline_fraction threshold:")
    for t, c in threshold_counts.items():
        pct = 100.0 * c / total_hours if total_hours > 0 else 0.0
        lines.append(f"  >= {int(t * 100):>3d}%: {c:>5d} hours ({pct:.2f}% of all hours)")
    lines.append("")
    lines.append("=== Top 20 hours by offline_fraction ===")
    lines.append(
        top_blackout[
            ["hour_utc", "active_count", "offline_count", "offline_fraction"]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append(
        "=== Hour-of-day distribution for hours with offline_fraction >= 80% ==="
    )
    lines.append(f"(total such hours: {len(high_blackout)})")
    lines.append(hod_dist_high.to_string())
    lines.append("")
    lines.append(
        "=== Hour-of-day distribution for hours with offline_fraction == 100% ==="
    )
    lines.append(f"(total such hours: {len(full_blackout)})")
    lines.append(hod_dist_full.to_string())
    lines.append("")
    lines.append("=== March 8 23:00 UTC specifically ===")
    if target_row is not None:
        lines.append(
            f"active_count: {int(target_row['active_count'])}, "
            f"offline_count: {int(target_row['offline_count'])}, "
            f"offline_fraction: {target_row['offline_fraction']:.3f}"
        )
    else:
        lines.append("Target hour not present in dataset.")

    report = "\n".join(lines) + "\n"
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
