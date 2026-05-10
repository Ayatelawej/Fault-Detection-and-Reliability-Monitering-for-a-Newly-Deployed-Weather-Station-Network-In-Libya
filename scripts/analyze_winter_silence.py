from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import HOURLY_ROW_STATES_PATH

OUTPUT_DIR = Path("docs/phase2_investigation_checks")
OUTPUT_TXT = OUTPUT_DIR / "winter_silence.txt"

INACTIVE_STATES = {
    "pre_install_padded_absence",
    "pre_install_padded_present",
    "pre_install_invalid_unknown",
    "terminal_padded_absence",
}

SILENCE_START = pd.Timestamp("2025-12-04 00:00:00", tz="UTC")
SILENCE_END = pd.Timestamp("2026-03-02 23:00:00", tz="UTC")


def main() -> None:
    states = pd.read_parquet(HOURLY_ROW_STATES_PATH)
    states = states[
        (states["hour_utc"] >= SILENCE_START)
        & (states["hour_utc"] <= SILENCE_END)
    ].copy()

    states["is_active"] = ~states["row_state"].isin(INACTIVE_STATES)
    states["is_offline"] = states["row_state"] == "true_outage_candidate"
    states["is_online"] = states["row_state"].isin(
        {"online_complete", "online_partial_missing"}
    )

    per_hour = states.groupby("hour_utc").agg(
        active_count=("is_active", "sum"),
        online_count=("is_online", "sum"),
        offline_count=("is_offline", "sum"),
    )

    midnight_hours = per_hour[per_hour.index.hour.isin([22, 23])]

    lines: list[str] = []
    lines.append("=== Winter silence diagnostic (Dec 4 2025 - Mar 2 2026) ===")
    lines.append(f"Hours analyzed: {len(per_hour)}")
    lines.append(
        f"Midnight-bin hours analyzed (hours 22-23 UTC): {len(midnight_hours)}"
    )
    lines.append("")

    lines.append("=== Online station count statistics across all hours ===")
    lines.append(f"Median online stations: {per_hour['online_count'].median():.1f}")
    lines.append(f"Mean online stations: {per_hour['online_count'].mean():.1f}")
    lines.append(f"Min online stations: {per_hour['online_count'].min()}")
    lines.append(f"Max online stations: {per_hour['online_count'].max()}")
    lines.append("")

    lines.append("=== Online station count at midnight bins (22-23 UTC) ===")
    lines.append(
        f"Median online stations: {midnight_hours['online_count'].median():.1f}"
    )
    lines.append(
        f"Hours with >= 10 online stations: "
        f"{(midnight_hours['online_count'] >= 10).sum()} of {len(midnight_hours)}"
    )
    lines.append(
        f"Hours with >= 15 online stations: "
        f"{(midnight_hours['online_count'] >= 15).sum()} of {len(midnight_hours)}"
    )
    lines.append(
        f"Hours with >= 20 online stations: "
        f"{(midnight_hours['online_count'] >= 20).sum()} of {len(midnight_hours)}"
    )
    lines.append("")

    lines.append("=== Online count distribution at midnight bins ===")
    lines.append(midnight_hours["online_count"].describe().to_string())
    lines.append("")

    lines.append("=== Per-week summary across the silence window ===")
    per_hour["iso_week"] = per_hour.index.strftime("%G-W%V")
    weekly = per_hour.groupby("iso_week").agg(
        mean_online=("online_count", "mean"),
        min_online=("online_count", "min"),
        mean_offline=("offline_count", "mean"),
    )
    lines.append(weekly.to_string())

    report = "\n".join(lines) + "\n"
    OUTPUT_TXT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
