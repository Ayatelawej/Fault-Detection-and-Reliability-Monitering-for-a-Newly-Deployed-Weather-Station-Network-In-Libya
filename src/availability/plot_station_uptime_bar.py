from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SUMMARY_PATH = Path("data/processed/station_reliability_summary.csv")
OUTPUT_PATH = Path("outputs/figures/station_uptime_bar.png")


def main() -> None:
    df = pd.read_csv(SUMMARY_PATH)
    df = df.sort_values("uptime_pct", ascending=True).reset_index(drop=True)

    df["downtime_local_h"] = df["total_active_hours"] * (
        df["local_event_count"] / df["total_event_count"].replace(0, 1)
    ) * (df["median_event_duration_h"].fillna(0))
    df["downtime_midnight_h"] = df["total_active_hours"] * (
        df["network_midnight_event_count"] / df["total_event_count"].replace(0, 1)
    ) * (df["median_event_duration_h"].fillna(0))
    df["downtime_other_h"] = df["total_active_hours"] * (
        df["network_other_event_count"] / df["total_event_count"].replace(0, 1)
    ) * (df["median_event_duration_h"].fillna(0))

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(df))
    ax.barh(
        y_pos, df["uptime_pct"],
        color="#2c5f8d", edgecolor="white",
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["station_id"], fontsize=8)
    ax.set_xlabel("Uptime (%)")
    ax.set_title("Per-station uptime, June 15 2025 - March 31 2026")
    ax.set_xlim(0, 100)
    ax.axvline(95.0, color="gray", linestyle="--", linewidth=0.8)
    ax.text(95.0, len(df) - 0.5, " 95%", fontsize=8, color="gray")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
