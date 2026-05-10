from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.config.paths import (
    HOURLY_ROW_STATES_PATH,
    NETWORK_OUTAGE_WINDOWS_PATH,
)

OUTPUT_PATH = Path("outputs/figures/network_offline_fraction_timeline.png")

INACTIVE_STATES = {
    "pre_install_padded_absence",
    "pre_install_padded_present",
    "pre_install_invalid_unknown",
    "terminal_padded_absence",
}


def main() -> None:
    states = pd.read_parquet(HOURLY_ROW_STATES_PATH)
    windows = pd.read_csv(NETWORK_OUTAGE_WINDOWS_PATH)

    states = states.copy()
    states["is_active"] = ~states["row_state"].isin(INACTIVE_STATES)
    states["is_offline"] = states["row_state"] == "true_outage_candidate"

    per_hour = states.groupby("hour_utc").agg(
        active_count=("is_active", "sum"),
        offline_count=("is_offline", "sum"),
    )
    per_hour = per_hour[per_hour["active_count"] >= 10].copy()
    per_hour["offline_fraction"] = (
        per_hour["offline_count"] / per_hour["active_count"]
    )
    per_hour = per_hour.reset_index()

    windows["window_start_utc"] = pd.to_datetime(
        windows["window_start_utc"], utc=True
    )
    midnight = windows[windows["outage_class"] == "network_midnight"]
    other = windows[windows["outage_class"] == "network_other"]

    fig, (ax_main, ax_events) = plt.subplots(
        2, 1, figsize=(14, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax_main.plot(
        per_hour["hour_utc"],
        per_hour["offline_fraction"],
        color="#2c5f8d", linewidth=0.7,
    )
    ax_main.set_ylabel("Offline fraction")
    ax_main.set_title(
        "Network offline fraction over time (mature network: >=10 active stations)"
    )
    ax_main.set_ylim(0, 1.05)
    ax_main.grid(axis="y", linestyle=":", alpha=0.5)

    ax_events.vlines(
        midnight["window_start_utc"],
        ymin=0,
        ymax=midnight["station_count"],
        color="#d9534f",
        linewidth=2.0,
        label=f"network_midnight (n={len(midnight)})",
    )
    ax_events.vlines(
        other["window_start_utc"],
        ymin=0,
        ymax=other["station_count"],
        color="#f0ad4e",
        linewidth=2.0,
        label=f"network_other (n={len(other)})",
    )

    ax_events.set_ylabel("Stations in window")
    ax_events.set_xlabel("Date (UTC)")
    ax_events.set_ylim(0, 25)
    ax_events.legend(loc="upper right", fontsize=9)
    ax_events.grid(axis="y", linestyle=":", alpha=0.5)

    ax_events.xaxis.set_major_locator(mdates.MonthLocator())
    ax_events.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(
        ax_events.xaxis.get_majorticklabels(),
        rotation=30, ha="right",
    )

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
