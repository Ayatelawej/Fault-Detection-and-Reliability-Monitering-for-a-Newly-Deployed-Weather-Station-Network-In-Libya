from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from concurrent_offline import main as concurrent_offline_main
from concurrent_offline_mature import main as concurrent_offline_mature_main
from midnight_prevalence import main as midnight_prevalence_main
from scripts.analyze_local_cooccurrence import main as local_cooccurrence_main
from scripts.analyze_local_cooccurrence_clean import (
    main as local_cooccurrence_clean_main,
)
from scripts.analyze_local_cooccurrence_event_clean import (
    main as local_cooccurrence_event_clean_main,
)
from scripts.analyze_midnight_cadence import main as midnight_cadence_main
from scripts.analyze_window_geography import main as window_geography_main
from scripts.analyze_window_durations_hourly import (
    main as window_durations_hourly_main,
)
from scripts.analyze_window_recovery_patterns import (
    main as window_recovery_patterns_main,
)
from scripts.analyze_winter_silence import main as winter_silence_main

OUTPUT_DIR = Path(__file__).resolve().parent
REGISTRY_CSV = Path("data/merged/station_registry.csv")
WINDOWS_CSV = Path("data/processed/network_outage_windows.csv")
OBSERVATIONS_CSV = Path(
    "data/external/wu_minute_pulls/NW_20260308T23/observations.csv"
)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")


def top_network_windows_geography() -> str:
    reg = pd.read_csv(REGISTRY_CSV)
    wins = pd.read_csv(WINDOWS_CSV)

    top = wins.nlargest(5, "station_count")
    sections = []

    for _, window in top.iterrows():
        station_ids = str(window["station_ids"]).split(";")
        sub = (
            reg.loc[
                reg["station_id"].isin(station_ids),
                ["station_id", "city", "latitude", "longitude"],
            ]
            .sort_values("city")
            .reset_index(drop=True)
        )
        sections.extend(
            [
                "",
                "=" * 70,
                (
                    f"{window['window_id']}  |  "
                    f"{window['station_count']} stations  |  "
                    f"{window['window_start_utc']}"
                ),
                "=" * 70,
                sub.to_string(index=False),
                f"Cities affected: {sub['city'].value_counts().to_dict()}",
                (
                    f"Lat range: {sub['latitude'].min():.2f} to "
                    f"{sub['latitude'].max():.2f}"
                ),
                (
                    f"Lon range: {sub['longitude'].min():.2f} to "
                    f"{sub['longitude'].max():.2f}"
                ),
            ]
        )

    return "\n".join(sections).strip() + "\n"


def network_window_hour_distribution() -> str:
    wins = pd.read_csv(WINDOWS_CSV)
    wins["hour"] = pd.to_datetime(
        wins["window_start_utc"],
        utc=True,
        errors="coerce",
    ).dt.hour

    lines = [
        wins["hour"].value_counts().sort_index().to_string(),
        "",
        f"Total windows: {len(wins)}",
        f"Windows at hour 22 or 23 UTC: {wins['hour'].isin([22, 23]).sum()}",
        f"Windows at other hours: {(~wins['hour'].isin([22, 23])).sum()}",
        "",
        "Non-midnight windows:",
        wins.loc[
            ~wins["hour"].isin([22, 23]),
            ["window_id", "window_start_utc", "station_count", "station_ids"],
        ].to_string(),
    ]
    return "\n".join(lines) + "\n"


def wu_observation_cadence() -> str:
    obs = pd.read_csv(OBSERVATIONS_CSV)
    obs["obs_time_utc"] = pd.to_datetime(
        obs["obs_time_utc"],
        utc=True,
        errors="coerce",
    )

    rows = []
    for station_id, group in obs.sort_values("obs_time_utc").groupby(
        "station_id"
    ):
        diffs = (
            group["obs_time_utc"]
            .diff()
            .dropna()
            .dt.total_seconds()
            .div(60)
        )
        rows.append(
            {
                "station_id": station_id,
                "n_obs": len(group),
                "min_gap_min": diffs.min() if len(diffs) else None,
                "median_gap_min": diffs.median() if len(diffs) else None,
                "max_gap_min": diffs.max() if len(diffs) else None,
                "first": group["obs_time_utc"].min(),
                "last": group["obs_time_utc"].max(),
            }
        )

    return pd.DataFrame(rows).to_string(index=False) + "\n"


def main() -> None:
    _write_text(
        OUTPUT_DIR / "top_network_windows_geography.txt",
        top_network_windows_geography(),
    )
    _write_text(
        OUTPUT_DIR / "network_window_hour_distribution.txt",
        network_window_hour_distribution(),
    )
    _write_text(
        OUTPUT_DIR / "wu_observation_cadence.txt",
        wu_observation_cadence(),
    )
    midnight_prevalence_main()
    concurrent_offline_main()
    concurrent_offline_mature_main()
    local_cooccurrence_main()
    local_cooccurrence_clean_main()
    local_cooccurrence_event_clean_main()
    midnight_cadence_main()
    window_geography_main()
    window_recovery_patterns_main()
    window_durations_hourly_main()
    winter_silence_main()


if __name__ == "__main__":
    main()
