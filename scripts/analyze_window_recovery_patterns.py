from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import NETWORK_OUTAGE_WINDOWS_PATH

PULLS_DIR = Path("data/external/wu_minute_pulls")
OUTPUT_DIR = Path("docs/availability_investigation_checks")
OUTPUT_TXT = OUTPUT_DIR / "window_recovery_patterns.txt"
OUTPUT_CSV = OUTPUT_DIR / "window_recovery_patterns.csv"


def analyze_window(window_id: str, window_meta: pd.Series) -> dict:
    window_dir = PULLS_DIR / window_id
    obs_path = window_dir / "observations.csv"
    manifest_path = window_dir / "manifest.json"

    result: dict = {
        "window_id": window_id,
        "window_start_utc": window_meta["window_start_utc"],
        "window_end_utc": window_meta["window_end_utc"],
        "station_count_in_window": int(window_meta["station_count"]),
        "outage_class": window_meta.get("outage_class", "unknown"),
    }

    if not obs_path.exists():
        result["status"] = "missing_observations"
        return result

    obs = pd.read_csv(obs_path, parse_dates=["obs_time_utc"])
    if obs.empty:
        result["status"] = "empty_observations"
        return result

    obs["obs_time_utc"] = pd.to_datetime(obs["obs_time_utc"], utc=True)
    window_start = pd.Timestamp(window_meta["window_start_utc"], tz="UTC")
    window_end = pd.Timestamp(window_meta["window_end_utc"], tz="UTC")

    with open(manifest_path) as f:
        manifest = json.load(f)
    analysis_start = pd.Timestamp(manifest["analysis_start_utc"], tz="UTC")
    analysis_end = pd.Timestamp(manifest["analysis_end_utc"], tz="UTC")

    target_stations = manifest.get("stations", [])
    result["stations_targeted"] = len(target_stations)

    last_before: list[pd.Timestamp] = []
    first_after: list[pd.Timestamp] = []
    for sid in target_stations:
        sid_obs = obs[obs["station_id"] == sid].sort_values("obs_time_utc")
        before = sid_obs[sid_obs["obs_time_utc"] < window_start]
        after = sid_obs[sid_obs["obs_time_utc"] > window_end]
        if not before.empty:
            last_before.append(before["obs_time_utc"].iloc[-1])
        if not after.empty:
            first_after.append(after["obs_time_utc"].iloc[0])

    result["n_with_last_before"] = len(last_before)
    result["n_with_first_after"] = len(first_after)

    if last_before:
        last_before_series = pd.Series(last_before)
        result["drop_min_utc"] = str(last_before_series.min())
        result["drop_max_utc"] = str(last_before_series.max())
        result["drop_spread_seconds"] = (
            last_before_series.max() - last_before_series.min()
        ).total_seconds()
        result["drop_median_utc"] = str(last_before_series.median())
    else:
        result["drop_min_utc"] = None
        result["drop_max_utc"] = None
        result["drop_spread_seconds"] = None
        result["drop_median_utc"] = None

    if first_after:
        first_after_series = pd.Series(first_after)
        result["recovery_min_utc"] = str(first_after_series.min())
        result["recovery_max_utc"] = str(first_after_series.max())
        result["recovery_spread_seconds"] = (
            first_after_series.max() - first_after_series.min()
        ).total_seconds()
        result["recovery_median_utc"] = str(first_after_series.median())

        if last_before:
            drop_median = pd.Series(last_before).median()
            recovery_median = pd.Series(first_after).median()
            result["outage_duration_minutes"] = (
                recovery_median - drop_median
            ).total_seconds() / 60.0
        else:
            result["outage_duration_minutes"] = None
    else:
        result["recovery_min_utc"] = None
        result["recovery_max_utc"] = None
        result["recovery_spread_seconds"] = None
        result["recovery_median_utc"] = None
        result["outage_duration_minutes"] = None

    result["status"] = "ok"
    return result


def assign_drop_sync_class(drop_spread_seconds: object) -> str:
    value = pd.to_numeric(
        pd.Series([drop_spread_seconds]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(value):
        return "unknown"
    if value < 120:
        return "tight"
    return "loose"


def append_drop_duration_summary(
    lines: list[str],
    title: str,
    frame: pd.DataFrame,
) -> None:
    lines.append(f"=== {title}: drop synchronization ===")
    lines.append(f"Events: {len(frame)}")
    if frame.empty:
        lines.append("")
        return

    drop_spreads = pd.to_numeric(
        frame["drop_spread_seconds"],
        errors="coerce",
    ).dropna()
    if not drop_spreads.empty:
        lines.append(
            f"Median drop spread (max - min last-obs): "
            f"{drop_spreads.median():.1f} seconds"
        )
        lines.append(f"Mean drop spread: {drop_spreads.mean():.1f} seconds")
        lines.append(
            f"Range: {drop_spreads.min():.1f}s to "
            f"{drop_spreads.max():.1f}s"
        )
    lines.append("")
    lines.append(f"=== {title}: outage duration ===")
    durs = pd.to_numeric(
        frame["outage_duration_minutes"],
        errors="coerce",
    ).dropna()
    if not durs.empty:
        lines.append(f"Median duration: {durs.median():.1f} minutes")
        lines.append(f"Mean duration: {durs.mean():.1f} minutes")
        lines.append(f"Range: {durs.min():.1f} to {durs.max():.1f} minutes")
        lines.append(
            f"Events with duration > 120 minutes: "
            f"{(durs > 120).sum()} of {len(durs)}"
        )
    else:
        lines.append("No complete recovery durations captured.")
    lines.append("")


def main() -> None:
    windows = pd.read_csv(NETWORK_OUTAGE_WINDOWS_PATH)
    windows = windows.sort_values("window_start_utc").reset_index(drop=True)

    results = []
    for _, window in windows.iterrows():
        results.append(analyze_window(window["window_id"], window))

    df = pd.DataFrame(results)
    df["drop_sync_class"] = df["drop_spread_seconds"].map(assign_drop_sync_class)
    df.to_csv(OUTPUT_CSV, index=False)

    midnight_mask = pd.to_datetime(df["window_start_utc"]).dt.hour.isin([22, 23])
    midnight = df[midnight_mask].copy()
    nonmidnight = df[~midnight_mask].copy()

    lines: list[str] = []
    lines.append("=== Window recovery pattern analysis ===")
    lines.append(f"Total windows analyzed: {len(df)}")
    lines.append(f"  Midnight (start hour 22 or 23 UTC): {len(midnight)}")
    lines.append(f"  Non-midnight: {len(nonmidnight)}")
    lines.append("")

    ok = df[df["status"] == "ok"]
    lines.append(f"Successfully characterized: {len(ok)}")
    lines.append(f"Missing or empty: {len(df) - len(ok)}")
    recovery_missing = ok.loc[
        pd.to_numeric(ok["n_with_first_after"], errors="coerce")
        .lt(pd.to_numeric(ok["stations_targeted"], errors="coerce"))
    ]
    lines.append(
        "Events with recovery still not captured within 48h padding: "
        f"{len(recovery_missing)}"
    )
    lines.append("")

    if not midnight.empty:
        mok = midnight[midnight["status"] == "ok"]
        if not mok.empty:
            append_drop_duration_summary(
                lines,
                "Tight-sync midnight events",
                mok.loc[mok["drop_sync_class"].eq("tight")],
            )
            append_drop_duration_summary(
                lines,
                "Loose-sync midnight events",
                mok.loc[mok["drop_sync_class"].eq("loose")],
            )

    if not nonmidnight.empty:
        nok = nonmidnight[nonmidnight["status"] == "ok"]
        if not nok.empty:
            lines.append("=== Non-midnight events: drop synchronization ===")
            lines.append(
                f"Median drop spread: "
                f"{nok['drop_spread_seconds'].median():.1f} seconds"
            )
            lines.append(
                f"Range: {nok['drop_spread_seconds'].min():.1f}s to "
                f"{nok['drop_spread_seconds'].max():.1f}s"
            )
            lines.append("")
            lines.append("=== Non-midnight events: outage duration ===")
            durs = nok["outage_duration_minutes"].dropna()
            if not durs.empty:
                lines.append(f"Median duration: {durs.median():.1f} minutes")
                lines.append(f"Range: {durs.min():.1f} to {durs.max():.1f} minutes")
            lines.append("")

    lines.append("=== Per-window summary ===")
    cols = [
        "window_id", "window_start_utc", "station_count_in_window",
        "drop_spread_seconds", "drop_sync_class", "outage_duration_minutes",
        "recovery_spread_seconds", "status",
    ]
    present_cols = [c for c in cols if c in df.columns]
    lines.append(df[present_cols].to_string(index=False))

    report = "\n".join(lines) + "\n"
    OUTPUT_TXT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
