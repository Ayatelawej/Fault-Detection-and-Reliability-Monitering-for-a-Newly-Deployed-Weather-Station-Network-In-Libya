from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


@dataclass
class WindowResult:
    window_id: str
    station_count: int
    status: str
    n_observations_in_window: int
    n_api_calls_hit_network: int
    error_message: str


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value for --skip-existing, got {value!r}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch pull WU minute-level observations for outage windows.",
    )
    parser.add_argument(
        "--windows-csv",
        default="data/processed/network_outage_windows.csv",
    )
    parser.add_argument("--padding-hours", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        default="data/external/wu_minute_pulls",
    )
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--skip-existing", type=parse_bool, default=True)
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--only-window", default=None)
    return parser.parse_args()


def read_manifest(output_dir: Path, window_id: str) -> dict[str, Any]:
    manifest_path = output_dir / window_id / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_windows(
    windows_csv: Path,
    *,
    only_window: str | None,
    start_from: str | None,
) -> pd.DataFrame:
    windows = pd.read_csv(windows_csv)
    required_columns = ["window_id", "station_count"]
    missing_columns = [
        column for column in required_columns
        if column not in windows.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise SystemExit(f"{windows_csv} is missing required columns: {missing}")

    if only_window:
        windows = windows.loc[windows["window_id"].eq(only_window)].copy()
        if windows.empty:
            raise SystemExit(f"Window id not found: {only_window}")
        return windows

    if start_from:
        matches = windows.index[windows["window_id"].eq(start_from)].tolist()
        if not matches:
            raise SystemExit(f"Start window id not found: {start_from}")
        windows = windows.loc[matches[0]:].copy()

    return windows.reset_index(drop=True)


def run_window(
    *,
    window_id: str,
    station_count: int,
    output_dir: Path,
    padding_hours: int,
    sleep_seconds: float,
    timeout_seconds: int,
    skip_existing: bool,
) -> WindowResult:
    window_output_dir = output_dir / window_id
    observations_path = window_output_dir / "observations.csv"
    if skip_existing and observations_path.exists():
        manifest = read_manifest(output_dir, window_id)
        return WindowResult(
            window_id=window_id,
            station_count=station_count,
            status="skipped",
            n_observations_in_window=int(
                manifest.get("n_observations_in_window", 0)
            ),
            n_api_calls_hit_network=int(
                manifest.get("n_api_calls_hit_network", 0)
            ),
            error_message="",
        )

    command = [
        sys.executable,
        "scripts/wu_minute_pull.py",
        "--window-id",
        window_id,
        "--padding-hours",
        str(padding_hours),
        "--output-dir",
        str(output_dir),
        "--sleep",
        str(sleep_seconds),
        "--timeout",
        str(timeout_seconds),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error_message = result.stderr.strip() or result.stdout.strip()
        return WindowResult(
            window_id=window_id,
            station_count=station_count,
            status="failed",
            n_observations_in_window=0,
            n_api_calls_hit_network=0,
            error_message=error_message,
        )

    manifest = read_manifest(output_dir, window_id)
    return WindowResult(
        window_id=window_id,
        station_count=station_count,
        status="success",
        n_observations_in_window=int(
            manifest.get("n_observations_in_window", 0)
        ),
        n_api_calls_hit_network=int(
            manifest.get("n_api_calls_hit_network", 0)
        ),
        error_message="",
    )


def print_window_result(result: WindowResult) -> None:
    print(
        f"{result.window_id} | "
        f"station_count={result.station_count} | "
        f"n_observations_in_window={result.n_observations_in_window} | "
        f"n_api_calls_hit_network={result.n_api_calls_hit_network} | "
        f"{result.status}"
    )
    if result.error_message:
        print(f"  error: {result.error_message}")


def print_final_summary(results: list[WindowResult], runtime_seconds: float) -> None:
    attempted = [
        result for result in results
        if result.status in {"success", "failed"}
    ]
    succeeded = [
        result for result in results
        if result.status == "success"
    ]
    failed = [
        result for result in results
        if result.status == "failed"
    ]
    skipped = [
        result for result in results
        if result.status == "skipped"
    ]
    total_api_hits = sum(
        result.n_api_calls_hit_network for result in results
    )

    print("")
    print("Final summary")
    print("-------------")
    print(f"Total windows selected: {len(results)}")
    print(f"Total windows attempted: {len(attempted)}")
    print(f"Total succeeded: {len(succeeded)}")
    print(f"Total skipped: {len(skipped)}")
    print(f"Total failed: {len(failed)}")
    print(
        "Failed window_ids: "
        + (", ".join(result.window_id for result in failed) if failed else "(none)")
    )
    print(f"Total API calls actually hit: {total_api_hits}")
    print(f"Total runtime seconds: {runtime_seconds:.1f}")


def main() -> None:
    started = time.monotonic()
    load_dotenv()
    if not os.environ.get("WU_API_KEY", "").strip():
        raise SystemExit("WU_API_KEY is not set. Add WU_API_KEY to .env.")

    args = parse_args()
    output_dir = Path(args.output_dir)
    windows = load_windows(
        Path(args.windows_csv),
        only_window=args.only_window,
        start_from=args.start_from,
    )

    results: list[WindowResult] = []
    for _, window in windows.iterrows():
        window_id = str(window["window_id"])
        station_count = int(
            pd.to_numeric(
                pd.Series([window["station_count"]]),
                errors="coerce",
            ).fillna(0).iloc[0]
        )
        result = run_window(
            window_id=window_id,
            station_count=station_count,
            output_dir=output_dir,
            padding_hours=int(args.padding_hours),
            sleep_seconds=float(args.sleep),
            timeout_seconds=int(args.timeout),
            skip_existing=bool(args.skip_existing),
        )
        results.append(result)
        print_window_result(result)
        time.sleep(1.0)

    runtime_seconds = time.monotonic() - started
    print_final_summary(results, runtime_seconds)


if __name__ == "__main__":
    main()
