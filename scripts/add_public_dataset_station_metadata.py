from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

DEFAULT_PUBLIC_DIR = Path(r"C:\Users\m\Desktop\Mozn Data")
CHUNKSIZE = 200_000
STATION_METADATA_COLUMNS = ["latitude", "longitude", "elevation_m"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add one clean copy of station latitude, longitude, and elevation "
            "to public observation CSVs from station_registry.csv."
        ),
    )
    parser.add_argument(
        "--public-dir",
        default=str(DEFAULT_PUBLIC_DIR),
        help="Public dataset directory to update.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=CHUNKSIZE,
        help="Rows per chunk for large CSV rewrites.",
    )
    return parser.parse_args()


def target_csvs(public_dir: Path) -> list[Path]:
    paths = [
        public_dir / "observations_pooled.csv",
        public_dir / "observations_complete.csv",
    ]

    for dirname in ["per_station_observations", "per_station_complete"]:
        directory = public_dir / dirname
        if directory.exists():
            paths.extend(sorted(directory.glob("*.csv")))
    return [path for path in paths if path.exists()]


def load_registry(public_dir: Path) -> pd.DataFrame:
    registry = pd.read_csv(public_dir / "station_registry.csv")
    required = ["station_id", *STATION_METADATA_COLUMNS]
    missing = [column for column in required if column not in registry.columns]
    if missing:
        raise ValueError(f"station_registry.csv is missing columns: {missing}")
    return registry[required].copy()


def ordered_columns(columns: list[str]) -> list[str]:
    columns = [
        column for column in columns
        if column not in STATION_METADATA_COLUMNS
    ]
    if "station_id" not in columns:
        raise ValueError("CSV is missing station_id.")
    station_index = columns.index("station_id")
    return (
        columns[: station_index + 1]
        + STATION_METADATA_COLUMNS
        + columns[station_index + 1 :]
    )


def rewrite_csv(path: Path, registry: pd.DataFrame, chunksize: int) -> None:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    output_columns = ordered_columns(header)
    temp_path = path.with_name(path.name + ".tmp")
    header_written = False

    try:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk = chunk.drop(columns=STATION_METADATA_COLUMNS, errors="ignore")
            chunk = chunk.merge(registry, on="station_id", how="left")
            chunk = chunk.reindex(columns=output_columns)
            chunk.to_csv(
                temp_path,
                mode="a",
                header=not header_written,
                index=False,
                encoding="utf-8-sig",
            )
            header_written = True
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    args = parse_args()
    public_dir = Path(args.public_dir)
    registry = load_registry(public_dir)

    paths = target_csvs(public_dir)
    for path in paths:
        rewrite_csv(path, registry, chunksize=args.chunksize)
        print(f"updated: {path}")
    print(f"CSV files updated: {len(paths)}")


if __name__ == "__main__":
    main()
