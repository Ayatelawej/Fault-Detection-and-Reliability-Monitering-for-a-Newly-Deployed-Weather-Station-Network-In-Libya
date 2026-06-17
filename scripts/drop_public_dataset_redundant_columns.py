from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

DEFAULT_PUBLIC_DIR = Path(r"C:\Users\m\Desktop\Mozn Data")
CHUNKSIZE = 200_000

REDUNDANT_COLUMNS = {
    "registry_station_id",
    "registry_station_name",
    "registry_city",
    "registry_country",
    "registry_latitude",
    "registry_longitude",
    "registry_elevation_m",
    "registry_install_date",
    "api_request_date",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove public dataset columns that duplicate station_registry.csv."
        ),
    )
    parser.add_argument(
        "--public-dir",
        default=str(DEFAULT_PUBLIC_DIR),
        help="Public dataset directory to clean.",
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


def clean_csv(path: Path, chunksize: int) -> bool:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    keep_columns = [
        column for column in header
        if column not in REDUNDANT_COLUMNS
    ]
    if keep_columns == header:
        return False

    temp_path = path.with_name(path.name + ".tmp")
    header_written = False
    try:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk = chunk.reindex(columns=keep_columns)
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
    return True


def main() -> None:
    args = parse_args()
    public_dir = Path(args.public_dir)

    updated = 0
    paths = target_csvs(public_dir)
    for path in paths:
        changed = clean_csv(path, chunksize=args.chunksize)
        updated += int(changed)
        status = "updated" if changed else "already clean"
        print(f"{status}: {path}")

    print(f"CSV files checked: {len(paths)}")
    print(f"CSV files updated: {updated}")


if __name__ == "__main__":
    main()
