from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.workflows.rebuild_detection_features import (
    DEFAULT_FIVE_MIN_DIR,
    DEFAULT_REFERENCE_DIR,
    rebuild_detection_features,
)
from src.config.paths import MERGED_DATASET_PATH, STATION_REGISTRY_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, default=MERGED_DATASET_PATH)
    parser.add_argument("--registry", type=Path, default=STATION_REGISTRY_PATH)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--five-min-dir", type=Path, default=DEFAULT_FIVE_MIN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = rebuild_detection_features(
        merged_path=args.merged,
        registry_path=args.registry,
        reference_dir=args.reference_dir,
        five_min_dir=args.five_min_dir,
    )
    print("DETECTION FEATURES REBUILT")
    for name, value in rows.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
