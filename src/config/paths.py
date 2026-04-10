from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_STATION_DIR = DATA_DIR / 'raw' / 'stations'
MERGED_DIR = DATA_DIR / 'merged'
EXTERNAL_DIR = DATA_DIR / 'external'
LABELS_DIR = DATA_DIR / 'labels'
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'

ALL_DIRS = [
    RAW_STATION_DIR,
    MERGED_DIR,
    EXTERNAL_DIR,
    LABELS_DIR,
    PROCESSED_DIR,
    FIGURES_DIR,
]


def ensure_directories() -> None:
    for path in ALL_DIRS:
        path.mkdir(parents=True, exist_ok=True)
