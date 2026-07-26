from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rules.result_figures import (
    RESULT_FIGURES_PATH,
    build_all_figures,
)
from src.workflows.prerequisites import require_files


def result_figure_inputs() -> dict[str, Path]:
    return {
        "external feature evidence": PROJECT_ROOT / "data/features/external_features.parquet",
        "external residual evidence": PROJECT_ROOT / "data/features/external_residuals.parquet",
        "spatial feature evidence": PROJECT_ROOT / "data/features/spatial_features.parquet",
        "spatial residual evidence": PROJECT_ROOT / "data/features/spatial_residuals.parquet",
        "systemic external evidence": PROJECT_ROOT / "data/features/systemic_external_evidence.csv",
        "reviewed external offset queue": PROJECT_ROOT / "data/labels/external_offset_queue_reviewed.csv",
        "spatial anomaly queue": PROJECT_ROOT / "data/labels/spatial_anomaly_queue.csv",
        "canonical merged dataset": PROJECT_ROOT / "data/merged/station_hourly_merged.csv",
    }


def require_result_inputs() -> None:
    require_files(
        "Result figure generation",
        result_figure_inputs(),
        "Use --set methodology in the public snapshot, or supply the preserved historical evidence inputs.",
    )


def _fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def main() -> None:
    require_result_inputs()
    external_features = pd.read_parquet(PROJECT_ROOT / "data/features/external_features.parquet")
    external_residuals = pd.read_parquet(PROJECT_ROOT / "data/features/external_residuals.parquet")
    spatial_features = pd.read_parquet(PROJECT_ROOT / "data/features/spatial_features.parquet")
    spatial_residuals = pd.read_parquet(PROJECT_ROOT / "data/features/spatial_residuals.parquet")
    evidence = pd.read_csv(PROJECT_ROOT / "data/features/systemic_external_evidence.csv")
    reviewed = pd.read_csv(PROJECT_ROOT / "data/labels/external_offset_queue_reviewed.csv")
    spatial_queue = pd.read_csv(PROJECT_ROOT / "data/labels/spatial_anomaly_queue.csv")
    merged = pd.read_csv(PROJECT_ROOT / "data/merged/station_hourly_merged.csv")
    result = build_all_figures(
        external_features,
        external_residuals,
        spatial_features,
        spatial_residuals,
        evidence,
        reviewed,
        spatial_queue,
        merged,
        RESULT_FIGURES_PATH,
    )

    print("RESULT FIGURE BUILD")
    print(f"output_dir={RESULT_FIGURES_PATH}")
    print()
    print("FIG 1 PRESSURE OFFSETS")
    print(
        result["pressure"].to_string(
            index=False,
            columns=[
                "station_id",
                "median",
                "p25",
                "p75",
                "n",
                "confirmed_offset",
                "confirmed_level_p50",
                "confirmed_events",
            ],
            formatters={
                "median": lambda value: _fmt(value, 3),
                "p25": lambda value: _fmt(value, 3),
                "p75": lambda value: _fmt(value, 3),
                "confirmed_level_p50": lambda value: _fmt(value, 3),
            },
        )
    )
    print()
    print("FIG 2 MONTHLY FLEET SOLAR RATIOS")
    print(
        result["solar_monthly"].to_string(
            index=False,
            formatters={
                "fleet_median": lambda value: _fmt(value, 3),
                "p25": lambda value: _fmt(value, 3),
                "p75": lambda value: _fmt(value, 3),
            },
        )
    )
    print()
    print("FIG 2 STATION CLEAR-DAY SOLAR RATIOS")
    print(
        result["solar_station"].to_string(
            index=False,
            formatters={"clear_day_ratio": lambda value: _fmt(value, 3)},
        )
    )
    print()
    print("FIG 3 SYSTEMIC SUPPORT COUNTS")
    print(result["systemic"].to_string(index=False))
    print(f"sum={int(result['systemic']['episodes'].sum())}")
    print()
    print("FIG 4 EXTERNAL-SPATIAL PRESSURE PAIRS")
    print(
        result["pairs"].to_string(
            index=False,
            formatters={
                "external_median": lambda value: _fmt(value, 3),
                "spatial_median": lambda value: _fmt(value, 3),
            },
        )
    )
    print()
    print("FIG 5 LOCALIZED SPATIAL WINDOWS")
    print(
        result["localized"].to_string(
            index=False,
            formatters={
                "spatial_level_p50": lambda value: _fmt(value, 3),
                "external_r_pressure_median": lambda value: _fmt(value, 3),
            },
        )
    )
    print()
    print("SAVED FIGURES")
    for key, path in result["paths"].items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
