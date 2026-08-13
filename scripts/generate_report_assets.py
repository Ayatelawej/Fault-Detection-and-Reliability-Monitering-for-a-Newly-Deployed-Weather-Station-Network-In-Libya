from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.workflows.build_methodology_figures import main as build_methodology
from src.workflows.build_result_figures import (
    main as build_results,
    require_result_inputs,
)
from src.workflows.build_july_evaluation_figures import (
    main as build_july_evaluation,
    require_july_evaluation_inputs,
)
from src.config.paths import HOURLY_ROW_STATES_PATH, STATION_REGISTRY_PATH
from src.workflows.prerequisites import require_files


def parse_args(argv: list[str] | None = None) -> str:
    parser = ArgumentParser(description="Generate report figure assets.")
    parser.add_argument(
        "--set",
        choices=("methodology", "results", "july-evaluation", "all"),
        default="methodology",
        dest="figure_set",
    )
    return str(parser.parse_args(argv).figure_set)


def main(argv: list[str] | None = None) -> None:
    figure_set = parse_args(argv)
    if figure_set in {"methodology", "all"}:
        require_files(
            "Methodology figure generation",
            {
                "station registry": STATION_REGISTRY_PATH,
                "hourly availability states": HOURLY_ROW_STATES_PATH,
            },
        )
    if figure_set in {"results", "all"}:
        require_result_inputs()
    if figure_set in {"july-evaluation", "all"}:
        require_july_evaluation_inputs()
    if figure_set in {"methodology", "all"}:
        build_methodology()
    if figure_set in {"results", "all"}:
        build_results()
    if figure_set in {"july-evaluation", "all"}:
        build_july_evaluation()


if __name__ == "__main__":
    main()
