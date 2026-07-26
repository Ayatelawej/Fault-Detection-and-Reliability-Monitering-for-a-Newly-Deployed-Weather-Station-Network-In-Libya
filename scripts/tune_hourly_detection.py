from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.workflows.resume_hourly_tuning import main as resume_tuning
from src.workflows.tune_hourly_detection import main as run_tuning


RUNNERS: dict[str, Callable[[list[str] | None], None]] = {
    "run": run_tuning,
    "resume": resume_tuning,
}


def parse_args(argv: list[str] | None = None) -> tuple[str, list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = ArgumentParser(
        description="Run or resume the hour-level RGFN tuning experiment.",
    )
    parser.add_argument("mode", choices=tuple(RUNNERS))
    if not values or values[0] in {"-h", "--help"}:
        parser.print_help()
        raise SystemExit(0)
    args = parser.parse_args(values[:1])
    return str(args.mode), values[1:]


def main(argv: list[str] | None = None) -> None:
    mode, remaining = parse_args(argv)
    RUNNERS[mode](remaining)


if __name__ == "__main__":
    main()
