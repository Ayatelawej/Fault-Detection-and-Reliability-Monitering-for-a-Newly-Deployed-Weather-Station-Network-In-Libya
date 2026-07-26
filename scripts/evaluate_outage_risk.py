from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.availability.risk_dataset import HORIZONS, build_risk_dataset
from src.availability.risk_eval import evaluate_horizon, event_recall
from src.config.paths import AVAILABILITY_EVENTS_PATH, HOURLY_ROW_STATES_PATH
from src.workflows.prerequisites import require_files

OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"
HOUR_METRICS_PATH = OUTPUT_DIR / "outage_risk_hour_metrics.csv"
EVENT_METRICS_PATH = OUTPUT_DIR / "outage_risk_event_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "outage_risk_predictions.parquet"


def evaluate_all_with_predictions(events_path: Path = AVAILABILITY_EVENTS_PATH) -> dict[str, pd.DataFrame]:
    dataset = build_risk_dataset()
    events = pd.read_parquet(events_path)
    metric_frames = []
    event_rows = []
    prediction_rows = []
    for horizon in HORIZONS:
        horizon_frame = dataset.for_horizon(horizon)
        result = evaluate_horizon(horizon_frame, horizon)
        metric_frames.append(result["metrics"])
        for model_name, prediction in result["predictions"].items():
            row = event_recall(result["split"]["test"], events, horizon, prediction.pred)
            event_rows.append({"horizon_h": horizon, "model": model_name, **row})
            frame = result["split"]["test"].loc[:, ["station_id", "hour_utc", "y"]].copy()
            frame["horizon_h"] = int(horizon)
            frame["model"] = model_name
            frame["prob"] = prediction.prob.astype(float)
            frame["pred"] = prediction.pred.astype(int)
            frame["threshold"] = float(prediction.threshold)
            frame["split"] = "test"
            prediction_rows.append(frame)
    return {
        "hour_metrics": pd.concat(metric_frames, ignore_index=True),
        "event_metrics": pd.DataFrame(event_rows),
        "predictions": pd.concat(prediction_rows, ignore_index=True),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate outage-risk models by forecast horizon.")
    parser.add_argument("--events", type=Path, default=AVAILABILITY_EVENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    require_files(
        "Outage-risk evaluation",
        {
            "hourly availability states": HOURLY_ROW_STATES_PATH,
            "availability events": args.events,
        },
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_all_with_predictions(Path(args.events))
    hour_metrics_path = output_dir / HOUR_METRICS_PATH.name
    event_metrics_path = output_dir / EVENT_METRICS_PATH.name
    predictions_path = output_dir / PREDICTIONS_PATH.name
    result["hour_metrics"].to_csv(hour_metrics_path, index=False)
    result["event_metrics"].to_csv(event_metrics_path, index=False)
    result["predictions"].to_parquet(predictions_path, index=False)
    print("OUTAGE RISK HOUR-LEVEL METRICS")
    print(result["hour_metrics"].to_string(index=False))
    print()
    print("OUTAGE RISK EVENT-LEVEL METRICS")
    print(result["event_metrics"].to_string(index=False))
    print()
    print(f"hour_metrics={hour_metrics_path}")
    print(f"event_metrics={event_metrics_path}")
    print(f"predictions={predictions_path}")


if __name__ == "__main__":
    main()
