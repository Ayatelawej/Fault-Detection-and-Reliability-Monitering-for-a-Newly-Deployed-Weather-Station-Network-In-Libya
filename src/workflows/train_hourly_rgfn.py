from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.hourly_baseline import filter_eligible_examples, load_hourly_tensor, write_metrics_json
from src.model.hourly_detection import MASK_MODE_PER_HOUR, MASK_MODES, SHORT_TENSOR_PATH
from src.model.hourly_rgfn import ENCODER_CONV, ENCODER_GRU
from src.model.hourly_rgfn_training import (
    DEVICE,
    RGFN_SEEDS,
    RGFN_THRESHOLDS,
    RGFN_WEIGHTS,
    HourlyRgfnTrainingConfig,
    comparison_report,
    load_calibrated_baseline,
    load_manifest_splits,
    master_comparison_frame,
    spaced_gap_frame,
    target_frame,
    train_hourly_rgfn_variant,
)
from src.workflows.prerequisites import require_files


OUTPUT_DIR = PROJECT_ROOT / "data" / "hourly_detection"
MANIFEST_PATH = OUTPUT_DIR / "hourly_baseline_split_manifest.csv"
BASELINE_METRICS_PATH = OUTPUT_DIR / "hourly_short_calibration_metrics.json"
METRICS_PATH = OUTPUT_DIR / "hourly_rgfn_comparison_metrics.json"
REPORT_PATH = OUTPUT_DIR / "hourly_rgfn_comparison_report.txt"
SWEEP_PATH = OUTPUT_DIR / "hourly_rgfn_comparison_validation_sweep.csv"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--tensor", type=Path, default=SHORT_TENSOR_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--baseline-metrics", type=Path, default=BASELINE_METRICS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--mask-mode", choices=MASK_MODES, default=MASK_MODE_PER_HOUR)
    return parser


def _sweep_frame(results: dict[str, dict[str, dict[str, object]]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, by_split in results.items():
        for split_name, result in by_split.items():
            selected = result["best_balanced"]
            for item in result["validation_seed_rows"]:
                metric = item["validation"]
                rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "row_type": "seed",
                        "seed": item["seed"],
                        "fault_class_weight": item["fault_class_weight"],
                        "threshold": item["threshold"],
                        "precision": metric["precision"],
                        "recall": metric["recall"],
                        "f1": metric["f1"],
                        "accuracy": metric["accuracy"],
                        "minimum_metric": min(metric["precision"], metric["recall"], metric["f1"]),
                        "precision_std": None,
                        "recall_std": None,
                        "f1_std": None,
                        "best_epoch": item["best_epoch"],
                        "best_validation_loss": item["best_validation_loss"],
                        "selected": bool(
                            float(item["fault_class_weight"]) == float(selected["fault_class_weight"])
                            and float(item["threshold"]) == float(selected["threshold"])
                        ),
                    }
                )
            for item in result["validation_grid"]:
                metric = item["validation"]
                deviation = item["validation_std"]
                rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "row_type": "aggregate",
                        "seed": None,
                        "fault_class_weight": item["fault_class_weight"],
                        "threshold": item["threshold"],
                        "precision": metric["precision"],
                        "recall": metric["recall"],
                        "f1": metric["f1"],
                        "accuracy": metric["accuracy"],
                        "minimum_metric": item["validation_minimum_metric"],
                        "precision_std": deviation["precision"],
                        "recall_std": deviation["recall"],
                        "f1_std": deviation["f1"],
                        "best_epoch": None,
                        "best_validation_loss": None,
                        "selected": bool(
                            float(item["fault_class_weight"]) == float(selected["fault_class_weight"])
                            and float(item["threshold"]) == float(selected["threshold"])
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["model", "split", "row_type", "fault_class_weight", "threshold", "seed"],
        kind="stable",
    ).reset_index(drop=True)


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(name): _jsonable(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main(argv: list[str] | None = None) -> None:
    args = parse_args().parse_args(argv)
    require_files(
        "Hourly RGFN training",
        {
            "short hourly tensor": args.tensor,
            "baseline split manifest": args.manifest,
            "calibrated baseline metrics": args.baseline_metrics,
        },
        "Run baseline and calibration through scripts/train_hourly_detection.py first.",
    )
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models" / "rgfn_hourly"
    examples, excluded_removed = filter_eligible_examples(load_hourly_tensor(Path(args.tensor)))
    splits = load_manifest_splits(examples, Path(args.manifest))
    baseline = load_calibrated_baseline(Path(args.baseline_metrics))
    training_config = HourlyRgfnTrainingConfig(mask_mode=args.mask_mode)
    print(f"device={DEVICE}", flush=True)
    print(f"eligible_hourly_examples={len(examples['y_binary'])}", flush=True)
    print(f"excluded_examples_removed={excluded_removed}", flush=True)
    results: dict[str, dict[str, dict[str, object]]] = {"RGFN-GRU": {}, "RGFN-CONV": {}}

    def progress(item: dict[str, object]) -> None:
        print(
            "candidate_complete="
            f"{item['encoder']}:{item['split']}:seed={item['seed']}:"
            f"weight={float(item['fault_class_weight']):.1f}:"
            f"epochs={item['epochs_completed']}:best_epoch={item['best_epoch']}:"
            f"validation_loss={float(item['best_validation_loss']):.6f}",
            flush=True,
        )

    for model_name, encoder in (("RGFN-GRU", ENCODER_GRU), ("RGFN-CONV", ENCODER_CONV)):
        for split_name in ("random", "spaced"):
            print(f"training={model_name} split={split_name} seeds={len(RGFN_SEEDS)} weights={len(RGFN_WEIGHTS)}", flush=True)
            results[model_name][split_name] = train_hourly_rgfn_variant(
                examples=examples,
                splits=splits[split_name],
                encoder=encoder,
                split_name=split_name,
                model_dir=model_dir,
                base_config=training_config,
                progress=progress,
            )
    gru = results["RGFN-GRU"]
    conv = results["RGFN-CONV"]
    payload = {
        "configuration": {
            "task": "hour-level binary fault detection",
            "window_hours": 7,
            "device": str(DEVICE),
            "training": asdict(training_config),
            "weights": list(RGFN_WEIGHTS),
            "thresholds": list(RGFN_THRESHOLDS),
            "seeds": list(RGFN_SEEDS),
            "selection": "aggregate validation maximum of minimum precision, recall, and f1",
            "test_evaluation": "one evaluation per selected seed model",
        },
        "inputs": {
            "tensor": str(Path(args.tensor)),
            "split_manifest": str(Path(args.manifest)),
            "baseline_metrics": str(Path(args.baseline_metrics)),
            "excluded_examples_removed": int(excluded_removed),
            "eligible_hourly_examples": int(len(examples["y_binary"])),
            "mask_mode": args.mask_mode,
        },
        "baseline": baseline,
        "rgfn_gru": gru,
        "rgfn_conv": conv,
        "master_test_comparison": master_comparison_frame(baseline, gru, conv).to_dict(orient="records"),
        "spaced_f1_answer": spaced_gap_frame(baseline, gru, conv).to_dict(orient="records"),
        "test_target_check": target_frame(baseline, gru, conv).to_dict(orient="records"),
    }
    report = comparison_report(baseline, gru, conv)
    metrics_path = output_dir / METRICS_PATH.name
    report_path = output_dir / REPORT_PATH.name
    sweep_path = output_dir / SWEEP_PATH.name
    write_metrics_json(_jsonable(payload), metrics_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    _sweep_frame(results).to_csv(sweep_path, index=False)
    print(report, flush=True)
    print(f"metrics={metrics_path}", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"validation_sweep={sweep_path}", flush=True)
    print(f"models={model_dir}", flush=True)


if __name__ == "__main__":
    main()
