# Weather Station Network Fault Detection and Reliability Monitoring

This repository contains the publishable analysis code and evidence for an
hour-level fault-detection and reliability-monitoring study of a 26-station
personal weather-station network in Libya. The published canonical dataset is
frozen through June 2026.

The live detection unit is a **station-hour**. The system builds reproducible
episode labels, expands them into hour-level fault/not-fault targets, trains a
gradient-boosted baseline and a Reliability-Aware Gated Fusion Network (RGFN),
and contains a separate 6/12/24-hour outage-risk prototype.

The outage-risk artifacts are June-current prototype evidence, not final
forecasting claims: their current labels use row offsets rather than verified
continuous clock hours and their split is not horizon-purged.

## Scope and reproducibility

This is an **analysis-code-and-evidence repository**, not a clean-clone
source-data reconstruction. It begins with the published canonical dataset:

```text
data/merged/station_hourly_merged.csv
```

Source retrieval, five-minute normalisation, staging, and canonical-dataset
upsert tooling are intentionally excluded. No credentials or source-acquisition
capability are included.

Public ERA5/Open-Meteo reference fetching is retained. Exact rebuilding of all
detection features also requires separately supplied complete five-minute
station files, one `<station_id>_complete.csv` file per station. Set
`FIVE_MINUTE_INPUT_DIR` or pass `--five-min-dir` to point at that directory.
Several large generated feature, tensor, model, and checkpoint artifacts are
also intentionally ignored; the repository preserves the code, tests, labels,
and published evidence needed to study the system without implying that every
historical result can be regenerated from a bare clone.

## What is included

- The publishable canonical station-hour dataset and station registry.
- The live reproducible label file, `data/labels/episode_labels.csv`.
- Current availability evidence and preliminary outage-risk artifacts.
- Analysis, labelling, modelling, evaluation, and report-asset code.
- Regression tests and architecture documentation for the live system.

## Pipeline

| Stage | Front door | Purpose |
|---|---|---|
| Public reference data | `scripts/fetch_reference_data.py` | Fetch/cache public ERA5/Open-Meteo reference observations. |
| Detection features | `scripts/rebuild_detection_features.py` | Rebuild statistical, reference, spatial, and feature-matrix evidence from the published dataset and separately supplied five-minute inputs. |
| Episode labels | `scripts/build_labels.py` | Apply the reproducible multi-label rules and write the live episode labels. |
| Hour-level dataset | `scripts/build_hourly_dataset.py` | Expand episode labels into past-only short and long station-hour tensors. |
| Model training | `scripts/train_hourly_detection.py` | Train the baseline or RGFN and run the fixed training workflows. |
| RGFN tuning | `scripts/tune_hourly_detection.py` | Run or resume RGFN architecture and feature experiments. |
| Outage-risk evaluation | `scripts/evaluate_outage_risk.py` | Evaluate the independent 6/12/24-hour future-outage task. |
| Report assets | `scripts/generate_report_assets.py` | Generate methodology and result figures. |

See [REPO_MAP.md](REPO_MAP.md) for the detailed data flow, modules, inputs,
outputs, and test coverage.

## Quick start

Install dependencies and run the regression suite:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The directly runnable outage-risk evaluator is retained as a prototype route,
not as a final report-headline result:

```bash
python scripts/evaluate_outage_risk.py
```

To obtain fresh public reference data:

```bash
python scripts/fetch_reference_data.py
```

To rebuild the full feature set when complete five-minute inputs are available:

```bash
python scripts/rebuild_detection_features.py --five-min-dir <path-to-complete-five-minute-files>
```

That feature rebuild writes generated local artifacts. Build labels, the
hour-level dataset, and models only after their required generated inputs have
been supplied or rebuilt; their exact dependencies are listed in
[REPO_MAP.md](REPO_MAP.md). Each front door checks its prerequisites before
writing outputs and reports every missing input together.

## Live labels and states

`data/labels/episode_labels.csv` is the live label source used by the hourly
path. It contains 4,799 episodes: 1,121 `fault`, 3,533 `benign`, and 145
`borderline_review` episodes. Borderline-review episodes are excluded from
training and evaluation.

The retired frozen legacy label inventory is not required. When it is present
locally, build_labels.py writes a comparison crosswalk; otherwise it writes
the live labels and reports that the optional comparison was skipped.

For a non-fault station-hour, `display_state` distinguishes `benign` (at least
one detector fired) from `clean` (no detector fired). This is display metadata,
not a separate model target.

## Repository structure

- `data/merged/` — published canonical station-hour dataset and registry.
- `data/labels/` — live episode labels and Layer 2 calibration evidence.
- `data/processed/` — published audit and availability evidence.
- `data/eval/` — preliminary outage-risk prototype artifacts.
- `src/` — analysis, labelling, feature, model, reliability, and report code.
- `scripts/` — the eight public pipeline front doors.
- `tests/` — regression tests for the retained public system.

## Licence and data use

Code is released under the repository licence. The included dataset and derived
artifacts are publishable for this study. Please cite the associated project
report when using its results.
