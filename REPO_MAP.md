# Repository map

This map describes the public analysis snapshot. It is organised by the
workflow rather than alphabetically so a reader can understand the system
without reconstructing its private source-acquisition history.

## Overview

The project monitors a 26-station personal weather-station network. Its
deployed modelling unit is a **station-hour**. The main fault-detection route
uses rule and reference evidence to label episodes, expands those labels into
past-only hourly windows, trains binary fault/not-fault models, and evaluates
retrospective event-level mechanism reason codes. The reliability route adds
full/partial outage classification, a causal current-health score, five
health-forecast horizons for transmitting stations, and a combined operational
scorecard. Corrected incident-risk experiments are retained as future work.
The final product layer is a read-only July replay dashboard over frozen binary
detection, health, forecast, and detector-evidence artifacts.

There are ten public front doors in `scripts/`. The public workflow begins
with the published canonical data file, not raw-source acquisition:

```text
data/merged/station_hourly_merged.csv
  -> public reference cache + separately supplied five-minute inputs
  -> detection features
  -> episode labels
  -> hourly tensors
  -> baseline / RGFN training, tuning, and retrospective reason codes
  -> full/partial availability evidence
  -> current health -> five-horizon health forecast -> station scorecard
  -> independent July scoring artifacts -> operational replay dashboard
  -> incident-risk experiments and report figures
```

### Scope boundary

This is an analysis-code-and-evidence snapshot. Source retrieval,
five-minute normalisation, staging, and canonical-data upsert code are
intentionally absent. Public ERA5/Open-Meteo fetching remains available.

A full feature rebuild needs complete five-minute files supplied separately:
one `<station_id>_complete.csv` per station. Set
`FIVE_MINUTE_INPUT_DIR` or pass `--five-min-dir` to the feature-rebuild
front door. Generated feature matrices, tensors, model checkpoints, and some
historical figure inputs are intentionally not versioned. A bare clone can
inspect the data, labels, code, and published evidence, but cannot reproduce
every historical model run without those inputs.

Health-forecast evaluation files and model bundles are generated locally and
ignored. The tracked operational scorecard is published evidence; regenerating
it requires first running the health forecast so the five selected
transmitting-origin model bundles exist.

## Pipeline

### 1. Public reference data

**Purpose.** Fetch/cache public ERA5/Open-Meteo observations for active
stations in the published registry.

**Front door.** `scripts/fetch_reference_data.py`

**Main code.** `src/rules/config.py` defines reference variables and cache
settings. The fetch front door owns the public API interaction.

**Inputs/outputs.** Reads the registry and canonical merged-data header;
writes ignored local files below `data/external/reference_hourly/` and a
reference manifest.

**Tests.** `tests/test_external_fetch.py`,
`tests/test_external_conventions.py`.

**Current limitation.** The frozen reference configuration covers
2025-06-15 through 2026-06-30 as an exact 9,144-hour UTC index. `IJANZO4` is
excluded from active fetches and retained as a historical cache. Extending the
dataset requires deliberately updating this configuration before rebuilding
reference-dependent features.

### 2. Detection features

**Purpose.** Rebuild statistical anomaly evidence, events, episodes, review
queue, external residuals/features, spatial residuals/features, and the
feature matrix.

**Front door.** `scripts/rebuild_detection_features.py`

**Main code.** `src/workflows/rebuild_detection_features.py` orchestrates
`src/rules/score.py`, `events.py`, `episodes.py`, `clustering.py`,
`review_queue.py`, `external_residuals.py`, `external_features.py`,
`spatial_residuals.py`, `spatial_offsets.py`, and `feature_matrix.py`.

**Inputs/outputs.** Reads the canonical merged dataset, station registry,
public reference cache, and supplied five-minute files. It writes generated
statistical, external, spatial, and feature-matrix artifacts. The tracked
`data/processed/review_queue.csv` is published evidence; most regenerated
artifacts are ignored. Current feature construction is retrospective, so the
resulting model metrics are not prospective deployment validation.

**Tests.** `tests/test_statistical_anomaly.py`,
`tests/test_external_residuals.py`, `tests/test_external_features.py`,
`tests/test_spatial_residuals.py`, `tests/test_spatial_offsets.py`,
`tests/test_feature_matrix.py`, and `tests/test_stuck_confirmation.py`.

### 3. Episode labelling

**Purpose.** Build reproducible multi-label episode states from rule,
contextual, external, spatial, and Layer 2 calibration evidence.

**Front door.** `scripts/build_labels.py`

**Main code.** `src/rules/labelling.py`, `statistical_gate.py`,
`layer2_calibration.py`, `channel_handlers.py`, `score.py`, `events.py`, and
`episodes.py`.

**Inputs/outputs.** Reads canonical data plus regenerated external/spatial
evidence. The live output is the tracked
`data/labels/episode_labels.csv`. It may also write Layer 2 evidence,
review files, and an optional legacy comparison crosswalk. The crosswalk is
written only when the retired frozen label inventory is supplied locally; it
is not required to build live labels.

**Tests.** `tests/test_labelling.py`, `tests/test_statistical_gate.py`,
`tests/test_layer2_calibration.py`, and
`tests/test_stuck_confirmation.py`.

**Live population.** `episode_labels.csv` contains 4,799 episodes: 1,121
`fault`, 3,533 `benign`, and 145 `borderline_review`. It is the only
episode label source used by the hourly training path.

### 4. Hour-level detection dataset

**Purpose.** Expand episode states across the full station-hour record and
build past-only short (7-hour) and long (49-hour) tensors for binary
fault/not-fault detection. Window assembly is past-only, while some underlying
detection features are retrospective snapshots.

**Front door.** `scripts/build_hourly_dataset.py`

**Main code.** `src/model/hourly_detection.py` and
`src/model/feature_spec.py`.

**Inputs/outputs.** Reads canonical data, `data/features/feature_matrix.parquet`,
and `data/labels/episode_labels.csv`. Writes `hourly_labels.csv` and
ignored short/long tensor files under `data/hourly_detection/`.

**Tests.** `tests/test_hourly_detection_dataset.py`.

### 5. Hour-level model training

**Purpose.** Train the gradient-boosted baseline or RGFN, compare split
strategies, and select an operating threshold using validation data. All
predefined configurations are reported; held-out test metrics do not select or
name a winner.

**Front door.** `scripts/train_hourly_detection.py` with `baseline`,
`split-comparison`, `calibration`, `rgfn`, or `reason-codes`.

**Main code.** `src/workflows/train_hourly_baseline.py`,
`train_hourly_split_comparison.py`, `train_hourly_calibration.py`, and
`train_hourly_rgfn.py`; `src/model/hourly_baseline.py`,
`hourly_calibration.py`, `hourly_rgfn.py`, and
`hourly_rgfn_training.py`.

**Inputs/outputs.** Reads generated tensors and writes ignored split manifests,
metrics, reports, feature importance, and model/checkpoint bundles. The
required order is baseline, calibration, then RGFN; the split-comparison run
is independent experiment evidence. Reason-code outputs are retrospective
event-level mechanism analyses; they are not causal live-dashboard diagnoses.

**Tests.** `tests/test_hourly_baseline.py`,
`tests/test_hourly_calibration.py`, `tests/test_rgfn_hourly.py`, and
`tests/test_front_door_scripts.py`.

### 6. RGFN tuning

**Purpose.** Screen and resume RGFN architecture/feature configurations while
keeping validation selection separate from final test evaluation. The report
keeps every predefined configuration rather than using a test metric to name
an arm.

**Front door.** `scripts/tune_hourly_detection.py` with `run` or
`resume`.

**Main code.** `src/workflows/tune_hourly_detection.py`,
`resume_hourly_tuning.py`; `src/model/hourly_rgfn_tuning.py`,
`hourly_rgfn_tuning_features.py`, and
`hourly_rgfn_tuning_logistic.py`.

**Tests.** `tests/test_rgfn_tuning.py`,
`tests/test_rgfn_tuning_logistic.py`, and
`tests/test_front_door_scripts.py`.

**Inputs/outputs.** A new tuning run requires the short tensor, baseline split
manifest, calibrated-baseline metrics, prior RGFN metrics, canonical merged
data, and the generated feature matrix. Resume additionally requires saved
tuning checkpoints. Missing prerequisites are reported together before any
tuning output is created.

### 7. Full and partial availability

**Purpose.** Preserve the frozen full-outage definition while additionally
classifying transmitting station-hours with one or more entirely absent sensor
groups as partial outages. Structural gaps are materialised as full outages for
continuous-clock analysis.

**Main code.** `src/availability/build_availability_events.py` and
`build_station_reliability_summary.py`.

**Inputs/outputs.** Reads the published row-state and measurement evidence. The
tracked outputs are `hourly_availability_classification.parquet`,
`partial_outage_events.parquet`, `structural_availability_gaps.csv`,
`availability_report.txt`, and the station reliability summary. The original
2,398 full-outage events and 47 network-wide windows remain unchanged.

**Tests.** `tests/test_availability.py`.

### 8. Station health, forecast, and operational scorecard

**Purpose.** Build a causal 0-100 current-health score, evaluate health at
1/3/6/12/24-hour horizons for transmitting stations, and assemble one current
operational row for each of the 26 stations.

**Front door.** `scripts/build_station_health.py`. The default builds current
health, `--forecast` trains/evaluates the five horizons, and `--scorecard`
assembles a parameterised station snapshot without retraining.

**Main code.** `src/availability/health_score.py`, `health_forecast.py`, and
`operational_scorecard.py`.

**Inputs/outputs.** Current health reads the canonical station-hour data and
the public exact-hour reference cache. Forecast evaluation/model artifacts are
generated below ignored `data/eval/health_forecast/` and
`data/model/health_forecast/`. The tracked score, summary, causal audits,
figures, scorecard, report, and invariants preserve the completed result.
Regenerating the scorecard requires the five selected models produced by
`--forecast`.

**Tests.** `tests/test_availability.py` and
`tests/test_front_door_scripts.py`.

### 9. Incident-risk experiments

**Purpose.** Construct future fault/outage targets on a continuous hourly grid
and evaluate direct and discrete-hazard comparisons at 6, 12, and 24 hours.

**Front door.** `scripts/evaluate_outage_risk.py`

**Main code.** `src/availability/risk_dataset.py`, `risk_model.py`,
`risk_eval.py`, and shared `src/model/binary_metrics.py`.

**Inputs/outputs.** Labels are defined strictly over `(t, t+H]`; partitions are
chronological by timestamp and independently purged by horizon. Saved
comparison artifacts are research evidence and future work, not deployed
forecasting claims, because the acceptance criterion was not met.

**Tests.** `tests/test_outage_risk.py`, `tests/test_availability.py`, and
`tests/test_binary_metrics.py`.

### 10. Report assets

**Purpose.** Produce methodology and results figures for the project report.

**Front door.** `scripts/generate_report_assets.py` with `--set methodology`,
`results`, `july-evaluation`, or `all`.

**Main code.** `src/workflows/build_methodology_figures.py`,
`build_result_figures.py`, `build_july_evaluation_figures.py`, and
`src/rules/result_figures.py`.

**Inputs/outputs.** Methodology figures read the registry and hourly row states.
Some result figures also require preserved historical evidence queues that are
not regenerated by the active public feature-rebuild route. Methodology is
the default public-safe set; results and all preflight their historical inputs
before either figure set writes output.

**Tests.** `tests/test_result_figures.py` and
`tests/test_front_door_scripts.py`.

### 11. July operational replay dashboard

**Purpose.** Replay the frozen July 2026 operational state without loading a
model, retraining, rescoring, or writing under `data/`.

**Front door.** `scripts/run_dashboard.py`

**Main code.** `src/dashboard/replay.py` loads the five frozen dashboard
artifacts, builds causal snapshots, segments consecutive predicted HGB-positive
hours, and prepares station/event evidence.

**Inputs/outputs.** Reads the published July health, forecast, binary-prediction,
statistical-score, spatial-neighbour, and station-registry files. The dashboard
has no output artifact. Network, Station, and Evidence tabs show operational
state rather than model-performance diagnostics.

**Tests.** `tests/test_availability.py` and
`tests/test_front_door_scripts.py`.

## Module index

Package markers are omitted below unless they contain runtime code.

### `src/availability/`

| Module | Role |
|---|---|
| `build_availability_events.py` | Builds frozen full-outage events plus operational full/partial station-hour classes, partial events, and structural-gap evidence. |
| `build_network_outage_windows.py` | Groups concurrent station outages into network windows and classifies them. |
| `build_station_reliability_summary.py` | Generates per-station full/partial outage and sensor-group availability summaries. |
| `health_score.py` | Builds the causal, transparent 0-100 station-health score, progressive outage penalty, audits, reports, and figures. |
| `health_forecast.py` | Builds chronological five-horizon health datasets, baselines, learned comparisons, validation selection, tests, and saved inference wrappers. |
| `operational_scorecard.py` | Joins current health, forecasts, availability, detector evidence, and reliability history into one causal station snapshot. |
| `plot_network_offline_fraction_timeline.py` | Historical availability timeline plotting utility. |
| `plot_station_uptime_bar.py` | Historical station-uptime plotting utility. |
| `risk_dataset.py` | Builds continuous-clock fault/outage targets, causal evidence features, event-history fields, and per-horizon purged partitions. |
| `risk_eval.py` | Provides timestamp splits, metrics, evaluation reports, and shared regression helpers used by health forecasting. |
| `risk_model.py` | Defines direct and discrete-hazard risk comparisons and validation-only operating-point selection. |

### `src/config/`

| Module | Role |
|---|---|
| `paths.py` | Central project/data paths, canonical schema expectations, and the neutral `FIVE_MINUTE_INPUT_DIR` setting. |
| `test_paths.py` | Active pytest data-contract checks; it is intentionally collected from `src/config/`. |

### `src/dashboard/`

| Module | Role |
|---|---|
| `replay.py` | Loads frozen July artifacts, builds station snapshots, segments predicted fault runs, and prepares compact operational evidence. |

### `src/features/`

| Module | Role |
|---|---|
| `build_station_registry.py` | Rebuilds registry presence-rate and status fields. |
| `row_state.py` | Assigns availability states such as warm-up, true outage, online, and padded absence. |
| `run_data_audit.py` | Builds row states, audit/missingness tables, and coverage plots. |

### `src/model/`

| Module | Role |
|---|---|
| `binary_metrics.py` | Shared binary precision, recall, F1, confusion metrics, and maximum-F1 threshold selection. |
| `feature_spec.py` | Feature vocabulary and fault mechanism/component axes. |
| `hourly_baseline.py` | Tensor loading, random/spaced splits, class-weighted detection, and retrospective multi-label reason-code comparison helpers. |
| `hourly_calibration.py` | Validation-only weight/threshold selection for the baseline. |
| `hourly_detection.py` | Core hourly labels, display states, and past-only tensor construction. |
| `hourly_rgfn.py` | RGFN architecture: temporal encoders, rule branch, learned gate, mask validation, and multi-label reason-code heads. |
| `hourly_rgfn_training.py` | RGFN split preparation, scaling, training, checkpoints, detection comparisons, and reason-code evaluation. |
| `hourly_rgfn_tuning.py` | RGFN tuning engine and result tables. |
| `hourly_rgfn_tuning_features.py` | Causal feature augmentations used by tuning arms. |
| `hourly_rgfn_tuning_logistic.py` | Logistic-regression tuning comparison. |

### `src/rules/`

| Module | Role |
|---|---|
| `baselines.py` | Station-specific and pooled median/MAD baseline selection. |
| `channel_handlers.py` | Channel transforms and component mapping. |
| `clustering.py` | Episode feature vectors and HDBSCAN clustering. |
| `config.py` | Detector/reference thresholds, mappings, paths, and label settings. |
| `detectors/robust_zscore.py` | Robust median/MAD detector. |
| `detectors/isolation_forest.py` | One-dimensional Isolation Forest detector. |
| `detectors/rolling_variance.py` | Rolling-variance flatline detector. |
| `episodes.py` | Merges aligned channel events into station episodes. |
| `events.py` | Groups contiguous flagged channel hours into events. |
| `external_features.py` | Leakage-guarded reference residual/offset/solar/fleet features. |
| `external_offsets.py` | Persistent reference-residual offset and solar-ratio run detection. |
| `external_residuals.py` | Aligns station, supplied five-minute, and reference readings into residual diagnostics. |
| `feature_matrix.py` | Joins statistical, reference, spatial, and context features. |
| `labelling.py` | Core multi-label classifier for spike, stuck, statistical, and calibration mechanisms. |
| `layer2_calibration.py` | Sustained calibration-offset analysis and borderline resolution evidence. |
| `physical_limits.py` | Physical-range rule checks. |
| `result_figures.py` | Result-figure data preparation and plotting helpers. |
| `review_queue.py` | Episode review-queue construction. |
| `score.py` | Combined detector score computation. |
| `spatial_offsets.py` | Spatial feature construction and neighbour-present counts. |
| `spatial_residuals.py` | Station-neighbour graph and spatial residual calculations. |
| `statistical_gate.py` | Contextual/external statistical-anomaly promotion gate. |
| `stuck_confirmation.py` | Supplied-five-minute evidence checks for suspected stuck episodes. |

### `src/workflows/`

| Module | Role |
|---|---|
| `build_methodology_figures.py` | Methodology figure generation. |
| `build_result_figures.py` | Results figure generation. |
| `build_july_evaluation_figures.py` | Selected-HGB class distribution, confusion matrix, ROC/PR curves, and July health-forecast figure generation. |
| `rebuild_detection_features.py` | Public analysis orchestration for the full feature rebuild. |
| `resume_hourly_tuning.py` | Resume helper for persisted tuning work. |
| `train_hourly_baseline.py` | Baseline and retrospective reason-code front-door workflow. |
| `train_hourly_calibration.py` | Baseline calibration workflow. |
| `train_hourly_rgfn.py` | RGFN training workflow. |
| `train_hourly_split_comparison.py` | Random-versus-spaced split comparison workflow. |
| `tune_hourly_detection.py` | Tuning front-door workflow. |

## Data map

| Artifact | Status | Producer / consumer |
|---|---|---|
| `data/merged/station_hourly_merged.csv` | Published canonical input | Public workflow entry point; read throughout analysis. |
| `data/merged/station_registry.csv` | Published canonical input | Station metadata, reference fetch, spatial features, figures. |
| `data/labels/episode_labels.csv` | Live tracked label source | Built by the labelling stage; read by hourly dataset construction. |
| `data/labels/calibration_offset_layer2.csv` | Published Layer 2 evidence | Calibration-offset resolution record. |
| `data/processed/hourly_row_states.parquet` | Published availability evidence | Read by outage-risk evaluation and methodology figures. |
| `data/processed/availability_events.parquet` | Published availability evidence | Read by outage-risk evaluation. |
| `data/processed/network_outage_windows.csv` | Published availability evidence | Network-outage context. |
| `data/processed/hourly_availability_classification.parquet` | Published operational availability evidence | Full/partial/online/excluded state for each materialised station-hour. |
| `data/processed/partial_outage_events.parquet` | Published operational availability evidence | Consecutive partial-outage events and absent sensor groups. |
| `data/processed/structural_availability_gaps.csv` | Published gap audit | Four omitted station spans materialised as full outages for continuous-clock calculations. |
| `data/processed/station_health_scores.parquet` and `station_health_summary.csv` | Published current-health evidence | Causal score components, total, band, summaries, and diagnostics. |
| `data/processed/station_operational_scorecard.csv` | Published operational snapshot | One current row per station, with health, forecast, availability, evidence, and reliability fields. |
| `data/eval/health_forecast/` and `data/model/health_forecast/` | Ignored generated artifacts | Created by `build_station_health.py --forecast`; required to regenerate the scorecard. |
| `data/eval/july_2026_health/station_health_scores_through_july.parquet` | Published dashboard input | Causal station-health history through July. |
| `data/eval/july_2026_health_forecast/july_health_forecast_predictions.parquet` | Published dashboard input | Frozen selected-policy forecasts at 1/3/6/12/24 hours. |
| `data/eval/july_2026_scoring/july_binary_predictions.parquet` | Published dashboard input | Frozen selected-HGB probabilities and predictions. |
| `data/eval/july_2026_features/statistical_anomaly_scores.parquet` and `spatial_neighbors.csv` | Published dashboard evidence | Detector margins, external evidence availability, and neighbour context. |
| `data/processed/data_audit_summary.csv` and `missingness_by_variable.csv` | Published audit evidence | Dataset quality summaries. |
| `data/eval/outage_risk_*.csv` and `outage_risk_predictions.parquet` | Retained historical prototype evidence | Pre-correction outputs; not final forecasting claims. The current front door rebuilds corrected labels/splits. |
| `data/external/reference_hourly/` | Ignored generated cache | Created by public reference fetch. |
| `data/external/five_minute_input/` | Separately supplied input | Required for exact external/stuck evidence rebuild. |
| `data/features/feature_matrix.parquet` | Ignored generated artifact | Required to build hourly tensors. |
| `data/hourly_detection/` tensors/models | Ignored generated artifacts | Required for training/tuning reproduction. |

Dated April-June backfill snapshots and retired V1 evaluation artifacts are
intentionally absent from this public snapshot. They are neither current
inputs nor evidence for the retained model path.

## Key concepts

- **Episode:** a contiguous station-level period assembled from aligned channel
  events. Episodes carry one or more mechanism and component labels.
- **Station-hour:** the deployed detection unit. Each hour receives binary
  `fault_hour` when it falls inside a fault episode.
- **Mechanisms:** `statistical_anomaly`, `stuck_flatline`,
  `spike_impossible`, and `calibration_offset`.
- **Components:** anemometer, barometer, light/UV, thermo-hygrometer, rain
  gauge, and wind vane.
- **Fault / benign / clean / excluded:** `fault` is a training positive;
  `benign` is a non-fault hour with at least one detector signal; `clean` is a
  non-fault hour with no detector signal; `excluded` is a borderline-review
  hour and carries no training label.
- **Three evidence layers:** detector/rule evidence creates candidates;
  contextual plus reference/spatial evidence controls statistical promotion;
  Layer 2 examines sustained calibration-offset candidates. Path A uses ERA5
  directional agreement, whereas the externally comparable single-detector
  Path B uses a strong absolute ERA5 residual: it is external discrepancy,
  not same-direction corroboration.
- **Mask mode:** hourly windows use a per-hour mask by default; the
  per-feature mask remains a supported reproducibility option.
- **Random versus spaced split:** the spaced split keeps positive source
  episodes together and is a harder retrospective generalisation check. It is
  not fully purged prospective validation because feature construction uses the
  full historical series.
- **RGFN / Evidence Gate:** the Reliability-Aware Gated Fusion Network combines
  temporal continuous features with rule evidence through a learned gate.
- **Full versus partial outage:** full means no transmission at that station-hour;
  partial means the station transmits while every channel in at least one sensor
  group is absent. The categories are mutually exclusive.
- **Health score:** a causal 0-100 operational summary built from transmission
  reliability, sensor completeness, rule-evidence burden, external-reference
  consistency, and recent stability. Active outages apply a monotonic,
  progressive duration penalty rather than an immediate hard zero.
- **Health forecast scope:** reported 1/3/6/12/24-hour forecasts apply to
  transmitting origins. Full-outage origins are marked not applicable in the
  operational scorecard.
- **Reason codes:** retrospective multi-label mechanism verdicts aggregated at
  reviewed event level. They are explanatory evaluation outputs, not causal
  live-event segmentation.
- **July replay:** a deterministic simulated clock over frozen out-of-time
  outputs. Events are segmented from predicted consecutive fault hours, never
  from reviewed episode identifiers.

## How to run the public workflow

```bash
python -m pip install -r requirements.txt
python -m pytest -q

# Optional: public reference cache
python scripts/fetch_reference_data.py

# Requires supplied complete five-minute files
python scripts/rebuild_detection_features.py --five-min-dir <path-to-complete-five-minute-files>

# Requires regenerated feature matrix and label evidence
python scripts/build_labels.py
python scripts/build_hourly_dataset.py

# Requires generated hourly tensors
python scripts/train_hourly_detection.py baseline
python scripts/train_hourly_detection.py calibration
python scripts/train_hourly_detection.py rgfn
python scripts/train_hourly_detection.py reason-codes
python scripts/tune_hourly_detection.py run

# Current health score; requires regenerated public exact-hour reference data
python scripts/build_station_health.py

# Generates ignored evaluation files and the models needed by --scorecard
python scripts/build_station_health.py --forecast
python scripts/build_station_health.py --scorecard

# Corrected label/split construction; model routes remain future-work evidence
python scripts/evaluate_outage_risk.py --target outage
python scripts/evaluate_outage_risk.py --target fault

# Public-safe report figures
python scripts/generate_report_assets.py --set methodology
python scripts/generate_report_assets.py --set july-evaluation

# Frozen July operational replay
python -m streamlit run scripts/run_dashboard.py
```

Use `--help` for the front doors that expose command-line options. The fixed
labelling workflow is documented in its source and in the pipeline section
above.
