# Weather Station Network Fault Detection and Reliability Monitoring

This repository contains the publishable analysis code and evidence for an
hour-level fault-detection and reliability-monitoring study of a 26-station
personal weather-station network in Libya. The published canonical dataset is
frozen through June 2026.

The live detection unit is a **station-hour**. The system builds reproducible
episode labels, expands them into hour-level fault/not-fault targets, trains a
gradient-boosted baseline and a Reliability-Aware Gated Fusion Network (RGFN),
and evaluates retrospective event-level mechanism reason codes. Its reliability
path adds full/partial outage classification, a causal 0-100 station-health
score, transmitting-station health forecasts at 1/3/6/12/24 hours, and a
combined 26-station operational scorecard.

The final system also includes a read-only Streamlit dashboard that replays the
independent July 2026 evaluation. The operational HGB model and all thresholds
were selected before July was scored. On 13,565 eligible July station-hours it
achieved 0.714 precision, 0.813 recall, 0.761 F1, 0.936 accuracy, 0.977 AUROC,
and 0.865 AUPRC without refitting.

The separate incident-risk experiments now use continuous clock-hour labels and
horizon-purged timestamp partitions. They remain future-work evidence rather
than deployable forecasting claims because their held-out precision, recall,
and F1 do not jointly meet the project's acceptance criterion.

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
- Current full/partial availability evidence, health-score artifacts, health
  forecast figures, and the latest operational scorecard.
- Frozen July binary predictions, health forecasts, detector evidence, selected
  HGB evaluation figures, and the read-only replay dashboard.
- Corrected incident-risk construction and comparison code, retained as
  future-work evidence rather than a delivered predictor.
- Analysis, labelling, modelling, evaluation, and report-asset code.
- Regression tests and architecture documentation for the live system.

## Pipeline

| Stage | Front door | Purpose |
|---|---|---|
| Public reference data | `scripts/fetch_reference_data.py` | Fetch/cache public ERA5/Open-Meteo reference observations. |
| Detection features | `scripts/rebuild_detection_features.py` | Rebuild statistical, reference, spatial, and feature-matrix evidence from the published dataset and separately supplied five-minute inputs. |
| Episode labels | `scripts/build_labels.py` | Apply the reproducible multi-label rules and write the live episode labels. |
| Hour-level dataset | `scripts/build_hourly_dataset.py` | Expand episode labels into past-only short and long station-hour tensors. |
| Model training and retrospective reason codes | `scripts/train_hourly_detection.py` | Train the baseline or RGFN, run fixed comparisons, and evaluate event-level mechanism reason codes. |
| RGFN tuning | `scripts/tune_hourly_detection.py` | Run or resume RGFN architecture and feature experiments. |
| Availability classification | `src/availability/build_availability_events.py` | Define full outages, partial sensor-group outages, structural gaps, and reliability summaries. |
| Station health, forecast, and scorecard | `scripts/build_station_health.py` | Build causal health scores; use `--forecast` for five horizons and `--scorecard` for the current station snapshot. |
| Incident-risk experiments | `scripts/evaluate_outage_risk.py` | Build corrected 6/12/24-hour fault/outage targets and run future-work comparisons. |
| Report assets | `scripts/generate_report_assets.py` | Generate methodology and result figures. |
| July replay dashboard | `scripts/run_dashboard.py` | Display frozen July network, station-health, forecast, and detector-evidence artifacts without inference or writes. |

See [REPO_MAP.md](REPO_MAP.md) for the detailed data flow, modules, inputs,
outputs, and test coverage.

## Quick start

Install dependencies and run the regression suite:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The incident-risk front door defaults to corrected label/split construction.
Its model routes are research comparisons, not final operational claims:

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

The current-health score is independently reproducible from the published
canonical data and a regenerated public reference cache:

```bash
python scripts/build_station_health.py
```

The forecast comparison and scorecard use the same front door:

```bash
python scripts/build_station_health.py --forecast
python scripts/build_station_health.py --scorecard
```

`--forecast` generates ignored evaluation files and five selected model bundles
under `data/eval/health_forecast/` and `data/model/health_forecast/`.
`--scorecard` requires those generated models. The tracked scorecard CSV,
report, invariants, and causal audit are published so the completed snapshot can
be inspected even when the ignored model bundles are not present in a clone.

Run the frozen July operational replay:

```bash
python -m streamlit run scripts/run_dashboard.py
```

The dashboard has Network, Station, and Evidence tabs. It segments events from
consecutive HGB-positive hours and never reads ground-truth episode IDs. Active
full outages receive clearly labelled continued-outage projections rather than
learned recovery forecasts.

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

Reason-code evaluation is retrospective and event-level. It assigns one or
more fault-mechanism codes after grouping by reviewed event identity; it is not
presented as a causal live-dashboard diagnosis. Component localisation remains
evaluated development work rather than a claimed delivered output.

## Repository structure

- `data/merged/` - published canonical station-hour dataset and registry.
- `data/labels/` - live episode labels and Layer 2 calibration evidence.
- `data/processed/` - published audit, availability, health, and scorecard evidence.
- `data/eval/` - retained research evidence plus the frozen July artifacts used
  by the read-only dashboard.
- `src/` - analysis, labelling, feature, model, reliability, and report code.
- `scripts/` - the ten public pipeline front doors.
- `tests/` - regression tests for the retained public system.

The final report is available in both Word and PDF form under `docs/report/`.
`docs/DEMO_GUIDE.md`, `docs/VIVA_STUDY_GUIDE.md`, and
`docs/SUBMISSION_CHECKLIST.md` provide a concise handoff.

## Licence and data use

Code is released under the repository licence. The included dataset and derived
artifacts are publishable for this study. Please cite the associated project
report when using its results.
