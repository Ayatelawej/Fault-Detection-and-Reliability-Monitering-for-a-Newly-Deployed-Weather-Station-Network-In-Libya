# Weather Station Network Fault Detection and Reliability Monitoring

This repository contains a reproducible analysis pipeline for a 26-station
weather network in Libya. The public dataset is frozen through March 2026 and
includes hourly station observations, station metadata, audit outputs,
availability events, statistical anomaly scores, fault episodes, cluster
assignments, and a review queue for assisted labeling.

The private acquisition and merge tooling is intentionally excluded because it
depends on credentials and source-specific operational details. The shared
repository focuses on the auditable research workflow: data quality checks,
availability analysis, fault detection, event construction, clustering,
label-review preparation, external/spatial comparison, forecasting, validation,
and dashboarding.

## Current Status

Completed:

- Dataset audit and station registry, `v0.1`
- Outage and availability engine, `v0.2`
- Statistical anomaly detection with channel events, fault episodes, clustering,
  and review queue generation, `v0.3` code complete with final figures pending

The anomaly workflow reduces 18,467 flagged station-hours into 2,719 fault
episodes, 32 cluster families, and a 412-row review queue. The queue includes
representatives, boundary cases, noise checks, sustained unclustered faults,
hard physical-limit breaches, and suspect-value breaches that require direct
review.

Next work:

- ERA5 reference residuals and nearest-neighbor spatial residuals
- Reviewed labels and validation set construction
- Hybrid detector with temporal modeling, fusion, reason codes, and cause
  classification
- Short-horizon risk prediction at 6, 12, and 24 hours
- Final validation against outages and reviewed labels
- Streamlit dashboard

The FT0360 device fault-injection extension is deferred as optional work.

## Repository Layout

- `data/merged/`: frozen public station-hour dataset and station registry
- `data/processed/`: reproducible pipeline outputs that support analysis and
  review
- `data/external/openmeteo/`: cached ERA5 reference inputs for upcoming
  residual analysis
- `src/features/`: data audit and row-state classification
- `src/availability/`: outage events, network outage windows, and reliability
  summaries
- `src/rules/`: statistical anomaly scoring, event building, episode merging,
  clustering, and review queue construction
- `src/references/`: external reference-data fetchers
- `scripts/`: reusable diagnostics, public-dataset utilities, and figure tools
- `docs/`: written audit notes and investigation checks
- `outputs/figures/`: generated figures used in analysis and reporting
- `tests/`: regression tests for the public workflow

## Key Outputs

- `data/processed/data_audit_summary.csv`
- `data/processed/hourly_row_states.parquet`
- `data/processed/missingness_by_variable.csv`
- `data/processed/availability_events.parquet`
- `data/processed/network_outage_windows.csv`
- `data/processed/station_reliability_summary.csv`
- `data/processed/review_queue.csv`
- `outputs/figures/station_coverage_timeline.png`
- `outputs/figures/missingness_heatmap.png`
- `outputs/figures/station_uptime_bar.png`
- `outputs/figures/network_offline_fraction_timeline.png`

Intermediate anomaly artifacts such as per-hour scores, channel events, fault
episodes, and cluster assignments are regenerated locally and ignored by git
unless a specific sharing package needs them.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the regression tests:

```bash
pytest
```

Regenerate the main audit and availability outputs:

```bash
python -m src.features.build_station_registry
python -m src.features.run_data_audit
python -m src.availability.build_availability_events
python -m src.availability.build_network_outage_windows
python -m src.availability.build_station_reliability_summary
```

Run the statistical-anomaly diagnostic workflow:

```bash
python scripts/run_anomaly_diagnostics.py
```
