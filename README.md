# Weather Station Network Fault Detection

## Phase 1 status

Phase 1 is complete. The project now has a station registry, canonical row-state
classification, station-level audit summary, variable missingness summary, and
two audit figures.

New Phase 1 artifacts:

- `data/merged/station_registry.csv`
- `data/processed/hourly_row_states.parquet`
- `data/processed/data_audit_summary.csv`
- `data/processed/missingness_by_variable.csv`
- `outputs/figures/station_coverage_timeline.png`
- `outputs/figures/missingness_heatmap.png`
- `docs/phase1_data_audit.md`

This project includes pre-merged data for 26 weather stations in a weather station network all across libya.
The merge pipeline and data aquisition pipelines are kept private because they include some private API keys that cannot be shared with the public, but this data was acquired to be open source and free for all.
This repo focuses on availability, analysis, fault detection, external/spatial comparison, event construction, risk prediction, validation and dashboarding.
The frozen dataset is through march 2026, but the stations continue to collect data as time goes on.
