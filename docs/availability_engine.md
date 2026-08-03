# Availability Engine

## Status: June 2026 availability evidence

The canonical availability artifacts are current through 2026-06-30. The
separate outage-risk model remains a preliminary prototype and is not a final
forecasting result.

## What the engine produces

### Frozen full-outage series

`data/processed/availability_events.parquet` contains 2,398 full-outage events
across 26 stations. A full outage is an active station-hour whose row state is
`true_outage_candidate`: the station transmitted no data during an expected
period. Its event count, network-window count, and duration statistics are
preserved as the historical availability baseline.

The current full-outage total is 34,406 observed station-hours. Event duration
has mean 14.35 hours, median 4 hours, and maximum 814 hours. Event classes are
1,968 local, 267 network-midnight, and 163 network-other.

### Network-wide windows

`data/processed/network_outage_windows.csv` contains 47 coordinated windows.
Windows are detected from full-outage event starts within one hour across at
least five stations. Partial outages never enter this calculation.

### Operational full/partial classification

`data/processed/hourly_availability_classification.parquet` records each
observed station-hour plus materialized structural station-time gaps:

`station_id, hour_utc, availability_scope, availability_class,
absent_sensor_groups, is_transmitting, row_state, source_kind`

The active operational classes are:

- `full_outage`: the frozen full-outage rule.
- `partial_outage`: the station transmitted, but every channel in at least one
  declared sensor group was absent.
- `online`: the station transmitted and no declared group was wholly absent.

`excluded` denotes terminal padded rows after a station's final transmission;
they are not evidence that a station was online or in outage. This is necessary
to avoid distorting uptime or the frozen full-outage series.

The six sensor groups are anemometer, barometer, light/UV, rain gauge,
thermo-hygrometer, and wind vane. Derived dew-point, wind-chill, and heat-index
columns are deliberately excluded from this rule.

`data/processed/partial_outage_events.parquet` merges consecutive partial
hours per station and carries the union of absent sensor groups. It is a
rule-derived operational indicator, not a supervised classifier: there is no
partial-outage ground truth, so precision, recall, and F1 are not applicable.

### Structural station-time gaps

`data/processed/structural_availability_gaps.csv` records holes in the current
station-hour key grid. The June dataset contains four gaps over one hour,
totalling 1,454 omitted station-hours, including a 481-hour timestamp span.
These hours are materialized as `full_outage` in the operational classification
table so they are visible to monitoring. They do not alter the frozen legacy
event table or its duration statistics.

### Per-station summary and report

`data/processed/station_reliability_summary.csv` is regenerated from the
June-inclusive data. It contains uptime, full- and partial-outage counts and
hours, dynamic data-end recency, and each sensor group's availability over
transmitting hours.

`data/processed/availability_report.txt` records the current channel mapping,
classification accounting, structural-gap count, partial-event duration table,
and network-wide sensor-group availability table.

### Figures

`outputs/figures/station_uptime_bar.png` and
`outputs/figures/network_offline_fraction_timeline.png` remain May-era
historical figures. They are not June-current report evidence until they are
regenerated from the current summary.

## Tests

`tests/test_availability.py` locks the 2,398 legacy full-outage events and 47
network windows, checks the six-group mapping, exercises full/partial/online
classification on synthetic data, verifies partial-event aggregation, and
accounts explicitly for structural gaps.
