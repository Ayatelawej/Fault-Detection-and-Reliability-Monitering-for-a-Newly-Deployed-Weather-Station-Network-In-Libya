# Availability Engine

## Status: June 2026 availability evidence

The canonical availability artifacts below are current through 2026-06-30.
The separate outage-risk model remains a preliminary prototype and is not a
final forecasting result.

## What this module produces

### Tier 1: events
`data/processed/availability_events.parquet`
2,398 outage events across 26 stations. Schema:
`event_id, station_id, start_utc, end_utc, duration_hours, outage_class`

Outage class taxonomy:
- `local`: not part of any detected coordinated network outage (1,968 events)
- `network_midnight`: part of a coordinated outage starting at hour 22
  or 23 UTC (267 events)
- `network_other`: part of a coordinated outage starting at any other
  hour (163 events)

### Tier 2: network-wide windows
`data/processed/network_outage_windows.csv`
47 detected coordinated network outage windows. Schema:
`window_id, window_start_utc, window_end_utc, station_count, station_ids,
outage_class, backfill_start_utc`

Windows are detected by clustering events that start within 1 hour of
each other across at least 5 stations. 22 windows are classified as
network_midnight (start hour 22 or 23 UTC), 25 as network_other.

### Tier 3: per-station reliability summary
`data/processed/station_reliability_summary.csv`
Per-station rollup with uptime, event counts by class, last outage
timestamp, days-since-last-outage-at-freeze. This file is a May-era historical
summary and must be regenerated before it is used as June-current evidence.

### Figures
- `outputs/figures/station_uptime_bar.png`: per-station uptime bar chart
- `outputs/figures/network_offline_fraction_timeline.png`: network-wide
  offline fraction over time, with network-wide windows highlighted

Both availability figures are May-era historical figures, not June-current
report evidence.

## Historical investigation findings

The following historical checks are useful provenance, but were not recomputed
after the June extension:
- Sub-minute drop synchronization confirmed for tight-sync midnight
  events (medians 23-59 seconds across 9 events)
- Median per-station outage duration in midnight windows: 24 hours
- Median per-station outage duration in non-midnight windows: 2 hours
- 90-day silence in midnight events Dec 2025 - Feb 2026, real (not a
  detection artifact)
- Local infrastructure clustering observed in some city-mate pairs
  (e.g., IJANZO2 <-> IJANZO3, 13 km apart, 5.6x co-occurrence ratio)
- Cause of midnight phenomenon not externally identifiable; consistent
  with daily upstream processing at the UTC-day boundary

## Tests

`tests/test_availability.py` verifies event counts, schema,
duration semantics, and outage_class assignments against the windows table.
