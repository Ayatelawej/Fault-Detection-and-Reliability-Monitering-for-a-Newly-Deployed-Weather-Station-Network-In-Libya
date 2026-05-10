# Phase 2 Investigation Checks

This folder preserves small exploratory checks used during Phase 2 root-cause
analysis. These are not canonical pipeline steps and are not pytest tests.

Run the checks from the project root with:

```bash
python docs/phase2_investigation_checks/run_checks.py
```

The script writes:

- `top_network_windows_geography.txt`
- `network_window_hour_distribution.txt`
- `wu_observation_cadence.txt`
- `midnight_prevalence.txt` checks whether outage-event starts are unusually concentrated around 22-23 UTC and what non-participating stations were doing before the March 8 drop.
- `concurrent_offline.txt` checks how often many active stations were offline at the same hour and whether those hours concentrate by time of day.
- `concurrent_offline_mature.txt` repeats the concurrent offline check after filtering to hours with at least 10 active stations.
- `local_cooccurrence.txt` summarizes pairwise station outage co-occurrence ratios by geographic distance bucket.
- `local_cooccurrence.csv` stores pairwise active-hour, offline-hour, expected co-failure, and co-occurrence ratio metrics.
- `local_cooccurrence_clean.txt` repeats pairwise co-occurrence after excluding hours inside detected network-wide outage windows.
- `local_cooccurrence_clean.csv` stores the cleaned pairwise co-occurrence metrics used by `local_cooccurrence_clean.txt`.
- `local_cooccurrence_event_clean.txt` repeats pairwise co-occurrence after excluding every station-hour belonging to a network outage event.
- `local_cooccurrence_event_clean.csv` stores the event-cleaned pairwise co-occurrence metrics.
- `midnight_cadence.txt` summarizes day-of-week, day-of-month, month, and inter-event gap cadence for midnight network windows.
- `midnight_cadence.csv` stores one row per midnight network window with calendar cadence fields and gap metrics.
- `window_geography.txt` summarizes geographic spread and pairwise station distances for network outage windows.
- `window_geography.csv` stores one row per network window with distance and latitude/longitude spread metrics.
- `window_recovery_patterns.txt` summarizes drop and recovery synchronization across all per-window WU minute-level pulls.
- `window_recovery_patterns.csv` stores one row per window with drop spread, recovery spread, and estimated outage duration metrics.
- `window_durations_hourly.txt` summarizes per-station outage duration and recovery using hourly row states for all network outage windows.
- `window_durations_hourly.csv` stores one row per network window with hourly duration, start spread, and recovery spread metrics.
- `winter_silence.txt` checks active and online station counts during the December 2025 to February 2026 silence in midnight events.
