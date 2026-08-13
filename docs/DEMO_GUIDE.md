# Dashboard Demo Guide

## Start

From the repository root:

```powershell
python -m streamlit run scripts/run_dashboard.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## Five-minute walkthrough

1. Explain that the clock is a deterministic July 2026 replay because direct live-station access is unavailable.
2. On **Network**, show the four mutually exclusive fleet counts and the sortable 26-station table. Point out full and partial outages separately.
3. Press **Play** or move the replay hour. Explain that the dashboard reads frozen output artifacts and performs no training or model selection.
4. On **Station**, choose a station. Show its current status, causal health history, five score components, six sensor-group states, and 1/3/6/12/24-hour health forecasts.
5. For a full outage, explain that the displayed values are continued-outage projections, not learned recovery forecasts.
6. On **Evidence**, choose a predicted event. Show detector, channel, score, frozen threshold, margin, and the compact external/spatial-evidence availability statements.
7. State that displayed events are segmented from consecutive HGB-positive hours and never from ground-truth episode identifiers.

## Key phrases

- “HGB was selected using validation data before July was evaluated.”
- “July is a frozen out-of-time test; no July metric changed the model or threshold.”
- “The dashboard is an operational replay, not a model-performance page.”
- “Current health remains causal during outages; a continued-outage projection assumes no recovery.”

## If something goes wrong

- Stop the app with `Ctrl+C` and restart the command above.
- If port 8501 is occupied, use `--server.port 8502` and open that port instead.
- Do not regenerate data during a demonstration. The dashboard is intentionally read-only.
