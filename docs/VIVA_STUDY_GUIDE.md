# Viva Study Guide

## Project in one sentence

The project turns a newly assembled 26-station weather dataset into an auditable reliability system covering availability, full and partial outages, fault detection, causal health, multi-horizon health forecasting, an operational scorecard, and a July replay dashboard.

## Numbers to know

| Topic | Result |
|---|---|
| Development dataset | 166,017 station-hours, through 30 June 2026 |
| Label catalogue | 4,799 episodes: 1,121 fault, 3,533 benign, 145 borderline |
| Full outages | 2,398 events; 47 coordinated network windows |
| Partial outages | 803 events; 11,158 transmitting station-hours |
| Selected development HGB | precision 0.899, recall 0.906, F1 0.902, accuracy 0.986 |
| Independent July HGB | precision 0.714, recall 0.813, F1 0.761, accuracy 0.936 |
| July discrimination | AUROC 0.977; AUPRC 0.865 |
| July health-band accuracy | 0.940 at 1 hour; 0.723 at 24 hours |
| July health MAE | 1.378 at 1 hour; 6.619 at 24 hours |

## Method decisions to defend

### Why HGB was deployed

HGB was selected on validation performance. It handles engineered nonlinear rule evidence and missingness effectively on the available dataset. The purpose-built Reliability-Aware Gated Fusion Network (RGFN) remained close but did not exceed it after tuning. The limited number of independent fault episodes reduces the advantage of a more complex temporal network.

### Why two development partitions exist

The random partition provides the operational model-selection comparison. The episode-grouped spaced partition is a harder retrospective sensitivity check that keeps positive episodes together; it is not claimed as prospective validation. July supplies the true later-period test.

### Why July precision fell

July introduced distribution shift. Temperature/humidity detector evidence appears in 384 of 553 false-positive hours, so thermal conditions are strongly associated with the increase, but this does not prove extreme heat caused every false alert. Ranking remained strong (AUROC 0.977 and AUPRC 0.865), while the frozen operating threshold became less precise.

### Why accuracy is not enough

Fault hours are the minority class. Precision, recall, F1, the confusion matrix, and especially AUPRC show minority-class behaviour. AUROC measures ranking across thresholds; AUPRC is more sensitive to false positives under imbalance.

### What reason codes mean

Mechanism reason codes were evaluated retrospectively during development. They are not claimed as an independently validated July subsystem. Live dashboard events are formed from predicted consecutive fault hours, never ground-truth episode IDs.

### What health forecasting means

The causal health score is a transparent 0–100 combination of five components. Forecast models predict future health for transmitting origins at 1, 3, 6, 12, and 24 hours. Active full outages instead receive an explicitly labelled deterministic continued-outage projection that assumes no recovery.

## Limitations to state directly

- Controlled hardware fault injection was planned but unavailable because a station could not be retained for testing.
- July has only 25 stations because IJANZO4 has no July observations.
- The July false-positive increase shows a need for additional annual cycles and seasonal adaptation.
- Mechanism/component support is uneven, especially for rare labels.
- The dashboard is a July replay until live Roaya API access is available.

## Architecture reading order

1. `REPO_MAP.md`
2. `scripts/run_dashboard.py`
3. `src/dashboard/replay.py`
4. `scripts/train_hourly_detection.py`
5. `src/model/hourly_baseline.py` and `src/model/hourly_rgfn.py`
6. `src/availability/health_score.py`, `health_forecast.py`, and `operational_scorecard.py`
7. Relevant tests for each module
