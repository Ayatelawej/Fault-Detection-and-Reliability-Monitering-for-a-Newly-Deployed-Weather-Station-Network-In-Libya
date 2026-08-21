# Two-to-seven-day health forecast experiment

## Purpose

This experiment was added after supervisor feedback asking how station-health forecasting behaves over a full week. It extends the existing 1, 3, 6, 12, and 24-hour evaluation to exact horizons of 48, 72, 96, 120, 144, and 168 hours. It does not replace or modify the deployed short-horizon models.

The reported outcome is exact four-band health accuracy for transmitting forecast origins. The bands are Healthy, Watch, Degraded, and Critical.

## Evaluation discipline

- Targets use exact continuous-clock horizons.
- Each horizon has its own chronological timestamp split and horizon-width boundary purge.
- Model family, feature set, recency weighting, tree iterations, and residual-correction alpha are selected using validation only.
- The selected configuration is refitted on train plus validation and evaluated once on test.
- The original 1/3/6/12/24-hour models and artifacts remain untouched.
- Full diagnostic metrics, confusion matrices, selections, predictions, models, and causality checks are retained locally below `data/eval/health_forecast_long_horizon/` and `data/model/health_forecast_long_horizon/`. These large generated directories remain gitignored.

## Experiment 1: existing feature representations

The first formal run used only the existing `core` and historical `full_engineered` representations. The exact-band test accuracy was:

| Forecast horizon | Accuracy |
|---|---:|
| Day 2 (48 h) | 62.16% |
| Day 3 (72 h) | 59.69% |
| Day 4 (96 h) | 59.28% |
| Day 5 (120 h) | 59.43% |
| Day 6 (144 h) | 57.28% |
| Day 7 (168 h) | 59.20% |

## Experiment 2: long-horizon features

The second formal run added an explicit `long_horizon` representation containing causal 7/14/30-day health summaries, health slopes, missing/transmitting fractions, sensor-group presence fractions, time spent in each health band, current band duration, and the calendar position of the forecast target. The original feature sets remained candidates, so validation could reject the new representation.

| Forecast horizon | Existing features | Validation-selected extended run | Change |
|---|---:|---:|---:|
| Day 2 (48 h) | 62.16% | 62.16% | 0.00 pp |
| Day 3 (72 h) | 59.69% | 59.98% | +0.29 pp |
| Day 4 (96 h) | 59.28% | 59.28% | 0.00 pp |
| Day 5 (120 h) | 59.43% | 59.43% | 0.00 pp |
| Day 6 (144 h) | 57.28% | 57.28% | 0.00 pp |
| Day 7 (168 h) | 59.20% | 59.20% | 0.00 pp |

Validation selected `long_horizon` only at 72 hours. It retained `core` at 48 and 96 hours and `full_engineered` at 120, 144, and 168 hours. The new representation therefore produced only a small Day-3 improvement and did not improve the one-week endpoint.

## Diagnostic: accuracy-oriented direct classifier

A separate read-only diagnostic tested an unweighted HGB classifier trained directly on the future health band, with its feature representation selected by validation accuracy. It produced 57.74%, 60.20%, 57.44%, 57.88%, 58.30%, and 45.72% test accuracy at Days 2–7. Because this was worse overall than the continuous-score forecast, it was not integrated as a delivered model.

## Conclusion

The best validated formulation remains the continuous health-score forecaster converted to the four operational bands. Adding slow multi-day summaries does not materially improve 2-to-7-day accuracy. The result indicates that the principal limitation is not a missing rolling feature set: future weather, future faults, outages, recoveries, and maintenance actions are not known at the forecast origin, so uncertainty grows materially beyond 24 hours.

The one-week result should be described as an additional post-development horizon study. The established 1/3/6/12/24-hour forecast remains the operational system.

## Reproduction

```powershell
python scripts/build_station_health.py --long-horizon-forecast --long-horizon-features current
python scripts/build_station_health.py --long-horizon-forecast --long-horizon-features extended
```

The compact tracked result table is `data/report/health_forecast_2_to_7_day_accuracy.csv`.
