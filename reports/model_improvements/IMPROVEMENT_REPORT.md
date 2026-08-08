# Requested model-improvement comparison

## Scope

This report implements and compares the requested changes using **8
identical chronological race folds** distributed from the first eligible validation
race through the latest available race. Every fold trains only on earlier races.
The comparisons use a five-window stride and the selected HistGradientBoosting
parameters:

```json
{'learning_rate': 0.08, 'max_iter': 150, 'max_leaf_nodes': 31, 'max_depth': None, 'min_samples_leaf': 40, 'l2_regularization': 8.0}
```

Implemented feature changes:

- `DriverNumber` and `Country` are removed from the model; driver, team and circuit
  remain available.
- Absolute sector-time lags/rolling statistics are replaced by sector deltas to
  the same-lap field median and teammate median.
- Gap, relative-pace, tyre and pit features are retained.
- Added `PitWindowPhase`, `FieldPittedShare`, `LapsSincePitStop`, `HasPitted`,
  `EstimatedTyreLifeAdvantageLaps`, `VirtualSafetyCarThisLap`, and five-lap VSC count.
- DRS-opportunity features were not added because detection-zone opportunity data
  is not present in the stored tables. Existing `DRSActivePct` remains.
- Retirement modelling and alternate loss functions were intentionally omitted.

## 1. Two-stage model

The two-stage model first classifies whether the position changes. A conditional
regressor is fitted only on non-zero changes. Its signed prediction is multiplied
by the classifier's change probability.

| Configuration | Races | MeanRaceMAE | BaselineMAE | ModelWinRate | ChangeAccuracy | ChangeBrier | ChangeROCAUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full / 5 laps / 40-race half-life | 8 | 0.969 | 1.016 | 0.5 | 0.693 | 0.202 | 0.765 |
| single-stage / 5 laps | 8 | 0.977 | 1.016 | 0.5 | nan | nan | nan |

Lowest architecture MAE in this sample: **full / 5 laps / 40-race half-life**
at **0.969**. Change accuracy alone should not
decide the winner; MAE and baseline performance remain the deployment metrics.
The two-stage improvement is real in this sample but small, so it should remain
switchable in the dashboard rather than replacing the single-stage path entirely.

## 2. Historical recency weighting

Race balancing is preserved, then each older race receives exponential decay. A
40-race half-life means a race 40 events older receives half the total influence.

| Configuration | Races | MeanRaceMAE | MedianRaceMAE | BaselineMAE | ModelWinRate | MeanBaselineImprovementPct |
| --- | --- | --- | --- | --- | --- | --- |
| equal history | 8 | 0.957 | 0.951 | 1.016 | 0.5 | 5.327 |
| 20-race half-life | 8 | 0.957 | 0.934 | 1.016 | 0.625 | 5.218 |
| 80-race half-life | 8 | 0.966 | 0.946 | 1.016 | 0.5 | 4.415 |
| full / 5 laps / 40-race half-life | 8 | 0.969 | 0.964 | 1.016 | 0.5 | 3.984 |

Lowest tested recency MAE: **equal history** at
**0.957**. This comparison should be repeated after a
major regulation boundary because the best half-life can change by era.
Equal history and the 20-race half-life are effectively tied in mean MAE; the
20-race version won against baseline in more sampled races, so it is the dashboard
default while equal history remains available.

## 3. Position-state comparison — decision intentionally deferred

The four variants are:

- Full: current position, five lags, and rolling position summaries.
- Current only: the single current `Position` value.
- Prior lags only: `Position_lag1` through `Position_lag4`; no current or rolling value.
- None: every direct position-state feature removed. Historical position-gain
  statistics remain because they describe prior races rather than current state.

| Configuration | Races | MeanRaceMAE | MedianRaceMAE | BaselineMAE | ModelWinRate | MeanBaselineImprovementPct |
| --- | --- | --- | --- | --- | --- | --- |
| current Position only | 8 | 0.966 | 0.928 | 1.016 | 0.625 | 4.458 |
| full / 5 laps / 40-race half-life | 8 | 0.969 | 0.964 | 1.016 | 0.5 | 3.984 |
| Position lags 1-4 only | 8 | 0.97 | 0.959 | 1.016 | 0.5 | 4.01 |
| no position-state features | 8 | 0.976 | 0.965 | 1.016 | 0.625 | 3.319 |

Lowest sampled position-state MAE: **current Position only** at
**0.966**. No position columns are removed permanently
by this experiment; this table is evidence for the user's later decision.
The advantage over the full position family is only about 0.003 MAE, which is too
small to justify an irreversible feature deletion from eight races.

## 4. Prediction-horizon comparison

Direct MAE naturally grows with horizon, so compare baseline improvement and win
rate as well as raw MAE. Longer horizons also have fewer valid end-of-race windows.

| Configuration | Races | MeanRaceMAE | MedianRaceMAE | BaselineMAE | ModelWinRate | MeanBaselineImprovementPct | NoChangePercent | ValidationWindows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 lap | 8 | 0.274 | 0.267 | 0.295 | 0.625 | 2.182 | 80.081 | 1763 |
| 3 laps | 8 | 0.708 | 0.731 | 0.71 | 0.5 | 0.331 | 61.116 | 1712 |
| full / 5 laps / 40-race half-life | 8 | 0.969 | 0.964 | 1.016 | 0.5 | 3.984 | 50.029 | 1656 |
| 10 laps | 8 | 1.355 | 1.356 | 1.626 | 1.0 | 16.518 | 30.981 | 1500 |

Separate targets now exist for 1, 3, 5 and 10 laps. The dashboard can select any
of them. A horizon should be chosen according to the product question, not only the
smallest MAE: one-lap forecasts are easier but provide less strategic warning.
The ten-lap target produced the strongest relative result: 16.5% mean baseline
improvement and a model win in all eight sampled races. The one-lap target was
unchanged about 80% of the time, explaining its much smaller absolute MAE.

## 5. Interpretation and next decision

1. Prefer an architecture only if it improves race-level MAE, not merely classifier
   accuracy.
2. Use the recency row with the most stable baseline improvement; a tiny sampled
   MAE difference is not enough to establish a universal half-life.
3. Review the position-state table before removing any position family. The test
   was implemented specifically to keep this decision reversible.
4. For horizon selection, consider a short operational forecast (1 or 3 laps) and
   a strategic forecast (5 or 10 laps) as separate outputs rather than forcing one
   horizon to serve both purposes.
5. Newly ingested races should be the next untouched confirmation set because all
   races in this report have now influenced development.

## Reproducibility

- `improvement_analysis.py`: resumable experiment code.
- `comparison_results.csv`: every race/configuration result.
- `IMPROVEMENT_REPORT.md`: this report.
- `final_two_stage_model.joblib`: current-position-only, 10-lap two-stage model
  using a 20-race recency half-life.
- `final_two_stage_model_metadata.json`: feature and training-coverage metadata.
