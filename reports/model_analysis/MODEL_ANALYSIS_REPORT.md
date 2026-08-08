# HistGradientBoosting F1 model analysis

## Executive summary

This experiment used **35,346 stride-5 prediction windows** from
**182 races**. The first 40 races formed the initial history;
the model was then freshly trained and validated chronologically on every remaining
race (**142 race-level folds**). No validation-race row was present in
its training set.

The selected configuration was **strong_regularization**:

```json
{
  "learning_rate": 0.08,
  "max_iter": 150,
  "max_leaf_nodes": 31,
  "max_depth": null,
  "min_samples_leaf": 40,
  "l2_regularization": 8.0
}
```

Across all evaluated races, mean race MAE was **1.093**
positions, compared with **1.098** for predicting
no position change. The model beat that baseline in **50.0%** of
races. On the later 2025-2026 evidence, mean race MAE was
**1.070** versus baseline **1.053**.

Important caution: 2018-2024 folds were used for model development and parameter
selection. The 2025-2026 section is the cleaner estimate of forward generalization.
No test is perfectly permanent once its result has been repeatedly inspected.

## 1. Dataset and validation design

- Feature window: five completed laps; target: position change five laps later.
- Window stride: 5, reducing four-lap overlap between neighbouring samples.
- Initial history: 40 races.
- Validation: one complete race at a time, with only earlier races in training.
- Training weights: each training race contributes equal total weight.
- Missing numeric values: handled natively by HistGradientBoosting.
- Completely empty early-fold features: removed using training data only.
- Scaling: not used because tree splits do not require standardized units.
- Training time represented in checkpoints: 74.4 minutes.

The dataset contains no missing target values. **10**
features exceed 10% missingness and **0**
exceed 50%.

Top missing features:

| Feature | MissingCount | MissingPercent |
| --- | --- | --- |
| DriverCircuitAvgQualifyingPosition | 10965 | 31.022 |
| DriverCircuitAvgPositionGain | 10921 | 30.897 |
| DriverCircuitLastFinish | 10768 | 30.465 |
| DriverCircuitAvgFinish | 10757 | 30.433 |
| DriverCircuitBestFinish | 10757 | 30.433 |
| TeamCircuitAvgQualifyingPosition | 10460 | 29.593 |
| TeamCircuitAvgFinish | 10261 | 29.03 |
| Sector1Seconds_std5 | 3647 | 10.318 |
| Sector1Seconds_mean5 | 3647 | 10.318 |
| Sector1Seconds_trend5 | 3547 | 10.035 |
| Sector1Seconds_lag4 | 3510 | 9.93 |
| GapToCarAheadSeconds_std5 | 2279 | 6.448 |
| GapToCarAheadSeconds_mean5 | 2279 | 6.448 |
| GapToCarAheadSeconds_trend5 | 2224 | 6.292 |
| GapToCarAheadSeconds_lag2 | 1942 | 5.494 |

## 2. Hyperparameter comparison

The same ten chronological development folds through 2024 were used for every
candidate. Mean race MAE was the primary selection metric; lower is better.

| Configuration | MeanRaceMAE | MedianRaceMAE | MeanRMSE | BaselineImprovementPct |
| --- | --- | --- | --- | --- |
| strong_regularization | 1.092 | 1.057 | 1.687 | -3.032 |
| default | 1.1 | 1.052 | 1.707 | -3.517 |
| small_trees | 1.102 | 1.033 | 1.709 | -3.893 |
| slower_regularized | 1.107 | 1.063 | 1.712 | -4.608 |
| shallow | 1.11 | 1.082 | 1.708 | -4.635 |
| fast_boosting | 1.113 | 1.096 | 1.724 | -5.132 |
| high_capacity | 1.116 | 1.124 | 1.719 | -5.843 |
| large_leaves | 1.119 | 1.117 | 1.712 | -5.718 |

## 3. Walk-forward accuracy by season

| Year | Races | MeanRaceMAE | MedianRaceMAE | MeanRMSE | BaselineMAE | ImprovementPct |
| --- | --- | --- | --- | --- | --- | --- |
| 2019 | 1 | 1.043 | 1.043 | 1.646 | 1.07 | 2.513 |
| 2020 | 17 | 1.135 | 1.084 | 1.803 | 1.135 | -2.432 |
| 2021 | 21 | 0.98 | 0.94 | 1.521 | 0.938 | -13.885 |
| 2022 | 22 | 1.114 | 1.084 | 1.707 | 1.137 | -6.987 |
| 2023 | 22 | 1.15 | 1.071 | 1.77 | 1.219 | -0.88 |
| 2024 | 24 | 1.124 | 1.162 | 1.689 | 1.13 | -16.745 |
| 2025 | 24 | 1.17 | 1.159 | 1.815 | 1.151 | -5.877 |
| 2026 | 11 | 0.853 | 0.89 | 1.293 | 0.837 | -7.154 |

The model beat the no-change baseline in **50.0%** of evaluated
races. This comparison matters because **50.8%** of individual
five-lap targets contain no position change.

At window level, rounded predictions exactly matched the target **42.6%**
of the time and continuous predictions were within one position **65.9%**
of the target. Mean signed prediction bias was **-0.028** positions. On the
harder windows where the true change was at least three positions, MAE was
**3.400**.

### Ten hardest validation races

| Year | RoundNumber | EventName | MAE | Baseline MAE |
| --- | --- | --- | --- | --- |
| 2023 | 17 | Qatar Grand Prix | 2.37 | 2.785 |
| 2020 | 1 | Austrian Grand Prix | 2.102 | 2.082 |
| 2024 | 22 | Las Vegas Grand Prix | 1.905 | 2.446 |
| 2022 | 19 | United States Grand Prix | 1.642 | 1.891 |
| 2023 | 22 | Abu Dhabi Grand Prix | 1.635 | 2.15 |
| 2023 | 21 | Las Vegas Grand Prix | 1.606 | 1.858 |
| 2025 | 21 | São Paulo Grand Prix | 1.594 | 1.762 |
| 2022 | 18 | Japanese Grand Prix | 1.538 | 1.472 |
| 2022 | 21 | São Paulo Grand Prix | 1.518 | 1.774 |
| 2020 | 8 | Italian Grand Prix | 1.518 | 1.448 |

### Ten easiest validation races

| Year | RoundNumber | EventName | MAE | Baseline MAE |
| --- | --- | --- | --- | --- |
| 2024 | 8 | Monaco Grand Prix | 0.46 | 0.107 |
| 2021 | 5 | Monaco Grand Prix | 0.493 | 0.257 |
| 2021 | 18 | Mexico City Grand Prix | 0.598 | 0.48 |
| 2021 | 11 | Hungarian Grand Prix | 0.647 | 0.461 |
| 2022 | 4 | Emilia Romagna Grand Prix | 0.648 | 0.293 |
| 2025 | 6 | Miami Grand Prix | 0.674 | 0.537 |
| 2020 | 17 | Abu Dhabi Grand Prix | 0.697 | 0.663 |
| 2026 | 6 | Monaco Grand Prix | 0.706 | 0.455 |
| 2023 | 3 | Australian Grand Prix | 0.711 | 0.655 |
| 2021 | 13 | Dutch Grand Prix | 0.717 | 0.611 |

## 4. Individual permutation importance

Importance is the increase in validation MAE after one feature is shuffled.
Positive values indicate useful unique information. Near-zero values can also
occur when correlated features substitute for one another.

Top 25 features across representative unseen races:

| Feature | MeanImportance | MedianImportance | ImportanceStdAcrossRaces | PositiveRaceShare | Races |
| --- | --- | --- | --- | --- | --- |
| Position | 0.227 | 0.241 | 0.086 | 1.0 | 10 |
| PaceToFieldSeconds_lag0 | 0.042 | 0.04 | 0.028 | 0.9 | 10 |
| GapToCarBehindSeconds_lag0 | 0.04 | 0.038 | 0.027 | 1.0 | 10 |
| GapToCarAheadSeconds_lag0 | 0.032 | 0.036 | 0.029 | 0.9 | 10 |
| TyreLife | 0.022 | 0.02 | 0.03 | 0.8 | 10 |
| Stint | 0.014 | 0.006 | 0.022 | 0.8 | 10 |
| PitThisLap | 0.013 | 0.008 | 0.019 | 0.9 | 10 |
| RaceProgress | 0.008 | 0.005 | 0.009 | 0.9 | 10 |
| QualifyingPosition | 0.008 | 0.004 | 0.009 | 0.9 | 10 |
| Position_std5 | 0.007 | 0.007 | 0.013 | 0.6 | 10 |
| GapToCarAheadSeconds_lag1 | 0.006 | 0.006 | 0.009 | 0.7 | 10 |
| PaceToFieldSeconds_lag1 | 0.006 | 0.003 | 0.01 | 0.6 | 10 |
| PaceToFieldSeconds_mean5 | 0.005 | 0.005 | 0.01 | 0.6 | 10 |
| LapTimeSeconds_std5 | 0.005 | 0.005 | 0.01 | 0.7 | 10 |
| Team | 0.005 | 0.003 | 0.014 | 0.7 | 10 |
| PaceToFieldSeconds_lag4 | 0.005 | 0.003 | 0.007 | 0.7 | 10 |
| PaceToFieldSeconds_lag2 | 0.004 | 0.003 | 0.005 | 0.9 | 10 |
| GapToLeaderSeconds_std5 | 0.004 | 0.002 | 0.005 | 0.9 | 10 |
| QualifyingDeltaToPoleSeconds | 0.004 | 0.001 | 0.009 | 0.5 | 10 |
| GridPosition | 0.004 | 0.002 | 0.008 | 0.6 | 10 |
| GapToLeaderSeconds_lag2 | 0.004 | 0.004 | 0.005 | 0.7 | 10 |
| PaceToTeammateSeconds_lag3 | 0.004 | 0.004 | 0.005 | 0.8 | 10 |
| Compound | 0.003 | 0.003 | 0.01 | 0.6 | 10 |
| TeamSeasonAvgFinish | 0.003 | 0.003 | 0.004 | 0.7 | 10 |
| PaceToTeammateSeconds_trend5 | 0.002 | 0.002 | 0.003 | 0.8 | 10 |

Features that were consistently non-positive are candidates for removal, but
they should be confirmed with retraining because permutation importance can
understate correlated inputs:

| Feature | MeanImportance | MedianImportance | ImportanceStdAcrossRaces | PositiveRaceShare | Races |
| --- | --- | --- | --- | --- | --- |
| TrackTemp_lag0 | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| Position_lag0 | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| EventFormat | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| TyreLife_lag0 | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| Rainfall_lag2 | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| Rainfall_lag1 | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| AirTemp_lag2 | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| Circuit | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| Country | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| TotalRaceLaps | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| TrackTemp_lag1 | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| Rainfall_lag3 | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| Rainfall_lag4 | 0.0 | 0.0 | 0.0 | 0.0 | 10 |
| Rainfall_lag0 | -0.0 | 0.0 | 0.0 | 0.0 | 10 |
| SafetyCarLapsLast5 | -0.0 | 0.0 | 0.0 | 0.0 | 10 |
| SafetyCarThisLap | -0.0 | 0.0 | 0.0 | 0.0 | 10 |
| Sector2Seconds_mean5 | -0.0 | 0.0 | 0.0 | 0.1 | 10 |
| Sector1Seconds_lag1 | -0.0 | 0.0 | 0.001 | 0.3 | 10 |
| TrackTemp_lag3 | -0.0 | 0.0 | 0.0 | 0.0 | 10 |
| TrackTemp_lag2 | -0.0 | 0.0 | 0.0 | 0.0 | 10 |

Domain-relevant features with unexpectedly low unique importance:

| Feature | MeanImportance | MedianImportance | ImportanceStdAcrossRaces | PositiveRaceShare | Races |
| --- | --- | --- | --- | --- | --- |
| GapToLeaderSeconds_lag4 | -0.002 | -0.002 | 0.004 | 0.4 | 10 |
| GapToCarAheadSeconds_trend5 | -0.001 | -0.001 | 0.005 | 0.4 | 10 |
| Sector3Seconds_std5 | -0.001 | -0.001 | 0.004 | 0.3 | 10 |
| DriverCircuitBestFinish | -0.001 | 0.0 | 0.004 | 0.5 | 10 |
| Sector3Seconds_lag0 | -0.001 | 0.0 | 0.002 | 0.1 | 10 |
| GapToCarBehindSeconds_lag3 | -0.001 | 0.0 | 0.002 | 0.5 | 10 |
| GapToCarAheadSeconds_lag2 | -0.0 | -0.0 | 0.001 | 0.5 | 10 |
| Sector1Seconds_std5 | -0.0 | -0.001 | 0.003 | 0.3 | 10 |
| GapToCarBehindSeconds_lag2 | -0.0 | 0.0 | 0.004 | 0.6 | 10 |
| Sector1Seconds_lag4 | -0.0 | -0.0 | 0.001 | 0.4 | 10 |
| GapToLeaderSeconds_mean5 | -0.0 | 0.0 | 0.001 | 0.5 | 10 |
| Sector2Seconds_lag1 | -0.0 | 0.0 | 0.0 | 0.0 | 10 |
| DriverCircuitAvgPositionGain | -0.0 | 0.0 | 0.001 | 0.5 | 10 |
| GapToCarAheadSeconds_lag3 | -0.0 | -0.0 | 0.002 | 0.4 | 10 |
| Sector2Seconds_lag4 | -0.0 | -0.0 | 0.001 | 0.1 | 10 |

These low values do not automatically mean sector, gap, qualifying, circuit, or
DRS information is useless. The lag, mean, standard-deviation and trend versions
are highly redundant, allowing the model to replace a shuffled feature with a
closely related one.

## 5. Feature-group ablation

Ablation is stronger evidence than individual permutation: the model was
retrained after removing an entire related group. A positive MAE increase means
the group helped; a negative value means the reduced model performed better.

| FeatureGroup | FeaturesDropped | MeanMAEIncrease | MedianMAEIncrease | HelpfulRaceShare | Races |
| --- | --- | --- | --- | --- | --- |
| gaps_and_relative_pace | 37 | 0.046 | 0.042 | 0.875 | 8 |
| tyres_and_pits | 15 | 0.019 | 0.019 | 0.875 | 8 |
| race_control | 3 | 0.005 | 0.005 | 0.625 | 8 |
| weather | 18 | 0.001 | 0.009 | 0.75 | 8 |
| lap_and_sector_timing | 32 | 0.0 | 0.004 | 0.625 | 8 |
| qualifying_and_grid | 5 | -0.0 | 0.003 | 0.5 | 8 |
| driver_and_team_history | 15 | -0.001 | 0.001 | 0.5 | 8 |
| drs | 5 | -0.003 | 0.004 | 0.625 | 8 |
| position_state | 9 | -0.003 | 0.002 | 0.5 | 8 |
| identity_and_circuit | 6 | -0.01 | -0.014 | 0.125 | 8 |

Groups with a negative or near-zero mean increase are the first candidates for
removal. Groups with a large positive increase should be retained. A low
HelpfulRaceShare means the group's usefulness is unstable across circuits.

### Evidence-based interpretation

- **Retain gaps and relative pace.** Removing these 37 inputs increased MAE by
  about 0.046 and hurt seven of eight diagnostic races. Current field-relative
  pace and immediate gaps contain the clearest stable signal.
- **Retain tyres and pit context.** Removing this group increased MAE by about
  0.019 and hurt seven of eight races. `TyreLife`, `Stint`, and `PitThisLap` also
  ranked highly in permutation importance.
- **Test removing identity and circuit categories first.** Their removal improved
  MAE by about 0.010 and helped seven of eight races. Driver/team identities can
  encourage memorization of performance that becomes stale after team or
  regulation changes. Keep leakage-safe numeric history separately during this test.
- **Current position is relied upon but may not generalize.** Shuffling the single
  `Position` feature caused the largest immediate performance loss, yet retraining
  without the nine-feature position-state family slightly improved mean MAE. This
  is not contradictory: the fitted model relies on position once it is available,
  but a model denied that shortcut can learn a combination that transfers better.
  Position also imposes floor/ceiling effects on a position-change target, which
  can encourage regression toward zero. Confirm this on more ablation folds.
- **Absolute timing contributes surprisingly little.** Removing 32 lap/sector
  timing inputs changed mean MAE by less than 0.001, while relative pace was useful.
  Absolute seconds vary heavily by circuit and conditions. Replace or supplement
  them with sector deltas to the field, teammate, and driver's clean-air baseline.
- **The current DRS summary is unstable.** Removing all five DRS lags improved mean
  MAE slightly, although DRS helped five of eight races. `DRSActivePct` measures use,
  not whether a pass was possible. Add detection-zone opportunity, gap under one
  second, and DRS-train context before deciding that DRS itself is unimportant.
- **Historical and qualifying groups are weak after the race is underway.** Their
  mean ablation effects were near zero. They may be redundant with current gaps,
  pace, and position, or too stale across seasons. Test recency weighting and
  regulation-era history rather than simply adding longer averages.

### Recommended next controlled experiment

Start from the selected parameters and compare the full model with a reduced model
that removes `identity_and_circuit`. Then test removing `position_state` and the
absolute `lap_and_sector_timing` family one group at a time. Keep gaps/relative pace
and tyres/pits in every candidate. Choose using race-level MAE through 2024 and wait
for newly ingested races before declaring a final deployment winner.

## 6. Practical improvement opportunities

1. **Optimize for the deployed objective.** The model fits squared error but is
   selected using MAE. Compare `loss="absolute_error"` in a future controlled run
   if MAE is the product metric.
2. **Treat unchanged position as a distinct process.** With 50.8%
   unchanged targets, test a two-stage system: classify whether a position change
   occurs, then regress its magnitude conditional on change.
3. **Constrain predictions to racing reality.** Continuous regression can predict
   fractional positions and impossible ranks. Compare rounded/clipped output and
   consider predicting pairwise overtaking probabilities or rank directly.
4. **Add survival awareness.** Current windows require a valid target five laps
   later, which excludes drivers who retire inside the horizon. Add a retirement
   classifier or explicit DNF outcome so deployment is not conditioned on survival.
5. **Improve gap reliability.** Missing leader/ahead gaps have racing meaning.
   Add explicit `HasCarAhead`, `IsLeader`, and timing-availability indicators rather
   than relying only on NaN routing.
6. **Reduce redundant lag families.** Use the ablation results before deleting
   individual lags. Highly correlated sector and lap summaries inflate complexity
   without necessarily adding stable information.
7. **Add event-state interactions.** Tyre age, pit status, safety-car state and rain
   affect pace differently together. Trees learn interactions, but clearer flags
   such as pit-window phase, laps since restart, and wet-compound state may help.
8. **Model regulation changes explicitly.** Add regulation-era or season-recency
   weighting so old team performance contributes less after major rule changes.
9. **Use multiple horizons.** A five-lap horizon behaves very differently during
   stable running and pit cycles. Compare 1-, 3-, 5- and 10-lap targets.
10. **Preserve a final future test.** After acting on this report, all races in it
    have influenced development. Use newly ingested races as the next untouched test.

## 7. Limitations

- FastF1 could not produce the 2018 Italian GP feature partition.
- The 2021 Belgian GP was too short for a five-lap window plus five-lap target.
- Historical circuit statistics are naturally missing for first appearances.
- Adjacent drivers and windows are dependent observations; race-level splitting
  prevents leakage but standard row-level confidence intervals would be invalid.
- Parameter search was deliberately finite, so the selected values are best among
  tested candidates rather than mathematically optimal.
- Repeatedly using the dashboard to choose settings from a race converts that race
  from untouched test data into development feedback.

## 8. Reproducibility files

- `tuning_results.csv`: every parameter/fold result.
- `best_parameters.json`: selected model settings.
- `walk_forward_metrics.csv`: one row per validation race.
- `predictions/`: window-level predictions for every evaluated race.
- `permutation_importance.csv`: feature importance by representative race.
- `feature_group_ablation.csv`: retraining results after dropping each group.
- `missing_values.csv`: complete missingness audit.
- `final_model.joblib`: final estimator fitted on all available feature windows.
- `final_model_metadata.json`: training coverage and settings for that artifact.
