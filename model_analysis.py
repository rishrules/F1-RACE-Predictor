r"""Reproducible, resumable analysis for the F1 position-change model.

The script intentionally keeps model development separate from final evidence:

* Ten chronological folds ending in 2024 are used to compare parameter sets.
* The chosen parameters are then evaluated on every race with 40 prior races.
* Results from 2025-2026 are reported separately as untouched later seasons.
* Permutation importance and feature-group ablation diagnose feature usefulness.

Run every stage with:

    .\.venv\Scripts\python.exe -u model_analysis.py --stage all

Every completed fit is written immediately. Re-running the command skips work
already present in the output directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

from main import (
    ModelParameters,
    PreparedTrainingData,
    TrainingResult,
    WalkForwardSplit,
    create_hist_gradient_boosting_regressor,
    load_and_prepare_training_data,
    make_walk_forward_split,
    prediction_frame,
    train_and_evaluate,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "reports" / "model_analysis"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
INITIAL_TRAINING_RACES = 40
LAP_STRIDE = 5
RANDOM_STATE = 42

TUNING_RESULTS_PATH = OUTPUT_DIR / "tuning_results.csv"
BEST_PARAMETERS_PATH = OUTPUT_DIR / "best_parameters.json"
EVALUATION_RESULTS_PATH = OUTPUT_DIR / "walk_forward_metrics.csv"
IMPORTANCE_RESULTS_PATH = OUTPUT_DIR / "permutation_importance.csv"
ABLATION_RESULTS_PATH = OUTPUT_DIR / "feature_group_ablation.csv"
MISSING_RESULTS_PATH = OUTPUT_DIR / "missing_values.csv"
REPORT_PATH = OUTPUT_DIR / "MODEL_ANALYSIS_REPORT.md"
FINAL_MODEL_PATH = OUTPUT_DIR / "final_model.joblib"
FINAL_MODEL_METADATA_PATH = OUTPUT_DIR / "final_model_metadata.json"


# Deliberately compact search space: each candidate represents a meaningful
# bias/variance trade-off rather than changing many values at random.
CANDIDATE_PARAMETERS: dict[str, ModelParameters] = {
    "default": ModelParameters(
        learning_rate=0.10,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
    ),
    "slower_regularized": ModelParameters(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=30,
        l2_regularization=1.0,
    ),
    "small_trees": ModelParameters(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=15,
        max_depth=None,
        min_samples_leaf=30,
        l2_regularization=1.0,
    ),
    "shallow": ModelParameters(
        learning_rate=0.08,
        max_iter=175,
        max_leaf_nodes=31,
        max_depth=6,
        min_samples_leaf=30,
        l2_regularization=1.0,
    ),
    "large_leaves": ModelParameters(
        learning_rate=0.10,
        max_iter=125,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=60,
        l2_regularization=1.0,
    ),
    "strong_regularization": ModelParameters(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=40,
        l2_regularization=8.0,
    ),
    "fast_boosting": ModelParameters(
        learning_rate=0.15,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=30,
        l2_regularization=1.0,
    ),
    "high_capacity": ModelParameters(
        learning_rate=0.05,
        max_iter=225,
        max_leaf_nodes=63,
        max_depth=None,
        min_samples_leaf=30,
        l2_regularization=2.0,
    ),
}


def feature_groups(columns: pd.Index) -> dict[str, list[str]]:
    """Map related inputs to groups for drop-one-group ablation tests."""
    names = list(columns)

    def containing(*parts: str) -> list[str]:
        return [name for name in names if any(part in name for part in parts)]

    groups = {
        "position_state": [
            name
            for name in names
            if name == "Position" or name.startswith("Position_")
        ],
        "lap_and_sector_timing": containing("LapTimeSeconds", "Sector1", "Sector2", "Sector3"),
        "gaps_and_relative_pace": containing("GapTo", "PaceTo"),
        "driver_and_team_history": containing("DriverSeason", "DriverCircuit", "TeamSeason", "TeamCircuit"),
        "qualifying_and_grid": containing("Qualifying", "GridPosition"),
        "tyres_and_pits": containing("Tyre", "Compound", "Stint", "Pit"),
        "weather": containing("AirTemp", "TrackTemp", "Rainfall"),
        "race_control": containing("TrackStatus", "SafetyCar"),
        "drs": containing("DRSActivePct"),
        "identity_and_circuit": [
            name
            for name in names
            if name in {"Driver", "DriverNumber", "Team", "Circuit", "Country", "EventFormat"}
        ],
    }
    return {group: values for group, values in groups.items() if values}


def event_table(prepared: PreparedTrainingData) -> pd.DataFrame:
    """Return one ordered row for each race represented by feature windows."""
    return (
        prepared.metadata[["Year", "RoundNumber", "EventName"]]
        .drop_duplicates(["Year", "RoundNumber"])
        .sort_values(["Year", "RoundNumber"])
        .reset_index(drop=True)
    )


def append_csv_row(path: Path, row: dict[str, object]) -> None:
    """Checkpoint one completed result immediately."""
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_keys(path: Path, columns: list[str]) -> set[tuple[object, ...]]:
    """Read keys already checkpointed by a previous or interrupted run."""
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    return set(frame[columns].itertuples(index=False, name=None))


def selected_tuning_indices(events: pd.DataFrame) -> list[int]:
    """Select ten well-spaced development folds, never extending past 2024."""
    last_2024_index = int(events.index[events["Year"].le(2024)].max())
    return sorted(
        set(
            np.linspace(
                INITIAL_TRAINING_RACES,
                last_2024_index,
                num=10,
                dtype=int,
            ).tolist()
        )
    )


def selected_diagnostic_indices(events: pd.DataFrame, count: int) -> list[int]:
    """Choose folds spanning the complete eligible period for diagnostics."""
    return sorted(
        set(
            np.linspace(
                INITIAL_TRAINING_RACES,
                len(events) - 1,
                num=count,
                dtype=int,
            ).tolist()
        )
    )


def tune_parameters(prepared: PreparedTrainingData, events: pd.DataFrame) -> ModelParameters:
    """Compare candidates on identical chronological folds through 2024."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done = completed_keys(TUNING_RESULTS_PATH, ["Configuration", "Year", "RoundNumber"])
    tuning_indices = selected_tuning_indices(events)
    total = len(CANDIDATE_PARAMETERS) * len(tuning_indices)
    completed = len(done)

    for config_name, parameters in CANDIDATE_PARAMETERS.items():
        for event_index in tuning_indices:
            event = events.iloc[event_index]
            key = (config_name, int(event.Year), int(event.RoundNumber))
            if key in done:
                continue

            print(
                f"TUNE {completed + 1}/{total}: {config_name} on "
                f"{int(event.Year)} R{int(event.RoundNumber)} {event.EventName}",
                flush=True,
            )
            split = make_walk_forward_split(
                prepared, int(event.Year), int(event.RoundNumber)
            )
            result = train_and_evaluate(split, parameters)
            append_csv_row(
                TUNING_RESULTS_PATH,
                {
                    "Configuration": config_name,
                    "Year": int(event.Year),
                    "RoundNumber": int(event.RoundNumber),
                    "EventName": event.EventName,
                    "Windows": len(split.y_validation),
                    **asdict(parameters),
                    **result.metrics,
                    "TrainingSeconds": result.training_seconds,
                },
            )
            completed += 1

    tuning = pd.read_csv(TUNING_RESULTS_PATH)
    summary = (
        tuning.groupby("Configuration")
        .agg(
            MeanRaceMAE=("MAE", "mean"),
            MedianRaceMAE=("MAE", "median"),
            MeanRMSE=("RMSE", "mean"),
            MeanImprovement=("Baseline improvement %", "mean"),
            Folds=("MAE", "size"),
        )
        .sort_values(["MeanRaceMAE", "MeanRMSE"])
    )
    best_name = str(summary.index[0])
    best = CANDIDATE_PARAMETERS[best_name]
    BEST_PARAMETERS_PATH.write_text(
        json.dumps(
            {"Configuration": best_name, **asdict(best)}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"BEST CONFIGURATION: {best_name} {asdict(best)}", flush=True)
    return best


def load_best_parameters() -> tuple[str, ModelParameters]:
    """Load the selected configuration produced by the tuning stage."""
    if not BEST_PARAMETERS_PATH.exists():
        raise FileNotFoundError("Run the tuning stage before evaluation")
    stored = json.loads(BEST_PARAMETERS_PATH.read_text(encoding="utf-8"))
    name = stored.pop("Configuration")
    return str(name), ModelParameters(**stored)


def evaluate_every_race(
    prepared: PreparedTrainingData,
    events: pd.DataFrame,
    parameters: ModelParameters,
) -> None:
    """Fit from scratch and validate every race after the 40-race seed."""
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    done = completed_keys(EVALUATION_RESULTS_PATH, ["Year", "RoundNumber"])
    eligible = list(range(INITIAL_TRAINING_RACES, len(events)))
    completed = len(done)

    for event_index in eligible:
        event = events.iloc[event_index]
        key = (int(event.Year), int(event.RoundNumber))
        if key in done:
            continue

        print(
            f"EVALUATE {completed + 1}/{len(eligible)}: "
            f"{int(event.Year)} R{int(event.RoundNumber)} {event.EventName}",
            flush=True,
        )
        split = make_walk_forward_split(
            prepared, int(event.Year), int(event.RoundNumber)
        )
        result = train_and_evaluate(split, parameters)
        predictions = prediction_frame(result)
        prediction_path = (
            PREDICTION_DIR
            / f"year={int(event.Year)}_round={int(event.RoundNumber):02d}.parquet"
        )
        temporary_path = prediction_path.with_suffix(".parquet.tmp")
        predictions.to_parquet(temporary_path, index=False)
        temporary_path.replace(prediction_path)

        append_csv_row(
            EVALUATION_RESULTS_PATH,
            {
                "Year": int(event.Year),
                "RoundNumber": int(event.RoundNumber),
                "EventName": event.EventName,
                "TrainingRaces": event_index,
                "TrainingWindows": len(split.y_train),
                "ValidationWindows": len(split.y_validation),
                "FeaturesUsed": split.X_train.shape[1],
                **result.metrics,
                "TrainingSeconds": result.training_seconds,
            },
        )
        completed += 1


def permutation_analysis(
    prepared: PreparedTrainingData,
    events: pd.DataFrame,
    parameters: ModelParameters,
) -> None:
    """Measure individual feature reliance on representative unseen races."""
    done = completed_keys(IMPORTANCE_RESULTS_PATH, ["Year", "RoundNumber", "Feature"])
    indices = selected_diagnostic_indices(events, count=10)

    for number, event_index in enumerate(indices, start=1):
        event = events.iloc[event_index]
        year, round_number = int(event.Year), int(event.RoundNumber)
        # A completed fold contains one row for every feature used by that fold.
        if any(key[0] == year and key[1] == round_number for key in done):
            continue

        print(
            f"IMPORTANCE {number}/{len(indices)}: {year} R{round_number} {event.EventName}",
            flush=True,
        )
        split = make_walk_forward_split(prepared, year, round_number)
        result = train_and_evaluate(split, parameters)
        started_at = perf_counter()
        importance = permutation_importance(
            result.model,
            split.X_validation,
            split.y_validation,
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        elapsed = perf_counter() - started_at
        fold_frame = pd.DataFrame(
            {
                "Year": year,
                "RoundNumber": round_number,
                "EventName": event.EventName,
                "Feature": split.X_validation.columns,
                "ImportanceMean": importance.importances_mean,
                "ImportanceStd": importance.importances_std,
                "FoldMAE": result.metrics["MAE"],
                "CalculationSeconds": elapsed,
            }
        )
        fold_frame.to_csv(
            IMPORTANCE_RESULTS_PATH,
            mode="a",
            header=not IMPORTANCE_RESULTS_PATH.exists(),
            index=False,
        )


def subset_split(split: WalkForwardSplit, dropped: list[str]) -> WalkForwardSplit:
    """Create an ablation split while preserving labels, metadata and weights."""
    retained = [column for column in split.X_train if column not in dropped]
    return replace(
        split,
        X_train=split.X_train[retained],
        X_validation=split.X_validation[retained],
    )


def ablation_analysis(
    prepared: PreparedTrainingData,
    events: pd.DataFrame,
    parameters: ModelParameters,
) -> None:
    """Retrain after dropping one coherent feature group at a time."""
    groups = feature_groups(prepared.X.columns)
    indices = selected_diagnostic_indices(events, count=8)
    done = completed_keys(ABLATION_RESULTS_PATH, ["Year", "RoundNumber", "FeatureGroup"])
    evaluation = pd.read_csv(EVALUATION_RESULTS_PATH).set_index(["Year", "RoundNumber"])
    total = len(indices) * len(groups)
    completed = len(done)

    for event_index in indices:
        event = events.iloc[event_index]
        year, round_number = int(event.Year), int(event.RoundNumber)
        full_mae = float(evaluation.loc[(year, round_number), "MAE"])
        split = make_walk_forward_split(prepared, year, round_number)

        for group_name, group_columns in groups.items():
            key = (year, round_number, group_name)
            if key in done:
                continue
            print(
                f"ABLATION {completed + 1}/{total}: drop {group_name} on "
                f"{year} R{round_number}",
                flush=True,
            )
            dropped = [column for column in group_columns if column in split.X_train]
            candidate = train_and_evaluate(subset_split(split, dropped), parameters)
            append_csv_row(
                ABLATION_RESULTS_PATH,
                {
                    "Year": year,
                    "RoundNumber": round_number,
                    "EventName": event.EventName,
                    "FeatureGroup": group_name,
                    "FeaturesDropped": len(dropped),
                    "FullModelMAE": full_mae,
                    "AblatedMAE": candidate.metrics["MAE"],
                    "MAEIncreaseWhenDropped": candidate.metrics["MAE"] - full_mae,
                    "TrainingSeconds": candidate.training_seconds,
                },
            )
            completed += 1


def train_final_model(
    prepared: PreparedTrainingData,
    events: pd.DataFrame,
    parameters: ModelParameters,
) -> None:
    """Fit the deployment artifact on all available data after evaluation."""
    X = prepared.X.copy()
    category_vocabularies: dict[str, list[object]] = {}
    for column in prepared.categorical_columns:
        X[column] = X[column].astype("category")
        category_vocabularies[column] = X[column].cat.categories.tolist()

    # This is determined using training data because every available row is now
    # part of the final fit. The walk-forward results—not this fit—measure quality.
    usable_columns = X.columns[~X.isna().all(axis=0)]
    X = X.loc[:, usable_columns]
    weights = prepared.sample_weight / prepared.sample_weight.mean()
    model = create_hist_gradient_boosting_regressor(
        **asdict(parameters), early_stopping=False
    )

    print(
        f"FINAL MODEL: fitting {len(X):,} windows and {X.shape[1]} features...",
        flush=True,
    )
    started_at = perf_counter()
    model.fit(X, prepared.y, sample_weight=weights)
    training_seconds = perf_counter() - started_at

    bundle = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "categorical_columns": [
            column for column in prepared.categorical_columns if column in X
        ],
        "category_vocabularies": category_vocabularies,
        "parameters": asdict(parameters),
        "target": "TargetPositionChange",
        "lap_stride": LAP_STRIDE,
    }
    temporary_path = FINAL_MODEL_PATH.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary_path)
    temporary_path.replace(FINAL_MODEL_PATH)

    latest = events.iloc[-1]
    FINAL_MODEL_METADATA_PATH.write_text(
        json.dumps(
            {
                "TrainingWindows": len(X),
                "Features": X.shape[1],
                "Races": len(events),
                "TrainingThrough": {
                    "Year": int(latest.Year),
                    "RoundNumber": int(latest.RoundNumber),
                    "EventName": latest.EventName,
                },
                "TrainingSeconds": training_seconds,
                "Parameters": asdict(parameters),
                "ValidationEvidence": "walk_forward_metrics.csv",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"FINAL MODEL WRITTEN: {FINAL_MODEL_PATH}", flush=True)


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    """Create a dependency-free Markdown table from a small DataFrame."""
    shown = frame.copy()
    for column in shown.select_dtypes(include="number"):
        shown[column] = shown[column].round(digits)
    headers = [str(column) for column in shown.columns]
    rows = [[str(value) for value in row] for row in shown.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(prepared: PreparedTrainingData, events: pd.DataFrame) -> None:
    """Compile metrics and diagnostics into one readable Markdown report."""
    config_name, parameters = load_best_parameters()
    tuning = pd.read_csv(TUNING_RESULTS_PATH)
    evaluation = pd.read_csv(EVALUATION_RESULTS_PATH).sort_values(["Year", "RoundNumber"])
    importance = pd.read_csv(IMPORTANCE_RESULTS_PATH)
    ablation = pd.read_csv(ABLATION_RESULTS_PATH)
    prediction_files = sorted(PREDICTION_DIR.glob("*.parquet"))
    predictions = pd.concat(
        [pd.read_parquet(path) for path in prediction_files], ignore_index=True
    )

    tuning_summary = (
        tuning.groupby("Configuration")
        .agg(
            MeanRaceMAE=("MAE", "mean"),
            MedianRaceMAE=("MAE", "median"),
            MeanRMSE=("RMSE", "mean"),
            BaselineImprovementPct=("Baseline improvement %", "mean"),
        )
        .sort_values("MeanRaceMAE")
        .reset_index()
    )
    yearly = (
        evaluation.groupby("Year")
        .agg(
            Races=("MAE", "size"),
            MeanRaceMAE=("MAE", "mean"),
            MedianRaceMAE=("MAE", "median"),
            MeanRMSE=("RMSE", "mean"),
            BaselineMAE=("Baseline MAE", "mean"),
            ImprovementPct=("Baseline improvement %", "mean"),
        )
        .reset_index()
    )
    later = evaluation[evaluation["Year"].ge(2025)]
    worst = evaluation.nlargest(10, "MAE")[["Year", "RoundNumber", "EventName", "MAE", "Baseline MAE"]]
    best = evaluation.nsmallest(10, "MAE")[["Year", "RoundNumber", "EventName", "MAE", "Baseline MAE"]]

    importance_summary = (
        importance.groupby("Feature")
        .agg(
            MeanImportance=("ImportanceMean", "mean"),
            MedianImportance=("ImportanceMean", "median"),
            ImportanceStdAcrossRaces=("ImportanceMean", "std"),
            PositiveRaceShare=("ImportanceMean", lambda values: (values > 0).mean()),
            Races=("ImportanceMean", "size"),
        )
        .sort_values("MeanImportance", ascending=False)
        .reset_index()
    )
    ablation_summary = (
        ablation.groupby("FeatureGroup")
        .agg(
            FeaturesDropped=("FeaturesDropped", "max"),
            MeanMAEIncrease=("MAEIncreaseWhenDropped", "mean"),
            MedianMAEIncrease=("MAEIncreaseWhenDropped", "median"),
            HelpfulRaceShare=("MAEIncreaseWhenDropped", lambda values: (values > 0).mean()),
            Races=("MAEIncreaseWhenDropped", "size"),
        )
        .sort_values("MeanMAEIncrease", ascending=False)
        .reset_index()
    )

    missing = pd.DataFrame(
        {
            "Feature": prepared.X.columns,
            "MissingCount": prepared.X.isna().sum().to_numpy(),
            "MissingPercent": prepared.X.isna().mean().mul(100).to_numpy(),
        }
    ).sort_values("MissingPercent", ascending=False)
    missing.to_csv(MISSING_RESULTS_PATH, index=False)

    predictions["AbsoluteError"] = predictions["Residual"].abs()
    actual = predictions["ActualPositionChange"]
    predicted = predictions["PredictedPositionChange"]
    no_change_share = float(actual.eq(0).mean() * 100)
    exact_rounded_share = float(np.rint(predicted).eq(actual).mean() * 100)
    within_one_share = float((predictions["AbsoluteError"] <= 1).mean() * 100)
    bias = float(predicted.sub(actual).mean())
    extreme = predictions.loc[actual.abs().ge(3), "AbsoluteError"]

    weak_expected_patterns = ("Sector", "GapTo", "Qualifying", "DriverCircuit", "TeamCircuit", "DRS")
    expected_features = importance_summary[
        importance_summary["Feature"].str.contains("|".join(weak_expected_patterns), regex=True)
    ]
    weak_expected = expected_features.nsmallest(15, "MeanImportance")
    consistently_nonpositive = importance_summary[
        (importance_summary["MeanImportance"] <= 0)
        & (importance_summary["PositiveRaceShare"] <= 0.4)
    ].head(20)

    total_seconds = float(
        tuning["TrainingSeconds"].sum()
        + evaluation["TrainingSeconds"].sum()
        + ablation["TrainingSeconds"].sum()
    )
    baseline_win_rate = float((evaluation["MAE"] < evaluation["Baseline MAE"]).mean() * 100)

    report = f"""# HistGradientBoosting F1 model analysis

## Executive summary

This experiment used **{len(prepared.X):,} stride-{LAP_STRIDE} prediction windows** from
**{len(events)} races**. The first {INITIAL_TRAINING_RACES} races formed the initial history;
the model was then freshly trained and validated chronologically on every remaining
race (**{len(evaluation)} race-level folds**). No validation-race row was present in
its training set.

The selected configuration was **{config_name}**:

```json
{json.dumps(asdict(parameters), indent=2)}
```

Across all evaluated races, mean race MAE was **{evaluation['MAE'].mean():.3f}**
positions, compared with **{evaluation['Baseline MAE'].mean():.3f}** for predicting
no position change. The model beat that baseline in **{baseline_win_rate:.1f}%** of
races. On the later 2025-2026 evidence, mean race MAE was
**{later['MAE'].mean():.3f}** versus baseline **{later['Baseline MAE'].mean():.3f}**.

Important caution: 2018-2024 folds were used for model development and parameter
selection. The 2025-2026 section is the cleaner estimate of forward generalization.
No test is perfectly permanent once its result has been repeatedly inspected.

## 1. Dataset and validation design

- Feature window: five completed laps; target: position change five laps later.
- Window stride: {LAP_STRIDE}, reducing four-lap overlap between neighbouring samples.
- Initial history: {INITIAL_TRAINING_RACES} races.
- Validation: one complete race at a time, with only earlier races in training.
- Training weights: each training race contributes equal total weight.
- Missing numeric values: handled natively by HistGradientBoosting.
- Completely empty early-fold features: removed using training data only.
- Scaling: not used because tree splits do not require standardized units.
- Training time represented in checkpoints: {total_seconds / 60:.1f} minutes.

The dataset contains no missing target values. **{int((missing['MissingPercent'] > 10).sum())}**
features exceed 10% missingness and **{int((missing['MissingPercent'] > 50).sum())}**
exceed 50%.

Top missing features:

{markdown_table(missing.head(15))}

## 2. Hyperparameter comparison

The same ten chronological development folds through 2024 were used for every
candidate. Mean race MAE was the primary selection metric; lower is better.

{markdown_table(tuning_summary)}

## 3. Walk-forward accuracy by season

{markdown_table(yearly)}

The model beat the no-change baseline in **{baseline_win_rate:.1f}%** of evaluated
races. This comparison matters because **{no_change_share:.1f}%** of individual
five-lap targets contain no position change.

At window level, rounded predictions exactly matched the target **{exact_rounded_share:.1f}%**
of the time and continuous predictions were within one position **{within_one_share:.1f}%**
of the target. Mean signed prediction bias was **{bias:.3f}** positions. On the
harder windows where the true change was at least three positions, MAE was
**{extreme.mean():.3f}**.

### Ten hardest validation races

{markdown_table(worst)}

### Ten easiest validation races

{markdown_table(best)}

## 4. Individual permutation importance

Importance is the increase in validation MAE after one feature is shuffled.
Positive values indicate useful unique information. Near-zero values can also
occur when correlated features substitute for one another.

Top 25 features across representative unseen races:

{markdown_table(importance_summary.head(25))}

Features that were consistently non-positive are candidates for removal, but
they should be confirmed with retraining because permutation importance can
understate correlated inputs:

{markdown_table(consistently_nonpositive if not consistently_nonpositive.empty else pd.DataFrame({'Result': ['None met the strict non-positive criterion']}))}

Domain-relevant features with unexpectedly low unique importance:

{markdown_table(weak_expected)}

These low values do not automatically mean sector, gap, qualifying, circuit, or
DRS information is useless. The lag, mean, standard-deviation and trend versions
are highly redundant, allowing the model to replace a shuffled feature with a
closely related one.

## 5. Feature-group ablation

Ablation is stronger evidence than individual permutation: the model was
retrained after removing an entire related group. A positive MAE increase means
the group helped; a negative value means the reduced model performed better.

{markdown_table(ablation_summary)}

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
2. **Treat unchanged position as a distinct process.** With {no_change_share:.1f}%
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
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"REPORT WRITTEN: {REPORT_PATH}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("tune", "evaluate", "importance", "ablation", "final", "report", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and preparing feature data...", flush=True)
    prepared = load_and_prepare_training_data(lap_stride=LAP_STRIDE)
    events = event_table(prepared)
    print(
        f"Loaded {len(prepared.X):,} windows, {prepared.X.shape[1]} features, "
        f"and {len(events)} races.",
        flush=True,
    )

    if args.stage in {"tune", "all"}:
        tune_parameters(prepared, events)
    if args.stage in {"evaluate", "importance", "ablation", "final", "report", "all"}:
        _, parameters = load_best_parameters()
    if args.stage in {"evaluate", "all"}:
        evaluate_every_race(prepared, events, parameters)
    if args.stage in {"importance", "all"}:
        permutation_analysis(prepared, events, parameters)
    if args.stage in {"ablation", "all"}:
        ablation_analysis(prepared, events, parameters)
    if args.stage in {"final", "all"}:
        train_final_model(prepared, events, parameters)
    if args.stage in {"report", "all"}:
        build_report(prepared, events)


if __name__ == "__main__":
    main()
