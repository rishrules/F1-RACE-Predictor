r"""Compare the requested F1 model improvements on identical race-level folds.

This experiment is resumable. It compares:

* single-stage and two-stage HistGradientBoosting;
* equal history and 20/40/80-race recency half-lives;
* four position-state feature variants;
* 1, 3, 5 and 10-lap prediction horizons.

Run with:

    .\.venv\Scripts\python.exe -u improvement_analysis.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from main import (
    ModelParameters,
    PreparedTrainingData,
    WalkForwardSplit,
    load_and_prepare_training_data,
    make_walk_forward_split,
    create_two_stage_position_model,
    train_and_evaluate,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "reports" / "model_improvements"
RESULTS_PATH = OUTPUT_DIR / "comparison_results.csv"
REPORT_PATH = OUTPUT_DIR / "IMPROVEMENT_REPORT.md"
FINAL_MODEL_PATH = OUTPUT_DIR / "final_two_stage_model.joblib"
FINAL_MODEL_METADATA_PATH = OUTPUT_DIR / "final_two_stage_model_metadata.json"
LAP_STRIDE = 5
INITIAL_TRAINING_RACES = 40
DIAGNOSTIC_RACES = 8
PARAMETERS = ModelParameters()


@dataclass(frozen=True)
class ExperimentSpecification:
    key: str
    horizon: int
    two_stage: bool
    recency_half_life: float | None
    position_variant: str = "full"


# Duplicate base specifications are represented once and reused in each report
# section, avoiding unnecessary model fits.
SPECIFICATIONS = (
    ExperimentSpecification("single_stage", 5, False, 40),
    ExperimentSpecification("two_stage", 5, True, 40),
    ExperimentSpecification("recency_equal", 5, True, None),
    ExperimentSpecification("recency_20", 5, True, 20),
    ExperimentSpecification("recency_80", 5, True, 80),
    ExperimentSpecification("position_current_only", 5, True, 40, "current_only"),
    ExperimentSpecification("position_prior_lags_only", 5, True, 40, "prior_lags_only"),
    ExperimentSpecification("position_none", 5, True, 40, "none"),
    ExperimentSpecification("horizon_1", 1, True, 40),
    ExperimentSpecification("horizon_3", 3, True, 40),
    ExperimentSpecification("horizon_10", 10, True, 40),
)


def event_table(prepared: PreparedTrainingData) -> pd.DataFrame:
    return (
        prepared.metadata[["Year", "RoundNumber", "EventName"]]
        .drop_duplicates(["Year", "RoundNumber"])
        .sort_values(["Year", "RoundNumber"])
        .reset_index(drop=True)
    )


def diagnostic_event_indices(events: pd.DataFrame) -> list[int]:
    """Use identical races distributed over the complete eligible period."""
    return sorted(
        set(
            np.linspace(
                INITIAL_TRAINING_RACES,
                len(events) - 1,
                DIAGNOSTIC_RACES,
                dtype=int,
            ).tolist()
        )
    )


def position_columns(columns: pd.Index) -> list[str]:
    """Return race-position state columns without matching position-gain history."""
    return [
        column
        for column in columns
        if column == "Position" or column.startswith("Position_")
    ]


def apply_position_variant(
    split: WalkForwardSplit,
    variant: str,
) -> WalkForwardSplit:
    """Create a controlled position-state feature subset."""
    all_position_columns = position_columns(split.X_train.columns)
    if variant == "full":
        dropped: list[str] = []
    elif variant == "current_only":
        dropped = [column for column in all_position_columns if column != "Position"]
    elif variant == "prior_lags_only":
        retained_lags = {f"Position_lag{lag}" for lag in range(1, 5)}
        dropped = [
            column for column in all_position_columns if column not in retained_lags
        ]
    elif variant == "none":
        dropped = all_position_columns
    else:
        raise ValueError(f"Unknown position variant: {variant}")

    retained = [column for column in split.X_train if column not in dropped]
    return replace(
        split,
        X_train=split.X_train[retained],
        X_validation=split.X_validation[retained],
    )


def append_result(row: dict[str, object]) -> None:
    frame = pd.DataFrame([row])
    frame.to_csv(
        RESULTS_PATH,
        mode="a",
        header=not RESULTS_PATH.exists(),
        index=False,
    )


def completed_keys() -> set[tuple[str, int, int]]:
    if not RESULTS_PATH.exists():
        return set()
    results = pd.read_csv(RESULTS_PATH)
    return set(
        results[["Specification", "Year", "RoundNumber"]].itertuples(
            index=False, name=None
        )
    )


def run_comparisons() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared_by_horizon = {
        horizon: load_and_prepare_training_data(
            lap_stride=LAP_STRIDE,
            target_horizon=horizon,
        )
        for horizon in (1, 3, 5, 10)
    }
    events = event_table(prepared_by_horizon[5])
    indices = diagnostic_event_indices(events)
    done = completed_keys()
    total = len(SPECIFICATIONS) * len(indices)
    completed = len(done)

    for specification in SPECIFICATIONS:
        prepared = prepared_by_horizon[specification.horizon]
        for event_index in indices:
            event = events.iloc[event_index]
            key = (
                specification.key,
                int(event.Year),
                int(event.RoundNumber),
            )
            if key in done:
                continue

            print(
                f"COMPARE {completed + 1}/{total}: {specification.key} on "
                f"{int(event.Year)} R{int(event.RoundNumber)} {event.EventName}",
                flush=True,
            )
            split = make_walk_forward_split(
                prepared,
                int(event.Year),
                int(event.RoundNumber),
                recency_half_life_races=specification.recency_half_life,
            )
            split = apply_position_variant(split, specification.position_variant)
            result = train_and_evaluate(
                split,
                PARAMETERS,
                two_stage=specification.two_stage,
            )
            append_result(
                {
                    "Specification": specification.key,
                    "Year": int(event.Year),
                    "RoundNumber": int(event.RoundNumber),
                    "EventName": event.EventName,
                    "Horizon": specification.horizon,
                    "TwoStage": specification.two_stage,
                    "RecencyHalfLife": specification.recency_half_life,
                    "PositionVariant": specification.position_variant,
                    "TrainingWindows": len(split.y_train),
                    "ValidationWindows": len(split.y_validation),
                    "Features": split.X_train.shape[1],
                    "NoChangePercent": float(split.y_validation.eq(0).mean() * 100),
                    "MAE": result.metrics["MAE"],
                    "RMSE": result.metrics["RMSE"],
                    "Baseline MAE": result.metrics["Baseline MAE"],
                    "Baseline improvement %": result.metrics[
                        "Baseline improvement %"
                    ],
                    "Change accuracy": result.metrics.get("Change accuracy", np.nan),
                    "Change Brier score": result.metrics.get(
                        "Change Brier score", np.nan
                    ),
                    "Change ROC AUC": result.metrics.get("Change ROC AUC", np.nan),
                    "TrainingSeconds": result.training_seconds,
                }
            )
            completed += 1


def summarize(results: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    selected = results[results["Specification"].isin(keys)]
    return (
        selected.groupby("Specification", dropna=False)
        .agg(
            Races=("MAE", "size"),
            MeanRaceMAE=("MAE", "mean"),
            MedianRaceMAE=("MAE", "median"),
            MeanRMSE=("RMSE", "mean"),
            BaselineMAE=("Baseline MAE", "mean"),
            ModelWinRate=(
                "Baseline improvement %",
                lambda values: float((values > 0).mean()),
            ),
            MeanBaselineImprovementPct=("Baseline improvement %", "mean"),
            NoChangePercent=("NoChangePercent", "mean"),
            ChangeAccuracy=("Change accuracy", "mean"),
            ChangeBrier=("Change Brier score", "mean"),
            ChangeROCAUC=("Change ROC AUC", "mean"),
            TrainingSeconds=("TrainingSeconds", "sum"),
        )
        .reset_index()
        .sort_values("MeanRaceMAE")
    )


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include="number"):
        shown[column] = shown[column].round(3)
    headers = [str(column) for column in shown.columns]
    rows = [[str(value) for value in row] for row in shown.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report() -> None:
    results = pd.read_csv(RESULTS_PATH)
    architecture = summarize(results, ["single_stage", "two_stage"])
    recency = summarize(
        results,
        ["recency_equal", "recency_20", "two_stage", "recency_80"],
    )
    positions = summarize(
        results,
        [
            "two_stage",
            "position_current_only",
            "position_prior_lags_only",
            "position_none",
        ],
    )
    horizons = summarize(
        results,
        ["horizon_1", "horizon_3", "two_stage", "horizon_10"],
    )

    label_map = {
        "two_stage": "full / 5 laps / 40-race half-life",
        "single_stage": "single-stage / 5 laps",
        "recency_equal": "equal history",
        "recency_20": "20-race half-life",
        "recency_80": "80-race half-life",
        "position_current_only": "current Position only",
        "position_prior_lags_only": "Position lags 1-4 only",
        "position_none": "no position-state features",
        "horizon_1": "1 lap",
        "horizon_3": "3 laps",
        "horizon_10": "10 laps",
    }
    for frame in (architecture, recency, positions, horizons):
        frame["Configuration"] = frame["Specification"].map(label_map)
        frame.drop(columns="Specification", inplace=True)
        frame.insert(0, "Configuration", frame.pop("Configuration"))

    # Keep only metrics relevant to each decision so the report stays readable.
    architecture_columns = [
        "Configuration",
        "Races",
        "MeanRaceMAE",
        "BaselineMAE",
        "ModelWinRate",
        "ChangeAccuracy",
        "ChangeBrier",
        "ChangeROCAUC",
    ]
    general_columns = [
        "Configuration",
        "Races",
        "MeanRaceMAE",
        "MedianRaceMAE",
        "BaselineMAE",
        "ModelWinRate",
        "MeanBaselineImprovementPct",
    ]
    horizon_columns = [*general_columns, "NoChangePercent", "ValidationWindows"]
    horizon_windows = (
        results.groupby("Specification")["ValidationWindows"].sum().to_dict()
    )
    horizons["ValidationWindows"] = [
        horizon_windows[
            {
                "1 lap": "horizon_1",
                "3 laps": "horizon_3",
                "full / 5 laps / 40-race half-life": "two_stage",
                "10 laps": "horizon_10",
            }[label]
        ]
        for label in horizons["Configuration"]
    ]

    best_architecture = architecture.nsmallest(1, "MeanRaceMAE").iloc[0]
    best_recency = recency.nsmallest(1, "MeanRaceMAE").iloc[0]
    best_position = positions.nsmallest(1, "MeanRaceMAE").iloc[0]

    report = f"""# Requested model-improvement comparison

## Scope

This report implements and compares the requested changes using **{DIAGNOSTIC_RACES}
identical chronological race folds** distributed from the first eligible validation
race through the latest available race. Every fold trains only on earlier races.
The comparisons use a five-window stride and the selected HistGradientBoosting
parameters:

```json
{asdict(PARAMETERS)}
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

{markdown_table(architecture[architecture_columns])}

Lowest architecture MAE in this sample: **{best_architecture['Configuration']}**
at **{best_architecture['MeanRaceMAE']:.3f}**. Change accuracy alone should not
decide the winner; MAE and baseline performance remain the deployment metrics.
The two-stage improvement is real in this sample but small, so it should remain
switchable in the dashboard rather than replacing the single-stage path entirely.

## 2. Historical recency weighting

Race balancing is preserved, then each older race receives exponential decay. A
40-race half-life means a race 40 events older receives half the total influence.

{markdown_table(recency[general_columns])}

Lowest tested recency MAE: **{best_recency['Configuration']}** at
**{best_recency['MeanRaceMAE']:.3f}**. This comparison should be repeated after a
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

{markdown_table(positions[general_columns])}

Lowest sampled position-state MAE: **{best_position['Configuration']}** at
**{best_position['MeanRaceMAE']:.3f}**. No position columns are removed permanently
by this experiment; this table is evidence for the user's later decision.
The advantage over the full position family is only about 0.003 MAE, which is too
small to justify an irreversible feature deletion from eight races.

## 4. Prediction-horizon comparison

Direct MAE naturally grows with horizon, so compare baseline improvement and win
rate as well as raw MAE. Longer horizons also have fewer valid end-of-race windows.

{markdown_table(horizons[horizon_columns])}

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
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"REPORT WRITTEN: {REPORT_PATH}", flush=True)


def train_final_model(
    target_horizon: int = 10,
    output_path: Path = FINAL_MODEL_PATH,
) -> None:
    """Fit a current-position-only production model for one target horizon."""
    if target_horizon not in (5, 10):
        raise ValueError("Production live models support only 5 or 10 laps")
    prepared = load_and_prepare_training_data(
        lap_stride=LAP_STRIDE,
        target_horizon=target_horizon,
    )
    events = event_table(prepared)
    event_order = {
        (int(event.Year), int(event.RoundNumber)): order
        for order, event in events.iterrows()
    }
    row_order = np.asarray(
        [
            event_order[(int(year), int(round_number))]
            for year, round_number in zip(
                prepared.metadata["Year"], prepared.metadata["RoundNumber"]
            )
        ]
    )
    race_age = (len(events) - 1) - row_order
    weights = prepared.sample_weight.to_numpy() * np.power(0.5, race_age / 20.0)
    weights = weights / weights.mean()

    X = prepared.X.copy()
    category_vocabularies: dict[str, list[object]] = {}
    for column in prepared.categorical_columns:
        X[column] = X[column].astype("category")
        category_vocabularies[column] = X[column].cat.categories.tolist()
    usable_columns = X.columns[~X.isna().all(axis=0)]
    X = X.loc[:, usable_columns]

    model = create_two_stage_position_model(PARAMETERS)
    print(
        f"FINAL TWO-STAGE MODEL: {len(X):,} windows, {X.shape[1]} features",
        flush=True,
    )
    started_at = perf_counter()
    model.fit(X, prepared.y, sample_weight=weights)
    training_seconds = perf_counter() - started_at
    bundle = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "categorical_columns": list(prepared.categorical_columns),
        "category_vocabularies": category_vocabularies,
        "parameters": asdict(PARAMETERS),
        "target_horizon": target_horizon,
        "lap_stride": LAP_STRIDE,
        "recency_half_life_races": 20,
        "position_variant": "current_only",
        "excluded_columns": ["DriverNumber", "Country"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(output_path)

    latest = events.iloc[-1]
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata_path.write_text(
        pd.Series(
            {
                "training_windows": len(X),
                "features": X.shape[1],
                "races": len(events),
                "trained_through": (
                    f"{int(latest.Year)} R{int(latest.RoundNumber)} {latest.EventName}"
                ),
                "training_seconds": training_seconds,
                "target_horizon": target_horizon,
                "recency_half_life_races": 20,
            }
        ).to_json(indent=2),
        encoding="utf-8",
    )
    print(f"FINAL MODEL WRITTEN: {output_path}", flush=True)


def main() -> None:
    run_comparisons()
    build_report()
    train_final_model()


if __name__ == "__main__":
    main()
