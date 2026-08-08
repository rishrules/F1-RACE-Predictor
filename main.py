from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator
from time import perf_counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


FEATURE_DATA_PATH = Path(__file__).resolve().parent / "data" / "features" / "position_5_laps"
TARGET_COLUMN = "TargetPositionChange"
PREDICTION_HORIZONS = (1, 3, 5, 10)
DEFAULT_TARGET_HORIZON = 10
DEFAULT_RECENCY_HALF_LIFE_RACES = 20.0

# These columns describe a sample or contain future labels. They are useful for
# splitting and evaluation, but must never be passed to the estimator as inputs.
TARGET_COLUMNS = (
    "TargetLapNumber",
    "TargetPosition",
    "TargetPositionChange",
    "TargetGapToLeaderSeconds",
)
MODEL_EXCLUDED_COLUMNS = ("DriverNumber", "Country")
METADATA_COLUMNS = (
    "Year",
    "RoundNumber",
    "EventName",
    "Session",
    # pandas adds these lowercase columns from the Parquet folder names.
    "year",
    "round",
)

# HistGradientBoosting can use pandas categorical columns directly. DriverNumber
# and Country are intentionally excluded from the estimator at the user's request.
CATEGORICAL_COLUMNS = (
    "Driver",
    "Team",
    "Circuit",
    "EventFormat",
    "Compound",
    "TrackStatus",
    "PitWindowPhase",
)


@dataclass
class PreparedTrainingData:
    """Features, label, metadata and weights before chronological splitting."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    sample_weight: pd.Series
    categorical_columns: tuple[str, ...]
    target_horizon: int


@dataclass
class WalkForwardSplit:
    """One race-level train/validation split in chronological order."""

    validation_year: int
    validation_round: int
    target_horizon: int
    X_train: pd.DataFrame
    y_train: pd.Series
    training_weights: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    validation_metadata: pd.DataFrame


@dataclass(frozen=True)
class ModelParameters:
    """The HistGradientBoosting settings controlled by the dashboard."""

    learning_rate: float = 0.08
    max_iter: int = 150
    max_leaf_nodes: int = 31
    max_depth: int | None = None
    min_samples_leaf: int = 40
    l2_regularization: float = 8.0


@dataclass
class TrainingResult:
    """Everything needed to evaluate one fitted model."""

    model: object
    split: WalkForwardSplit
    parameters: ModelParameters
    predictions: np.ndarray
    metrics: dict[str, float]
    training_seconds: float
    model_type: str = "Single-stage regressor"
    change_probabilities: np.ndarray | None = None


def create_hist_gradient_boosting_regressor(
    learning_rate: float = 0.08,
    max_iter: int = 150,
    max_leaf_nodes: int | None = 31,
    max_depth: int | None = None,
    min_samples_leaf: int = 40,
    l2_regularization: float = 8.0,
    early_stopping: bool | str = "auto",
    random_state: int | None = 42,
) -> HistGradientBoostingRegressor:
    """Create an untrained HistGradientBoosting position regressor.

    The returned estimator is intentionally not fitted. Pass training data to
    its ``fit`` method later, after preprocessing and chronological splitting.
    """
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=early_stopping,
        random_state=random_state,
    )


class TwoStagePositionChangeModel(RegressorMixin, BaseEstimator):
    """Classify whether a change occurs, then estimate its signed magnitude.

    The final prediction is the conditional position-change estimate multiplied
    by the probability that any change occurs. This is the expected signed
    change and avoids a hard discontinuity at an arbitrary probability cutoff.
    """

    def __init__(
        self,
        learning_rate: float = 0.08,
        max_iter: int = 150,
        max_leaf_nodes: int = 31,
        max_depth: int | None = None,
        min_samples_leaf: int = 40,
        l2_regularization: float = 8.0,
        random_state: int | None = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def _common_parameters(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "early_stopping": False,
            "random_state": self.random_state,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> "TwoStagePositionChangeModel":
        target = pd.Series(np.asarray(y), index=X.index, dtype=float)
        change_target = target.ne(0).astype(int)
        if change_target.nunique() < 2:
            raise ValueError("Two-stage training requires changed and unchanged targets")

        self.change_classifier_ = HistGradientBoostingClassifier(
            loss="log_loss", **self._common_parameters()
        )
        self.change_classifier_.fit(
            X, change_target, sample_weight=sample_weight
        )

        changed = change_target.eq(1)
        conditional_weights = None
        if sample_weight is not None:
            conditional_weights = np.asarray(sample_weight)[changed.to_numpy()]
            conditional_weights = conditional_weights / conditional_weights.mean()
        self.change_regressor_ = HistGradientBoostingRegressor(
            loss="squared_error", **self._common_parameters()
        )
        self.change_regressor_.fit(
            X.loc[changed],
            target.loc[changed],
            sample_weight=conditional_weights,
        )
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_components(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        change_probability = self.change_classifier_.predict_proba(X)[:, 1]
        conditional_change = self.change_regressor_.predict(X)
        return change_probability, conditional_change

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        change_probability, conditional_change = self.predict_components(X)
        return change_probability * conditional_change

    def staged_predict(self, X: pd.DataFrame) -> Iterator[np.ndarray]:
        """Yield probability-weighted predictions after each boosting stage."""
        probability_stages = self.change_classifier_.staged_predict_proba(X)
        regression_stages = self.change_regressor_.staged_predict(X)
        for probabilities, conditional_change in zip(
            probability_stages, regression_stages
        ):
            yield probabilities[:, 1] * conditional_change


def create_two_stage_position_model(
    parameters: ModelParameters,
) -> TwoStagePositionChangeModel:
    """Create an unfitted two-stage model from dashboard parameters."""
    return TwoStagePositionChangeModel(**asdict(parameters))


def target_column_for_horizon(data: pd.DataFrame, horizon: int) -> str:
    """Resolve the requested target while supporting legacy five-lap tables."""
    if horizon not in PREDICTION_HORIZONS:
        raise ValueError(f"target_horizon must be one of {PREDICTION_HORIZONS}")
    multi_horizon_name = f"TargetPositionChange_h{horizon}"
    if multi_horizon_name in data:
        return multi_horizon_name
    if horizon == 5 and TARGET_COLUMN in data:
        return TARGET_COLUMN
    raise ValueError(
        f"Feature dataset does not contain the {horizon}-lap target; "
        "regenerate features with f1_data.py --stage features"
    )


def load_and_prepare_training_data(
    feature_path: Path | str = FEATURE_DATA_PATH,
    lap_stride: int = 1,
    target_horizon: int = DEFAULT_TARGET_HORIZON,
) -> PreparedTrainingData:
    """Load the Parquet feature dataset and prepare it without fitting a model.

    ``lap_stride=1`` keeps every available five-lap window. A value of 5 keeps
    prediction points at laps 5, 10, 15, ... and therefore reduces the heavy
    overlap between neighbouring windows.
    """
    feature_frame = pd.read_parquet(feature_path)
    return prepare_training_frame(
        feature_frame,
        lap_stride=lap_stride,
        target_horizon=target_horizon,
    )


def prepare_training_frame(
    data: pd.DataFrame,
    lap_stride: int = 1,
    target_horizon: int = DEFAULT_TARGET_HORIZON,
) -> PreparedTrainingData:
    """Perform target selection, separation, validation and race weighting."""
    if lap_stride < 1:
        raise ValueError("lap_stride must be at least 1")

    selected_target = target_column_for_horizon(data, target_horizon)
    required_columns = {
        "Year",
        "RoundNumber",
        "EventName",
        "DriverNumber",
        "LapNumber",
        "Position",
        selected_target,
    }
    missing_required = required_columns.difference(data.columns)
    if missing_required:
        raise ValueError(
            "Feature dataset is missing required columns: "
            f"{sorted(missing_required)}"
        )

    prepared = data.copy()

    # STEP 1: choose position change over the selected horizon as the label.
    # A negative value means the driver gained positions; positive means lost.
    prepared[selected_target] = pd.to_numeric(
        prepared[selected_target], errors="coerce"
    )
    # Tail rows can have a short-horizon answer but no longer-horizon answer.
    # Filter only the target selected for this training run.
    prepared = prepared.loc[prepared[selected_target].notna()].copy()
    if prepared.empty:
        raise ValueError(f"No valid {target_horizon}-lap targets are available")

    # STEP 6: optionally reduce overlapping samples. With stride 5, adjacent
    # retained rows no longer share four out of their five input laps.
    if lap_stride > 1:
        lap_numbers = pd.to_numeric(prepared["LapNumber"], errors="coerce")
        prepared = prepared.loc[lap_numbers.mod(lap_stride).eq(0)].copy()

    prepared = prepared.sort_values(
        ["Year", "RoundNumber", "DriverNumber", "LapNumber"]
    ).reset_index(drop=True)

    # Each driver/race/lap should identify exactly one prediction window.
    window_key = ["Year", "RoundNumber", "DriverNumber", "LapNumber"]
    duplicate_count = int(prepared.duplicated(window_key).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate prediction windows")

    # STEP 2: keep identifiers for splitting, while excluding identifiers and
    # every future target from the estimator's input matrix.
    # Keep evaluation-only columns beside the identifiers. They make plots
    # understandable, but remain completely separate from the estimator's X.
    selected_target_position = f"TargetPosition_h{target_horizon}"
    evaluation_columns = (
        "Driver",
        "Team",
        "Circuit",
        "Country",
        "Position",
        selected_target_position,
        "TrackStatus",
    )
    available_metadata = [
        column
        for column in (*METADATA_COLUMNS, *evaluation_columns)
        if column in prepared.columns
    ]
    metadata = prepared[available_metadata].copy()
    if selected_target_position in metadata:
        metadata["TargetPosition"] = metadata.pop(selected_target_position)
    elif "TargetPosition" in prepared:
        metadata["TargetPosition"] = prepared["TargetPosition"]
    metadata["DriverNumber"] = prepared["DriverNumber"].astype("string")
    metadata["LapNumber"] = pd.to_numeric(prepared["LapNumber"], errors="coerce")

    y = prepared[selected_target].rename(
        f"PositionChangeNext{target_horizon}Laps"
    )
    # Every Target* column describes the future. Dropping by prefix prevents a
    # different horizon from leaking its answer into the selected model.
    columns_to_remove = [
        column
        for column in prepared.columns
        if column.startswith("Target")
        or column in METADATA_COLUMNS
        or column in MODEL_EXCLUDED_COLUMNS
        # Retain the current position, but remove lagged and rolling position
        # inputs. Controlled validation found the current value sufficient.
        or (column.startswith("Position_") and column != "Position")
    ]
    X = prepared.drop(columns=columns_to_remove)

    # STEP 3: preserve categories as readable strings for now. Categories are
    # learned from each training fold later so future driver/team names do not
    # leak backwards and unseen validation categories become missing values.
    available_categorical = tuple(
        column for column in CATEGORICAL_COLUMNS if column in X.columns
    )
    for column in available_categorical:
        X[column] = X[column].astype("string")

    # STEP 6: make every race contribute equal total weight. Without this,
    # longer races and races with more surviving drivers dominate the loss.
    race_window_count = metadata.groupby(["Year", "RoundNumber"])[
        "RoundNumber"
    ].transform("size")
    sample_weight = (1.0 / race_window_count).rename("RaceBalancedWeight")
    sample_weight = sample_weight / sample_weight.mean()

    return PreparedTrainingData(
        X=X,
        y=y,
        metadata=metadata,
        sample_weight=sample_weight,
        categorical_columns=available_categorical,
        target_horizon=target_horizon,
    )


def missing_value_report(prepared: PreparedTrainingData) -> pd.DataFrame:
    """STEP 4: summarize missing values without modifying or imputing them.

    Numeric NaNs are deliberately retained because HistGradientBoosting handles
    them natively and missing history can itself be informative.
    """
    report = pd.DataFrame(
        {
            "DataType": prepared.X.dtypes.astype(str),
            "MissingCount": prepared.X.isna().sum(),
            "MissingPercent": prepared.X.isna().mean().mul(100),
        }
    )
    return report.sort_values(
        ["MissingPercent", "MissingCount"], ascending=False
    )


def align_categorical_features(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    categorical_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn categorical vocabularies from training data only.

    A driver, team or compound unseen during training becomes a missing category
    in validation, which HistGradientBoosting can handle without target leakage.
    """
    train = X_train.copy()
    validation = X_validation.copy()

    for column in categorical_columns:
        train[column] = train[column].astype("category")
        training_categories = train[column].cat.categories
        validation[column] = pd.Categorical(
            validation[column], categories=training_categories
        )
    return train, validation


def remove_empty_training_features(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns that contain no usable value in the training period.

    Very early folds have no previous-season or previous-circuit history. Some
    corresponding columns are therefore entirely NaN. HistGradientBoosting
    cannot find bin boundaries for an all-missing numeric feature, so the
    training fold decides which columns are usable and validation follows it.
    """
    usable_columns = X_train.columns[~X_train.isna().all(axis=0)]
    return X_train.loc[:, usable_columns], X_validation.loc[:, usable_columns]


def recency_adjusted_training_weights(
    prepared: PreparedTrainingData,
    training_mask: pd.Series,
    row_order: pd.Series,
    validation_order: int,
    recency_half_life_races: float | None,
) -> pd.Series:
    """Combine equal-race weights with optional exponential recency decay.

    A half-life of 40 means a race 40 events older than the newest training
    race receives half as much total influence. The calculation uses only the
    chronological event order available before validation.
    """
    weights = prepared.sample_weight.loc[training_mask].copy()
    if recency_half_life_races is not None:
        if recency_half_life_races <= 0:
            raise ValueError("recency_half_life_races must be positive or None")
        race_age = (validation_order - 1) - row_order.loc[training_mask]
        weights *= np.power(0.5, race_age / recency_half_life_races)
    return weights / weights.mean()


def walk_forward_splits(
    prepared: PreparedTrainingData,
    min_training_races: int = 40,
    recency_half_life_races: float | None = DEFAULT_RECENCY_HALF_LIFE_RACES,
) -> Iterator[WalkForwardSplit]:
    """STEP 5: yield one chronological validation race at a time.

    No row from the validation race, or any later race, is included in its
    training data. All overlapping windows from a race stay in the same split.
    """
    if min_training_races < 1:
        raise ValueError("min_training_races must be at least 1")

    events = (
        prepared.metadata[["Year", "RoundNumber"]]
        .drop_duplicates()
        .sort_values(["Year", "RoundNumber"])
        .reset_index(drop=True)
    )
    if len(events) <= min_training_races:
        raise ValueError(
            f"Need more than {min_training_races} races; found {len(events)}"
        )

    event_order = {
        (int(event.Year), int(event.RoundNumber)): order
        for order, event in events.iterrows()
    }
    row_order = pd.Series(
        [
            event_order[(int(year), int(round_number))]
            for year, round_number in zip(
                prepared.metadata["Year"], prepared.metadata["RoundNumber"]
            )
        ],
        index=prepared.metadata.index,
    )

    for validation_order in range(min_training_races, len(events)):
        validation_event = events.iloc[validation_order]
        training_mask = row_order.lt(validation_order)
        validation_mask = row_order.eq(validation_order)

        X_train, X_validation = align_categorical_features(
            prepared.X.loc[training_mask],
            prepared.X.loc[validation_mask],
            prepared.categorical_columns,
        )
        X_train, X_validation = remove_empty_training_features(
            X_train, X_validation
        )
        training_weights = recency_adjusted_training_weights(
            prepared,
            training_mask,
            row_order,
            validation_order,
            recency_half_life_races,
        )

        yield WalkForwardSplit(
            validation_year=int(validation_event["Year"]),
            validation_round=int(validation_event["RoundNumber"]),
            target_horizon=prepared.target_horizon,
            X_train=X_train,
            y_train=prepared.y.loc[training_mask],
            training_weights=training_weights,
            X_validation=X_validation,
            y_validation=prepared.y.loc[validation_mask],
            validation_metadata=prepared.metadata.loc[validation_mask],
        )


def make_walk_forward_split(
    prepared: PreparedTrainingData,
    validation_year: int,
    validation_round: int,
    recency_half_life_races: float | None = DEFAULT_RECENCY_HALF_LIFE_RACES,
) -> WalkForwardSplit:
    """Create one requested race-level split without building every other fold."""
    year = prepared.metadata["Year"].astype(int)
    round_number = prepared.metadata["RoundNumber"].astype(int)
    training_mask = year.lt(validation_year) | (
        year.eq(validation_year) & round_number.lt(validation_round)
    )
    validation_mask = year.eq(validation_year) & round_number.eq(validation_round)

    if not validation_mask.any():
        raise ValueError("The requested validation race is not in the feature data")
    if not training_mask.any():
        raise ValueError("The requested validation race has no earlier training races")

    X_train, X_validation = align_categorical_features(
        prepared.X.loc[training_mask],
        prepared.X.loc[validation_mask],
        prepared.categorical_columns,
    )
    X_train, X_validation = remove_empty_training_features(
        X_train, X_validation
    )
    events = (
        prepared.metadata[["Year", "RoundNumber"]]
        .drop_duplicates()
        .sort_values(["Year", "RoundNumber"])
        .reset_index(drop=True)
    )
    event_order = {
        (int(event.Year), int(event.RoundNumber)): order
        for order, event in events.iterrows()
    }
    row_order = pd.Series(
        [
            event_order[(int(event_year), int(event_round))]
            for event_year, event_round in zip(year, round_number)
        ],
        index=prepared.metadata.index,
    )
    validation_order = event_order[(validation_year, validation_round)]
    weights = recency_adjusted_training_weights(
        prepared,
        training_mask,
        row_order,
        validation_order,
        recency_half_life_races,
    )

    return WalkForwardSplit(
        validation_year=validation_year,
        validation_round=validation_round,
        target_horizon=prepared.target_horizon,
        X_train=X_train,
        y_train=prepared.y.loc[training_mask],
        training_weights=weights,
        X_validation=X_validation,
        y_validation=prepared.y.loc[validation_mask],
        validation_metadata=prepared.metadata.loc[validation_mask],
    )


def train_and_evaluate(
    split: WalkForwardSplit,
    parameters: ModelParameters,
    two_stage: bool = False,
) -> TrainingResult:
    """Fit one model and evaluate it against the no-position-change baseline."""
    if two_stage:
        model = create_two_stage_position_model(parameters)
        model_type = "Two-stage classifier + conditional regressor"
    else:
        model = create_hist_gradient_boosting_regressor(
            **asdict(parameters),
            # Internal random validation would violate our chronological design.
            # The dashboard performs explicit race-level validation instead.
            early_stopping=False,
        )
        model_type = "Single-stage regressor"

    started_at = perf_counter()
    model.fit(
        split.X_train,
        split.y_train,
        sample_weight=split.training_weights,
    )
    training_seconds = perf_counter() - started_at
    predictions = model.predict(split.X_validation)
    change_probabilities = None
    if two_stage:
        change_probabilities, _ = model.predict_components(split.X_validation)
    baseline_predictions = np.zeros(len(split.y_validation))

    model_mae = mean_absolute_error(split.y_validation, predictions)
    baseline_mae = mean_absolute_error(split.y_validation, baseline_predictions)
    metrics = {
        "MAE": float(model_mae),
        "RMSE": float(
            np.sqrt(mean_squared_error(split.y_validation, predictions))
        ),
        "Baseline MAE": float(baseline_mae),
        "Baseline improvement %": float(
            100 * (baseline_mae - model_mae) / baseline_mae
            if baseline_mae > 0
            else 0.0
        ),
    }
    if change_probabilities is not None:
        actual_change = split.y_validation.ne(0).astype(int)
        predicted_change = change_probabilities >= 0.5
        metrics["Change accuracy"] = float(
            accuracy_score(actual_change, predicted_change)
        )
        metrics["Change Brier score"] = float(
            brier_score_loss(actual_change, change_probabilities)
        )
        if actual_change.nunique() == 2:
            metrics["Change ROC AUC"] = float(
                roc_auc_score(actual_change, change_probabilities)
            )
    return TrainingResult(
        model=model,
        split=split,
        parameters=parameters,
        predictions=predictions,
        metrics=metrics,
        training_seconds=training_seconds,
        model_type=model_type,
        change_probabilities=change_probabilities,
    )


def prediction_frame(result: TrainingResult) -> pd.DataFrame:
    """Combine predictions with labels and readable race metadata."""
    frame = result.split.validation_metadata.reset_index(drop=True).copy()
    frame["ActualPositionChange"] = result.split.y_validation.to_numpy()
    frame["PredictedPositionChange"] = result.predictions
    frame["Residual"] = (
        frame["ActualPositionChange"] - frame["PredictedPositionChange"]
    )
    frame["AbsoluteError"] = frame["Residual"].abs()
    if result.change_probabilities is not None:
        frame["PredictedChangeProbability"] = result.change_probabilities

    if "Position" in frame:
        current_position = pd.to_numeric(frame["Position"], errors="coerce")
        frame["PredictedTargetPosition"] = (
            current_position + frame["PredictedPositionChange"]
        ).clip(lower=1)
    return frame


def record_experiment(
    result: TrainingResult,
    label: str,
) -> None:
    """Keep compact experiment results for comparison during this app session."""
    if "experiments" not in st.session_state:
        st.session_state.experiments = []

    record = {
        "Experiment": len(st.session_state.experiments) + 1,
        "Label": label,
        "ValidationRace": (
            f"{result.split.validation_year} R{result.split.validation_round}"
        ),
        "TargetHorizon": result.split.target_horizon,
        "ModelType": result.model_type,
        **asdict(result.parameters),
        **result.metrics,
        "TrainingSeconds": result.training_seconds,
    }
    st.session_state.experiments.append(record)


def staged_error_frame(result: TrainingResult) -> pd.DataFrame:
    """Calculate training and validation MAE after every boosting iteration."""
    # Cap the training sample to keep this diagnostic responsive. The validation
    # curve still uses every row from the held-out race.
    sample_size = min(20_000, len(result.split.X_train))
    sample_rows = np.linspace(
        0, len(result.split.X_train) - 1, sample_size, dtype=int
    )
    X_train_sample = result.split.X_train.iloc[sample_rows]
    y_train_sample = result.split.y_train.iloc[sample_rows]

    rows: list[dict[str, float | int | str]] = []
    staged_train = result.model.staged_predict(X_train_sample)
    staged_validation = result.model.staged_predict(result.split.X_validation)
    for iteration, (train_prediction, validation_prediction) in enumerate(
        zip(staged_train, staged_validation), start=1
    ):
        rows.extend(
            (
                {
                    "Iteration": iteration,
                    "Dataset": "Training sample",
                    "MAE": mean_absolute_error(y_train_sample, train_prediction),
                },
                {
                    "Iteration": iteration,
                    "Dataset": "Validation race",
                    "MAE": mean_absolute_error(
                        result.split.y_validation, validation_prediction
                    ),
                },
            )
        )
    return pd.DataFrame(rows)


def parameter_choices(parameter_name: str) -> list[float | int]:
    """Small, interpretable search ranges used by the interactive experiments."""
    choices = {
        "learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2],
        "max_iter": [50, 100, 150, 200, 300],
        "max_leaf_nodes": [7, 15, 31, 63, 127],
        "max_depth": [3, 5, 7, 10, 15],
        "min_samples_leaf": [10, 20, 30, 50, 100],
        "l2_regularization": [0.0, 0.1, 1.0, 5.0, 10.0],
    }
    return choices[parameter_name]


def parameters_with_change(
    parameters: ModelParameters,
    parameter_name: str,
    value: float | int,
) -> ModelParameters:
    """Return a new immutable parameter set with one value replaced."""
    values = asdict(parameters)
    integer_parameters = {
        "max_iter",
        "max_leaf_nodes",
        "max_depth",
        "min_samples_leaf",
    }
    values[parameter_name] = (
        int(value) if parameter_name in integer_parameters else float(value)
    )
    return ModelParameters(**values)


@st.cache_data(show_spinner="Loading the feature dataset...")
def cached_training_data(
    lap_stride: int,
    target_horizon: int,
) -> PreparedTrainingData:
    """Load and prepare Parquet once for each stride/horizon combination."""
    return load_and_prepare_training_data(
        lap_stride=lap_stride,
        target_horizon=target_horizon,
    )


def model_parameter_controls() -> ModelParameters:
    """Render readable sidebar controls and return their current values."""
    st.sidebar.header("Model parameters")
    learning_rate = st.sidebar.slider(
        "Learning rate", 0.01, 0.30, 0.08, 0.01,
        help="Contribution made by each new tree.",
    )
    max_iter = st.sidebar.slider(
        "Boosting iterations", 25, 400, 150, 25,
        help="Maximum number of trees added sequentially.",
    )
    max_leaf_nodes = st.sidebar.slider(
        "Maximum leaf nodes", 3, 127, 31, 2,
    )
    limit_depth = st.sidebar.checkbox("Limit tree depth", value=False)
    max_depth = (
        st.sidebar.slider("Maximum tree depth", 2, 20, 8)
        if limit_depth
        else None
    )
    min_samples_leaf = st.sidebar.slider(
        "Minimum samples per leaf", 5, 200, 40, 5,
    )
    l2_regularization = st.sidebar.slider(
        "L2 regularization", 0.0, 20.0, 8.0, 0.1,
    )
    return ModelParameters(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
    )


def render_training_curve(result: TrainingResult) -> None:
    """VIEW 1: training and validation error by boosting iteration."""
    curve = staged_error_frame(result)
    chart = (
        alt.Chart(curve)
        .mark_line()
        .encode(
            x=alt.X("Iteration:Q", title="Boosting iteration"),
            y=alt.Y("MAE:Q", title="Mean absolute error"),
            color="Dataset:N",
            tooltip=["Iteration:Q", "Dataset:N", alt.Tooltip("MAE:Q", format=".3f")],
        )
        .interactive()
    )
    st.altair_chart(chart, width="stretch")


def render_prediction_diagnostics(result: TrainingResult) -> None:
    """Render views 5, 6 and 7 from the selected validation race."""
    predictions = prediction_frame(result)

    st.subheader("5. Model versus baseline")
    comparison = pd.DataFrame(
        {
            "Method": [result.model_type, "No position change"],
            "MAE": [result.metrics["MAE"], result.metrics["Baseline MAE"]],
        }
    )
    st.altair_chart(
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            x=alt.X("Method:N", sort=None),
            y=alt.Y("MAE:Q"),
            color="Method:N",
            tooltip=["Method:N", alt.Tooltip("MAE:Q", format=".3f")],
        ),
        width="stretch",
    )

    st.subheader("6. Actual versus predicted position change")
    scatter = (
        alt.Chart(predictions)
        .mark_circle(opacity=0.55)
        .encode(
            x=alt.X("ActualPositionChange:Q", title="Actual change"),
            y=alt.Y("PredictedPositionChange:Q", title="Predicted change"),
            color=alt.Color("Driver:N"),
            tooltip=[
                "Driver:N",
                "LapNumber:Q",
                alt.Tooltip("ActualPositionChange:Q", format=".2f"),
                alt.Tooltip("PredictedPositionChange:Q", format=".2f"),
            ],
        )
        .interactive()
    )
    domain = pd.concat(
        [predictions["ActualPositionChange"], predictions["PredictedPositionChange"]]
    )
    diagonal_data = pd.DataFrame({"value": [domain.min(), domain.max()]})
    diagonal = alt.Chart(diagonal_data).mark_line(color="gray", strokeDash=[5, 5]).encode(
        x="value:Q", y="value:Q"
    )
    st.altair_chart(scatter + diagonal, width="stretch")

    st.subheader("7. Residual distribution")
    histogram = (
        alt.Chart(predictions)
        .mark_bar()
        .encode(
            x=alt.X("Residual:Q", bin=alt.Bin(maxbins=35)),
            y=alt.Y("count():Q", title="Prediction windows"),
            tooltip=["count():Q"],
        )
        .interactive()
    )
    boxplot = alt.Chart(predictions).mark_boxplot(size=35).encode(
        x=alt.X("Residual:Q", title="Actual minus predicted change")
    )
    st.altair_chart(alt.vconcat(histogram, boxplot), width="stretch")


def render_error_breakdown(result: TrainingResult) -> None:
    """VIEW 8: compare errors across readable racing groups."""
    predictions = prediction_frame(result)
    available_groups = [
        column
        for column in ("Driver", "Team", "Circuit", "Country", "TrackStatus")
        if column in predictions
    ]
    group = st.selectbox("Break errors down by", available_groups, key="error_group")
    grouped = (
        predictions.groupby(group, dropna=False)
        .agg(MAE=("AbsoluteError", "mean"), Windows=("AbsoluteError", "size"))
        .reset_index()
        .sort_values("MAE", ascending=False)
    )
    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X("MAE:Q"),
            y=alt.Y(f"{group}:N", sort="-x"),
            color=alt.Color("MAE:Q", scale=alt.Scale(scheme="orangered")),
            tooltip=[f"{group}:N", alt.Tooltip("MAE:Q", format=".3f"), "Windows:Q"],
        )
        .interactive()
    )
    st.altair_chart(chart, width="stretch")


def render_race_timeline(result: TrainingResult) -> None:
    """VIEW 9: compare actual, predicted and current positions through a race."""
    predictions = prediction_frame(result)
    if "Driver" not in predictions or "TargetPosition" not in predictions:
        st.info("Driver or target-position metadata is unavailable.")
        return

    drivers = sorted(predictions["Driver"].dropna().astype(str).unique())
    selected_driver = st.selectbox("Driver", drivers, key="timeline_driver")
    driver_rows = predictions.loc[
        predictions["Driver"].astype(str).eq(selected_driver)
    ].copy()
    driver_rows["Actual target position"] = pd.to_numeric(
        driver_rows["TargetPosition"], errors="coerce"
    )
    driver_rows["Current position"] = pd.to_numeric(
        driver_rows["Position"], errors="coerce"
    )
    timeline = driver_rows.melt(
        id_vars="LapNumber",
        value_vars=[
            "Actual target position",
            "PredictedTargetPosition",
            "Current position",
        ],
        var_name="Series",
        value_name="PlottedPosition",
    )
    chart = (
        alt.Chart(timeline)
        .mark_line(point=True)
        .encode(
            x=alt.X("LapNumber:Q", title="Prediction lap"),
            y=alt.Y("PlottedPosition:Q", title="Position", sort="descending"),
            color="Series:N",
            tooltip=[
                "LapNumber:Q",
                "Series:N",
                alt.Tooltip("PlottedPosition:Q", title="Position", format=".2f"),
            ],
        )
        .interactive()
    )
    st.altair_chart(chart, width="stretch")


def render_permutation_importance(result: TrainingResult) -> None:
    """VIEW 10: validation-set permutation importance."""
    repeats = st.slider("Permutation repeats", 2, 10, 3)
    if st.button("Calculate permutation importance"):
        with st.spinner("Permuting validation features..."):
            importance = permutation_importance(
                result.model,
                result.split.X_validation,
                result.split.y_validation,
                scoring="neg_mean_absolute_error",
                n_repeats=repeats,
                random_state=42,
                n_jobs=-1,
            )
        importance_frame = pd.DataFrame(
            {
                "Feature": result.split.X_validation.columns,
                "Importance": importance.importances_mean,
                "StandardDeviation": importance.importances_std,
            }
        ).nlargest(25, "Importance")
        st.session_state.permutation_frame = importance_frame

    if "permutation_frame" in st.session_state:
        chart = (
            alt.Chart(st.session_state.permutation_frame)
            .mark_bar()
            .encode(
                x=alt.X("Importance:Q", title="Increase in validation MAE"),
                y=alt.Y("Feature:N", sort="-x"),
                color=alt.Color("Importance:Q", scale=alt.Scale(scheme="blues")),
                tooltip=[
                    "Feature:N",
                    alt.Tooltip("Importance:Q", format=".4f"),
                    alt.Tooltip("StandardDeviation:Q", format=".4f"),
                ],
            )
            .interactive()
        )
        st.altair_chart(chart, width="stretch")


def render_partial_dependence(result: TrainingResult) -> None:
    """VIEW 11: model response while one numeric feature is varied."""
    numeric_features = [
        column
        for column in result.split.X_validation.columns
        if pd.api.types.is_numeric_dtype(result.split.X_validation[column])
        and result.split.X_validation[column].nunique(dropna=True) > 1
    ]
    feature = st.selectbox("Numeric feature", numeric_features, key="pdp_feature")
    grid_points = st.slider("Grid points", 10, 50, 25)

    if st.button("Calculate partial dependence"):
        source = result.split.X_validation
        if len(source) > 2_000:
            source = source.sample(2_000, random_state=42)
        valid_values = pd.to_numeric(source[feature], errors="coerce").dropna()
        low, high = valid_values.quantile([0.05, 0.95])
        grid = np.linspace(low, high, grid_points)
        rows = []
        for value in grid:
            changed = source.copy()
            changed[feature] = value
            rows.append(
                {
                    "FeatureValue": value,
                    "AveragePrediction": result.model.predict(changed).mean(),
                }
            )
        st.session_state.partial_dependence_frame = pd.DataFrame(rows)
        st.session_state.partial_dependence_feature = feature

    if "partial_dependence_frame" in st.session_state:
        shown_feature = st.session_state.partial_dependence_feature
        chart = (
            alt.Chart(st.session_state.partial_dependence_frame)
            .mark_line(point=True)
            .encode(
                x=alt.X("FeatureValue:Q", title=shown_feature),
                y=alt.Y(
                    "AveragePrediction:Q",
                    title="Average predicted position change",
                ),
                tooltip=[
                    alt.Tooltip("FeatureValue:Q", format=".3f"),
                    alt.Tooltip("AveragePrediction:Q", format=".3f"),
                ],
            )
            .interactive()
        )
        st.altair_chart(chart, width="stretch")


def render_experiment_history() -> None:
    """VIEW 12: experiment table and normalized parallel-coordinates plot."""
    experiments = pd.DataFrame(st.session_state.get("experiments", []))
    if experiments.empty:
        st.info("Train a model or run a parameter experiment to create history.")
        return

    st.dataframe(experiments, width="stretch", hide_index=True)
    st.download_button(
        "Download experiment history",
        experiments.to_csv(index=False),
        file_name="hist_gradient_boosting_experiments.csv",
        mime="text/csv",
    )

    plot_columns = [
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "MAE",
        "RMSE",
    ]
    normalized = experiments[["Experiment", *plot_columns]].copy()
    for column in plot_columns:
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        value_range = numeric.max() - numeric.min()
        normalized[column] = (
            (numeric - numeric.min()) / value_range if value_range else 0.5
        )
    long_frame = normalized.melt(
        id_vars="Experiment",
        var_name="ParameterOrMetric",
        value_name="NormalizedValue",
    )
    chart = (
        alt.Chart(long_frame)
        .mark_line(point=True, opacity=0.65)
        .encode(
            x=alt.X("ParameterOrMetric:N", sort=plot_columns, title=None),
            y=alt.Y("NormalizedValue:Q", title="Normalized value (0 to 1)"),
            detail="Experiment:N",
            color=alt.Color("Experiment:N"),
            tooltip=["Experiment:N", "ParameterOrMetric:N", "NormalizedValue:Q"],
        )
        .interactive()
    )
    st.altair_chart(chart, width="stretch")


def run_dashboard() -> None:
    """Interactive training and diagnostics application containing all 12 views."""
    st.set_page_config(page_title="F1 Position Model", layout="wide")
    st.title("F1 10-lap position prediction")
    st.caption(
        "HistGradientBoostingRegressor trained only on races before the selected "
        "validation race. Negative target values mean positions were gained."
    )

    parameters = model_parameter_controls()
    st.sidebar.header("Training data")
    target_horizon = DEFAULT_TARGET_HORIZON
    recency_half_life = DEFAULT_RECENCY_HALF_LIFE_RACES
    st.sidebar.info(
        "Fixed model: two-stage, 10-lap horizon, 20-race recency half-life, "
        "and current Position only."
    )
    lap_stride = st.sidebar.select_slider(
        "Lap-window stride",
        options=[1, 2, 5, 10],
        value=5,
        help="Stride 5 reduces overlap and makes interactive experiments faster.",
    )
    try:
        prepared = cached_training_data(lap_stride, target_horizon)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Could not load training features: {error}")
        st.stop()

    events = (
        prepared.metadata[["Year", "RoundNumber", "EventName"]]
        .drop_duplicates(["Year", "RoundNumber"])
        .sort_values(["Year", "RoundNumber"])
        .reset_index(drop=True)
    )
    # Require a useful amount of history while still allowing 2020 onward.
    eligible_events = events.iloc[min(40, max(1, len(events) - 1)) :].copy()
    eligible_events["Label"] = eligible_events.apply(
        lambda row: f"{int(row['Year'])} R{int(row['RoundNumber'])}: {row['EventName']}",
        axis=1,
    )
    selected_label = st.sidebar.selectbox(
        "Validation race",
        eligible_events["Label"].tolist(),
        index=len(eligible_events) - 1,
    )
    selected_event = eligible_events.loc[eligible_events["Label"].eq(selected_label)].iloc[0]

    st.sidebar.info(
        f"{len(prepared.X):,} prediction windows and {prepared.X.shape[1]} input features"
    )
    if st.sidebar.button("Train selected model", type="primary"):
        split = make_walk_forward_split(
            prepared,
            int(selected_event["Year"]),
            int(selected_event["RoundNumber"]),
            recency_half_life_races=recency_half_life,
        )
        with st.spinner("Training two-stage model..."):
            result = train_and_evaluate(
                split,
                parameters,
                two_stage=True,
            )
        st.session_state.training_result = result
        record_experiment(result, "Selected model")
        # Diagnostics calculated for an older model must not be shown as current.
        for key in ("permutation_frame", "partial_dependence_frame"):
            st.session_state.pop(key, None)

    result = st.session_state.get("training_result")
    # Discard a result retained by Streamlit from an older configurable build.
    if result is not None and (
        result.split.target_horizon != DEFAULT_TARGET_HORIZON
        or result.change_probabilities is None
        or any(
            column.startswith("Position_")
            for column in result.split.X_train.columns
        )
    ):
        st.session_state.pop("training_result", None)
        result = None
    if result is None:
        st.info("Choose the parameters and validation race, then click **Train selected model**.")
        with st.expander("Missing-value report"):
            st.dataframe(missing_value_report(prepared), width="stretch")
        return

    st.success(
        f"{result.model_type} trained on {len(result.split.X_train):,} windows in "
        f"{result.training_seconds:.1f} seconds; validated on "
        f"{result.split.validation_year} round {result.split.validation_round} "
        f"at a {result.split.target_horizon}-lap horizon."
    )
    metric_columns = st.columns(4)
    for column, (name, value) in zip(metric_columns, result.metrics.items()):
        column.metric(name, f"{value:.3f}")
    if result.change_probabilities is not None:
        st.caption(
            "Change classifier — "
            f"accuracy: {result.metrics['Change accuracy']:.3f}, "
            f"Brier score: {result.metrics['Change Brier score']:.3f}, "
            f"ROC AUC: {result.metrics.get('Change ROC AUC', float('nan')):.3f}."
        )

    with st.expander("1. Training and validation error by iteration", expanded=True):
        render_training_curve(result)

    with st.expander("2. One-parameter performance experiment"):
        parameter_name = st.selectbox(
            "Parameter to vary",
            list(asdict(parameters)),
            key="sweep_parameter",
        )
        values = parameter_choices(parameter_name)
        if st.button("Run parameter sweep"):
            rows = []
            progress = st.progress(0)
            for position, value in enumerate(values, start=1):
                changed = parameters_with_change(parameters, parameter_name, value)
                candidate = train_and_evaluate(
                    result.split,
                    changed,
                    two_stage=True,
                )
                record_experiment(candidate, f"Sweep {parameter_name}={value}")
                rows.append({"Value": value, **candidate.metrics})
                progress.progress(position / len(values))
            st.session_state.parameter_sweep = pd.DataFrame(rows)
            st.session_state.parameter_sweep_name = parameter_name
        if "parameter_sweep" in st.session_state:
            chart = (
                alt.Chart(st.session_state.parameter_sweep)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Value:Q", title=st.session_state.parameter_sweep_name),
                    y=alt.Y("MAE:Q"),
                    tooltip=["Value:Q", alt.Tooltip("MAE:Q", format=".3f")],
                )
                .interactive()
            )
            st.altair_chart(chart, width="stretch")

    with st.expander("3. Two-parameter performance heatmap"):
        parameter_names = list(asdict(parameters))
        first_parameter = st.selectbox("First parameter", parameter_names, index=0)
        second_parameter = st.selectbox("Second parameter", parameter_names, index=1)
        if first_parameter == second_parameter:
            st.warning("Choose two different parameters.")
        elif st.button("Run heatmap experiment"):
            first_values = parameter_choices(first_parameter)[1:4]
            second_values = parameter_choices(second_parameter)[1:4]
            rows = []
            progress = st.progress(0)
            total = len(first_values) * len(second_values)
            completed = 0
            for first_value in first_values:
                for second_value in second_values:
                    changed = parameters_with_change(
                        parameters, first_parameter, first_value
                    )
                    changed = parameters_with_change(
                        changed, second_parameter, second_value
                    )
                    candidate = train_and_evaluate(
                        result.split,
                        changed,
                        two_stage=True,
                    )
                    record_experiment(
                        candidate,
                        f"Heatmap {first_parameter}={first_value}, "
                        f"{second_parameter}={second_value}",
                    )
                    rows.append(
                        {
                            "FirstValue": str(first_value),
                            "SecondValue": str(second_value),
                            "MAE": candidate.metrics["MAE"],
                        }
                    )
                    completed += 1
                    progress.progress(completed / total)
            st.session_state.heatmap_frame = pd.DataFrame(rows)
            st.session_state.heatmap_names = (first_parameter, second_parameter)
        if "heatmap_frame" in st.session_state:
            first_name, second_name = st.session_state.heatmap_names
            heatmap = (
                alt.Chart(st.session_state.heatmap_frame)
                .mark_rect()
                .encode(
                    x=alt.X("FirstValue:N", title=first_name),
                    y=alt.Y("SecondValue:N", title=second_name),
                    color=alt.Color("MAE:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True)),
                    tooltip=["FirstValue:N", "SecondValue:N", alt.Tooltip("MAE:Q", format=".3f")],
                )
            )
            labels = heatmap.mark_text().encode(text=alt.Text("MAE:Q", format=".3f"), color=alt.value("black"))
            st.altair_chart(heatmap + labels, width="stretch")

    with st.expander("4. Walk-forward performance across races"):
        race_count = st.slider("Number of recent validation races", 2, 20, 5)
        if st.button("Run walk-forward evaluation"):
            selected_index = events.index[
                events["Year"].eq(int(selected_event["Year"]))
                & events["RoundNumber"].eq(int(selected_event["RoundNumber"]))
            ][0]
            validation_events = events.iloc[
                max(1, selected_index - race_count + 1) : selected_index + 1
            ]
            rows = []
            progress = st.progress(0)
            for position, event in enumerate(validation_events.itertuples(), start=1):
                split = make_walk_forward_split(
                    prepared,
                    int(event.Year),
                    int(event.RoundNumber),
                    recency_half_life_races=recency_half_life,
                )
                candidate = train_and_evaluate(
                    split,
                    parameters,
                    two_stage=True,
                )
                record_experiment(candidate, "Walk-forward")
                rows.extend(
                    [
                        {
                            "Race": f"{event.Year} R{event.RoundNumber}",
                            "Method": "Model",
                            "MAE": candidate.metrics["MAE"],
                        },
                        {
                            "Race": f"{event.Year} R{event.RoundNumber}",
                            "Method": "Baseline",
                            "MAE": candidate.metrics["Baseline MAE"],
                        },
                    ]
                )
                progress.progress(position / len(validation_events))
            st.session_state.walk_forward_frame = pd.DataFrame(rows)
        if "walk_forward_frame" in st.session_state:
            walk_chart = (
                alt.Chart(st.session_state.walk_forward_frame)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Race:N", sort=None),
                    y="MAE:Q",
                    color="Method:N",
                    tooltip=["Race:N", "Method:N", alt.Tooltip("MAE:Q", format=".3f")],
                )
                .interactive()
            )
            st.altair_chart(walk_chart, width="stretch")

    with st.expander("5–7. Baseline, prediction and residual diagnostics"):
        render_prediction_diagnostics(result)
    with st.expander("8. Error by driver, team and race conditions"):
        render_error_breakdown(result)
    with st.expander("9. Driver race-position timeline"):
        render_race_timeline(result)
    with st.expander("10. Permutation feature importance"):
        render_permutation_importance(result)
    with st.expander("11. Partial-dependence explorer"):
        render_partial_dependence(result)
    with st.expander("12. Experiment history and parallel coordinates"):
        render_experiment_history()

    with st.expander("Data-quality: missing values"):
        st.dataframe(missing_value_report(prepared), width="stretch")


if __name__ == "__main__":
    run_dashboard()
