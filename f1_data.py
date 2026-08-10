"""Ingest FastF1 data and build five-lap, model-ready feature tables.

This module deliberately stops at dataset creation. It does not create, fit,
evaluate, or save a machine-learning model.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import fastf1
import numpy as np
import pandas as pd
from fastf1.exceptions import DataNotLoadedError, RateLimitExceededError


START_YEAR = 2018
END_YEAR = 2026
WINDOW_LAPS = 5
PREDICTION_HORIZONS = (1, 3, 5, 10)
PRIMARY_PREDICTION_HORIZON = 5

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "raw" / "fastf1_cache"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_DIR = ROOT / "data" / "features" / "position_5_laps"
REQUIRED_FEATURE_COLUMNS = {
    "TargetPositionChange_h1",
    "TargetPositionChange_h3",
    "TargetPositionChange_h5",
    "TargetPositionChange_h10",
    "Sector1ToFieldSeconds_lag0",
    "VirtualSafetyCarThisLap",
    "PitWindowPhase",
    "LapsSincePitStop",
    "EstimatedTyreLifeAdvantageLaps",
}
BASE_TABLES = ("laps", "results", "weather", "qualifying")
EVENT_CONTEXT: dict[tuple[int, int], dict[str, str]] = {}
TABLE_CACHE: dict[str, pd.DataFrame] = {}


def completed_race(event: pd.Series, now: pd.Timestamp) -> bool:
    """Return whether an event's race session has already started."""
    race_date = event.get("Session5DateUtc")
    if pd.isna(race_date):
        race_date = event.get("EventDate")
    if pd.isna(race_date):
        return False

    race_date = pd.Timestamp(race_date)
    if race_date.tzinfo is None:
        race_date = race_date.tz_localize("UTC")
    else:
        race_date = race_date.tz_convert("UTC")
    return race_date <= now


def partition_path(root: Path, year: int, round_number: int) -> Path:
    """Return a year/round partition path that cannot collide across years."""
    return root / f"year={year}" / f"round={round_number:02d}" / "data.parquet"


def table_path(table: str, year: int, round_number: int) -> Path:
    return partition_path(PROCESSED_DIR / table, year, round_number)


def feature_path(year: int, round_number: int) -> Path:
    return partition_path(FEATURE_DIR, year, round_number)


def feature_partition_is_current(path: Path) -> bool:
    """Return whether a stored feature partition uses the current schema."""
    if not path.exists():
        return False
    try:
        stored = pd.read_parquet(path)
    except Exception:
        return False
    # Known source races can legitimately produce no five-lap windows.
    if stored.empty:
        return True
    columns = set(stored.columns)
    return REQUIRED_FEATURE_COLUMNS.issubset(columns)


def add_identifiers(
    frame: pd.DataFrame,
    *,
    year: int,
    round_number: int,
    event_name: str,
    session_name: str,
) -> pd.DataFrame:
    """Add the composite event identifier to every stored row."""
    result = pd.DataFrame(frame).copy()
    for column in ("Year", "RoundNumber", "EventName", "Session"):
        if column in result:
            result = result.drop(columns=column)
    result.insert(0, "Session", session_name)
    result.insert(0, "EventName", event_name)
    result.insert(0, "RoundNumber", round_number)
    result.insert(0, "Year", year)
    return result


def write_partition(
    destination: Path,
    frame: pd.DataFrame,
    *,
    year: int,
    round_number: int,
    event_name: str,
    session_name: str,
) -> None:
    """Write a compressed Parquet partition atomically enough for resuming."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.tmp")
    add_identifiers(
        frame,
        year=year,
        round_number=round_number,
        event_name=event_name,
        session_name=session_name,
    ).to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(destination)


def loaded_table(session: fastf1.core.Session, attribute: str) -> pd.DataFrame:
    """Return a loaded FastF1 table or an empty table when unavailable."""
    try:
        return pd.DataFrame(getattr(session, attribute)).copy()
    except DataNotLoadedError:
        logging.warning(
            "%s is unavailable for %s - %s; storing an empty partition",
            attribute,
            session.event["EventName"],
            session.name,
        )
        return pd.DataFrame()


def save_manifest(path: Path, stage: str, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "stage": stage,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_processed_table(table: str) -> pd.DataFrame:
    """Load and cache every existing partition for a small processed table."""
    if table not in TABLE_CACHE:
        files = sorted((PROCESSED_DIR / table).rglob("data.parquet"))
        frames = [pd.read_parquet(path) for path in files]
        TABLE_CACHE[table] = (
            pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        )
    return TABLE_CACHE[table].copy()


def add_circuit_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the circuit location associated with each year/round partition."""
    result = frame.copy()
    result["Circuit"] = [
        EVENT_CONTEXT.get((int(year), int(round_number)), {}).get(
            "Circuit", str(event_name)
        )
        for year, round_number, event_name in zip(
            result["Year"], result["RoundNumber"], result["EventName"]
        )
    ]
    return result


def historical_features(
    year: int,
    round_number: int,
    current_laps: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate driver/team form using only races before the current race."""
    participants = (
        current_laps[["Driver", "Team"]]
        .dropna()
        .drop_duplicates()
        .astype(str)
    )
    if participants.empty:
        return participants

    all_results = load_processed_table("results")
    all_qualifying = load_processed_table("qualifying")
    if all_results.empty:
        return participants

    prior_mask = (all_results["Year"] < year) | (
        (all_results["Year"] == year)
        & (all_results["RoundNumber"] < round_number)
    )
    prior = add_circuit_context(all_results.loc[prior_mask]).rename(
        columns={"Abbreviation": "Driver", "TeamName": "Team"}
    )
    prior["Driver"] = prior["Driver"].astype(str)
    prior["Team"] = prior["Team"].astype(str)
    prior["FinishPosition"] = pd.to_numeric(prior["Position"], errors="coerce")
    prior["Grid"] = pd.to_numeric(prior["GridPosition"], errors="coerce").where(
        lambda values: values > 0
    )
    prior["PositionGain"] = prior["Grid"] - prior["FinishPosition"]
    status = prior.get("Status", pd.Series("", index=prior.index)).fillna("").astype(str)
    prior["DNF"] = ~(
        status.eq("Finished") | status.str.match(r"^\+\d+ Lap", na=False)
    )
    prior = prior.sort_values(["Year", "RoundNumber"])

    if all_qualifying.empty:
        prior_qualifying = pd.DataFrame(
            columns=["Driver", "Team", "Circuit", "QualifyingPosition"]
        )
    else:
        qualifying_mask = (all_qualifying["Year"] < year) | (
            (all_qualifying["Year"] == year)
            & (all_qualifying["RoundNumber"] < round_number)
        )
        prior_qualifying = add_circuit_context(
            all_qualifying.loc[qualifying_mask]
        ).rename(columns={"Abbreviation": "Driver", "TeamName": "Team"})
        prior_qualifying["Driver"] = prior_qualifying["Driver"].astype(str)
        prior_qualifying["Team"] = prior_qualifying["Team"].astype(str)
        prior_qualifying["QualifyingPosition"] = pd.to_numeric(
            prior_qualifying["Position"], errors="coerce"
        )

    current_circuit = EVENT_CONTEXT.get((year, round_number), {}).get("Circuit")
    rows: list[dict[str, object]] = []
    for participant in participants.itertuples(index=False):
        driver = participant.Driver
        team = participant.Team
        driver_history = prior[prior["Driver"] == driver]
        driver_season = driver_history[driver_history["Year"] == year]
        driver_circuit = driver_history[
            driver_history["Circuit"] == current_circuit
        ]
        driver_circuit_qualifying = prior_qualifying[
            (prior_qualifying["Driver"] == driver)
            & (prior_qualifying["Circuit"] == current_circuit)
        ]

        team_history = prior[prior["Team"] == team]
        team_season = team_history[team_history["Year"] == year]
        team_circuit = team_history[team_history["Circuit"] == current_circuit]
        team_circuit_qualifying = prior_qualifying[
            (prior_qualifying["Team"] == team)
            & (prior_qualifying["Circuit"] == current_circuit)
        ]
        team_recent_events = (
            team_season.groupby("RoundNumber", sort=True)["FinishPosition"]
            .mean()
            .tail(3)
        )

        rows.append(
            {
                "Driver": driver,
                "Team": team,
                "DriverSeasonAvgFinish": driver_season["FinishPosition"].mean(),
                "DriverSeasonMedianFinish": driver_season["FinishPosition"].median(),
                "DriverRecent3AvgFinish": driver_season["FinishPosition"].tail(3).mean(),
                "DriverSeasonBestFinish": driver_season["FinishPosition"].min(),
                "DriverSeasonDNFRate": driver_season["DNF"].mean(),
                "DriverSeasonAvgPositionGain": driver_season["PositionGain"].mean(),
                "DriverCircuitAvgFinish": driver_circuit["FinishPosition"].mean(),
                "DriverCircuitLastFinish": driver_circuit["FinishPosition"].iloc[-1]
                if not driver_circuit.empty
                else np.nan,
                "DriverCircuitBestFinish": driver_circuit["FinishPosition"].min(),
                "DriverCircuitAvgPositionGain": driver_circuit["PositionGain"].mean(),
                "DriverCircuitAvgQualifyingPosition": driver_circuit_qualifying[
                    "QualifyingPosition"
                ].mean(),
                "DriverCircuitStarts": len(driver_circuit),
                "TeamSeasonAvgFinish": team_season["FinishPosition"].mean(),
                "TeamRecent3AvgFinish": team_recent_events.mean(),
                "TeamSeasonDNFRate": team_season["DNF"].mean(),
                "TeamCircuitAvgFinish": team_circuit["FinishPosition"].mean(),
                "TeamCircuitAvgQualifyingPosition": team_circuit_qualifying[
                    "QualifyingPosition"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def ingest_event(year: int, event: pd.Series) -> dict[str, object]:
    """Store laps, results, weather, and qualifying for one race weekend."""
    round_number = int(event["RoundNumber"])
    event_name = str(event["EventName"])
    destinations = {
        table: table_path(table, year, round_number) for table in BASE_TABLES
    }

    if all(path.exists() for path in destinations.values()):
        logging.info("Base tables already exist for %s R%d", year, round_number)
        return {"year": year, "round": round_number, "event": event_name, "status": "cached"}

    logging.info("Ingesting %s R%d: %s", year, round_number, event_name)
    race = event.get_session("R")
    race.load(laps=True, telemetry=False, weather=True, messages=False)

    for table, attribute in (
        ("laps", "laps"),
        ("results", "results"),
        ("weather", "weather_data"),
    ):
        write_partition(
            destinations[table],
            loaded_table(race, attribute),
            year=year,
            round_number=round_number,
            event_name=event_name,
            session_name="Race",
        )

    qualifying = event.get_session("Q")
    qualifying.load(laps=True, telemetry=False, weather=False, messages=False)
    write_partition(
        destinations["qualifying"],
        loaded_table(qualifying, "results"),
        year=year,
        round_number=round_number,
        event_name=event_name,
        session_name="Qualifying",
    )

    return {"year": year, "round": round_number, "event": event_name, "status": "processed"}


def aggregate_lap_telemetry(session: fastf1.core.Session) -> pd.DataFrame:
    """Reduce variable-length car telemetry to one fixed row per driver/lap."""
    rows: list[dict[str, object]] = []

    for _, lap in session.laps.iterrows():
        if pd.isna(lap.get("LapNumber")) or pd.isna(lap.get("DriverNumber")):
            continue
        try:
            telemetry = lap.get_car_data()
        except (DataNotLoadedError, KeyError, ValueError):
            continue
        if telemetry.empty:
            continue

        drs = pd.to_numeric(telemetry.get("DRS"), errors="coerce")

        rows.append(
            {
                "DriverNumber": str(lap["DriverNumber"]),
                "LapNumber": float(lap["LapNumber"]),
                "DRSActivePct": drs.isin((10, 12, 14)).mean() * 100,
            }
        )

    return pd.DataFrame(rows)


def timedelta_seconds(values: pd.Series) -> pd.Series:
    return pd.to_timedelta(values, errors="coerce").dt.total_seconds()


def attach_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest known weather observation at each lap end."""
    weather_columns = (
        "AirTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "TrackTemp",
        "WindDirection",
        "WindSpeed",
    )
    result = laps.copy()
    for column in weather_columns:
        if column not in result:
            result[column] = np.nan
    if weather.empty or "Time" not in weather or "Time" not in result:
        return result

    observations = weather.copy()
    observations["WeatherTimeSeconds"] = timedelta_seconds(observations["Time"])
    available = [column for column in weather_columns if column in observations]
    observations = observations[["WeatherTimeSeconds", *available]].dropna(
        subset=["WeatherTimeSeconds"]
    )
    if observations.empty:
        return result

    result["SessionTimeSeconds"] = timedelta_seconds(result["Time"])
    result["_row_order"] = np.arange(len(result))
    valid = result[result["SessionTimeSeconds"].notna()].drop(columns=available)
    invalid = result[result["SessionTimeSeconds"].isna()]
    valid = pd.merge_asof(
        valid.sort_values("SessionTimeSeconds"),
        observations.sort_values("WeatherTimeSeconds"),
        left_on="SessionTimeSeconds",
        right_on="WeatherTimeSeconds",
        direction="backward",
    )
    result = pd.concat([valid, invalid], ignore_index=True).sort_values("_row_order")
    return result.drop(columns=["_row_order", "WeatherTimeSeconds"], errors="ignore")


def qualifying_features(qualifying: pd.DataFrame) -> pd.DataFrame:
    """Create pre-race qualifying position and delta-to-pole features."""
    if qualifying.empty or "DriverNumber" not in qualifying:
        return pd.DataFrame(
            columns=["DriverNumber", "QualifyingPosition", "QualifyingDeltaToPoleSeconds"]
        )

    output = pd.DataFrame(
        {"DriverNumber": qualifying["DriverNumber"].astype(str)}
    )
    output["QualifyingPosition"] = pd.to_numeric(
        qualifying.get("Position"), errors="coerce"
    )
    phase_times = pd.DataFrame(index=qualifying.index)
    for phase in ("Q1", "Q2", "Q3"):
        if phase in qualifying:
            phase_times[phase] = timedelta_seconds(qualifying[phase])
    if phase_times.empty:
        output["QualifyingDeltaToPoleSeconds"] = np.nan
    else:
        best = phase_times.min(axis=1, skipna=True)
        output["QualifyingDeltaToPoleSeconds"] = best - best.min()
    return output


def build_five_lap_features(
    laps: pd.DataFrame,
    telemetry: pd.DataFrame,
    weather: pd.DataFrame,
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
    history: pd.DataFrame,
    event: pd.Series,
    *,
    total_race_laps: int | None = None,
    require_future_targets: bool = True,
) -> pd.DataFrame:
    """Create one row per driver/window with several future race horizons.

    Offline ingestion keeps ``require_future_targets=True`` so training rows
    always have a known label. A live caller sets it to false to retain the
    newest completed five-lap window. Live callers must also pass the scheduled
    race distance because the largest observed lap is not the total distance.
    """
    if laps.empty:
        return pd.DataFrame()

    data = laps.copy()
    data["DriverNumber"] = data["DriverNumber"].astype(str)
    data["LapNumber"] = pd.to_numeric(data["LapNumber"], errors="coerce")
    data["Position"] = pd.to_numeric(data["Position"], errors="coerce")

    for source, destination in (
        ("LapTime", "LapTimeSeconds"),
        ("Sector1Time", "Sector1Seconds"),
        ("Sector2Time", "Sector2Seconds"),
        ("Sector3Time", "Sector3Seconds"),
    ):
        data[destination] = timedelta_seconds(data[source]) if source in data else np.nan

    if not telemetry.empty:
        telemetry = telemetry.copy()
        telemetry["DriverNumber"] = telemetry["DriverNumber"].astype(str)
        telemetry["LapNumber"] = pd.to_numeric(telemetry["LapNumber"], errors="coerce")
        data = data.merge(telemetry, on=["DriverNumber", "LapNumber"], how="left")

    data = attach_weather(data, weather)

    race_results = results.copy()
    if not race_results.empty and "DriverNumber" in race_results:
        race_results["DriverNumber"] = race_results["DriverNumber"].astype(str)
        if "GridPosition" in race_results:
            grid = race_results[["DriverNumber", "GridPosition"]].copy()
            grid["GridPosition"] = pd.to_numeric(grid["GridPosition"], errors="coerce")
            data = data.merge(grid, on="DriverNumber", how="left")
    if "GridPosition" not in data:
        data["GridPosition"] = np.nan

    data = data.merge(qualifying_features(qualifying), on="DriverNumber", how="left")
    if not history.empty:
        data = data.merge(history, on=["Driver", "Team"], how="left")
    data["Circuit"] = str(event.get("Location", event["EventName"]))
    data["Country"] = str(event.get("Country", "Unknown"))
    data["EventFormat"] = str(event.get("EventFormat", "conventional"))

    total_laps = (
        int(total_race_laps)
        if total_race_laps is not None
        else pd.to_numeric(data["LapNumber"], errors="coerce").max()
    )
    if pd.isna(total_laps) or float(total_laps) <= 0:
        raise ValueError("total_race_laps must be a positive number")
    data["TotalRaceLaps"] = total_laps
    data["RaceProgress"] = data["LapNumber"] / total_laps
    data["PitThisLap"] = (
        data.get("PitInTime", pd.Series(index=data.index, dtype="object")).notna()
        | data.get("PitOutTime", pd.Series(index=data.index, dtype="object")).notna()
    ).astype(int)
    track_status = data.get(
        "TrackStatus", pd.Series("", index=data.index)
    ).astype(str)
    # FastF1 status 4 is safety car; 6/7 are VSC deployed/ending. Keeping
    # these separate prevents the model from treating both interventions alike.
    data["SafetyCarThisLap"] = track_status.str.contains("4", regex=False).astype(int)
    data["VirtualSafetyCarThisLap"] = track_status.str.contains(
        r"[67]", regex=True
    ).astype(int)

    # Only current and past pit information is used. Before a driver's first
    # stop, LapsSincePitStop remains missing and HasPitted distinguishes it.
    last_pit_lap = data["LapNumber"].where(data["PitThisLap"].eq(1)).groupby(
        data["DriverNumber"]
    ).ffill()
    data["HasPitted"] = last_pit_lap.notna().astype(int)
    data["LapsSincePitStop"] = data["LapNumber"] - last_pit_lap

    # Pit-window phase is based on the fraction of the starting field that has
    # made a first stop by the current lap. This is causal and circuit agnostic.
    first_pit_lap = (
        data.loc[data["PitThisLap"].eq(1)]
        .groupby("DriverNumber")["LapNumber"]
        .min()
    )
    field_size = max(int(data["DriverNumber"].nunique()), 1)
    unique_laps = np.sort(data["LapNumber"].dropna().unique())
    pitted_share_by_lap = {
        lap: float(first_pit_lap.le(lap).sum() / field_size)
        for lap in unique_laps
    }
    data["FieldPittedShare"] = data["LapNumber"].map(pitted_share_by_lap)
    data["PitWindowPhase"] = pd.cut(
        data["FieldPittedShare"],
        bins=[-np.inf, 0.15, 0.35, 0.75, np.inf],
        labels=["PreWindow", "Opening", "Active", "Closing"],
        include_lowest=True,
    ).astype("string")

    data["FieldMedianLapSeconds"] = data.groupby("LapNumber")[
        "LapTimeSeconds"
    ].transform("median")
    data["PaceToFieldSeconds"] = (
        data["LapTimeSeconds"] - data["FieldMedianLapSeconds"]
    )
    if "Team" in data:
        data["TeamMedianLapSeconds"] = data.groupby(["LapNumber", "Team"])[
            "LapTimeSeconds"
        ].transform("median")
        data["PaceToTeammateSeconds"] = (
            data["LapTimeSeconds"] - data["TeamMedianLapSeconds"]
        )
    else:
        data["PaceToTeammateSeconds"] = np.nan

    # Absolute sector seconds are difficult to compare between circuits. These
    # deltas express each sector relative to the field and teammate on that lap.
    for sector in ("Sector1", "Sector2", "Sector3"):
        seconds = f"{sector}Seconds"
        field_median = data.groupby("LapNumber")[seconds].transform("median")
        data[f"{sector}ToFieldSeconds"] = data[seconds] - field_median
        if "Team" in data:
            team_median = data.groupby(["LapNumber", "Team"])[seconds].transform(
                "median"
            )
            data[f"{sector}ToTeammateSeconds"] = data[seconds] - team_median
        else:
            data[f"{sector}ToTeammateSeconds"] = np.nan

    if "SessionTimeSeconds" not in data:
        data["SessionTimeSeconds"] = timedelta_seconds(data["Time"])
    data["GapToLeaderSeconds"] = data["SessionTimeSeconds"] - data.groupby(
        "LapNumber"
    )["SessionTimeSeconds"].transform("min")
    ordered = data.sort_values(["LapNumber", "Position", "SessionTimeSeconds"])
    lap_groups = ordered.groupby("LapNumber", sort=False)["SessionTimeSeconds"]
    ordered["GapToCarAheadSeconds"] = (
        ordered["SessionTimeSeconds"] - lap_groups.shift(1)
    ).clip(lower=0)
    ordered["GapToCarBehindSeconds"] = (
        lap_groups.shift(-1) - ordered["SessionTimeSeconds"]
    ).clip(lower=0)
    tyre_life = pd.to_numeric(ordered.get("TyreLife"), errors="coerce")
    car_ahead_tyre_life = tyre_life.groupby(ordered["LapNumber"], sort=False).shift(1)
    # Positive means the current driver has the fresher tyre by this many laps.
    ordered["EstimatedTyreLifeAdvantageLaps"] = car_ahead_tyre_life - tyre_life
    data = ordered.sort_values(["DriverNumber", "LapNumber"]).reset_index(drop=True)

    lag_columns = [
        "LapTimeSeconds",
        "Sector1ToFieldSeconds",
        "Sector2ToFieldSeconds",
        "Sector3ToFieldSeconds",
        "Sector1ToTeammateSeconds",
        "Sector2ToTeammateSeconds",
        "Sector3ToTeammateSeconds",
        "Position",
        "TyreLife",
        "DRSActivePct",
        "GapToLeaderSeconds",
        "GapToCarAheadSeconds",
        "GapToCarBehindSeconds",
        "PaceToFieldSeconds",
        "PaceToTeammateSeconds",
        "LapsSincePitStop",
        "EstimatedTyreLifeAdvantageLaps",
        "FieldPittedShare",
        "VirtualSafetyCarThisLap",
        "AirTemp",
        "TrackTemp",
        "Rainfall",
    ]
    for column in lag_columns:
        if column not in data:
            data[column] = np.nan

    generated: dict[str, pd.Series] = {}
    for column in lag_columns:
        numeric = pd.to_numeric(data[column], errors="coerce")
        for lag in range(WINDOW_LAPS):
            generated[f"{column}_lag{lag}"] = numeric.groupby(
                data["DriverNumber"]
            ).shift(lag)

    rolling_columns = [
        "LapTimeSeconds",
        "Sector1ToFieldSeconds",
        "Sector2ToFieldSeconds",
        "Sector3ToFieldSeconds",
        "Sector1ToTeammateSeconds",
        "Sector2ToTeammateSeconds",
        "Sector3ToTeammateSeconds",
        "Position",
        "TyreLife",
        "GapToLeaderSeconds",
        "GapToCarAheadSeconds",
        "PaceToFieldSeconds",
        "PaceToTeammateSeconds",
        "LapsSincePitStop",
        "EstimatedTyreLifeAdvantageLaps",
        "FieldPittedShare",
        "TrackTemp",
    ]
    for column in rolling_columns:
        numeric = pd.to_numeric(data[column], errors="coerce")
        group_numeric = numeric.groupby(data["DriverNumber"])
        generated[f"{column}_mean5"] = group_numeric.transform(
            lambda values: values.rolling(WINDOW_LAPS, min_periods=WINDOW_LAPS).mean()
        )
        generated[f"{column}_std5"] = group_numeric.transform(
            lambda values: values.rolling(WINDOW_LAPS, min_periods=WINDOW_LAPS).std()
        )
        generated[f"{column}_trend5"] = numeric - group_numeric.shift(WINDOW_LAPS - 1)

    data = pd.concat([data, pd.DataFrame(generated, index=data.index)], axis=1)

    grouped = data.groupby("DriverNumber", sort=False)
    compound_history = pd.concat(
        [grouped["Compound"].shift(lag) for lag in range(WINDOW_LAPS)], axis=1
    )
    target_values: dict[str, pd.Series] = {}
    for horizon in PREDICTION_HORIZONS:
        target_lap = grouped["LapNumber"].shift(-horizon)
        contiguous = target_lap.sub(data["LapNumber"]).eq(horizon)
        target_position = grouped["Position"].shift(-horizon).where(contiguous)
        target_values[f"TargetLapNumber_h{horizon}"] = target_lap.where(contiguous)
        target_values[f"TargetPosition_h{horizon}"] = target_position
        target_values[f"TargetPositionChange_h{horizon}"] = (
            target_position - data["Position"]
        )
        target_values[f"TargetGapToLeaderSeconds_h{horizon}"] = grouped[
            "GapToLeaderSeconds"
        ].shift(-horizon).where(contiguous)

    additions = pd.DataFrame(
        {
            "PitLapsLast5": grouped["PitThisLap"].transform(
                lambda values: values.rolling(
                    WINDOW_LAPS, min_periods=WINDOW_LAPS
                ).sum()
            ),
            "SafetyCarLapsLast5": grouped["SafetyCarThisLap"].transform(
                lambda values: values.rolling(
                    WINDOW_LAPS, min_periods=WINDOW_LAPS
                ).sum()
            ),
            "VirtualSafetyCarLapsLast5": grouped[
                "VirtualSafetyCarThisLap"
            ].transform(
                lambda values: values.rolling(
                    WINDOW_LAPS, min_periods=WINDOW_LAPS
                ).sum()
            ),
            "CompoundChangedLast5": compound_history.nunique(
                axis=1, dropna=True
            ).gt(1),
            "_driver_lap_index": grouped.cumcount(),
            **target_values,
        },
        index=data.index,
    )
    # Preserve the original five-lap names for existing consumers.
    additions["TargetLapNumber"] = additions[
        f"TargetLapNumber_h{PRIMARY_PREDICTION_HORIZON}"
    ]
    additions["TargetPosition"] = additions[
        f"TargetPosition_h{PRIMARY_PREDICTION_HORIZON}"
    ]
    additions["TargetPositionChange"] = additions[
        f"TargetPositionChange_h{PRIMARY_PREDICTION_HORIZON}"
    ]
    additions["TargetGapToLeaderSeconds"] = additions[
        f"TargetGapToLeaderSeconds_h{PRIMARY_PREDICTION_HORIZON}"
    ]
    data = pd.concat([data, additions], axis=1)

    context_columns = [
        "DriverNumber",
        "Driver",
        "Team",
        "Circuit",
        "Country",
        "EventFormat",
        "LapNumber",
        "TotalRaceLaps",
        "RaceProgress",
        "Position",
        "GridPosition",
        "QualifyingPosition",
        "QualifyingDeltaToPoleSeconds",
        "DriverSeasonAvgFinish",
        "DriverSeasonMedianFinish",
        "DriverRecent3AvgFinish",
        "DriverSeasonBestFinish",
        "DriverSeasonDNFRate",
        "DriverSeasonAvgPositionGain",
        "DriverCircuitAvgFinish",
        "DriverCircuitLastFinish",
        "DriverCircuitBestFinish",
        "DriverCircuitAvgPositionGain",
        "DriverCircuitAvgQualifyingPosition",
        "DriverCircuitStarts",
        "TeamSeasonAvgFinish",
        "TeamRecent3AvgFinish",
        "TeamSeasonDNFRate",
        "TeamCircuitAvgFinish",
        "TeamCircuitAvgQualifyingPosition",
        "Stint",
        "Compound",
        "FreshTyre",
        "TyreLife",
        "TrackStatus",
        "PitThisLap",
        "SafetyCarThisLap",
        "VirtualSafetyCarThisLap",
        "HasPitted",
        "LapsSincePitStop",
        "FieldPittedShare",
        "PitWindowPhase",
        "EstimatedTyreLifeAdvantageLaps",
        "PitLapsLast5",
        "SafetyCarLapsLast5",
        "VirtualSafetyCarLapsLast5",
        "CompoundChangedLast5",
    ]
    engineered_columns = [
        column
        for column in data.columns
        if "_lag" in column or column.endswith(("_mean5", "_std5", "_trend5"))
    ]
    target_columns = [
        "TargetLapNumber",
        "TargetPosition",
        "TargetPositionChange",
        "TargetGapToLeaderSeconds",
        *[
            f"Target{target}_h{horizon}"
            for horizon in PREDICTION_HORIZONS
            for target in (
                "LapNumber",
                "Position",
                "PositionChange",
                "GapToLeaderSeconds",
            )
        ],
    ]
    # Training keeps only labelled rows. Live inference retains the newest
    # completed window even though no future target exists yet.
    valid = data["_driver_lap_index"].ge(WINDOW_LAPS - 1)
    if require_future_targets:
        valid &= data["TargetPosition_h1"].notna()
    columns = [
        column
        for column in [*context_columns, *engineered_columns, *target_columns]
        if column in data
    ]
    return data.loc[valid, columns].reset_index(drop=True)


def build_event_features(year: int, event: pd.Series) -> dict[str, object]:
    """Build telemetry aggregates and five-lap windows for one race."""
    round_number = int(event["RoundNumber"])
    event_name = str(event["EventName"])
    telemetry_destination = table_path("telemetry_laps", year, round_number)
    features_destination = feature_path(year, round_number)
    if telemetry_destination.exists() and feature_partition_is_current(
        features_destination
    ):
        logging.info("Features already exist for %s R%d", year, round_number)
        return {"year": year, "round": round_number, "event": event_name, "status": "cached"}

    base_paths = {table: table_path(table, year, round_number) for table in BASE_TABLES}
    if not all(path.exists() for path in base_paths.values()):
        return {"year": year, "round": round_number, "event": event_name, "status": "base_missing"}

    logging.info("Building features for %s R%d: %s", year, round_number, event_name)
    if telemetry_destination.exists():
        telemetry = pd.read_parquet(telemetry_destination)
    else:
        # Only races without an existing telemetry-lap partition require a
        # FastF1 session load. Normal schema upgrades rebuild entirely offline.
        race = event.get_session("R")
        race.load(laps=True, telemetry=True, weather=True, messages=False)
        telemetry = aggregate_lap_telemetry(race)
        write_partition(
            telemetry_destination,
            telemetry,
            year=year,
            round_number=round_number,
            event_name=event_name,
            session_name="Race",
        )

    features = build_five_lap_features(
        pd.read_parquet(base_paths["laps"]),
        telemetry,
        pd.read_parquet(base_paths["weather"]),
        pd.read_parquet(base_paths["results"]),
        pd.read_parquet(base_paths["qualifying"]),
        historical_features(
            year,
            round_number,
            pd.read_parquet(base_paths["laps"]),
        ),
        event,
    )
    write_partition(
        features_destination,
        features,
        year=year,
        round_number=round_number,
        event_name=event_name,
        session_name="Race",
    )
    return {
        "year": year,
        "round": round_number,
        "event": event_name,
        "status": "processed",
        "feature_rows": len(features),
    }


def run_stage(
    stage: str,
    processor: Callable[[int, pd.Series], dict[str, object]],
) -> bool:
    """Run a resumable stage; return False when the hourly limit is reached."""
    records: list[dict[str, object]] = []
    manifest_path = (
        PROCESSED_DIR / "ingestion_manifest.json"
        if stage == "ingestion"
        else FEATURE_DIR.parent / "feature_manifest.json"
    )
    now = pd.Timestamp.now(tz="UTC")

    for year in range(START_YEAR, END_YEAR + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except RateLimitExceededError as error:
            logging.error("FastF1 hourly limit reached before %s: %s", year, error)
            records.append({"year": year, "status": "rate_limited"})
            save_manifest(manifest_path, stage, records)
            return False

        for _, scheduled_event in schedule.iterrows():
            scheduled_round = int(scheduled_event["RoundNumber"])
            if scheduled_round > 0:
                EVENT_CONTEXT[(year, scheduled_round)] = {
                    "Circuit": str(
                        scheduled_event.get(
                            "Location", scheduled_event.get("EventName", "Unknown")
                        )
                    ),
                    "Country": str(scheduled_event.get("Country", "Unknown")),
                }

        for _, event in schedule.iterrows():
            if int(event["RoundNumber"]) == 0 or not completed_race(event, now):
                continue
            try:
                record = processor(year, event)
            except RateLimitExceededError as error:
                logging.error("FastF1 hourly limit reached: %s", error)
                records.append(
                    {
                        "year": year,
                        "round": int(event["RoundNumber"]),
                        "event": str(event["EventName"]),
                        "status": "rate_limited",
                    }
                )
                save_manifest(manifest_path, stage, records)
                return False
            except Exception as error:
                logging.exception("Failed %s for %s %s", stage, year, event["EventName"])
                record = {
                    "year": year,
                    "round": int(event["RoundNumber"]),
                    "event": str(event["EventName"]),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            records.append(record)
            save_manifest(manifest_path, stage, records)
    return True


def main(stage: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    if stage in ("all", "ingest"):
        ingestion_complete = run_stage("ingestion", ingest_event)
        if not ingestion_complete:
            logging.warning(
                "Ingestion paused at FastF1's hourly limit. Run this command again "
                "after the limit resets; completed partitions will be skipped."
            )
            return
    if stage in ("all", "features"):
        feature_complete = run_stage("features", build_event_features)
        if not feature_complete:
            logging.warning(
                "Feature generation paused at FastF1's hourly limit. Run this "
                "command again after the limit resets to resume."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "ingest", "features"),
        default="all",
        help="Run base ingestion, feature generation, or both (default: all).",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(arguments.stage)
