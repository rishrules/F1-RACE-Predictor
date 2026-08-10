r"""Record, parse, store and predict from the FastF1 live timing stream.

The pipeline deliberately keeps the official FastF1 recording untouched:

    SignalRClient -> raw append-only text recording -> SQLite time-series store
                  -> FastF1 snapshot parser -> five-lap features -> model

FastF1 documents its live client as a recorder, not a real-time parser. This
module therefore parses complete snapshots on a best-effort basis after new
TimingData messages arrive. Predictions are only emitted from completed laps.

Examples
--------
Record and predict during a race::

    .\.venv\Scripts\python.exe live_predictor.py record \
        --year 2026 --round 12 --total-laps 70 --horizons 5 10

Process a recording produced by FastF1 in another terminal::

    .\.venv\Scripts\python.exe live_predictor.py process \
        --year 2026 --round 12 --total-laps 70 \
        --raw data/live/2026_r12_raw.txt --horizons 5 10

Show the newest persisted leaderboard::

    .\.venv\Scripts\python.exe live_predictor.py show \
        --db data/live/live_timing.sqlite
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import json
import logging
import sqlite3
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import fastf1
import joblib
import numpy as np
import pandas as pd
from fastf1.livetiming.client import SignalRClient
from fastf1.livetiming.data import LiveTimingData

from f1_data import (
    EVENT_CONTEXT,
    aggregate_lap_telemetry,
    build_five_lap_features,
    historical_features,
)


ROOT = Path(__file__).resolve().parent
LIVE_DIR = ROOT / "data" / "live"
DEFAULT_DATABASE = LIVE_DIR / "live_timing.sqlite"
DEFAULT_MODEL_PATHS = {
    5: ROOT / "reports" / "model_improvements" / "final_two_stage_model_h5.joblib",
    10: ROOT / "reports" / "model_improvements" / "final_two_stage_model.joblib",
}
TIMING_TOPICS = {
    "TimingData",
    "TimingAppData",
    "CarData.z",
    "WeatherData",
    "TrackStatus",
    "DriverList",
    "LapCount",
    "SessionStatus",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: object) -> object:
    """Convert pandas/numpy values to JSON-safe scalar values."""
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


class TimeSeriesStore:
    """Small embedded time-series store backed by indexed SQLite tables.

    SQLite keeps the live runner self-contained: no database server or extra
    driver is needed on race day. WAL mode permits the recorder, predictor and
    a dashboard/query process to access the database concurrently.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    @contextmanager
    def connection(self):
        """Commit or roll back, then always release the Windows file handle."""
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_key TEXT NOT NULL,
                    received_utc TEXT NOT NULL,
                    stream_utc TEXT,
                    topic TEXT NOT NULL,
                    payload_json BLOB NOT NULL,
                    raw_line BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_raw_messages_time
                    ON raw_messages(session_key, received_utc, topic);

                CREATE TABLE IF NOT EXISTS stream_offsets (
                    file_path TEXT PRIMARY KEY,
                    byte_offset INTEGER NOT NULL,
                    updated_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lap_features (
                    session_key TEXT NOT NULL,
                    observed_utc TEXT NOT NULL,
                    reference_lap INTEGER NOT NULL,
                    driver_number TEXT NOT NULL,
                    driver TEXT,
                    completed_lap REAL,
                    features_json BLOB NOT NULL,
                    PRIMARY KEY (session_key, reference_lap, driver_number)
                );
                CREATE INDEX IF NOT EXISTS ix_lap_features_time
                    ON lap_features(session_key, observed_utc);

                CREATE TABLE IF NOT EXISTS predictions (
                    session_key TEXT NOT NULL,
                    predicted_utc TEXT NOT NULL,
                    reference_lap INTEGER NOT NULL,
                    horizon INTEGER NOT NULL,
                    predicted_rank INTEGER NOT NULL,
                    driver_number TEXT NOT NULL,
                    driver TEXT,
                    current_position REAL,
                    change_probability REAL,
                    conditional_change REAL,
                    predicted_change REAL,
                    predicted_position_score REAL,
                    PRIMARY KEY (
                        session_key, reference_lap, horizon, driver_number
                    )
                );
                CREATE INDEX IF NOT EXISTS ix_predictions_latest
                    ON predictions(session_key, reference_lap, horizon,
                                   predicted_rank);
                """
            )

    def offset_for(self, raw_path: Path) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT byte_offset FROM stream_offsets WHERE file_path = ?",
                (str(raw_path.resolve()),),
            ).fetchone()
        return int(row[0]) if row else 0

    def append_messages(
        self,
        session_key: str,
        raw_path: Path,
        new_offset: int,
        messages: list[tuple[str, str | None, bytes, bytes]],
    ) -> None:
        now = utc_now()
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO raw_messages(
                    session_key, received_utc, stream_utc, topic,
                    payload_json, raw_line
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (session_key, now, stream_utc, topic, payload, raw_line)
                    for topic, stream_utc, payload, raw_line in messages
                ],
            )
            connection.execute(
                """
                INSERT INTO stream_offsets(file_path, byte_offset, updated_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    byte_offset = excluded.byte_offset,
                    updated_utc = excluded.updated_utc
                """,
                (str(raw_path.resolve()), new_offset, now),
            )

    def save_feature_rows(
        self,
        session_key: str,
        reference_lap: int,
        features: pd.DataFrame,
    ) -> None:
        observed = utc_now()
        rows = []
        for record in features.to_dict(orient="records"):
            rows.append(
                (
                    session_key,
                    observed,
                    reference_lap,
                    str(record.get("DriverNumber", "")),
                    str(record.get("Driver", "")),
                    float(record.get("LapNumber", np.nan)),
                    # pandas serializes NaN/NaT as JSON null, while the stdlib
                    # encoder would otherwise emit non-standard NaN tokens.
                    pd.Series(record).to_json(date_format="iso"),
                )
            )
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO lap_features(
                    session_key, observed_utc, reference_lap, driver_number,
                    driver, completed_lap, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def save_predictions(
        self,
        session_key: str,
        reference_lap: int,
        horizon: int,
        frame: pd.DataFrame,
    ) -> None:
        predicted = utc_now()
        rows = [
            (
                session_key,
                predicted,
                reference_lap,
                horizon,
                int(row.PredictedPosition),
                str(row.DriverNumber),
                str(row.Driver),
                float(row.CurrentPosition),
                float(row.ChangeProbability),
                float(row.ConditionalChange),
                float(row.PredictedChange),
                float(row.PredictedPositionScore),
            )
            for row in frame.itertuples(index=False)
        ]
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO predictions(
                    session_key, predicted_utc, reference_lap, horizon,
                    predicted_rank, driver_number, driver, current_position,
                    change_probability, conditional_change, predicted_change,
                    predicted_position_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def latest_predictions(self, session_key: str | None = None) -> pd.DataFrame:
        where = "WHERE session_key = ?" if session_key else ""
        parameters: tuple[object, ...] = (session_key,) if session_key else ()
        query = f"""
            WITH newest AS (
                SELECT session_key, MAX(reference_lap) AS reference_lap
                FROM predictions {where}
                GROUP BY session_key
            )
            SELECT p.session_key AS Session, p.reference_lap AS ReferenceLap,
                   p.horizon AS Horizon, p.predicted_rank AS PredictedPosition,
                   p.driver AS Driver, p.driver_number AS DriverNumber,
                   p.current_position AS CurrentPosition,
                   p.change_probability AS ChangeProbability,
                   p.predicted_change AS PredictedChange
            FROM predictions p
            JOIN newest n ON n.session_key = p.session_key
                         AND n.reference_lap = p.reference_lap
            ORDER BY p.horizon, p.predicted_rank
        """
        with self.connection() as connection:
            return pd.read_sql_query(query, connection, params=parameters)


def parse_recording_line(raw_line: bytes) -> tuple[str, str | None, bytes] | None:
    """Parse one complete SignalRClient recording line without modifying it."""
    try:
        decoded = raw_line.decode("utf-8").strip()
        if not decoded:
            return None
        item = ast.literal_eval(decoded)
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        topic, payload, stream_utc = item
        payload_bytes = json.dumps(
            payload, default=json_default, separators=(",", ":")
        ).encode("utf-8")
        return str(topic), str(stream_utc) if stream_utc else None, payload_bytes
    except (UnicodeDecodeError, SyntaxError, ValueError, TypeError):
        return None


class RawStreamTailer:
    """Read only complete new lines and mirror them into the time-series DB."""

    def __init__(
        self,
        raw_path: Path,
        store: TimeSeriesStore,
        session_key: str,
    ) -> None:
        self.raw_path = raw_path.resolve()
        self.store = store
        self.session_key = session_key

    def ingest_available(self) -> set[str]:
        if not self.raw_path.exists():
            return set()
        offset = self.store.offset_for(self.raw_path)
        file_size = self.raw_path.stat().st_size
        if file_size < offset:  # recorder was restarted without append mode
            offset = 0

        messages: list[tuple[str, str | None, bytes, bytes]] = []
        topics: set[str] = set()
        with self.raw_path.open("rb") as source:
            source.seek(offset)
            while True:
                line_start = source.tell()
                raw_line = source.readline()
                if not raw_line:
                    break
                if not raw_line.endswith(b"\n"):
                    # SignalRClient may still be writing this line.
                    source.seek(line_start)
                    break
                parsed = parse_recording_line(raw_line)
                if parsed is not None:
                    topic, stream_utc, payload = parsed
                    messages.append((topic, stream_utc, payload, raw_line))
                    topics.add(topic)
            new_offset = source.tell()

        if messages or new_offset != offset:
            self.store.append_messages(
                self.session_key,
                self.raw_path,
                new_offset,
                messages,
            )
        return topics


def atomic_recording_snapshot(raw_path: Path) -> Path:
    """Copy all complete recording lines to a stable temporary file."""
    temporary = tempfile.NamedTemporaryFile(
        mode="wb", suffix="_fastf1_live.txt", delete=False
    )
    snapshot = Path(temporary.name)
    with temporary, raw_path.open("rb") as source:
        while True:
            line = source.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            temporary.write(line)
    return snapshot


@dataclass(frozen=True)
class LoadedModel:
    horizon: int
    model: object
    feature_columns: list[str]
    categorical_columns: list[str]
    category_vocabularies: dict[str, list[object]]


def load_models(
    horizons: Iterable[int],
    model_paths: dict[int, Path] | None = None,
) -> dict[int, LoadedModel]:
    paths = model_paths or DEFAULT_MODEL_PATHS
    loaded: dict[int, LoadedModel] = {}
    for horizon in horizons:
        path = paths[horizon]
        if not path.exists():
            raise FileNotFoundError(
                f"No {horizon}-lap model exists at {path}. "
                "Train the required artifact before starting the live client."
            )
        bundle = joblib.load(path)
        stored_horizon = int(bundle["target_horizon"])
        if stored_horizon != horizon:
            raise ValueError(
                f"{path} contains a {stored_horizon}-lap model, not {horizon}"
            )
        loaded[horizon] = LoadedModel(
            horizon=horizon,
            model=bundle["model"],
            feature_columns=list(bundle["feature_columns"]),
            categorical_columns=list(bundle["categorical_columns"]),
            category_vocabularies=dict(bundle["category_vocabularies"]),
        )
    return loaded


def align_live_features(
    features: pd.DataFrame,
    loaded: LoadedModel,
) -> pd.DataFrame:
    """Apply the exact training columns and categorical vocabularies."""
    X = features.reindex(columns=loaded.feature_columns).copy()
    for column in loaded.categorical_columns:
        if column in X:
            X[column] = pd.Categorical(
                X[column],
                categories=loaded.category_vocabularies.get(column, []),
            )
    return X


def predict_leaderboard(
    feature_rows: pd.DataFrame,
    loaded: LoadedModel,
) -> pd.DataFrame:
    """Predict independent changes and rank them into one coherent order."""
    X = align_live_features(feature_rows, loaded)
    probability, conditional = loaded.model.predict_components(X)
    predicted_change = probability * conditional
    current_position = pd.to_numeric(
        feature_rows["Position"], errors="coerce"
    ).to_numpy(dtype=float)
    score = current_position + predicted_change

    result = pd.DataFrame(
        {
            "DriverNumber": feature_rows["DriverNumber"].astype(str).to_numpy(),
            "Driver": feature_rows["Driver"].astype(str).to_numpy(),
            "CurrentPosition": current_position,
            "ChangeProbability": probability,
            "ConditionalChange": conditional,
            "PredictedChange": predicted_change,
            "PredictedPositionScore": score,
        }
    )
    # Ranking avoids duplicate/non-integer positions from independent driver
    # regressions. The continuous score remains available for diagnostics.
    result = result.sort_values(
        ["PredictedPositionScore", "CurrentPosition"], na_position="last"
    ).reset_index(drop=True)
    result["PredictedPosition"] = np.arange(1, len(result) + 1)
    return result


class LivePredictionEngine:
    """Parse a recording snapshot, compile features and persist predictions."""

    def __init__(
        self,
        *,
        year: int,
        round_number: int,
        total_laps: int,
        horizons: Iterable[int],
        store: TimeSeriesStore,
        output_path: Path,
    ) -> None:
        self.year = year
        self.round_number = round_number
        self.total_laps = total_laps
        self.session_key = f"{year}-R{round_number:02d}"
        self.store = store
        self.output_path = output_path.resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.models = load_models(horizons)
        self.last_reference_lap = -1

        # Load static event and qualifying context once, before disabling the
        # FastF1 request cache for changing live snapshots.
        self.race_template = fastf1.get_session(year, round_number, "R")
        self.event = self.race_template.event
        circuit = str(self.event.get("Location", self.event["EventName"]))
        EVENT_CONTEXT[(year, round_number)] = {
            "Circuit": circuit,
            "Country": str(self.event.get("Country", "Unknown")),
            "EventName": str(self.event["EventName"]),
        }
        self.qualifying = self._load_qualifying()

    def _load_qualifying(self) -> pd.DataFrame:
        try:
            qualifying = fastf1.get_session(
                self.year, self.round_number, "Q"
            )
            qualifying.load(
                laps=False, telemetry=False, weather=False, messages=False
            )
            return pd.DataFrame(qualifying.results).copy()
        except Exception as error:  # live operation should continue with NaNs
            logging.warning("Qualifying context unavailable: %s", error)
            return pd.DataFrame()

    def _load_live_session(self, snapshot: Path) -> fastf1.core.Session:
        livedata = LiveTimingData(str(snapshot))
        session = fastf1.get_session(self.year, self.round_number, "R")
        # A growing recording must never reuse a cached parse from an older
        # snapshot. Static qualifying context was already loaded above.
        fastf1.Cache.set_disabled()
        session.load(
            laps=True,
            telemetry=True,
            weather=True,
            messages=False,
            livedata=livedata,
        )
        return session

    def update_from_recording(self, raw_path: Path) -> bool:
        snapshot = atomic_recording_snapshot(raw_path)
        try:
            session = self._load_live_session(snapshot)
            laps = pd.DataFrame(session.laps).copy()
            if laps.empty:
                return False
            completed = pd.to_numeric(laps["LapNumber"], errors="coerce")
            reference_lap = int(completed.max())
            if reference_lap <= self.last_reference_lap:
                return False

            telemetry = aggregate_lap_telemetry(session)
            weather = pd.DataFrame(session.weather_data).copy()
            results = pd.DataFrame(session.results).copy()
            history = historical_features(
                self.year, self.round_number, laps
            )
            features = build_five_lap_features(
                laps=laps,
                telemetry=telemetry,
                weather=weather,
                results=results,
                qualifying=self.qualifying,
                history=history,
                event=self.event,
                total_race_laps=self.total_laps,
                require_future_targets=False,
            )
            if features.empty:
                logging.info(
                    "Lap %d received; waiting for five completed laps per driver",
                    reference_lap,
                )
                self.last_reference_lap = reference_lap
                return False

            latest = (
                features.sort_values(["DriverNumber", "LapNumber"])
                .groupby("DriverNumber", as_index=False, sort=False)
                .tail(1)
                .reset_index(drop=True)
            )
            self.store.save_feature_rows(
                self.session_key, reference_lap, latest
            )

            output_frames = []
            for horizon, loaded in sorted(self.models.items()):
                prediction = predict_leaderboard(latest, loaded)
                prediction.insert(0, "Horizon", horizon)
                prediction.insert(0, "ReferenceLap", reference_lap)
                prediction["PredictedAtLap"] = reference_lap + horizon
                self.store.save_predictions(
                    self.session_key, reference_lap, horizon, prediction
                )
                output_frames.append(prediction)

            combined = pd.concat(output_frames, ignore_index=True)
            self._write_latest_json(combined)
            self._print_predictions(combined)
            self.last_reference_lap = reference_lap
            return True
        finally:
            snapshot.unlink(missing_ok=True)

    def _write_latest_json(self, frame: pd.DataFrame) -> None:
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(
            frame.to_json(orient="records", indent=2), encoding="utf-8"
        )
        temporary.replace(self.output_path)

    @staticmethod
    def _print_predictions(frame: pd.DataFrame) -> None:
        for horizon, rows in frame.groupby("Horizon", sort=True):
            reference_lap = int(rows["ReferenceLap"].iloc[0])
            print(
                f"\nPrediction after lap {reference_lap}: "
                f"leaderboard {horizon} laps later",
                flush=True,
            )
            shown = rows[
                [
                    "PredictedPosition",
                    "Driver",
                    "DriverNumber",
                    "CurrentPosition",
                    "ChangeProbability",
                    "PredictedChange",
                ]
            ].copy()
            print(
                shown.to_string(
                    index=False,
                    formatters={
                        "ChangeProbability": "{:.1%}".format,
                        "PredictedChange": "{:+.2f}".format,
                    },
                ),
                flush=True,
            )


def run_processor(
    *,
    raw_path: Path,
    engine: LivePredictionEngine,
    store: TimeSeriesStore,
    poll_seconds: float,
    parse_seconds: float,
    stop_event: threading.Event | None = None,
    once: bool = False,
) -> None:
    tailer = RawStreamTailer(raw_path, store, engine.session_key)
    last_parse = 0.0
    while stop_event is None or not stop_event.is_set():
        topics = tailer.ingest_available()
        now = time.monotonic()
        should_parse = bool(topics.intersection(TIMING_TOPICS))
        if should_parse and now - last_parse >= parse_seconds:
            try:
                engine.update_from_recording(raw_path)
            except Exception:
                # Partial live snapshots are normal while FastF1 is still
                # assembling the session. Preserve the recording and retry.
                logging.exception("Live snapshot could not be compiled yet")
            last_parse = now
        if once:
            return
        time.sleep(poll_seconds)


def common_engine(arguments: argparse.Namespace) -> tuple[TimeSeriesStore, LivePredictionEngine]:
    if arguments.total_laps <= 0:
        raise ValueError("--total-laps must be positive")
    store = TimeSeriesStore(arguments.db)
    engine = LivePredictionEngine(
        year=arguments.year,
        round_number=arguments.round_number,
        total_laps=arguments.total_laps,
        horizons=arguments.horizons,
        store=store,
        output_path=arguments.output,
    )
    return store, engine


def record_command(arguments: argparse.Namespace) -> None:
    raw_path = arguments.raw.resolve()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    store, engine = common_engine(arguments)
    stop_event = threading.Event()
    processor = threading.Thread(
        target=run_processor,
        kwargs={
            "raw_path": raw_path,
            "engine": engine,
            "store": store,
            "poll_seconds": arguments.poll_seconds,
            "parse_seconds": arguments.parse_seconds,
            "stop_event": stop_event,
        },
        name="live-feature-processor",
        daemon=True,
    )
    processor.start()
    client = SignalRClient(
        filename=str(raw_path),
        filemode="a",
        timeout=arguments.timeout,
        no_auth=arguments.no_auth,
    )
    try:
        client.start()
    finally:
        stop_event.set()
        processor.join(timeout=max(10.0, arguments.parse_seconds + 2.0))
        # Capture any messages flushed immediately before shutdown.
        RawStreamTailer(raw_path, store, engine.session_key).ingest_available()


def process_command(arguments: argparse.Namespace) -> None:
    store, engine = common_engine(arguments)
    run_processor(
        raw_path=arguments.raw.resolve(),
        engine=engine,
        store=store,
        poll_seconds=arguments.poll_seconds,
        parse_seconds=arguments.parse_seconds,
        once=arguments.once,
    )


def show_command(arguments: argparse.Namespace) -> None:
    frame = TimeSeriesStore(arguments.db).latest_predictions(arguments.session)
    if frame.empty:
        print("No predictions have been stored yet.")
        return
    print(frame.to_string(index=False))


def add_live_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--total-laps", type=int, required=True)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        choices=sorted(DEFAULT_MODEL_PATHS),
        default=[5, 10],
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--output", type=Path, default=LIVE_DIR / "latest_predictions.json"
    )
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--parse-seconds",
        type=float,
        default=15.0,
        help="Minimum seconds between expensive FastF1 snapshot parses.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser("record", help="Record and predict live")
    add_live_arguments(record)
    record.add_argument("--raw", type=Path)
    record.add_argument("--timeout", type=int, default=120)
    record.add_argument(
        "--no-auth",
        action="store_true",
        help="Attempt an unauthenticated FastF1 live connection.",
    )
    record.set_defaults(handler=record_command)

    process = commands.add_parser(
        "process", help="Tail a raw recording made by another FastF1 client"
    )
    add_live_arguments(process)
    process.add_argument("--raw", type=Path, required=True)
    process.add_argument("--once", action="store_true")
    process.set_defaults(handler=process_command)

    show = commands.add_parser("show", help="Print newest stored predictions")
    show.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    show.add_argument("--session", help="Optional session key, for example 2026-R12")
    show.set_defaults(handler=show_command)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(arguments.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if arguments.command == "record" and arguments.raw is None:
        arguments.raw = LIVE_DIR / (
            f"{arguments.year}_r{arguments.round_number:02d}_raw.txt"
        )
    try:
        arguments.handler(arguments)
    except KeyboardInterrupt:
        logging.info("Stopped by user")
    except Exception as error:
        logging.error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
