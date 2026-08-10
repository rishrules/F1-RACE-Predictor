"""Focused offline tests for the live timing bridge."""

from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

import numpy as np
import pandas as pd

from live_predictor import (
    RawStreamTailer,
    TimeSeriesStore,
    load_models,
    parse_recording_line,
)


class LivePredictorTests(unittest.TestCase):
    def test_recording_line_parser(self) -> None:
        raw = b"['TimingData', {'Lines': {'1': {'Position': '1'}}}, '12:00:00.000']\n"
        parsed = parse_recording_line(raw)
        self.assertIsNotNone(parsed)
        topic, timestamp, payload = parsed
        self.assertEqual(topic, "TimingData")
        self.assertEqual(timestamp, "12:00:00.000")
        self.assertIn(b'\"Position\":\"1\"', payload)

    def test_tailer_is_incremental_and_ignores_partial_line(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "recording.txt"
            raw_path.write_bytes(
                b"['TimingData', {'Lap': 5}, '12:00:00.000']\n"
                b"['WeatherData', {'AirTemp': '24.0'}, '12:00:01.000']"
            )
            store = TimeSeriesStore(root / "live.sqlite")
            tailer = RawStreamTailer(raw_path, store, "2026-R01")

            self.assertEqual(tailer.ingest_available(), {"TimingData"})
            self.assertEqual(tailer.ingest_available(), set())

            with raw_path.open("ab") as output:
                output.write(b"\n")
            self.assertEqual(tailer.ingest_available(), {"WeatherData"})

            connection = sqlite3.connect(store.path)
            try:
                count = connection.execute(
                    "SELECT COUNT(*) FROM raw_messages"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(count, 2)

            store.save_feature_rows(
                "2026-R01",
                5,
                pd.DataFrame(
                    [
                        {
                            "DriverNumber": "1",
                            "Driver": "AAA",
                            "LapNumber": 5.0,
                            "MissingFeature": np.nan,
                        }
                    ]
                ),
            )
            store.save_predictions(
                "2026-R01",
                5,
                10,
                pd.DataFrame(
                    [
                        {
                            "PredictedPosition": 1,
                            "DriverNumber": "1",
                            "Driver": "AAA",
                            "CurrentPosition": 2.0,
                            "ChangeProbability": 0.8,
                            "ConditionalChange": -1.0,
                            "PredictedChange": -0.8,
                            "PredictedPositionScore": 1.2,
                        }
                    ]
                ),
            )
            latest = store.latest_predictions("2026-R01")
            self.assertEqual(len(latest), 1)
            self.assertEqual(int(latest.iloc[0]["Horizon"]), 10)

    def test_both_live_model_horizons_are_distinct(self) -> None:
        models = load_models([5, 10])
        self.assertEqual(set(models), {5, 10})
        self.assertEqual(models[5].horizon, 5)
        self.assertEqual(models[10].horizon, 10)
        self.assertEqual(
            [c for c in models[5].feature_columns if c.startswith("Position")],
            ["Position"],
        )


if __name__ == "__main__":
    unittest.main()
