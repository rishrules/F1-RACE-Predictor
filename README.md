# F1 Race Predictor

This project predicts a driver's position change over the next five race laps.

## Open the interactive training dashboard

Run Streamlit with the Python executable inside the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run main.py
```

Choose a validation race and model parameters in the sidebar, then select
**Train selected model**. Parameter sweeps, heatmaps, walk-forward validation,
permutation importance, and partial dependence run only when their buttons are
selected because they require additional model fits.

## Model analysis report

The completed chronological tuning, all-race validation, feature-importance, and
ablation report is available at
[`reports/model_analysis/MODEL_ANALYSIS_REPORT.md`](reports/model_analysis/MODEL_ANALYSIS_REPORT.md).

The multi-horizon, two-stage, recency, and position-state comparison is available
at [`reports/model_improvements/IMPROVEMENT_REPORT.md`](reports/model_improvements/IMPROVEMENT_REPORT.md).

The production training configuration is fixed to the two-stage model with a
10-lap target, a 20-race recency half-life, and only the current `Position`
input (lagged and rolling position-state inputs are excluded).

Reproduce or resume the improvement comparison with:

```powershell
.\.venv\Scripts\python.exe -u improvement_analysis.py
```

## Live race predictions

`live_predictor.py` records FastF1's append-only live stream, mirrors every
complete message into an indexed SQLite time-series database, compiles the
latest completed five-lap window, and prints separate five- and ten-lap
leaderboards.

Start it two or three minutes before the race so the FastF1 recording contains
the initial session and driver messages. Supply the scheduled race distance;
do not use the current lap as `--total-laps`.

```powershell
.\.venv\Scripts\python.exe live_predictor.py record `
  --year 2026 `
  --round 12 `
  --total-laps 70 `
  --horizons 5 10
```

The command creates:

- `data/live/2026_r12_raw.txt`: untouched FastF1 raw stream recording.
- `data/live/live_timing.sqlite`: raw messages, compiled lap features, and
  prediction time series. SQLite uses WAL mode so another process can read it.
- `data/live/latest_predictions.json`: atomically replaced latest leaderboard.

If FastF1 is already recording in another terminal, tail its file instead:

```powershell
.\.venv\Scripts\python.exe live_predictor.py process `
  --year 2026 `
  --round 12 `
  --total-laps 70 `
  --raw data/live/2026_r12_raw.txt `
  --horizons 5 10
```

Read the newest predictions without stopping the recorder:

```powershell
.\.venv\Scripts\python.exe live_predictor.py show
```

The five-lap input window and five-/ten-lap target horizons are different
concepts. After lap 25, both models use laps 21–25; one predicts lap 30 and the
other lap 35. Predictions begin only after a driver has five completed laps.
The pipeline never treats an incomplete lap as model input.

The experiment is resumable and can be reproduced with:

```powershell
.\.venv\Scripts\python.exe -u model_analysis.py --stage all
```
