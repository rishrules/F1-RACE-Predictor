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

The experiment is resumable and can be reproduced with:

```powershell
.\.venv\Scripts\python.exe -u model_analysis.py --stage all
```
