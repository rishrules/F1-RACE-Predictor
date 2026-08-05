from typing import Any

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingRegressor


def create_hist_gradient_boosting_regressor(
    **parameters: Any,
) -> HistGradientBoostingRegressor:
    """Create an untrained position regressor with overridable parameters.

    The returned estimator is intentionally not fitted. Pass training data to
    its ``fit`` method later, after preprocessing and chronological splitting.
    """
    defaults: dict[str, Any] = {
        "loss": "squared_error",
        "learning_rate": 0.1,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "early_stopping": "auto",
        "random_state": 42,
    }
    defaults.update(parameters)
    return HistGradientBoostingRegressor(**defaults)
