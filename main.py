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
    learning_rate: float = 0.1,
    max_iter: int = 100,
    max_leaf_nodes: int | None = 31,
    max_depth: int | None = None,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.0,
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
