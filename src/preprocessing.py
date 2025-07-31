from sklearn.preprocessing import StandardScaler
from typing import Tuple
import pandas as pd
import numpy as np


def preprocess_data(
    X_train: pd.DataFrame,X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale the training and testing datasets using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.

    Returns:
        Tuple: Scaled X_train, scaled X_test, fitted scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
