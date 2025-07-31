from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Tuple


def load_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the Iris dataset and split it into training and testing sets.

    Args:
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    rng = np.random.default_rng(random_state)
    X += rng.normal(0, 0.2, X.shape)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
