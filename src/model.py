from sklearn.linear_model import LogisticRegression
import joblib
from typing import Any


def train_model(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state (int): Random seed for reproducibility.

    Returns:
        LogisticRegression: Trained model.
    """
    model = LogisticRegression(max_iter=200, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def save_model(model: Any, path: str = "logistic_model.pkl") -> None:
    """
    Save the trained model to disk.

    Args:
        model (Any): Trained model instance.
        path (str): File path for saving the model.
    """
    joblib.dump(model, path)


def load_model(path: str = "logistic_model.pkl") -> Any:
    """
    Load a trained model from disk.

    Args:
        path (str): File path to the saved model.

    Returns:
        Any: Loaded model instance.
    """
    return joblib.load(path)
