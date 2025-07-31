from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple


def evaluate_model(model, X_test, y_test) -> Tuple[float, str]:
    """
    Evaluate the trained model using accuracy and classification report.

    Args:
        model: Trained model instance.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Tuple: Accuracy score, classification report (string)
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report
