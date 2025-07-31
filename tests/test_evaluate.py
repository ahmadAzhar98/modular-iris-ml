from src.model import train_model
from src.evaluate import evaluate_model


def test_evaluate_model(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    model = train_model(X_train, y_train, n_estimators=10)
    accuracy, report = evaluate_model(model, X_test, y_test)

    assert isinstance(accuracy, float)
    assert "precision" in report
