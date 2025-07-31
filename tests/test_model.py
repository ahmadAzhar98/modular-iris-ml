from src.preprocessing import preprocess_data


def test_preprocess_data(iris_data):
    X_train, X_test, _, _ = iris_data
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert hasattr(scaler, "transform")
