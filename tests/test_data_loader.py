from src.data_loader import load_data


def test_load_data():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0
