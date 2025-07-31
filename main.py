from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model, save_model
from src.evaluate import evaluate_model
from src.utils import setup_logger


def main() -> None:
    """
    Orchestrates the data loading, preprocessing, training,
    evaluation, and model saving pipeline.
    """
    logger = setup_logger()

    logger.info("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    logger.info("Preprocessing data...")
    X_train_scaled, X_test_scaled, _ = preprocess_data(X_train, X_test)

    logger.info("Training Logistic Regression model...")
    model = train_model(X_train_scaled, y_train)

    logger.info("Evaluating model...")
    accuracy, report = evaluate_model(model, X_test_scaled, y_test)
    logger.info("Model Accuracy: %.4f", accuracy)
    logger.info("Classification Report:\n%s", report)

    logger.info("Saving trained model...")
    save_model(model, path="logistic_model.pkl")

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
