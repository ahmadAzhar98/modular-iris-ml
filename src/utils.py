import logging


def setup_logger() -> logging.Logger:
    """
    Set up and return a logger instance for the project.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)
