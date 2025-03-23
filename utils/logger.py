import logging


def get_logger(name=__name__):
    """
    Creates and returns a configured logger instance.
    Args:
        name (str): Logger name.
    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
