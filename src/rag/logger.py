import logging
import os
from datetime import datetime

def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create and configure a logger with both console and file output handlers.

    This utility ensures consistent logging across the RAG application by creating:
    - A daily log file stored under the specified directory.
    - A console stream handler for runtime visibility.
    - A check to avoid attaching duplicate handlers.

    Parameters
    ----------
    name : str
        Name of the logger instance, usually set to ``__name__`` of the module
        requesting logging.
    log_dir : str, optional
        Directory where log files will be stored. Defaults to ``"logs"``.

    Returns
    -------
    logging.Logger
        A configured logger instance with INFO-level logging enabled.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now():%Y-%m-%d}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if logger already exists
    if logger.hasHandlers():
        return logger

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
