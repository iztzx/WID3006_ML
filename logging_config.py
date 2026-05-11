"""Structured JSON logging configuration for IntentSight v3."""

import logging
import sys
from pythonjsonlogger import jsonlogger


def setup_logging() -> logging.Logger:
    """Configure and return the root logger with JSON formatting."""
    logger = logging.getLogger("intentsight")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on reload
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"levelname": "severity", "asctime": "timestamp"},
        static_fields={"service": "intentsight", "environment": "production"},
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("flaml").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logger


logger = setup_logging()