"""
utils/logging_config.py

Structured logging configuration. Log level is driven by the LOG_LEVEL env var.
"""
import logging
import os
import sys

import structlog
from dotenv import load_dotenv

load_dotenv()


def configure_logging() -> None:
    """Set up structlog with PrintLoggerFactory (no stdlib bridge needed)."""
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Basic stdlib logging (for third-party libs that use it)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__):
    return structlog.get_logger(name)
