# polymarket/utils/logging.py
"""Structured logging configuration."""

import logging
import sys
from typing import Literal

_loggers: dict[str, logging.Logger] = {}


def setup_logging(
    level: str = "INFO",
    json: bool = False,
) -> logging.Logger:
    """
    Configure root logger with appropriate handlers.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json: If True, use JSON formatting; otherwise human-readable

    Returns:
        Configured root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)

    if json:
        # JSON formatting
        from structlog.stdlib import ProcessorFormatter

        handler.setFormatter(ProcessorFormatter())
    else:
        # Human-readable formatting with color
        from colorlog import ColoredFormatter

        handler.setFormatter(
            ColoredFormatter(
                "%(log_color)s%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        )

    root_logger.addHandler(handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name (typically __name__ of module)

    Returns:
        Logger instance
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]
