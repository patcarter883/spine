"""SPINE logging configuration — configure logging levels from CLI flags."""

from __future__ import annotations

import logging
import sys


def configure_logging(verbose: bool = False) -> None:
    """
    Configure the root logger with appropriate settings.

    Args:
        verbose: If True, set DEBUG level with detailed format.
                 If False, set WARNING level with simple format.
    """
    if verbose:
        level = logging.DEBUG
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        level = logging.WARNING
        format_str = "%(levelname)s: %(message)s"

    # Force reconfiguration by removing existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format=format_str,
        stream=sys.stdout,
        force=True
    )
