import logging
import sys

from src.config import LOG_LEVEL


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s | %(message)s"
)

_DATE_FORMAT = "%H:%M:%S"


def configure_logging() -> None:
    """
    Configure application-wide console logging.

    This function is safe to call more than once.
    """

    root_logger = logging.getLogger()

    if root_logger.handlers:
        return

    numeric_level = getattr(
        logging,
        LOG_LEVEL.upper(),
        logging.INFO,
    )

    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        fmt=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )

    handler.setFormatter(formatter)

    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)