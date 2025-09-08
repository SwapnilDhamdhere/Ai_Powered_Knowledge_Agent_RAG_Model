import logging
import sys
from app.core.config import settings

def get_logger(name: str = "ai-knowledge-agent") -> logging.Logger:
    """Creates a structured logger for the app."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers in interactive environments
    if logger.handlers:
        return logger

    # Log format
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )

    # Set level based on DEBUG mode
    level = logging.DEBUG if settings.DEBUG else logging.INFO
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    return logger

# Global app logger
logger = get_logger()
