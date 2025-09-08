"""
App package initializer.

Holds lightweight metadata and a shared logger so other modules can:
    from app import logger, __version__
"""

from logging import getLogger, basicConfig, INFO

# Basic logging config (tweak later in core/logger.py if needed)
basicConfig(level=INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

logger = getLogger("ai-knowledge-agent")

__app_name__ = "ai-knowledge-agent"
__version__ = "0.1.0"

__all__ = ["logger", "__app_name__", "__version__"]
