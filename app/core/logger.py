import logging
import sys
import json
from app.core.config import settings

class JsonFormatter(logging.Formatter):
    def format(self, record):
        record_dict = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "file": f"{record.filename}:{record.lineno}",
            "message": record.getMessage()
        }
        # include extra fields if present
        if hasattr(record, "request_id"):
            record_dict["request_id"] = record.request_id
        if hasattr(record, "duration"):
            record_dict["duration"] = record.duration
        return json.dumps(record_dict)

def get_logger(name: str = "ai-knowledge-agent"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = logging.DEBUG if settings.DEBUG else logging.INFO
    logger.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(JsonFormatter())
    logger.addHandler(ch)
    return logger

logger = get_logger()