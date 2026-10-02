"""Structured JSON logging configuration for production observability."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any


class JSONLogFormatter(logging.Formatter):
    """Formats log records as newline-delimited JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include structured context if present
        for key in ("request_id", "organization_id", "duration_ms", "status_code", "path", "method"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def configure_structured_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with the structured JSON formatter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONLogFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers on re-configuration
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
