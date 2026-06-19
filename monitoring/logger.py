"""
Structured JSON logger.
Every log line is a JSON object — easy to ship to Datadog, Logtail, CloudWatch, etc.
"""

import logging
import json
import time
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats every log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Merge any extra kwargs passed as keyword arguments
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


class StructuredLogger(logging.Logger):
    """Logger that accepts keyword kwargs and merges them into the JSON line."""

    def _log_structured(self, level: int, event: str, **kwargs):
        if self.isEnabledFor(level):
            record = self.makeRecord(
                self.name, level, "(unknown)", 0, event, (), None
            )
            record.extra_fields = kwargs  # type: ignore[attr-defined]
            self.handle(record)

    def info(self, event: str, *args, **kwargs):  # type: ignore[override]
        if kwargs:
            self._log_structured(logging.INFO, event, **kwargs)
        else:
            super().info(event, *args)

    def error(self, event: str, *args, **kwargs):  # type: ignore[override]
        if kwargs:
            self._log_structured(logging.ERROR, event, **kwargs)
        else:
            super().error(event, *args)

    def warning(self, event: str, *args, **kwargs):  # type: ignore[override]
        if kwargs:
            self._log_structured(logging.WARNING, event, **kwargs)
        else:
            super().warning(event, *args)


def get_logger(name: str) -> StructuredLogger:
    logging.setLoggerClass(StructuredLogger)
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger  # type: ignore[return-value]


def log_request(logger: StructuredLogger, request_id: str, **kwargs):
    """Convenience wrapper for request-scoped logs."""
    logger.info("request", request_id=request_id, **kwargs)