# src/utils/helpers.py

import logging
import logging.handlers
import time
import psutil
from datetime import datetime
from typing import Any
import json
from contextlib import contextmanager
from config.settings import LOGGING_CONFIG, LOGS_DIR


def setup_logging(name: str = "market_intelligence") -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOGGING_CONFIG["LEVEL"], "INFO"))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    formatter = logging.Formatter(LOGGING_CONFIG["FORMAT"])

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler (all logs)
    log_file = LOGS_DIR / f"{name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=LOGGING_CONFIG["MAX_BYTES"],
        backupCount=LOGGING_CONFIG["BACKUP_COUNT"],
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Rotating file handler (errors)
    error_file = LOGS_DIR / f"{name}_errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=LOGGING_CONFIG["MAX_BYTES"],
        backupCount=LOGGING_CONFIG["BACKUP_COUNT"],
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger


@contextmanager
def performance_monitor(operation_name: str):
    perf_logger = logging.getLogger("performance")
    start_time = time.time()
    proc = psutil.Process()
    start_memory = proc.memory_info().rss / (1024 * 1024)
    perf_logger.info(f"Starting {operation_name}")
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = proc.memory_info().rss / (1024 * 1024)
        perf_logger.info(
            f"Completed {operation_name}: Duration={end_time - start_time:.2f}s, "
            f"Memory Delta={end_memory - start_memory:.2f}MB, Peak Memory={end_memory:.2f}MB"
        )


def safe_json_serialize(obj: Any) -> str:
    def json_serializer(o: Any):
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    try:
        return json.dumps(obj, default=json_serializer, indent=2)
    except Exception as e:
        logging.getLogger(__name__).error(f"JSON serialization failed: {e}")
        try:
            return str(obj)
        except Exception:
            return "<unserializable object>"
