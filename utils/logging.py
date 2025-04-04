# ct2fea/utils/logging.py
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import sys
from .errors import CT2FEAError


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON

        Args:
            record: Log record to format

        Returns:
            JSON formatted log string
        """
        # Basic log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            log_data["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
            }

            # Add detailed error info for our custom exceptions
            if isinstance(exc_value, CT2FEAError) and exc_value.details:
                log_data["exception"]["details"] = exc_value.details

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data)


class PipelineLogger:
    """Enhanced logger for pipeline operations"""

    def __init__(self, name: str):
        """Initialize pipeline logger

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.stage = None
        self.start_time = None

    def start_stage(self, stage: str) -> None:
        """Start timing a pipeline stage

        Args:
            stage: Stage name
        """
        self.stage = stage
        self.start_time = datetime.now()
        self.logger.info(f"Starting stage: {stage}")

    def end_stage(self, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """End timing a pipeline stage

        Args:
            extra_data: Optional additional data to log
        """
        if self.stage and self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            log_data = {"stage": self.stage, "duration_seconds": duration}
            if extra_data:
                log_data.update(extra_data)

            self.logger.info(
                f"Completed stage: {self.stage} in {duration:.2f}s",
                extra={"extra_data": log_data},
            )

        self.stage = None
        self.start_time = None


def setup_logging(
    output_dir: Path,
    level: int = logging.INFO,
    enable_console: bool = True,
    log_format: str = "structured",
) -> None:
    """Configure logging with both console and file handlers

    Args:
        output_dir: Directory where log files will be saved
        level: Logging level
        enable_console: Whether to enable console output
        log_format: Format type ("structured" or "simple")
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    if log_format == "structured":
        file_formatter = StructuredFormatter()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        file_formatter = console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # File handler (JSON structured logging)
    log_file = output_dir / "processing.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Debug log file (all messages)
    debug_file = output_dir / "debug.log"
    debug_handler = logging.FileHandler(debug_file)
    debug_handler.setFormatter(file_formatter)
    debug_handler.setLevel(logging.DEBUG)
    logger.addHandler(debug_handler)

    # Error log file (errors only)
    error_file = output_dir / "errors.log"
    error_handler = logging.FileHandler(error_file)
    error_handler.setFormatter(file_formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)


def log_error(logger: logging.Logger, error: Exception, stage: str) -> None:
    """Log an error with enhanced context

    Args:
        logger: Logger instance
        error: Exception to log
        stage: Pipeline stage where error occurred
    """
    if isinstance(error, CT2FEAError):
        logger.error(
            f"Error in {stage}: {error.message}",
            extra={"extra_data": {"stage": stage, "error_details": error.details}},
        )
    else:
        logger.error(
            f"Unexpected error in {stage}: {str(error)}",
            exc_info=True,
            extra={"extra_data": {"stage": stage}},
        )
