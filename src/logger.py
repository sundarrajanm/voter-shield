"""
Centralized logging configuration.

Provides structured logging with:
- Console output (INFO or DEBUG based on environment)
- File output (always DEBUG for troubleshooting)
- Automatic log file rotation

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.debug("Detailed debug info")  # Only shown when DEBUG=1
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import get_config


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "electorials",
    log_dir: Optional[Path] = None,
    debug: Optional[bool] = None,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Setup and configure a logger.
    
    Args:
        name: Logger name (typically __name__)
        log_dir: Directory for log files (default from config)
        debug: Enable debug mode (default from environment/config)
        log_to_file: Whether to write logs to file
    
    Returns:
        Configured logger instance
    """
    config = get_config()
    
    if debug is None:
        debug = config.debug
    
    if log_dir is None:
        log_dir = config.logs_dir
    
    # Determine log level
    level = logging.DEBUG if debug else logging.INFO
    
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Use colored formatter for console if terminal supports it
    if sys.stdout.isatty():
        console_format = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (always DEBUG level for troubleshooting)
    if log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_filename = f"{datetime.now():%Y%m%d_%H%M%S}.log"
        log_file = log_dir / log_filename
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        # Log the log file location
        logger.debug(f"Log file: {log_file}")
    
    return logger


# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str = "electorials") -> logging.Logger:
    """
    Get a logger instance.
    
    Creates and caches logger instances. Use this for consistent logging
    throughout the application.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Starting processing")
    """
    if name not in _loggers:
        _loggers[name] = setup_logger(name)
    return _loggers[name]


def log_timing(logger: logging.Logger, operation: str, duration_sec: float) -> None:
    """Log timing information for an operation."""
    if duration_sec < 1:
        logger.debug(f"{operation}: {duration_sec * 1000:.1f}ms")
    elif duration_sec < 60:
        logger.info(f"{operation}: {duration_sec:.2f}s")
    else:
        minutes = int(duration_sec // 60)
        seconds = duration_sec % 60
        logger.info(f"{operation}: {minutes}m {seconds:.1f}s")


def log_progress(logger: logging.Logger, current: int, total: int, item: str = "item") -> None:
    """Log progress information."""
    percent = (current / total * 100) if total > 0 else 0
    logger.info(f"Progress: {current}/{total} ({percent:.1f}%) - {item}")
