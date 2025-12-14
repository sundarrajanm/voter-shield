# logger.py
import logging
from rich.logging import RichHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "votershield.log"

def setup_logger():
    logger = logging.getLogger("votershield")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate logs

    # --- Console (Rich) handler ---
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False
    )
    console_handler.setLevel(logging.INFO)

    # --- File handler ---
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Attach handlers (only once)
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
