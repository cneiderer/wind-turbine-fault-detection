import logging
from pathlib import Path

from wtfd.config import load_config, PROJECT_ROOT


def get_logger(name: str = "wtfd") -> logging.Logger:
    """
    Configure and return a project logger.

    This logger writes to both the console and a log file.
    Repeated calls with the same logger name will not duplicate handlers.
    """
    config = load_config()

    log_level_str = config.get("logging", {}).get("level", "INFO").upper()
    log_file_name = config.get("logging", {}).get("file_name", "wtfd.log")
    logs_dir = PROJECT_ROOT / config["paths"]["logs"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level_str, logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(logs_dir / log_file_name)
    file_handler.setLevel(getattr(logging, log_level_str, logging.INFO))
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, log_level_str, logging.INFO))
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger