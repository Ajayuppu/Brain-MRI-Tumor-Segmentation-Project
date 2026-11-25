"""One logging format for every script: time, level, message; optionally also written to a file."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)
