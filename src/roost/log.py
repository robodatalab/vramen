"""Logging setup utilities."""

import faulthandler
import logging
import signal
import sys
from typing import Any

FORMAT = "%(asctime)s %(process)6d %(name)-12s %(message)s"
TIME_FORMAT = "%H:%M:%S"

MARK = "_roost_handler"


def setup() -> None:
    root = logging.getLogger()
    if any(getattr(handler, MARK, False) for handler in root.handlers):
        return

    handler: Any = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT, datefmt=TIME_FORMAT))
    setattr(handler, MARK, True)

    root.addHandler(handler)
    root.setLevel(logging.INFO)

    faulthandler.enable()

    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)


def logger(name: str) -> logging.Logger:
    """Short names: the format column is 12 wide, and `roost.resource_manager`
    would run past it."""
    return logging.getLogger(name.rsplit(".", 1)[-1])
