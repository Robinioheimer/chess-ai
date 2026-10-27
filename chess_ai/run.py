"""Application entry point."""
from __future__ import annotations

import logging
import sys

from .app.controller import AppController
from .config import AppConfig
from .logging_setup import configure_logging, get_logger

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    if "--uci-engine" in argv:
        from .engine.builtin_engine import main as uci_main

        return uci_main()
    configure_logging(logging.INFO)
    LOGGER.info("Starting Chess AI Overlay")

    # Imported lazily so headless/CLI tests never require a display.
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication(argv)
    app.setApplicationName("Chess AI Overlay")
    app.setQuitOnLastWindowClosed(False)  # hidden overlay must keep running

    config = AppConfig.load()
    problems = config.validate()
    if problems:
        LOGGER.warning("Configuration issues: %s", "; ".join(problems))

    controller = AppController(config)
    controller.start()

    def _shutdown() -> None:
        controller.shutdown()

    app.aboutToQuit.connect(_shutdown)
    exit_code = app.exec_()
    controller.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
