"""Global hotkey support.

Uses ``pynput`` when available so hotkeys work while another application
(for example the chess website in a browser) has focus. Imports are lazy:
the rest of the application works without pynput installed, only the global
hotkey is replaced by an in-window shortcut.
"""
from __future__ import annotations

from typing import Callable

from ..logging_setup import get_logger

LOGGER = get_logger(__name__)


class HotkeyService:
    """Registers a global show/hide hotkey through pynput."""

    def __init__(self, hotkey: str, on_toggle: Callable[[], None]) -> None:
        self.hotkey = hotkey
        self.on_toggle = on_toggle
        self._listener = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> bool:
        try:
            from pynput import keyboard
        except ImportError:
            LOGGER.warning("pynput not installed; global hotkey disabled.")
            return False
        try:
            # pynput uses the same '<ctrl>+<alt>+s' notation as our config.
            self._listener = keyboard.GlobalHotKeys({self.hotkey: self._handle})
            self._listener.daemon = True
            self._listener.start()
            self._available = True
            LOGGER.info("Global hotkey registered: %s", self.hotkey)
            return True
        except Exception as exc:  # pragma: no cover - platform specific
            LOGGER.warning("Failed to register global hotkey: %s", exc)
            self._available = False
            return False

    def _handle(self) -> None:
        try:
            self.on_toggle()
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.exception("Hotkey callback failed")

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            self._available = False

    def restart_with(self, hotkey: str) -> bool:
        """Register a different hotkey, returning whether it succeeded."""
        self.stop()
        self.hotkey = hotkey
        return self.start()
