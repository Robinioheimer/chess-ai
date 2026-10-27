"""Thread-safe bridge between worker threads and the Qt GUI.

Workers (screen watcher, analysis manager) call plain Python callbacks. This
bridge converts those calls into Qt signals, so every GUI update executes on
the main thread and the interface stays responsive during engine searches.
"""
from __future__ import annotations

from PyQt5 import QtCore


class SignalBridge(QtCore.QObject):
    recognition_ready = QtCore.pyqtSignal(object)
    position_ready = QtCore.pyqtSignal(object, object)
    analysis_ready = QtCore.pyqtSignal(object)
    status_message = QtCore.pyqtSignal(str)
    error_message = QtCore.pyqtSignal(str)
    # state token (connected / analyzing / unavailable / not_connected), text
    engine_state = QtCore.pyqtSignal(str, str)
