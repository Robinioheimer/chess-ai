"""Top-level application controller.

Owns and wires every component - configuration, vision pipeline, position
tracker, screen watcher, the single long-lived Stockfish process, the
analysis manager and the overlay window. Worker threads communicate with the
GUI exclusively through :class:`SignalBridge` signals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import chess

from ..config import AppConfig, config_dir
from ..core.tracker import PositionTracker, PositionUpdate
from ..engine.analyzer import AnalysisManager
from ..engine.stockfish_setup import StockfishInfo, ensure_stockfish, probe_stockfish
from ..engine.uci import UciEngine, UciEngineError
from ..logging_setup import get_logger
from ..overlay.dialogs import CorrectionDialog, FenDialog, SettingsDialog
from ..overlay.main_window import MainOverlay
from ..overlay.qt_bridge import SignalBridge
from ..vision.capture import ScreenCapture
from ..vision.pipeline import RecognitionPipeline, RecognitionResult
from ..vision.recognizer import PieceRecognizer
from .hotkeys import HotkeyService
from .screen_watcher import ScreenWatcher

LOGGER = get_logger(__name__)
TEMPLATE_PATH = config_dir() / "templates.npz"


class AppController:
    """Composition root for the running application."""

    def __init__(self, config: AppConfig, capture: ScreenCapture | None = None) -> None:
        self.config = config
        self.bridge = SignalBridge()
        initial_side = (
            chess.BLACK if config.vision.side_to_move_default == "b" else chess.WHITE
        )
        self.tracker = PositionTracker(side_to_move=initial_side)
        self.recognizer = PieceRecognizer()
        self._load_templates()
        self.pipeline = RecognitionPipeline(
            recognizer=self.recognizer,
            warp_size=config.vision.warp_size,
            stable_readings=config.vision.stable_readings,
            min_board_fraction=config.vision.min_board_fraction,
        )
        self._apply_perspective_setting()
        self.window = MainOverlay(config)
        self.capture = capture or ScreenCapture()
        self.watcher = ScreenWatcher(
            self.pipeline,
            self.tracker,
            config.vision,
            capture=self.capture,
            on_recognition=self.bridge.recognition_ready.emit,
            on_position=self._on_position,
            on_error=self.bridge.error_message.emit,
        )

        self.engine: Optional[UciEngine] = None
        self.analyzer: Optional[AnalysisManager] = None
        self.engine_info: Optional[StockfishInfo] = None
        self.hotkeys = HotkeyService(
            config.overlay.show_hotkey, self._toggle_overlay_visibility
        )
        self._connect_window()
        self._connect_bridge()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.capture.__enter__()
        self._start_engine()
        self.watcher.start()
        if self.analyzer is not None:
            self.analyzer.start()
        self.hotkeys.start()
        self.window.cover_screen()
        self.window.set_click_through(self.config.overlay.click_through)
        self.window.show()
        self.bridge.status_message.emit("Overlay ready.")

    def shutdown(self) -> None:
        LOGGER.info("Shutting down Chess AI")
        try:
            self.hotkeys.stop()
            self.watcher.stop()
            if self.analyzer is not None:
                self.analyzer.shutdown()
        finally:
            self.capture.__exit__(None, None, None)
            self.config.save()

    # ---------------------------------------------------------------- engine
    def _start_engine(self) -> None:
        try:
            self.engine_info = ensure_stockfish(self.config.engine.stockfish_path)
            probe_stockfish(self.engine_info.path)
        except UciEngineError as exc:
            LOGGER.error("Stockfish unavailable: %s", exc)
            self.bridge.engine_state.emit(
                "unavailable", "Engine: Stockfish unavailable"
            )
            self.bridge.error_message.emit(
                f"Stockfish is not available ({exc}). Analysis disabled until a "
                "valid engine path is configured in Settings."
            )
            return
        except Exception as exc:  # network failures during auto-download
            LOGGER.exception("Stockfish setup failed")
            self.bridge.engine_state.emit(
                "unavailable", "Engine: automatic setup failed"
            )
            self.bridge.error_message.emit(
                f"Stockfish automatic setup failed: {exc}. Configure the engine "
                "path manually in Settings."
            )
            return
        self.config.engine.stockfish_path = self.engine_info.path
        self.engine = UciEngine(self.engine_info.path)
        self.analyzer = AnalysisManager(
            self.engine,
            self.config.engine,
            on_result=self.bridge.analysis_ready.emit,
            on_status=self.bridge.status_message.emit,
        )
        self.bridge.engine_state.emit(
            "connected", f"Engine: {self.engine_info.name} — ready"
        )
        LOGGER.info("Using Stockfish: %s (%s)", self.engine_info.name, self.engine_info.path)

    def _restart_engine(self) -> None:
        if self.analyzer is not None:
            self.analyzer.shutdown()
            self.analyzer = None
            self.engine = None
        self._start_engine()
        if self.analyzer is not None:
            self.analyzer.start()
            if self.tracker.fen:
                self.analyzer.request_analysis(self.tracker.fen)

    # -------------------------------------------------------------- wiring
    def _connect_window(self) -> None:
        window = self.window
        window.paused_changed.connect(self._set_paused)
        window.reanalyze_requested.connect(self._reanalyze)
        window.fen_requested.connect(self._open_fen_dialog)
        window.correction_requested.connect(self._open_correction_dialog)
        window.calibration_requested.connect(self._calibrate)
        window.settings_requested.connect(self._open_settings)

    def _connect_bridge(self) -> None:
        self.bridge.recognition_ready.connect(self.window.update_recognition)
        self.bridge.analysis_ready.connect(self.window.update_analysis)
        self.bridge.status_message.connect(self.window.set_status)
        self.bridge.error_message.connect(self.window.show_error)
        self.bridge.engine_state.connect(self.window.set_engine_state)
        # Cross-thread position events are delivered to the GUI thread via the
        # queued connection before any widget state is touched.
        self.bridge.position_ready.connect(self._handle_position_on_gui_thread)

    def _on_position(self, update: PositionUpdate, result: RecognitionResult) -> None:
        # Runs on the screen-watcher thread: only thread-safe work here.
        if self.analyzer is not None and update.fen:
            self.analyzer.request_analysis(update.fen)
        self.bridge.position_ready.emit(update, result)

    def _handle_position_on_gui_thread(
        self, update: PositionUpdate, result: RecognitionResult
    ) -> None:
        self.window.reset_hidden_moves()

    # ------------------------------------------------------------- controls
    def _toggle_overlay_visibility(self) -> None:
        if self.window.isVisible():
            self.window.hide()
        else:
            self.window.show()
            self.window.raise_()
            self.window.activateWindow()

    def _set_paused(self, paused: bool) -> None:
        self.config.paused = paused
        if paused:
            self.watcher.pause()
            if self.analyzer is not None:
                self.analyzer.pause()
        else:
            self.watcher.resume()
            if self.analyzer is not None:
                self.analyzer.resume()

    def _reanalyze(self) -> None:
        if self.analyzer is None:
            self.bridge.error_message.emit("No Stockfish engine is configured.")
            return
        if self.tracker.fen is None:
            self.bridge.error_message.emit("No position has been detected yet.")
            return
        self.analyzer.reanalyze()

    def _open_fen_dialog(self) -> None:
        current = self.tracker.fen or chess.STARTING_FEN
        dialog = FenDialog(current, self.window)
        if dialog.exec_() == FenDialog.Accepted:
            update = self.tracker.set_manual_fen(dialog.fen())
            if update.accepted and self.analyzer is not None:
                self.analyzer.request_analysis(update.fen)
            self.bridge.status_message.emit("Manual FEN applied.")

    def _open_correction_dialog(self) -> None:
        if self.window._recognition is None or not self.window._recognition.success:
            self.bridge.error_message.emit("No recognized board available to correct.")
            return
        latest = self.window._recognition
        dialog = CorrectionDialog(
            latest.screen_grid, latest.perspective_white, self.window
        )
        if dialog.exec_() == CorrectionDialog.Accepted:
            fen = dialog.corrected_fen()
            update = self.tracker.set_manual_fen(fen)
            if update.accepted and self.analyzer is not None:
                self.analyzer.request_analysis(update.fen)
            self.bridge.status_message.emit("Manual correction applied.")

    def _calibrate(self) -> None:
        try:
            frame = self.capture.grab()
            count = self.pipeline.calibrate_from_starting_position(frame)
        except ValueError as exc:
            self.bridge.error_message.emit(str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Calibration failed")
            self.bridge.error_message.emit(f"Calibration failed: {exc}")
            return
        try:
            self.recognizer.save(TEMPLATE_PATH)
        except ValueError:
            pass
        self.bridge.status_message.emit(
            f"Calibrated {count} piece templates from the visible board."
        )

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self.window)
        if dialog.exec_() != SettingsDialog.Accepted:
            return
        previous_hotkey = self.hotkeys.hotkey
        self.config.save()
        self.window.apply_opacity(self.config.overlay.opacity)
        self.window.apply_scale(self.config.overlay.scale)
        self.window.set_click_through(self.config.overlay.click_through)
        self.pipeline.stable_readings = self.config.vision.stable_readings
        self._apply_perspective_setting()
        if self.analyzer is not None:
            engine_path_changed = (
                self.engine_info is None
                or self.config.engine.stockfish_path != self.engine_info.path
            )
            if engine_path_changed:
                self._restart_engine()
            else:
                self.analyzer.update_config(self.config.engine)
        elif self.config.engine.stockfish_path:
            self._restart_engine()
        if previous_hotkey != self.config.overlay.show_hotkey:
            if not self.hotkeys.restart_with(self.config.overlay.show_hotkey):
                self.bridge.error_message.emit(
                    "Could not register the new global hotkey; the old one is inactive."
                )
        self.bridge.status_message.emit("Settings saved.")

    def _apply_perspective_setting(self) -> None:
        mode = self.config.vision.perspective
        if mode == "white":
            self.pipeline.set_manual_perspective(True)
        elif mode == "black":
            self.pipeline.set_manual_perspective(False)
        else:
            self.pipeline.clear_manual_perspective()

    # ------------------------------------------------------------ templates
    def _load_templates(self) -> None:
        path = Path(TEMPLATE_PATH)
        if path.exists():
            try:
                self.recognizer.load(path)
                LOGGER.info("Loaded piece templates from %s", path)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Could not load templates: %s", exc)
