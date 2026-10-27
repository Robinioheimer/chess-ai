# Chess AI Overlay

A desktop application that **automatically detects a chessboard on your screen**,
recognizes the position, analyzes it continuously with **Stockfish (UCI)**, and
paints the best moves directly over the board in a **transparent, movable,
resizable overlay** — without modifying the underlying application.

> Intended for analyzing personal games, training and studying chess positions.
> The app never executes moves for you. Respect the rules of the platform you use.

---

## Features

### Board recognition
- Automatic chessboard detection from screenshots at different sizes, board
  colors, rotations and perspectives (White or Black at the bottom).
- Full 8×8 piece recognition combining shape heuristics with an optional
  one-click **template calibration** learned from a visible starting position
  (handles custom piece designs).
- Robust to screenshot noise and JPEG blur: an Otsu contrast floor and a
  largest-component gate reject sensor/compression artifacts on empty squares.
- Supports the standard starting position and arbitrary legal positions.
- Minimum supported on-screen board size is about **48 px per square**
  (~384 px board width); smaller boards are below the reliable recognition
  limit and should be zoomed in.
- Stability filtering (N identical readings) plus a stateful position tracker
  suppress flicker and duplicate move detections; the side to move alternates
  automatically after every detected move.
- **Manual FEN input** and an 8×8 **correction grid** for misrecognized squares;
  low-confidence squares are listed in the panel.

### Stockfish analysis
- Raw UCI integration over a long-lived subprocess — the engine is started
  **once** and reused (`stop`/`go`) instead of being restarted per move.
- Configurable Stockfish executable/version, threads, hash memory, search
  depth (or move-time cap) and **MultiPV** (top 5–10 candidates, up to 500).
- "Analyze every legal move" mode sets MultiPV to the number of legal moves.
- Continuous analysis that restarts automatically after every detected move,
  with throttled live updates so the UI never freezes.
- Scores are always presented **from the perspective of the side to move**.

### Honest percentages — three different numbers
The UI keeps these concepts strictly separate (see panel legend):

1. **Eval** — Stockfish's engine evaluation in centipawns (`+1.23`) or
   mate-in-N notation (`M5`). A search output, not a probability.
2. **Win % (est.)** — a *calibrated estimate* of the expected game result using
   the Lichess logistic model `P = 1 / (1 + exp(-0.003682237 · cp))`, with mate
   scores mapped to bounded pseudo-centipawns. It is an expectation over many
   games, **never a guarantee that a move wins this game**.
3. **Search confidence** — technical reliability of the current search (depth
   reached vs. target and best-move stability), shown independently from the
   winning probability.

### Overlay
- Frameless, translucent, always-on-top PyQt5 window; drag the title bar to
  move it, grip to resize, opacity/scale sliders in Settings.
- **Best move**: dark-blue corner markers (`#0B2A6B`) on source and
  destination squares, drawn over the detected quadrilateral (works with
  rotated/perspective boards).
- Alternative candidates get distinct colors and arrows.
- Candidate table with per-move: SAN notation, evaluation, estimated win %,
  opponent-perspective evaluation, depth, and a short auto-generated
  explanation (captures, checks, castling, promotion, tactical gaps…).
- Per-move show/hide checkboxes, live progress bar and depth/nodes/nps status.
- A colored **engine-state indicator** (green = connected and ready,
  amber = analyzing, red = unavailable, grey = not connected) shows the
  Stockfish connection at a glance.
- Errors appear in a non-modal, dismissible **error banner** with a
  **✕ Clear** button, so recognition and analysis keep running underneath
  instead of being blocked by popup dialogs.

### Controls
- Global show/hide hotkey (default `Ctrl+Alt+S`, pynput — works while another
  app is focused; the app also runs without pynput).
- Pause/resume, Reanalyze, manual FEN, correction grid, piece calibration,
  Settings dialog.
- Automatic White/Black perspective detection with manual override.
- Clear error messages for recognition and engine communication failures
  (dismissible banner plus rotating diagnostic logs).

---

## Architecture

```
chess_ai/
├── core/        # Qt-free chess logic: FEN/grids, calibrated probabilities,
│                # SAN/notation, move explanations, position tracker
├── vision/      # screen capture, quadrilateral detection + grid scoring,
│                # perspective warp, cell features, heuristic/template recognizer
├── engine/      # UCI client, continuous analysis manager, Stockfish setup
├── overlay/     # PyQt5 transparent window, markers, candidate panel, dialogs
├── app/         # screen watcher thread, global hotkeys, controller (wiring)
├── config.py    # persisted JSON settings
├── doctor.py    # `python -m chess_ai.doctor` self-test
└── run.py       # entry point (`chess-ai` / `python -m chess_ai`)
tests/           # 150+ pytest tests, including a deterministic fake UCI engine
```

Worker threads (screen watcher, UCI reader, analysis worker) communicate with
the GUI only through Qt signals, keeping the interface responsive during
searches.

---

## Installation (Windows 10/11)

Requires **Python 3.10 or newer**.

```bat
git clone https://github.com/<your-user>/chess-ai.git
cd chess-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Stockfish

On first launch the app tries, in order:

1. The path configured in **Settings → Engine**
2. The `STOCKFISH_PATH` environment variable
3. `stockfish` on your `PATH`
4. Common install locations (`Program Files`, Scoop, Chocolatey, local `bin/`)
5. **Automatic download** of the matching official release from the
   [Stockfish GitHub releases](https://github.com/official-stockfish/Stockfish/releases)
   into `%APPDATA%\ChessAI\bin`

To install manually instead: download Stockfish, unzip it, and either put
`stockfish.exe` on your `PATH` or point Settings at the full `.exe` path.

Settings and learned piece templates live in `%APPDATA%\ChessAI\`
(logs under `logs\`).

### Linux/macOS (development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
# system Stockfish, e.g. Debian/Ubuntu:
sudo apt install stockfish
```

---

## Running

```bash
python -m chess_ai
# or, after `pip install .`:
chess-ai
```

1. Open a chessboard (browser, desktop app, PGN viewer…).
2. The overlay locks onto the board and starts analysis; the status bar shows
   detection confidence and any uncertain squares.
3. Use **◈ Calibrate pieces** once on a visible starting position if you use a
   custom piece theme.
4. `Ctrl+Alt+S` hides/restores the overlay; **⏸ Pause** freezes recognition and
   analysis; **⟳ Reanalyze** re-runs the current position.

### Diagnostics

```bash
python -m chess_ai.doctor     # dependency, vision and Stockfish self-test
```

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers board detection/warping, feature extraction and piece
classification, FEN parsing and grid transforms, SAN/notation, the calibrated
probability model, position tracking/de-duplication, UCI parsing and the full
analysis manager against a deterministic **fake UCI engine**, plus headless
Qt smoke tests for the overlay and an end-to-end controller test.

---

## Packaging as a Windows `.exe`

```bat
pip install -r requirements-dev.txt
packaging\build_windows.bat
```

or manually:

```bat
pyinstaller packaging\chess_ai.spec --noconfirm --clean
```

The onedir build appears at `dist\ChessAI\ChessAI.exe`. Ship the whole
`dist\ChessAI` folder. Stockfish is downloaded automatically on first run (or
place a portable `stockfish.exe` in a `bin\` folder next to the executable —
the app checks there as well). Logs remain in `%APPDATA%\ChessAI\logs`.

---

## Performance and reliability notes

- Polling-based capture (default 1 s) keeps CPU usage low; recognition of a
  480 px warped board takes only a few milliseconds.
- Engine options are applied once; position changes reuse the same process and
  hash tables, and stale searches are interrupted immediately.
- The position tracker validates every reading with python-chess, rejects
  illegal placements, de-duplicates identical frames and alternates the side to
  move only on real changes.
- Rotating logs (1 MB × 3) provide diagnostics without flooding the disk.

## Limitations

- Heuristic recognition targets conventional piece silhouettes; heavily
  stylized themes work best after template calibration.
- Castling rights and en-passant squares cannot be inferred from a single
  screenshot; FENs emitted by the vision pipeline omit them (the engine handles
  the resulting positions correctly for analysis).
- Board coordinates assume a roughly square board; extreme perspective angles
  reduce detection confidence.
