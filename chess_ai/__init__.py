"""Chess AI Overlay.

Automatic screen chessboard detection, Stockfish analysis and a transparent
on-screen overlay.

The package is organised into decoupled modules:

* :mod:`chess_ai.core`     - FEN, notation, probabilities, position tracking
* :mod:`chess_ai.vision`   - screen capture, board detection, piece recognition
* :mod:`chess_ai.engine`   - UCI/Stockfish process management and analysis
* :mod:`chess_ai.overlay`  - PyQt5 transparent desktop overlay
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
