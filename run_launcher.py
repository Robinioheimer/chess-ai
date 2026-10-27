"""PyInstaller entry point (kept at repository root for simple packaging)."""
from chess_ai.run import main

if __name__ == "__main__":
    raise SystemExit(main())
