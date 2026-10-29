@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Erstelle Umgebung und installiere Abhaengigkeiten...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
".venv\Scripts\python.exe" -m chess_ai
pause