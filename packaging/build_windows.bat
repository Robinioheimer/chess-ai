@echo off
REM Build Chess AI Overlay into dist\ChessAI\ChessAI.exe on Windows.
setlocal
cd /d "%~dp0\.."

if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

pyinstaller packaging\chess_ai.spec --noconfirm --clean
echo.
echo Build finished. Executable: dist\ChessAI\ChessAI.exe
endlocal
