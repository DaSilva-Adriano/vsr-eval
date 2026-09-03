@echo off
setlocal
title VSR-Eval
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo VSR-Eval could not find the virtualenv.
    echo Expected:
    echo   %PY%
    echo.
    echo Create it first. In PowerShell, from this folder:
    echo   uv venv --python 3.12 .venv
    echo   .\.venv\Scripts\Activate.ps1
    echo   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    echo   uv pip install -r requirements.txt
    echo   uv pip install -e .
    echo.
    pause
    exit /b 1
)

echo Starting VSR-Eval GUI...
echo Keep this window open while the UI is running. Close it or press Ctrl+C to stop.
echo.
"%PY%" -m vsr_eval --gui
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
    echo.
    echo VSR-Eval exited with code %ERR%.
    pause
)
exit /b %ERR%
