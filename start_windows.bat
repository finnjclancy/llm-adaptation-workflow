@echo off
REM ====================================================================
REM  LLM Adaptation Workflow - Windows launcher
REM  Double-click this file to start the app. No terminal needed.
REM ====================================================================

cd /d "%~dp0"
title LLM Adaptation Workflow

REM Find a Python launcher: prefer "py", fall back to "python".
where py >nul 2>nul
if %errorlevel%==0 (
    py run_app.py
    goto :end
)

where python >nul 2>nul
if %errorlevel%==0 (
    python run_app.py
    goto :end
)

echo.
echo  Python was not found on this computer.
echo.
echo  Please install Python 3.10+ from:
echo      https://www.python.org/downloads/windows/
echo.
echo  IMPORTANT: on the first install screen, tick
echo  "Add Python to PATH", then run this file again.
echo.
pause

:end
