@echo off
setlocal
cd /d "%~dp0"

rem UTF-8 console so emoji/CJK in logs don't crash on cp1252
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo Starting RIFT-SVC GUI...

rem Prefer uv (project uses uv.lock); fall back to the local venv, then system python
where uv >nul 2>nul
if %errorlevel%==0 (
    uv run python gui_infer.py %*
    goto :end
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" gui_infer.py %*
    goto :end
)

python gui_infer.py %*

:end
if %errorlevel% neq 0 (
    echo.
    echo The GUI exited with an error ^(code %errorlevel%^).
)
pause
