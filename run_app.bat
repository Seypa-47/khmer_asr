@echo off
title Khmer ASR Interactive Testing Web App
echo ============================================================
echo Starting Khmer Speech Recognition Web App (Gradio)...
echo URL: http://127.0.0.1:7860
echo ============================================================
echo.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)
pause
