@echo off
title Khmer ASR - WSL GPU Runtime
cd /d "%~dp0"
echo Starting Khmer ASR at http://127.0.0.1:7860 using Ubuntu WSL...
set "KHMER_WINDOWS_PROFILE=%USERPROFILE%"
if defined WSLENV (set "WSLENV=KHMER_WINDOWS_PROFILE/p:%WSLENV%") else set "WSLENV=KHMER_WINDOWS_PROFILE/p"
wsl -d Ubuntu -- bash src/run_app_wsl.sh
pause
