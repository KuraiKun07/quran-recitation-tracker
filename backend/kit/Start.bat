@echo off
setlocal
title Quran Tracker
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\run.ps1"
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
  echo.
  echo Something went wrong. Press any key to close this window.
  pause >nul
)
endlocal & exit /b %RC%
