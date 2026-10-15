@echo off
setlocal
cd /d "%~dp0"
title Perceptinet Teacher Demo Launcher
echo =================================================
echo     PERCEPTINET LIVE LOOPING TEACHER DEMO
echo =================================================
echo.
echo Starting real CARLA + onboard / RSU + Dashboard.
echo Press Q in the camera window or launcher console to exit.
echo.
where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python launcher 'py' was not found.
  echo Install Python 3.12 or run Perceptinet_Teacher_Demo.py using
  echo the Python environment used for your original CARLA script.
  pause
  exit /b 1
)
py -3.12 "%~dp0Perceptinet_Teacher_Demo.py"
set "DEMO_EXIT_CODE=%ERRORLEVEL%"
if not "%DEMO_EXIT_CODE%"=="0" (
  echo.
  echo Demo returned exit code %DEMO_EXIT_CODE%.
  echo The details should be visible in this window.
  pause
)
exit /b %DEMO_EXIT_CODE%
