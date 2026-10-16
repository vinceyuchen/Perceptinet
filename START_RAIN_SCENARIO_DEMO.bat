@echo off
setlocal
cd /d "%~dp0"
title Perceptinet Scenario 2 - Heavy Rain V2I Fault
echo ==========================================================
echo  PERCEPTINET SCENARIO 2 - HEAVY RAIN + INJECTED V2I FAULT
echo ==========================================================
echo Weather is a visual CARLA setting, V2I delay/loss is injected.
echo Close Dashboard or press Q in the camera / console to exit.
echo.
where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python 3.12 launcher not available.
  pause
  exit /b 1
)
py -3.12 "%~dp0Perceptinet_Teacher_Demo.py" --scenario rain_v2i
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" pause
exit /b %RESULT%
