@echo off
setlocal
cd /d "%~dp0"
title Perceptinet Comparative Teacher Demo
echo =============================================================
echo  PERCEPTINET: CLEAR SUCCESS VS HEAVY-RAIN V2I FAULT INJECTION
echo =============================================================
echo.
echo This run alternates clear baseline and heavy-rain degraded V2I.
echo Rain changes visuals. Communication faults are separately injected.
echo Outcomes are measured, not predetermined physical collisions.
echo.
where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python launcher 'py' was not found.
  pause
  exit /b 1
)
py -3.12 "%~dp0Perceptinet_Teacher_Demo.py" --scenario alternate
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Comparative demo failed with code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
