@echo off
rem Opens the HMS window. Double-click this file.
cd /d "%~dp0"
set "PY=python"
set "PYW=pythonw"
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"
if exist .venv\Scripts\pythonw.exe set "PYW=.venv\Scripts\pythonw.exe"

rem Check that HMS and its libraries load before opening the window, because
rem pythonw shows no errors: without this, a missing install does nothing at all.
set "CHECK=import hms.gui, hms.cli, hms.structural, hms.preprocess, hms.extract, xgboost"
"%PY%" -c "%CHECK%" >nul 2>&1 || goto problem
start "" "%PYW%" -m hms gui
exit /b 0

:problem
echo HMS could not start. The error is below; README.md section 1 explains the setup.
echo.
"%PY%" -c "%CHECK%"
echo.
pause
exit /b 1
