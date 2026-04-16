@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

title Parking Detection - Pipeline Complet

set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"

echo =================================================
echo  Parking Detection - Pipeline Complet
echo  1. Selection des sous-dossiers
echo  2. Analyse avec YOLOv8
echo  3. Mise a jour du monitoring
echo =================================================
echo.

"%PYTHON_EXE%" "%~dp0interface_selection_data.py"

if errorlevel 1 (
  echo.
  echo Une erreur est survenue dans le pipeline.
  pause
)

pause
