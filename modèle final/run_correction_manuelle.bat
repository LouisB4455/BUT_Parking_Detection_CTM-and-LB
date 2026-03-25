@echo off
setlocal
cd /d "%~dp0"

echo Lancement de la correction manuelle...
set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
"%PYTHON_EXE%" "%~dp0lancer_correction_manuelle.py"
if errorlevel 1 (
  echo Une erreur est survenue lors du lancement de la correction manuelle.
  pause
)
