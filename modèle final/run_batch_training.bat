@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"

echo ================================================
echo  Batch Training YOLO
echo ================================================
echo.

REM Example end-to-end with 80/10/10 split:
REM   1) Optional offline augmentation
REM   2) Dataset preparation
REM   3) Transfer learning training

"%PYTHON_EXE%" "prepare_yolo_dataset.py" ^
  --images-dir "..\DATA\DATA_1" ^
  --labels-dir "..\DATA\LABELS_1" ^
  --output-dir "batch_dataset" ^
  --train-ratio 0.8 ^
  --val-ratio 0.1 ^
  --test-ratio 0.1 ^
  --class-names "car"
if errorlevel 1 goto :error

"%PYTHON_EXE%" "train_batch_yolo.py" ^
  --data "batch_dataset\dataset.yaml" ^
  --weights "yolov8n.pt" ^
  --epochs 100 ^
  --imgsz 640 ^
  --batch 16 ^
  --project "training_runs" ^
  --name "batch_yolo" ^
  --report-json "training_batch_last_report.json" ^
  --report-html "training_batch_last_report.html"
if errorlevel 1 goto :error

echo.
echo [OK] Batch training complete.
if exist "training_batch_last_report.html" start "" "training_batch_last_report.html"
pause
exit /b 0

:error
echo.
echo [ERREUR] Echec de la pipeline batch training.
pause
exit /b 1
