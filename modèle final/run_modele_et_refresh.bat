@echo off
setlocal
cd /d "%~dp0"

echo [1/4] Nettoyage du check manuel...
"..\.venv\Scripts\python.exe" "nettoyer_check_manuel.py" --csv "check_manuel_results.csv"
if errorlevel 1 goto :error

echo [2/4] Lancement du modele final...
"..\.venv\Scripts\python.exe" "analyse_modele_final.py"
if errorlevel 1 goto :error

echo [3/4] Mise a jour de monitoring_officiel.html...
"..\.venv\Scripts\python.exe" "mettre_a_jour_monitoring_html.py"
if errorlevel 1 goto :error

echo [4/4] Ouverture du monitoring HTML...
start "" "monitoring_officiel.html"

echo Termine.
goto :eof

:error
echo Une erreur est survenue. Voir les messages ci-dessus.
pause
