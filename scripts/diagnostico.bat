@echo off
cd /d "%~dp0.."
echo === Python activo === > resultado.txt
python --version >> resultado.txt 2>&1
where python >> resultado.txt 2>&1
echo. >> resultado.txt
echo === PyInstaller via python -m === >> resultado.txt
python -m PyInstaller --version >> resultado.txt 2>&1
echo. >> resultado.txt
echo === Qt instalado (python -m pip) === >> resultado.txt
python -m pip show PyQt6 >> resultado.txt 2>&1
python -m pip show PyQt5 >> resultado.txt 2>&1
echo. >> resultado.txt
echo === Todos los paquetes === >> resultado.txt
python -m pip list >> resultado.txt 2>&1
echo. >> resultado.txt
echo Listo. Abre resultado.txt
pause
