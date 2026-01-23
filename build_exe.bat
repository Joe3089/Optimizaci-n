@echo off
REM ==========================================
REM Construir .exe con PyInstaller (Windows)
REM ==========================================
REM 1) Instala PyInstaller:
REM    pip install pyinstaller
REM
REM 2) (Recomendado) Crea un icono .ico:
REM    Convierte Menu\icono_app.png a Menu\icono_app.ico (PNG -> ICO)
REM
REM 3) Ejecuta este .bat en la carpeta del proyecto (donde está main.py)
REM
pyinstaller --noconsole --onefile ^
  --name "OptimizadorFunciones" ^
  --add-data "Menu\\fondo_optimizacion.png;Menu" ^
  --add-data "Menu\\icono_app.png;Menu" ^
  --icon "Menu\\icono_app.ico" ^
  main.py

echo.
echo EXE generado en: dist\\OptimizadorFunciones.exe
pause
