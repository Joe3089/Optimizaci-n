@echo off
echo.
echo ============================================================
echo   Optimizador de Funciones - Generando EXE...
echo ============================================================
echo.

if not exist "main.py" (
    echo [ERROR] No se encontro main.py
    echo         Ejecuta desde la carpeta raiz del proyecto.
    pause
    exit /b 1
)

if not exist "Menu\WindowsIcon-min.ico" (
    echo [ERROR] Falta Menu\WindowsIcon-min.ico
    echo  Convierte el PNG en: https://convertio.co/es/png-ico/
    pause
    exit /b 1
)

echo [1/2] Limpiando builds anteriores...
if exist "build" rmdir /s /q "build"
if exist "dist\OptimizadorFunciones.exe" del /q "dist\OptimizadorFunciones.exe"

echo [2/2] Compilando con PyInstaller...
echo.
python -m PyInstaller OptimizadorFunciones.spec --noconfirm

echo.
if exist "dist\OptimizadorFunciones.exe" (
    echo ============================================================
    echo   EXITO: dist\OptimizadorFunciones.exe generado
    echo ============================================================
    echo.
    echo   Clic derecho sobre dist\OptimizadorFunciones.exe
    echo   Enviar a - Escritorio (crear acceso directo)
    echo.
) else (
    echo ============================================================
    echo   ERROR: No se genero el EXE. Revisa los mensajes arriba.
    echo ============================================================
)
pause
