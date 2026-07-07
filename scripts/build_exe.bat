@echo off
cd /d "%~dp0.."
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

echo [1/3] Limpiando builds anteriores...
if exist "build" rmdir /s /q "build"
if exist "dist\OptimizadorFunciones.exe" del /q "dist\OptimizadorFunciones.exe"

echo [2/3] Compilando con PyInstaller...
echo.
python -m PyInstaller OptimizadorFunciones.spec --noconfirm

echo.
if exist "dist\OptimizadorFunciones.exe" (
    echo ============================================================
    echo   EXITO: dist\OptimizadorFunciones.exe generado
    echo ============================================================
    echo.
    echo [3/3] Copiando el EXE al Escritorio...
    copy /y "dist\OptimizadorFunciones.exe" "%USERPROFILE%\Desktop\OptimizadorFunciones.exe" >nul
    if errorlevel 1 (
        echo   No se pudo copiar al Escritorio automaticamente.
        echo   Clic derecho sobre dist\OptimizadorFunciones.exe
        echo   Enviar a - Escritorio (crear acceso directo)
    ) else (
        echo   Copiado a: %USERPROFILE%\Desktop\OptimizadorFunciones.exe
        echo   ^(el EXE es autonomo: puedes moverlo/copiarlo a donde quieras^)
    )
    echo.
) else (
    echo ============================================================
    echo   ERROR: No se genero el EXE. Revisa los mensajes arriba.
    echo ============================================================
)
pause
