@echo off
cd /d "%~dp0.."
echo.
echo ============================================================
echo   Optimizador de Funciones - Generando INSTALADOR...
echo ============================================================
echo.

set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
    echo [ERROR] No se encontro Inno Setup 6.
    echo         Instalalo con:  winget install JRSoftware.InnoSetup
    pause
    exit /b 1
)

echo [1/2] Compilando el EXE con PyInstaller...
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -m PyInstaller OptimizadorFunciones.spec --noconfirm
) else (
    python -m PyInstaller OptimizadorFunciones.spec --noconfirm
)
if not exist "dist\OptimizadorFunciones.exe" (
    echo [ERROR] No se genero dist\OptimizadorFunciones.exe
    pause
    exit /b 1
)

echo.
echo [2/2] Empaquetando el instalador con Inno Setup...
"%ISCC%" "installer\OptimizadorFunciones.iss"
if errorlevel 1 (
    echo [ERROR] Fallo la compilacion del instalador.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   EXITO: instalador en installer\Output\
echo   Copia ese unico archivo a la otra computadora y ejecutalo.
echo ============================================================
explorer "installer\Output"
pause
