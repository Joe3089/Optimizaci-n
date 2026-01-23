# Documentación técnica — Proyecto de Optimización (PyQt5)

## 1. Descripción general
Aplicación de escritorio en Python orientada a la optimización de funciones. Permite:
- Ingresar funciones y parámetros desde una interfaz gráfica.
- Ejecutar métodos de búsqueda/optimización (Fibonacci, Búsqueda Local, Armijo, Wolfe).
- Visualizar resultados en tablas y gráficas 2D/3D.
- Exportar resultados a CSV y a Excel en formato reporte.
- Empaquetar la aplicación como ejecutable `.exe` (Windows) con PyInstaller.

## 2. Tecnologías utilizadas
### Lenguaje
- Python 3.x

### Interfaz de usuario (Desktop)
- PyQt5 (widgets, layouts, señales/slots)

### Gráficas
- Matplotlib
  - Embebido en PyQt5 con FigureCanvasQTAgg
  - 3D con mpl_toolkits.mplot3d (Axes3D)
  - Rotación automática 3D usando QTimer (ver rotacion_3d.py)

### Exportación
- CSV: módulo estándar csv
- Excel reporte: openpyxl
  - Hojas por método
  - Estilos de tabla
  - Inserción de imágenes (gráficas)

### Empaquetado / Distribución
- PyInstaller
  - `.exe` sin consola (`--noconsole`)
  - Inclusión de recursos (carpeta Menu/) con `--add-data`
  - Icono con `--icon` (.ico)

## 3. Estructura de carpetas (referencial)
Proyecto/
- main.py
- splash_screen.py
- interfaz_qt.py
- fibonacci.py
- busqueda_local.py
- armijo.py
- wolfe.py
- Menu/
  - fondo_optimizacion.png
  - WindowsIcon-min.ico
  - Fondo/
    - Imgen de fondo.jpg

## 4. Flujo de ejecución
1. main.py crea QApplication, asigna icono y muestra SplashScreen.
2. Al iniciar, se instancia InterfazOptimizacion (ventana principal).
3. El usuario elige método y parámetros, ejecuta cálculo.
4. Se muestran tablas y gráficas.
5. Exportación a CSV y Excel.

## 5. Rotación dinámica de la gráfica 3D
Ver rotacion_3d.py

## 6. Exportación a Excel tipo reporte
Ver reporte_export.py

## 7. Construcción del ejecutable (.exe)
Ejemplo (PowerShell):
python -m PyInstaller --clean --noconsole --onefile --name "OptimizadorFunciones" --add-data "Menu;Menu" --icon "Menu\WindowsIcon-min.ico" main.py

## 8. Requisitos de instalación (desarrollo)
pip install pyqt5 matplotlib openpyxl pyinstaller
