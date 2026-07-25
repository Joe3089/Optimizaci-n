# -*- coding: utf-8 -*-
import sys
import os

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

from app.ui.splash_screen import SplashScreen
# InterfazOptimizacion NO se importa aquí arriba a propósito: ese import por
# sí solo arrastra sympy/matplotlib/numpy/reportlab/openpyxl/python-docx y
# tarda ~2-4s. Si estuviera aquí, bloquearía la app ANTES de poder mostrar
# el splash — el usuario vería una ventana en blanco/nada al hacer doble
# clic. Se importa de forma diferida dentro de iniciar_interfaz(), que se
# llama recién cuando el splash YA está pintado en pantalla (percepción de
# apertura inmediata mientras la carga pesada ocurre detrás).


def resource_path(relative_path: str) -> str:
    """Ruta absoluta al recurso (dev y PyInstaller)."""
    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base_path, relative_path)


def iniciar_interfaz():
    from app.ui.main_window import InterfazOptimizacion
    app.main_window = InterfazOptimizacion(resource_path=resource_path)
    app.main_window.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    icon_path = resource_path(os.path.join('Menu', 'WindowsIcon-min.ico'))
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    splash = SplashScreen(on_start_callback=iniciar_interfaz, resource_path=resource_path)
    splash.show()

    sys.exit(app.exec_())
