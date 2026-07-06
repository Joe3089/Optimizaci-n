# -*- coding: utf-8 -*-
import sys
import os

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

from splash_screen import SplashScreen
from interfaz_qt import InterfazOptimizacion


def resource_path(relative_path: str) -> str:
    """Ruta absoluta al recurso (dev y PyInstaller)."""
    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base_path, relative_path)


def iniciar_interfaz():
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
