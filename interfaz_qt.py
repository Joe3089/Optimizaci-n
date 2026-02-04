# interfaz_qt.py (FIXIMPORTS + DIAGNOSTICO) compatible con resource_path
import os, sys, traceback, importlib

# Asegura que la carpeta del archivo (interfaz_qt.py) esté en sys.path
try:
    from rotacion_3d import Rotating3DCanvas
except Exception:  # pragma: no cover
    Rotating3DCanvas = None

try:
    from canvas_2d import Function2DCanvas
except Exception:  # pragma: no cover
    Function2DCanvas = None

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QLabel
from PyQt5.QtCore import Qt

def _import_module(name):
    try:
        return importlib.import_module(name), None
    except Exception:
        return None, traceback.format_exc()

def _try_import_canvases():
    mod2d, err2d = _import_module("canvas_2d_estilo_fixed")
    mod3d, err3d = _import_module("rotacion_3d_superficie")

    Function2DCanvas = getattr(mod2d, "Function2DCanvas", None) if mod2d else None
    Rotating3DCanvas = getattr(mod3d, "Rotating3DCanvas", None) if mod3d else None

    diag = []
    diag.append(f"Working dir: {os.getcwd()}")
    diag.append(f"File dir: {_HERE}")
    diag.append("sys.path[0..3]: " + repr(sys.path[:4]))

    if Function2DCanvas is None:
        diag.append("\n❌ No se pudo cargar Function2DCanvas desde canvas_2d_estilo_fixed.py")
        if err2d:
            diag.append("Traceback import 2D:\n" + err2d)
        else:
            diag.append("El módulo 2D cargó, pero NO existe la clase Function2DCanvas (revisa el nombre).")

    if Rotating3DCanvas is None:
        diag.append("\n❌ No se pudo cargar Rotating3DCanvas desde rotacion_3d_superficie.py")
        if err3d:
            diag.append("Traceback import 3D:\n" + err3d)
        else:
            diag.append("El módulo 3D cargó, pero NO existe la clase Rotating3DCanvas (revisa el nombre).")

    return Function2DCanvas, Rotating3DCanvas, "\n".join(diag)


class InterfazOptimizacion(QMainWindow):
    def __init__(self, resource_path=None, **kwargs):
        super().__init__()
        self.resource_path = resource_path

        self.setWindowTitle("Optimizador de Funciones")
        self.resize(1200, 700)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.left_panel = QWidget()
        self.right_panel = QWidget()
        root.addWidget(self.left_panel)
        root.addWidget(self.right_panel, 1)

        right_layout = QVBoxLayout(self.right_panel)
        splitter = QSplitter(Qt.Vertical)
        right_layout.addWidget(splitter)

        Function2DCanvas, Rotating3DCanvas, diag = _try_import_canvases()

        if Function2DCanvas is None:
            self.canvas_2d = QLabel("ERROR IMPORT 2D\n\n" + diag)
            self.canvas_2d.setWordWrap(True)
            self.canvas_2d.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        else:
            self.canvas_2d = Function2DCanvas()
            try:
                self.canvas_2d.set_point_changed_callback(self._on_canvas_point_changed)
            except Exception:
                pass

        if Rotating3DCanvas is None:
            self.canvas_3d = QLabel("ERROR IMPORT 3D\n\n" + diag)
            self.canvas_3d.setWordWrap(True)
            self.canvas_3d.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        else:
            self.canvas_3d = Rotating3DCanvas()

        splitter.addWidget(self.canvas_2d)
        splitter.addWidget(self.canvas_3d)
        splitter.setSizes([320, 420])

    def _on_canvas_point_changed(self, x, y):
        print(f"Punto seleccionado: ({x:.6f}, {y:.6f})")
