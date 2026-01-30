# -*- coding: utf-8 -*-
import os
import sys
import math
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional

import numpy as np
import csv

# Métodos (con compatibilidad: algunos archivos definen wrappers metodo_* )
try:
    from busqueda_local import busqueda_local
except Exception:  # pragma: no cover
    busqueda_local = None

try:
    from fibonacci import metodo_fibonacci, fibonacci_search
except Exception:  # pragma: no cover
    metodo_fibonacci = None
    fibonacci_search = None

try:
    from armijo import metodo_armijo, armijo_search
except Exception:  # pragma: no cover
    metodo_armijo = None
    armijo_search = None

try:
    from wolfe import metodo_wolfe, wolfe_search
except Exception:  # pragma: no cover
    metodo_wolfe = None
    wolfe_search = None

# --- Métodos Multidimensionales (sin pip: carga por ruta) ---
md_penalty_newton = None
md_barrier_newton = None
md_penalty_bfgs = None
md_penalty_nelder = None
_mdw = None

def _load_md_wrappers_by_path():
    """Carga metodos_multidimensional/wrappers.py por ruta (sin depender de package imports)."""
    global _mdw, md_penalty_newton, md_barrier_newton, md_penalty_bfgs, md_penalty_nelder
    if _mdw is not None:
        return _mdw

    import os
    import importlib.util

    base_dir = os.path.dirname(os.path.abspath(__file__))
    md_dir = os.path.join(base_dir, "metodos_multidimensional")
    md_path = os.path.join(md_dir, "wrappers.py")

    if not os.path.exists(md_path):
        raise ImportError(f"No se encontró wrappers.py en: {md_path}")

    spec = importlib.util.spec_from_file_location("md_wrappers_runtime", md_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo crear spec para: {md_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    _mdw = mod
    md_penalty_newton = getattr(mod, "penalty_method_newton", None)
    md_barrier_newton = getattr(mod, "barrier_method_newton", None)
    md_penalty_bfgs = getattr(mod, "penalty_method_bfgs", None)
    md_penalty_nelder = getattr(mod, "penalty_method_nelder", None)
    return mod
try:
    from rotacion_3d import Rotating3DCanvas
except Exception:  # pragma: no cover
    Rotating3DCanvas = None

try:
    from reporte_export import ReportItem, exportar_reporte_excel, exportar_reporte_pdf
except Exception:  # pragma: no cover
    ReportItem = None
    exportar_reporte_excel = None
    exportar_reporte_pdf = None


from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QColor, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QLineEdit,
    QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QMessageBox,
    QFileDialog, QTableWidget, QTableWidgetItem, QSizePolicy, QSpacerItem, QMenu, QToolButton, QAction
)




SAFE_MATH = {
    # constantes
    "pi": math.pi,
    "e": math.e,
    # funciones básicas (numpy)
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "abs": np.abs, "power": np.power
}


def resource_path(relative_path: str) -> str:
    """
    Devuelve ruta absoluta para recursos (compatible con PyInstaller).
    """
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class InterfazOptimizacion(QMainWindow):
    """
    UI principal: panel de control (izq) + dashboard (der).
    - Mínimo/Máximo son los límites del intervalo [a,b]
    - Parámetros internos (x0, paso, max_iter) se calculan automáticamente.
    - Tolerancia SOLO para Fibonacci.
    - Exportar Excel: un solo archivo por función, con pestañas por método ejecutado,
      incluyendo tabla + imagen de la gráfica.
    """

    def __init__(self, resource_path_base: str = ".", resource_path: str = None):
        super().__init__()
        # Compatibilidad con código que usa resource_path
        if resource_path is not None:
            resource_path_base = resource_path

        
        # estado: expr -> method -> data
        self._results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._current_expr: str = ""

        self._canvas: Optional[Rotating3DCanvas] = None
        self._build_ui()
        self._apply_styles()
        self._apply_background()

        self.on_metodo_change()  # set visibilities

    # ---------------- UI ----------------
    def _build_ui(self):
        self.setWindowTitle("Optimizador de Funciones")
        ico = self._find_icon()
        if ico:
            self.setWindowIcon(QIcon(ico))

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        # Left panel
        self.left_panel = QFrame()
        self.left_panel.setObjectName("leftPanel")
        self.left_panel.setFixedWidth(360)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Panel de control")
        title.setObjectName("panelTitle")
        left_layout.addWidget(title)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        # Función
        form.addWidget(QLabel("Función f(x):"), 0, 0)
        self.func_input = QLineEdit()
        self.func_input.setPlaceholderText("Ej: -(x-3)**2 + 10")
        form.addWidget(self.func_input, 0, 1)

        # --- Campos para Métodos Multidimensionales (MD) ---
        form.addWidget(QLabel("Restricción g(x):"), 0, 2)
        self.constraint_input = QLineEdit()
        self.constraint_input.setPlaceholderText("Ej: x1**2 + x2**2 - 1")
        self.constraint_input.setFixedWidth(90)
        form.addWidget(self.constraint_input, 0, 3)

        form.addWidget(QLabel("Variables:"), 1, 2)
        self.vars_input = QLineEdit()
        self.vars_input.setPlaceholderText("Ej: x1 x2")
        self.vars_input.setFixedWidth(90)
        form.addWidget(self.vars_input, 1, 3)

        form.addWidget(QLabel("x0 (coma):"), 2, 2)
        self.x0_input = QLineEdit()
        self.x0_input.setPlaceholderText("Ej: 0.5,0.5")
        self.x0_input.setFixedWidth(90)
        form.addWidget(self.x0_input, 2, 3)

        # ocultos hasta seleccionar un método MD
        self.constraint_input.setVisible(False)
        self.vars_input.setVisible(False)
        self.x0_input.setVisible(False)

        # Método
        form.addWidget(QLabel("Método:"), 1, 0)
        self.metodo_menu = QComboBox()
        self.metodo_menu.addItems([
            "Búsqueda Local", "Fibonacci", "Armijo", "Wolfe",
            "MD: Penalización (Newton)",
            "MD: Barrera (Newton)",
            "MD: Penalización (BFGS)",
            "MD: Penalización (Nelder-Mead)"
        ])
        self.metodo_menu.currentIndexChanged.connect(self.on_metodo_change)
        form.addWidget(self.metodo_menu, 1, 1)

        # Mínimo / Máximo
        form.addWidget(QLabel("mínimo:"), 2, 0)
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setDecimals(6)
        self.min_spin.setValue(0.0)
        form.addWidget(self.min_spin, 2, 1)

        form.addWidget(QLabel("máximo:"), 3, 0)
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setDecimals(6)
        self.max_spin.setValue(5.0)
        form.addWidget(self.max_spin, 3, 1)

        # Tolerancia (solo Fibonacci)
        self.tol_label = QLabel("tolerancia:")
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setDecimals(8)
        self.tol_spin.setRange(1e-12, 1e6)
        self.tol_spin.setValue(1e-3)
        self.tol_spin.setSingleStep(1e-3)
        form.addWidget(self.tol_label, 4, 0)
        form.addWidget(self.tol_spin, 4, 1)

        left_layout.addLayout(form)

        # Buttons
        btn_row1 = QHBoxLayout()
        self.btn_calc = QPushButton("Calcular")
        self.btn_calc.setFixedHeight(44)
        self.btn_calc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_calc.clicked.connect(self.ejecutar_metodo)
        self.btn_clear = QPushButton("Limpiar")
        self.btn_clear.setFixedHeight(44)
        self.btn_clear.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_clear.clicked.connect(self.limpiar)
        btn_row1.addWidget(self.btn_calc)
        btn_row1.addWidget(self.btn_clear)
        btn_row1.setStretch(0, 1)
        btn_row1.setStretch(1, 1)
        left_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()

        # Exportar (botón con menú desplegable)
        self.export_btn = QToolButton()
        self.export_btn.setObjectName("exportButton")
        self.export_btn.setText("Exportar")
        self.export_btn.setFixedHeight(44)
        self.export_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.export_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.export_btn.setPopupMode(QToolButton.InstantPopup)

        export_menu = QMenu(self.export_btn)
        export_menu.addAction("Exportar CSV", self.exportar_csv)
        export_menu.addAction("Exportar Excel", self.exportar_excel)
        export_menu.addAction("Exportar PDF", self.exportar_pdf)
        self.export_btn.setMenu(export_menu)

        # Salir (mismo tamaño que Calcular)
        self.btn_exit = QPushButton("Salir")
        self.btn_exit.setFixedHeight(44)
        self.btn_exit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_exit.clicked.connect(self._confirm_exit)

        btn_row2.addWidget(self.export_btn)
        btn_row2.addWidget(self.btn_exit)
        btn_row2.setStretch(0, 1)
        btn_row2.setStretch(1, 1)
        left_layout.addLayout(btn_row2)

        # status line
        self.status_lbl = QLabel("")
        self.status_lbl.setObjectName("statusLabel")
        self.status_lbl.setWordWrap(True)
        left_layout.addWidget(self.status_lbl)
        left_layout.addItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Right dashboard
        self.right_panel = QFrame()
        self.right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        dash_title = QLabel("Dashboard")
        dash_title.setObjectName("dashTitle")
        right_layout.addWidget(dash_title)

        self.table = QTableWidget()
        self.table.setObjectName("resultTable")
        self.table.setAlternatingRowColors(False)
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        # Mejoras UI: ocultar numeración de filas (recuadro rojo) y hacer cabeceras legibles
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        right_layout.addWidget(self.table, 2)

        self.plot_frame = QFrame()
        self.plot_frame.setObjectName("plotFrame")
        self.plot_layout = QVBoxLayout(self.plot_frame)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(0)

        self.plot_placeholder = QLabel("Ejecute un método para ver la gráfica.")
        self.plot_placeholder.setAlignment(Qt.AlignCenter)
        self.plot_placeholder.setObjectName("plotPlaceholder")
        self.plot_layout.addWidget(self.plot_placeholder)

        right_layout.addWidget(self.plot_frame, 3)

        root.addWidget(self.left_panel)
        root.addWidget(self.right_panel, 1)

        self._install_spanish_context_menus()
        self.resize(1200, 700)

    def _apply_styles(self):
        # estilos: paneles oscuros con texto blanco; menús contextuales (click derecho) blancos
        self.setStyleSheet("""
        QMainWindow { background: transparent; }
        QFrame#leftPanel, QFrame#rightPanel {
            background-color: rgba(2, 6, 23, 0.78);
            border-radius: 18px;
        }
        QLabel { color: white; font-size: 12px; }
        QLabel#panelTitle { font-size: 20px; font-weight: 800; }
        QLabel#dashTitle { font-size: 20px; font-weight: 800; }
        QLabel#statusLabel { color: #d6e0ff; font-size: 12px; }
        QLineEdit, QComboBox, QDoubleSpinBox {
            background-color: rgba(255,255,255,0.10);
            color: white;
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 10px;
            padding: 6px 10px;
            min-height: 30px;
        }
        QComboBox QAbstractItemView {
            background-color: rgba(9, 12, 20, 0.98);
            color: white;
            selection-background-color: rgba(0, 140, 255, 0.35);
            border: 1px solid rgba(255,255,255,0.15);
        }
        QPushButton {
            background-color: rgba(0, 140, 255, 0.55);
            color: white;
            border: 0px;
            border-radius: 14px;
            padding: 10px 14px;
            font-weight: 700;
        }
        QPushButton:hover { background-color: rgba(0, 140, 255, 0.70); }
        QPushButton:pressed { background-color: rgba(0, 140, 255, 0.85); }


        QToolButton#exportButton {
            background-color: rgba(0, 140, 255, 0.55);
            color: white;
            border: 0px;
            border-radius: 14px;
            padding: 10px 14px;
            font-weight: 700;
        }
        QToolButton#exportButton:hover { background-color: rgba(0, 140, 255, 0.70); }
        QToolButton#exportButton:pressed { background-color: rgba(0, 140, 255, 0.85); }

        QComboBox#exportCombo {
            background-color: rgba(0, 140, 255, 0.55);
            color: white;
            border: 0px;
            border-radius: 14px;
            padding: 10px 14px;
            font-weight: 700;
            min-height: 30px;
        }
        QComboBox#exportCombo:hover { background-color: rgba(0, 140, 255, 0.70); }
        QComboBox#exportCombo::drop-down { border: 0px; width: 30px; }
        QComboBox#exportCombo::down-arrow { /* default arrow */ }


        QTableWidget#resultTable {
            alternate-background-color: rgba(0,0,0,0.45);
            background-color: rgba(0,0,0,0.55);
            color: white;
            gridline-color: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 12px;
        }
        QHeaderView::section {
            background-color: rgba(0,0,0,0.55);
            color: white;
            border: 0px;
            padding: 6px;
            font-weight: 700;
        }
        QTableCornerButton::section {
            background-color: rgba(0,0,0,0.55);
            border: 0px;
        }
        QTableWidget::item { padding: 4px; background-color: rgba(0,0,0,0.35); color: white; }
        QTableWidget::item:alternate { background-color: rgba(0,0,0,0.35); }
        QTableWidget::item:disabled { color: rgba(255,255,255,0.75); }
        QTableWidget::item:selected { background-color: rgba(0, 140, 255, 0.30); }

        QLabel#plotPlaceholder { color: rgba(255,255,255,0.85); }

        /* Menú contextual (click derecho) */
        QMenu {
            background-color: rgba(9, 12, 20, 0.98);
            color: white;
            border: 1px solid rgba(255,255,255,0.15);
        }
        QMenu::item:selected { background-color: rgba(0, 140, 255, 0.35); }

        """)

    def _apply_background(self):
        # Fondo: Menu/Fondo/Imgen de fondo.jpg
        bg_path = resource_path(os.path.join("Menu", "Fondo", "Imgen de fondo.jpg"))
        if not os.path.exists(bg_path):
            bg_path = resource_path(os.path.join("Menu", "Fondo", "Imagen de fondo.jpg"))
        if not os.path.exists(bg_path):
            return

        # usar un label como wallpaper
        self._bg_label = QLabel(self)
        self._bg_label.setObjectName("bgLabel")
        self._bg_label.setScaledContents(True)
        pix = QPixmap(bg_path)
        self._bg_label.setPixmap(pix)
        self._bg_label.lower()
        self._bg_label.setGeometry(self.rect())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_bg_label"):
            self._bg_label.setGeometry(self.rect())

    def _find_icon(self) -> Optional[str]:
        for rel in [
            os.path.join("Menu", "WindowsIcon-min.ico"),
            os.path.join("Menu", "WindowsIcon.ico"),
            os.path.join("Menu", "WindowsIcon-min.png"),
        ]:
            p = resource_path(rel)
            if os.path.exists(p):
                return p
        return None

    # ---------------- Behaviors ----------------
    def on_metodo_change(self):
        metodo = self.metodo_menu.currentText()
        is_fibo = (metodo == "Fibonacci")
        is_md = metodo.startswith("MD:")
        self.tol_label.setVisible(is_fibo)
        self.tol_spin.setVisible(is_fibo)

        if hasattr(self, "constraint_input"):
            self.constraint_input.setVisible(is_md)
            self.vars_input.setVisible(is_md)
            self.x0_input.setVisible(is_md)

        self.min_spin.setEnabled(not is_md)
        self.max_spin.setEnabled(not is_md)

    def limpiar(self):
        self.status_lbl.setText("")
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)

        # limpiar gráfica
        if self._canvas is not None:
            self.plot_layout.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas = None
        self.plot_placeholder.show()

    # ---------------- Core logic ----------------
    def _parse_function(self, expr: str) -> Callable[[float], float]:
        expr = expr.strip()
        if not expr:
            raise ValueError("Debe ingresar una función.")

        def f(x):
            return eval(expr, {"__builtins__": {}}, {"x": x, "np": np, **SAFE_MATH})

        # prueba rápida
        _ = f(0.0)
        return f

    def _parse_md_x0(self, s: str) -> List[float]:
        s = (s or "").strip()
        if not s:
            raise ValueError("Para métodos MD, debe ingresar x0 (ej: 0.5, 0.5).")
        try:
            return [float(x.strip()) for x in s.split(",") if x.strip()]
        except Exception:
            raise ValueError("x0 inválido. Use valores separados por coma. Ej: 0.5, 0.5")

    def _parse_md_vars(self, s: str) -> str:
        s = (s or "").strip()
        if not s:
            raise ValueError("Para métodos MD, debe ingresar las variables (ej: x1 x2).")
        return s

    def _md_diag(self) -> str:
        try:
            mod = _load_md_wrappers_by_path()
            return f"wrappers: {getattr(mod, '__file__', 'desconocido')}"
        except Exception as e:
            return "wrappers: NO CARGADO (revisa metodos_multidimensional/wrappers.py)\n" + f"detalle: {e}"

    def _parse_md_constraint(self, s: str) -> str:
        s = (s or "").strip()
        if not s:
            raise ValueError("Para métodos MD, debe ingresar la restricción g(x)<=0.")
        return s

    def _auto_params(self, a: float, b: float) -> Dict[str, float]:
        if a == b:
            b = a + 1.0
        if a > b:
            a, b = b, a
        x0 = (a + b) / 2.0
        span = max(abs(b - a), 1e-6)
        paso = span / 50.0
        max_iter = 200
        return {"a": a, "b": b, "x0": x0, "paso": paso, "max_iter": max_iter}

    def ejecutar_metodo(self):
        try:
            expr = self.func_input.text().strip()
            metodo = self.metodo_menu.currentText()

            if metodo.startswith("MD:"):
                func_str = expr
                constraint_str = self._parse_md_constraint(self.constraint_input.text())
                var_str = self._parse_md_vars(self.vars_input.text())
                x0_md = self._parse_md_x0(self.x0_input.text())

                # Carga por ruta (evita problemas de package/__init__.py y no requiere pip)
                try:
                    _load_md_wrappers_by_path()
                except Exception:
                    pass

                if metodo == "MD: Penalización (Newton)":
                    if md_penalty_newton is None:
                        raise ImportError("No se pudo importar penalty_method_newton\n" + self._md_diag())
                    out = md_penalty_newton(func_str, constraint_str, var_str, x0_md)
                elif metodo == "MD: Barrera (Newton)":
                    if md_barrier_newton is None:
                        raise ImportError("No se pudo importar barrier_method_newton\n" + self._md_diag())
                    out = md_barrier_newton(func_str, constraint_str, var_str, x0_md)
                elif metodo == "MD: Penalización (BFGS)":
                    if md_penalty_bfgs is None:
                        raise ImportError("No se pudo importar penalty_method_bfgs\n" + self._md_diag())
                    out = md_penalty_bfgs(func_str, constraint_str, var_str, x0_md)
                elif metodo == "MD: Penalización (Nelder-Mead)":
                    if md_penalty_nelder is None:
                        raise ImportError("No se pudo importar penalty_method_nelder\n" + self._md_diag())
                    out = md_penalty_nelder(func_str, constraint_str, var_str, x0_md)
                else:
                    raise ValueError("Método no soportado.")

                history = []
                if isinstance(out, (tuple, list)) and len(out) >= 3 and isinstance(out[2], list):
                    history = out[2]
                self._current_expr = expr
                self._update_table(self._coerce_history(history))
                try:
                    self.status_lbl.setText(f"✓ {metodo} → x*={out[0]}, f(x*)={out[1]}")
                except Exception:
                    self.status_lbl.setText(f"✓ {metodo} ejecutado.")

                # MD: no graficar en 1D
                if self._canvas is not None:
                    self.plot_layout.removeWidget(self._canvas)
                    self._canvas.setParent(None)
                    self._canvas = None
                self.plot_placeholder.setText("Método multidimensional: gráfico 1D no aplica.")
                self.plot_placeholder.show()
                return

            f = self._parse_function(expr)

            a_in = float(self.min_spin.value())
            b_in = float(self.max_spin.value())
            params = self._auto_params(a_in, b_in)
            a, b, x0, paso, max_iter = params["a"], params["b"], params["x0"], params["paso"], params["max_iter"]

            metodo = self.metodo_menu.currentText()

            # Ejecutar método con compatibilidad
            if metodo == "Búsqueda Local":
                if busqueda_local is None:
                    raise ImportError("No se pudo importar busqueda_local()")
                out = busqueda_local(f, x0=x0, paso=paso, max_iter=max_iter, tolerancia=1e-5, return_history=True)

            elif metodo == "Fibonacci":
                tol = float(self.tol_spin.value())
                fn = metodo_fibonacci if metodo_fibonacci is not None else fibonacci_search
                if fn is None:
                    raise ImportError("No se pudo importar metodo_fibonacci()/fibonacci_search()")
                out = fn(f, a=a, b=b, tolerancia=tol, max_n=max_iter, return_history=True)

            elif metodo == "Armijo":
                if metodo_armijo is not None:
                    out = metodo_armijo(f, a=a, b=b, max_iter=max_iter, return_history=True)
                elif armijo_search is not None:
                    x_new, f_new, iters, dist, hist = armijo_search(f, x0=x0, alpha0=1.0, rho=0.5, c=1e-4, max_iter=max_iter, return_history=True)
                    out = (x_new, f_new, iters, dist, hist)
                else:
                    raise ImportError("No se pudo importar metodo_armijo()/armijo_search()")

            elif metodo == "Wolfe":
                if metodo_wolfe is not None:
                    out = metodo_wolfe(f, a=a, b=b, max_iter=max_iter, return_history=True)
                elif wolfe_search is not None:
                    x_new, f_new, iters, dist, hist = wolfe_search(f, x0=x0, alpha0=1.0, rho=0.5, c1=1e-4, c2=0.9, max_iter=max_iter, return_history=True)
                    out = (x_new, f_new, iters, dist, hist)
                else:
                    raise ImportError("No se pudo importar metodo_wolfe()/wolfe_search()")
            else:
                raise ValueError("Método no soportado.")

            parsed = self._parse_method_output(metodo, out)
            resumen = parsed["resumen"]
            history = parsed["history"]

            # actualizar tabla + gráfica
            self._current_expr = expr
            self._update_table(history)
            self._render_plot(metodo, f, history)

            # Guardar para exportación
            img_path = self._save_plot_image(metodo, f, history)
            if ReportItem is not None:
                item = ReportItem(funcion=expr, metodo=metodo, iteraciones=history, resumen=resumen, grafica_path=img_path)
            else:
                item = None

            self._results.setdefault(expr, {})[metodo] = {"resumen": resumen, "history": history, "item": item}

            # status
            if "x_opt" in resumen and "f_opt" in resumen:
                try:
                    self.status_lbl.setText(f"✓ {metodo} → x*={float(resumen['x_opt']):.6g}, f(x*)={float(resumen['f_opt']):.6g}")
                except Exception:
                    self.status_lbl.setText(f"✓ {metodo} ejecutado.")
            else:
                self.status_lbl.setText(f"✓ {metodo} ejecutado.")

        except Exception as e:
            self._show_msg("Error", f"Ocurrió un error al ejecutar el método:\n{e}", icon=QMessageBox.Critical)

    def _update_table(self, history: List[Dict[str, Any]]):
        self.table.clear()
        if not history:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        columns = list(history[0].keys())
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(history))

        for r, row in enumerate(history):
            for c, col in enumerate(columns):
                val = row.get(col, "")
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                item.setForeground(QBrush(QColor(255, 255, 255)))
                item.setBackground(QBrush(QColor(0, 0, 0, 115)))
                self.table.setItem(r, c, item)

        self.table.resizeColumnsToContents()

    def _history_to_xyz(self, f: Callable[[float], float], history: List[Dict[str, Any]]):
        iters, xs, zs = [], [], []
        for i, row in enumerate(history, start=1):
            it = row.get("iter", i)
            x = row.get("x", None)
            if x is None:
                if "x1" in row and "x2" in row:
                    x = (row["x1"] + row["x2"]) / 2.0
                elif "a" in row and "b" in row:
                    x = (row["a"] + row["b"]) / 2.0
            if x is None:
                continue
            try:
                z = row.get("f(x)", row.get("fx", None))
                if z is None:
                    z = float(f(float(x)))
            except Exception:
                z = np.nan
            iters.append(float(it))
            xs.append(float(x))
            zs.append(float(z))
        return np.array(iters), np.array(xs), np.array(zs)

    def _render_plot(self, metodo: str, f: Callable[[float], float], history: List[Dict[str, Any]]):
        iters, xs, zs = self._history_to_xyz(f, history)
        if iters.size == 0:
            return

        self.plot_placeholder.hide()

        # Re-crear el canvas por ejecución para que SIEMPRE coincida con el método ejecutado
        if self._canvas is not None:
            self.plot_layout.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas = None

        titulo = f"Gráfica 3D - {metodo}"
        self._canvas = Rotating3DCanvas(self.plot_frame, title=titulo)
        self.plot_layout.addWidget(self._canvas)

        self._canvas.set_data(iters, xs, zs)
        self._canvas.start_rotation()

    def _save_plot_image(self, metodo: str, f: Callable[[float], float], history: List[Dict[str, Any]]) -> Optional[str]:
        # genera PNG 3D para Excel
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa

            iters, xs, zs = self._history_to_xyz(f, history)
            if iters.size == 0:
                return None

            fig = plt.figure(figsize=(7.2, 4.8), dpi=150)
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(iters, xs, zs, linewidth=2)
            ax.scatter([iters[-1]], [xs[-1]], [zs[-1]], s=35)
            ax.set_title(f"{metodo} - f(x)")
            ax.set_xlabel("Iteración")
            ax.set_ylabel("x")
            ax.set_zlabel("f(x)")
            fig.tight_layout()

            fd, path = tempfile.mkstemp(prefix="plot_", suffix=".png")
            os.close(fd)
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            return path
        except Exception:
            return None

    # ---------------- Export ----------------

    def _confirm_exit(self):
        m = QMessageBox(self)
        m.setWindowTitle("Salir")
        m.setIcon(QMessageBox.Question)
        m.setText("¿Está seguro de que desea salir de la aplicación?")
        m.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        m.setDefaultButton(QMessageBox.No)
        # Texto y botones legibles (no heredar estilos oscuros globales)
        m.setStyleSheet(
            "QLabel{color:#111; font-size:12px;} "
            "QPushButton{min-width:90px; min-height:30px; font-weight:600;}"
        )
        m.button(QMessageBox.Yes).setText("Sí")
        m.button(QMessageBox.No).setText("No")
        if m.exec_() == QMessageBox.Yes:
            self.close()
    def _on_export_selected(self, idx: int):
        # idx 0 = "Exportar" (placeholder)
        if idx <= 0:
            return
        choice = self.export_combo.currentText().strip().lower()
        # volver al placeholder inmediatamente para que el combo actúe como botón desplegable
        self.export_combo.blockSignals(True)
        self.export_combo.setCurrentIndex(0)
        self.export_combo.blockSignals(False)

        if "csv" in choice:
            self.exportar_csv()
        elif "excel" in choice:
            self.exportar_excel()
        elif "pdf" in choice:
            self.exportar_pdf()

    def _coerce_history(self, history: Any) -> List[Dict[str, Any]]:
        if history is None:
            return []
        if isinstance(history, list):
            # asegurar dicts
            out = []
            for row in history:
                if isinstance(row, dict):
                    out.append(row)
                elif isinstance(row, (list, tuple)):
                    out.append({f"c{i+1}": v for i, v in enumerate(row)})
                else:
                    out.append({"value": row})
            return out
        if isinstance(history, tuple):
            return self._coerce_history(list(history))
        if isinstance(history, dict):
            return [history]
        return [{"value": history}]

    def _parse_method_output(self, metodo: str, out: Any) -> Dict[str, Any]:
        """Normaliza salidas de métodos a un dict: {'resumen':..., 'history':...}."""
        metodo_norm = metodo.lower()
        resumen: Dict[str, Any] = {"metodo": metodo, "funcion": self.func_input.text().strip()}

        # Búsqueda Local: (x, f, iter, tipo, distancia[, history])
        if "búsqueda" in metodo_norm or "busqueda" in metodo_norm:
            if isinstance(out, (tuple, list)):
                if len(out) >= 5:
                    x_opt, f_opt, iters, tipo, dist = out[:5]
                    resumen.update({"x_opt": x_opt, "f_opt": f_opt, "iter": iters, "tipo": tipo, "distancia": dist})
                history = out[5] if len(out) >= 6 else []
            else:
                history = []
            return {"resumen": resumen, "history": self._coerce_history(history)}

        # Fibonacci: (x_opt, f_opt, iter, tipo, distancia[, history, limit_reached, target])
        if "fibonacci" in metodo_norm:
            if isinstance(out, (tuple, list)) and len(out) >= 5:
                x_opt, f_opt, iters, tipo, dist = out[:5]
                resumen.update({"x_opt": x_opt, "f_opt": f_opt, "iter": iters, "tipo": tipo, "distancia": dist})
                history = out[5] if len(out) >= 6 else []
                if len(out) >= 7:
                    resumen["limit_reached"] = bool(out[6])
                if len(out) >= 8:
                    resumen["target"] = out[7]
            else:
                history = []
            return {"resumen": resumen, "history": self._coerce_history(history)}

        # Armijo/Wolfe wrappers pueden devolver solo history cuando return_history=True
        if "armijo" in metodo_norm or "wolfe" in metodo_norm:
            if isinstance(out, list) and (len(out) == 0 or isinstance(out[0], dict)):
                history = out
                # intentar inferir x_opt/f_opt del último registro
                if history:
                    last = history[-1]
                    for k in ("x_new", "x", "x_opt", "x*"):
                        if k in last:
                            try:
                                resumen["x_opt"] = float(last[k])
                                break
                            except Exception:
                                pass
                    for k in ("f_new", "fx", "f(x)", "f_opt"):
                        if k in last:
                            try:
                                resumen["f_opt"] = float(last[k])
                                break
                            except Exception:
                                pass
                    resumen["iter"] = len(history)
                return {"resumen": resumen, "history": self._coerce_history(history)}
            # si devuelve tuple: (x_new,f_new,iters,dist[,history])
            if isinstance(out, (tuple, list)) and len(out) >= 4:
                x_opt, f_opt, iters, dist = out[:4]
                resumen.update({"x_opt": x_opt, "f_opt": f_opt, "iter": iters, "distancia": dist})
                history = out[4] if len(out) >= 5 else []
                return {"resumen": resumen, "history": self._coerce_history(history)}
            return {"resumen": resumen, "history": []}

        return {"resumen": resumen, "history": []}

    def exportar_csv(self):
        expr = self.func_input.text().strip()
        if not expr or expr not in self._results or not self._results[expr]:
            self._show_msg("Aviso", "No hay resultados para exportar para esta función.\nEjecute al menos un método.")
            return

        metodo = self.metodo_menu.currentText()
        data = self._results[expr].get(metodo)
        if not data:
            self._show_msg("Aviso", f"No hay resultados guardados para el método '{metodo}'.\nEjecute el método primero.")
            return

        history = data["history"]

        filepath, _ = QFileDialog.getSaveFileName(self, "Guardar CSV", f"resultados_{metodo}.csv", "CSV (*.csv)")
        if not filepath:
            return
        # Exportación CSV sin pandas (evita dependencia extra)
        try:
            with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
                if history:
                    fieldnames = list(history[0].keys())
                else:
                    fieldnames = []
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in history:
                    writer.writerow(row)
        except Exception as e:
            self._show_msg("Error", f"No se pudo exportar el CSV:\n{e}", icon=QMessageBox.Critical)
            return
        self._show_msg("Éxito", "Archivo CSV exportado exitosamente.", icon=QMessageBox.Information)

    def exportar_excel(self):
        expr = self.func_input.text().strip()
        if not expr or expr not in self._results or not self._results[expr]:
            self._show_msg("Aviso", "No hay resultados para exportar para esta función.\nEjecute al menos un método.")
            return

        # un solo excel: una pestaña por método ejecutado
        items: List[ReportItem] = []
        for metodo, data in self._results[expr].items():
            items.append(data["item"])

        default_name = "reporte_optimizacion.xlsx"
        filepath, _ = QFileDialog.getSaveFileName(self, "Guardar Reporte Excel", default_name, "Excel (*.xlsx)")
        if not filepath:
            return

        try:
            exportar_reporte_excel(items, output_path=filepath, app_title="Optimizador de Funciones")
            self._show_msg("Éxito", "Reporte Excel exportado exitosamente.", icon=QMessageBox.Information)
        except Exception as e:
            # si falla, reportar claro (texto negro en QMessageBox por defecto)
            self._show_msg("Error", f"No se pudo exportar el Excel:\n{e}", icon=QMessageBox.Critical)

    def exportar_pdf(self):
        expr = self.func_input.text().strip()
        if not expr or expr not in self._results or not self._results[expr]:
            self._show_msg("Aviso", "No hay resultados para exportar para esta función.\nEjecute al menos un método.")
            return
        items: List[ReportItem] = []
        for metodo, data in self._results[expr].items():
            if data.get("item") is not None:
                items.append(data["item"])
        if exportar_reporte_pdf is None:
            self._show_msg("Error", "No se pudo exportar el PDF: dependencia no disponible (reportlab).", icon=QMessageBox.Critical)
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Guardar Reporte PDF", "reporte_optimizacion.pdf", "PDF (*.pdf)")
        if not filepath:
            return
        try:
            exportar_reporte_pdf(items, output_path=filepath, app_title="Optimizador de Funciones")
            self._show_msg("Éxito", "Reporte PDF exportado exitosamente.", icon=QMessageBox.Information)
        except Exception as e:
            self._show_msg("Error", f"No se pudo exportar el PDF:\n{e}", icon=QMessageBox.Critical)

    # ---------------- Helpers ----------------
    def _install_spanish_context_menus(self):
        # Tabla: menú contextual en español (Copiar, Seleccionar todo)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_table_context_menu)

        # LineEdits relevantes: menú contextual en español (Cortar/Copiar/Pegar/Seleccionar todo)
        edits = [self.func_input]
        for sp in (self.min_spin, self.max_spin, self.tol_spin):
            le = sp.lineEdit()
            if le is not None:
                edits.append(le)

        for le in edits:
            le.setContextMenuPolicy(Qt.CustomContextMenu)
            le.customContextMenuRequested.connect(lambda pos, w=le: self._show_lineedit_context_menu(w, pos))

    def _show_table_context_menu(self, pos):
        menu = QMenu(self.table)
        act_copy = menu.addAction("Copiar")
        act_select_all = menu.addAction("Seleccionar todo")

        sel = self.table.selectedRanges()
        act_copy.setEnabled(bool(sel))

        action = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if action == act_copy:
            self._copy_table_selection()
        elif action == act_select_all:
            self.table.selectAll()

    def _copy_table_selection(self):
        sel = self.table.selectedRanges()
        if not sel:
            return
        r = sel[0]
        rows = []
        for row in range(r.topRow(), r.bottomRow() + 1):
            cols = []
            for col in range(r.leftColumn(), r.rightColumn() + 1):
                item = self.table.item(row, col)
                cols.append("" if item is None else item.text())
            rows.append("\t".join(cols))
        QApplication.clipboard().setText("\n".join(rows))

    def _show_lineedit_context_menu(self, widget, pos):
        menu = QMenu(widget)
        act_cut = menu.addAction("Cortar")
        act_copy = menu.addAction("Copiar")
        act_paste = menu.addAction("Pegar")
        menu.addSeparator()
        act_select_all = menu.addAction("Seleccionar todo")

        has_sel = widget.hasSelectedText()
        act_cut.setEnabled(has_sel and not widget.isReadOnly())
        act_copy.setEnabled(has_sel)
        act_paste.setEnabled(not widget.isReadOnly())

        action = menu.exec_(widget.mapToGlobal(pos))
        if action == act_cut:
            widget.cut()
        elif action == act_copy:
            widget.copy()
        elif action == act_paste:
            widget.paste()
        elif action == act_select_all:
            widget.selectAll()

    def _show_msg(self, title: str, text: str, icon=QMessageBox.Warning):
        # Forzar legibilidad: el stylesheet global pinta QLabel en blanco.
        m = QMessageBox(self)
        m.setWindowTitle(title)
        m.setText(text)
        m.setIcon(icon)
        m.setStyleSheet("""
            QLabel { color: #111; font-size: 12px; }
            QPushButton { min-width: 90px; padding: 6px 14px; }
        """)
        m.exec_()
