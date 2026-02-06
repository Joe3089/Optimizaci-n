# -*- coding: utf-8 -*-
import os
import sys
from typing import Any, Dict, List, Callable

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QLineEdit,
    QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QMessageBox,
    QTableWidget, QTableWidgetItem
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MD_DIR = os.path.join(BASE_DIR, "multi-dimensionals")
if os.path.isdir(MD_DIR) and MD_DIR not in sys.path:
    sys.path.insert(0, MD_DIR)

from armijo import metodo_armijo
from wolfe import metodo_wolfe
from fibonacci import metodo_fibonacci
from busqueda_local import busqueda_local

try:
    from wrappers import (
        penalty_method_newton,
        barrier_method_newton,
        penalty_method_bfgs,
        weighted_sum_bfgs,
        penalty_method_nelder,
        weighted_sum_nelder,
        weighted_sum_newton,
    )
except Exception:
    import wrappers as _w

    def _safe(name):
        fn = getattr(_w, name, None)
        if callable(fn):
            return fn
        def _missing(*args, **kwargs):
            raise NotImplementedError(f"Metodo no disponible: {name}")
        return _missing
    
    penalty_method_newton = _safe("penalty_method_newton")
    barrier_method_newton = _safe("barrier_method_newton")
    penalty_method_bfgs = _safe("penalty_method_bfgs")
    weighted_sum_bfgs = _safe("weighted_sum_bfgs")
    penalty_method_nelder = _safe("penalty_method_nelder")
    weighted_sum_nelder = _safe("weighted_sum_nelder")
    weighted_sum_newton = _safe("weighted_sum_newton")

from canvas_2d import Function2DCanvas
from rotacion_3d import Rotating3DCanvas

SAFE_MATH = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
    "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi, "e": np.e, "pow": np.power,
    "arctan": np.arctan, "arcsin": np.arcsin, "arccos": np.arccos,
}

def _parse_vars(var_str: str) -> List[str]:
    s = (var_str or "").strip().replace(",", " ")
    parts = [p.strip() for p in s.split() if p.strip()]
    return parts if parts else ["x"]

def _parse_x0(x0_str: str) -> List[float]:
    s = (x0_str or "").strip()
    if not s:
        return [0.0]
    s = s.replace(";", ",").replace(" ", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]

def _compile_expr(expr: str, vars_list: List[str]) -> Callable:
    code = compile(expr, "<expr>", "eval")
    def f(*args):
        env = dict(SAFE_MATH)
        for name, val in zip(vars_list, args):
            env[name] = float(val)
        return float(eval(code, {"__builtins__": {}}, env))
    return f


class InterfazOptimizacion(QMainWindow):
    def __init__(self, resource_path=None):
        super().__init__()
        self.resource_path = resource_path if isinstance(resource_path, str) else None
        self.setWindowTitle("Optimizador de Funciones")
        self.resize(1250, 720)

        # Fondo
        self.bg_label = QLabel(self)
        self.bg_label.setScaledContents(True)
        self.bg_label.lower()

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        # ---- Izquierda ----
        self.left_panel = QFrame()
        self.left_panel.setObjectName("leftPanel")
        left = QVBoxLayout(self.left_panel)
        left.setContentsMargins(18, 18, 18, 18)
        left.setSpacing(10)

        title = QLabel("Panel de control")
        title.setObjectName("panelTitle")
        left.addWidget(title)

        left.addWidget(QLabel("Método:"))
        self.metodo_menu = QComboBox()
        self.metodo_menu.addItems([
            "Armijo",
            "Wolfe",
            "Fibonacci",
            "Búsqueda Local",
            "MD: Penalización (Newton)",
            "MD: Barrera (Newton)",
            "MD: Penalización (BFGS)",
            "MD: Suma Ponderada (BFGS)",
            "MD: Penalización (Nelder-Mead)",
            "MD: Suma Ponderada (Nelder-Mead)",
            "MD: Suma Ponderada (Newton)",
        ])
        left.addWidget(self.metodo_menu)

        left.addWidget(QLabel("Función f(x):"))
        self.func_input = QLineEdit()
        self.func_input.setPlaceholderText("Ej: (x1-4)**2 + (x2-4)**2")
        left.addWidget(self.func_input)

        left.addWidget(QLabel("Restricción g(x) (<=0):"))
        self.constraint_input = QLineEdit()
        self.constraint_input.setPlaceholderText("Ej: x1**2 + x2**2 - 1")
        left.addWidget(self.constraint_input)

        left.addWidget(QLabel("Variables:"))
        self.vars_input = QLineEdit()
        self.vars_input.setPlaceholderText("Ej: x   o   x1 x2")
        left.addWidget(self.vars_input)

        left.addWidget(QLabel("x0 (coma):"))
        self.x0_input = QLineEdit()
        self.x0_input.setPlaceholderText("Ej: 0.5   o   0.7,0.7")
        left.addWidget(self.x0_input)

        grid = QGridLayout()
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(6)
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setValue(0.0)
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(6)
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setValue(5.0)
        grid.addWidget(QLabel("mínimo:"), 0, 0)
        grid.addWidget(self.min_spin, 0, 1)
        grid.addWidget(QLabel("máximo:"), 1, 0)
        grid.addWidget(self.max_spin, 1, 1)
        left.addLayout(grid)

        btn_row = QHBoxLayout()
        self.btn_calc = QPushButton("Calcular")
        self.btn_clear = QPushButton("Limpiar")
        btn_row.addWidget(self.btn_calc)
        btn_row.addWidget(self.btn_clear)
        left.addLayout(btn_row)

        self.btn_exit = QPushButton("Salir")
        left.addWidget(self.btn_exit)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        left.addWidget(self.status_lbl)
        left.addStretch(1)

        # ---- Derecha ----
        self.right_panel = QFrame()
        self.right_panel.setObjectName("rightPanel")
        right = QVBoxLayout(self.right_panel)
        right.setContentsMargins(18, 18, 18, 18)
        right.setSpacing(10)

        dash = QLabel("Dashboard")
        dash.setObjectName("dashTitle")
        right.addWidget(dash)

        self.table = QTableWidget(0, 6)
        self.table.setFixedHeight(190)
        right.addWidget(self.table)

        self.plot_frame = QFrame()
        self.plot_frame.setObjectName("plotFrame")
        pl = QVBoxLayout(self.plot_frame)
        pl.setContentsMargins(10, 10, 10, 10)
        pl.setSpacing(10)

        self.lbl2d = QLabel("Ejecute 'Calcular' para ver la gráfica 2D.")
        self.lbl2d.setObjectName("plotPlaceholder")
        self.lbl2d.setAlignment(Qt.AlignCenter)
        self.canvas2d = Function2DCanvas(self.plot_frame, title="Gráfica 2D")
        self.canvas2d.setVisible(False)
        self.canvas2d.setFixedHeight(240)

        self.lbl3d = QLabel("Ejecute 'Calcular' para ver la gráfica 3D.")
        self.lbl3d.setObjectName("plotPlaceholder")
        self.lbl3d.setAlignment(Qt.AlignCenter)
        self.canvas3d = Rotating3DCanvas(self.plot_frame, title="Gráfica 3D")
        self.canvas3d.setVisible(False)

        pl.addWidget(self.lbl2d)
        pl.addWidget(self.canvas2d)
        pl.addWidget(self.lbl3d)
        pl.addWidget(self.canvas3d, 1)

        right.addWidget(self.plot_frame, 1)

        root.addWidget(self.left_panel, 0)
        root.addWidget(self.right_panel, 1)

        self._apply_styles()
        self._apply_background()

        self.btn_exit.clicked.connect(self._confirm_exit)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_calc.clicked.connect(self.ejecutar_metodo)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.bg_label.setGeometry(0, 0, self.width(), self.height())

    def _apply_background(self):
        candidates = []
        if self.resource_path and os.path.isdir(self.resource_path):
            candidates += [
                os.path.join(self.resource_path, "bg.png"),
                os.path.join(self.resource_path, "background.png"),
                os.path.join(self.resource_path, "fondo.png"),
            ]
        candidates += [
            os.path.join(BASE_DIR, "report_assets", "bg.png"),
            os.path.join(BASE_DIR, "report_assets", "background.png"),
            os.path.join(BASE_DIR, "report_assets", "fondo.png"),
        ]
        chosen = None
        for p in candidates:
            if os.path.exists(p):
                chosen = p
                break
        if chosen:
            self.bg_label.setPixmap(QPixmap(chosen))
        else:
            self.bg_label.setStyleSheet("background-color:#0b1220;")

    def _apply_styles(self):
        self.setStyleSheet(r"""
        QMainWindow { background: transparent; }
        QFrame#leftPanel, QFrame#rightPanel {
            background-color: rgba(2, 6, 23, 0.78);
            border-radius: 18px;
        }
        QLabel { color: white; font-size: 12px; }
        QLabel#panelTitle { font-size: 20px; font-weight: 800; }
        QLabel#dashTitle { font-size: 20px; font-weight: 800; }
        QLabel#plotPlaceholder { color: #d6e0ff; font-size: 12px; }
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
            border-radius: 12px;
            padding: 10px 12px;
            font-weight: 700;
            min-height: 38px;
        }
        QPushButton:hover { background-color: rgba(0, 140, 255, 0.70); }
        QPushButton:pressed { background-color: rgba(0, 140, 255, 0.85); }
        QFrame#plotFrame {
            background-color: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
        }
        QTableWidget {
            background-color: rgba(255,255,255,0.03);
            color: white;
            gridline-color: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
        }
        QHeaderView::section {
            background-color: rgba(255,255,255,0.06);
            color: white;
            padding: 6px;
            border: 0px;
        }
        """)

    def _confirm_exit(self):
        m = QMessageBox(self)
        m.setWindowTitle("Salir")
        m.setIcon(QMessageBox.Question)
        m.setText("¿Está seguro de que desea salir de la aplicación?")
        m.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        m.setDefaultButton(QMessageBox.No)
        m.button(QMessageBox.Yes).setText("Sí")
        m.button(QMessageBox.No).setText("No")
        if m.exec_() == QMessageBox.Yes:
            self.close()

    def _on_clear(self):
        self.table.setRowCount(0)
        self.canvas2d.setVisible(False)
        self.canvas3d.setVisible(False)
        self.lbl2d.setVisible(True)
        self.lbl3d.setVisible(True)
        self.status_lbl.setText("")

    def _fill_table_generic(self, history: List[Dict[str, Any]]):
        self.table.setRowCount(0)
        if not history:
            return
        cols = list(history[0].keys())
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        for r, row in enumerate(history):
            self.table.insertRow(r)
            for c, k in enumerate(cols):
                it = QTableWidgetItem(str(row.get(k, "")))
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, c, it)

    def ejecutar_metodo(self):
        try:
            metodo = self.metodo_menu.currentText().strip()
            f_str = self.func_input.text().strip()
            g_str = self.constraint_input.text().strip()
            vars_list = _parse_vars(self.vars_input.text())
            x0 = _parse_x0(self.x0_input.text())
            a = float(self.min_spin.value())
            b = float(self.max_spin.value())

            if not f_str:
                QMessageBox.warning(self, "Falta dato", "Escriba la función f(x).")
                return

            # -------- 1D --------
            if not metodo.startswith("MD:"):
                if len(vars_list) != 1:
                    QMessageBox.warning(self, "Variables", "Para métodos 1D use una sola variable (ej: x).")
                    return
                f = _compile_expr(f_str, vars_list)

                if metodo == "Armijo":
                    history = metodo_armijo(f, a=a, b=b, max_iter=200, return_history=True)
                elif metodo == "Wolfe":
                    history = metodo_wolfe(f, a=a, b=b, max_iter=200, return_history=True)
                elif metodo == "Fibonacci":
                    out = metodo_fibonacci(f, a=a, b=b, tolerancia=1e-5, max_n=200, return_history=True)
                    history = out[5] if isinstance(out, (list, tuple)) and len(out) >= 6 else []
                elif metodo == "Búsqueda Local":
                    x0v = float(x0[0]) if x0 else (a + b) / 2.0
                    out = busqueda_local(f, x0=x0v, paso=0.1, max_iter=200, tolerancia=1e-5, return_history=True)
                    history = out[5] if isinstance(out, (list, tuple)) and len(out) >= 6 else []
                else:
                    raise ValueError("Método 1D no soportado.")

                self._fill_table_generic(history)

                xs = []
                for row in history:
                    for key in ("x_new", "x", "x_k"):
                        if key in row and row[key] is not None and not isinstance(row[key], (list, tuple)):
                            xs.append(float(row[key]))
                            break

                x_curve = np.linspace(a, b, 500)
                y_curve = np.array([f(float(xx)) for xx in x_curve], dtype=float)
                y_path = np.array([f(float(xx)) for xx in xs], dtype=float) if xs else np.array([])

                self.lbl2d.setVisible(False)
                self.canvas2d.setVisible(True)
                self.canvas2d.set_title("Gráfica 2D - " + metodo)
                self.canvas2d.plot_1d(x_curve, y_curve, np.array(xs), y_path)

                self.lbl3d.setVisible(False)
                self.canvas3d.setVisible(True)
                self.canvas3d.set_title("Gráfica 3D - " + metodo)
                iters = np.arange(len(xs), dtype=float)
                zs = np.array([f(float(xx)) for xx in xs], dtype=float) if xs else np.array([])
                if len(xs) > 0:
                    self.canvas3d.set_curve3d(iters, np.array(xs), zs, labels=("k", "x", "f(x)"))
                else:
                    self.canvas3d.clear("Sin trayectoria.")

                self.status_lbl.setText(f"✓ {metodo} ejecutado.")
                return

            # -------- MD --------
            if len(vars_list) < 2:
                QMessageBox.warning(self, "Variables", "Para métodos MD use al menos 2 variables (ej: x1 x2).")
                return
            if not g_str:
                QMessageBox.warning(self, "Restricción", "Para métodos MD escriba g(x) (<=0).")
                return
            if len(x0) < 2:
                QMessageBox.warning(self, "x0", "Para métodos MD escriba al menos 2 valores en x0.")
                return

            var_str = " ".join(vars_list)

            if metodo == "MD: Penalización (Newton)":
                x_opt, f_opt, history = penalty_method_newton(f_str, g_str, var_str, x0)
            elif metodo == "MD: Barrera (Newton)":
                x_opt, f_opt, history = barrier_method_newton(f_str, g_str, var_str, x0)
            elif metodo == "MD: Penalización (BFGS)":
                x_opt, f_opt, history = penalty_method_bfgs(f_str, g_str, var_str, x0)
            elif metodo == "MD: Suma Ponderada (BFGS)":
                x_opt, f_opt, history = weighted_sum_bfgs(f_str, g_str, var_str, x0)
            elif metodo == "MD: Penalización (Nelder-Mead)":
                x_opt, f_opt, history = penalty_method_nelder(f_str, g_str, var_str, x0)
            elif metodo == "MD: Suma Ponderada (Nelder-Mead)":
                x_opt, f_opt, history = weighted_sum_nelder(f_str, g_str, var_str, x0)
            elif metodo == "MD: Suma Ponderada (Newton)":
                x_opt, f_opt, history = weighted_sum_newton(f_str, g_str, var_str, x0)
            else:
                raise ValueError("Método MD no soportado.")

            self._fill_table_generic(history)

            pts = []
            for row in history:
                xk = row.get("x_k", None)
                if isinstance(xk, (list, tuple)) and len(xk) >= 2:
                    pts.append((float(xk[0]), float(xk[1])))

            f2 = _compile_expr(f_str, vars_list[:2])
            g2 = _compile_expr(g_str, vars_list[:2])

            X = np.linspace(a, b, 120)
            Y = np.linspace(a, b, 120)
            XX, YY = np.meshgrid(X, Y)
            Z = np.zeros_like(XX, dtype=float)
            GZ = np.zeros_like(XX, dtype=float)
            for i in range(XX.shape[0]):
                for j in range(XX.shape[1]):
                    Z[i, j] = float(f2(float(XX[i, j]), float(YY[i, j])))
                    GZ[i, j] = float(g2(float(XX[i, j]), float(YY[i, j])))

            path_x = np.array([p[0] for p in pts], dtype=float) if pts else np.array([])
            path_y = np.array([p[1] for p in pts], dtype=float) if pts else np.array([])

            self.lbl2d.setVisible(False)
            self.canvas2d.setVisible(True)
            self.canvas2d.set_title("Gráfica 2D - " + metodo)
            self.canvas2d.plot_contours(XX, YY, Z, path_x, path_y, gZ=GZ)

            X3 = np.linspace(a, b, 55)
            Y3 = np.linspace(a, b, 55)
            XX3, YY3 = np.meshgrid(X3, Y3)
            Z3 = np.zeros_like(XX3, dtype=float)
            for i in range(XX3.shape[0]):
                for j in range(XX3.shape[1]):
                    Z3[i, j] = float(f2(float(XX3[i, j]), float(YY3[i, j])))

            pz = np.array([float(f2(float(px), float(py))) for px, py in pts], dtype=float) if pts else np.array([])

            self.lbl3d.setVisible(False)
            self.canvas3d.setVisible(True)
            self.canvas3d.set_title("Gráfica 3D - " + metodo)
            self.canvas3d.set_surface_and_path(XX3, YY3, Z3, path_x, path_y, pz, labels=("x1", "x2", "f(x)"))

            self.status_lbl.setText(f"✓ {metodo} → x*={x_opt}, f*={f_opt}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ocurrió un error:\\n{e}")
