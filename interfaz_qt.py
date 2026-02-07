# interfaz_qt.py
from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# --------- Import canvases (must be in same folder as main.py) ----------
try:
    from canvas_2d import Function2DCanvas
except Exception:
    Function2DCanvas = None  # type: ignore

try:
    from rotacion_3d import Rotating3DCanvas
except Exception:
    Rotating3DCanvas = None  # type: ignore

# --------- Ensure multi-dimensionals is importable ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_DIR = os.path.join(BASE_DIR, "multi-dimensionals")
if os.path.isdir(MD_DIR) and MD_DIR not in sys.path:
    sys.path.insert(0, MD_DIR)

# 1D methods in root
try:
    import armijo as _armijo
except Exception:
    _armijo = None
try:
    import wolfe as _wolfe
except Exception:
    _wolfe = None
try:
    import fibonacci as _fibo
except Exception:
    _fibo = None
try:
    import busqueda_local as _blocal
except Exception:
    _blocal = None

# MD methods in multi-dimensionals
try:
    import wrappers as _wrap
except Exception:
    _wrap = None
try:
    import line_search as _md_linesearch
except Exception:
    _md_linesearch = None
try:
    import localsearch as _md_localsearch
except Exception:
    _md_localsearch = None

# sympy parsing (for plotting)
import sympy as sp


@dataclass
class MethodSpec:
    key: str
    label: str
    kind: str  # '1d' or 'md'
    needs_g: bool = False


METHODS: List[MethodSpec] = [
    MethodSpec("armijo_1d", "Armijo (1D)", "1d"),
    MethodSpec("wolfe_1d", "Wolfe (1D)", "1d"),
    MethodSpec("fibonacci_1d", "Fibonacci (1D)", "1d"),
    MethodSpec("busqueda_local_1d", "Búsqueda local (1D)", "1d"),

    MethodSpec("penalty_newton", "MD: Penalización (Newton)", "md", needs_g=True),
    MethodSpec("barrier_newton", "MD: Barrera (Newton)", "md", needs_g=True),

    # wrappers extras (si existen en wrappers.py)
    MethodSpec("penalty_bfgs", "MD: Penalización (BFGS)", "md", needs_g=True),
    MethodSpec("weighted_sum_bfgs", "MD: Suma ponderada (BFGS)", "md", needs_g=True),
    MethodSpec("penalty_nelder", "MD: Penalización (Nelder)", "md", needs_g=True),
    MethodSpec("weighted_sum_nelder", "MD: Suma ponderada (Nelder)", "md", needs_g=True),
    MethodSpec("weighted_sum_newton", "MD: Suma ponderada (Newton)", "md", needs_g=True),

    # line_search module (si existe)
    MethodSpec("md_newton_armijo", "MD: Newton + Armijo", "md", needs_g=False),
    MethodSpec("md_newton_wolfe", "MD: Newton + Wolfe", "md", needs_g=False),
    MethodSpec("md_wolfe_line_search", "MD: Wolfe line search", "md", needs_g=False),

    # localsearch module (si existe)
    MethodSpec("md_fibonacci_search", "MD: Fibonacci search", "md", needs_g=False),
    MethodSpec("md_local_search", "MD: Local search", "md", needs_g=False),
    MethodSpec("md_scipy_minimize", "MD: SciPy minimize", "md", needs_g=False),
]


def _safe_getattr(mod, name: str) -> Optional[Callable]:
    if mod is None:
        return None
    return getattr(mod, name, None)


def _parse_vars(vars_text: str) -> List[str]:
    txt = (vars_text or "").replace(",", " ").strip()
    parts = [p for p in txt.split() if p]
    return parts


def _parse_x0(x0_text: str) -> List[float]:
    txt = (x0_text or "").replace(";", ",").strip()
    if not txt:
        return []
    parts = [p.strip() for p in txt.split(",")]
    out = []
    for p in parts:
        if p:
            out.append(float(p))
    return out


def _compile_sympy_func(expr_text: str, var_names: List[str]) -> Callable:
    expr = sp.sympify(expr_text)
    syms = [sp.Symbol(v) for v in var_names]
    f = sp.lambdify(syms, expr, modules=["numpy"])
    return f


def _grid(minv: float, maxv: float, n: int = 80):
    xs = np.linspace(minv, maxv, n)
    ys = np.linspace(minv, maxv, n)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def _history_to_path_xy(history: Any) -> List[Tuple[float, float]]:
    """
    Intenta extraer una trayectoria (x1,x2) desde varios formatos de history.
    Soporta:
      - list of dict con 'x_k' o 'x'
      - list of list/tuple
      - ndarray Nx2
    """
    if history is None:
        return []
    if isinstance(history, np.ndarray):
        if history.ndim == 2 and history.shape[1] >= 2:
            return [(float(r[0]), float(r[1])) for r in history]
    if isinstance(history, list):
        out = []
        for h in history:
            if isinstance(h, dict):
                x = h.get("x_k", h.get("x", None))
                if x is None:
                    continue
                if isinstance(x, (list, tuple, np.ndarray)) and len(x) >= 2:
                    out.append((float(x[0]), float(x[1])))
            elif isinstance(h, (list, tuple, np.ndarray)) and len(h) >= 2:
                out.append((float(h[0]), float(h[1])))
        return out
    return []


class InterfazOptimizacion(QMainWindow):
    def __init__(self, resource_path=None):
        super().__init__()
        self.setWindowTitle("Optimizador de Funciones")
        self.resize(1200, 720)

        # resource_path puede venir como función desde main.py
        if callable(resource_path):
            try:
                resource_path = resource_path()
            except Exception:
                resource_path = None
        self.resource_path = resource_path

        self._build_ui()
        self._apply_style()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(12)

        # LEFT: Panel de control
        self.left_panel = QFrame()
        self.left_panel.setObjectName("LeftPanel")
        self.left_panel.setFixedWidth(360)
        left = QVBoxLayout(self.left_panel)
        left.setSpacing(10)

        title = QLabel("Panel de control")
        title.setObjectName("TitleLabel")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        left.addWidget(title)

        self.method_combo = QComboBox()
        self.method_combo.setObjectName("Combo")
        for m in METHODS:
            self.method_combo.addItem(m.label, m.key)
        left.addWidget(QLabel("Método:"))
        left.addWidget(self.method_combo)

        self.f_edit = QLineEdit()
        self.f_edit.setPlaceholderText("Ej: (x1-4)**2 + (x2-4)**2")
        left.addWidget(QLabel("Función f(x):"))
        left.addWidget(self.f_edit)

        self.g_edit = QLineEdit()
        self.g_edit.setPlaceholderText("Ej: x1**2 + x2**2 - 1  (<= 0)")
        left.addWidget(QLabel("Restricción g(x) (<=0):"))
        left.addWidget(self.g_edit)

        self.vars_edit = QLineEdit()
        self.vars_edit.setPlaceholderText("Ej: x1 x2")
        left.addWidget(QLabel("Variables:"))
        left.addWidget(self.vars_edit)

        self.x0_edit = QLineEdit()
        self.x0_edit.setPlaceholderText("Ej: 0.5, 0.5")
        left.addWidget(QLabel("x0 (coma):"))
        left.addWidget(self.x0_edit)

        # min/max
        mm = QHBoxLayout()
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(6)
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setValue(0.0)
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(6)
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setValue(5.0)
        mm.addWidget(QLabel("mínimo:"))
        mm.addWidget(self.min_spin)
        mm.addSpacing(8)
        mm.addWidget(QLabel("máximo:"))
        mm.addWidget(self.max_spin)
        left.addLayout(mm)

        # Buttons
        btnrow1 = QHBoxLayout()
        self.btn_calc = QPushButton("Calcular")
        self.btn_clear = QPushButton("Limpiar")
        btnrow1.addWidget(self.btn_calc)
        btnrow1.addWidget(self.btn_clear)
        left.addLayout(btnrow1)

        btnrow2 = QHBoxLayout()
        self.btn_export = QPushButton("Exportar")
        self.btn_exit = QPushButton("Salir")
        btnrow2.addWidget(self.btn_export)
        btnrow2.addWidget(self.btn_exit)
        left.addLayout(btnrow2)

        self.status = QLabel("")
        self.status.setObjectName("StatusLabel")
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        left.addStretch(1)

        main.addWidget(self.left_panel)

        # RIGHT: Dashboard + plots
        self.right_panel = QFrame()
        self.right_panel.setObjectName("RightPanel")
        right = QVBoxLayout(self.right_panel)
        right.setSpacing(8)

        dash_title = QLabel("Dashboard")
        dash_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        dash_title.setObjectName("TitleLabel")
        right.addWidget(dash_title)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["k", "x_k", "x_k_str", "f_k", "grad_norm", "alpha_k"])
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(self.table, stretch=3)

        # Splitter for 2D / 3D
        self.splitter = QSplitter(Qt.Vertical)
        self.plot2d_frame = QFrame()
        self.plot3d_frame = QFrame()
        self.splitter.addWidget(self.plot2d_frame)
        self.splitter.addWidget(self.plot3d_frame)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        right.addWidget(self.splitter, stretch=4)

        # 2D canvas
        p2 = QVBoxLayout(self.plot2d_frame)
        p2.setContentsMargins(0, 0, 0, 0)
        if Function2DCanvas is None:
            self.canvas2d = QLabel("No se pudo cargar el canvas 2D.")
            p2.addWidget(self.canvas2d)
        else:
            self.canvas2d = Function2DCanvas(self.plot2d_frame, title="Gráfica 2D")
            p2.addWidget(self.canvas2d)

        # 3D canvas
        p3 = QVBoxLayout(self.plot3d_frame)
        p3.setContentsMargins(0, 0, 0, 0)
        if Rotating3DCanvas is None:
            self.canvas3d = QLabel("No se pudo cargar el canvas 3D.")
            p3.addWidget(self.canvas3d)
        else:
            self.canvas3d = Rotating3DCanvas(self.plot3d_frame, title="Gráfica 3D")
            p3.addWidget(self.canvas3d)
            self.canvas3d.start_rotation(60)

        main.addWidget(self.right_panel, stretch=1)

        # Signals
        self.btn_exit.clicked.connect(self._confirm_exit)
        self.btn_clear.clicked.connect(self._clear)
        self.btn_calc.clicked.connect(self._calculate)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._on_method_changed()

    def _apply_style(self):
        # Minimal style; keep compatible with your existing background/assets.
        self.setStyleSheet("""
        QMainWindow { background: #0b1220; }
        #LeftPanel, #RightPanel { background: rgba(10, 18, 32, 0.75); border-radius: 18px; }
        QLabel { color: #e6eefc; font-family: Segoe UI; }
        #TitleLabel { color: #ffffff; }
        QLineEdit, QComboBox, QDoubleSpinBox {
            background: rgba(255,255,255,0.08);
            color: #ffffff;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 10px;
            padding: 6px;
        }
        QPushButton {
            background: #0b67c2;
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px;
            font-weight: 600;
        }
        QPushButton:hover { background: #0a5aa8; }
        QTableWidget { background: rgba(0,0,0,0.35); color: #e6eefc; gridline-color: rgba(255,255,255,0.12); }
        QHeaderView::section { background: rgba(255,255,255,0.10); color: #ffffff; padding: 6px; border: none; }
        #StatusLabel { color: #b9d3ff; }
        """)

    # ---------------- behavior ----------------
    def _on_method_changed(self):
        key = self.method_combo.currentData()
        spec = next((m for m in METHODS if m.key == key), None)
        needs_g = bool(spec.needs_g) if spec else False
        self.g_edit.setEnabled(needs_g)
        if not needs_g:
            self.g_edit.setPlaceholderText("(no requerido para este método)")
        else:
            self.g_edit.setPlaceholderText("Ej: x1**2 + x2**2 - 1  (<= 0)")

    def _confirm_exit(self):
        resp = QMessageBox.question(
            self,
            "Salir",
            "¿Seguro que quieres salir?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp == QMessageBox.Yes:
            self.close()

    def _clear(self):
        self.f_edit.clear()
        self.g_edit.clear()
        self.vars_edit.clear()
        self.x0_edit.clear()
        self.table.setRowCount(0)
        self.status.setText("")
        if hasattr(self.canvas2d, "clear"):
            self.canvas2d.clear()
        if hasattr(self.canvas3d, "clear"):
            self.canvas3d.clear()

    def _fill_table(self, rows: List[Dict[str, Any]]):
        self.table.setRowCount(0)
        for r in rows:
            k = self.table.rowCount()
            self.table.insertRow(k)
            for c, key in enumerate(["k", "x_k", "x_k_str", "f_k", "grad_norm", "alpha_k"]):
                val = r.get(key, "")
                self.table.setItem(k, c, QTableWidgetItem(str(val)))

    def _calculate(self):
        try:
            method_key = self.method_combo.currentData()
            f_text = self.f_edit.text().strip()
            g_text = self.g_edit.text().strip()
            var_names = _parse_vars(self.vars_edit.text())
            x0 = _parse_x0(self.x0_edit.text())
            minv = float(self.min_spin.value())
            maxv = float(self.max_spin.value())

            if not f_text:
                raise ValueError("Debes escribir f(x).")
            if not var_names:
                # default: 1D variable x
                var_names = ["x"] if len(x0) <= 1 else [f"x{i+1}" for i in range(len(x0))]

            # ensure x0 length
            if not x0:
                x0 = [0.5] * len(var_names)

            # Prepare plotting function
            f_np = _compile_sympy_func(f_text, var_names)

            # Draw base function (contour/surface) if 2 variables
            if len(var_names) >= 2 and hasattr(self.canvas2d, "set_contour"):
                X, Y = _grid(minv, maxv, n=90)
                Z = f_np(X, Y)
                self.canvas2d.set_contour(X, Y, Z)
                if hasattr(self.canvas3d, "set_surface"):
                    self.canvas3d.set_surface(X, Y, Z)

            # Execute method
            table_rows: List[Dict[str, Any]] = []
            path_xy: List[Tuple[float, float]] = []

            if method_key == "armijo_1d":
                if _armijo is None:
                    raise ImportError("armijo.py no se pudo importar.")
                func1 = lambda x: float(_compile_sympy_func(f_text, [var_names[0]])(x))
                a, b = minv, maxv
                xopt, fopt, hist = _armijo.metodo_armijo(func1, a, b, return_history=True)
                table_rows = [{"k": i, "x_k": h.get("x_k",""), "x_k_str": h.get("x_k",""), "f_k": h.get("f_k",""), "grad_norm": h.get("grad_norm",""), "alpha_k": h.get("alpha_k","")} for i,h in enumerate(hist or [])]
                # 2D plot for 1D: y=f(x), path on x axis
                if hasattr(self.canvas2d, "ax"):
                    self.canvas2d.ax.clear()
                    xs = np.linspace(a, b, 400)
                    ys = _compile_sympy_func(f_text, [var_names[0]])(xs)
                    self.canvas2d.ax.plot(xs, ys, linewidth=2.0)
                    if hist:
                        xs_k = [float(h["x_k"]) for h in hist if "x_k" in h]
                        ys_k = [_compile_sympy_func(f_text, [var_names[0]])(xk) for xk in xs_k]
                        self.canvas2d.ax.plot(xs_k, ys_k, linewidth=1.5)
                        self.canvas2d.ax.scatter(xs_k, ys_k, s=35)
                    self.canvas2d.ax.set_title("Gráfica 2D")
                    self.canvas2d.ax.grid(True, alpha=0.3)
                    self.canvas2d.draw_idle()
                self.status.setText(f"x*={xopt}, f*={fopt}")

            elif method_key == "wolfe_1d":
                if _wolfe is None:
                    raise ImportError("wolfe.py no se pudo importar.")
                func1 = lambda x: float(_compile_sympy_func(f_text, [var_names[0]])(x))
                a, b = minv, maxv
                xopt, fopt, hist = _wolfe.metodo_wolfe(func1, a, b, return_history=True)
                table_rows = [{"k": i, "x_k": h.get("x_k",""), "x_k_str": h.get("x_k",""), "f_k": h.get("f_k",""), "grad_norm": h.get("grad_norm",""), "alpha_k": h.get("alpha_k","")} for i,h in enumerate(hist or [])]
                self.status.setText(f"x*={xopt}, f*={fopt}")

            elif method_key == "fibonacci_1d":
                if _fibo is None:
                    raise ImportError("fibonacci.py no se pudo importar.")
                func1 = lambda x: float(_compile_sympy_func(f_text, [var_names[0]])(x))
                a, b = minv, maxv
                xopt, fopt, hist = _fibo.metodo_fibonacci(func1, a, b, return_history=True)
                table_rows = [{"k": i, "x_k": h.get("x_k",""), "x_k_str": h.get("x_k",""), "f_k": h.get("f_k",""), "grad_norm": "", "alpha_k": ""} for i,h in enumerate(hist or [])] if isinstance(hist, list) else []
                self.status.setText(f"x*={xopt}, f*={fopt}")

            elif method_key == "busqueda_local_1d":
                if _blocal is None:
                    raise ImportError("busqueda_local.py no se pudo importar.")
                func1 = lambda x: float(_compile_sympy_func(f_text, [var_names[0]])(x))
                xopt, fopt, hist = _blocal.busqueda_local(func1, float(x0[0]), return_history=True)
                table_rows = [{"k": i, "x_k": h.get("x_k",""), "x_k_str": h.get("x_k",""), "f_k": h.get("f_k",""), "grad_norm": "", "alpha_k": ""} for i,h in enumerate(hist or [])] if isinstance(hist, list) else []
                self.status.setText(f"x*={xopt}, f*={fopt}")

            else:
                # MD family
                if method_key in ("penalty_newton","barrier_newton","penalty_bfgs","weighted_sum_bfgs","penalty_nelder","weighted_sum_nelder","weighted_sum_newton"):
                    if _wrap is None:
                        raise ImportError("No se pudo importar multi-dimensionals/wrappers.py")
                    if not g_text:
                        raise ValueError("Este método requiere g(x) (<=0).")

                    fn_map = {
                        "penalty_newton": "penalty_method_newton",
                        "barrier_newton": "barrier_method_newton",
                        "penalty_bfgs": "penalty_method_bfgs",
                        "weighted_sum_bfgs": "weighted_sum_bfgs",
                        "penalty_nelder": "penalty_method_nelder",
                        "weighted_sum_nelder": "weighted_sum_nelder",
                        "weighted_sum_newton": "weighted_sum_newton",
                    }
                    fn_name = fn_map[method_key]
                    fn = _safe_getattr(_wrap, fn_name)
                    if fn is None:
                        raise ImportError(f"wrappers.py no tiene '{fn_name}'")
                    xopt, fopt, hist = fn(f_text, g_text, " ".join(var_names), x0)
                    # history may be list of dict
                    path_xy = _history_to_path_xy(hist)
                    table_rows = []
                    if isinstance(hist, list):
                        for i, h in enumerate(hist):
                            if isinstance(h, dict):
                                table_rows.append({
                                    "k": h.get("k", i),
                                    "x_k": h.get("x_k", ""),
                                    "x_k_str": h.get("x_k_str", h.get("x_k", "")),
                                    "f_k": h.get("f_k", ""),
                                    "grad_norm": h.get("grad_norm", ""),
                                    "alpha_k": h.get("alpha_k", ""),
                                })
                    self.status.setText(f"x*={xopt}, f*={fopt}")

                elif method_key == "md_newton_armijo":
                    if _md_linesearch is None or not hasattr(_md_linesearch, "newton_armijo"):
                        raise ImportError("multi-dimensionals/line_search.py no tiene newton_armijo")
                    xopt, fopt, log_data = _md_linesearch.newton_armijo(f_text, " ".join(var_names), x0)
                    path_xy = _history_to_path_xy(log_data)
                    if isinstance(log_data, list):
                        for i, h in enumerate(log_data):
                            if isinstance(h, dict):
                                table_rows.append({
                                    "k": h.get("k", i),
                                    "x_k": h.get("x_k", ""),
                                    "x_k_str": h.get("x_k_str", h.get("x_k", "")),
                                    "f_k": h.get("f_k", ""),
                                    "grad_norm": h.get("grad_norm", ""),
                                    "alpha_k": h.get("alpha_k", ""),
                                })
                    self.status.setText(f"x*={xopt}, f*={fopt}")

                elif method_key == "md_newton_wolfe":
                    if _md_linesearch is None or not hasattr(_md_linesearch, "newton_wolfe_step"):
                        raise ImportError("multi-dimensionals/line_search.py no tiene newton_wolfe_step")
                    xopt, fopt, log_data = _md_linesearch.newton_wolfe_step(f_text, " ".join(var_names), x0)
                    path_xy = _history_to_path_xy(log_data)
                    if isinstance(log_data, list):
                        for i, h in enumerate(log_data):
                            if isinstance(h, dict):
                                table_rows.append({
                                    "k": h.get("k", i),
                                    "x_k": h.get("x_k", ""),
                                    "x_k_str": h.get("x_k_str", h.get("x_k", "")),
                                    "f_k": h.get("f_k", ""),
                                    "grad_norm": h.get("grad_norm", ""),
                                    "alpha_k": h.get("alpha_k", ""),
                                })
                    self.status.setText(f"x*={xopt}, f*={fopt}")

                elif method_key == "md_wolfe_line_search":
                    if _md_linesearch is None or not hasattr(_md_linesearch, "wolfe_line_search"):
                        raise ImportError("multi-dimensionals/line_search.py no tiene wolfe_line_search")
                    # Here we just run and show outputs (this is a line-search primitive)
                    out = _md_linesearch.wolfe_line_search(f_text, " ".join(var_names), x0)
                    self.status.setText(f"Salida: {out}")

                elif method_key in ("md_fibonacci_search","md_local_search","md_scipy_minimize"):
                    if _md_localsearch is None:
                        raise ImportError("multi-dimensionals/localsearch.py no disponible")
                    fn_map = {
                        "md_fibonacci_search": "fibonacci_search",
                        "md_local_search": "local_search",
                        "md_scipy_minimize": "scipy_minimize",
                    }
                    fn = _safe_getattr(_md_localsearch, fn_map[method_key])
                    if fn is None:
                        raise ImportError(f"localsearch.py no tiene {fn_map[method_key]}")
                    out = fn(f_text, " ".join(var_names), x0)
                    self.status.setText(f"Salida: {out}")

                else:
                    raise ValueError("Método no reconocido.")

            # Update table
            if table_rows:
                self._fill_table(table_rows)

            # Update plots with path
            if len(var_names) >= 2:
                if path_xy and hasattr(self.canvas2d, "set_path"):
                    self.canvas2d.set_path(path_xy)
                # 3D path: z=f(x,y)
                if path_xy and hasattr(self.canvas3d, "set_path_xyz"):
                    zs = [float(f_np(x, y)) for x, y in path_xy]
                    path_xyz = [(x, y, z) for (x, y), z in zip(path_xy, zs)]
                    self.canvas3d.set_path_xyz(path_xyz)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ocurrió un error al ejecutar el método:\n{e}")
