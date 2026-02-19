# interfaz_qt.py
# Interfaz principal (robusta) - NO cambia la distribución visual existente.
# Solo corrige imports/errores típicos y conecta canvases 2D/3D de forma segura.

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# ---------- Qt (PyQt6 / PyQt5) ----------
_QT = None
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import Qt
    _QT = "PyQt6"
except Exception:
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    from PyQt5.QtCore import Qt  # type: ignore
    _QT = "PyQt5"

# QtWidgets aliases (para evitar NameError por imports parciales)
QApplication = QtWidgets.QApplication
QMainWindow = QtWidgets.QMainWindow
QWidget = QtWidgets.QWidget
QVBoxLayout = QtWidgets.QVBoxLayout
QHBoxLayout = QtWidgets.QHBoxLayout
QGridLayout = QtWidgets.QGridLayout
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QComboBox = QtWidgets.QComboBox
QPushButton = QtWidgets.QPushButton
QFrame = QtWidgets.QFrame
QTableWidget = QtWidgets.QTableWidget
QTableWidgetItem = QtWidgets.QTableWidgetItem
QHeaderView = QtWidgets.QHeaderView
QDoubleSpinBox = QtWidgets.QDoubleSpinBox
QMessageBox = QtWidgets.QMessageBox
QSpacerItem = QtWidgets.QSpacerItem
QSizePolicy = QtWidgets.QSizePolicy

# ---------- Dependencias de cálculo ----------
import numpy as np

try:
    import sympy as sp
    from sympy import lambdify
except Exception:
    sp = None  # type: ignore
    lambdify = None  # type: ignore

# ---------- Canvases (respeta nombres puente si existen) ----------
Function2DCanvas = None
Rotating3DCanvas = None

def _import_canvas_2d():
    global Function2DCanvas
    # 1) puente legacy
    for modname in ("canvas_2d_estilo_fixed", "canvas_2d"):
        try:
            m = __import__(modname, fromlist=["Function2DCanvas"])
            Function2DCanvas = getattr(m, "Function2DCanvas")
            return
        except Exception:
            continue

def _import_canvas_3d():
    global Rotating3DCanvas
    for modname in ("rotacion_3d_superficie", "rotacion_3d"):
        try:
            m = __import__(modname, fromlist=["Rotating3DCanvas"])
            Rotating3DCanvas = getattr(m, "Rotating3DCanvas")
            return
        except Exception:
            continue

_import_canvas_2d()
_import_canvas_3d()

# ---------- Métodos (carga opcional; NO rompe si faltan) ----------
def _safe_import(name: str, attr: Optional[str] = None):
    try:
        m = __import__(name, fromlist=[attr] if attr else [])
        return getattr(m, attr) if attr else m
    except Exception:
        return None

armijo = _safe_import("armijo", "armijo")
wolfe = _safe_import("wolfe", "wolfe")
fibonacci = _safe_import("fibonacci", "fibonacci")
busqueda_local = _safe_import("busqueda_local", "busqueda_local")

# wrappers multidimensionales: suele vivir en multi-dimensionals/wrappers.py
_wrappers = None
def _load_wrappers():
    global _wrappers
    if _wrappers is not None:
        return _wrappers
    # intenta localizar carpeta "multi-dimensionals" relativa al proyecto
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "multi-dimensionals")
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)
    _wrappers = _safe_import("wrappers", None)
    return _wrappers

# ---------- Utilidades ----------
def _show_error(parent: QWidget, title: str, msg: str, detail: str = ""):
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(msg)
    if detail:
        box.setDetailedText(detail)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()

def _parse_vars(text: str) -> List[str]:
    # acepta "x", "x1 x2", "x1,x2"
    raw = (text or "").replace(",", " ").split()
    return [v.strip() for v in raw if v.strip()]

def _parse_x0(text: str) -> List[float]:
    # acepta "0.5", "0.5, 0.7", "0.5 0.7"
    t = (text or "").replace(";", ",").replace(" ", ",")
    parts = [p for p in t.split(",") if p.strip() != ""]
    return [float(p) for p in parts]

def _sympy_ready() -> bool:
    return sp is not None and lambdify is not None

def _make_callable(expr: str, var_names: Sequence[str]) -> Callable[..., Any]:
    if not _sympy_ready():
        raise RuntimeError("SymPy no está disponible en tu entorno. Instala sympy.")
    locals_map = {vn: sp.Symbol(vn) for vn in var_names}
    ex = sp.sympify(expr, locals=locals_map)
    syms = [locals_map[vn] for vn in var_names]
    f_np = sp.lambdify(syms, ex, modules=["numpy"])
    return f_np

def _try_eval_f(expr: str, var_names: Sequence[str], xs: np.ndarray) -> np.ndarray:
    f = _make_callable(expr, var_names)
    # xs shape: (n,) para 1D o (2, n) / etc.
    if len(var_names) == 1:
        return np.asarray(f(xs), dtype=float)
    raise ValueError("Evaluación 1D solo soporta 1 variable.")

def _make_1d_grid(xmin: float, xmax: float, n: int = 400) -> np.ndarray:
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return np.linspace(xmin, xmax, int(max(50, n)))

# ---------- UI ----------
class InterfazOptimizacion(QMainWindow):
    """
    Nota: este archivo intenta ser tolerante a errores (imports/métodos faltantes)
    y NO modifica tu layout visual (fondos/QSS). Si tu proyecto ya aplica un QSS
    externo/fondo, se respeta: este archivo solo evita que reviente por errores.
    """

    def __init__(self, resource_path: Optional[str] = None):
        super().__init__()
        self.resource_path = resource_path
        self.setWindowTitle("Optimizador de Funciones")

        # widgets principales (sin tocar tu QSS, si existe en tu proyecto)
        self.central = QWidget()
        self.setCentralWidget(self.central)

        self.root_layout = QHBoxLayout(self.central)
        self.root_layout.setContentsMargins(20, 20, 20, 20)
        self.root_layout.setSpacing(18)

        # Panel izquierdo (control)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setSpacing(12)

        # Panel derecho (dashboard)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setSpacing(12)

        self.root_layout.addWidget(self.left_panel, 1)
        self.root_layout.addWidget(self.right_panel, 2)

        self._build_left()
        self._build_right()

        self._wire()
        self._reset_dashboard_placeholders()

    # ---------- Construcción UI ----------
    def _build_left(self):
        title = QLabel("Panel de control")
        title.setObjectName("panelTitle")
        self.left_layout.addWidget(title)

        # Método
        row = QWidget()
        row_l = QVBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("Método:")
        self.cmb_metodo = QComboBox()
        self.cmb_metodo.setObjectName("methodCombo")
        row_l.addWidget(lbl)
        row_l.addWidget(self.cmb_metodo)
        self.left_layout.addWidget(row)

        # Campos
        self.edt_fx = self._make_labeled_edit("Función f(x):", "Ej: (x1-4)**2 + (x2-4)**2")
        self.edt_gx = self._make_labeled_edit("Restricción g(x):", "Ej: x1**2 + x2**2 - 1")
        self.edt_vars = self._make_labeled_edit("Variables:", "Ej: x1 x2")
        self.edt_x0 = self._make_labeled_edit("x0 (coma):", "Ej: 0.5, 0.5")

        self.left_layout.addWidget(self.edt_fx["w"])
        self.left_layout.addWidget(self.edt_gx["w"])
        self.left_layout.addWidget(self.edt_vars["w"])
        self.left_layout.addWidget(self.edt_x0["w"])

        # Rango 1D
        range_row = QWidget()
        range_l = QHBoxLayout(range_row)
        range_l.setContentsMargins(0, 0, 0, 0)
        self.spin_min = QDoubleSpinBox()
        self.spin_max = QDoubleSpinBox()
        for spn in (self.spin_min, self.spin_max):
            spn.setDecimals(6)
            spn.setRange(-1e9, 1e9)
        self.spin_min.setValue(0.0)
        self.spin_max.setValue(5.0)

        range_l.addWidget(QLabel("mínimo:"))
        range_l.addWidget(self.spin_min)
        range_l.addSpacing(10)
        range_l.addWidget(QLabel("máximo:"))
        range_l.addWidget(self.spin_max)

        self.left_layout.addWidget(range_row)

        # Botones
        btn_row1 = QWidget()
        btn_l1 = QHBoxLayout(btn_row1)
        btn_l1.setContentsMargins(0, 0, 0, 0)
        self.btn_calc = QPushButton("Calcular")
        self.btn_clear = QPushButton("Limpiar")
        btn_l1.addWidget(self.btn_calc)
        btn_l1.addWidget(self.btn_clear)

        btn_row2 = QWidget()
        btn_l2 = QHBoxLayout(btn_row2)
        btn_l2.setContentsMargins(0, 0, 0, 0)
        self.btn_export = QPushButton("Exportar")
        self.btn_exit = QPushButton("Salir")
        btn_l2.addWidget(self.btn_export)
        btn_l2.addWidget(self.btn_exit)

        self.left_layout.addSpacing(10)
        self.left_layout.addWidget(btn_row1)
        self.left_layout.addWidget(btn_row2)
        self.left_layout.addStretch(1)

        self._populate_methods()

    def _make_labeled_edit(self, label: str, placeholder: str):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(label)
        edt = QLineEdit()
        edt.setPlaceholderText(placeholder)
        l.addWidget(lab)
        l.addWidget(edt)
        return {"w": w, "label": lab, "edit": edt}

    def _build_right(self):
        title = QLabel("Dashboard")
        title.setObjectName("dashTitle")
        self.right_layout.addWidget(title)

        # Tabla
        self.table = QTableWidget(0, 0)
        self.table.setObjectName("dashTable")
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.right_layout.addWidget(self.table, 1)

        # Área gráficas (stack vertical)
        self.plot_container = QWidget()
        pc_l = QVBoxLayout(self.plot_container)
        pc_l.setContentsMargins(0, 0, 0, 0)
        pc_l.setSpacing(10)

        # 2D frame
        self.frame_2d = QFrame()
        self.frame_2d.setObjectName("frame2d")
        f2_l = QVBoxLayout(self.frame_2d)
        f2_l.setContentsMargins(0, 0, 0, 0)
        self.lbl_2d_placeholder = QLabel("Ejecute un método para ver la gráfica 2D.")
        self.lbl_2d_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter if _QT=="PyQt6" else Qt.AlignCenter)
        f2_l.addWidget(self.lbl_2d_placeholder)

        # 3D frame
        self.frame_3d = QFrame()
        self.frame_3d.setObjectName("frame3d")
        f3_l = QVBoxLayout(self.frame_3d)
        f3_l.setContentsMargins(0, 0, 0, 0)
        self.lbl_3d_placeholder = QLabel("Ejecute un método para ver la gráfica 3D.")
        self.lbl_3d_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter if _QT=="PyQt6" else Qt.AlignCenter)
        f3_l.addWidget(self.lbl_3d_placeholder)

        pc_l.addWidget(self.frame_2d, 1)
        pc_l.addWidget(self.frame_3d, 1)

        self.right_layout.addWidget(self.plot_container, 2)

        # Instanciar canvases SIN 'title' kw (evita tu error)
        self.canvas2d = None
        self.canvas3d = None
        if Function2DCanvas is not None:
            try:
                self.canvas2d = Function2DCanvas(self.frame_2d)
            except Exception:
                self.canvas2d = None
        if Rotating3DCanvas is not None:
            try:
                self.canvas3d = Rotating3DCanvas(self.frame_3d)
            except Exception:
                self.canvas3d = None

        # Solo se muestran cuando hay cálculo
        if self.canvas2d is not None:
            self.canvas2d.setVisible(False)
        if self.canvas3d is not None:
            self.canvas3d.setVisible(False)

    def _wire(self):
        self.btn_calc.clicked.connect(self.ejecutar_metodo)  # type: ignore
        self.btn_clear.clicked.connect(self.limpiar_todo)  # type: ignore
        self.btn_exit.clicked.connect(self.confirmar_salida)  # type: ignore
        self.btn_export.clicked.connect(self.exportar_csv)  # type: ignore

    # ---------- Métodos ----------
    def _populate_methods(self):
        # Mantén nombres amigables; no rompe si algo no existe.
        items = [
            "Armijo (1D)",
            "Wolfe (1D)",
            "Fibonacci (1D)",
            "Búsqueda Local",
            "MD: Penalización (Newton)",
            "MD: Barrera (Newton)",
        ]
        self.cmb_metodo.clear()
        self.cmb_metodo.addItems(items)

    def _reset_dashboard_placeholders(self):
        # vacía tabla
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        # ocultar canvases, mostrar placeholders
        self.lbl_2d_placeholder.setVisible(True)
        self.lbl_3d_placeholder.setVisible(True)
        if self.canvas2d is not None:
            self.canvas2d.setVisible(False)
        if self.canvas3d is not None:
            self.canvas3d.setVisible(False)

    def limpiar_todo(self):
        # limpiar campos panel (recuadro rojo en tu imagen)
        for edt in (self.edt_fx["edit"], self.edt_gx["edit"], self.edt_vars["edit"], self.edt_x0["edit"]):
            edt.clear()
        self.spin_min.setValue(0.0)
        self.spin_max.setValue(5.0)
        self._reset_dashboard_placeholders()

    def confirmar_salida(self):
        res = QMessageBox.question(
            self,
            "Salir",
            "¿Desea salir de la aplicación?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if res == QMessageBox.StandardButton.Yes:
            self.close()

    # ---------- Acciones ----------
    def ejecutar_metodo(self):
        try:
            metodo = self.cmb_metodo.currentText()
            fx = self.edt_fx["edit"].text().strip()
            gx = self.edt_gx["edit"].text().strip()
            vars_ = _parse_vars(self.edt_vars["edit"].text().strip())
            x0 = _parse_x0(self.edt_x0["edit"].text().strip()) if self.edt_x0["edit"].text().strip() else []

            if not fx:
                raise ValueError("Debes ingresar la función f(x).")
            if not vars_:
                # fallback: inferir por método
                vars_ = ["x"] if ("(1D)" in metodo or "Búsqueda Local" in metodo) else ["x1", "x2"]

            # Determina 1D vs 2D (para gráficos)
            is_1d = (len(vars_) == 1)

            # Ejecuta método (si disponible). Si no, igual grafica f(x) con el/los puntos.
            hist = self._run_method(metodo, fx, gx, vars_, x0)

            # Update tabla + gráficas
            self._render_table(hist)
            self._render_plots(metodo, fx, vars_, hist, x0)

        except Exception as e:
            _show_error(self, "Error", f"Ocurrió un error al ejecutar el método:\n{e}", traceback.format_exc())

    def _run_method(self, metodo: str, fx: str, gx: str, vars_: List[str], x0: List[float]):
        # Historial estándar: lista de dicts
        hist: List[Dict[str, Any]] = []

        # -------- 1D --------
        if "Armijo" in metodo and armijo is not None:
            # La firma varía; intentamos compat.
            try:
                res = armijo(fx, self.spin_min.value(), self.spin_max.value())
            except Exception:
                res = armijo(fx)
            hist = self._coerce_hist(res)
            return hist

        if "Wolfe" in metodo and wolfe is not None:
            try:
                res = wolfe(fx, self.spin_min.value(), self.spin_max.value())
            except Exception:
                res = wolfe(fx)
            hist = self._coerce_hist(res)
            return hist

        if "Fibonacci" in metodo and fibonacci is not None:
            try:
                res = fibonacci(fx, self.spin_min.value(), self.spin_max.value())
            except Exception:
                res = fibonacci(fx)
            hist = self._coerce_hist(res)
            return hist

        if "Búsqueda Local" in metodo and busqueda_local is not None:
            try:
                res = busqueda_local(fx, self.spin_min.value(), self.spin_max.value())
            except Exception:
                res = busqueda_local(fx)
            hist = self._coerce_hist(res)
            return hist

        # -------- MD (wrappers) --------
        if metodo.startswith("MD:"):
            w = _load_wrappers()
            if w is None:
                raise RuntimeError("No se encontró multi-dimensionals/wrappers.py en tu estructura.")
            # Preferencias
            fn = None
            if "Penalización" in metodo:
                fn = getattr(w, "penalty_method_newton", None)
            elif "Barrera" in metodo:
                fn = getattr(w, "barrier_method_newton", None)

            if fn is None:
                raise RuntimeError(f"El método {metodo} no está implementado en wrappers.py")

            # Intento de llamada flexible:
            # Muchos wrappers usan (f_str, g_str, x0, var_names) o similar.
            try:
                res = fn(fx, gx, x0, vars_)
            except Exception:
                try:
                    res = fn(fx, gx, x0)
                except Exception:
                    res = fn(fx, gx)
            hist = self._coerce_hist(res)
            return hist

        # Fallback: sin método; devuelve un "historial" mínimo con x0 como trayectoria
        if x0:
            if len(vars_) == 1:
                hist = [{"k": 0, "x": x0[0]}]
            else:
                hist = [{"k": 0, "x": x0}]
        return hist

    def _coerce_hist(self, res: Any) -> List[Dict[str, Any]]:
        # Normaliza formatos comunes (list[dict], dict, pandas, etc.)
        if res is None:
            return []
        if isinstance(res, list):
            if res and isinstance(res[0], dict):
                return res
            # lista de tuplas -> dicts
            out = []
            for i, row in enumerate(res):
                if isinstance(row, dict):
                    out.append(row)
                elif isinstance(row, (list, tuple)):
                    out.append({"k": i, "row": row})
                else:
                    out.append({"k": i, "value": row})
            return out
        if isinstance(res, dict):
            # puede ser {"history":[...]}
            if "history" in res and isinstance(res["history"], list):
                return self._coerce_hist(res["history"])
            return [res]
        # pandas DataFrame
        try:
            import pandas as pd  # type: ignore
            if isinstance(res, pd.DataFrame):
                return res.to_dict(orient="records")
        except Exception:
            pass
        return [{"value": res}]

    def _render_table(self, hist: List[Dict[str, Any]]):
        if not hist:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        # columns union
        cols = []
        for r in hist:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(hist))
        for i, r in enumerate(hist):
            for j, c in enumerate(cols):
                val = r.get(c, "")
                self.table.setItem(i, j, QTableWidgetItem(str(val)))
        try:
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)  # type: ignore
        except Exception:
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # type: ignore

    def _render_plots(self, metodo: str, fx: str, vars_: List[str], hist: List[Dict[str, Any]], x0: List[float]):
        # Trajectory extraction
        traj = self._extract_traj(vars_, hist, x0)

        # 2D
        if self.canvas2d is not None:
            try:
                self.lbl_2d_placeholder.setVisible(False)
                self.canvas2d.setVisible(True)
                if len(vars_) == 1:
                    x = _make_1d_grid(self.spin_min.value(), self.spin_max.value(), 400)
                    y = _try_eval_f(fx, vars_, x)
                    tx = traj[0] if traj else np.array([])
                    ty = _try_eval_f(fx, vars_, tx) if tx.size else np.array([])
                    (
                    self.canvas2d.animate_1d(x, y, tx, ty, title=f"Gráfica 2D - {metodo}")
                    if hasattr(self.canvas2d, 'animate_1d') and tx.size>1 else
                    self.canvas2d.plot_1d(x, y, tx, ty, title=f"Gráfica 2D - {metodo}")
                )
                else:
                    # 2D contour (x1,x2) si hay 2 variables
                    xmin, xmax = self.spin_min.value(), self.spin_max.value()
                    grid = np.linspace(min(xmin,xmax), max(xmin,xmax), 140)
                    X1, X2 = np.meshgrid(grid, grid)
                    f = _make_callable(fx, vars_[:2])
                    Z = np.asarray(f(X1, X2), dtype=float)
                    path_x = traj[0] if len(traj) > 0 else np.array([])
                    path_y = traj[1] if len(traj) > 1 else np.array([])
                    (
                    self.canvas2d.animate_2d_contour(X1, X2, Z, path_x, path_y, title=f"Gráfica 2D - {metodo}")
                    if hasattr(self.canvas2d, 'animate_2d_contour') and path_x.size>1 else
                    self.canvas2d.plot_2d_contour(X1, X2, Z, path_x, path_y, title=f"Gráfica 2D - {metodo}")
                )
            except Exception:
                self.lbl_2d_placeholder.setVisible(True)
                self.canvas2d.setVisible(False)

        # 3D
        if self.canvas3d is not None:
            try:
                self.lbl_3d_placeholder.setVisible(False)
                self.canvas3d.setVisible(True)
                if len(vars_) == 1:
                    x = _make_1d_grid(self.spin_min.value(), self.spin_max.value(), 300)
                    y = _try_eval_f(fx, vars_, x)
                    tx = traj[0] if traj else np.array([])
                    ty = _try_eval_f(fx, vars_, tx) if tx.size else np.array([])
                    # 3D simple: (iter, x, f(x))
                    it = np.arange(x.size)
                    self.canvas3d.set_curve3d(it, x, y, title=f"Gráfica 3D - {metodo}", path_it=np.arange(tx.size), path_x=tx, path_z=ty)
                else:
                    xmin, xmax = self.spin_min.value(), self.spin_max.value()
                    grid = np.linspace(min(xmin,xmax), max(xmin,xmax), 80)
                    X1, X2 = np.meshgrid(grid, grid)
                    f = _make_callable(fx, vars_[:2])
                    Z = np.asarray(f(X1, X2), dtype=float)
                    path_x = traj[0] if len(traj) > 0 else np.array([])
                    path_y = traj[1] if len(traj) > 1 else np.array([])
                    path_z = np.asarray(f(path_x, path_y), dtype=float) if (path_x.size and path_y.size) else np.array([])
                    (
                    self.canvas3d.animate_surface_and_path(X1, X2, Z, path_x, path_y, path_z, title=f"Gráfica 3D - {metodo}", rotate=True)
                    if hasattr(self.canvas3d, 'animate_surface_and_path') and path_x.size>1 else
                    self.canvas3d.set_surface_and_path(X1, X2, Z, path_x, path_y, path_z, title=f"Gráfica 3D - {metodo}")
                )
            except Exception:
                self.lbl_3d_placeholder.setVisible(True)
                self.canvas3d.setVisible(False)

    def _extract_traj(self, vars_: List[str], hist: List[Dict[str, Any]], x0: List[float]) -> List[np.ndarray]:
        if not hist and x0:
            if len(vars_) == 1:
                return [np.asarray([x0[0]], dtype=float)]
            if len(vars_) >= 2 and len(x0) >= 2:
                return [np.asarray([x0[0]]), np.asarray([x0[1]])]
        # intenta keys comunes
        if len(vars_) == 1:
            xs: List[float] = []
            for r in hist:
                if "x" in r and isinstance(r["x"], (int, float, np.number)):
                    xs.append(float(r["x"]))
                elif "x_k" in r:
                    try:
                        xs.append(float(r["x_k"]))
                    except Exception:
                        pass
            return [np.asarray(xs, dtype=float)] if xs else []
        # multi: busca 'x' como lista/tuple
        x1, x2 = [], []
        for r in hist:
            v = r.get("x") or r.get("x_k")
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    x1.append(float(v[0])); x2.append(float(v[1]))
                except Exception:
                    pass
        if x1 and x2:
            return [np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)]
        # fallback x0
        if len(x0) >= 2:
            return [np.asarray([x0[0]]), np.asarray([x0[1]])]
        return []

    # ---------- Export ----------
    def exportar_csv(self):
        if self.table.rowCount() == 0 or self.table.columnCount() == 0:
            QMessageBox.information(self, "Exportar", "No hay datos en la tabla para exportar.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Guardar CSV", "reporte.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            cols = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
            rows = []
            for r in range(self.table.rowCount()):
                row = []
                for c in range(self.table.columnCount()):
                    it = self.table.item(r, c)
                    row.append("" if it is None else it.text())
                rows.append(row)
            import csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(cols)
                w.writerows(rows)
            QMessageBox.information(self, "Exportar", f"CSV guardado en:\n{path}")
        except Exception as e:
            _show_error(self, "Error", f"No se pudo exportar CSV: {e}", traceback.format_exc())
