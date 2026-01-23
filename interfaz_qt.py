# -*- coding: utf-8 -*-
import os
import sys
import math
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional

import numpy as np
import csv

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QLineEdit,
    QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QMessageBox,
    QFileDialog, QTableWidget, QTableWidgetItem, QSizePolicy, QSpacerItem
)

from busqueda_local import busqueda_local
from fibonacci import fibonacci_search
from armijo import armijo_search
from wolfe import wolfe_search

from rotacion_3d import Rotating3DCanvas
from reporte_export import ReportItem, export_reporte_excel


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

    def __init__(self, resource_path_base: str = "."):
        super().__init__()
        self.resource_base = resource_path_base

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

        # Método
        form.addWidget(QLabel("Método:"), 1, 0)
        self.metodo_menu = QComboBox()
        self.metodo_menu.addItems(["Búsqueda Local", "Fibonacci", "Armijo", "Wolfe"])
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
        self.btn_calc.clicked.connect(self.ejecutar_metodo)
        self.btn_clear = QPushButton("Limpiar")
        self.btn_clear.clicked.connect(self.limpiar)
        btn_row1.addWidget(self.btn_calc)
        btn_row1.addWidget(self.btn_clear)
        left_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.btn_csv = QPushButton("Exportar CSV")
        self.btn_csv.clicked.connect(self.exportar_csv)
        self.btn_excel = QPushButton("Exportar Excel")
        self.btn_excel.clicked.connect(self.exportar_excel)
        btn_row2.addWidget(self.btn_csv)
        btn_row2.addWidget(self.btn_excel)
        left_layout.addLayout(btn_row2)

        self.btn_exit = QPushButton("Salir")
        self.btn_exit.clicked.connect(self.close)
        left_layout.addWidget(self.btn_exit)

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
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
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

        QTableWidget#resultTable {
            background-color: rgba(0,0,0,0.25);
            color: white;
            gridline-color: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 12px;
        }
        QHeaderView::section {
            background-color: rgba(255,255,255,0.08);
            color: white;
            border: 0px;
            padding: 6px;
            font-weight: 700;
        }
        QTableWidget::item { padding: 4px; }
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
        self.tol_label.setVisible(is_fibo)
        self.tol_spin.setVisible(is_fibo)

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
            f = self._parse_function(expr)

            a_in = float(self.min_spin.value())
            b_in = float(self.max_spin.value())
            params = self._auto_params(a_in, b_in)
            a, b, x0, paso, max_iter = params["a"], params["b"], params["x0"], params["paso"], params["max_iter"]

            metodo = self.metodo_menu.currentText()
            history: List[Dict[str, Any]] = []
            resumen: Dict[str, Any] = {"metodo": metodo, "funcion": expr}

            if metodo == "Búsqueda Local":
                resumen_out, history = busqueda_local(f, x0=x0, paso=paso, max_iter=max_iter)
                resumen.update(resumen_out)

            elif metodo == "Fibonacci":
                tol = float(self.tol_spin.value())
                x_opt, f_opt, history = fibonacci_search(f, a=a, b=b, tolerance=tol, max_iter=max_iter, return_history=True)
                resumen.update({"x_opt": x_opt, "f_opt": f_opt, "a": a, "b": b, "tolerance": tol, "iter": len(history)})

            elif metodo == "Armijo":
                x_opt, f_opt, alpha, history = armijo_search(
                    f, x0=x0, alpha0=1.0, rho=0.5, c=1e-4, max_iter=max_iter, return_history=True
                )
                resumen.update({"x_opt": x_opt, "f_opt": f_opt, "alpha": alpha, "iter": len(history)})

            elif metodo == "Wolfe":
                x_opt, f_opt, alpha, history = wolfe_search(
                    f, x0=x0, alpha0=1.0, c1=1e-4, c2=0.9, max_iter=max_iter, return_history=True
                )
                resumen.update({"x_opt": x_opt, "f_opt": f_opt, "alpha": alpha, "iter": len(history)})

            else:
                raise ValueError("Método no soportado.")

            # actualizar tabla + gráfica
            self._current_expr = expr
            self._update_table(history)
            self._render_plot(metodo, f, history)

            # Guardar resultado para exportación (sin depender de pandas)
            img_path = self._save_plot_image(metodo, f, history)
            item = ReportItem(
                metodo=metodo,
                funcion=expr,
                resumen=resumen,
                tabla=history,   # lista de dicts
                image_path=img_path
            )
            self._results.setdefault(expr, {})[metodo] = {"resumen": resumen, "history": history, "item": item}

            # status
            if "x_opt" in resumen and "f_opt" in resumen:
                self.status_lbl.setText(f"✓ {metodo} → x*={resumen['x_opt']:.6g}, f(x*)={resumen['f_opt']:.6g}")
            else:
                # búsqueda local devuelve x_max/f_max
                xk = resumen.get("x_max", resumen.get("x_min", None))
                fk = resumen.get("f_max", resumen.get("f_min", None))
                if xk is not None and fk is not None:
                    self.status_lbl.setText(f"✓ {metodo} → x={xk:.6g}, f(x)={fk:.6g}")
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

        if self._canvas is None:
            self._canvas = Rotating3DCanvas(self.plot_frame, title=f"Gráfica 3D - {metodo}")
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
            export_reporte_excel(filepath, items, app_title="Optimizador de Funciones")
            self._show_msg("Éxito", "Reporte Excel exportado exitosamente.", icon=QMessageBox.Information)
        except Exception as e:
            # si falla, reportar claro (texto negro en QMessageBox por defecto)
            self._show_msg("Error", f"No se pudo exportar el Excel:\n{e}", icon=QMessageBox.Critical)

    # ---------------- Helpers ----------------
    def _show_msg(self, title: str, text: str, icon=QMessageBox.Warning):
        # QMessageBox usa estilo del sistema (texto negro normalmente),
        # para evitar letras blancas en mensajes.
        m = QMessageBox(self)
        m.setWindowTitle(title)
        m.setText(text)
        m.setIcon(icon)
        m.exec_()
