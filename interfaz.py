import sys
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QComboBox, QFormLayout,
    QFrame, QGraphicsDropShadowEffect,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QStackedWidget
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from busqueda_local import busqueda_local
from fibonacci import metodo_fibonacci
from armijo import armijo_search
from wolfe import wolfe_search



def resource_path(relative_path: str) -> str:
    """Rutas correctas en desarrollo y en .exe (PyInstaller)."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

class InterfazOptimizacion(QWidget):
    def __init__(self):
        super().__init__()
        # Icono ventana (barra superior)
        icon_ico = resource_path(r"Menu\\WindowsIcon-min.ico")
        if os.path.exists(icon_ico):
            self.setWindowIcon(QIcon(icon_ico))
        self.setWindowTitle("Optimizador de Funciones - Proyecto Joe Verde")
        self.setGeometry(200, 100, 1200, 750)

        # =========================
        # Fondo (imagen) + overlays
        # =========================
        self.background_label = QLabel(self)
        if hasattr(self, 'set_background_image'):

            self.set_background_image()
        self.overlay_dark = QLabel(self)
        self.overlay_dark.setGeometry(0, 0, self.width(), self.height())
        self.overlay_dark.setStyleSheet("background-color: rgba(0, 0, 0, 55);")
        self.overlay_dark.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.overlay_light = QLabel(self)
        self.overlay_light.setGeometry(0, 0, self.width(), self.height())
        self.overlay_light.setStyleSheet("background-color: rgba(255, 255, 255, 18);")
        self.overlay_light.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.background_label.lower()
        self.overlay_dark.raise_()
        self.overlay_light.raise_()

        # =========================
        # Layout principal
        # =========================
        self.root_layout = QHBoxLayout(self)
        self.root_layout.setContentsMargins(24, 18, 24, 24)
        self.root_layout.setSpacing(18)

        self.setStyleSheet(self._global_stylesheet())


        # Asegura que menús/contexto (click derecho) hereden el estilo global
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(self._global_stylesheet())
        left = self._build_left_panel()
        left.setFixedWidth(450)

        right = self._build_right_dashboard()

        self.root_layout.addWidget(left)
        self.root_layout.addWidget(right, 1)

        # Estado inicial: dashboard vacío
        self._set_dashboard_state(active=False)

    def _global_stylesheet(self) -> str:
        return """
        QWidget {
            font-family: 'Segoe UI';
            font-size: 14px;
            color: rgba(255,255,255,250);
        }

        QFrame#LeftPanel {
            background-color: rgba(12, 24, 42, 200);
            border: 1px solid rgba(255,255,255,28);
            border-radius: 18px;
        }

        QLabel#SectionTitle { font-size: 16px; font-weight: 900; color: rgba(255,255,255,250); }
        QLabel#MainTitle { font-size: 28px; font-weight: 900; color: rgba(255,255,255,250); }

        QLabel { font-weight: 700; color: rgba(255,255,255,245); }

        QLineEdit, QComboBox {
            background-color: rgba(255,255,255,22);
            border: 1px solid rgba(255,255,255,30);
            border-radius: 12px;
            padding: 11px 12px;
            color: rgba(255,255,255,250);
            font-weight: 700;
        }

        QLineEdit::placeholder { color: rgba(255,255,255,170); font-weight: 600; }

        QComboBox QAbstractItemView {
            background-color: rgba(12, 24, 42, 240);
            color: rgba(255,255,255,245);
            selection-background-color: rgba(0, 170, 255, 220);
            selection-color: white;
            border: 1px solid rgba(255,255,255,18);
        }

        QPushButton[action="primary"] {
            background-color: rgba(0, 170, 255, 230);
            border: none;
            border-radius: 12px;
            padding: 13px;
            font-weight: 900;
            color: white;
        }
        QPushButton[action="primary"]:hover { background-color: rgba(0, 170, 255, 255); }

        QTableWidget {
            background-color: rgba(12, 24, 42, 180);
            border: 1px solid rgba(255,255,255,20);
            border-radius: 14px;
            gridline-color: rgba(255,255,255,10);
            color: rgba(255,255,255,245);
        }
        QHeaderView::section {
            background-color: rgba(12, 24, 42, 235);
            color: rgba(255,255,255,255);
            border: none;
            border-bottom: 1px solid rgba(255,255,255,45);
            padding: 10px 8px;
            font-weight: 900;
        }

        QTableCornerButton::section {
            background-color: rgba(12, 24, 42, 235);
            border: none;
            border-bottom: 1px solid rgba(255,255,255,45);
        }

        /* ===== FIX: QMessageBox LEGIBLE ===== */
        QMessageBox {
            background-color: white;
        }
        QMessageBox QLabel {
            color: #000000;
            font-size: 13px;
            font-weight: 600;
        }
        QMessageBox QPushButton {
            color: #000000;
            font-weight: 700;
            padding: 6px 14px;
        }

        /* ===== FIX: QMenu (click derecho) LEGIBLE ===== */
        QMenu {
            background-color: white;
            color: #000000;
            border: 1px solid rgba(0,0,0,40);
        }
        QMenu::item {
            padding: 6px 18px;
            color: #000000;
            font-weight: 600;
        }
        QMenu::item:selected {
            background-color: rgba(0,170,255,60);
            color: #000000;
        }
        QMenu::separator {
            height: 1px;
            background: rgba(0,0,0,30);
            margin: 4px 8px;
        }
        """

    # =========================
    # Panel izquierdo
    # =========================
    def _build_left_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("LeftPanel")

        shadow = QGraphicsDropShadowEffect(panel)
        shadow.setBlurRadius(32)
        shadow.setOffset(0, 12)
        shadow.setColor(Qt.black)
        panel.setGraphicsEffect(shadow)

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(16)

        title = QLabel("Parámetros")
        title.setObjectName("SectionTitle")
        lay.addWidget(title)

        form = QFormLayout()
        form.setVerticalSpacing(14)
        form.setHorizontalSpacing(14)

        self.metodo_menu = QComboBox()
        self.metodo_menu.addItem("Seleccione...")
        self.metodo_menu.addItems(["Búsqueda Local", "Fibonacci", "Armijo", "Wolfe"])
        self.metodo_menu.setCurrentIndex(0)
        try:
            self.metodo_menu.model().item(0).setEnabled(False)
        except Exception:
            pass
        self.metodo_menu.currentIndexChanged.connect(self.on_metodo_change)
        form.addRow("Método", self.metodo_menu)

        self.func_input = QLineEdit()
        self.func_input.setPlaceholderText("Ej: -(x-3)**2 + 10")
        form.addRow("Función f(x)", self.func_input)

        self.xmin_input = QLineEdit()
        self.xmin_input.setPlaceholderText("Desde (x_min) o x0")

        self.xmax_input = QLineEdit()
        self.xmax_input.setPlaceholderText("Hasta (x_max) o (opcional) b")

        range_row = QHBoxLayout()
        range_row.setSpacing(10)
        range_row.addWidget(self.xmin_input)
        range_row.addWidget(self.xmax_input)
        form.addRow("Rango / parámetros", range_row)

        # Tolerancia: SOLO Fibonacci y VACÍA (sin "Seleccione")
        self.tol_label = QLabel("Tolerancia")
        self.tol_input = QLineEdit()
        form.addRow(self.tol_label, self.tol_input)

        lay.addLayout(form)

        btns = QHBoxLayout()
        btns.setSpacing(12)

        self.btn_calc = QPushButton("Calcular")
        self.btn_calc.setProperty("action", "primary")
        self.btn_calc.clicked.connect(self.ejecutar_metodo)

        self.btn_clear = QPushButton("Limpiar")
        self.btn_clear.setProperty("action", "primary")
        self.btn_clear.clicked.connect(self.limpiar)

        btns.addWidget(self.btn_calc)
        btns.addWidget(self.btn_clear)
        lay.addLayout(btns)

        self.resultado_label = QLabel("")
        self.resultado_label.setWordWrap(True)
        lay.addWidget(self.resultado_label)

        self._show_tolerancia(False)
        return panel

    def _show_tolerancia(self, show: bool):
        self.tol_label.setVisible(show)
        self.tol_input.setVisible(show)
        if not show:
            self.tol_input.clear()

    # =========================
    # Dashboard (vacío hasta Calcular)
    # =========================
    def _build_right_dashboard(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self.dashboard_stack = QStackedWidget()
        self.page_empty = self._build_dashboard_empty_page()
        self.page_results = self._build_dashboard_results_page()

        self.dashboard_stack.addWidget(self.page_empty)   # 0
        self.dashboard_stack.addWidget(self.page_results) # 1

        lay.addWidget(self.dashboard_stack)
        return wrap

    def _build_dashboard_empty_page(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(10)

        title = QLabel("OPTIMIZADOR DE FUNCIONES")
        title.setObjectName("MainTitle")
        l.addWidget(title)
        l.addStretch(1)
        return w

    def _build_dashboard_results_page(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(12)

        title = QLabel("OPTIMIZADOR DE FUNCIONES")
        title.setObjectName("MainTitle")
        l.addWidget(title)

        self.table = QTableWidget()
        self.table.setColumnCount(0)
        self.table.setRowCount(0)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        l.addWidget(self.table, 1)

        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        l.addWidget(self.canvas, 2)

        return w

    def _set_dashboard_state(self, active: bool):
        self.dashboard_stack.setCurrentIndex(1 if active else 0)
        if not active:
            if hasattr(self, "table"):
                self.table.setRowCount(0)
                self.table.setColumnCount(0)
            if hasattr(self, "fig"):
                self.fig.clear()
                self.canvas.draw()

    # =========================
    # Mensajes legibles
    # =========================
    def _show_warning(self, title: str, message: str, details: str = ""):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        if details:
            msg.setInformativeText(details)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    # =========================
    # Tabla y gráfica
    # =========================
    def _populate_table_from_history(self, history):
        if not history:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        cols = list(history[0].keys())
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(history))

        for r, row in enumerate(history):
            for c, k in enumerate(cols):
                val = row.get(k, "")
                txt = f"{val:.6g}" if isinstance(val, float) else str(val)
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)

    def _plot_no_data(self, title: str, message: str):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_title(title)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, transform=ax.transAxes)
        self.canvas.draw()

    def _plot_3d_history(self, history, metodo, funcion, x_opt=None):
        if not history:
            self._plot_no_data(
                f"Gráfica 3D - {metodo}",
                "No hay iteraciones para graficar.\n\n"
                "Sugerencia: use una tolerancia más pequeña\n"
                "para generar más iteraciones."
            )
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="3d")

        it, xs, fs = [], [], []

        for h in history:
            if "x" in h and "f(x)" in h:
                x = float(h["x"]); f = float(h["f(x)"])
            elif "x_new" in h and "f_new" in h:
                x = float(h["x_new"]); f = float(h["f_new"])
            elif "x1" in h and "f1" in h:
                x = float(h["x1"]); f = float(h["f1"])
            else:
                continue

            it.append(int(h.get("iter", len(it) + 1)))
            xs.append(x)
            fs.append(f)

        if len(it) >= 2:
            ax.plot(it, xs, fs)
            ax.scatter(it, xs, fs, s=18)
        else:
            ax.scatter(it, xs, fs, s=30)

        if x_opt is not None and it:
            try:
                fopt = funcion(x_opt)
                ax.scatter([max(it) + 1], [x_opt], [fopt], s=60)
            except Exception:
                pass

        ax.set_title(f"Gráfica 3D - {metodo}")
        ax.set_xlabel("Iteración")
        ax.set_ylabel("x")
        ax.set_zlabel("f(x)")

        # Ajuste para que la gráfica ocupe casi todo el recuadro (dejando espacio al título)
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.90)

        # Habilitar arrastre de puntos (mantiene la vista 3D)
        self._enable_3d_point_drag(ax, it, xs, funcion)

        self.canvas.draw()



    # =========================
    # Interacción: arrastrar puntos en la gráfica 3D
    # =========================
    def _enable_3d_point_drag(self, ax, it, xs, funcion):
        """Permite seleccionar (click) y arrastrar puntos en la gráfica 3D.

        Limitaciones prácticas de Matplotlib 3D: el arrastre se mapea de forma estable a cambios
        en el eje Y (x) manteniendo fija la iteración (eje X). Al mover x se recalcula f(x).
        """
        # Guardar referencias para el arrastre
        self._drag3d = {
            "ax": ax,
            "it": list(it),
            "xs": list(xs),
            "func": funcion,
            "active": False,
            "idx": None,
            "press": None,
            "sc": None,
            "cid": getattr(self, "_drag3d_cids", None),
        }

        # Dibujar puntos "pickeables"
        # (volvemos a crear el scatter con picker para que sea seleccionable)
        ax.collections.clear()
        fs = [float(funcion(x)) for x in self._drag3d["xs"]]
        sc = ax.scatter(self._drag3d["it"], self._drag3d["xs"], fs, s=30, picker=8)
        self._drag3d["sc"] = sc

        # Conectar eventos una sola vez por canvas
        if not hasattr(self, "_drag3d_cids"):
            self._drag3d_cids = []
            self._drag3d_cids.append(self.canvas.mpl_connect("button_press_event", self._on_drag3d_press))
            self._drag3d_cids.append(self.canvas.mpl_connect("button_release_event", self._on_drag3d_release))
            self._drag3d_cids.append(self.canvas.mpl_connect("motion_notify_event", self._on_drag3d_motion))
            self._drag3d_cids.append(self.canvas.mpl_connect("pick_event", self._on_drag3d_pick))

    def _on_drag3d_pick(self, event):
        if not hasattr(self, "_drag3d") or self._drag3d is None:
            return
        if event.artist != self._drag3d.get("sc"):
            return
        ind = getattr(event, "ind", None)
        if not ind:
            return
        self._drag3d["idx"] = int(ind[0])

    def _on_drag3d_press(self, event):
        if not hasattr(self, "_drag3d") or self._drag3d is None:
            return
        if event.inaxes != self._drag3d.get("ax"):
            return
        if self._drag3d.get("idx") is None:
            return
        self._drag3d["active"] = True
        self._drag3d["press"] = (event.x, event.y)

    def _on_drag3d_release(self, event):
        if not hasattr(self, "_drag3d") or self._drag3d is None:
            return
        self._drag3d["active"] = False
        self._drag3d["press"] = None
        self._drag3d["idx"] = None

    def _on_drag3d_motion(self, event):
        if not hasattr(self, "_drag3d") or self._drag3d is None:
            return
        if not self._drag3d.get("active"):
            return
        if event.inaxes != self._drag3d.get("ax"):
            return
        if self._drag3d.get("press") is None:
            return

        ax = self._drag3d["ax"]
        idx = self._drag3d.get("idx")
        if idx is None:
            return

        x0, y0 = self._drag3d["press"]
        dx = (event.x - x0)
        dy = (event.y - y0)

        # Mapear movimiento vertical del mouse a cambios en el eje y (x)
        y_min, y_max = ax.get_ylim()
        span = (y_max - y_min) if (y_max - y_min) != 0 else 1.0

        # Sensibilidad ajustable: píxeles -> unidades del eje
        # 250 px ~ 1 span completo (suave y controlable)
        delta_x = (-dy / 250.0) * span

        new_x = self._drag3d["xs"][idx] + delta_x

        # Clamp a límites actuales del eje para evitar saltos raros
        new_x = max(min(new_x, y_max), y_min)

        self._drag3d["xs"][idx] = new_x

        # Recalcular f(x)
        try:
            new_f = float(self._drag3d["func"](new_x))
        except Exception:
            return

        # Actualizar offsets del scatter (mplot3d usa atributos privados, pero es estable)
        sc = self._drag3d["sc"]
        sc._offsets3d = (self._drag3d["it"], self._drag3d["xs"], [float(self._drag3d["func"](x)) for x in self._drag3d["xs"]])

        self.canvas.draw_idle()

        # Actualizar punto de referencia para arrastre continuo
        self._drag3d["press"] = (event.x, event.y)

        # =========================
        # Eventos
        # =========================
        def on_metodo_change(self, index: int):
            if index == 0:
                self._show_tolerancia(False)
                return
            metodo = self.metodo_menu.currentText()
            self._show_tolerancia(metodo == "Fibonacci")

        # =========================
        # Ejecutar
        # =========================
        def ejecutar_metodo(self):
            if self.metodo_menu.currentIndex() == 0:
                QMessageBox.warning(self, "Aviso", "Por favor seleccione un método.")
                return

            funcion_str = self.func_input.text().strip()
            if not funcion_str:
                QMessageBox.warning(self, "Aviso", "Ingrese una función f(x).")
                return

            try:
                funcion = lambda x: eval(funcion_str, {"x": x, "np": np})
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Función inválida:\n{e}")
                return

            metodo = self.metodo_menu.currentText()

            # Activar dashboard SOLO al calcular
            self._set_dashboard_state(active=True)

            try:
                if metodo == "Fibonacci":
                    a = float(self.xmin_input.text())
                    b = float(self.xmax_input.text())

                    if not self.tol_input.text().strip():
                        QMessageBox.warning(self, "Aviso", "Ingrese tolerancia para Fibonacci.")
                        self._set_dashboard_state(active=False)
                        return

                    tol = float(self.tol_input.text())

                    res = metodo_fibonacci(funcion, a, b, tolerancia=tol, max_n=200, return_history=True)

                    if len(res) == 8:
                        x_opt, f_opt, iters, tipo, dist, history, limit_reached, target = res
                    else:
                        x_opt, f_opt, iters, tipo, dist, history = res
                        limit_reached, target = False, None

                    self.resultado_label.setText(
                        f"✅ {tipo}: x={x_opt:.6g}, f={f_opt:.6g}, iter={iters}, intervalo={dist:.6g}"
                    )
                    self._populate_table_from_history(history)
                    self._plot_3d_history(history, metodo, funcion, x_opt=x_opt)

                    if iters == 0 and not history:
                        self._show_warning(
                            "Sin iteraciones",
                            "La tolerancia es demasiado grande para este rango.",
                            (
                                "El método no necesitó iterar y devolvió el punto medio del intervalo.\n\n"
                                f"Rango: [{a}, {b}]  (L={abs(b-a)})\n"
                                f"Tolerancia: {tol}\n\n"
                                "Sugerencia: use una tolerancia más pequeña (ej: 0.1, 0.01, 0.001)."
                            )
                        )

                    if limit_reached:
                        self._show_warning(
                            "Límite alcanzado",
                            "Se alcanzó el límite máximo de iteraciones para esta tolerancia.",
                            (
                                "No existen más iteraciones disponibles para seguir refinando.\n\n"
                                f"Tolerancia solicitada: {tol}\n"
                                f"Intervalo final alcanzado: {dist:.6g}\n"
                                f"Límite interno (max_n): 200\n"
                                + (f"Objetivo (L/tolerancia): {target:.6g}" if target is not None else "")
                            )
                        )

                elif metodo == "Búsqueda Local":
                    x_min = float(self.xmin_input.text())
                    x_max = float(self.xmax_input.text()) if self.xmax_input.text().strip() else x_min + 1.0
                    x0 = (x_min + x_max) / 2.0

                    x_opt, f_opt, iters, tipo, dist, history = busqueda_local(
                        funcion, x0, paso=0.1, max_iter=200, tolerancia=1e-5, return_history=True
                    )
                    self.resultado_label.setText(
                        f"✅ {tipo}: x={x_opt:.6g}, f={f_opt:.6g}, iter={iters}, dist={dist:.3g}"
                    )
                    self._populate_table_from_history(history)
                    self._plot_3d_history(history, metodo, funcion, x_opt=x_opt)

                elif metodo == "Armijo":
                    x0 = float(self.xmin_input.text())
                    x_new, f_new, iters, dist, history = armijo_search(
                        funcion, x0, alpha0=1.0, rho=0.5, c=1e-4, max_iter=50, return_history=True
                    )
                    self.resultado_label.setText(
                        f"✅ x={x_new:.6g}, f={f_new:.6g}, iter={iters}, dist={dist:.3g}"
                    )
                    self._populate_table_from_history(history)
                    self._plot_3d_history(history, metodo, funcion, x_opt=x_new)

                elif metodo == "Wolfe":
                    x0 = float(self.xmin_input.text())
                    x_new, f_new, iters, dist, history = wolfe_search(
                        funcion, x0, alpha0=1.0, rho=0.5, c1=1e-4, c2=0.9, max_iter=50, return_history=True
                    )
                    self.resultado_label.setText(
                        f"✅ x={x_new:.6g}, f={f_new:.6g}, iter={iters}, dist={dist:.3g}"
                    )
                    self._populate_table_from_history(history)
                    self._plot_3d_history(history, metodo, funcion, x_opt=x_new)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Ocurrió un error al ejecutar el método:\n{e}")
                self._set_dashboard_state(active=False)

        def limpiar(self):
            self.metodo_menu.setCurrentIndex(0)
            self.func_input.clear()
            self.xmin_input.clear()
            self.xmax_input.clear()
            self.tol_input.clear()
            self._show_tolerancia(False)
            self.resultado_label.clear()
            self._set_dashboard_state(active=False)

        # =========================
        # Fondo
        # =========================

    def buscar_imagen_recursivamente(self, root_dir: str):
        """Busca la primera imagen (.png/.jpg/.jpeg/.bmp/.gif) dentro de root_dir y subcarpetas.
        Devuelve la ruta completa o None si no encuentra.
        """
        extensiones = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        try:
            for base, _, files in os.walk(root_dir):
                for name in files:
                    if name.lower().endswith(extensiones):
                        return os.path.join(base, name)
        except Exception:
            return None
        return None


    def set_background_image(self):
        # Fondo del dashboard (APP principal): Menu/Fondo/Imgen de fondo.*
        candidates = [
            resource_path(r"Menu\Fondo\Imgen de fondo.png"),
            resource_path(r"Menu\Fondo\Imgen de fondo.jpg"),
            resource_path(r"Menu\Fondo\Imgen de fondo.jpeg"),
            resource_path(r"Menu\Fondo\Imagen de fondo.png"),
            resource_path(r"Menu\Fondo\Imagen de fondo.jpg"),
            resource_path(r"Menu\Fondo\Imagen de fondo.jpeg"),
        ]

        image_path = None
        for p in candidates:
            if os.path.exists(p):
                image_path = p
                break

        # Si no coincide el nombre exacto, usa la primera imagen dentro de Menu/Fondo
        if image_path is None:
            fondo_dir = resource_path(r"Menu\Fondo")
            if os.path.isdir(fondo_dir):
                for name in os.listdir(fondo_dir):
                    if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        image_path = os.path.join(fondo_dir, name)
                        break

        if image_path:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    self.width(),
                    self.height(),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation
                )
                self.background_label.setPixmap(pixmap)
                self.background_label.setGeometry(0, 0, self.width(), self.height())

    def resizeEvent(self, event):
        self.background_label.setGeometry(0, 0, self.width(), self.height())
        if self.background_label.pixmap() and not self.background_label.pixmap().isNull():
            self.set_background_image()

        self.overlay_dark.setGeometry(0, 0, self.width(), self.height())
        self.overlay_light.setGeometry(0, 0, self.width(), self.height())
        self.overlay_dark.raise_()
        self.overlay_light.raise_()
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = InterfazOptimizacion()
    w.show()
    sys.exit(app.exec_())


