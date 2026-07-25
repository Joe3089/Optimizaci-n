# app/ui/inventory_panel.py
# Panel dinámico del módulo de Inventario — QWidget autocontenido, NO un
# módulo/app aparte: solo recolecta parámetros, construye la función de
# costo vía app.domain.inventory y la entrega a la ventana principal, que
# la ejecuta con el MISMO motor (app.application.optimization_service) y
# la muestra en el MISMO dashboard/tabla/gráficas/exportaciones de siempre.
#
# Se mantiene oculto (setVisible(False), sin reservar espacio ni cargar
# nada) hasta que el usuario elige uno de los 4 métodos "Inventario: …" en
# el combo principal (ver main_window._on_method / _populate_grouped_combo).
from __future__ import annotations

from typing import Any, Dict, List, Optional

_QT = None
try:
    from PyQt6 import QtCore, QtWidgets
    from PyQt6.QtCore import Qt
    _QT = "PyQt6"
except Exception:
    from PyQt5 import QtCore, QtWidgets           # type: ignore
    from PyQt5.QtCore import Qt                   # type: ignore
    _QT = "PyQt5"

Signal = QtCore.pyqtSignal if _QT == "PyQt5" else QtCore.pyqtSignal

from app.domain.inventory import (
    MODEL_BUILDERS, validate_inventory_inputs, INVENTORY_FUNCTIONS,
    INVENTORY_MODEL_LABELS, INVENTORY_LABEL_TO_KEY, INVENTORY_METHOD_LABELS,
)

# Mismo estilo de botón (gradiente azul→turquesa) que Calcular/Limpiar/
# Exportar/Salir en main_window.py — se duplica aquí en vez de importarse
# porque allá está definido como variable local dentro de _build_ui().
_BTN_STYLE = (
    "QPushButton {"
    "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
    "    stop:0 #1a6ee8, stop:1 #0fb8c9);"
    "  color: #ffffff;"
    "  border: none;"
    "  border-radius: 10px;"
    "  font-weight: bold;"
    "  font-size: 13px;"
    "  min-height: 38px;"
    "  letter-spacing: 0.5px;"
    "}"
    "QPushButton:hover {"
    "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
    "    stop:0 #2e88f5, stop:1 #1fd0e0);"
    "}"
    "QPushButton:pressed {"
    "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
    "    stop:0 #1050b0, stop:1 #0a90a0);"
    "}"
)

# Catálogo por clave, para mostrar métodos recomendados sin reconstruir el
# modelo (solo necesitamos "metodos_recomendados", ya fijo por modelo).
_CATALOG_BY_MODEL = {
    "eoq_clasico": next(f for f in INVENTORY_FUNCTIONS if "Clásico" in f["nombre"]),
    "eoq_backorders": next(f for f in INVENTORY_FUNCTIONS if "Faltantes" in f["nombre"]),
    "eoq_descuentos": next(f for f in INVENTORY_FUNCTIONS if "Descuentos" in f["nombre"]),
    "rop_probabilistico": next(f for f in INVENTORY_FUNCTIONS if "Reorden" in f["nombre"]),
}

# Métodos recomendados por modelo (mismos que devuelve cada builder; se
# repiten aquí como constante para poblar el combo de algoritmo sin tener
# que construir un modelo de prueba solo para leer la lista).
_RECOMMENDED_METHODS = {
    "eoq_clasico": ["Búsqueda Local", "Fibonacci", "Sección Áurea", "Sección Dorada",
                    "Goldstein", "Newton-Raphson", "Grad. Conjugado", "Armijo", "Wolfe"],
    "eoq_backorders": ["Armijo", "Wolfe", "MD: Penalización (Newton)",
                       "MD: Barreras (Newton)", "MD: Pen. (BFGS)", "MD: Pes. (BFGS)",
                       "MD: Pes. (Nelder-Mead)", "MD: Sum. (BFGS)"],
    "eoq_descuentos": ["Búsqueda Local", "SA: Recocido Simulado", "PSO: Enjambre",
                       "GA: Algoritmo Genético"],
    "rop_probabilistico": ["Búsqueda Local", "Fibonacci", "Sección Áurea", "Sección Dorada",
                           "Goldstein", "Newton-Raphson", "Grad. Conjugado", "Armijo", "Wolfe"],
}


def _dspin(minv=0.0001, maxv=1_000_000.0, val=1.0, decimals=4, step=1.0):
    """
    QDoubleSpinBox con rango/ancho acotados. `maxv` bajó de 1e9 a 1e6 por
    defecto: Qt reserva el ancho del spinbox según el valor máximo posible
    ("1000000000.0000" con maxv=1e9 exige mucho más espacio horizontal que
    el panel de control tiene disponible — 375px con márgenes — y el layout
    no podía comprimirlo, desbordando todo el panel de Inventario fuera de
    sus márgenes). `setMaximumWidth` es la garantía final, sin importar el
    rango que se use.
    """
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(minv, maxv); s.setDecimals(decimals)
    s.setValue(val); s.setSingleStep(step)
    s.setMaximumWidth(150)
    return s


class InventoryPanel(QtWidgets.QFrame):
    """
    Panel de parámetros de inventario. `built` se emite con (model_dict,
    algoritmo_elegido) cuando el usuario genera la función con éxito;
    main_window la conecta para volcar fx/gx/vars/x0/rango en los campos
    genéricos existentes y luego deja que el flujo normal de `ejecutar()`
    haga el resto (sin duplicar lógica de cálculo/dashboard/exportación).
    """
    built = Signal(dict, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("invSec")
        self._model_key: Optional[str] = None
        self._last_model: Optional[Dict[str, Any]] = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6); lay.setSpacing(5)

        self.lbl_title = QtWidgets.QLabel("Parámetros de Inventario")
        self.lbl_title.setStyleSheet("font-weight:bold;color:#7aaaea;")
        # CRÍTICO: sin word-wrap, un QLabel calcula su ancho MÍNIMO para
        # mostrar todo el texto en una sola línea — con el título largo que
        # arma set_model() (p.ej. "... — Revisión Periódica / Punto de
        # Reorden (ROP) Probabilístico") esto forzaba TODO el panel (y por
        # cascada, el panel de control completo) a expandirse muy por
        # encima de sus 375px habituales. Con wrap, el label se ajusta al
        # ancho real del panel en vez de imponerle el suyo.
        self.lbl_title.setWordWrap(True)
        lay.addWidget(self.lbl_title)

        # ── Campos comunes a varios modelos ──────────────────────────────
        self.row_D = self._field_row(lay, "Demanda anual (D):", "D", 1000.0)
        self.row_S = self._field_row(lay, "Costo de pedido (S):", "S", 50.0)
        self.row_H = self._field_row(lay, "Costo de mantenimiento (H):", "H", 2.0)
        self.row_B = self._field_row(lay, "Costo de faltante (B):", "B", 8.0)

        # ── Descuentos por cantidad: tasa + tabla de tramos ─────────────
        self.row_rate = self._field_row(
            lay, "Tasa de mantenimiento (fracción, ej. 0.2):", "rate", 0.2, decimals=4,
            minv=0.0001, maxv=0.9999)

        self.lbl_breaks = QtWidgets.QLabel("Tramos de precio (cantidad mínima, precio unitario):")
        self.lbl_breaks.setWordWrap(True)
        lay.addWidget(self.lbl_breaks)
        self.tbl_breaks = QtWidgets.QTableWidget(3, 2)
        self.tbl_breaks.setHorizontalHeaderLabels(["Cant. mínima", "Precio unit."])
        self.tbl_breaks.verticalHeader().setVisible(False)
        self.tbl_breaks.setMaximumHeight(110)
        for r, (q, p) in enumerate([(0, 10.0), (100, 9.0), (500, 8.0)]):
            self.tbl_breaks.setItem(r, 0, QtWidgets.QTableWidgetItem(str(q)))
            self.tbl_breaks.setItem(r, 1, QtWidgets.QTableWidgetItem(str(p)))
        lay.addWidget(self.tbl_breaks)
        row_break_btns = QtWidgets.QHBoxLayout()
        self.btn_add_break = QtWidgets.QPushButton("+ tramo")
        self.btn_del_break = QtWidgets.QPushButton("− tramo")
        self.btn_add_break.setStyleSheet(_BTN_STYLE)
        self.btn_del_break.setStyleSheet(_BTN_STYLE)
        self.btn_add_break.clicked.connect(self._add_break_row)
        self.btn_del_break.clicked.connect(self._del_break_row)
        row_break_btns.addWidget(self.btn_add_break); row_break_btns.addWidget(self.btn_del_break)
        self.row_breaks_widget = QtWidgets.QWidget(); self.row_breaks_widget.setLayout(row_break_btns)
        lay.addWidget(self.row_breaks_widget)

        # ── ROP probabilístico ───────────────────────────────────────────
        self.row_d_diario = self._field_row(lay, "Demanda diaria promedio (d̄):", "d_diario", 20.0)
        self.row_sigma_d  = self._field_row(lay, "Desv. estándar diaria (σ_d):", "sigma_d", 5.0)
        self.row_lead     = self._field_row(lay, "Lead time (días):", "lead_time", 7.0)
        self.row_nivel    = self._field_row(
            lay, "Nivel de servicio (0-1):", "nivel_servicio", 0.95, decimals=4, step=0.01,
            minv=0.0001, maxv=0.9999)

        # ── Algoritmo compatible (filtrado por modelo) ──────────────────
        # El combo ya solo lista los métodos compatibles con el modelo
        # elegido — eso por sí solo impide seleccionar uno incompatible.
        # (Se eliminó el texto informativo "No compatibles con este
        # modelo: …" que iba debajo: no tenía ninguna función interactiva,
        # solo repetía en prosa lo que el propio combo ya garantiza.)
        lay.addWidget(QtWidgets.QLabel("Algoritmo a usar:"))
        self.cmb_algo = QtWidgets.QComboBox()
        lay.addWidget(self.cmb_algo)

        # ── Generar / error / resultados analíticos ─────────────────────
        self.btn_generar = QtWidgets.QPushButton("Generar función →")
        self.btn_generar.setStyleSheet(_BTN_STYLE)
        self.btn_generar.clicked.connect(self._on_generar)
        lay.addWidget(self.btn_generar)

        self.lbl_error = QtWidgets.QLabel("")
        self.lbl_error.setStyleSheet("color:#ff6b6b; font-size:11px;")
        self.lbl_error.setWordWrap(True)
        lay.addWidget(self.lbl_error)

        self.lbl_analitico = QtWidgets.QLabel("")
        self.lbl_analitico.setStyleSheet("color:#8fd0a0; font-size:11px;")
        self.lbl_analitico.setWordWrap(True)
        lay.addWidget(self.lbl_analitico)

        self.lbl_resultado = QtWidgets.QLabel("")
        self.lbl_resultado.setStyleSheet("color:#ffd76a; font-size:11px; font-weight:bold;")
        self.lbl_resultado.setWordWrap(True)
        lay.addWidget(self.lbl_resultado)

        self.setVisible(False)

    # ── helpers de construcción ──────────────────────────────────────────
    def _field_row(self, lay, label_text, attr_prefix, default, decimals=4, step=1.0,
                  minv=0.0001, maxv=1_000_000.0):
        """
        Etiqueta arriba + campo abajo a todo el ancho — mismo patrón que el
        resto del "Panel de control" (p.ej. "Función f(x):" / edt_fx), en
        vez de etiqueta y campo lado a lado. Con etiquetas largas como
        "Demanda diaria promedio (d̄):" el layout lado a lado desbordaba el
        ancho del panel y quedaba desalineado con el resto de la app.
        """
        row = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(row); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(3)
        lbl = QtWidgets.QLabel(label_text)
        lbl.setWordWrap(True)   # ver nota en lbl_title: evita que una etiqueta
                                # larga fuerce el ancho de todo el panel
        v.addWidget(lbl)
        spin = _dspin(val=default, decimals=decimals, step=step, minv=minv, maxv=maxv)
        setattr(self, f"spin_{attr_prefix}", spin)
        v.addWidget(spin)
        lay.addWidget(row)
        return row

    def _add_break_row(self):
        r = self.tbl_breaks.rowCount()
        self.tbl_breaks.insertRow(r)
        self.tbl_breaks.setItem(r, 0, QtWidgets.QTableWidgetItem("0"))
        self.tbl_breaks.setItem(r, 1, QtWidgets.QTableWidgetItem("0"))

    def _del_break_row(self):
        r = self.tbl_breaks.rowCount()
        if r > 1:
            self.tbl_breaks.removeRow(r - 1)

    # ── activación / cambio de submodelo ──────────────────────────────────
    def set_model(self, model_key: str) -> None:
        """Muestra solo los campos relevantes para `model_key` y filtra el
        combo de algoritmo a los métodos compatibles con ese modelo."""
        self._model_key = model_key
        self.lbl_error.setText(""); self.lbl_resultado.setText("")
        self.lbl_analitico.setText("")

        is_backorders = model_key == "eoq_backorders"
        is_descuentos = model_key == "eoq_descuentos"
        is_rop        = model_key == "rop_probabilistico"
        is_clasico    = model_key == "eoq_clasico"

        self.row_D.setVisible(not is_rop)
        self.row_S.setVisible(True)
        self.row_H.setVisible(not is_descuentos)
        self.row_B.setVisible(is_backorders)
        self.row_rate.setVisible(is_descuentos)
        self.lbl_breaks.setVisible(is_descuentos)
        self.tbl_breaks.setVisible(is_descuentos)
        self.row_breaks_widget.setVisible(is_descuentos)
        self.row_d_diario.setVisible(is_rop)
        self.row_sigma_d.setVisible(is_rop)
        self.row_lead.setVisible(is_rop)
        self.row_nivel.setVisible(is_rop)

        self.cmb_algo.clear()
        self.cmb_algo.addItems(_RECOMMENDED_METHODS.get(model_key, []))

        cat = _CATALOG_BY_MODEL.get(model_key, {})
        titulo = cat.get("nombre", "Inventario")
        self.lbl_title.setText(f"Parámetros de Inventario — {titulo}")

    def model_key(self) -> Optional[str]:
        return self._model_key

    def selected_algorithm(self) -> str:
        return self.cmb_algo.currentText().strip()

    def current_model(self) -> Optional[Dict[str, Any]]:
        return self._last_model

    # ── generar función (valida + construye) ─────────────────────────────
    def _read_breaks(self) -> List[tuple]:
        breaks = []
        for r in range(self.tbl_breaks.rowCount()):
            it_q = self.tbl_breaks.item(r, 0)
            it_p = self.tbl_breaks.item(r, 1)
            q_txt = it_q.text().strip() if it_q else ""
            p_txt = it_p.text().strip() if it_p else ""
            if not q_txt or not p_txt:
                continue
            breaks.append((float(q_txt), float(p_txt)))
        return breaks

    def _on_generar(self):
        self.lbl_error.setText(""); self.lbl_resultado.setText("")
        if not self._model_key:
            return
        try:
            if self._model_key == "eoq_clasico":
                kwargs = dict(D=self.spin_D.value(), S=self.spin_S.value(), H=self.spin_H.value())
            elif self._model_key == "eoq_backorders":
                kwargs = dict(D=self.spin_D.value(), S=self.spin_S.value(),
                              H=self.spin_H.value(), B=self.spin_B.value())
            elif self._model_key == "eoq_descuentos":
                kwargs = dict(D=self.spin_D.value(), S=self.spin_S.value(),
                              holding_cost_rate=self.spin_rate.value(),
                              price_breaks=self._read_breaks())
            elif self._model_key == "rop_probabilistico":
                kwargs = dict(d_diario=self.spin_d_diario.value(),
                              sigma_d_diario=self.spin_sigma_d.value(),
                              lead_time_dias=self.spin_lead_time.value(),
                              nivel_servicio=self.spin_nivel_servicio.value(),
                              D_anual=self.spin_D.value() if hasattr(self, "spin_D") else 1000.0,
                              S=self.spin_S.value(), H=self.spin_H.value())
            else:
                return

            validate_inventory_inputs(self._model_key, **kwargs)
            model = MODEL_BUILDERS[self._model_key](**kwargs)
        except ValueError as ex:
            self.lbl_error.setText(str(ex))
            return
        except Exception as ex:
            self.lbl_error.setText(f"Error al construir el modelo: {ex}")
            return

        self._last_model = model
        extra_txt = "  ·  ".join(f"{k}: {v}" for k, v in model["extra_results"].items())
        self.lbl_analitico.setText(f"Referencia analítica — {extra_txt}")

        algo = self.selected_algorithm()
        self.built.emit(model, algo)

    # ── resultados tras ejecutar() (Dashboard ya calculó todo) ───────────
    def show_results(self, session_entry: Dict[str, Any]) -> None:
        """Pinta, dentro del propio panel, el resultado ya calculado por el
        Dashboard principal (mismo `hist`/`session`, sin recálculo) — deja
        ambas vistas sincronizadas."""
        model = self._last_model
        if not model or not session_entry:
            return
        hist = session_entry.get("hist") or []
        if not hist:
            return
        last = hist[-1]
        x_val = last.get("x", last.get("x_k"))
        labels = model.get("dashboard_labels", {})
        var_names = model.get("var_names", [])
        parts = []
        if len(var_names) == 1 and x_val is not None:
            parts.append(f"{labels.get(var_names[0], var_names[0])} = {float(x_val):.4f}")
        elif isinstance(x_val, (list, tuple)):
            for name, v in zip(var_names, x_val):
                parts.append(f"{labels.get(name, name)} = {float(v):.4f}")
        costo = last.get("f(x)", last.get("f_k", last.get("f_lambda")))
        if costo is not None:
            try:
                parts.append(f"Costo total = {float(costo):.4f}")
            except (TypeError, ValueError):
                pass
        if "ROP" in labels:
            parts.append(f"{labels['ROP']} = {model['extra_results'].get('Punto de reorden (ROP)')}")
        self.lbl_resultado.setText("Resultado obtenido — " + "  ·  ".join(parts))
