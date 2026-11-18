# app/ui/problem_panel.py
# Formulario genérico para los métodos de app.application.problem_catalog.
# Lo construye a partir de la especificación del método; el cálculo y la
# presentación de resultados los hace main_window.
from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Optional

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

from app.infrastructure import data_import as di

from app.application.problem_catalog import SPEC_BY_LABEL, ProblemSpec

_INPUT_STYLE = (
    "QPlainTextEdit { background: rgba(8,20,55,200); color: #e0ecff;"
    " border: 1px solid #2a4ea0; border-radius: 4px; }"
)


class ProblemPanel(QtWidgets.QFrame):
    export_requested = pyqtSignal()       # «💾 Exportar»: recolectar los resultados

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mdSec")
        self._spec: ProblemSpec | None = None
        self._widgets: Dict[str, QtWidgets.QWidget] = {}
        self._saved: Dict[str, Dict[str, str]] = {}

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(8, 6, 8, 6)
        self._layout.setSpacing(5)
        self.lbl_info = QtWidgets.QLabel()
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color:#7aaaea;")
        self._layout.addWidget(self.lbl_info)

        # Datos del usuario: el panel arranca vacío (el ejemplo es solo una guía)
        bar = data_buttons(self)
        self.btn_load, self.btn_example = bar.buttons
        self.btn_load.clicked.connect(self._load_file_dialog)
        self.btn_example.clicked.connect(self.load_example)
        self._layout.addLayout(bar)
        self.lbl_file = QtWidgets.QLabel("")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color:#3cff8a;font-size:10px;")
        self.lbl_file.setVisible(False)
        self._layout.addWidget(self.lbl_file)

        self._form = QtWidgets.QWidget()
        self._form_layout = QtWidgets.QVBoxLayout(self._form)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        self._form_layout.setSpacing(4)
        self._layout.addWidget(self._form)
        self.setVisible(False)

    def set_method(self, label: str) -> None:
        if self._spec is not None:
            self._saved[self._spec.label] = self.values()
        spec = SPEC_BY_LABEL[label]
        self._spec = spec
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._widgets.clear()
        self.lbl_info.setText(spec.info)
        self.lbl_file.setVisible(False)
        saved = self._saved.get(label, {})
        for f in spec.fields:
            # Sin word-wrap, una etiqueta larga fija el ancho mínimo del panel.
            lbl = QtWidgets.QLabel(f.label)
            lbl.setWordWrap(True)
            self._form_layout.addWidget(lbl)
            # Los campos parten vacíos: el usuario ingresa SU problema. El
            # ejemplo se ve como texto guía y se carga con «Usar ejemplo».
            first = f.default if f.kind == "choice" else ""
            self._add_input(f, saved.get(f.key, first))
            w = self._widgets[f.key]
            if f.kind != "choice" and f.default and hasattr(w, "setPlaceholderText"):
                ph = f.default if f.kind == "text" else f"Ej.: {f.default}"
                w.setPlaceholderText(ph)

    def _add_input(self, f, value: str) -> None:
        if f.kind == "choice":
            w = QtWidgets.QComboBox()
            w.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(12)
            w.addItems(list(f.options))
            w.setCurrentText(value)
            self._form_layout.addWidget(w)
        elif f.kind == "text":
            w = QtWidgets.QPlainTextEdit(value)
            w.setFixedHeight(92)
            w.setStyleSheet(_INPUT_STYLE)
            self._form_layout.addWidget(w)
        elif f.kind == "file":
            row = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            w = QtWidgets.QLineEdit(value)
            btn = QtWidgets.QPushButton("Examinar…")
            btn.clicked.connect(lambda: self._browse(w))
            lay.addWidget(w, 1)
            lay.addWidget(btn)
            self._form_layout.addWidget(row)
        else:
            w = QtWidgets.QLineEdit(value)
            self._form_layout.addWidget(w)
        self._widgets[f.key] = w

    def _browse(self, target: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Selecciona el inventario", "",
            "Inventario (*.csv *.txt *.xlsx *.xlsm)")
        if path:
            target.setText(path)

    # ── Datos del usuario ────────────────────────────────────────────────────
    def has_user_data(self) -> bool:
        """True si el usuario escribió o cargó algún dato (no solo opciones)."""
        if self._spec is None:
            return False
        return any(v.strip() for k, v in self.values().items()
                   if self._kind(k) != "choice")

    def _kind(self, key: str) -> str:
        for f in (self._spec.fields if self._spec else []):
            if f.key == key:
                return f.kind
        return "line"

    def load_example(self) -> None:
        if self._spec is not None:
            self.set_values({f.key: f.default for f in self._spec.fields})
            self.lbl_file.setText("📋 Ejemplo cargado: reemplázalo por tus datos si quieres.")
            self.lbl_file.setVisible(True)

    def clear_values(self) -> None:
        if self._spec is not None:
            self.set_values({f.key: "" for f in self._spec.fields if f.kind != "choice"})
            self._saved.pop(self._spec.label, None)
            self.lbl_file.setVisible(False)

    def _data_field(self):
        """Campo que recibe un archivo: el primero de tipo archivo o texto."""
        fields = self._spec.fields if self._spec else []
        for kind in ("file", "text"):
            for f in fields:
                if f.kind == kind:
                    return f
        return None

    def _load_file_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Selecciona tus datos", "", di.FILE_FILTER)
        if path:
            try:
                self.load_file(path)
            except ValueError as ex:
                QtWidgets.QMessageBox.warning(self, "Cargar archivo", str(ex))

    def _match(self, pairs: Dict[str, str]) -> Dict[str, str]:
        """Asigna cada «clave: valor» del archivo al campo con esa clave o etiqueta."""
        out: Dict[str, str] = {}
        for f in self._spec.fields:
            names = {di.norm_key(f.key), di.norm_key(f.label.split("(")[0])}
            names.discard("")
            for k, v in pairs.items():
                if k in names or (len(k) >= 3 and any(n.startswith(k) for n in names)):
                    if f.kind == "choice":
                        opt = next((o for o in f.options if di.norm_key(o).startswith(
                            di.norm_key(v)[:6])), None)
                        if opt:
                            out[f.key] = opt
                    else:
                        out[f.key] = v
                    break
        return out

    def load_file(self, path: str, exercise: Optional[int] = None) -> List[str]:
        """Vuelca un archivo CSV/TXT/Excel/Word/PDF en los campos del método:
        líneas «campo: valor» van al campo con ese nombre y las filas numéricas,
        al campo de tabla. Devuelve las claves llenadas."""
        if self._spec is None:
            raise ValueError("Primero elige un método.")
        from app.ui.exercise_dialog import choose_exercise
        _r = di.resolve_exercise(path, exercise,
                                 lambda exs, name: choose_exercise(self, exs, name))
        if _r is None:
            return []                    # el usuario canceló la selección del ejercicio
        path, shown = _r
        fields = self._spec.fields
        ext = os.path.splitext(path)[1].lower()
        file_f = next((f for f in fields if f.kind == "file"), None)
        if file_f is not None:          # el método ya lee CSV/Excel por sí mismo (inventario)
            text_f = next((f for f in fields if f.kind == "text"), None)
            if ext in (".csv", ".txt", ".xlsx", ".xlsm") or text_f is None:
                self.set_values({file_f.key: path})
                filled = [file_f.key]
            else:                        # Word/PDF: la tabla se pega en «O pega los datos»
                rows = di.read_rows(path)
                if not rows:
                    raise ValueError("El archivo no contiene datos.")
                self.set_values({file_f.key: "",
                                 text_f.key: "\n".join(", ".join(r) for r in rows)})
                filled = [text_f.key]
        else:
            values = self._match(di.read_pairs(path))
            text_f = next((f for f in fields if f.kind == "text" and f.key not in values), None)
            if text_f is not None:
                rows = [r for r in di.read_table(path) if all(di.is_number(c) for c in r if c)]
                if not rows and not values:
                    rows = di.read_table(path)
                if rows:
                    values[text_f.key] = "\n".join(", ".join(c for c in r if c) for r in rows)
            if not values:
                names = ", ".join(f"«{f.label.rstrip(':')}»" for f in fields if f.kind != "choice")
                raise ValueError("No se reconocieron datos para este método en el archivo. "
                                 "Escribe una línea «campo: valor» por dato (" + names + ") "
                                 "o una tabla numérica.")
            self.set_values(values)
            filled = list(values)
            # Si llegó una muestra y el método tiene modo «Datos de una muestra» (Chebyshev)
            if any(f.kind == "text" and f.key in values for f in fields):
                for g in fields:
                    opt = next((o for o in g.options if o.lower().startswith("datos")), None) \
                        if g.kind == "choice" else None
                    if opt and g.key not in values:
                        self.set_values({g.key: opt})
        names = ", ".join(next(f.label.rstrip(":") for f in fields if f.key == k) for k in filled)
        self.lbl_file.setText(f"📂 Cargado: {shown} → {names}")
        self.lbl_file.setVisible(True)
        return filled

    def values(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for key, w in self._widgets.items():
            if isinstance(w, QtWidgets.QComboBox):
                out[key] = w.currentText()
            elif isinstance(w, QtWidgets.QPlainTextEdit):
                out[key] = w.toPlainText()
            else:
                out[key] = w.text()
        return out

    def set_values(self, values: Dict[str, str]) -> None:
        for key, value in values.items():
            w = self._widgets.get(key)
            if isinstance(w, QtWidgets.QComboBox):
                w.setCurrentText(value)
            elif isinstance(w, QtWidgets.QPlainTextEdit):
                w.setPlainText(value)
            elif w is not None:
                w.setText(value)

    def forget(self) -> None:
        """Descarta lo escrito y vuelve a los ejemplos precargados."""
        self._saved.clear()
        self._spec = None



_BAR_STYLE = ("QPushButton{color:#cfe3ff;background:rgba(40,70,140,150);"
              "border:1px solid #2a4ea0;border-radius:5px;padding:3px 4px;font-size:10px;}"
              "QPushButton:hover{background:rgba(60,100,190,190);}")


def data_buttons(parent=None) -> QtWidgets.QGridLayout:
    """Barra común «📂 Cargar archivo · 📋 Usar ejemplo». Se usa en los métodos con
    f(x), en el catálogo y en inventario."""
    grid = QtWidgets.QGridLayout()
    grid.setSpacing(4)
    grid.setContentsMargins(0, 0, 0, 0)
    # Vaciar y Exportar ya los hacen los botones azules «Limpiar» y «Exportar».
    specs = [("📂 Cargar archivo", "Carga tus datos desde CSV, Excel, Word o PDF"),
             ("📋 Usar ejemplo", "Rellena los campos con un ejemplo de este método")]
    grid.buttons = []
    for i, (txt, tip) in enumerate(specs):
        b = QtWidgets.QPushButton(txt, parent)
        b.setToolTip(tip)
        b.setStyleSheet(_BAR_STYLE)
        b.setMinimumHeight(26)
        grid.addWidget(b, i // 2, i % 2)
        grid.buttons.append(b)
    return grid


def read_table(path: str) -> List[List[str]]:
    """Compatibilidad: ver app.infrastructure.data_import.read_table."""
    return di.read_table(path)
