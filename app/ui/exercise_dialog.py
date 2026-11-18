# app/ui/exercise_dialog.py
# Cuadro para elegir qué ejercicio estudiar cuando el archivo cargado (Word,
# PDF, Excel o CSV) contiene varios.
from __future__ import annotations

from typing import List, Optional

from PyQt5 import QtCore, QtWidgets

_STYLE = (
    "QDialog{background:#0a1228;}"
    "QLabel{color:#dce8ff;}"
    "QLabel#exPreview{color:#8fb4ff;font-size:11px;padding-left:22px;}"
    "QRadioButton{color:#ffffff;font-weight:bold;font-size:12px;}"
    "QFrame#exCard{background:rgba(20,40,90,170);border:1px solid #2a4ea0;border-radius:6px;}"
    "QScrollArea{border:none;background:transparent;}"
    "QPushButton{color:#cfe3ff;background:rgba(40,70,140,190);border:1px solid #2a4ea0;"
    "border-radius:5px;padding:6px 16px;}"
    "QPushButton:hover{background:rgba(60,100,190,220);}"
    "QPushButton:default{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
    "stop:0 #1e6fe0,stop:1 #10b4c8);color:#ffffff;font-weight:bold;}")


class ExerciseDialog(QtWidgets.QDialog):
    def __init__(self, exercises: List, filename: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Selecciona el ejercicio a estudiar")
        self.setStyleSheet(_STYLE)
        self.setMinimumSize(520, 360)
        lay = QtWidgets.QVBoxLayout(self)
        head = QtWidgets.QLabel(
            f"El archivo «{filename}» contiene <b>{len(exercises)} ejercicios</b>.<br>"
            "Indica cuál quieres estudiar y pulsa <b>Aceptar</b>:")
        head.setWordWrap(True)
        lay.addWidget(head)

        inner = QtWidgets.QWidget()
        inner.setStyleSheet("background:transparent;")
        il = QtWidgets.QVBoxLayout(inner)
        il.setSpacing(6)
        self._group = QtWidgets.QButtonGroup(self)
        self.radios: List[QtWidgets.QRadioButton] = []
        for i, ex in enumerate(exercises):
            card = QtWidgets.QFrame(); card.setObjectName("exCard")
            cl = QtWidgets.QVBoxLayout(card); cl.setContentsMargins(8, 6, 8, 6); cl.setSpacing(2)
            rb = QtWidgets.QRadioButton(f"{i + 1}. {ex.title}")
            self._group.addButton(rb, i)
            self.radios.append(rb)
            cl.addWidget(rb)
            pv = QtWidgets.QLabel(ex.preview() or "(sin contenido)")
            pv.setObjectName("exPreview"); pv.setWordWrap(True)
            cl.addWidget(pv)
            il.addWidget(card)
        il.addStretch(1)
        if self.radios:
            self.radios[0].setChecked(True)
        sc = QtWidgets.QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(inner)
        lay.addWidget(sc, 1)

        box = QtWidgets.QDialogButtonBox()
        self.btn_ok = box.addButton("Aceptar", QtWidgets.QDialogButtonBox.AcceptRole)
        self.btn_cancel = box.addButton("Cancelar", QtWidgets.QDialogButtonBox.RejectRole)
        self.btn_ok.setDefault(True)
        for b in (self.btn_ok, self.btn_cancel):
            b.setMinimumWidth(120)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def selected(self) -> int:
        return max(self._group.checkedId(), 0)


def choose_exercise(parent, exercises: List, filename: str) -> Optional[int]:
    """Muestra el cuadro y devuelve el índice elegido, o None si se canceló."""
    dlg = ExerciseDialog(exercises, filename, parent)
    ok = dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()
    return dlg.selected() if ok == QtWidgets.QDialog.Accepted else None
