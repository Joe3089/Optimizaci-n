# -*- coding: utf-8 -*-
"""
result_box.py — Recuadro de resultado / error con la explicación de la IA.

Se conecta a AutoInterventionController: muestra el evento en cuanto el
motor termina y añade debajo la explicación (IA o respaldo local). Controles
compactos: voz automática on/off, detener, repetir y ver/ocultar explicación.
Incluye además el diálogo de diagnóstico que ejecuta
test_ai_automatic_intervention() sin abrir la terminal.
"""
from __future__ import annotations

import html

try:
    from PyQt5 import QtWidgets
    from PyQt5.QtCore import Qt
except ImportError:  # pragma: no cover
    from PyQt6 import QtWidgets  # type: ignore
    from PyQt6.QtCore import Qt  # type: ignore

from app.application import result_analysis as ra
from app.infrastructure import auto_intervention as ai

_RED = "#ff5c5c"
# color de acento, icono
_STYLE = {
    ra.SUCCESS:          ("#2ecc71", "✅"),
    ra.WARNING:          ("#f5b041", "⚠️"),
    ra.CONVERGENCE:      ("#ff9f43", "🔁"),
    ra.MAX_ITERATIONS:   ("#ff9f43", "🔁"),
    ra.PARAMETER_ERROR:  ("#f5b041", "🛠️"),
    ra.DOMAIN_ERROR:     (_RED, "⛔"),
    ra.DIVISION_BY_ZERO: (_RED, "➗"),
    ra.SYNTAX_ERROR:     (_RED, "✏️"),
    ra.EXECUTION_ERROR:  (_RED, "✗"),
}

_SPEECH_TXT = {
    ai.TEXT_READY: "📝 Texto generado",
    ai.REQUESTED: "⏳ Preparando la voz…",
    ai.PLAYING: "🔊 Reproduciendo…",
    ai.FINISHED: "✔ Lectura finalizada",
    ai.STOPPED: "⏹ Detenida",
    ai.OFF: "🔇 Voz automática desactivada",
}


class ResultBox(QtWidgets.QFrame):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.ctrl = controller
        self.setObjectName("resultBox")
        self.last_event = None
        self.last_explanation = ""
        self.last_ai_status = ""
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(3)

        top = QtWidgets.QHBoxLayout()
        self.lbl_title = QtWidgets.QLabel("")
        self.lbl_title.setWordWrap(True)          # nunca impone un ancho mayor al del panel
        self.lbl_title.setMinimumWidth(60)
        top.addWidget(self.lbl_title, 1)
        self.lbl_speech = QtWidgets.QLabel("")
        self.lbl_speech.setWordWrap(True)
        self.lbl_speech.setMaximumWidth(130)
        self.lbl_speech.setStyleSheet("color:#8fb4ff;font-size:11px;border:none;")
        top.addWidget(self.lbl_speech)

        btn_row = QtWidgets.QHBoxLayout()   # botones en su fila: el título usa todo el ancho
        btn_row.setSpacing(4)

        def _btn(txt, tip):
            b = QtWidgets.QToolButton()
            b.setText(txt); b.setToolTip(tip)
            b.setStyleSheet("QToolButton{color:#cfe3ff;background:rgba(40,70,140,150);"
                            "border:1px solid #2a4ea0;border-radius:5px;padding:2px 6px;}"
                            "QToolButton:checked{background:rgba(0,150,200,170);}"
                            "QToolButton:disabled{color:#5a6a8a;}")
            btn_row.addWidget(b)
            return b

        self.btn_voice = _btn("🔊", "Explicación automática por voz (activar/desactivar)")
        self.btn_voice.setCheckable(True)
        self.btn_voice.setChecked(controller.voice_enabled)
        self.btn_stop = _btn("⏹", "Detener la reproducción")
        self.btn_repeat = _btn("🔁", "Repetir la explicación en voz")
        self.btn_text = _btn("📄", "Mostrar u ocultar la explicación escrita")
        self.btn_text.setCheckable(True)
        self.btn_text.setChecked(True)
        self.btn_stop.setEnabled(False)
        self.btn_repeat.setEnabled(False)
        btn_row.addStretch(1)
        lay.addLayout(top)
        lay.addLayout(btn_row)

        self.lbl_facts = QtWidgets.QLabel("")
        self.lbl_facts.setWordWrap(True)
        self.lbl_facts.setTextFormat(Qt.RichText)
        self.lbl_facts.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.lbl_facts)

        self.lbl_ai = QtWidgets.QLabel("")
        self.lbl_ai.setWordWrap(True)
        self.lbl_ai.setTextFormat(Qt.RichText)
        self.lbl_ai.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_ai.setStyleSheet("color:#bfe9ff; background:rgba(0,90,140,70);"
                                  "border-left:3px solid #00c8ff; padding:4px 6px;")
        lay.addWidget(self.lbl_ai)

        self.btn_voice.toggled.connect(self._on_voice_toggled)
        self.btn_stop.clicked.connect(controller.stop)
        self.btn_repeat.clicked.connect(controller.repeat)
        self.btn_text.toggled.connect(self.lbl_ai.setVisible)
        controller.event_presented.connect(self.show_event)
        controller.explanation_ready.connect(self.show_explanation)
        controller.ai_status.connect(self._on_ai_status)
        controller.speech_state.connect(self._on_speech_state)
        self.setVisible(False)

    def set_chat_hosted(self, hosted: bool = True):
        """Dentro del chat, la explicación de Opti ya aparece como mensaje: el
        texto del recuadro queda disponible con 📄 pero oculto, sin duplicarlo."""
        self.btn_text.setChecked(not hosted)
        self.lbl_ai.setVisible(not hosted)
        if hosted:
            self.btn_text.setToolTip("Mostrar también aquí la explicación escrita "
                                     "(Opti ya la escribe en el chat)")

    # ── Slots ────────────────────────────────────────────────────────────────
    def show_event(self, ev):
        self.last_event = ev
        self.last_explanation = ""
        self.last_ai_status = ""
        color, icon = _STYLE.get(ev.kind, ("#a8d0ff", "ℹ️"))
        self.setStyleSheet(
            f"QFrame#resultBox{{background:rgba(8,20,55,215);border:1px solid {color};"
            f"border-left:5px solid {color};border-radius:6px;}}"
            "QFrame#resultBox QLabel{color:#dce8ff;font-size:12px;border:none;}")
        self.lbl_title.setText(f"{icon}  {ev.title}")
        self.lbl_title.setStyleSheet(f"font-weight:bold;font-size:13px;color:{color};border:none;")
        rows = [f"<b style='color:#8fb4ff'>{html.escape(k)}:</b> {html.escape(str(v))}"
                for k, v in ev.facts()]
        if ev.recommendation and ev.kind != ra.SUCCESS:
            rows.append(f"<b style='color:{color}'>Recomendación:</b> "
                        f"{html.escape(ev.recommendation)}")
        self.lbl_facts.setText("<br>".join(rows))
        self.lbl_facts.setToolTip(ev.technical_detail[:1500] if ev.technical_detail else "")
        # Explicación preliminar inmediata (datos del motor); la IA la reemplaza al responder
        if ev.kind in ra.CHAT_KINDS:
            self.lbl_ai.setText("<i>🤖 La IA está analizando el resultado… Explicación preliminar:</i> "
                                + html.escape(ra.local_explanation(ev)))
        else:   # resultado correcto: Opti no interviene
            self.lbl_ai.setText("<b>📊 Resumen del motor:</b> " + html.escape(ra.local_explanation(ev)))
        self.lbl_speech.setText("")
        self.btn_repeat.setEnabled(False)
        self.setVisible(True)

    def _on_ai_status(self, ev, status):
        if ev is self.last_event:
            self.last_ai_status = status

    def show_explanation(self, ev, text, source):
        if ev is not self.last_event:
            return
        self.last_explanation = text
        st = self.last_ai_status
        if source == "local" and st.startswith("sin intervención"):
            tag = "Resumen del motor (Opti no interviene en resultados correctos)"
        elif source == "local" and st.startswith("ai_comm_error"):
            tag = ("⚠️ Error de comunicación con la IA — explicación generada localmente "
                   "con los datos del motor")
        elif source == "local":
            tag = f"Explicación local (IA {st or 'no disponible'})"
        elif source == "ia-simulada":
            tag = "IA (modo de prueba)"
        else:
            tag = "Asistente IA"
        self.lbl_ai.setText(f"<b>🤖 {html.escape(tag)}:</b> {html.escape(text)}")
        self.btn_repeat.setEnabled(True)

    def _on_voice_toggled(self, on):
        self.ctrl.set_voice_enabled(on)
        self.btn_voice.setText("🔊" if on else "🔇")

    def _on_speech_state(self, state):
        self.btn_stop.setEnabled(state in (ai.REQUESTED, ai.PLAYING))
        if state.startswith("error:"):
            self.lbl_speech.setText("🔇 Voz no disponible")
            self.lbl_speech.setToolTip(state[6:])
        else:
            self.lbl_speech.setText(_SPEECH_TXT.get(state, ""))


class DiagnosticDialog(QtWidgets.QDialog):
    """Ejecuta test_ai_automatic_intervention() desde la propia aplicación."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diagnóstico: intervención automática IA + voz")
        self.resize(820, 560)
        lay = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            "<b>Prueba simulada</b>: IA y voz de prueba, sin consumir API (repetible).<br>"
            "<b>Prueba real</b>: usa tu API Key configurada y la voz del sistema "
            "(se escuchará audio y tardará unos minutos).")
        info.setWordWrap(True)
        lay.addWidget(info)
        row = QtWidgets.QHBoxLayout()
        self.btn_sim = QtWidgets.QPushButton("▶ Prueba simulada")
        self.btn_real = QtWidgets.QPushButton("▶ Prueba real (IA + voz)")
        row.addWidget(self.btn_sim); row.addWidget(self.btn_real); row.addStretch(1)
        lay.addLayout(row)
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setStyleSheet("font-family:Consolas,monospace;font-size:11px;")
        lay.addWidget(self.out, 1)
        self.btn_sim.clicked.connect(lambda: self._run(False))
        self.btn_real.clicked.connect(lambda: self._run(True))

    def _run(self, real: bool):
        from app.infrastructure.intervention_selftest import (
            run_ai_intervention_suite, format_report)
        self.btn_sim.setEnabled(False); self.btn_real.setEnabled(False)
        self.out.setPlainText("Ejecutando pruebas… (la ventana sigue respondiendo)")
        try:
            res = run_ai_intervention_suite(real=real, verbose=False)
            self.out.setPlainText(format_report(res))
        except Exception as ex:
            self.out.setPlainText(f"No se pudo ejecutar el diagnóstico: {ex}")
        finally:
            self.btn_sim.setEnabled(True); self.btn_real.setEnabled(True)
