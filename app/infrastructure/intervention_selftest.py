# -*- coding: utf-8 -*-
"""
intervention_selftest.py — Suite de la intervención automática IA + voz.

`run_ai_intervention_suite()` ejecuta escenarios reales a través de la
ventana principal:  InterfazOptimizacion.ejecutar() → motor matemático →
detector de eventos → controlador → IA → explicación → TTS → ResultBox.

  • Modo simulado (real=False): IA simulada + TTS de prueba. Sin API, sin
    audio, repetible. La voz se aprueba solo si el mecanismo de reproducción
    fue INVOCADO con la explicación exacta (no basta con generar texto), y
    el informe deja claro que no hubo audio real.
  • Modo real (real=True): IA configurada en la app (API Key guardada por el
    asistente; nunca se lee ni se imprime aquí) + voz SAPI5 real. Exige que la
    IA real haya respondido y que el motor TTS haya notificado inicio y fin
    de la reproducción; además genera un WAV para confirmar síntesis de audio.

Lo usan tests/test_ai_automatic_intervention.py (pytest / --real) y el
botón «🩺 Diagnóstico IA y voz» de la aplicación.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import List

from app.application import result_analysis as ra
from app.infrastructure.auto_intervention import SILENT

SCENARIOS = [
    dict(id="T1", name="Resultado satisfactorio", metodo="Newton-Raphson",
         fx="(x-3)**2 + 2", x0="0", lo=0.0, hi=6.0,
         kinds={ra.SUCCESS}, expected="mínimo en x = 3, valor mínimo 2; Opti NO interviene (sin IA ni voz)",
         keywords=["3", "2", "mínimo", "convergió"], check_values=("3", 2.0)),
    dict(id="T2", name="Error de dominio", metodo="Newton-Raphson",
         fx="sqrt(x-5)", x0="2", lo=0.0, hi=10.0,
         kinds={ra.DOMAIN_ERROR}, expected="punto inicial x = 2 inválido; probar x = 6",
         keywords=["raíz", "negativo"], rec_keywords=["x = 6"], bad_point=[2.0]),
    dict(id="T3", name="División entre cero", metodo="Newton-Raphson",
         fx="1/(x-2)", x0="2", lo=0.0, hi=5.0,
         kinds={ra.DIVISION_BY_ZERO}, expected="denominador nulo en x = 2; evitar x = 2",
         keywords=["denominador", "0"], rec_keywords=["evita x = 2"], bad_point=[2.0]),
    dict(id="T4", name="Falta de convergencia", metodo="Newton-Raphson",
         fx="log(1+x**2)", x0="2", lo=-5.0, hi=5.0,
         kinds={ra.MAX_ITERATIONS, ra.CONVERGENCE},
         expected="60 iteraciones sin cumplir la tolerancia (no es excepción); ajustar parámetros",
         keywords=["iteraciones", "precisión"], rec_keywords=["punto inicial", "tolerancia"],
         no_exception=True),
    dict(id="T5", name="Error de sintaxis", metodo="Newton-Raphson",
         fx="(x-3**2 +", x0="0", lo=0.0, hi=6.0,
         kinds={ra.SYNTAX_ERROR}, expected="expresión inválida; cómo corregirla",
         keywords=["sintaxis", "paréntesis"], rec_keywords=["**", "paréntesis"]),
    dict(id="T10", name="Resultado NaN (caso C)", metodo="Newton-Raphson",
         fx="cosh(x) - sinh(x)", x0="1000", lo=-10.0, hi=10.0,
         kinds={ra.NAN_RESULT}, expected="NaN por ∞ − ∞ (desbordamiento intermedio); reescribir como exp(-x)",
         keywords=["nan"], rec_keywords=["exp(-x)", "punto"]),
    dict(id="T11", name="Desbordamiento numérico (caso D)", metodo="Newton-Raphson",
         fx="exp(x)", x0="1000", lo=-10.0, hi=10.0,
         kinds={ra.OVERFLOW}, expected="exp(1000) supera 1.8·10^308; usar valores menores o reescalar",
         keywords=["desbordamiento"], rec_keywords=["pequeños", "reescala"], bad_point=[1000.0]),
    dict(id="T12", name="Desbordamiento durante las iteraciones", metodo="Grad. Conjugado",
         fx="-x**3", x0="1", lo=-5.0, hi=5.0,
         kinds={ra.OVERFLOW}, expected="los valores crecen sin límite (OverflowError); acotar el rango",
         keywords=["desbordamiento"], rec_keywords=["acota", "rango"]),
]


def _app():
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:  # pragma: no cover
        from PyQt6.QtWidgets import QApplication  # type: ignore
    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    app.processEvents()
    return bool(cond())


def _voice_status(entry, real: bool) -> tuple:
    """(aprobado, descripción). Simulado: solo 'solicitada'. Real: exige inicio y fin."""
    if not entry or not entry.get("speech_requested"):
        return False, "reproducción NO solicitada"
    if entry.get("tts_error"):
        return False, f"error TTS: {entry['tts_error']}"
    if not real:
        return True, "reproducción solicitada al TTS de prueba (sin audio real)"
    if entry.get("t_started") and entry.get("t_finished"):
        dur = entry["t_finished"] - entry["t_started"]
        return True, f"reproducción iniciada y finalizada ({dur:.1f} s de audio)"
    if entry.get("t_started"):
        return False, "reproducción iniciada pero no finalizó"
    return False, "solicitada, pero el motor TTS nunca notificó el inicio"


def _set_inputs(win, sc):
    win.cmb.setCurrentText(sc["metodo"])
    win.edt_fx.setText(sc["fx"])
    win.edt_gx.setText("")
    win.edt_vars.setText("")
    win.edt_x0.setText(sc["x0"])
    win.smin.setValue(sc["lo"]); win.smax.setValue(sc["hi"])


def _execute(app, win, ctrl, real):
    n_log = len(ctrl.log)
    win.ejecutar()
    _wait(app, lambda: len(ctrl.log) > n_log, 5)
    entry = ctrl.log[-1] if len(ctrl.log) > n_log else None
    if entry:
        _wait(app, lambda: "explanation" in entry.get("spoken_parts", []) or
              entry.get("ai_status") == SILENT or
              (entry["explanation"] and not ctrl.voice_enabled), 70 if real else 5)
        if real and entry["speech_requested"]:
            _wait(app, lambda: entry.get("t_finished") or entry.get("tts_error"), 180)
    return entry


def _run_scenario(app, win, ctrl, explainer, sc, real: bool) -> dict:
    t0 = time.time()
    checks = {}
    spoken = getattr(ctrl.speaker, "spoken", None)
    n_spoken = len(spoken) if spoken is not None else 0
    n_prompts = len(getattr(explainer, "prompts", []) or [])
    _set_inputs(win, sc)
    entry = _execute(app, win, ctrl, real)
    ev = entry["event"] if entry else None

    checks["motor recibió la función"] = bool(ev and ev.fx == sc["fx"] and ev.metodo == sc["metodo"])
    checks["tipo de evento"] = bool(ev and ev.kind in sc["kinds"])
    if ev and sc.get("check_values"):
        x_exp, f_exp = sc["check_values"]
        checks["valores del motor (x*, f*)"] = (ev.x_opt == x_exp and ev.f_opt is not None
                                               and abs(ev.f_opt - f_exp) < 1e-6
                                               and ev.converged is True)
    if ra.SUCCESS in sc["kinds"]:
        # Resultado correcto: la IA no interviene por su cuenta (regla del usuario)
        box = getattr(win, "result_box", None)
        expl = entry["explanation"] if entry else ""
        checks["la IA no se consulta"] = bool(entry and entry["ai_status"] == SILENT and (
            real or len(getattr(explainer, "prompts", []) or []) == n_prompts))
        checks["sin voz automática"] = bool(entry and not entry["speech_requested"] and (
            spoken is None or len(spoken) == n_spoken))
        checks["el chat no se abre solo"] = bool(entry and not entry.get("t_chat_open"))
        checks["recuadro con el resumen del motor"] = bool(
            box is not None and box.last_event is ev and box.last_explanation == expl
            and "Resumen del motor" in box.lbl_ai.text())
        checks["se presenta como éxito"] = win.lbl_st.text().startswith("✓")
        obtained = (f"{ev.title}: x*={ev.x_opt}, f*={ra._fmt_num(ev.f_opt)}, "
                    f"iters={ev.iterations}" if ev else "sin evento")
        return _record(sc["id"], sc["name"], sc["fx"], sc["metodo"], sc["expected"], obtained,
                       ev.kind if ev else "—", expl, "", "sin voz (no interviene)", checks, t0,
                       source=entry["source"] if entry else "",
                       ai_status=entry["ai_status"] if entry else "")
    if sc.get("bad_point"):
        checks["punto problemático identificado"] = bool(ev and ev.point == sc["bad_point"])
    if sc.get("no_exception"):
        checks["distingue no-convergencia de excepción"] = bool(
            ev and not ev.exception_type and ev.iterations > 0)
    expl = entry["explanation"] if entry else ""
    if real:
        # La IA real redacta libremente: se exige que responda y que use datos del evento
        checks["IA real respondió"] = bool(entry and entry["source"] == "ia"
                                           and entry["ai_status"] == "ok")
        checks["explicación generada"] = len(expl) > 40
    else:
        low = expl.lower()
        checks["explicación contextual"] = bool(expl) and all(k.lower() in low
                                                              for k in sc["keywords"])
        checks["IA recibió el evento"] = bool(explainer.prompts and sc["fx"] in explainer.prompts[-1])
    rec = ev.recommendation if ev else ""
    if sc.get("rec_keywords"):
        checks["recomendación concreta"] = bool(rec) and any(k.lower() in rec.lower()
                                                             for k in sc["rec_keywords"])
    voice_ok, voice_state = _voice_status(entry, real)
    if spoken is not None:
        voice_ok = voice_ok and len(spoken) > n_spoken and spoken[-1] == expl
    checks["voz"] = voice_ok
    box = getattr(win, "result_box", None)
    checks["interfaz recibió el evento"] = bool(
        box is not None and ev is not None and box.last_event is ev
        and box.isVisibleTo(box.parentWidget()) and box.last_explanation == expl)
    if ra.SUCCESS not in sc["kinds"]:
        checks["no se presenta como éxito"] = not win.lbl_st.text().startswith("✓")
    if ev is not None and ev.kind in ra.CHAT_KINDS:
        checks.update(_chat_checks(win, entry, expl))

    obtained = "sin evento"
    if ev and (ev.kind == ra.SUCCESS or ev.kind in ra.NON_CONVERGED):
        obtained = (f"{ev.title}: x*={ev.x_opt}, f*={ra._fmt_num(ev.f_opt)}, "
                    f"iters={ev.iterations}, ‖∇f‖={ra._fmt_num(ev.grad_norm)}")
    elif ev:
        obtained = f"{ev.title}: {ev.error_message}"
    return _record(sc["id"], sc["name"], sc["fx"], sc["metodo"], sc["expected"], obtained,
                   ev.kind if ev else "—", expl, rec, voice_state, checks, t0,
                   source=entry["source"] if entry else "",
                   ai_status=entry["ai_status"] if entry else "")


_INVITE = ("pregunt", "escríbeme", "escribeme", "micrófono", "microfono", "chat",
           "consúltame", "consultame", "duda")


def _chat_checks(win, entry, expl) -> dict:
    """Etapas de Opti + chat desplegado + sincronización voz/chat."""
    p = win.ai_panel
    msgs = list(getattr(p, "opti_messages", []))
    greet = entry.get("greeting", "") if entry else ""
    parts = entry.get("spoken_parts", []) if entry else []
    c = {
        "chat desplegado automáticamente": bool(p is not None and win.btn_ai.isChecked()
                                                and not p.isHidden()),
        "saludo de Opti generado": bool(greet),
        "saludo escrito en el chat": ("greeting", greet) in msgs,
        "explicación + recomendación en el chat": ("explanation", expl) in msgs,
        "invitación final a seguir consultando": any(k in expl.lower()[-260:] for k in _INVITE),
        "voz en orden saludo → explicación": parts[:2] == ["greeting", "explanation"],
        "chat abierto al iniciar la voz": bool(entry and entry.get("t_chat_open")
                                               and entry.get("t_requested")
                                               and entry["t_chat_open"] <= entry["t_requested"]),
        "explicación en el historial (contexto)": bool(
            p is not None and p._messages and p._messages[-1].get("content") == expl),
    }
    chats = [e for e in win.intervention.log if e.get("chat")]
    if chats and chats[0] is entry:
        c["primera intervención de la sesión: «Soy Opti»"] = "Soy Opti" in greet
    return c


def _record(id_, name, fx, metodo, expected, obtained, kind, expl, rec, voice, checks, t0,
            source="", ai_status="") -> dict:
    return {"id": id_, "prueba": name, "funcion": fx, "metodo": metodo, "esperado": expected,
            "obtenido": obtained, "evento": kind, "explicacion": expl, "recomendacion": rec,
            "fuente": source, "estado_ia": ai_status, "voz": voice, "checks": checks,
            "duracion_s": round(time.time() - t0, 2),
            "estado": "APROBADO" if checks and all(checks.values()) else "FALLIDO"}


# ─── T6: indisponibilidad de la IA ───────────────────────────────────────────
def _test_ai_unavailable(app, win, ctrl, real) -> dict:
    from app.infrastructure.auto_intervention import MockAIExplainer
    t0 = time.time()
    prev_ai, prev_enabled = ctrl._ai, ctrl.ai_enabled
    ctrl.set_backends(explainer=MockAIExplainer(fail=True))
    ctrl.ai_enabled = True
    sc = SCENARIOS[1]          # error de dominio: Opti sí consulta la IA
    _set_inputs(win, sc)
    t_exec = time.time()
    entry = _execute(app, win, ctrl, real)
    ui_time = time.time() - t_exec
    ev = entry["event"] if entry else None
    box = win.result_box
    checks = {
        "el motor conserva el diagnóstico": bool(ev and ev.kind in sc["kinds"]),
        "estado ai_comm_error registrado": bool(entry and entry["ai_status"].startswith(
            "ai_comm_error")),
        "no se confirma falsamente la IA": bool(entry and entry["source"] == "local"),
        "mensaje alternativo en pantalla": "Error de comunicación con la IA" in box.lbl_ai.text(),
        "la interfaz no se bloquea": ui_time < (150 if real else 5),
    }
    voice_ok, voice = _voice_status(entry, real)
    checks["voz con la explicación alternativa"] = voice_ok
    ctrl._ai, ctrl.ai_enabled = prev_ai, prev_enabled
    return _record("T6", "Error de comunicación con la IA", sc["fx"], sc["metodo"],
                   "diagnóstico intacto + mensaje alternativo, sin fingir respuesta de IA",
                   f"{ev.title if ev else '—'} · IA: {entry['ai_status'] if entry else '—'}",
                   ra.AI_COMM_ERROR, entry["explanation"] if entry else "", "",
                   voice, checks, t0, source=entry["source"] if entry else "",
                   ai_status=entry["ai_status"] if entry else "")


# ─── Robustez (modo simulado) ────────────────────────────────────────────────
def _ev(fx="x**2"):
    """Evento que sí dispara a Opti (los resultados correctos no lo hacen)."""
    return _problem_event(fx)


def _test_no_overlap(app, ctrl) -> dict:
    t0 = time.time()
    sp = ctrl.speaker
    n_spoken, n_stops = len(sp.spoken), sp.stops
    ctrl.submit(_ev("x**2 + 1"))
    ctrl.submit(_ev("x**2 + 7"))
    _wait(app, lambda: len(sp.spoken) > n_spoken, 3)
    _wait(app, lambda: False, 0.6)
    _wait(app, lambda: False, 0.8)
    new = sp.spoken[n_spoken:]
    checks = {"solo se lee la última intervención": bool(new) and
              ctrl.log[-1]["explanation"] in new and
              all(t != ctrl.log[-2]["explanation"] for t in new),
              "explicación obsoleta nunca leída": (ctrl.log[-1]["speech_requested"]
                                                   and "explanation" not in
                                                   ctrl.log[-2].get("spoken_parts", [])),
              "voz anterior detenida": sp.stops > n_stops}
    return _record("T7", "Ejecuciones seguidas sin superposición", "—", "—",
                   "solo se lee el último resultado", f"{len(new)} reproducción(es)", "—",
                   "", "", "—", checks, t0)


def _test_controls(app, win, ctrl) -> dict:
    t0 = time.time()
    sp = ctrl.speaker
    win.result_box.btn_voice.setChecked(False)
    n = len(sp.spoken)
    ctrl.submit(_ev())
    _wait(app, lambda: bool(ctrl.log[-1]["explanation"]), 3)
    _wait(app, lambda: False, 0.5)
    checks = {"texto disponible con voz desactivada": bool(win.result_box.last_explanation),
              "no se reproduce con voz desactivada": len(sp.spoken) == n}
    win.result_box.btn_voice.setChecked(True)
    n = len(sp.spoken)
    win.result_box.btn_repeat.click()
    checks["botón repetir"] = len(sp.spoken) == n + 1
    win.result_box.btn_stop.setEnabled(True)
    win.result_box.btn_stop.click()
    checks["botón detener"] = not sp.is_speaking()
    return _record("T8", "Controles de voz (on/off, repetir, detener)", "—", "—",
                   "controles operativos", "controles", "—", "", "", "—", checks, t0)


def _test_parameter_event(app, win, ctrl) -> dict:
    t0 = time.time()
    n = len(ctrl.log)
    win.cmb.setCurrentText("Fibonacci")
    win.edt_fx.setText("x**2"); win.smin.setValue(0.0); win.smax.setValue(5.0)
    win.edt_tol.setText("-1")
    win.ejecutar()
    win.edt_tol.setText("0.1")
    _wait(app, lambda: len(ctrl.log) > n, 3)
    ev = ctrl.log[-1]["event"] if len(ctrl.log) > n else None
    checks = {"parámetro incorrecto detectado": bool(ev and ev.kind == ra.PARAMETER_ERROR),
              "recomendación": bool(ev and "positiva" in ev.recommendation)}
    return _record("T9", "Parámetros incorrectos (tolerancia negativa)", "x**2", "Fibonacci",
                   "evento parameter_error", ev.error_message if ev else "sin evento",
                   ev.kind if ev else "—", "", ev.recommendation if ev else "", "—", checks, t0)


def _test_detector(app) -> dict:
    """Detector puro: ±∞, NaN, desbordamiento y que un negativo válido NO es error."""
    t0 = time.time()
    inf, nan = float("inf"), float("nan")

    def cr(rows, fx="x**2", x0=(0.0,)):
        return ra.classify_result("Newton-Raphson", fx, ["x"], list(x0), -5, 5, rows).kind
    checks = {
        "infinito positivo → infinite_result": cr([{"x_k": 1.0, "f_k": inf}]) == ra.INFINITE_RESULT,
        "infinito negativo → infinite_result": cr([{"x_k": 1.0, "f_k": -inf}]) == ra.INFINITE_RESULT,
        "f → −∞ (no acotada) → infinite_result": cr([{"x_k": 9.0, "f_k": -1e15,
                                                     "grad_norm": 0.0}]) == ra.INFINITE_RESULT,
        "NaN → nan_result": cr([{"x_k": 1.0, "f_k": nan}]) == ra.NAN_RESULT,
        "∞ por magnitud excesiva → overflow": cr([{"x_k": 1000.0, "f_k": inf}], "exp(x)",
                                                 (1000.0,)) == ra.OVERFLOW,
        "OverflowError del motor → overflow": ra.classify_exception(
            "Grad. Conjugado", "-x**3", ["x"], [1.0], -5, 5,
            OverflowError(34, "Result too large")).kind == ra.OVERFLOW,
        "negativo válido NO es error": cr([{"x_k": 0.0, "f_k": -5.0, "grad_norm": 0.0,
                                            "converged": True}], "x**2 - 5") == ra.SUCCESS,
        "mínimo local NO es inválido": cr([{"x_k": 1.0, "f_k": -0.5, "grad_norm": 1e-9,
                                            "converged": True}], "x**4 - x**2") == ra.SUCCESS,
    }
    return _record("T13", "Detector: infinito / NaN / desbordamiento / negativos válidos",
                   "—", "detector", "cada condición con su tipo", "8 casos", "—", "", "", "—",
                   checks, t0)


def _problem_event(fx="1/(x-2)", kind=None):
    return ra.OptimizationEvent(kind=kind or ra.DIVISION_BY_ZERO, metodo="Newton-Raphson",
                                fx=fx, vars_=["x"], x0=[2.0], lo=0.0, hi=5.0, point=[2.0],
                                error_message="La función no se puede evaluar en x = 2.",
                                cause="En x = 2 el denominador x - 2 vale 0.",
                                recommendation="Evita x = 2.")


def _test_identity(app, win, ctrl) -> dict:
    """Opti se presenta solo en la primera intervención de la sesión."""
    t0 = time.time()
    ctrl.presented = False
    ctrl.submit(_problem_event("1/(x-2)"))
    g1 = ctrl.log[-1]["greeting"]
    _wait(app, lambda: bool(ctrl.log[-1]["explanation"]), 3)
    ctrl.submit(_problem_event("1/(x-3)"))
    g2 = ctrl.log[-1]["greeting"]
    _wait(app, lambda: bool(ctrl.log[-1]["explanation"]), 3)
    checks = {"primera intervención: se presenta como Opti": "Soy Opti" in g1,
              "saludo natural (pregunta cómo estás / juntos)": ("¿Cómo estás?" in g1
                                                                 or "juntos" in g1),
              "no culpa al usuario": not any(w in g1.lower() for w in ("culpa", "te equivocaste",
                                                                      "error tuyo")),
              "siguientes: saludo breve sin repetir la presentación": bool(g2) and
                  "Soy Opti" not in g2 and len(g2) < len(g1)}
    return _record("T14", "Identidad de Opti (presentación una vez por sesión)", "—", "—",
                   "«Soy Opti…» solo la primera vez", g1[:60] + "… / " + g2[:40] + "…", "—",
                   g1, "", "—", checks, t0)


class _FakeWorkerFactory:
    """Sustituye a _AIWorker del chat en modo simulado (no hay red)."""
    def __init__(self):
        self.calls: list = []

    def __call__(self, key, key2, messages, system, parent=None):
        from PyQt5.QtCore import QObject, QTimer, pyqtSignal
        factory = self

        class _W(QObject):
            response_ready = pyqtSignal(str)
            error_occurred = pyqtSignal(str)
            provider_used = pyqtSignal(str)

            def start(self_):
                factory.calls.append({"messages": messages, "system": system})
                last = messages[-1]["content"]
                QTimer.singleShot(0, lambda: self_.response_ready.emit(
                    f"[Opti simulado] Sobre tu pregunta «{last[:50]}»: otra forma es "
                    f"cambiar el punto inicial."))

            def isRunning(self_):
                return False
        return _W()


def _test_chat_queries(app, win, ctrl, real) -> dict:
    """Caso E: consulta posterior por texto y por voz con contexto."""
    import app.infrastructure.ai_assistant as aa
    t0 = time.time()
    p = win.ai_panel
    checks = {}
    if p is None:
        return _record("T15", "Consultas posteriores (texto y voz)", "—", "—", "—",
                       "sin panel", "—", "", "", "—", {"panel de chat": False}, t0)
    # Intervención previa (división entre cero) para que haya contexto
    sc = SCENARIOS[2]
    _set_inputs(win, sc)
    entry = _execute(app, win, ctrl, real)
    expl = entry["explanation"] if entry else ""
    spoken_calls = []
    orig_speak, orig_worker, orig_key = p._speak_response, aa._AIWorker, p._api_key
    p._speak_response = lambda text: spoken_calls.append(text)
    fake = None
    if not real:
        fake = _FakeWorkerFactory()
        aa._AIWorker = fake
        p._api_key = p._api_key or "sk-or-simulada"
    stops_before = getattr(ctrl.speaker, "stops", 0)
    try:
        n_hist = len(p._messages)
        q1 = "¿Puedes explicarme otra forma de resolverlo?"
        p.edt_msg.setText(q1)
        p._on_send()                                   # entrada por TEXTO
        _wait(app, lambda: not p._busy and len(p._messages) >= n_hist + 2, 90 if real else 5)
        r1 = p._messages[-1]["content"] if p._messages else ""
        checks["consulta por texto recibida"] = any(m.get("content") == q1 for m in p._messages)
        checks["Opti respondió por texto"] = (p._messages[-1]["role"] == "assistant"
                                              and len(r1) > 10)
        hist1 = (fake.calls[-1]["messages"] if fake and fake.calls else p._messages)
        joined = " ".join(m.get("content", "") for m in hist1)
        checks["contexto: la IA recibe el evento y la explicación previa"] = (
            sc["fx"] in joined and expl[:60] in joined)
        checks["respuesta del chat también en voz"] = bool(spoken_calls)
        checks["Opti deja de hablar cuando el usuario pregunta"] = bool(
            entry and entry.get("user_took_over")) and (
            getattr(ctrl.speaker, "stops", 1) > stops_before or real)
        # Entrada por VOZ: mismo slot que emite el reconocimiento (_STTWorker.text_ready)
        q2 = "¿Qué puedo hacer para corregirlo?"
        n_hist = len(p._messages)
        p._on_transcription_ready(q2)
        _wait(app, lambda: not p._busy and len(p._messages) >= n_hist + 2, 90 if real else 5)
        checks["consulta por voz (transcrita) recibida"] = any(
            m.get("content") == q2 for m in p._messages)
        hist2 = (fake.calls[-1]["messages"] if fake and fake.calls else p._messages)
        checks["continuidad: conserva la pregunta y respuesta anteriores"] = (
            any(m.get("content") == q1 for m in hist2)
            and any(m.get("content") == r1 for m in hist2))
    finally:
        p._speak_response, aa._AIWorker, p._api_key = orig_speak, orig_worker, orig_key
    return _record("T15", "Consultas posteriores por texto y voz (caso E)", sc["fx"],
                   sc["metodo"], "respuesta contextual sin volver a pedir datos",
                   (p._messages[-1]["content"][:120] if p._messages else "—"), "—",
                   "", "", "—", checks, t0)


def _test_sync_interrupt(app, win, ctrl) -> dict:
    """Cola saludo→explicación, sin superposición, interrupciones y duplicados."""
    from app.infrastructure.auto_intervention import MockSpeaker
    t0 = time.time()
    prev = ctrl.speaker
    sp = MockSpeaker(ctrl, hold=True)
    ctrl.set_backends(speaker=sp)
    checks = {}
    try:
        ctrl.presented = True
        ctrl.submit(_problem_event("1/(x-4)"))
        e1 = ctrl.log[-1]
        _wait(app, lambda: bool(e1["explanation"]), 3)
        _wait(app, lambda: False, 0.5)
        checks["mientras saluda, la explicación espera (sin superponer)"] = (
            sp.spoken == [e1["greeting"]] and len(ctrl._queue) == 1)
        sp.finish()
        _wait(app, lambda: len(sp.spoken) == 2, 2)
        checks["al terminar el saludo, se lee la explicación"] = (
            len(sp.spoken) == 2 and sp.spoken[1] == e1["explanation"])
        # Nueva intervención mientras habla → la anterior se corta
        stops = sp.stops
        ctrl.submit(_problem_event("1/(x-5)"))
        e2 = ctrl.log[-1]
        _wait(app, lambda: False, 0.2)
        checks["nuevo evento detiene la voz anterior"] = sp.stops > stops
        checks["se habla el saludo del evento nuevo"] = sp.spoken[-1] == e2["greeting"]
        _wait(app, lambda: bool(e2["explanation"]), 3)
        # Detener desde el chat: no lee lo pendiente y el chat sigue abierto
        win.ai_panel.btn_player_stop.setEnabled(True)
        win.ai_panel.btn_player_stop.click()
        n = len(sp.spoken)
        _wait(app, lambda: False, 0.5)
        checks["detener (⏹ del chat) descarta lo pendiente"] = (not ctrl._queue
                                                                and len(sp.spoken) == n)
        checks["el chat sigue abierto tras detener"] = not win.ai_panel.isHidden()
        # Respuesta obsoleta: la IA contesta TARDE a un evento ya reemplazado
        class _SlowAI:
            name = "ia-simulada"
            def __init__(self): self.pending = []
            def available(self): return True
            def explain(self, event, on_ok, on_fail): self.pending.append(on_ok)
        slow, prev_ai = _SlowAI(), ctrl._ai
        ctrl.set_backends(explainer=slow)
        ctrl.submit(_problem_event("1/(x-7)"))
        old_entry = ctrl.log[-1]
        ctrl.submit(_problem_event("1/(x-8)"))
        n_msgs = len(win.ai_panel.opti_messages)
        slow.pending[0]("respuesta antigua que no debe aparecer")
        _wait(app, lambda: False, 0.4)
        checks["respuestas antiguas descartadas"] = (
            not old_entry["explanation"] and all(t != "respuesta antigua que no debe aparecer"
                                                 for _, t in win.ai_panel.opti_messages[n_msgs:]))
        ctrl.set_backends(explainer=prev_ai)
        # Duplicado: el mismo evento dos veces seguidas
        n_log = len(ctrl.log)
        ev = _problem_event("1/(x-6)")
        ctrl.submit(ev)
        ctrl.submit(_problem_event("1/(x-6)"))
        checks["sin intervención duplicada para el mismo evento"] = len(ctrl.log) == n_log + 1
    finally:
        ctrl.set_backends(speaker=prev)
    return _record("T16", "Sincronización voz/chat e interrupciones", "—", "—",
                   "una voz a la vez, en orden, cancelable", f"{len(sp.spoken)} locuciones",
                   "—", "", "", "—", checks, t0)


# ─── Generación de audio real (archivo WAV) ──────────────────────────────────
def _test_audio_file(app) -> dict:
    t0 = time.time()
    path = os.path.join(tempfile.gettempdir(), "optimizador_tts_check.wav")
    result = {}

    def work():
        try:
            try:
                import comtypes
                comtypes.CoInitialize()
            except Exception:
                pass
            import pyttsx3
            from app.infrastructure.ai_assistant import _find_spanish_voice
            eng = pyttsx3.init()
            vid = _find_spanish_voice(eng)
            if vid:
                eng.setProperty("voice", vid)
            eng.save_to_file("El mínimo se encuentra en x igual a tres.", path)
            eng.runAndWait()
            result["voice"] = bool(vid)
        except Exception as ex:
            result["error"] = str(ex)

    if os.path.exists(path):
        os.remove(path)
    th = threading.Thread(target=work, daemon=True)
    th.start()
    _wait(app, lambda: not th.is_alive(), 30)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    checks = {"voz en español instalada": bool(result.get("voice")),
              "audio sintetizado (WAV > 0 bytes)": size > 1000}
    return _record("T0", "Generación de audio TTS real", "—", "SAPI5/pyttsx3",
                   "archivo WAV generado", result.get("error") or f"WAV de {size} bytes",
                   "—", "", "", f"audio generado: {size} bytes", checks, t0)


# ─── Punto de entrada ────────────────────────────────────────────────────────
def run_ai_intervention_suite(real: bool = False, verbose: bool = True) -> List[dict]:
    if not real:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = _app()
    try:
        from PyQt5.QtWidgets import QMessageBox
    except ImportError:  # pragma: no cover
        from PyQt6.QtWidgets import QMessageBox  # type: ignore
    import app.ui.main_window as mw
    from app.infrastructure.auto_intervention import (
        MockAIExplainer, MockSpeaker, PanelAIExplainer, QtTTSSpeaker)

    # Los diálogos modales de la app no deben bloquear la prueba
    orig = (QMessageBox.warning, QMessageBox.critical, QMessageBox.information, mw._err)
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.critical = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    mw._err = lambda *a, **k: None
    results: List[dict] = []
    meta = {"modo": "REAL" if real else "SIMULADO"}
    try:
        win = mw.InterfazOptimizacion()
        ctrl = win.intervention
        assert ctrl is not None, "El controlador de intervención no se inicializó"
        if real:
            explainer = PanelAIExplainer(win.ai_panel)
            meta["ia"] = ("API Key configurada" if explainer.available()
                          else "SIN API Key configurada")
            ctrl.set_backends(explainer=explainer, speaker=QtTTSSpeaker(win.ai_panel, ctrl))
            results.append(_test_audio_file(app))
        else:
            explainer = MockAIExplainer()
            ctrl.set_backends(explainer=explainer, speaker=MockSpeaker(ctrl))
        ctrl.ai_enabled = True
        ctrl.set_voice_enabled(True)
        win.result_box.btn_voice.setChecked(True)

        for sc in SCENARIOS:
            try:
                results.append(_run_scenario(app, win, ctrl, explainer, sc, real))
            except Exception as ex:
                import traceback
                r = _record(sc["id"], sc["name"], sc["fx"], sc["metodo"], sc["expected"],
                            f"EXCEPCIÓN {ex}", "—", "", "", "—", {"sin excepción": False},
                            time.time())
                r["traceback"] = traceback.format_exc()
                results.append(r)
        results.append(_test_ai_unavailable(app, win, ctrl, real))
        results.append(_test_chat_queries(app, win, ctrl, real))
        if not real:
            results.append(_test_detector(app))
            results.append(_test_identity(app, win, ctrl))
            results.append(_test_sync_interrupt(app, win, ctrl))
            results.append(_test_no_overlap(app, ctrl))
            results.append(_test_controls(app, win, ctrl))
            results.append(_test_parameter_event(app, win, ctrl))
        ctrl.stop()
        win.close()
    finally:
        (QMessageBox.warning, QMessageBox.critical, QMessageBox.information, mw._err) = (
            staticmethod(orig[0]), staticmethod(orig[1]), staticmethod(orig[2]), orig[3])

    results.append({"_meta": meta})
    if verbose:
        print(format_report(results))
    return results


# Alias con el nombre solicitado (la versión pytest vive en tests/)
def test_ai_automatic_intervention(real: bool = False, verbose: bool = True) -> List[dict]:
    return run_ai_intervention_suite(real=real, verbose=verbose)


def scenario_results(results: List[dict]) -> List[dict]:
    return [r for r in results if "_meta" not in r]


def format_report(results: List[dict]) -> str:
    meta = next((r["_meta"] for r in results if "_meta" in r), {})
    rows = scenario_results(results)
    out = [f"MODO: {meta.get('modo', '?')}"
           + (" — IA simulada + TTS de prueba (no se consume API ni se reproduce audio)"
              if meta.get("modo") == "SIMULADO" else
              f" — IA real ({meta.get('ia', '')}) + voz SAPI5 real")]
    for r in rows:
        out.append("=" * 78)
        out.append(f"[{r['id']}] {r['prueba']}  →  {r['estado']}   ({r['duracion_s']} s)")
        for k, lbl in (("funcion", "función"), ("metodo", "método"), ("esperado", "esperado"),
                       ("obtenido", "obtenido"), ("evento", "evento"), ("fuente", "fuente"),
                       ("estado_ia", "estado IA"), ("recomendacion", "recomendación"),
                       ("voz", "estado TTS")):
            if r.get(k) not in ("", "—", None):
                out.append(f"  {lbl:<14}: {r[k]}")
        if r.get("explicacion"):
            out.append(f"  explicación   : {r['explicacion']}")
        for k, v in r["checks"].items():
            out.append(f"    [{'OK' if v else 'XX'}] {k}")
        if r.get("traceback"):
            out.append(r["traceback"])
    ok = sum(r["estado"] == "APROBADO" for r in rows)
    failed = [r for r in rows if r["estado"] != "APROBADO"]
    ai_rows = [r for r in rows if r["fuente"]]
    if meta.get("modo") == "REAL":
        ia_ok = [r for r in ai_rows if r["fuente"] == "ia"]
        ia_state = (f"REAL — respondió en {len(ia_ok)}/{len(ai_rows) - 1} escenarios "
                    f"(T6 fuerza la caída a propósito)")
        v_ok = sum("finalizada" in r["voz"] for r in rows)
        voice_state = f"REAL — reproducción iniciada y finalizada en {v_ok} prueba(s)"
    else:
        ia_state = "SIMULADA — prompts recibidos y respuestas entregadas"
        voice_state = "SIMULADA — reproducción solicitada (sin audio real)"
    out += ["=" * 78,
            f"TOTAL: {len(rows)}   APROBADAS: {ok}   FALLIDAS: {len(failed)}",
            "Errores: " + ("ninguno" if not failed else
                           "; ".join(f"{r['id']} ({', '.join(k for k, v in r['checks'].items() if not v)})"
                                     for r in failed)),
            f"Integración IA : {ia_state}",
            f"Integración voz: {voice_state}"]
    return "\n".join(out)
