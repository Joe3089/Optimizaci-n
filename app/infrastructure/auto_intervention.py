# -*- coding: utf-8 -*-
"""
auto_intervention.py — Intervención automática de la IA (texto + voz).

Flujo:  motor → detector (result_analysis) → AutoInterventionController
        → [IA: explicación contextual]  → TTS  → UI (ResultBox)

No es un segundo asistente: reutiliza el `_AIWorker` (Groq/OpenRouter, con
las mismas API Keys guardadas por el AIAssistantPanel) y el `_TTSWorker`
(SAPI5/pyttsx3) del panel existente. Los backends son sustituibles
(`set_backends`) para que las pruebas usen IA simulada y TTS de prueba.

Garantías:
  • El resultado se muestra ANTES de pedir la explicación; la red y el audio
    corren en QThread, nunca en el hilo de la UI.
  • Cada evento tiene un número de generación: respuestas o voces de un
    cálculo anterior se descartan y la voz anterior se detiene antes de
    empezar la nueva (nunca dos lecturas simultáneas).
  • Si la IA falla o tarda demasiado → estado `ai_comm_error` y explicación
    local; nunca se presenta como si la IA hubiese respondido.
  • Estados de voz diferenciados: texto listo → reproducción solicitada →
    reproducción iniciada (evento real del motor TTS) → finalizada.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

try:
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal as Signal
except ImportError:  # pragma: no cover
    from PyQt6.QtCore import QObject, QTimer, pyqtSignal as Signal  # type: ignore

from app.application.result_analysis import (
    AI_SYSTEM_PROMPT, CHAT_KINDS, OptimizationEvent, build_ai_prompt, ensure_invitation,
    local_explanation, opti_greeting,
)

AI_TIMEOUT_MS = 60000
SPEECH_DELAY_MS = 300
DUPLICATE_WINDOW_S = 2.0
# Estado de un resultado correcto: la IA no interviene por su cuenta.
SILENT = "sin intervención (resultado correcto)"   # el mismo evento repetido en < 2 s no vuelve a intervenir

# Estados de voz emitidos por speech_state
TEXT_READY = "text_ready"
REQUESTED = "requested"
PLAYING = "playing"
FINISHED = "finished"
STOPPED = "stopped"
OFF = "off"


# ═════════════════════════ Backends de explicación ═══════════════════════════
class MockAIExplainer:
    """IA simulada para pruebas: registra el prompt recibido y responde con
    una explicación contextual derivada del evento (sin API externa)."""
    name = "ia-simulada"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.prompts: list = []

    def available(self) -> bool:
        return True

    def explain(self, event, on_ok, on_fail):
        self.prompts.append(build_ai_prompt(event))
        if self.fail:
            QTimer.singleShot(0, lambda: on_fail("servicio de IA simulado no disponible"))
            return
        text = "[IA simulada] " + local_explanation(event)
        QTimer.singleShot(0, lambda: on_ok(text))


class PanelAIExplainer:
    """Usa el proveedor real del AIAssistantPanel (mismas keys y _AIWorker,
    que ya hace fallback Groq → OpenRouter en su propio QThread)."""
    name = "ia"

    def __init__(self, panel):
        self.panel = panel
        self._workers: list = []
        self._cache: dict = {}   # mismo evento -> misma explicación (ahorra cupo)

    @staticmethod
    def _cache_key(ev):
        return (ev.kind, ev.metodo, ev.fx, str(ev.x0), ev.lo, ev.hi, str(ev.x_opt),
                str(ev.f_opt), ev.iterations, ev.error_message)

    def available(self) -> bool:
        return bool(self.panel is not None and getattr(self.panel, "_api_key", ""))

    MAX_ATTEMPTS = 3   # igual que el chat manual: los modelos gratuitos fallan a ratos

    def explain(self, event, on_ok, on_fail, attempt: int = 1):
        from app.infrastructure.ai_assistant import _AIWorker, is_quota_or_block_error
        ck = self._cache_key(event)
        if ck in self._cache:
            text = self._cache[ck]
            QTimer.singleShot(0, lambda: on_ok(text))
            return
        w = _AIWorker(self.panel._api_key, getattr(self.panel, "_api_key_backup", ""),
                      [{"role": "user", "content": build_ai_prompt(event)}],
                      AI_SYSTEM_PROMPT)

        def _err(msg):
            auth = any(x in msg for x in ("inválida", "401"))
            hard = auth or is_quota_or_block_error(msg)   # reintentar solo gastaría cupo
            if not hard and attempt < self.MAX_ATTEMPTS:
                QTimer.singleShot(700 * attempt,
                                  lambda: self.explain(event, on_ok, on_fail, attempt + 1))
            else:
                on_fail(f"{msg} (tras {attempt} intento(s))")

        def _ok(text):
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[ck] = text
            on_ok(text)

        w.response_ready.connect(_ok)
        w.error_occurred.connect(_err)
        w.finished.connect(lambda w=w: self._workers.remove(w) if w in self._workers else None)
        self._workers.append(w)
        w.start()


# ═════════════════════════════ Backends de voz ═══════════════════════════════
class MockSpeaker(QObject):
    """TTS de prueba: confirma que la reproducción fue SOLICITADA. No emite
    'started/finished' porque no hay audio real: así una prueba simulada
    nunca puede afirmar que el usuario escuchó algo."""
    requested = Signal()
    started = Signal()
    finished = Signal()
    error = Signal(str)

    def __init__(self, parent=None, hold: bool = False):
        super().__init__(parent)
        self.spoken: list = []
        self.stops = 0
        self._active = False
        # hold=False: cada texto se "consume" al instante (señal `done`, no
        # `finished`: no hubo audio). hold=True: queda "hablando" hasta finish().
        self.hold = hold

    done = Signal()

    def speak(self, text: str):
        self.spoken.append(text)
        self._active = True
        self.requested.emit()
        if not self.hold:
            QTimer.singleShot(0, self._consume)

    def _consume(self):
        if self._active:
            self._active = False
            self.done.emit()

    def finish(self):
        """Simula que terminó la locución en curso (solo con hold=True)."""
        self._consume()

    def stop(self):
        self.stops += 1
        self._active = False

    def is_speaking(self) -> bool:
        return self._active


class QtTTSSpeaker(QObject):
    """Voz real con el `_TTSWorker` del asistente. Comparte canal con el
    reproductor del chat: antes de hablar detiene cualquier lectura en curso
    (la suya o la del panel) y espera a que termine, así nunca se superponen."""
    requested = Signal()
    started = Signal()    # SAPI empezó a emitir audio
    finished = Signal()   # la lectura terminó completa
    error = Signal(str)
    done = Signal()       # la locución acabó (completa, detenida o con error)

    def __init__(self, panel=None, parent=None):
        super().__init__(parent)
        self.panel = panel
        self._worker = None
        self._pending: Optional[str] = None
        self._poll = QTimer(self)
        self._poll.setInterval(60)
        self._poll.timeout.connect(self._try_flush)
        self._poll_ticks = 0

    def _running(self) -> list:
        ws = [self._worker, getattr(self.panel, "_tts_worker", None)]
        return [w for w in ws if w is not None and w.isRunning()]

    def speak(self, text: str):
        from app.infrastructure.ai_assistant import _strip_markdown_for_speech
        plain = _strip_markdown_for_speech(text)
        if not plain:
            return
        self._pending = plain
        self._stop_all()
        self._poll_ticks = 0
        self.requested.emit()
        self._try_flush()
        if self._pending is not None:
            self._poll.start()

    def _try_flush(self):
        self._poll_ticks += 1
        if self._running() and self._poll_ticks < 60:   # espera ≤ ~3.6 s
            return
        self._poll.stop()
        text, self._pending = self._pending, None
        if not text:
            return
        from app.infrastructure.ai_assistant import _TTSWorker
        w = _TTSWorker(text, parent=self)
        w.playback_started.connect(self.started.emit)
        w.finished_speaking.connect(self.finished.emit)
        w.error_occurred.connect(self.error.emit)
        # `done` solo si no hay otra locución esperando turno (si no, un worker
        # detenido anunciaría como terminada la locución que aún no empezó)
        w.finished.connect(lambda w=w: self.done.emit()
                           if (w is self._worker and self._pending is None) else None)
        self._worker = w
        p = self.panel
        if p is not None and hasattr(p, "btn_player_stop"):
            # El reproductor del chat controla también la voz de Opti
            try:
                p._tts_worker = w
                p._tts_last_text = text
                p._tts_paused = False
                p.btn_player_playpause.setText("⏸")
                p.btn_player_stop.setEnabled(True)
                w.finished_speaking.connect(p._on_tts_finished)
            except Exception:
                pass
        w.start()

    def _stop_all(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
        p = self.panel
        if p is not None and getattr(p, "_tts_worker", None) is not None \
                and p._tts_worker is not self._worker and p._tts_worker.isRunning():
            try:
                p._on_player_stop()
            except Exception:
                p._tts_worker.request_stop()

    def stop(self):
        self._pending = None
        self._poll.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()

    def is_speaking(self) -> bool:
        return self._pending is not None or (self._worker is not None
                                             and self._worker.isRunning())


# ═══════════════════════════════ Controlador ═════════════════════════════════
class AutoInterventionController(QObject):
    """Recibe eventos del motor y orquesta explicación + voz."""
    event_presented = Signal(object)              # OptimizationEvent (mostrar ya)
    explanation_ready = Signal(object, str, str)  # evento, texto, fuente ("ia"|"local"|…)
    ai_status = Signal(object, str)               # evento, "ok" | "ai_comm_error: …" | …
    speech_state = Signal(str)                    # TEXT_READY/REQUESTED/PLAYING/FINISHED/…
    # Chat de Opti: abrir panel y publicar mensajes (parte: greeting/thinking/explanation)
    chat_open_requested = Signal(object)
    chat_message = Signal(object, str, str)

    def __init__(self, panel=None, parent=None):
        super().__init__(parent)
        self.panel = panel
        self.voice_enabled = True
        self.ai_enabled = True
        self._gen = 0
        self.current_event: Optional[OptimizationEvent] = None
        self.current_explanation = ""
        self.current_source = ""
        self.log: list = []               # registro de intervenciones (pruebas/diagnóstico)
        self.presented = False            # Opti ya se presentó en esta sesión
        self._n_interventions = 0         # para variar saludos e invitaciones
        self._queue: list = []            # [(gen, entry, part, text)] locuciones pendientes
        self._now = None                  # (gen, entry, part) locución en curso
        self._last_sig = None             # (firma, t) para descartar duplicados
        self._ai = PanelAIExplainer(panel)
        self._speaker = None
        self.set_speaker(QtTTSSpeaker(panel, self))

    # ── Inyección de backends (pruebas / integración real) ───────────────────
    def set_backends(self, explainer=None, speaker=None):
        if explainer is not None:
            self._ai = explainer
        if speaker is not None:
            self.set_speaker(speaker)

    def set_speaker(self, speaker):
        if self._speaker is not None:
            try:
                self._speaker.stop()
            except Exception:
                pass
        self._speaker = speaker
        speaker.requested.connect(lambda: self._mark("t_requested", REQUESTED))
        speaker.started.connect(lambda: self._mark("t_started", PLAYING))
        speaker.finished.connect(self._on_part_finished)
        speaker.error.connect(self._on_tts_error)
        if hasattr(speaker, "done"):
            speaker.done.connect(self._on_part_done)

    @property
    def speaker(self):
        return self._speaker

    def _current_entry(self):
        return self.log[-1] if self.log else None

    def _mark(self, key, state):
        e = self._current_entry()
        if e is not None and e.get("speech_requested") and key not in e:
            e[key] = time.time()
        self.speech_state.emit(state)

    def _on_tts_error(self, msg):
        e = self._current_entry()
        if e is not None:
            e["tts_error"] = msg
        self.speech_state.emit(f"error:{msg}")

    def _on_part_finished(self):
        """Audio real terminado. t_finished solo cuando acaba la ÚLTIMA parte
        (la explicación): el saludo solo no cuenta como intervención leída."""
        now = self._now
        if now is not None:
            gen, entry, part = now
            if entry.get("t_started"):
                entry[f"t_finished_{part}"] = time.time()
                if part == "explanation" and "t_finished" not in entry:
                    entry["t_finished"] = time.time()
                    self.speech_state.emit(FINISHED)

    def _on_part_done(self):
        self._now = None
        self._flush_queue()

    def _flush_queue(self):
        while self._queue and self._now is None:
            gen, entry, part, text = self._queue.pop(0)
            if gen == self._gen and self.voice_enabled:
                self._say(gen, entry, part, text)

    def _say(self, gen, entry, part, text):
        """Locución en orden: si Opti está diciendo otra parte del MISMO evento,
        se encola; nunca se superponen dos voces."""
        if gen != self._gen or not self.voice_enabled or not text:
            return
        if self._now is not None and self._now[0] == gen:
            self._queue.append((gen, entry, part, text))
            return
        self._now = (gen, entry, part)
        entry["speech_requested"] = True
        entry.setdefault("spoken_parts", []).append(part)
        try:
            self._speaker.speak(text)
        except Exception as ex:
            self._now = None
            self._on_tts_error(str(ex))

    def cancel_speech(self):
        """Detiene la voz y descarta lo pendiente (sin cerrar el chat)."""
        self._queue.clear()
        self._now = None
        self._speaker.stop()

    @staticmethod
    def _signature(ev):
        return (ev.kind, ev.metodo, ev.fx, tuple(ev.x0 or ()), ev.lo, ev.hi,
                ev.error_message, ev.x_opt)

    # ── Entrada principal ───────────────────────────────────────────────────
    def submit(self, event: OptimizationEvent) -> int:
        sig, now_t = self._signature(event), time.time()
        if self._last_sig and self._last_sig[0] == sig and now_t - self._last_sig[1] < DUPLICATE_WINDOW_S:
            return self._gen                 # mismo evento repetido → sin intervención duplicada
        self._last_sig = (sig, now_t)
        self._gen += 1
        gen = self._gen
        self.cancel_speech()                 # voz desactualizada → fuera
        self.current_event = event
        self.current_explanation = ""
        self.current_source = ""
        chat = event.kind in CHAT_KINDS
        if not chat:
            # Resultado correcto: la IA NO interviene (ni consulta, ni voz, ni chat).
            # El recuadro muestra el resumen del motor; Opti responde solo si el
            # usuario le escribe o le habla.
            text = local_explanation(event)
            entry = {"gen": gen, "event": event, "explanation": text, "source": "local",
                     "ai_status": SILENT, "speech_requested": False, "t_submit": now_t,
                     "greeting": "", "chat": False, "n": self._n_interventions}
            self.log.append(entry)
            self.current_explanation, self.current_source = text, "local"
            self.event_presented.emit(event)
            self.ai_status.emit(event, SILENT)
            self.explanation_ready.emit(event, text, "local")
            return gen
        n = self._n_interventions
        self._n_interventions += 1
        greeting = opti_greeting(event, first=not self.presented, n=n)
        entry = {"gen": gen, "event": event, "explanation": "", "source": "",
                 "ai_status": "", "speech_requested": False, "t_submit": now_t,
                 "greeting": greeting, "chat": chat, "n": n}
        self.log.append(entry)
        self.event_presented.emit(event)     # 1) el resultado se muestra de inmediato
        if greeting:
            self.presented = True
        if chat:
            # 2) el chat se despliega y Opti saluda (texto + voz a la vez)
            entry["t_chat_open"] = time.time()
            self.chat_open_requested.emit(event)
            if greeting:
                self.chat_message.emit(event, "greeting", greeting)
            self.chat_message.emit(event, "thinking", "")
        if greeting and self.voice_enabled:
            self._say(gen, entry, "greeting", greeting)

        def ok(text, src):
            if gen != self._gen or entry["explanation"]:
                return                       # obsoleta o duplicada
            text = (text or "").strip()
            if not text:
                return fail("respuesta vacía")
            entry["ai_status"] = "ok"
            self.ai_status.emit(event, "ok")
            self._deliver(gen, entry, event, text, src)

        def fail(msg):
            if gen != self._gen or entry["explanation"]:
                return
            entry["ai_status"] = f"ai_comm_error: {msg}"
            self.ai_status.emit(event, entry["ai_status"])
            self._deliver(gen, entry, event, local_explanation(event), "local")

        ai = self._ai
        if not self.ai_enabled or ai is None:
            entry["ai_status"] = "desactivada"
            self._deliver(gen, entry, event, local_explanation(event), "local")
        elif not ai.available():
            entry["ai_status"] = "no configurada (sin API Key)"
            self.ai_status.emit(event, entry["ai_status"])
            self._deliver(gen, entry, event, local_explanation(event), "local")
        else:
            QTimer.singleShot(AI_TIMEOUT_MS, lambda: fail("sin respuesta en "
                              f"{AI_TIMEOUT_MS // 1000} s"))
            try:
                ai.explain(event, lambda t: ok(t, ai.name), fail)
            except Exception as ex:
                fail(str(ex))
        return gen

    def _deliver(self, gen, entry, event, text, source):
        if entry.get("chat"):
            text = ensure_invitation(text, entry.get("n", 0))   # etapa 4
        entry["explanation"], entry["source"] = text, source
        entry["t_text"] = time.time()
        self.current_explanation, self.current_source = text, source
        self.explanation_ready.emit(event, text, source)
        if entry.get("chat"):
            self.chat_message.emit(event, "explanation", text)
        self.speech_state.emit(TEXT_READY)
        if self.voice_enabled:
            # La voz empieza DESPUÉS de que el recuadro y el texto estén en pantalla
            QTimer.singleShot(SPEECH_DELAY_MS, lambda: self._speak_if_current(gen, entry))
        else:
            self.speech_state.emit(OFF)

    def _speak_if_current(self, gen, entry):
        if gen != self._gen or not self.voice_enabled or entry.get("user_took_over")                 or entry.get("voice_cancelled"):
            return
        self._say(gen, entry, "explanation", entry["explanation"])

    def user_took_over(self):
        """El usuario escribió/habló en el chat: Opti calla y no lee lo pendiente."""
        e = self._current_entry()
        if e is not None:
            e["user_took_over"] = True
        self.cancel_speech()

    # ── Controles ───────────────────────────────────────────────────────────
    def set_voice_enabled(self, on: bool):
        self.voice_enabled = bool(on)
        if not on:
            self.cancel_speech()
            self.speech_state.emit(OFF)

    def stop(self):
        """⏹ (recuadro o chat): calla y no lee lo que quede de esta intervención."""
        e = self._current_entry()
        if e is not None:
            e["voice_cancelled"] = True
        self.cancel_speech()
        self.speech_state.emit(STOPPED)

    def repeat(self):
        if self.current_explanation:
            e = self._current_entry()
            self.cancel_speech()
            if e is not None:
                e["user_took_over"] = e["voice_cancelled"] = False
                self._say(self._gen, e, "repeat", self.current_explanation)
