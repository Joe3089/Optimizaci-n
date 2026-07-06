"""
ai_assistant.py — Asistente de IA para el Optimizador de Funciones
===================================================================
Integra Google Gemini (GRATIS) como asistente inteligente
estrictamente enfocado en optimización matemática y en los métodos
implementados en esta aplicación.

Características:
  • Responde SOLO preguntas relacionadas con el optimizador
  • Rechaza educadamente cualquier consulta fuera del dominio
  • Tiene contexto del estado actual (método, función, resultados)
  • Usa QThread para no bloquear la interfaz durante las llamadas
  • Historial de conversación por sesión
  • Sugerencias inteligentes de métodos según la función ingresada
"""

from __future__ import annotations
import json
import os
import re
from typing import List, Dict, Optional

# ─── Ruta del archivo de configuración (misma carpeta que ai_assistant.py) ───
_CFG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".optimizer_config.json")

# ─── Qt ──────────────────────────────────────────────────────────────────────
try:
    from PyQt6.QtCore import QThread, pyqtSignal as Signal, QObject, QTimer
    from PyQt6 import QtWidgets, QtGui, QtCore
    from PyQt6.QtCore import Qt
    _QT = "PyQt6"
except ImportError:
    from PyQt5.QtCore import QThread, pyqtSignal as Signal, QObject, QTimer  # type: ignore
    from PyQt5 import QtWidgets, QtGui, QtCore                                # type: ignore
    from PyQt5.QtCore import Qt                                               # type: ignore
    _QT = "PyQt5"

# ─── HTTP ─────────────────────────────────────────────────────────────────────
try:
    import urllib.request, urllib.error
    _HTTP_OK = True
except ImportError:
    _HTTP_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  GESTIÓN DE API KEY — 3 fuentes automáticas
# ══════════════════════════════════════════════════════════════════════════════
def _is_valid_key(key: str) -> bool:
    """Valida que la key sea de un proveedor gratuito soportado."""
    return bool(key and (
        key.startswith("AIza") or    # Google Gemini (gratis)
        key.startswith("gsk_")       # Groq (gratis)
    ))


def _cleanup_old_keys():
    """Elimina del archivo de config cualquier key de Anthropic (sk-) guardada."""
    try:
        if os.path.exists(_CFG_FILE):
            with open(_CFG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = data.get("api_key", "")
            if key and not _is_valid_key(key):
                data.pop("api_key", None)
                with open(_CFG_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
    except Exception:
        pass


def _load_api_key_auto() -> str:
    """
    Busca la API Key GRATUITA en este orden:
      0. Key embebida en el código (_EMBEDDED_KEY)
      1. Variable de entorno  GEMINI_API_KEY  o  GROQ_API_KEY
      2. Archivo de config    .optimizer_config.json
      3. Archivo .env
    Solo acepta Gemini (AIza...) y Groq (gsk_...).
    """
    # 0 ── Key embebida
    _EMBEDDED_KEY = "PEGA_AQUI_TU_API_KEY_DE_GROQ"
    if _is_valid_key(_EMBEDDED_KEY):
        return _EMBEDDED_KEY

    # 1 ── Variables de entorno (Groq primero)
    for env_var in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        key = os.environ.get(env_var, "").strip()
        if _is_valid_key(key):
            return key

    # 2 ── Archivo de configuración
    try:
        if os.path.exists(_CFG_FILE):
            with open(_CFG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = data.get("api_key", "").strip()
            if _is_valid_key(key):
                return key
    except Exception:
        pass

    # 3 ── Archivo .env
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    for var in ("GROQ_API_KEY", "GEMINI_API_KEY"):
                        if line.startswith(var):
                            parts = line.split("=", 1)
                            if len(parts) == 2:
                                key = parts[1].strip().strip('"').strip("'")
                                if _is_valid_key(key):
                                    return key
    except Exception:
        pass

    return ""


def _save_api_key(key: str) -> bool:
    """Guarda la API Key en el archivo de configuración local."""
    try:
        data = {}
        if os.path.exists(_CFG_FILE):
            try:
                with open(_CFG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data["api_key"] = key
        with open(_CFG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def _delete_saved_key() -> bool:
    """Elimina la API Key guardada del archivo de configuración."""
    try:
        if os.path.exists(_CFG_FILE):
            with open(_CFG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.pop("api_key", None)
            with open(_CFG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT — Define el dominio estricto del asistente
# ══════════════════════════════════════════════════════════════════════════════
_SYSTEM_PROMPT = """Eres el Asistente de IA del Optimizador de Funciones, una aplicación \
de optimización matemática desarrollada en Python con PyQt.

## TU ROL
Eres un experto en optimización numérica y metaheurísticas. Tu ÚNICA función es asistir \
al usuario con todo lo relacionado a esta aplicación y sus métodos matemáticos.

## MÉTODOS DISPONIBLES EN LA APLICACIÓN
### Búsqueda 1D clásica:
- Búsqueda Local: exploración por vecindad, paso adaptativo
- Fibonacci: reduce intervalo con secuencia de Fibonacci (sin derivadas)
- Sección Áurea: proporción τ=(√5-1)/2, convergencia garantizada

### Búsqueda de línea:
- Armijo: condición de descenso suficiente f(x+αd) ≤ f(x)+c·α·∇f·d
- Wolfe: Armijo + condición de curvatura |∇f(x+αd)·d| ≤ c₂|∇f(x)·d|
- Goldstein: Armijo + cota inferior para evitar pasos muy pequeños

### Newton clásico:
- Newton-Raphson: x_{k+1} = x_k - H⁻¹·∇f, convergencia cuadrática
- Gradiente Conjugado: Fletcher-Reeves, β=‖∇f_{k+1}‖²/‖∇f_k‖²

### Multidimensional (MD):
- MD: Penalización (Newton): φ(x) = f(x) + μ·max(0,g(x))²
- MD: Barreras (Newton): φ(x) = f(x) - (1/t)·ln(-g(x))
- MD: Pen./Pes./Sum. (BFGS): cuasi-Newton, aproxima H⁻¹ con gradientes
- MD: Pes. (Nelder-Mead): simplex sin derivadas

### Metaheurísticos:
- SA: Recocido Simulado: acepta peores con P=exp(-ΔE/T), T decrece
- PSO: Enjambre: v_i = w·v_i + c₁·r₁·(pbest-x_i) + c₂·r₂·(gbest-x_i)
- GA: Algoritmo Genético: selección torneo + cruce aritmético + mutación gaussiana

## FUNCIONES SOPORTADAS (sintaxis Python/SymPy):
- Polinomiales: x**2 - 4*x + 5, 2*(x1-3)**2 + x1*x2**3
- Trigonométricas: cos(x)*exp(-x**2), sin(x)/x
- Exponenciales: exp(-x**2), log(x+1)
- Benchmark: Rosenbrock, Himmelblau, Rastrigin, Ackley, Griewank, Beale
- Operadores: ** para potencia (no ^), * para multiplicación

## PUEDES AYUDAR CON:
1. Recomendar el mejor método para una función dada
2. Explicar por qué un método converge o diverge
3. Interpretar la tabla de iteraciones y resultados
4. Sugerir parámetros (rango, tolerancia, punto inicial)
5. Explicar las condiciones matemáticas (Armijo, Wolfe, Goldstein)
6. Comparar métodos clásicos vs metaheurísticos
7. Ayudar a ingresar funciones en la sintaxis correcta
8. Diagnosticar por qué una función da error
9. Explicar conceptos de convexidad, gradiente, Hessiano
10. Guiar en la interpretación de gráficas 2D y 3D

## REGLA ABSOLUTA — FUERA DE DOMINIO:
Si el usuario pregunta sobre temas NO relacionados con:
- Esta aplicación de optimización
- Los métodos matemáticos implementados
- Las funciones matemáticas a optimizar
- Conceptos de optimización numérica/metaheurística

DEBES responder EXACTAMENTE:
"⚠️ Esta consulta está fuera del dominio del Optimizador de Funciones. \
Solo puedo asistirte con temas relacionados a la optimización matemática \
y los métodos implementados en esta aplicación (Armijo, Wolfe, Fibonacci, \
Newton-Raphson, PSO, GA, Recocido Simulado, etc.). \
¿En qué puedo ayudarte con el optimizador?"

## ESTILO DE RESPUESTA:
- Respuestas concisas y directas (máx 200 palabras salvo que se pida más detalle)
- Usa notación matemática clara: f(x), ∇f, α, μ, etc.
- Cuando recomiendes un método, explica brevemente POR QUÉ
- Si hay un error en la función del usuario, muestra la corrección
- Idioma: SIEMPRE español

## CONTEXTO ACTUAL (se actualiza con cada cálculo):
{context}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Detectar proveedor por formato de key
# ══════════════════════════════════════════════════════════════════════════════
def _detect_provider(key: str) -> str:
    """
    Detecta el proveedor según el formato de la API Key:
      • 'AIza...' → Google Gemini (GRATIS)
      • 'gsk_...' → Groq          (GRATIS)
    """
    if key.startswith("AIza"): return "gemini"
    if key.startswith("gsk_"): return "groq"
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  Worker Thread — soporta Gemini (gratis) y Groq (gratis)
# ══════════════════════════════════════════════════════════════════════════════
class _AIWorker(QThread):
    """QThread que realiza la llamada HTTP al proveedor de IA seleccionado."""
    response_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, api_key: str, messages: list, system: str, parent=None):
        super().__init__(parent)
        self._api_key  = api_key
        self._messages = messages
        self._system   = system
        self._provider = _detect_provider(api_key)

    def run(self):
        try:
            if self._provider == "groq":
                self._call_groq()
            else:
                # Gemini es el proveedor por defecto
                self._call_gemini()
        except Exception as ex:
            self.error_occurred.emit(f"Error de conexión: {ex}")

    # ── Google Gemini (GRATIS) ─────────────────────────────────────────────────
    def _call_gemini(self):
        """
        Llama a Google Gemini 1.5 Flash (gratis: 15 req/min, 1M tokens/día).
        Endpoint: generativelanguage.googleapis.com
        """
        # Convertir historial al formato de Gemini (roles: user/model)
        contents = []
        # Inyectar system prompt como primer turno user/model
        sys_text = (f"[Instrucciones del sistema]:\n{self._system}\n\n"
                    "Confirma que entendiste respondiendo 'Entendido, estoy listo.'")
        contents.append({"role": "user",  "parts": [{"text": sys_text}]})
        contents.append({"role": "model", "parts": [{"text": "Entendido, estoy listo."}]})
        # Agregar historial real
        for msg in self._messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        payload = json.dumps({
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": 600,
                "temperature"    : 0.7,
            }
        }).encode("utf-8")

        url = (f"https://generativelanguage.googleapis.com/v1beta/"
               f"models/gemini-2.0-flash:generateContent?key={self._api_key}")

        req = urllib.request.Request(
            url,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                self.response_ready.emit(text)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                msg = json.loads(body).get("error", {}).get("message", body[:200])
            except Exception:
                msg = body[:200]
            if e.code == 400:
                self.error_occurred.emit(f"API Key de Gemini inválida: {msg[:120]}")
            elif e.code == 429:
                self.error_occurred.emit(
                    "Límite de Gemini alcanzado (15 req/min). Espera un momento.")
            else:
                self.error_occurred.emit(f"Error Gemini {e.code}: {msg[:120]}")

    # ── Groq (GRATIS) ──────────────────────────────────────────────────────────
    def _call_groq(self):
        """
        Llama a Groq con Llama 3 (gratis: 14,400 req/día).
        Endpoint: api.groq.com — compatible con formato OpenAI.
        """
        # Groq usa formato OpenAI: system + messages
        groq_messages = [{"role": "system", "content": self._system}]
        groq_messages += self._messages

        payload = json.dumps({
            "model"      : "llama-3.1-8b-instant",
            "messages"   : groq_messages,
            "max_tokens" : 600,
            "temperature": 0.7,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data    = payload,
            headers = {
                "Content-Type" : "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"]
                self.response_ready.emit(text)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                msg = json.loads(body).get("error", {}).get("message", body[:200])
            except Exception:
                msg = body[:200]
            if e.code == 401:
                self.error_occurred.emit("API Key de Groq inválida.")
            elif e.code == 429:
                self.error_occurred.emit(
                    "Límite de Groq alcanzado. Espera un momento.")
            else:
                self.error_occurred.emit(f"Error Groq {e.code}: {msg[:120]}")



# ══════════════════════════════════════════════════════════════════════════════
#  Panel del Asistente (widget Qt embebido)
# ══════════════════════════════════════════════════════════════════════════════
class AIAssistantPanel(QtWidgets.QWidget):
    """
    Panel lateral/flotante con el chat del asistente de IA.
    Se integra directamente en la interfaz principal.
    """

    # ─── Estilos ──────────────────────────────────────────────────────────────
    _QSS = """
    QWidget#aiPanel {
        background: rgba(6,12,32,245);
        border-left: 2px solid rgba(60,120,255,160);
        border-top: 2px solid rgba(60,120,255,160);
        border-radius: 10px 0 0 10px;
    }
    QLabel#aiTitle {
        color: #7eb8ff;
        font-size: 13px;
        font-weight: bold;
        background: transparent;
        padding: 2px 0;
    }
    QLabel#aiStatus {
        color: #3cefff;
        font-size: 11px;
        background: transparent;
    }
    QTextEdit#chatHistory {
        background: rgba(8,16,45,220);
        color: #c8ddf8;
        border: 1px solid rgba(40,70,160,180);
        border-radius: 6px;
        font-size: 12px;
        font-family: 'Segoe UI', Consolas;
        padding: 6px;
        selection-background-color: #1a3e80;
    }
    QLineEdit#chatInput {
        background: rgba(12,22,60,230);
        color: #dce8f8;
        border: 1px solid rgba(60,100,200,180);
        border-radius: 6px;
        font-size: 12px;
        padding: 6px 10px;
        min-height: 30px;
    }
    QLineEdit#chatInput:focus {
        border: 1.5px solid #4a9eff;
    }
    QPushButton#sendBtn {
        background: #1a50c0;
        color: #fff;
        border: none;
        border-radius: 6px;
        font-weight: bold;
        font-size: 13px;
        min-width: 38px;
        min-height: 36px;
    }
    QPushButton#sendBtn:hover   { background: #2462d8; }
    QPushButton#sendBtn:pressed { background: #1340a0; }
    QPushButton#sendBtn:disabled { background: #1a2a50; color: #4a6090; }
    QPushButton#clearBtn {
        background: rgba(30,20,60,200);
        color: #7090c0;
        border: 1px solid rgba(60,80,150,140);
        border-radius: 5px;
        font-size: 11px;
        padding: 3px 10px;
    }
    QPushButton#clearBtn:hover { background: rgba(50,30,90,220); color: #a0b8e0; }
    QPushButton#apiKeyBtn {
        background: rgba(20,40,100,200);
        color: #6090d0;
        border: 1px solid rgba(50,80,160,140);
        border-radius: 5px;
        font-size: 11px;
        padding: 3px 10px;
    }
    QPushButton#apiKeyBtn:hover { background: rgba(30,60,140,220); color: #90b8f0; }
    QLineEdit#apiKeyInput {
        background: rgba(10,18,50,230);
        color: #b0c8e8;
        border: 1px solid rgba(50,80,160,160);
        border-radius: 5px;
        font-size: 11px;
        padding: 4px 8px;
    }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("aiPanel")
        self.setStyleSheet(self._QSS)
        self.setFixedWidth(340)

        self._api_key:  str  = ""
        self._messages: List[Dict] = []
        self._context:  str  = "Sin cálculo realizado aún."
        self._worker:   Optional[_AIWorker] = None
        self._busy:     bool = False
        self._pending_explain: str = ""   # prompt guardado del último cálculo

        self._build_ui()
        _cleanup_old_keys()

        auto_key = _load_api_key_auto()
        if auto_key:
            self._apply_key(auto_key, source="auto")

    # ─── Construcción de UI ────────────────────────────────────────────────────
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── Título ──────────────────────────────────────────────────────────
        hdr = QtWidgets.QHBoxLayout()
        ico = QtWidgets.QLabel("🤖")
        ico.setStyleSheet("font-size:18px; background:transparent;")
        title = QtWidgets.QLabel("Asistente de IA")
        title.setObjectName("aiTitle")
        self.lbl_status = QtWidgets.QLabel("● Sin conexión")
        self.lbl_status.setObjectName("aiStatus")
        hdr.addWidget(ico)
        hdr.addWidget(title, 1)
        hdr.addWidget(self.lbl_status)
        root.addLayout(hdr)

        # ── Sección API Key ──────────────────────────────────────────────────
        self.key_widget = QtWidgets.QWidget()
        self.key_widget.setStyleSheet("background: transparent;")
        key_layout = QtWidgets.QVBoxLayout(self.key_widget)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.setSpacing(4)

        key_row = QtWidgets.QHBoxLayout()
        self.edt_key = QtWidgets.QLineEdit()
        self.edt_key.setObjectName("apiKeyInput")
        self.edt_key.setPlaceholderText("API Key Groq (gsk_...) o Gemini (AIza...)")
        self.edt_key.setEchoMode(
            QtWidgets.QLineEdit.EchoMode.Password if _QT=="PyQt6"
            else QtWidgets.QLineEdit.Password)
        self.edt_key.setToolTip(
            "🥇 GROQ — recomendado (30 req/min, gratis)\n"
            "   1. Ve a: console.groq.com/keys\n"
            "   2. Crea una cuenta gratuita\n"
            "   3. Genera una API Key\n"
            "   La key empieza con 'gsk_'\n\n"
            "🥈 Google Gemini (15 req/min, gratis)\n"
            "   aistudio.google.com/apikey\n"
            "   La key empieza con 'AIza'")
        self.edt_key.returnPressed.connect(self._set_api_key)

        self.btn_key = QtWidgets.QPushButton("✓ Conectar")
        self.btn_key.setObjectName("apiKeyBtn")
        self.btn_key.clicked.connect(self._set_api_key)
        key_row.addWidget(self.edt_key, 1)
        key_row.addWidget(self.btn_key)
        key_layout.addLayout(key_row)

        opts_row = QtWidgets.QHBoxLayout()
        self.chk_save = QtWidgets.QCheckBox("Recordar key")
        self.chk_save.setStyleSheet(
            "QCheckBox { color:#5a8ac0; font-size:10px; background:transparent; spacing:5px; }"
            "QCheckBox::indicator { width:13px; height:13px; border:1px solid #2a4a90;"
            " border-radius:3px; background:rgba(10,20,60,200); }"
            "QCheckBox::indicator:checked { background:#1a50c0; border-color:#4a9eff; }"
        )
        self.chk_save.setChecked(True)
        self.btn_forget = QtWidgets.QPushButton("🗑 Olvidar key")
        self.btn_forget.setObjectName("clearBtn")
        self.btn_forget.clicked.connect(self._forget_key)
        self.btn_forget.setVisible(False)
        opts_row.addWidget(self.chk_save)
        opts_row.addStretch()
        opts_row.addWidget(self.btn_forget)
        key_layout.addLayout(opts_row)

        root.addWidget(self.key_widget)

        # ── Separador ────────────────────────────────────────────────────────
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine if _QT=="PyQt6"
                          else QtWidgets.QFrame.HLine)
        sep.setStyleSheet("background:rgba(50,90,200,100); max-height:1px; border:none;")
        root.addWidget(sep)

        # ── Historial de chat ─────────────────────────────────────────────────
        self.chat_view = QtWidgets.QTextEdit()
        self.chat_view.setObjectName("chatHistory")
        self.chat_view.setReadOnly(True)
        self.chat_view.setMinimumHeight(200)
        self._append_system_msg(
            "👋 Hola, soy tu Asistente de IA para el Optimizador.\n\n"
            "Para activarme necesito una API Key gratuita:\n\n"
            "  🥇  Groq (recomendado — 30 req/min, sin restricciones)\n"
            "      console.groq.com/keys  →  key: gsk_...\n\n"
            "  🥈  Google Gemini (15 req/min)\n"
            "      aistudio.google.com/apikey  →  key: AIza...\n\n"
            "Pega tu key arriba y pulsa '✓ Conectar'.\n"
            "✅ Se recordará automáticamente para la próxima sesión.")
        root.addWidget(self.chat_view, 1)

        # ── Botón Explicar resultado (se activa tras cada cálculo) ────────────
        self.btn_explain = QtWidgets.QPushButton("📊 Explicar último resultado")
        self.btn_explain.setObjectName("explainBtn")
        self.btn_explain.setStyleSheet(
            "QPushButton#explainBtn {"
            "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "    stop:0 rgba(20,70,160,210), stop:1 rgba(10,120,80,200));"
            "  color: #c8eeff; border: 1px solid rgba(50,140,200,160);"
            "  border-radius: 7px; font-size: 11px; font-weight: bold;"
            "  padding: 6px 10px; text-align: left; }"
            "QPushButton#explainBtn:hover {"
            "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "    stop:0 rgba(30,100,210,240), stop:1 rgba(20,160,110,220));"
            "  color: #ffffff; }"
            "QPushButton#explainBtn:disabled {"
            "  background: rgba(15,25,55,140); color: #2a3d58;"
            "  border-color: rgba(25,40,80,80); }")
        self.btn_explain.setCursor(QtGui.QCursor(
            Qt.CursorShape.PointingHandCursor if _QT=="PyQt6" else Qt.PointingHandCursor))
        self.btn_explain.clicked.connect(self._on_explain_click)
        self.btn_explain.setEnabled(False)
        root.addWidget(self.btn_explain)

        # ── Sugerencias rápidas ───────────────────────────────────────────────
        sug_lbl = QtWidgets.QLabel("Sugerencias:")
        sug_lbl.setStyleSheet("color:#5a7ab0; font-size:11px; background:transparent;")
        root.addWidget(sug_lbl)

        sug_grid = QtWidgets.QGridLayout()
        sug_grid.setSpacing(4)
        suggestions = [
            ("¿Qué método usar?",        "¿Qué método me recomiendas para la función actual y por qué?"),
            ("Comparar métodos",          "¿Cuál es la diferencia práctica entre Armijo, Wolfe y Goldstein?"),
            ("Método para Rosenbrock",    "¿Qué método es mejor para optimizar la función de Rosenbrock y por qué?"),
            ("¿Cuándo usar PSO o GA?",    "¿Cuándo conviene usar metaheurísticas (PSO, GA, SA) en vez de métodos clásicos?"),
            ("Convergencia lenta",        "El método converge muy lento, ¿qué puedo cambiar para mejorar?"),
            ("Ayuda con la función",      "¿Cómo escribo correctamente una función en la sintaxis del optimizador?"),
        ]
        for i, (label, prompt) in enumerate(suggestions):
            btn = QtWidgets.QPushButton(label)
            btn.setStyleSheet(
                "QPushButton { background: rgba(20,35,90,200); color:#7aaad8;"
                " border:1px solid rgba(40,70,150,160); border-radius:5px;"
                " font-size:10px; padding:4px 6px; text-align:left; }"
                "QPushButton:hover { background: rgba(30,55,130,220); color:#a0c8f0; }")
            btn.setCursor(QtGui.QCursor(
                Qt.CursorShape.PointingHandCursor if _QT=="PyQt6" else Qt.PointingHandCursor))
            # Siempre mostrar burbuja de usuario al enviar sugerencia
            btn.clicked.connect(lambda _, p=prompt: self._send_message(p))
            sug_grid.addWidget(btn, i // 2, i % 2)
        root.addLayout(sug_grid)

        # ── Input + botón enviar ──────────────────────────────────────────────
        inp_row = QtWidgets.QHBoxLayout()
        inp_row.setSpacing(6)
        self.edt_msg = QtWidgets.QLineEdit()
        self.edt_msg.setObjectName("chatInput")
        self.edt_msg.setPlaceholderText("Escribe tu pregunta sobre el optimizador…")
        self.edt_msg.returnPressed.connect(self._on_send)

        self.btn_send = QtWidgets.QPushButton("➤")
        self.btn_send.setObjectName("sendBtn")
        self.btn_send.clicked.connect(self._on_send)
        self.btn_send.setEnabled(False)

        inp_row.addWidget(self.edt_msg, 1)
        inp_row.addWidget(self.btn_send)
        root.addLayout(inp_row)

        # ── Pie ───────────────────────────────────────────────────────────────
        foot = QtWidgets.QHBoxLayout()
        self.btn_clear_chat = QtWidgets.QPushButton("🗑 Limpiar chat")
        self.btn_clear_chat.setObjectName("clearBtn")
        self.btn_clear_chat.clicked.connect(self._clear_chat)
        powered = QtWidgets.QLabel("Groq · Gemini — 100% Gratis")
        powered.setStyleSheet("color:#2a6040; font-size:10px; background:transparent;")
        foot.addWidget(self.btn_clear_chat)
        foot.addStretch()
        foot.addWidget(powered)
        root.addLayout(foot)

    # ─── API Key ─────────────────────────────────────────────────────────────────
    def _set_api_key(self):
        key = self.edt_key.text().strip()
        if not key:
            self._append_system_msg("⚠️ Pega tu API Key primero.")
            return
        provider = _detect_provider(key)
        if provider == "unknown":
            self._append_system_msg(
                "⚠️ Key no reconocida.\n\n"
                "Proveedores gratuitos aceptados:\n"
                "  🥇 Groq  →  gsk_...   (console.groq.com/keys)\n"
                "  🥈 Gemini →  AIza...  (aistudio.google.com/apikey)")
            self.edt_key.clear()
            return
        if self.chk_save.isChecked():
            _save_api_key(key)
        self.edt_key.clear()
        self._apply_key(key, source="manual")

    def _apply_key(self, key: str, source: str = "manual", extra_msg: str = ""):
        self._api_key = key
        self.btn_send.setEnabled(True)
        if self._pending_explain:
            self.btn_explain.setEnabled(True)

        provider = _detect_provider(key)
        prov_labels = {"groq": "Groq / Llama 🟢", "gemini": "Google Gemini 🟡"}
        prov_label  = prov_labels.get(provider, "IA conectada 🟢")

        self.lbl_status.setText("● Conectado")
        self.lbl_status.setStyleSheet("color: #3cff8a; font-size:11px; background:transparent;")
        self.key_widget.setVisible(False)
        self.btn_forget.setVisible(True)
        self.edt_key.setVisible(False)

        # Limpiar chat y mostrar saludo estático — SIN llamar a la API
        self.chat_view.clear()
        self._messages.clear()
        self._append_ai_msg(
            f"¡Hola! Soy tu asistente de optimización matemática, conectado con {prov_label}.\n\n"
            "Puedo ayudarte a:\n"
            "  • Recomendar el método más adecuado para tu función\n"
            "  • Explicar resultados, iteraciones y convergencia\n"
            "  • Comparar métodos (Armijo, Wolfe, Newton, PSO, GA…)\n"
            "  • Diagnosticar por qué un método falla o converge lento\n"
            "  • Resolver dudas sobre optimización numérica\n\n"
            "Usa las sugerencias de abajo o escríbeme directamente. "
            "Después de cada cálculo aparecerá el botón 📊 para que yo explique el resultado."
        )

    def _forget_key(self):
        """Elimina la key guardada y muestra el panel de conexión."""
        _delete_saved_key()
        self._api_key = ""
        self.btn_send.setEnabled(False)
        self.btn_explain.setEnabled(False)
        self.lbl_status.setText("● Sin conexión")
        self.lbl_status.setStyleSheet(
            "color: #3cefff; font-size:11px; background:transparent;")
        self.key_widget.setVisible(True)
        self.btn_forget.setVisible(False)
        self.edt_key.setVisible(True)
        self.chat_view.clear()
        self._messages.clear()
        self._append_system_msg(
            "🗑 Key eliminada.\n\n"
            "Pega una nueva API Key arriba para reconectarte.\n"
            "  🥇 Groq → console.groq.com/keys\n"
            "  🥈 Gemini → aistudio.google.com/apikey")

    # ─── Botón Explicar resultado ────────────────────────────────────────────────
    def _on_explain_click(self):
        """Envía el prompt guardado del último cálculo al pulsar el botón."""
        if not self._pending_explain:
            return
        prompt = self._pending_explain
        self._pending_explain = ""
        self.btn_explain.setEnabled(False)
        self.btn_explain.setText("📊 Explicar último resultado")
        self._send_message(prompt)

    # ─── Enviar mensaje ──────────────────────────────────────────────────────────
    def _on_send(self):
        text = self.edt_msg.text().strip()
        if not text:
            return
        self.edt_msg.clear()
        self._send_message(text)

    def _send_message(self, text: str, is_auto: bool = False):
        """
        Envía un mensaje al asistente.
        is_auto=True: no muestra burbuja de usuario (usado internamente si hiciera falta).
        is_auto=False (default): siempre muestra la burbuja del usuario en el chat.
        """
        if not self._api_key:
            self._append_system_msg("⚠️ Conecta tu API Key primero.")
            return

        if self._busy:
            self._append_system_msg("⏳ Espera, procesando respuesta anterior…")
            return

        # Mostrar mensaje del usuario en el chat
        if not is_auto:
            self._append_user_msg(text)

        self._messages.append({"role": "user", "content": text})
        self._busy = True
        self._append_system_msg("⏳ Analizando…")
        self.btn_send.setEnabled(False)
        self.btn_send.setText("…")

        system = _SYSTEM_PROMPT.replace("{context}", self._context)
        self._worker = _AIWorker(self._api_key, list(self._messages), system)
        self._worker.response_ready.connect(self._on_response)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_response(self, text: str):
        """Llamado cuando el worker recibe respuesta exitosa."""
        self._remove_last_system_msg()   # quitar "Analizando…"
        self._append_ai_msg(text)
        self._messages.append({"role": "assistant", "content": text})
        # Limitar historial (10 turnos)
        if len(self._messages) > 20:
            self._messages = self._messages[-20:]
        self._busy = False
        self.btn_send.setEnabled(True)
        self.btn_send.setText("➤")

    def _on_error(self, error: str):
        """Llamado cuando el worker falla."""
        self._remove_last_system_msg()
        # Sacar el mensaje del usuario del historial para que pueda reenviar
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages.pop()
        self._append_system_msg(f"❌ {error}\n↩ Puedes intentarlo de nuevo.")
        self._busy = False
        self.btn_send.setEnabled(bool(self._api_key))
        self.btn_send.setText("➤")

    # _process_queue kept for QTimer compatibility (no-op now)
    def _process_queue(self):
        pass

    # ─── Actualizar contexto desde la app principal ─────────────────────────────
    def update_context(self, metodo: str = "", fx: str = "",
                       result_summary: str = "", n_iters: int = 0,
                       vars_: list = None, x0: list = None,
                       lo: float = 0.0, hi: float = 5.0):
        """
        Llamado por la ventana principal tras cada cálculo para
        darle al asistente contexto del estado actual.
        """
        parts = []
        if metodo:
            parts.append(f"Método activo: {metodo}")
        if fx:
            parts.append(f"Función: f(x) = {fx}")
        if vars_:
            parts.append(f"Variables: {', '.join(vars_)}")
        if x0:
            parts.append(f"Punto inicial x0: {x0}")
        parts.append(f"Rango de búsqueda: [{lo:.4g}, {hi:.4g}]")
        if n_iters:
            parts.append(f"Iteraciones ejecutadas: {n_iters}")
        if result_summary:
            parts.append(f"Resultado: {result_summary}")

        self._context = "\n".join(parts) if parts else "Sin cálculo realizado aún."

    def explain_calculation(self, metodo: str, fx: str, hist: list,
                            vars_: list, x0: list, lo: float, hi: float,
                            gx: str = ""):
        """
        Llamado automáticamente después de cada cálculo.
        Construye un prompt detallado y lo envía al asistente.
        """
        if not self._api_key:
            return   # Sin API key no hacemos nada automático

        if not hist:
            return

        # Extraer datos clave del historial para el prompt
        first = hist[0]
        last  = hist[-1]
        n     = len(hist)

        def _fmt_val(v):
            if v is None: return "N/D"
            if hasattr(v, '__iter__') and not isinstance(v, str):
                try:
                    return "[" + ", ".join(f"{float(x):.5g}" for x in v) + "]"
                except Exception:
                    return str(v)
            try:   return f"{float(v):.6g}"
            except: return str(v)

        # Valor inicial
        f_ini = _fmt_val(first.get("f_k") or first.get("f(x)") or
                         first.get("f_lambda") or first.get("f_best"))
        # Valor final
        f_fin = _fmt_val(last.get("f_k")  or last.get("f(x)") or
                         last.get("f_lambda") or last.get("f_best"))
        # Punto óptimo
        x_fin = _fmt_val(last.get("x_k")  or last.get("x") or
                         last.get("lambda_k") or last.get("x_k_str"))
        # Gradiente final si existe
        grad_fin = _fmt_val(last.get("grad_norm"))
        has_grad = last.get("grad_norm") is not None

        # Restricción
        restr_info = f"\nRestricción activa: g(x) = {gx}" if gx else ""

        # ── Construir prompt automático ──────────────────────────────────────
        prompt = (
            f"Acabo de ejecutar el método **{metodo}** sobre la función "
            f"f({', '.join(vars_)}) = {fx}.{restr_info}\n\n"
            f"**Parámetros usados:**\n"
            f"  • Rango: [{lo:.4g}, {hi:.4g}]\n"
            f"  • Punto inicial x₀: {x0}\n"
            f"  • Iteraciones: {n}\n\n"
            f"**Resumen del historial:**\n"
            f"  • Valor inicial:  f(x₀) = {f_ini}\n"
            f"  • Valor final:    f(x*) = {f_fin}\n"
            f"  • Punto óptimo:   x* = {x_fin}\n"
        )
        if has_grad:
            prompt += f"  • ‖∇f(x*)‖ = {grad_fin}\n"

        prompt += (
            f"\nPor favor explica de forma clara e intuitiva:\n"
            f"1. ¿Qué hizo el método {metodo} paso a paso?\n"
            f"2. ¿Qué significan los valores obtenidos?\n"
            f"3. ¿Convergió bien? ¿Por qué?\n"
            f"4. ¿Qué representa el punto x* encontrado?"
        )

        # Guardar prompt y activar botón — sin consumir API hasta que el usuario lo pida
        self._pending_explain = prompt
        self.btn_explain.setText(f"📊 Explicar: {metodo}  f={fx[:22]}{'…' if len(fx)>22 else ''}")
        self.btn_explain.setEnabled(True)

    # ─── Métodos de visualización del chat ─────────────────────────────────────
    def _append_user_msg(self, text: str):
        cursor = self.chat_view.textCursor()
        cursor.movePosition(
            QtGui.QTextCursor.MoveOperation.End if _QT=="PyQt6"
            else QtGui.QTextCursor.End)
        fmt_name = QtGui.QTextCharFormat()
        fmt_name.setForeground(QtGui.QColor("#4a9eff"))
        fmt_name.setFontWeight(
            QtGui.QFont.Weight.Bold if _QT=="PyQt6" else QtGui.QFont.Bold)
        fmt_name.setFontPointSize(11)
        cursor.insertText("\n👤 Tú\n", fmt_name)

        fmt_body = QtGui.QTextCharFormat()
        fmt_body.setForeground(QtGui.QColor("#d8eaf8"))
        fmt_body.setFontPointSize(11)
        cursor.insertText(text + "\n", fmt_body)
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

    def _append_ai_msg(self, text: str):
        cursor = self.chat_view.textCursor()
        cursor.movePosition(
            QtGui.QTextCursor.MoveOperation.End if _QT=="PyQt6"
            else QtGui.QTextCursor.End)
        fmt_name = QtGui.QTextCharFormat()
        fmt_name.setForeground(QtGui.QColor("#3cff8a"))
        fmt_name.setFontWeight(
            QtGui.QFont.Weight.Bold if _QT=="PyQt6" else QtGui.QFont.Bold)
        fmt_name.setFontPointSize(11)
        cursor.insertText("\n🤖 Asistente\n", fmt_name)

        fmt_body = QtGui.QTextCharFormat()
        fmt_body.setForeground(QtGui.QColor("#c8f0d8"))
        fmt_body.setFontPointSize(11)
        cursor.insertText(text + "\n", fmt_body)
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

    def _append_system_msg(self, text: str):
        cursor = self.chat_view.textCursor()
        cursor.movePosition(
            QtGui.QTextCursor.MoveOperation.End if _QT=="PyQt6"
            else QtGui.QTextCursor.End)
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor("#7a9ac0"))
        fmt.setFontItalic(True)
        fmt.setFontPointSize(10)
        cursor.insertText("\n" + text + "\n", fmt)
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

    def _remove_last_system_msg(self):
        """Elimina el último bloque 'Pensando…'."""
        html = self.chat_view.toPlainText()
        lines = html.split("\n")
        # Eliminar líneas finales vacías + la línea "⏳ Pensando…"
        while lines and lines[-1].strip() == "":
            lines.pop()
        if lines and ("Pensando" in lines[-1] or "Analizando" in lines[-1]):
            lines.pop()
        while lines and lines[-1].strip() == "":
            lines.pop()
        self.chat_view.setPlainText("\n".join(lines))
        cursor = self.chat_view.textCursor()
        cursor.movePosition(
            QtGui.QTextCursor.MoveOperation.End if _QT=="PyQt6"
            else QtGui.QTextCursor.End)
        self.chat_view.setTextCursor(cursor)

    def _clear_chat(self):
        self.chat_view.clear()
        self._messages.clear()
        self._busy = False
        self.btn_send.setEnabled(bool(self._api_key))
        self.btn_send.setText("➤")
        self._append_system_msg("🗑 Chat limpiado. ¿En qué puedo ayudarte?")
