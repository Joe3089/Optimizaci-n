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
import shutil
import sys
from typing import List, Dict, Optional

# ─── Directorio raíz de la app (solo lectura: iconos/fondos ya viajan dentro
# del EXE vía _MEIPASS; esto es únicamente fallback legado para .env/config) ──
def _get_app_dir() -> str:
    """Devuelve la carpeta de la app: directorio del EXE (frozen) o del script."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # __file__ vive en app/infrastructure/; la raíz del proyecto está dos niveles arriba.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Carpeta de datos de usuario ───────────────────────────────────────────────
# La API Key y el config NO pueden depender de dónde vive el .exe: un acceso
# directo en el Escritorio, movido o copiado a cualquier lugar, debe seguir
# encontrando la key guardada. Por eso se persisten en %APPDATA% (igual que
# cualquier app profesional de Windows), no "junto al EXE".
def _get_user_data_dir() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "OptimizadorFunciones")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d

_APP_DIR       = _get_app_dir()
_USER_DATA_DIR = _get_user_data_dir()
_CFG_FILE        = os.path.join(_USER_DATA_DIR, ".optimizer_config.json")
_LEGACY_CFG_FILE = os.path.join(_APP_DIR, ".optimizer_config.json")


def _migrate_legacy_config() -> None:
    """Copia (una sola vez, sin borrar el original) un config/.env legado que
    viva junto al EXE hacia %APPDATA%, para no perder una key ya guardada al
    aplicar este cambio."""
    try:
        if not os.path.exists(_CFG_FILE) and os.path.exists(_LEGACY_CFG_FILE):
            shutil.copy2(_LEGACY_CFG_FILE, _CFG_FILE)
    except Exception:
        pass
    try:
        _new_env = os.path.join(_USER_DATA_DIR, ".env")
        _old_env = os.path.join(_APP_DIR, ".env")
        if not os.path.exists(_new_env) and os.path.exists(_old_env):
            shutil.copy2(_old_env, _new_env)
    except Exception:
        pass


_migrate_legacy_config()

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

# ─── Voz: TTS (pyttsx3 / SAPI5, offline) ──────────────────────────────────────
try:
    import pyttsx3
    _TTS_OK = True
except Exception:
    pyttsx3 = None
    _TTS_OK = False

# ─── Voz: grabación de micrófono (sounddevice) ────────────────────────────────
try:
    import sounddevice as _sd
    import numpy as _np
    import wave as _wave
    import tempfile as _tempfile
    import uuid as _uuid
    _REC_OK = True
except Exception:
    _sd = None
    _REC_OK = False

import ssl as _ssl
def _ssl_ctx():
    """
    Crea un SSL context que funciona en script normal Y en EXE PyInstaller.

    Problema común con PyInstaller: certifi.where() apunta a _MEIPASS/certifi/cacert.pem
    pero si certifi no está en el .spec como datas, el archivo no existe → SSL falla →
    Groq no conecta → todo cae a OpenRouter → rate limit 429.
    Esta función busca el cacert.pem en múltiples ubicaciones posibles.
    """
    import os as _os

    # 1) Buscar cacert.pem junto al EXE (usuario puede copiarlo manualmente)
    _exe_dir = _os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else ""

    # 2) Dentro de _MEIPASS (PyInstaller temp dir)
    _meipass  = getattr(sys, "_MEIPASS", "")

    candidates = []
    if _meipass:
        candidates += [
            _os.path.join(_meipass, "certifi", "cacert.pem"),
            _os.path.join(_meipass, "cacert.pem"),
        ]
    if _exe_dir:
        candidates += [
            _os.path.join(_exe_dir, "certifi", "cacert.pem"),
            _os.path.join(_exe_dir, "cacert.pem"),
        ]

    # 3) certifi instalado normalmente (modo script)
    try:
        import certifi as _certifi
        candidates.append(_certifi.where())
    except Exception:
        pass

    for cafile in candidates:
        if cafile and _os.path.isfile(cafile):
            try:
                ctx = _ssl.create_default_context(cafile=cafile)
                return ctx
            except Exception:
                continue

    # Último recurso: contexto por defecto del sistema (puede fallar en Windows sin certs)
    try:
        ctx = _ssl.create_default_context()
        return ctx
    except Exception:
        # Sin verificación — solo si todo lo anterior falló
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode    = _ssl.CERT_NONE
        return ctx


# ══════════════════════════════════════════════════════════════════════════════
#  GESTIÓN DE API KEY — 3 fuentes automáticas
# ══════════════════════════════════════════════════════════════════════════════
def _is_valid_key(key: str) -> bool:
    """Valida que la key sea de un proveedor soportado."""
    return bool(key and (
        key.startswith("gsk_")   or   # Groq       (gratis)
        key.startswith("sk-or-")      # OpenRouter (gratis)
    ))


def _cleanup_old_keys():
    """Elimina del archivo de config cualquier key inválida guardada."""
    try:
        if os.path.exists(_CFG_FILE):
            with open(_CFG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            changed = False
            for field in ("api_key", "api_key_backup"):
                k = data.get(field, "")
                if k and not _is_valid_key(k):
                    data.pop(field, None)
                    changed = True
            if changed:
                with open(_CFG_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
    except Exception:
        pass


def _load_api_key_auto() -> tuple:
    """
    Devuelve (key_principal, key_respaldo).
    Busca en: embebida → variables de entorno → config → .env
    Soporta Groq (gsk_...) y OpenRouter (sk-or-...).
    """
    keys_found: list = []

    # 0 ── Keys embebidas
    _EMBEDDED_GROQ = "PEGA_AQUI_TU_KEY_DE_GROQ"
    _EMBEDDED_OR   = "PEGA_AQUI_TU_KEY_DE_OPENROUTER"
    for k in (_EMBEDDED_GROQ, _EMBEDDED_OR):
        if _is_valid_key(k) and k not in keys_found:
            keys_found.append(k)

    # 1 ── Variables de entorno
    for env_var in ("GROQ_API_KEY", "OPENROUTER_API_KEY"):
        k = os.environ.get(env_var, "").strip()
        if _is_valid_key(k) and k not in keys_found:
            keys_found.append(k)

    # 2 ── Archivo de configuración (%APPDATA% primero; junto al EXE como fallback)
    for cfg_path in (_CFG_FILE, _LEGACY_CFG_FILE):
        try:
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for field in ("api_key", "api_key_backup"):
                    k = data.get(field, "").strip()
                    if _is_valid_key(k) and k not in keys_found:
                        keys_found.append(k)
        except Exception:
            pass

    # 3 ── Archivo .env (%APPDATA%\OptimizadorFunciones primero; junto al
    #      EXE/script como fallback legado — nunca en _MEIPASS, se borra al cerrar)
    for env_path in (os.path.join(_USER_DATA_DIR, ".env"), os.path.join(_APP_DIR, ".env")):
        try:
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        for var in ("GROQ_API_KEY", "OPENROUTER_API_KEY"):
                            if line.startswith(var + "="):
                                parts = line.split("=", 1)
                                if len(parts) == 2:
                                    k = parts[1].strip().strip('"').strip("'")
                                    if _is_valid_key(k) and k not in keys_found:
                                        keys_found.append(k)
        except Exception:
            pass

    primary = keys_found[0] if len(keys_found) > 0 else ""
    backup  = keys_found[1] if len(keys_found) > 1 else ""
    return primary, backup


def _save_api_key(key: str, is_backup: bool = False) -> bool:
    """Guarda la API Key principal o de respaldo en el archivo de configuración."""
    try:
        data = {}
        if os.path.exists(_CFG_FILE):
            try:
                with open(_CFG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data["api_key_backup" if is_backup else "api_key"] = key
        with open(_CFG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def _delete_saved_key() -> bool:
    """Elimina las API Keys guardadas del archivo de configuración."""
    try:
        if os.path.exists(_CFG_FILE):
            with open(_CFG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.pop("api_key", None)
            data.pop("api_key_backup", None)
            with open(_CFG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  RENDERIZADO MATEMÁTICO — Convierte texto con notación math a HTML
# ══════════════════════════════════════════════════════════════════════════════
def _md_to_html(text: str) -> str:
    """
    Convierte la respuesta del asistente (markdown + notación matemática)
    a HTML para renderizado enriquecido en QTextEdit.

    Soporta:
      • **negrita** y *cursiva*
      • x**2 / x^2  →  x<sup>2</sup>
      • x_{k+1}     →  x<sub>k+1</sub>
      • Fracciones comunes: 1/2 → ½, etc.
      • Encabezados ##, ###
      • Listas con - / • / 1.
    """
    import html as _hl

    COLOR_BODY   = "#c8f0d8"
    COLOR_HEAD   = "#7eb8ff"
    COLOR_HEAD3  = "#a0c8ff"
    COLOR_NUM    = "#ffd080"

    # Mapa de fracciones Unicode
    _FRACS = [("1/2","½"),("1/3","⅓"),("1/4","¼"),("2/3","⅔"),("3/4","¾"),
              ("1/8","⅛"),("3/8","⅜"),("5/8","⅝"),("7/8","⅞")]

    def _apply_math(s: str) -> str:
        """Aplica superíndices y subíndices (se llama DESPUÉS del escape HTML)."""
        # Superíndices: x**{k+1}, x**2, x^{n}, x^2
        s = re.sub(r'([a-zA-Z\u03b1-\u03c9\u0391-\u03a9\u2207\)\]])\*\*\{([^}]+)\}',
                   lambda m: f"{m.group(1)}<sup>{m.group(2)}</sup>", s)
        s = re.sub(r'([a-zA-Z\u03b1-\u03c9\u0391-\u03a9\u2207\)\]])\*\*(-?\d+(?:\.\d+)?)',
                   lambda m: f"{m.group(1)}<sup>{m.group(2)}</sup>", s)
        s = re.sub(r'([a-zA-Z\u03b1-\u03c9\u0391-\u03a9\)\]])\^\{([^}]+)\}',
                   lambda m: f"{m.group(1)}<sup>{m.group(2)}</sup>", s)
        s = re.sub(r'([a-zA-Z\u03b1-\u03c9\u0391-\u03a9\)\]])\^(-?\d+)',
                   lambda m: f"{m.group(1)}<sup>{m.group(2)}</sup>", s)
        # Subíndices: x_{k+1}, x_k, α_k
        s = re.sub(r'([a-zA-Z\u03b1-\u03c9\u0391-\u03a9])_\{([^}]+)\}',
                   lambda m: f"{m.group(1)}<sub>{m.group(2)}</sub>", s)
        s = re.sub(r'([a-zA-Z\u03b1-\u03c9\u0391-\u03a9])_([a-zA-Z0-9\+\-\*]+)',
                   lambda m: f"{m.group(1)}<sub>{m.group(2)}</sub>", s)
        return s

    result = []
    for line in text.split('\n'):
        esc = _hl.escape(line)           # escapa < > & primero

        # Fracciones comunes
        for raw, uni in _FRACS:
            esc = esc.replace(raw, uni)

        # Matemática (super/subíndices)
        esc = _apply_math(esc)

        # Bold (**texto**)  — después de superíndices para no confundir x**2
        esc = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', esc)
        # Italic (*texto*)
        esc = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'<i>\1</i>', esc)

        # Encabezados
        if esc.startswith('## '):
            esc = (f'<br/><b><span style="color:{COLOR_HEAD}; font-size:11pt;">'
                   f'{esc[3:]}</span></b>')
        elif esc.startswith('### '):
            esc = (f'<b><span style="color:{COLOR_HEAD3}; font-size:11pt;">'
                   f'{esc[4:]}</span></b>')
        elif esc.startswith('# '):
            esc = (f'<br/><b><span style="color:{COLOR_HEAD}; font-size:11pt;">'
                   f'{esc[2:]}</span></b>')
        else:
            # Listas con bullet o guión
            bm = re.match(r'^(\s*)[•\-\*]\s+(.+)$', esc)
            if bm:
                indent = '&nbsp;&nbsp;' * (len(bm.group(1)) // 2 + 1)
                esc = f'{indent}<span style="color:{COLOR_BODY};">• {bm.group(2)}</span>'
            else:
                # Listas numeradas
                nm = re.match(r'^(\d+)\.\s+(.+)$', esc)
                if nm:
                    esc = (f'&nbsp;&nbsp;<b><span style="color:{COLOR_NUM};">'
                           f'{nm.group(1)}.</span></b> '
                           f'<span style="color:{COLOR_BODY};">{nm.group(2)}</span>')
                else:
                    esc = f'<span style="color:{COLOR_BODY};">{esc}</span>'

        result.append(esc)

    return '<br/>'.join(result)


def _table_md_to_html(raw_rows: str) -> str:
    """Convierte una tabla markdown (| col | col |, fila separadora ---) a una
    <table> HTML legible dentro del QTextEdit del chat (estilo Minimax:
    encabezado resaltado, filas alternadas)."""
    import html as _hl
    lines = [ln.strip() for ln in raw_rows.split('\n') if ln.strip()]
    if len(lines) < 2:
        return _md_to_html(raw_rows)

    def _cells(line: str) -> list:
        inner = line.strip().strip('|')
        return [c.strip() for c in inner.split('|')]

    header = _cells(lines[0])
    rows = [_cells(ln) for ln in lines[2:]]  # lines[1] es la fila separadora ---

    th = "".join(
        f'<th style="padding:4px 8px;border:1px solid rgba(80,120,200,140);'
        f'background:rgba(30,60,140,200);color:#dce8f8;">{_hl.escape(c)}</th>'
        for c in header)
    trs = []
    for i, row in enumerate(rows):
        bg = "rgba(10,20,50,160)" if i % 2 == 0 else "rgba(15,28,65,160)"
        tds = "".join(
            f'<td style="padding:4px 8px;border:1px solid rgba(60,90,160,110);'
            f'color:#c8ddf8;background:{bg};">{_hl.escape(c)}</td>'
            for c in row)
        trs.append(f'<tr>{tds}</tr>')

    return (f'<table cellspacing="0" style="border-collapse:collapse;font-size:10pt;'
            f'font-family:\'Segoe UI\',Consolas;margin:4px 0;">'
            f'<tr>{th}</tr>{"".join(trs)}</table>')


# ══════════════════════════════════════════════════════════════════════════════
#  RENDERIZADO MATEMÁTICO CON MATPLOTLIB — Genera imágenes PNG inline
# ══════════════════════════════════════════════════════════════════════════════
import re as _re_math

# ── Pre-inicializar matplotlib con backend Agg UNA SOLA VEZ al importar ──────
# Esto evita el crash cuando Qt llama a matplotlib desde el hilo principal
# (matplotlib.use() después de pyplot importado lanza RuntimeError)
_MPL_OK = False
try:
    import matplotlib as _mpl
    _mpl.use('Agg')          # debe llamarse ANTES del primer import de pyplot
    import matplotlib.pyplot as _plt
    _MPL_OK = True
except Exception:
    pass

# Patrón para detectar líneas con notación matemática
_MATH_PAT = _re_math.compile(
    r'\*\*[\d\{]|_\{[a-zA-Z\d]|_k\b|_\{k|∇|‖'
    r'|[αβγδλμσωφθ∂∞≤≥]'
    r'|\bexp\(|\bsqrt\(|\bsin\(|\bcos\(|\blog\('
    r'|x_k|f_k|d_k|x_\{|f_\{|\\alpha|\\nabla'
    r'|\bH\^|H_\{|f\(x\)|∑|∏|∫'
)

def _line_has_math(line: str) -> bool:
    """
    True SOLO si la línea contiene notación matemática real (sub/superíndices,
    letras griegas, normas, funciones matemáticas...) Y no es prosa normal.

    Regla clave: si hay 3+ palabras puramente alfabéticas (>=3 letras), la
    línea es texto descriptivo y se renderiza como HTML, no como PNG matplotlib
    (que causaba palabras pegadas y letras enormes).
    """
    s = line.strip().lstrip('•-* \t')
    if len(s) < 2:
        return False
    # Si hay 3+ palabras de texto (letras), es prosa — usar HTML normal
    prose_words = _re_math.findall(r'[a-zA-ZáéíóúñÁÉÍÓÚÑüÜ]{3,}', s)
    if len(prose_words) >= 3:
        return False
    return bool(_MATH_PAT.search(s))


def _to_mathtext(s: str) -> str:
    """
    Convierte una línea con notación Python/mixta al formato mathtext
    de matplotlib ($...$), listo para renderizar.
    Compatible con Python 3.14+ (sin bad escape en re.sub replacements).
    """
    s = _re_math.sub(r'^[\s•\-\*\t]+', '', s).strip()

    # Unicode → strings con replace() (evita bad-escape en Python 3.14)
    UNI = {
        'α': '\\alpha',  'β': '\\beta',   'γ': '\\gamma',
        'δ': '\\delta',  'λ': '\\lambda', 'μ': '\\mu',
        'σ': '\\sigma',  'ω': '\\omega',  'φ': '\\phi',
        'θ': '\\theta',  '∇': '\\nabla',  '∂': '\\partial',
        '∞': '\\infty',  '≤': '\\leq',    '≥': '\\geq',
        '≠': '\\neq',    '·': '\\cdot',   '×': '\\times',
        '±': '\\pm',     '‖': '\\|',      '∑': '\\sum',
        '∏': '\\prod',   '∫': '\\int',
        '½': '\\frac{1}{2}', '⅓': '\\frac{1}{3}', '¼': '\\frac{1}{4}',
    }
    for uc, lt in UNI.items():
        s = s.replace(uc, lt)

    # Palabras griegas → mathtext con lambda (evita bad escape Python 3.14+)
    for word, repl in [('alpha', '\\alpha'), ('beta',  '\\beta'),
                       ('lambda','\\lambda'), ('nabla', '\\nabla'),
                       ('mu',    '\\mu'),    ('sigma', '\\sigma')]:
        s = _re_math.sub(r'\b' + word + r'\b', lambda m, r=repl: r, s)

    # Potencias con lambda (evita r'\1^{\2}' interpretado como bad escape)
    s = _re_math.sub(r'([a-zA-Z0-9\)\|])\*\*\{([^}]+)\}',
                     lambda m: f"{m.group(1)}^{{{m.group(2)}}}", s)
    s = _re_math.sub(r'([a-zA-Z0-9\)\|])\*\*(-?\d+)',
                     lambda m: f"{m.group(1)}^{{{m.group(2)}}}", s)

    # Subíndices
    s = _re_math.sub(r'([a-zA-Z])_\{([^}]+)\}',
                     lambda m: f"{m.group(1)}_{{{m.group(2)}}}", s)
    s = _re_math.sub(r'([a-zA-Z])_([a-zA-Z0-9])',
                     lambda m: f"{m.group(1)}_{{{m.group(2)}}}", s)

    # Multiplicación
    s = _re_math.sub(r'(?<!\^)(?<!\{)\*(?!\*)(?!\{)',
                     lambda m: ' \\cdot ', s)

    # Funciones matemáticas
    s = _re_math.sub(r'exp\(([^)]+)\)',
                     lambda m: f"e^{{{m.group(1)}}}", s)
    s = _re_math.sub(r'sqrt\(([^)]+)\)',
                     lambda m: '\\sqrt{' + m.group(1) + '}', s)
    s = _re_math.sub(r'sin\(([^)]+)\)',
                     lambda m: '\\sin(' + m.group(1) + ')', s)
    s = _re_math.sub(r'cos\(([^)]+)\)',
                     lambda m: '\\cos(' + m.group(1) + ')', s)
    s = _re_math.sub(r'log\(([^)]+)\)',
                     lambda m: '\\log(' + m.group(1) + ')', s)

    # Fracciones dígito/dígito
    s = _re_math.sub(r'\b(\d+)/(\d+)\b',
                     lambda m: '\\frac{' + m.group(1) + '}{' + m.group(2) + '}', s)

    return f'${s}$'


def _render_math_img(mathtext_expr: str,
                     fontsize: float = 13,
                     fg: str = 'white',
                     dpi: int = 130) -> 'bytes | None':
    """Renderiza mathtext a PNG transparente usando el backend Agg pre-inicializado."""
    if not _MPL_OK:
        return None
    try:
        from io import BytesIO
        fig = _plt.figure(figsize=(0.01, 0.01))
        fig.patch.set_alpha(0.0)
        fig.text(0.5, 0.5, mathtext_expr,
                 fontsize=fontsize, color=fg,
                 ha='center', va='center')
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi,
                    bbox_inches='tight', pad_inches=0.08,
                    transparent=True)
        _plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ── Versión con display mode (más grande, centrada) ───────────────────────────
def _render_math_img_v2(mathtext_expr: str,
                        fontsize: float = 13,
                        fg: str = 'white',
                        dpi: int = 150,
                        display: bool = False) -> 'bytes | None':
    """
    Renderiza mathtext a PNG transparente.
    display=True → fuente 1.7× más grande, centrada (estilo imagen 2).
    """
    if not _MPL_OK:
        return None
    try:
        from io import BytesIO
        fs  = fontsize   # mismo tamaño para todos los bloques math
        pad = 0.12
        fig = _plt.figure(figsize=(0.01, 0.01))
        fig.patch.set_alpha(0.0)
        fig.text(0.5, 0.5, mathtext_expr,
                 fontsize=fs, color=fg,
                 ha='center', va='center')
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi,
                    bbox_inches='tight', pad_inches=pad,
                    transparent=True)
        _plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


_TABLE_ROW = _re_math.compile(r'^\|.*\|\s*$')
_TABLE_SEP = _re_math.compile(r'^\|?[\s:|-]+\|[\s:|-]*\|?$')


def _parse_response_blocks(text: str) -> list:
    """
    Divide la respuesta del AI en bloques tipados para renderizado rico:
      ('code',         lang,     content)  — bloque ```lang ... ```
      ('table',        '',       raw_rows) — tabla markdown (| col | col |, fila separadora --- )
      ('display_math', '',       expr)     — expresión math sola en su línea (grande, centrada)
      ('line',         '',       raw_line) — línea normal (texto / bullet / math inline)
    """
    CODE_FENCE = _re_math.compile(r'^```(\w*)\s*$')
    blocks = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        m = CODE_FENCE.match(lines[i].strip())
        if m:
            lang = m.group(1) or 'code'
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            blocks.append(('code', lang, '\n'.join(code_lines)))
            i += 1
            continue

        # Tabla markdown: fila de encabezado | ... | seguida de fila separadora
        # |---|---| (con opcionales : para alineación) — al menos 2 líneas.
        if (_TABLE_ROW.match(lines[i].strip())
                and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1].strip())):
            table_lines = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and _TABLE_ROW.match(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1
            blocks.append(('table', '', '\n'.join(table_lines)))
            continue

        raw      = lines[i]
        stripped = raw.strip()

        # Display math: corta (≤10 palabras), sin palabra española inicial larga
        # Se excluyen líneas que empiezan con ≥3 letras seguidas de espacio (prosa normal)
        is_display = (
            stripped
            and _line_has_math(stripped)
            and len(stripped.split()) <= 10
            and not _re_math.match(r'^[A-ZÁ-Úa-záéíóúñ]{3,}\s', stripped)
        )
        if is_display:
            blocks.append(('display_math', '', stripped))
        else:
            blocks.append(('line', '', raw))
        i += 1

    return blocks


# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEMAS PREDEFINIDOS — importables desde la interfaz principal
#  Uso: from ai_assistant import PROBLEM_PRESETS
#  Cada entrada contiene los datos listos para cargar en el optimizador.
# ══════════════════════════════════════════════════════════════════════════════
PROBLEM_PRESETS = {
    # ── Búsqueda 1D (Local / Fibonacci / Sección Áurea) ───────────────────────
    "búsqueda_local_aurea": {
        "label"      : "Búsqueda Local / Áurea",
        "fx"         : "x1**2 + 2*x2**2 + 1",
        "vars"       : ["x1", "x2"],
        "x0"         : [0.0, 0.0],
        "restriction": "",
        "lo"         : -5.0,
        "hi"         :  5.0,
        "metodos"    : ["Búsqueda Local", "Sección Áurea"],
    },
    # ── Newton-Raphson con restricción ────────────────────────────────────────
    "newton_raphson": {
        "label"      : "Newton-Raphson",
        "fx"         : "x1**2 + 2*x2**2 + 1",
        "vars"       : ["x1", "x2"],
        "x0"         : [0.0, 0.0],
        "restriction": "x1 - x2 - 1",
        "lo"         : -5.0,
        "hi"         :  5.0,
        "metodos"    : ["Newton-Raphson"],
    },
    # ── Gradiente Conjugado con restricción ───────────────────────────────────
    "gradiente_conjugado": {
        "label"      : "Gradiente Conjugado",
        "fx"         : "(x1+1)**2 + (x2-1)**2 + 2",
        "vars"       : ["x1", "x2"],
        "x0"         : [0.0, 0.0],
        "restriction": "2*x1 + x2 - 2",
        "lo"         : -5.0,
        "hi"         :  5.0,
        "metodos"    : ["Gradiente Conjugado"],
    },
    # ── Wolfe / Penalización / Barreras / Pen.BFGS / Pes.BFGS ────────────────
    "wolfe_md": {
        "label"      : "Wolfe / Penalización / Barreras / BFGS",
        "fx"         : "3*x1**2 + x2**2 + x1*x2",
        "vars"       : ["x1", "x2"],
        "x0"         : [0.0, 0.0],
        "restriction": "x1 + 2*x2 - 3",
        "lo"         : -5.0,
        "hi"         :  5.0,
        "metodos"    : ["Wolfe", "MD: Penalización (Newton)",
                        "MD: Barreras (Newton)",
                        "MD: Pen./Pes./Sum. (BFGS)",
                        "MD: Pes. (Nelder-Mead)"],
    },
    # ── Armijo ────────────────────────────────────────────────────────────────
    "armijo": {
        "label"      : "Armijo",
        "fx"         : "(x1-1)**2 + 2*(x2+2)**2 + exp(x1+x2)",
        "vars"       : ["x1", "x2"],
        "x0"         : [2.0, -1.0],
        "restriction": "",
        "lo"         : -5.0,
        "hi"         :  5.0,
        "metodos"    : ["Armijo"],
    },
    # ── Fibonacci 1D ─────────────────────────────────────────────────────────
    "fibonacci": {
        "label"      : "Fibonacci",
        "fx"         : "(x1-3)**2 + 4",
        "vars"       : ["x1"],
        "x0"         : [0.0],
        "restriction": "x1 - 2",
        "lo"         : -5.0,
        "hi"         :  5.0,
        "metodos"    : ["Fibonacci"],
    },
    # ── Nelder-Mead ───────────────────────────────────────────────────────────
    "nelder_mead": {
        "label"      : "Nelder-Mead",
        "fx"         : "(x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2",
        "vars"       : ["x1", "x2"],
        "x0"         : [0.0, 0.0],
        "restriction": "x1 + x2 - 5",
        "lo"         : -6.0,
        "hi"         :  6.0,
        "metodos"    : ["MD: Pes. (Nelder-Mead)"],
    },
    # ── Sum.BFGS / Recocido Simulado / Algoritmo Genético ────────────────────
    "metaheuristicos": {
        "label"      : "Sum.BFGS / Recocido Simulado / GA",
        "fx"         : "(x1-1)**2 + (x2-2)**2 + log(1+exp(x1+x2))",
        "vars"       : ["x1", "x2"],
        "x0"         : [0.0, 0.0],
        "restriction": "x1 + x2 - 2",
        "lo"         : -5.0,
        "hi"         :  5.0,
        "metodos"    : ["MD: Pen./Pes./Sum. (BFGS)",
                        "SA: Recocido Simulado",
                        "GA: Algoritmo Genético"],
    },
    # ── Goldstein ─────────────────────────────────────────────────────────────
    "goldstein": {
        "label"      : "Goldstein",
        "fx"         : "0.5*(x1**2 + 4*x2**2 + 2*x1*x2)",
        "vars"       : ["x1", "x2"],
        "x0"         : [0.0, 0.0],
        "restriction": "x1 - 2*x2",
        "lo"         : -5.0,
        "hi"         :  5.0,
        "metodos"    : ["Goldstein"],
    },
    # ── Enjambre PSO 1 ────────────────────────────────────────────────────────
    "pso_1": {
        "label"      : "PSO — Enjambre 1",
        "fx"         : "x1**4 + x2**4 - 3*x1**2 - 3*x2**2 + 2",
        "vars"       : ["x1", "x2"],
        "x0"         : [0.0, 0.0],
        "restriction": "x1 + x2 - 1",
        "lo"         : -3.0,
        "hi"         :  3.0,
        "metodos"    : ["PSO: Enjambre"],
    },
    # ── Enjambre PSO 2 ────────────────────────────────────────────────────────
    "pso_2": {
        "label"      : "PSO — Enjambre 2",
        "fx"         : "sin(x1) + cos(x2) + 0.1*(x1**2 + x2**2)",
        "vars"       : ["x1", "x2"],
        "x0"         : [2.0, 2.0],
        "restriction": "x1 - x2",
        "lo"         : -5.0,
        "hi"         :  5.0,
        "metodos"    : ["PSO: Enjambre"],
    },
    # ── Prueba general ────────────────────────────────────────────────────────
    "prueba": {
        "label"      : "Prueba general",
        "fx"         : "(x1-2)**2 + exp(x1)",
        "vars"       : ["x1"],
        "x0"         : [0.0],
        "restriction": "",
        "lo"         : -2.0,
        "hi"         :  4.0,
        "metodos"    : ["Búsqueda Local", "Fibonacci", "Sección Áurea",
                        "Armijo", "Wolfe", "Goldstein"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  REGLAS DE COMPATIBILIDAD — método → requisitos mínimos
#  La app principal llama: panel_ai.check_method_compatibility(metodo, fx,
#                                           n_vars, tiene_restriccion)
#  y el asistente muestra un aviso instantáneo si hay incompatibilidad.
# ══════════════════════════════════════════════════════════════════════════════
_METHOD_RULES: Dict[str, Dict] = {
    # ── 1D estrictos ──────────────────────────────────────────────────────────
    "Búsqueda Local": {
        "max_vars"         : 1,
        "necesita_restriccion": False,
        "acepta_restriccion"  : False,
        "necesita_gradiente"  : False,
        "categoria"           : "1D clásico",
        "alternativas_nd"     : ["Armijo", "Wolfe", "MD: Pes. (Nelder-Mead)",
                                  "PSO: Enjambre", "GA: Algoritmo Genético"],
    },
    "Fibonacci": {
        "max_vars"         : 1,
        "necesita_restriccion": False,
        "acepta_restriccion"  : False,
        "necesita_gradiente"  : False,
        "categoria"           : "1D clásico",
        "alternativas_nd"     : ["Armijo", "Wolfe", "MD: Pes. (Nelder-Mead)",
                                  "PSO: Enjambre", "GA: Algoritmo Genético"],
    },
    "Sección Áurea": {
        "max_vars"         : 1,
        "necesita_restriccion": False,
        "acepta_restriccion"  : False,
        "necesita_gradiente"  : False,
        "categoria"           : "1D clásico",
        "alternativas_nd"     : ["Armijo", "Wolfe", "MD: Pes. (Nelder-Mead)",
                                  "PSO: Enjambre", "GA: Algoritmo Genético"],
    },
    # ── Búsqueda de línea (sin restricción nativa) ────────────────────────────
    "Armijo": {
        "max_vars"            : None,   # sin límite
        "necesita_restriccion": False,
        "acepta_restriccion"  : False,
        "necesita_gradiente"  : True,
        "categoria"           : "búsqueda de línea",
        "alternativas_restriccion": ["MD: Penalización (Newton)",
                                      "MD: Pen./Pes./Sum. (BFGS)",
                                      "SA: Recocido Simulado",
                                      "GA: Algoritmo Genético"],
    },
    "Wolfe": {
        "max_vars"            : None,
        "necesita_restriccion": False,
        "acepta_restriccion"  : False,
        "necesita_gradiente"  : True,
        "categoria"           : "búsqueda de línea",
        "alternativas_restriccion": ["MD: Penalización (Newton)",
                                      "MD: Pen./Pes./Sum. (BFGS)",
                                      "SA: Recocido Simulado",
                                      "GA: Algoritmo Genético"],
    },
    "Goldstein": {
        "max_vars"            : None,
        "necesita_restriccion": False,
        "acepta_restriccion"  : False,
        "necesita_gradiente"  : True,
        "categoria"           : "búsqueda de línea",
        "alternativas_restriccion": ["MD: Penalización (Newton)",
                                      "MD: Pen./Pes./Sum. (BFGS)",
                                      "SA: Recocido Simulado",
                                      "GA: Algoritmo Genético"],
    },
    # ── Newton clásico ────────────────────────────────────────────────────────
    "Newton-Raphson": {
        "max_vars"            : None,
        "necesita_restriccion": False,
        "acepta_restriccion"  : True,
        "necesita_gradiente"  : True,
        "necesita_hessiano"   : True,
        "categoria"           : "Newton",
        "alternativas_restriccion": ["MD: Penalización (Newton)",
                                      "MD: Barreras (Newton)",
                                      "MD: Pen./Pes./Sum. (BFGS)"],
    },
    "Gradiente Conjugado": {
        "max_vars"            : None,
        "necesita_restriccion": False,
        "acepta_restriccion"  : True,
        "necesita_gradiente"  : True,
        "categoria"           : "Newton",
        "alternativas_restriccion": ["MD: Penalización (Newton)",
                                      "MD: Pen./Pes./Sum. (BFGS)"],
    },
    # ── MD con restricción ────────────────────────────────────────────────────
    "MD: Penalización (Newton)": {
        "max_vars"            : None,
        "necesita_restriccion": True,
        "acepta_restriccion"  : True,
        "necesita_gradiente"  : True,
        "categoria"           : "MD restringido",
        "alternativas_sin_restriccion": ["Newton-Raphson", "Wolfe", "Armijo",
                                          "Gradiente Conjugado"],
    },
    "MD: Barreras (Newton)": {
        "max_vars"            : None,
        "necesita_restriccion": True,
        "acepta_restriccion"  : True,
        "necesita_gradiente"  : True,
        "categoria"           : "MD restringido",
        "alternativas_sin_restriccion": ["Newton-Raphson", "Wolfe", "Armijo"],
    },
    "MD: Pen./Pes./Sum. (BFGS)": {
        "max_vars"            : None,
        "necesita_restriccion": True,
        "acepta_restriccion"  : True,
        "necesita_gradiente"  : True,
        "categoria"           : "MD restringido",
        "alternativas_sin_restriccion": ["Wolfe", "Newton-Raphson",
                                          "Gradiente Conjugado"],
    },
    "MD: Pes. (Nelder-Mead)": {
        "max_vars"            : None,
        "necesita_restriccion": True,
        "acepta_restriccion"  : True,
        "necesita_gradiente"  : False,
        "categoria"           : "MD restringido",
        "alternativas_sin_restriccion": ["Búsqueda Local", "Fibonacci",
                                          "PSO: Enjambre", "GA: Algoritmo Genético"],
    },
    # ── Metaheurísticos (más flexibles) ──────────────────────────────────────
    "SA: Recocido Simulado": {
        "max_vars"            : None,
        "necesita_restriccion": False,
        "acepta_restriccion"  : True,
        "necesita_gradiente"  : False,
        "categoria"           : "metaheurístico",
    },
    "PSO: Enjambre": {
        "max_vars"            : None,
        "necesita_restriccion": False,
        "acepta_restriccion"  : True,
        "necesita_gradiente"  : False,
        "categoria"           : "metaheurístico",
    },
    "GA: Algoritmo Genético": {
        "max_vars"            : None,
        "necesita_restriccion": False,
        "acepta_restriccion"  : True,
        "necesita_gradiente"  : False,
        "categoria"           : "metaheurístico",
    },
}


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

### Multidimensional (MD) — restringidos, los 6 REQUIEREN g(x):
- MD: Penalización (Newton): φ(x) = f(x) + μ·max(0,g(x))², μ crece x10, Newton propio
- MD: Barreras (Newton): φ(x) = f(x) - (1/t)·ln(-g(x)), requiere punto factible interior
- MD: Pen. (BFGS): misma penalización cuadrática que Penalización (Newton), resuelta con BFGS
- MD: Pes. (BFGS): penalización LINEAL φ=f(x)+w·max(0,g(x)), peso w creciente, con BFGS
- MD: Pes. (Nelder-Mead): misma penalización lineal que Pes. (BFGS), pero sin derivadas (simplex)
- MD: Sum. (BFGS): suma ingenua h(x)=f(x)+g(x) de una sola pasada — contraejemplo pedagógico,
  NO garantiza factibilidad (a diferencia de los otros 5)
Si el usuario pide un método MD sin haber definido g(x), explica que esos 6 métodos
necesitan una restricción para tener sentido — la app ya bloquea la ejecución en ese caso.

### Metaheurísticos:
- SA: Recocido Simulado: acepta peores con P=exp(-ΔE/T), T decrece
- PSO: Enjambre: v_i = w·v_i + c₁·r₁·(pbest-x_i) + c₂·r₂·(gbest-x_i)
- GA: Algoritmo Genético: selección torneo + cruce aritmético + mutación gaussiana

### Multiobjetivo (MO) — 14 métodos, todos parten de una f(x) mono-objetivo de 1 variable
(la app genera automáticamente F(x)=[f1,f2] con f1=f(x), f2=(x-x_centro)²):
- Escalarización (Suma Ponderada): φ=w1·f1+w2·f2 — simple, no cubre frentes cóncavos
- MO — Bisección / MO — Sección Dorada: la misma suma ponderada, resuelta con esos solvers
- MO — Frente de Pareto: barre α∈[0,1] y devuelve el frente completo, no un solo punto
- MO — Análisis Jacobiano: diagnóstico de estacionariedad de Pareto en una malla de puntos
- MO — Lexicográfico: prioridad estricta f1≻f2
- MO — Goal Programming: minimiza desviación respecto a metas por objetivo
- MO — ε-Constraint: deja un objetivo libre, el otro como restricción fᵢ≤εᵢ
- MO — ASF (Logro) / MO — Chebyshev / MO — Punto de Referencia: cubren TODO el frente
  de Pareto (incluso regiones cóncavas), minimizando distancia a un punto de referencia
- MO — NBI / MO — Restricción Normal: distribución uniforme de puntos en el frente
- MO — Peso Adaptativo: suma ponderada + penalización de cercanía a puntos ya explorados
Si el usuario da una función de 2+ variables para un método MO, la app la rechaza — estos
métodos siempre parten de f(x) de 1 sola variable.

## FUNCIONES SOPORTADAS (sintaxis Python/SymPy):
- Polinomiales: x**2 - 4*x + 5, 2*(x1-3)**2 + x1*x2**3
- Trigonométricas: cos(x)*exp(-x**2), sin(x)/x
- Exponenciales: exp(-x**2), log(x+1)
- Valor absoluto y funciones especiales: Abs(x-2), Max(x,y) (con mayúscula inicial, notación Sympy)
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
Solo rechaza con el mensaje de abajo cuando la consulta NO tenga NINGUNA relación con:
- Matemáticas, álgebra, cálculo, análisis numérico o estadística
- Programación, Python, algoritmos o software
- Optimización, convergencia, funciones, gráficas
- Esta aplicación y sus métodos

Ejemplos de consultas que SÍ debes responder (aunque parezcan generales):
- "¿qué es un gradiente?" → responde, es matemática
- "¿cómo funciona exp()?" → responde, es una función soportada
- "¿qué significa converger?" → responde, es del dominio
- "explícame Armijo aunque no lo haya usado" → responde
- cualquier pregunta sobre métodos, funciones, parámetros o resultados del optimizador

Solo debes responder con el mensaje de abajo ante preguntas completamente ajenas
como: recetas de cocina, deportes, noticias, música, chistes, política, etc.

DEBES responder EXACTAMENTE este mensaje solo en esos casos:
"⚠️ Esta consulta está fuera del dominio del Optimizador de Funciones. \
Solo puedo asistirte con temas relacionados a la optimización matemática \
y los métodos implementados en esta aplicación (Armijo, Wolfe, Fibonacci, \
Newton-Raphson, PSO, GA, Recocido Simulado, etc.). \
¿En qué puedo ayudarte con el optimizador?"

## TONO Y ESTILO CONVERSACIONAL:
- Hablas como un asistente humano cercano, no como un manual — español latino
  neutro, natural, directo. Nada de "estimado usuario" ni tono de formulario.
- Puedes abrir con una frase breve y cálida ("Buena pregunta", "Vamos a verlo")
  antes de entrar en materia, sin sonar forzado ni repetir la misma muletilla siempre.
- Sé claro y ve al punto: no rellenes con paja ni repitas la pregunta del usuario
  antes de responder.
- Respuestas completas y detalladas según lo que se pida (sin límite de palabras
  arbitrario), pero organizadas: usa encabezados (##), listas y **negritas** para
  que se lea fácil, no un bloque de texto corrido.
- Usa notación matemática clara: f(x), ∇f, α, μ, etc. Cuando ayude a entender,
  arma una tabla en markdown (| columna | columna |) en vez de una lista larga.
- Cuando recomiendes un método, explica brevemente POR QUÉ — no des una
  recomendación sin razón.
- Si hay un error en la función del usuario, muestra la corrección concreta,
  no solo "está mal".
- Idioma: SIEMPRE español latino.

## CONTEXTO ACTUAL (se actualiza con cada cálculo):
{context}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Detectar proveedor por formato de key
# ══════════════════════════════════════════════════════════════════════════════
def _detect_provider(key: str) -> str:
    """
    Detecta el proveedor según el formato de la API Key:
      • 'gsk_...'   → Groq       (gratis, 30 req/min)
      • 'sk-or-...' → OpenRouter (gratis, modelos variados)
    """
    if key.startswith("gsk_"):   return "groq"
    if key.startswith("sk-or-"): return "openrouter"
    return "unknown"


# Base64 de la imagen de fondo del panel AI (340×340 JPEG)
_AI_PANEL_BG_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5Ojf/2wBDAQoKCg0MDRoPDxo3JR8lNzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzf/wAARCAFUAVQDASIAAhEBAxEB/8QAGwABAQADAQEBAAAAAAAAAAAAAAECBAUDBgf/xABJEAABAwIEAwUDCAcFBgcAAAABAAIDBBEFEiExQVFxBhMiYYEykbEUIzNCUnKhwRVDYoKy0eEHJFPw8RY1RFRjdCU0ZHOSwtL/xAAZAQEBAQEBAQAAAAAAAAAAAAAAAQIDBAX/xAArEQEBAAICAQMCBQQDAAAAAAAAAQIRAyExBBJBIlEFEzJhcRQzwfBCUtH/2gAMAwEAAhEDEQA/APgEURfReIREugihS6hRUKxKpUKgixKpUKKLFUqKCFRUrFFEuixUqhKmZCsVBbqFynFQoooVVCoIVFSoooiKICXRRQCiiIoVFSoghREUEREUEREVaEVRQYlFVEBERB37qrFLr0vOpUS6l0BQoSpdQCsSqViUBYqlQqKhUREVCpwVKhUEKiFRFQqFCsXOUVVCgOiKAsSqogFRCiiiiqhQFERQRE4ptv7lFUC5sFi4WJGmnJCfcoSgKIqAiolivVkZcbALrUOAVdW0FjAHPZmha7Tvrbhp2uOSuONy8OefJjhN5VxMpSy+hHZqtdYRiKSQ7xMkBcPy9xWlXYTNSZi4B8bXZTLHqzNxbfiQtXjynmM4+o48rqVyii9HxlvBeZC5u2xRCoiqiiIO7dLrG6L0POyupdREFUuihKKFRFCUAlYlCooBUQooqLE7rJYoqXQooUGJWLhcrI7KKKg0RVRRRRFEBEQoIoqopQUOiv8AVYu/JRYt7AeairmOaxjiNHA2PPVYlQFEXp4YwNMzyAddhfy4qiNie5uYNNjt59Oa3cNw6atc7umEsZYyOAvkbfcgakdFpZnOdmcSTzX0fZCQHGqUvY9zwTlcxmYg2NiRxAK1xyXLVcefPLDjuUdegwakw9jJal15CAQWuu48WviI24XDgu1R4fV4jdtPGKeB5zSZRbOebraX6aeS88Rwx8lQ6ohcHPkcSWh+ZrzxyO4/dOo811cHxgQMbTua6zdCx3tNPl/Jejn/ADOPjv5OO6+b6a8PNyS8+fXz+3/n89rN2TiZFmhqH96OBFh6FcephfE/u8SYbgZGzhoc5jTvYHS552v1X18+IwNiz5w7S+Ub/wBPVfNYviPy9vyeJmd5PhawXI9eK8H4fz+q5MrOWdffw+n+J+n9HhjLx5fV9p3199/H+fs+Vxfs80RmemLGstmNnXja22njdu4ngAvlKmB8MhZIxzHDdrhYhfp2GUb6Vsk80gyNBc4WzMZbjfi4cLe9fnGJyvmnfJLI6R7jcucbk+q9vPhJNvJ6LmyyyuO9yfLQKipUXlfSEREHaul1jdLr0ODK6XWN1LoMrqKXS6gqxJQlRAUKXUKKql1EKgXUKXWJRQqISooIURFFBuo7h0VG6h4dEEURHAtNioooiFBCg3RQbqC/1WLlSViVFgoiyawFuZzrNvbTcqiMaXuDWgknYALadTvLtbANY3M6+g0Xh3mmVgytO9tz1K6tC3O13Rv8PJWRnK1o/JwfozmtuLWK9aKslpHkxPLQ4WeASA9t72NuC3GBrcwYYyct/m7/AOfRcsPLGvc22YWseSvhmyZTVffYR2ga9rIZAA59gYclw4X0ZHGNB95xuuy+ngrmFzHXkZZrsr8zo3fZD9n9N/Nfk0c7mm9zddij7QVVPGGhwc5jcsT369yOOUbAnna69GHPPGT5vN+H2X3cV1X2UNHNLO+N9SHDiIyS49QbZf3ltSfJ8PjkY1oJZ9KGguycnP2c5vTRfFjtZiYaxrqova3Qte0EPHJ3P1WlX41JVWbbJC2/dxhx+bB4B29vLZavPjJ05/0XNndZXr9n0WMY8XuJpn5poycuR/0LhxjcPaYR9UhfGVk5nmfK62Z7iTlFhfos3yd4C4uLgPrgeIdR+a8ZDf6XjtI3W/Xn8V5uTkuT6PBwY8U1I8CVFXtLHlptcclFyesRRFB10updF6HBboovSKMyOIHAXQeaLKRuR5aeCwQUlRFLoooUuhUBQqFEVFCqsSoBQBFsQwOe0u2a3c8kLdPAtWBW73IfcREucPqkarVe2wuEJWDd0OvRRL6FZVCbbaKIiKKFLoUEKIiKhUVUUAgjcELJx+Zb94/kvWpqBM1gDA22p8yvF30TfvH8kRiN10IKkwOGgILW6enArnL3qKkzyNd3cceVjWZY22BsLX6nikuizbdkrPCbEvJA1cALeg4+a1Gsa+mqHmaNrm5Msbicz9eHTivAuKxJS5J7UVzFRRZbXMUuoiDIOLTmaSCNiF61NrOAAHzh0HQLXOxWxU7P/wDcPwCfCfMYT/TO9PgvNZ1H0zvT4LzS+VngRERXUurdYou7iyuq17mG7TYrFS6gyJLjcm5UuooSgqKXRFFEuogql0UUUKhRTdBW7rp0Fu7eCLnMNjqNDsuXe22i26WoMV9Lg7hIzlG+Axodlc11xrlbYf6rkSG4K3ZKvchznEjd1tFoPcDdW1MXmSmmU33uiiw6CiIoCizYx8hysaT+S2Y6ZjdZCHnlw/qqm2tFE+X2RpxJ2Cxe3K4tvexsum5j2taXNIaR4dLA9F4w4XWV7pH0cDprSZS1m40vfp/NL0sc9QrpDAsVcAW0MzwQ03YAbZrWvyOo6XF1g3BsRfKGCkkBuLk7AeHXpZzTccCs7VoL1e6A0kbWiTvw9xeSRly6Wtxvotx2CVjaN9XaMxMfKxxD9QY7ZtPPhzseS52yspZtFFla6WUVii2YaKeaHvYoy9vetis3Ul7gSABx2K9RhOId4I/kU2YvyAFu7rXsOenJQaCLpDB6pzah3gAp4WTSXJuA4aCwG+vQcSsP0NifeGMUFRnBaC3Jr4tvfYoNBFvOwevZSPqX00jY2WzAjUAgkOty0Wiioditqq7nunWe/v8AvTdmXw5co1vfe/BatrrYxKnmpa2aCojMcrD4mnhpdX4ZvmPOo+md6fBea9Kj6Z3p8AvNSrPAiIiukil0uuziqKJdFVS6XUQVS6JuoCFUKO4dEEKiKXUUug3CAcToEJFxYWQRUOQ5bE3N+VlggzLlgUUUBEtc2G5WzDSSPdlcHZjsxouf6INdrS5wDQSTyWzFSgaym5+y0/muhFQtax2oJAuWxnRv3nbei94qeNhzlrXlx8Nwcn7rd3fgETe/DTjp3uZdrQyIHVx0aF6tfFB9G3vHcHPboOg/mtisMd297I+7RYRXFx6DRvRc9VGUsr5XF0ji4niVqsr6ml76GCQNje/M4FoNzp/ILKWZkehNzyC0nuzOLjpc3spWo6Rx/EiLGVntB30Y3Gl/cAPQK0/aHE6aKKKOduWKEQxh0bSWsBJte37R9LcgtKmpJalj3RZbNIabk3uduHkvOaCWAnvGEAG2bgdLhY3N6b1dbbtNiVU7vKZ1Y2CCoEjJXOjzCz3BzrgC+4G2o4LVraGoonsE7BlkGaORjg5kg5tcNCPhxstcLeoa+WlY6Ehk1NIbyU8ouxx58w79oWKDSaLrMN9V1BhrK5pkwgve4DM+kebytHEt/wAQdPEOI4rSbGs2tyPegxCpoGkU/d2MjZQXxhxa9t7EX2OpWyceru8hka2mBgFobQC0TbWyt5DS/VajYSdgsu4sNln3Rr2PehqsVnxGSoowx1Y6P6TI0Fg9m7SbAHUDzutqKv7RTunZFTuzPcO9vTWNzfS52uWnTmDZauHy1VNUOFEcs87O4a7iMxGx4G4Gq2pO09Z3McPcwBsUeRpBdodfFvvqferKzY5M+KVpzMlyNdYtee6DXOuwsJd55SQucVt19TJW1UlTLbPJbNbyAH5LVI5LTLFDrfmiIPeuNOap5pHSOhNsplaA7YXuB53XgnVFakmpoREQdBFEXVzVFEQVRS6IrIapw9Ao06q8PREL/msXcOipKxJUUUREFO6xJR26igIiICiqiDoYZG2SzcryXF18lsxsBpc7BdMMDYzYMbFxDXEM9X7uPkFzcMbcsBAN3O0yF19BwG661s5NwSWixIcCWjbV3ssHkNUqal7QZnANa3xN8TGZdvMM2A83LN4uDJ9X6zi7T95/Ho3RadRVwwNc0Bj9dhcMPXi49VzKivlnIu4m2gJ4DyHBRpsVMsTHEh3E7C1+gWlJUOkFm+EfivF29zqSvajdCx0rp25vmzkFr+LS35pctRNa7eB0/moutidFFFSR1kMcrYZfDG8tIY8g6gc9Neq5VtFnHOZTcMcvdNs4p5IR8261nh9vMXt8Su1iGHSVdHTT0U0lSHwtkljc0BzSLjwjctAuONrLg8V9XTEsw+gLSQRTtNx95y481uOso9XBJlvGvlCcpW1DRVMjc0cLnjT2CCdTYaDXU6Lu1dHTYiS6cd1Of17W7/eHHqNeq5FXhs1BI0TMFnaskbq1w5gq48sy6+Uy4rh/Drx4Q04XTVNIZm1mZ2eN/hJLTbweYP8Am6ybLHiLy3E2mOpGnypjdSf+q3633h4ud1uYHI84VA17i5hMjQHeKwzA6cteS+hOGU1e0OfE+KWxtKAT0FuI4an1XXlw3jLHn4eSzK45eNvmJMImpy0SNaWuF2PYczHjmDxXlLREDZfsnYfsrBHRyuxCNsznv0Y/Vgt9ax4+ay7ddmKAYVNWU1PHDNAMx7sZQ9vEED4r5uWeU7+H0pcLfa/F6KB1LUOq5GEMpm5wCN3keAe/XoCuHMwDQa2X13anEP0lO17adlOxrQO6jcSy4AF7HjYAcdl8rONSvTx5bcc8dNF68nL3fqV5Ef54LvHnrzVc1zLZmkXFxcbhCQPM+ajnOcbuJJ21K0iInRFACIiDeRS6Ls5KiAFzg1ouTsq9jmWLhodkVil1EugyBUJUuooLdLqKta57gGgkngEEuso2OkNmi/5LYZSgaym5+y0/Er2sALAAAcBsrpNtWSERsBLsxvw2C8XWzHLe3C62qsEMAIIN+K1FKQRFLqKJdLp0Qb+HytiiD5AXAOdpmLb6DiOCwqK98nhaQG30a0WaOgWlckWJNt7J0U2Mw2SZ9gHPda9gLlYPDmmzgWnkVnDM6EuLbEuYWai9gV9E6ikr+zf6QgMUkmd0TqaO5kaBZxd58NORXPPk9lm/FZyy9utvmbpa+yiE+i6NvsMImL+zlNBU2np+9lHcyatsC21uI3Oo5rl4hgReXS4XmkG5pnG8jfu/bH4+RW7g5zYHTX4TTa//ABXUbheIGITNoKsxnUPED7db2Xzfdlhnbj968fuyxzvtfBbGzgdDbzC+wwemdX/oijjcGmdkcQcdhmeRf8UxCigr3WxBroqjhUNbZ/7wNsw/HzXX7LYbNH2gwKlY9kndGOV8sZ8Ija8uLvIADiumfNOSSfL6PpOXHK376bk/Z2gqn1MPZ+snqK2lc5slJUsa18oabF0dtHbezuuDE4gOjkjbJG42fFKNCfyPnuF6YhWGTGqqtpHujLql8sT2GxF3Egg8F2D2hoq353G8Eiq6se1UwzOgdJ98AWJ89Fyuq903F7M0TaiGsFVUPosNoaqSN9S8j50udcRs2u64224rvRdo/kUpiwmjZStB1mn+dndzJJ0F97Bfn3azH6jFcQdTmOKmo6R72QUsIsxmurvNx4krHDMZkhDY6i8sQ0Gvib0P5H8F7cc94yZPBlxaytxfsWCdp5WFzq175gdc19RYLk9r+2seIYeaSjikZHN7T5BYuA4D818zSV7ZYy+nkD2ZXXI0I8J0I4LgU1eTGYZiMrh4c2ovppqfCCbXI1Fl5+fh7lny9HByY3e53GpXPDnHVc6aJpic4nXldbs4jqZMkDwyXNlMb3DKXXPsu2sBbcrk1JdG7JK1zSQHAOFiQdj0W+OaOTLda0lh5rwcbr0kK8nLvHCsSOSxVJUVZEREBERBuJdRF1c2TXFrg4bg3CzmmdMQX202svJRBUURBUGpsN16QQGUjWwJsABck+QXUhoREcrwWutfu26vI8zwRLdOfFSuc6z7g/ZGp/otrII7xhobY2IH5ldCJrTEWtDWstZ/dus3UfWed+gRkEbLSvY0Z9i9ptt9Vm7up0TbPbTjp3ubnd4Ir+27b05r0ika1wbSxZ5eD3i5HQcOq26iSIPzSEgkbaF5B6aN/quZU1zWNMcQDG/ZZx6lNrp41znPbme4ucXaklaSykmdJ7RsOACwUrUhfmiibKKJsl77rdw9tI6zaqwLpQLk2AbY8eGtlZNpbqbaV77qHRdPGsM/R3cHLI3vw57e8Frs0ym3Dc+5cy9vNQxsym4WX0uAuLcGBaSLVT9f3WL5o6r7nsNhcFbgNVV4hUupqCkqCZJGMzPc5zWBrGjiTqvP6ibwc+WW49NKsoaXEbunHc1J/wCIY32vvt49Rr1XzeI4fU0DwJ2DI72JWG7H9D+W6/QcWwaOCjZiWF1PyzDXuyGUsyvhf9l7eB5HYrlMks10bmtfG/22PF2u6j/JXmw5cuPr4ccc8uPq+HQ/s4gj+QGunjbI3D4aqqbG8XDnjIG3HK5B9Fk7tNjr5XSnF64PcbkidwHuGgXQ7MRUsOE4xBRZmE4ZO7uXEusXSsAseIu06HXqvmnAtcQQQQbEHgsZZbu5+6ZZdbnzt34+2OPBmR9f37eVREyT4hcvtB24xdr46GQwijnpmmaOnhbC513O+s0babHQrTXF7T6VlNf/AJVnxct8X1Z6rpwcmU5J26sTo54TPTSCWMe1bRzPvDh1281c17X2XysE81NK2WCR0cjdnNP+fcu/heIxV8scEzRDUvcGsLR4JCeFvqn8Oi3ycVx7ncfX4+aZdXqtHFTbFaw/9d/xK8mSLZx2iqYMYqxJEQXSvcAHDa+/4rnuD2Zc7S3MLi43C7YauM04ZbmV262G4pNhtSKinLe8DS3xC4sfJdClraGqlp2SwtzNEAkHd37w5/nAABckgjrY2svmM5XrR1b6SrhqIwC6J4eAdiQbrWrrSbm9/LrYm+ANjgl+bmjpm+MtIu4X8GW3G48XMLRhFZNaP5M+ojyiTu3MJ8NiGkEaga6WNj5rapu0s9OyFjoIZWxm5DrgPdcEE24i2nUrylx+WSpfMaaHUxvaxpc0Mcy+Uix4ZjokxS5ObUiJzs1EyfumsbmMtiQ63i1GltCRxstdjHTPayJpc9zg1rW6kk7BfQzdq5TJ83SsMYLXBsjyfEBbUbW1NhsFjD2sq2NPeU1PM/KGiSQuuAMv5tB6krbL598MrImzPje2J5Ia8tIDrb2KGCUSmExvEoNizL4gei6tDjzqKnbEKWKUtzfOPe64BN8o5Nvu3Y67XK2D2pmzPkbRwB79Hlr3DMASQLctfVEfPXtyRduq7SVFRTyQfJ4WMkZkflvqLEW6DNcDYWXE3QERERs3VWKLowt1VisgNL+SAirjp6rEqjqYUCS3KXg2f7Dg07Didguiy2SxEYiJ3uWxk/xPP4LmYYWAMMhYG+LV7MwG3DiVs1NeGkmMnNa3ev1eR5fZ9ERtzERBpmJFhpnALh91mzepWlU4iWl2Qlhdu693uHmVzZal7ybEi+5J1K1z5ps09pKhz9Aco8uK8UV4eSjStawscS4h4Iyttv6qbbp0QOH1hfTTXZRA+SxXQhwx08QdHK3N3QkLTwBv/IrUqIjBKY5CCRbVvmL/AJq2XW0mUt08lb2318lCeWyijT7ORwr8PoWYiDUs+Sx5bus9mm7Tw/EHiuDiOCTU0bqild8ophq4gWfGP2m/mLjou3DpRUP/AGsXwXo17mPD2uLXjYtNiF83DPLjt9vh4sc7henxW6/QsHHcf2b0zT/xeKyyOPkxjWj4lceuwqlrsz2ltJUHXMB808+YHsnzGnku4yB9J2FwWlrHRwzOrqvu2vdpJ7Hsu26a6rvny454XXl3vJM8LrymCYtJhNS4mNtRSTt7uppn+zMzl5HkeBW/XdmZahvy3s02TEcPedBGM00B+xI3e458V8+WuY4tc0tcNC0i1lnFUSwOzU00kT/tRvLT7wvLL93nl61W/wBqnO7O9jzhlVGYsWxVze8jd7UVMx5cL8i5x25BfJUOPvGWLEQ6dgFhMPpWf/oeR1816dq3ukFA6RznPLJMznG5Jz8SuAvZx4Y58clj1YY454Tp9q1rXwiogkZNA42EjNr8jxB8iuH2pINbTW/5RnxcudRVlTRSGWllLCRZwOrXjk4bELs11ZhGIyidsEsRhospa9x8UuYWygG9hd3H4LGHFcOSfMYx47hnL8PnwDx2WQdY3aSD+K7GJ4fQQQyOpqjM9kZLm5r5DnAbfnmab6bLiL1PS3I8RqWva4yFxYLNLiTY5s1+t151FS6ctLgAQDtxJJJPvK8OHiXrFC+TVos37R2WfbPiNe6/LG9va9y9o4XvFz4WeY39OK9ooo47EDM77Th8AsySTcnVb9rHuc4uWJKp310WJPostLfmsSiXQROitv8ARZSvL5HOda5NzYWVGCdURRBE9UQe6KotsHVZcPRYK30t5KjJ23qsVSdPLdS6D2ZOWRBjRqCTddHC6vDo6SVuIwCeUzNcwOabZdAdQQds2nOy46t1FfSyVfZtzcwpCXXZ4QHtsAxo56kkOJ1F7iyxw6p7PMdFPUwSMla1j3MawlokB1DddvCNDvmPJfObITdB2ziOH/oiWldSn5TllMUojAPjkaQD5ZRoeBuOK4hN1vU2IERsp6xjZ6dhGTOLuh11ynl+ybg8l0MOoqXGA+OCNrahjXyFpOW410uALm+XhtdYzzmE3fDeGFzup5cBXff3pq3RwNxuDwUvdbYbFJUvpaiKWPUtc02ds6xvY+S+h7QYcMTxGoqKdzYqhzyDE42Y+2mh4HTjp5hfLNPiHVfa1gtVz2/xHfFd+KTLGyvJz24ZzKee/wDD4yaKWnldFPG6ORvtMeLELEe5fXTMhqohFVx96xvsuvZ7Punh01C4GI4RLS5poXGenGpeBZzPvDh12WM+K49zw6cfPMur1X0ER/uVEdv7pH8FbEbix5FfRdm+6wzs3Bj88TH1Hcx0+HskaC3vMt3SW45Ra3mvWLHIsbZ8j7UyZr/QYi1g7yB37VvaZzHBfHs7cvbPm9vmble3be0fY3szEfakmrJsv7OZrfyK6tR2SxyKobFBQvqmSH5qem8cUg4EOGgHWy5P9qrqeOqwigpJRJHQUZp3PafC6QOu8j1P4LrwTWfbpwzWXb57D8ekiY2CuaZ6dos0g/ORj9knceR9LLtx5J4e/pZGzQbF7RYtPJw3aevoSvjL3335r3pKqooZxNTTOikAtdp3HI8CPIrtycEy7x6refDL3PLr9qPYw9tv1cmv764Vra7/AAXdxPF8PrqSlPyV7auOF7ZNu7zEg3aDe3E/gtWqGHiGTuHNLxHbjvmGW3PS904rccZLDitxxkscs6qXtsrvtvyVZG57srWlzuXJd3ZB4t/esmRuebMbe25OwW3T0LpDc+IDc3sxvUrowQsa1ugLSfC94IZ0a3dxVT3fZzY6ZrAHyDOSdz7P9VvNpnuDXzvETDa2bcjyby/BbzYvnCQ13eWJLrAubtw9lnqpZoaHuy5CfE5xOUnzdu700Ta6+7Qe9sZywMy20Lnak/yXidOq3JamEXMMYdLYDvXi2W32W/mVzpqhrSc5zPJubLSOed0usshOvDmoRZYbSyX5KIgKkX13CidFAtzREQEREHuqoi2wqKXS90BES/JBbW3UupdOiC3Tfb3J1UJ9yC7brpdnhmxBw/8ATy/wrmX5rqdnP94v4/3eX+Fc+X9FdOL+5HXq4KevbaraRLwqGDx/vD6w/HzXAr8NqKIhzgJIXGzZmatPl5HyK+g6rJkjmZgLFrhZzXC7XDkRxXjw5MsP4e3k4sc/5fJCwcOOq+zrDernJ/xHfFcuqwaKpfnoCIZb6wPd4Xfdcduh966te1zK6oa5pBEjrgi3FfU9LyY5y6fG9bxZYZTf7vG62KCmqKusggowTUSvDI7HidF129me4ijdi2K0OHSyND2wTlzpA07EtaDa/mvWKtw3s5FK/Cqs4hisjCxtUIiyKmadCWA6ucRpfYL03Of8XlmH/bw1e3OL0VFX0GEMaTQ0MBgjniH6wOtI7LsQXctdOOy5WUmNszHNkhf7ErDdrvXn5HVcbtOQ5tDz7uT+JczD8RqsPe51NJZrvbjeLseP2hx+K+X6j031W4vRjx/mYe6eX2MVbVwxOip6qeKN3tRslc1p9AVwu04+aw87eGX+ILfoK+kxKzYiKepP6iR3hcf2HH4HXzK0+1YLIqBj2kOAluCLEeILy8Us5ZL/AL0nHLOSSvnr8kvZLcRqlr9F73sLX29yW5+5L22Uvff3oKT7l70Z8br6jKPiFrnRbVALvdfTQfEJPJXfyZwxrhc/UaWfwR//AGcqGEOc47j2n9557Ofw+61RwAa9wLRFfxOzEMJ83byHyGi51dVxvIvI4hugzAAejRsqnjw2X1rI4wyNokLTpcWYDzy8T5laFRVXdmneS7h/otOWqcdGDKOZ3Wv5kptdb8vaWoe/RvhHlxWEbbnmVhf3L1gNni3PZC9RuGnERySNLnbaGw9Oa8KumMJF9jtzHVdmRpeC5ubjrGN/5LQxQtBjaLeEELdnTEvbl21RZMfkeCWtd5OFwsTa1wuboiIiiiIlkBERB7Ioi2yqKXvuiC35opumyAre2ym/VCgt7qapZL22QLc11Ozh/wDEX20/u8v8K5dr7Lp9nP8AeLv+3l/hXLl/RXTi/uR2t1Fbc0zcPxXgfRL2/kvrcHgp6zFX4tUBslDh1K2plYTqXfq2EebregK+RDXOcGtBcSbAAak8l0u29dL2eo6Ls5SF0Nc1sdViUrDYmTLaOO/JrdTwuV6fTZXHLTyer45ljL8vCtq562smq6l5fNM8vefMrX3WjRYzBUEMr8sEvCVg8DvvAez1Gnkuk9haRe1nC7XNNw4cweK+xjljlOnwc8csb9Ti9pbAUV9fA/8AiXFJvuu32nBAob/4b/41w915eX9de70/9uf78qdPMLqPxGqxlmGYbUd3kheImShvjIcQNSd7LljT+St77aEbBcbjLZb8Otxlst+Ha/QcckUk9PUl0cbwDexJbnLLtGmY3F7cAuNK0slfG4jMxxaSNjY2UZLIwtLHuaWG7SDseYWFrqtKUAS4G+qHXW90C9tteq9qR4Y9ziQNOPULwTdBuVmIy1L8znEu5nh5AcFpG7jcn1Ktxx1PNZRyPjLi0jxNLTcX0KDG9lFd1LFFRZx6HVY6IiOgKxxblfd+ltXH8ea16md0pu434DyWvmKXurtJiiuw9VOicPVZaEREEVREFRREHpdETqtsiuynRFBd/wCSiK9UE3VuB5qFEFJ4k6c1lHG+X6ON7+PgaT8Fs4PiEuE4rS4hAxr5KeQPDHgEPA3ab8CLhftnbXtngVZhdHhENVV0kWM0zZflFBExxiY42DHN0OpuDbXTzUt0sj8HW5hdYyirBNJGZGFjmOANjZwtcea+hl7ECeV7cF7RYLiGVxb3bqj5PLcG3svsPcV4u/s97Wg+DBJpR9qGSN4PqHKXWU1Vm5dx7MfHPEZ6WQSxD2iBZzPvDh8PNRWl7D9o6CVtTWvpcFa39fW1kcdh90Ek9LLu0uO9m8HkAo5osRxYN8NbJAY6ON/NrDqT5kZb8l48uK4+O49mHPL58tnDYW9l6ePFcQYDikjc1BSPGsd/10g4fsjidV8N2rmfUYo2aR7nyvp43Pe43LiQbkldmvmqairkmrpHyVEhzOe83LvO/EdNF8/2i/8APRf9tF8FeG/WnPPo7cvit7DcVqcPORmWSAm7oZNWnzHI+YWkNtdk22969stl3HhyxmU1Xfx+Smr6WgnoZC94jd3tPuYbuvqeui4T43sJa5jmkGxBCzpqqSmz93bxZb312N171mIvqou6MbGtu2xF7gNvb111PFbyymXd8sYYXCe2eGkiu++nmodNtFzdF+9/VLE6AX6LFetPL3MoeW5haxHNFeR0QXvovSZ3eyufbLmOy8yeA0QXTjuoSVCl0BEsl0Ut/oh4KKk3AQRVFEFURFAVzeHLYWve9tVEQEREBERUVERQZoom60yK9VEQbDaKqcyN7aeQxyWyvDfCdS3fYagjVejsMrmzCF1JM2QuDcpaRYk2F+W/FbFHj9dRwQwQmPu4b5QW73JOuuu5WyztRXtLnNZT5nbkxkncE8eJAKK5XyOo7qKUROLZQSwDUuA3OUa20OtraKCkqXGwp5ybXsI3bWvy5Alb1Bj1Xh9OIKcRBl3Ekh1zfzvpsNrbC916VHaSuqIhDI2DuhGYyxrC0FpHkfXRBonDqwVElO6BzZIml7w4hoa0cSTpbUar0w/vqSshqZaWd4hHeMGQ765DttmsvU47VurWVkoifLG1zW5gbDMbnQEX32OltLLd/wBrKyKR7qaGnjBADSWlzmnS5uTxyjTbQaKDhywTxC08MjbOLT3jCPENxrxWDSWew4jyaSFvV2L1FbSR0j2xsgjeXNZHmsLkm2pN9XO1313XPRBxzOu7U8yiXulrboOhh+LT0jBC9onpgb908+z5tP1T+HMLc7QwRVFZBLh8wqYnwMAIsCCNLH8PVcO62Ya6aFrWtILWtIaDsDmDr+8Bcrx6y92LrOS3H2ZeHhIx8brPFiRf0WK9JpTO8EgCzbAD3/Eleey6zfy5038k26qEpdBSbpdS3JNOqC2vspe2yl1b33QRL80T8UFG1+uqWt5IDp70Fr+oQYqqKoqdVb6BUgBoJIN+F9uqhQRERQEREBVEQRVEQEREBERBkiKKsrdFEVFCzawu2BJ8li3Vdn5MGtYIniO4BNzvpe91ZNpbpx3NWNl066JghY9ou4nV22bTkuY7dLNEuy6l0RZVUUS/NFXol1EQXonVNtk6oF0vzURATql7bJ0QL+iiJbmgIrdRFE6IlkAJxCKIKfNREUBERAREQEVUQVERAREQEREBFUQVRVFURFVEGbN12xK10bDGQDaxLWZuHEc1wRovZkxbsSOi1jdMWbb+ISNMLG3Ge93X9rbiuWd1k591hdLdrJoS6iLLSooiC7Je6iIKonVCgt7KIoiiu6IgIhUQVERAURVBEVUUBERARFUBERAREQEREBERAREQVERBUURVFRREBERBEREBOiIgIiIoNSFeO3oFBuFk3h1OyIwRFbooiit0BERAREQRERQEREBEVQRVEQEREBVREFRRVAUREBEVQRVEVBERQRFUVQREQEREVLIqiCKKqjZBLIQqiIgCy0ACiIMSiIiiIiAiIgIiICIigIiqAiIgIiICKqICIiAiqiAiKoIiq2cOoZ8RqmU1Kwue46ng0czyCDWRfQ9q8AdhcjJ6ZhdSFjWucB7LwLG/K+/vXzyQERFQRS6XQVFLpdBUUuiCooigIiIFyiIgIiKgoqogIiKAiIgIqiCKoiAiIgIiqCIiICIiAiKoIqiICi9YYXS3LeG9lg5pa4tO4NkEX0PZrtK/C3Mp6iON1ITZzmsAe3zuPa9V88iD9C7VdpBhwFJRiOSeRmZznDM1jTtpxJX5/I90kjpHWzONzlaAPcNApJI+V2aR5c6wFyb6AWH4KJICKIgiIiqipRERERFAREQEREBERBQiIgKIiAiIgIiICqIgIiICIiAiIgIiICIiAiIgqiIg9I5XxE5HWvoVgSSbk6lEQFERAREQEREH/9k="


# ══════════════════════════════════════════════════════════════════════════════
#  Worker para renderizado matemático — corre matplotlib en hilo separado
# ══════════════════════════════════════════════════════════════════════════════
class _MathRenderWorker(QThread):
    """
    Renderiza en background todos los bloques de un mensaje AI que requieren
    matplotlib, y emite la lista de resultados cuando termina.
    Así el hilo principal de Qt nunca se bloquea con matplotlib.
    """
    done = Signal(list)   # emite lista de (btype, extra, content_or_png_bytes)

    def __init__(self, blocks: list, parent=None):
        super().__init__(parent)
        self._blocks = blocks

    def run(self):
        results = []
        try:
            for btype, extra, content in self._blocks:
                if btype == 'display_math':
                    png = _render_math_img_v2(
                        _to_mathtext(content),
                        fontsize=12, fg='white', display=False)
                    results.append(('display_math', '', png or content))
                elif btype == 'line':
                    raw      = content
                    stripped = raw.strip()
                    bm       = _re_math.match(r'^(\s*[•\-\*]\s+)(.*)', raw)
                    math_src = bm.group(2) if bm else stripped
                    if stripped and _line_has_math(stripped):
                        try:
                            png = _render_math_img(
                                _to_mathtext(math_src),
                                fontsize=12, fg='white')
                        except Exception:
                            png = None
                        if png:
                            results.append(('math_line', 'bullet' if bm else '', png))
                            continue
                    # línea de texto normal
                    results.append(('line', '', raw))
                else:
                    results.append((btype, extra, content))
        except Exception:
            # Emitir lo que se haya procesado — never let the thread die silently
            pass
        self.done.emit(results)


# ══════════════════════════════════════════════════════════════════════════════
#  Worker Thread — Groq principal + OpenRouter respaldo (fallback automático)
# ══════════════════════════════════════════════════════════════════════════════
class _AIWorker(QThread):
    """
    QThread que llama al proveedor principal (Groq) y si falla por cualquier
    razón de red intenta automáticamente con el respaldo (OpenRouter).
    """
    response_ready = Signal(str)
    error_occurred = Signal(str)
    provider_used  = Signal(str)   # emite nombre del proveedor que respondió

    def __init__(self, api_key: str, api_key_backup: str,
                 messages: list, system: str, parent=None):
        super().__init__(parent)
        self._key1     = api_key
        self._key2     = api_key_backup
        self._messages = messages
        self._system   = system

    def run(self):
        """Intenta con key1; si falla por red/bloqueo prueba key2."""
        pairs = [(self._key1, _detect_provider(self._key1))]
        if self._key2 and self._key2 != self._key1:
            pairs.append((self._key2, _detect_provider(self._key2)))

        last_err = "Sin proveedor disponible."
        for key, provider in pairs:
            if provider == "unknown":
                continue
            try:
                ok = self._call(key, provider)
                if ok is True:
                    self.provider_used.emit(provider)
                    return
                last_err = ok  # string de error
                # Error de autenticación → no tiene sentido probar respaldo
                if any(x in ok for x in ("inválida", "401", "403 Forbidden")):
                    break
                # Error de red → intentar respaldo
            except Exception as ex:
                last_err = f"Error de conexión: {ex}"

        self.error_occurred.emit(last_err)

    def _call(self, key: str, provider: str):
        """Despacha al método correcto según el proveedor."""
        if provider == "groq":
            return self._call_groq(key)
        if provider == "openrouter":
            return self._call_openrouter(key)
        return f"Proveedor desconocido: {provider}"

    # ── Groq ──────────────────────────────────────────────────────────────────
    def _call_groq(self, key: str):
        """Retorna True si OK, o string de error si falla."""
        messages = [{"role": "system", "content": self._system}] + self._messages
        payload = json.dumps({
            "model"      : "llama-3.1-8b-instant",
            "messages"   : messages,
            "max_tokens" : 900,    # reducido para respuestas más rápidas
            "temperature": 0.7,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx()) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.response_ready.emit(data["choices"][0]["message"]["content"])
                return True
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:    msg = json.loads(body).get("error", {}).get("message", body[:150])
            except: msg = body[:150]
            if e.code == 401: return "API Key de Groq inválida."
            if e.code == 403: return "Groq bloqueado por la red (403)."
            if e.code == 429:
                import time as _time
                _time.sleep(0.5)   # pausa mínima antes del fallback
                return "Límite de Groq alcanzado — pasando al respaldo."
            return f"Error Groq {e.code}: {msg[:100]}"
        except Exception as ex:
            return f"Groq no disponible: {ex}"

    # ── OpenRouter ────────────────────────────────────────────────────────────
    def _call_openrouter(self, key: str):
        """
        Llama a OpenRouter con fallback en cascada:
        1) openrouter/free  (router automático de OpenRouter)
        2) Modelos gratuitos conocidos y activos (marzo 2026)
        Si uno retorna respuesta vacía o 404, pasa al siguiente.
        """
        # Modelos ordenados por velocidad: los más rápidos primero
        models = [
            "openrouter/free",
            "meta-llama/llama-4-scout:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "meta-llama/llama-4-maverick:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "google/gemma-3-12b-it:free",
            "deepseek/deepseek-r1:free",
        ]
        messages = [{"role": "system", "content": self._system}] + self._messages
        last_err  = "OpenRouter: ningún modelo respondió."

        for model in models:
            payload = json.dumps({
                "model"      : model,
                "messages"   : messages,
                "max_tokens" : 900,    # reducido para respuestas más rápidas
                "temperature": 0.7,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data    = payload,
                headers = {
                    "Content-Type" : "application/json",
                    "Authorization": f"Bearer {key}",
                    "HTTP-Referer" : "https://optimizer-app",
                    "X-Title"      : "Optimizador de Funciones",
                },
                method = "POST")
            try:
                with urllib.request.urlopen(req, timeout=12, context=_ssl_ctx()) as resp:
                    data    = json.loads(resp.read().decode("utf-8"))
                    content = data["choices"][0]["message"]["content"]
                    if content and content.strip():
                        self.response_ready.emit(content.strip())
                        return True
                    # Respuesta vacía → probar siguiente modelo
                    last_err = f"OpenRouter ({model}): respuesta vacía."
                    continue
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                try:    msg = json.loads(body).get("error", {}).get("message", body[:120])
                except: msg = body[:120]
                if e.code == 401:
                    return "API Key de OpenRouter inválida."
                if e.code == 402:
                    return "OpenRouter: límite diario alcanzado (50 req/día en plan gratuito)."
                if e.code == 429:
                    import time as _time
                    _time.sleep(0.5)   # pausa mínima y pasar al siguiente modelo
                    last_err = "OpenRouter: demasiadas solicitudes — probando otro modelo."
                    continue
                # 404 / 503 → modelo no disponible, probar el siguiente
                last_err = f"OpenRouter ({model}): {msg[:80]}"
                continue
            except Exception as ex:
                last_err = f"OpenRouter no disponible: {ex}"
                break   # error de red total — no seguir intentando

        return last_err


# ══════════════════════════════════════════════════════════════════════════════
#  VOZ — Text-to-Speech (SAPI5/pyttsx3, offline) y Speech-to-Text (Groq Whisper)
# ══════════════════════════════════════════════════════════════════════════════
_REC_SAMPLERATE = 16000  # Hz — suficiente para voz, tamaño de archivo razonable


def _strip_markdown_for_speech(text: str) -> str:
    """Limpia markdown/símbolos matemáticos para que el TTS no los lea en
    voz alta literalmente (p. ej. no diga 'asterisco asterisco')."""
    s = text
    s = re.sub(r'```.*?```', ' [bloque de código omitido] ', s, flags=re.DOTALL)
    s = re.sub(r'`([^`]+)`', r'\1', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'\1', s)
    s = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'\1', s)
    s = re.sub(r'^#{1,3}\s+', '', s, flags=re.MULTILINE)
    s = re.sub(r'^[•\-]\s+', '', s, flags=re.MULTILINE)
    s = re.sub(r'[_#>]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _find_spanish_voice(engine) -> Optional[str]:
    """Busca una voz en español entre las instaladas en Windows (SAPI5).
    Devuelve el id de voz o None si no hay ninguna instalada."""
    try:
        for v in engine.getProperty("voices"):
            langs = [str(l).lower() for l in (getattr(v, "languages", None) or [])]
            if ("es-" in v.id.lower() or "_es" in v.id.lower()
                    or "spanish" in v.name.lower() or "español" in v.name.lower()
                    or any(l.startswith("es") for l in langs)):
                return v.id
    except Exception:
        pass
    return None


class _TTSWorker(QThread):
    """Sintetiza y reproduce la respuesta en voz (SAPI5 vía pyttsx3), en un
    hilo aparte para no bloquear la UI mientras habla."""
    error_occurred = Signal(str)
    finished_speaking = Signal()

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._text = text

    def run(self):
        if not _TTS_OK:
            self.error_occurred.emit(
                "La síntesis de voz no está disponible en este equipo "
                "(falta el paquete pyttsx3).")
            return
        try:
            engine = pyttsx3.init()
            voice_id = _find_spanish_voice(engine)
            if not voice_id:
                self.error_occurred.emit(
                    "No encontré una voz en español instalada en Windows, así "
                    "que no puedo leer la respuesta en voz alta. Para agregarla: "
                    "Configuración → Hora e idioma → Idioma y región → "
                    "Agregar idioma → Español, y marca \"Instalar voz\". "
                    "Mientras tanto, sigo respondiendo por escrito.")
                return
            engine.setProperty("voice", voice_id)
            engine.setProperty("rate", 178)
            engine.say(self._text)
            engine.runAndWait()
            self.finished_speaking.emit()
        except Exception as ex:
            self.error_occurred.emit(f"No se pudo reproducir la voz: {ex}")


class _RecordWorker(QThread):
    """Graba audio del micrófono en un hilo aparte hasta que se llama a
    stop(); guarda un WAV temporal en 16 kHz mono (formato que espera Whisper)."""
    error_occurred = Signal(str)
    recording_ready = Signal(str)   # ruta del WAV grabado

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames: list = []
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        if not _REC_OK:
            self.error_occurred.emit(
                "La grabación de audio no está disponible en este equipo "
                "(falta el paquete sounddevice) o no se detectó un micrófono.")
            return
        self._frames = []
        try:
            def _callback(indata, frames, time_info, status):
                self._frames.append(indata.copy())

            with _sd.InputStream(samplerate=_REC_SAMPLERATE, channels=1,
                                  dtype="int16", callback=_callback):
                while not self._stop_flag:
                    self.msleep(50)
        except Exception as ex:
            self.error_occurred.emit(
                f"No se pudo acceder al micrófono: {ex}\n\n"
                f"Verifica que Windows tenga permiso de micrófono para esta "
                f"app (Configuración → Privacidad → Micrófono) y que haya un "
                f"micrófono conectado.")
            return

        if not self._frames:
            self.error_occurred.emit("No se grabó audio (grabación demasiado corta).")
            return

        audio = _np.concatenate(self._frames, axis=0)
        fd, path = _tempfile.mkstemp(suffix=".wav", prefix="optimizador_voz_")
        os.close(fd)
        with _wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(_REC_SAMPLERATE)
            wf.writeframes(audio.tobytes())
        self.recording_ready.emit(path)


def _transcribe_audio_groq(key: str, wav_path: str, language: str = "es") -> str:
    """Sube un WAV a la API de transcripción de Groq (Whisper) y devuelve el
    texto transcrito. Lanza RuntimeError con mensaje en español si falla —
    mismo patrón de manejo de errores que _AIWorker._call_groq."""
    boundary = _uuid.uuid4().hex

    def _field(name: str, value: str) -> bytes:
        return (f'--{boundary}\r\nContent-Disposition: form-data; '
                f'name="{name}"\r\n\r\n{value}\r\n').encode("utf-8")

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    body = b"".join([
        _field("model", "whisper-large-v3-turbo"),
        _field("language", language),
        _field("response_format", "json"),
        (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
         f'filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode("utf-8"),
        audio_bytes,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ])

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Authorization": f"Bearer {key}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20, context=_ssl_ctx()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return (data.get("text") or "").strip()
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="replace")
        try:    msg = json.loads(body_txt).get("error", {}).get("message", body_txt[:150])
        except Exception: msg = body_txt[:150]
        if e.code == 401:
            raise RuntimeError("API Key de Groq inválida para transcripción de voz.")
        raise RuntimeError(f"No se pudo transcribir el audio (Groq {e.code}): {msg[:120]}")
    except Exception as ex:
        raise RuntimeError(f"No se pudo transcribir el audio: {ex}")


class _STTWorker(QThread):
    """Sube el WAV grabado a Groq Whisper en un hilo aparte (no bloquea la UI)."""
    text_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, key: str, wav_path: str, parent=None):
        super().__init__(parent)
        self._key = key
        self._wav_path = wav_path

    def run(self):
        try:
            text = _transcribe_audio_groq(self._key, self._wav_path)
            if text:
                self.text_ready.emit(text)
            else:
                self.error_occurred.emit(
                    "No detecté voz en la grabación — intenta de nuevo hablando "
                    "un poco más cerca del micrófono.")
        except Exception as ex:
            self.error_occurred.emit(str(ex))
        finally:
            try: os.remove(self._wav_path)
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  QTextEdit con menú contextual en español
# ══════════════════════════════════════════════════════════════════════════════
class _ChatView(QtWidgets.QTextEdit):
    """QTextEdit con menú de clic derecho completamente en español."""

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: rgba(10,18,55,245); color: #c8ddf8;"
            "  border: 1px solid rgba(50,90,200,180); border-radius: 6px;"
            "  font-size: 12px; padding: 4px; }"
            "QMenu::item { padding: 6px 24px; border-radius: 4px; }"
            "QMenu::item:selected { background: rgba(30,60,160,200); color: #ffffff; }"
            "QMenu::item:disabled { color: #3a5878; }"
            "QMenu::separator { height: 1px; background: rgba(50,90,180,120);"
            "  margin: 4px 8px; }"
        )

        act_copy = menu.addAction("📋  Copiar")
        act_copy.setShortcut("Ctrl+C")
        act_copy.setEnabled(self.textCursor().hasSelection())

        menu.addSeparator()

        act_selall = menu.addAction("☰  Seleccionar todo")
        act_selall.setShortcut("Ctrl+A")

        menu.addSeparator()

        act_clear = menu.addAction("🗑  Limpiar chat")

        chosen = menu.exec(
            event.globalPos() if _QT == "PyQt6"
            else event.globalPos())

        if chosen == act_copy:
            self.copy()
        elif chosen == act_selall:
            self.selectAll()
        elif chosen == act_clear:
            # Señal hacia el panel padre
            parent = self.parent()
            while parent:
                if hasattr(parent, "_clear_chat"):
                    parent._clear_chat()
                    break
                parent = parent.parent() if hasattr(parent, "parent") else None



class _ChatInput(QtWidgets.QLineEdit):
    """QLineEdit con menú de clic derecho completamente en español."""

    _MENU_CSS = (
        "QMenu { background: rgba(10,18,55,245); color: #c8ddf8;"
        "  border: 1px solid rgba(50,90,200,180); border-radius: 6px;"
        "  font-size: 12px; padding: 4px; }"
        "QMenu::item { padding: 6px 24px; border-radius: 4px; }"
        "QMenu::item:selected { background: rgba(30,60,160,200); color: #ffffff; }"
        "QMenu::item:disabled { color: #3a5878; }"
        "QMenu::separator { height: 1px; background: rgba(50,90,180,120);"
        "  margin: 4px 8px; }"
    )

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(self._MENU_CSS)

        act_undo = menu.addAction("↩  Deshacer")
        act_undo.setShortcut("Ctrl+Z")
        act_undo.setEnabled(self.isUndoAvailable())

        act_redo = menu.addAction("↪  Rehacer")
        act_redo.setShortcut("Ctrl+Y")
        act_redo.setEnabled(self.isRedoAvailable())

        menu.addSeparator()

        act_cut = menu.addAction("✂  Cortar")
        act_cut.setShortcut("Ctrl+X")
        act_cut.setEnabled(self.hasSelectedText())

        act_copy = menu.addAction("📋  Copiar")
        act_copy.setShortcut("Ctrl+C")
        act_copy.setEnabled(self.hasSelectedText())

        act_paste = menu.addAction("📌  Pegar")
        act_paste.setShortcut("Ctrl+V")

        act_delete = menu.addAction("🗑  Eliminar")
        act_delete.setEnabled(self.hasSelectedText())

        menu.addSeparator()

        act_selall = menu.addAction("☰  Seleccionar todo")
        act_selall.setShortcut("Ctrl+A")
        act_selall.setEnabled(bool(self.text()))

        chosen = menu.exec(
            event.globalPos() if _QT == "PyQt6"
            else event.globalPos())

        if chosen == act_undo:
            self.undo()
        elif chosen == act_redo:
            self.redo()
        elif chosen == act_cut:
            self.cut()
        elif chosen == act_copy:
            self.copy()
        elif chosen == act_paste:
            self.paste()
        elif chosen == act_delete:
            self.del_()
        elif chosen == act_selall:
            self.selectAll()


class AIAssistantPanel(QtWidgets.QWidget):
    """
    Panel lateral/flotante con el chat del asistente de IA.
    Se integra directamente en la interfaz principal.
    """

    # ─── Estilos ──────────────────────────────────────────────────────────────
    _QSS = """
    QWidget#aiPanel {
        background: transparent;
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
        background: rgba(5,10,28,200);
        color: #c8ddf8;
        border: 1px solid rgba(40,70,160,180);
        border-radius: 6px;
        font-size: 11px;
        font-family: 'Segoe UI', Consolas;
        padding: 5px;
        selection-background-color: #1a3e80;
    }
    QLineEdit#chatInput {
        background: rgba(5,10,28,210);
        color: #dce8f8;
        border: 1px solid rgba(60,100,200,180);
        border-radius: 6px;
        font-size: 11px;
        padding: 5px 8px;
        min-height: 26px;
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
        min-width: 34px;
        min-height: 32px;
    }
    QPushButton#sendBtn:hover   { background: #2462d8; }
    QPushButton#sendBtn:pressed { background: #1340a0; }
    QPushButton#sendBtn:disabled { background: #1a2a50; color: #4a6090; }
    QPushButton#clearBtn {
        background: rgba(30,40,90,180);
        color: #7090c0;
        border: 1px solid rgba(60,80,150,140);
        border-radius: 5px;
        font-size: 10px;
        padding: 5px 8px;
    }
    QPushButton#clearBtn:hover { background: rgba(50,60,130,220); color: #a0b8e0; }
    QPushButton#exportPdfBtn {
        background: rgba(15,45,35,180);
        color: #50b870;
        border: 1px solid rgba(40,110,70,150);
        border-radius: 5px;
        font-size: 10px;
        padding: 5px 8px;
    }
    QPushButton#exportPdfBtn:hover { background: rgba(20,70,50,210); color: #80e0a0; }
    QPushButton#exportPdfBtn:disabled { background: rgba(10,20,15,120); color: #2a4535; }
    QPushButton#apiKeyBtn {
        background: rgba(20,40,100,200);
        color: #6090d0;
        border: 1px solid rgba(50,80,160,140);
        border-radius: 5px;
        font-size: 11px;
        padding: 3px 10px;
    }
    QPushButton#apiKeyBtn:hover { background: rgba(30,60,140,220); color: #90b8f0; }
    QPushButton#continueBtn {
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 rgba(10,60,130,220), stop:1 rgba(10,100,70,200));
        color: #a8deff;
        border: 1px solid rgba(50,120,200,150);
        border-radius: 6px;
        font-size: 11px;
        font-weight: bold;
        padding: 5px 10px;
    }
    QPushButton#continueBtn:hover {
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 rgba(20,90,190,240), stop:1 rgba(20,150,100,220));
        color: #ffffff;
    }
    QLineEdit#apiKeyInput {
        background: rgba(5,10,28,200);
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
        self.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground if _QT == "PyQt6"
            else Qt.WA_StyledBackground, True)

        self._api_key:        str  = ""
        self._api_key_backup: str  = ""
        self._messages: List[Dict] = []
        self._context:  str  = "Sin cálculo realizado aún."
        self._worker:   Optional[_AIWorker] = None
        self._busy:     bool = False
        self._pending_explain: str = ""
        self._msg_queue: list = []
        self._welcome_shown: bool = False
        self._continuation_blocks: list = []   # bloques pendientes de respuesta larga
        # ── Reintentos silenciosos ────────────────────────────────────────────
        self._retry_count: int = 0
        self._MAX_RETRIES: int = 3
        self._retry_msg:   str = ""

        # ── Modo de respuesta (voz): None = aún no se preguntó esta sesión ────
        self._response_mode: Optional[str] = None   # "escrita" | "voz" | "ambas"
        self._pending_first_message: str = ""
        self._tts_worker:    Optional[_TTSWorker]  = None
        self._record_worker: Optional[_RecordWorker] = None
        self._stt_worker:    Optional[_STTWorker]  = None
        self._recording: bool = False

        # ── Imagen de fondo ───────────────────────────────────────────────────
        import base64 as _b64
        _raw = _b64.b64decode(_AI_PANEL_BG_B64)
        self._bg_pixmap = QtGui.QPixmap()
        self._bg_pixmap.loadFromData(_raw)

        # ── Construir UI e inicializar keys ───────────────────────────────────
        self._build_ui()
        _cleanup_old_keys()
        primary, backup = _load_api_key_auto()
        if primary:
            self._apply_keys(primary, backup, source="auto")

    # ── Fondo personalizado ───────────────────────────────────────────────────
    def paintEvent(self, event):
        """Dibuja la imagen de fondo con overlay semi-transparente para legibilidad."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(
            QtGui.QPainter.RenderHint.SmoothPixmapTransform
            if _QT == "PyQt6" else QtGui.QPainter.SmoothPixmapTransform)
        # 1) Fondo sólido base
        painter.fillRect(self.rect(), QtGui.QColor(11, 19, 45, 255))
        # 2) Imagen escalada al tamaño actual del panel
        if hasattr(self, "_bg_pixmap") and not self._bg_pixmap.isNull():
            scaled = self._bg_pixmap.scaled(
                self.width(), self.height(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding
                if _QT == "PyQt6" else Qt.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
                if _QT == "PyQt6" else Qt.SmoothTransformation)
            x = (self.width()  - scaled.width())  // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        # 3) Overlay oscuro para legibilidad del texto
        painter.fillRect(self.rect(), QtGui.QColor(5, 10, 28, 185))
        painter.end()

    # ─── Construcción de UI ────────────────────────────────────────────────────
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(5)

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
        self.key_widget.setStyleSheet("background: rgba(5,10,28,160); border-radius:6px;")
        key_layout = QtWidgets.QVBoxLayout(self.key_widget)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.setSpacing(4)

        key_row = QtWidgets.QHBoxLayout()
        self.edt_key = QtWidgets.QLineEdit()
        self.edt_key.setObjectName("apiKeyInput")
        self.edt_key.setPlaceholderText("Key principal: gsk_... (Groq)")
        self.edt_key.setEchoMode(
            QtWidgets.QLineEdit.EchoMode.Password if _QT=="PyQt6"
            else QtWidgets.QLineEdit.Password)
        self.edt_key.setToolTip(
            "Key PRINCIPAL — se usa primero\n\n"
            "🥇 Groq (recomendado, 30 req/min, gratis)\n"
            "   console.groq.com/keys\n"
            "   La key empieza con 'gsk_'\n\n"
            "También puedes poner OpenRouter aquí (sk-or-...)\n"
            "   openrouter.ai → Sign In → Keys")
        self.edt_key.returnPressed.connect(self._set_api_key)

        self.btn_key = QtWidgets.QPushButton("✓ Conectar")
        self.btn_key.setObjectName("apiKeyBtn")
        self.btn_key.clicked.connect(self._set_api_key)
        key_row.addWidget(self.edt_key, 1)
        key_row.addWidget(self.btn_key)
        key_layout.addLayout(key_row)

        # Fila key de respaldo
        backup_row = QtWidgets.QHBoxLayout()
        lbl_bk = QtWidgets.QLabel("↩")
        lbl_bk.setStyleSheet("color:#4a7ab0; background:transparent; font-size:13px;")
        lbl_bk.setToolTip("Key de respaldo — se usa si la principal falla")
        self.edt_key_backup = QtWidgets.QLineEdit()
        self.edt_key_backup.setObjectName("apiKeyInput")
        self.edt_key_backup.setPlaceholderText("Key de respaldo: sk-or-... (OpenRouter)")
        self.edt_key_backup.setEchoMode(
            QtWidgets.QLineEdit.EchoMode.Password if _QT=="PyQt6"
            else QtWidgets.QLineEdit.Password)
        self.edt_key_backup.setToolTip(
            "Key de RESPALDO — si la principal falla, se usa ésta.\n"
            "Recomendado: pon Groq de principal y OpenRouter de respaldo.\n\n"
            "OpenRouter (gratis): openrouter.ai → Sign In → Keys")
        backup_row.addWidget(lbl_bk)
        backup_row.addWidget(self.edt_key_backup, 1)
        key_layout.addLayout(backup_row)

        opts_row = QtWidgets.QHBoxLayout()
        self.chk_save = QtWidgets.QCheckBox("Recordar keys")
        self.chk_save.setStyleSheet(
            "QCheckBox { color:#5a8ac0; font-size:10px; background:transparent; spacing:5px; }"
            "QCheckBox::indicator { width:13px; height:13px; border:1px solid #2a4a90;"
            " border-radius:3px; background:rgba(10,20,60,200); }"
            "QCheckBox::indicator:checked { background:#1a50c0; border-color:#4a9eff; }"
        )
        self.chk_save.setChecked(True)
        self.btn_forget = QtWidgets.QPushButton("🗑 Olvidar keys")
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
        self.chat_view = _ChatView()
        self.chat_view.setObjectName("chatHistory")
        self.chat_view.setReadOnly(True)
        self.chat_view.setMinimumHeight(120)
        self.chat_view.setMaximumHeight(220)
        self._append_system_msg(
            "👋 Hola, soy tu Asistente de IA para el Optimizador.\n\n"
            "Necesito al menos una API Key gratuita:\n"
            "  🥇 Groq → console.groq.com/keys  (gsk_...)\n"
            "  🥈 OpenRouter → openrouter.ai → Keys  (sk-or-...)\n\n"
            "Puedo ayudarte a:\n"
            "  • Recomendar métodos · Explicar resultados\n"
            "  • Comparar métodos · Diagnosticar convergencia\n\n"
            "Escríbeme o usa las sugerencias de abajo 👇")
        root.addWidget(self.chat_view, 1)

        # ── Botón "Continuar" para respuestas largas ──────────────────────────
        self.btn_continue = QtWidgets.QPushButton("▼  La respuesta continúa — clic para ver más")
        self.btn_continue.setObjectName("continueBtn")
        self.btn_continue.setCursor(QtGui.QCursor(
            Qt.CursorShape.PointingHandCursor if _QT=="PyQt6" else Qt.PointingHandCursor))
        self.btn_continue.clicked.connect(self._on_continue_click)
        self.btn_continue.setVisible(False)
        root.addWidget(self.btn_continue)

        # ── Botón Explicar resultado (se activa tras cada cálculo) ────────────
        self.btn_explain = QtWidgets.QPushButton("📊 Explicar último resultado")
        self.btn_explain.setObjectName("explainBtn")
        self.btn_explain.setStyleSheet(
            "QPushButton#explainBtn {"
            "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "    stop:0 rgba(20,70,160,210), stop:1 rgba(10,120,80,200));"
            "  color: #c8eeff; border: 1px solid rgba(50,140,200,160);"
            "  border-radius: 6px; font-size: 10px; font-weight: bold;"
            "  padding: 4px 10px; text-align: left; }"
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
        sug_lbl = QtWidgets.QLabel("Sugerencias rápidas:")
        sug_lbl.setStyleSheet("color:#4a6a9a; font-size:10px; background:transparent; margin-top:2px;")
        root.addWidget(sug_lbl)

        sug_grid = QtWidgets.QGridLayout()
        sug_grid.setSpacing(3)
        sug_grid.setContentsMargins(0, 0, 0, 0)
        # ── Botones de sugerencias fijas ─────────────────────────────────────
        static_suggestions = [
            ("¿Qué método usar?",     "¿Qué método me recomiendas para la función actual y por qué?"),
            ("Método para Rosenbrock","¿Qué método es mejor para optimizar la función de Rosenbrock y por qué?"),
            ("¿Cuándo usar PSO o GA?","¿Cuándo conviene usar metaheurísticas (PSO, GA, SA) en vez de métodos clásicos?"),
            ("Convergencia lenta",    "El método converge muy lento, ¿qué puedo cambiar para mejorar?"),
            ("Ayuda con la función",  "¿Cómo escribo correctamente una función en la sintaxis del optimizador?"),
        ]

        # Botón dinámico "Comparar métodos" — usa el contexto actual
        self._btn_compare = QtWidgets.QPushButton("Comparar métodos utilizados")
        self._btn_compare.setStyleSheet(
            "QPushButton { background: rgba(20,35,90,200); color:#7aaad8;"
            " border:1px solid rgba(40,70,150,160); border-radius:5px;"
            " font-size:9px; padding:3px 5px; text-align:left; }"
            "QPushButton:hover { background: rgba(30,55,130,220); color:#a0c8f0; }")
        self._btn_compare.setCursor(QtGui.QCursor(
            Qt.CursorShape.PointingHandCursor if _QT=="PyQt6" else Qt.PointingHandCursor))
        self._btn_compare.clicked.connect(self._on_compare_click)
        sug_grid.addWidget(self._btn_compare, 0, 0)

        for i, (label, prompt) in enumerate(static_suggestions):
            btn = QtWidgets.QPushButton(label)
            btn.setStyleSheet(
                "QPushButton { background: rgba(20,35,90,200); color:#7aaad8;"
                " border:1px solid rgba(40,70,150,160); border-radius:5px;"
                " font-size:9px; padding:3px 5px; text-align:left; }"
                "QPushButton:hover { background: rgba(30,55,130,220); color:#a0c8f0; }")
            btn.setCursor(QtGui.QCursor(
                Qt.CursorShape.PointingHandCursor if _QT=="PyQt6" else Qt.PointingHandCursor))
            btn.clicked.connect(lambda _, p=prompt: self._send_message(p))
            # +1 en fila porque el botón comparar ocupa (0,0), ajustar grid
            row = (i + 1) // 2
            col = (i + 1) % 2
            sug_grid.addWidget(btn, row, col)
        root.addLayout(sug_grid)

        # ── Modo de respuesta: Escrita / Voz / Ambas ───────────────────────────
        # Se pregunta una vez al primer mensaje de la sesión (ver _send_message /
        # _ask_response_mode); esta misma fila queda visible después como
        # selector para cambiar de modo sin volver a interrumpir con la pregunta.
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(4)
        mode_lbl = QtWidgets.QLabel("🔊 Respuesta:")
        mode_lbl.setStyleSheet("color:#7090c0; font-size:10px; background:transparent;")
        self.btn_mode_escrita = QtWidgets.QPushButton("✍ Escrita")
        self.btn_mode_voz     = QtWidgets.QPushButton("🔊 Voz")
        self.btn_mode_ambas   = QtWidgets.QPushButton("✍🔊 Ambas")
        for b in (self.btn_mode_escrita, self.btn_mode_voz, self.btn_mode_ambas):
            b.setObjectName("clearBtn")
            b.setCheckable(True)
            b.setAutoExclusive(True)
        self.btn_mode_escrita.clicked.connect(lambda: self._set_response_mode("escrita"))
        self.btn_mode_voz.clicked.connect(lambda: self._set_response_mode("voz"))
        self.btn_mode_ambas.clicked.connect(lambda: self._set_response_mode("ambas"))
        mode_row.addWidget(mode_lbl)
        mode_row.addWidget(self.btn_mode_escrita)
        mode_row.addWidget(self.btn_mode_voz)
        mode_row.addWidget(self.btn_mode_ambas)
        root.addLayout(mode_row)

        # ── Input + micrófono + botón enviar ───────────────────────────────────
        inp_row = QtWidgets.QHBoxLayout()
        inp_row.setSpacing(5)
        self.edt_msg = _ChatInput()
        self.edt_msg.setObjectName("chatInput")
        self.edt_msg.setPlaceholderText("Escribe tu pregunta sobre el optimizador…")
        self.edt_msg.returnPressed.connect(self._on_send)

        self.btn_mic = QtWidgets.QPushButton("🎤")
        self.btn_mic.setObjectName("clearBtn")
        self.btn_mic.setToolTip("Grabar pregunta por voz (haz clic para empezar/detener)")
        self.btn_mic.setEnabled(_REC_OK)
        if not _REC_OK:
            self.btn_mic.setToolTip("Grabación de voz no disponible (falta sounddevice)")
        self.btn_mic.clicked.connect(self._toggle_recording)

        self.btn_send = QtWidgets.QPushButton("➤")
        self.btn_send.setObjectName("sendBtn")
        self.btn_send.clicked.connect(self._on_send)
        self.btn_send.setEnabled(False)

        inp_row.addWidget(self.edt_msg, 1)
        inp_row.addWidget(self.btn_mic)
        inp_row.addWidget(self.btn_send)
        root.addLayout(inp_row)

        # ── Pie ───────────────────────────────────────────────────────────────
        foot = QtWidgets.QVBoxLayout()
        foot.setSpacing(3)

        # Fila 1: botones que ocupan todo el ancho por igual
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_clear_chat = QtWidgets.QPushButton("🗑  Limpiar chat")
        self.btn_clear_chat.setObjectName("clearBtn")
        self.btn_clear_chat.clicked.connect(self._clear_chat)
        self.btn_clear_chat.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding if _QT == "PyQt6"
            else QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed if _QT == "PyQt6"
            else QtWidgets.QSizePolicy.Fixed)
        self.btn_export_pdf = QtWidgets.QPushButton("📄  Exportar PDF")
        self.btn_export_pdf.setObjectName("exportPdfBtn")
        self.btn_export_pdf.setToolTip("Exportar toda la conversación a un archivo PDF")
        self.btn_export_pdf.clicked.connect(self._export_chat_pdf)
        self.btn_export_pdf.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding if _QT == "PyQt6"
            else QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed if _QT == "PyQt6"
            else QtWidgets.QSizePolicy.Fixed)
        btn_row.addWidget(self.btn_clear_chat)
        btn_row.addWidget(self.btn_export_pdf)
        foot.addLayout(btn_row)

        # Fila 2: marca centrada debajo de los botones
        powered = QtWidgets.QLabel("Groq · OpenRouter — 100% Gratis")
        powered.setAlignment(
            Qt.AlignmentFlag.AlignCenter if _QT == "PyQt6" else Qt.AlignCenter)
        powered.setStyleSheet("color:#2a5040; font-size:9px; background:transparent;")
        foot.addWidget(powered)

        root.addLayout(foot)

    # ─── API Key ─────────────────────────────────────────────────────────────────
    def _set_api_key(self):
        key   = self.edt_key.text().strip()
        key_b = self.edt_key_backup.text().strip()
        if not key:
            self._append_system_msg("⚠️ Pega tu API Key principal primero.")
            return
        if _detect_provider(key) == "unknown":
            self._append_system_msg(
                "⚠️ Key principal no reconocida.\n"
                "  Groq       →  gsk_...   (console.groq.com/keys)\n"
                "  OpenRouter →  sk-or-... (openrouter.ai → Keys)")
            self.edt_key.clear()
            return
        if key_b and _detect_provider(key_b) == "unknown":
            self._append_system_msg("⚠️ Key de respaldo no válida — se ignorará.")
            key_b = ""
        if self.chk_save.isChecked():
            _save_api_key(key, is_backup=False)
            if key_b:
                _save_api_key(key_b, is_backup=True)
        self.edt_key.clear()
        self.edt_key_backup.clear()
        self._apply_keys(key, key_b, source="manual")

    def _apply_keys(self, key: str, key_backup: str = "", source: str = "manual"):
        self._api_key        = key
        self._api_key_backup = key_backup
        self.btn_send.setEnabled(True)
        if self._pending_explain:
            self.btn_explain.setEnabled(True)

        prov_names = {"groq": "Groq 🟢", "openrouter": "OpenRouter 🟢"}
        p1 = prov_names.get(_detect_provider(key), "IA 🟢")
        p2 = prov_names.get(_detect_provider(key_backup), "") if key_backup else ""
        prov_txt = p1 + (f" + {p2} respaldo" if p2 else "")

        self.lbl_status.setText("● Conectado")
        self.lbl_status.setStyleSheet("color: #3cff8a; font-size:11px; background:transparent;")
        self.key_widget.setVisible(False)
        self.btn_forget.setVisible(True)

        self.chat_view.clear()
        self._messages.clear()

        backup_note = (
            f" Con {p2} como respaldo automático — si {p1} falla, cambio sin interrupciones."
            if p2 else
            "\n💡 Agrega una key de OpenRouter como respaldo para máxima disponibilidad."
        )
        self._append_ai_msg(
            f"¡Hola! Soy tu asistente de optimización, conectado con {prov_txt}.{backup_note}\n\n"
            "Puedo ayudarte a:\n"
            "  • Recomendar el mejor método para tu función\n"
            "  • Explicar resultados, iteraciones y convergencia\n"
            "  • Comparar métodos (Armijo, Wolfe, Newton, PSO, GA…)\n"
            "  • Diagnosticar por qué un método falla o converge lento\n\n"
            "Escríbeme o usa las sugerencias de abajo. "
            "Tras cada cálculo aparece el botón 📊 para pedir explicación."
        )
        self._welcome_shown = True   # El chat solo tiene el saludo — se borrará al primer mensaje

    def _forget_key(self):
        _delete_saved_key()
        self._api_key        = ""
        self._api_key_backup = ""
        self.btn_send.setEnabled(False)
        self.btn_explain.setEnabled(False)
        self.lbl_status.setText("● Sin conexión")
        self.lbl_status.setStyleSheet("color: #3cefff; font-size:11px; background:transparent;")
        self.key_widget.setVisible(True)
        self.btn_forget.setVisible(False)
        self.chat_view.clear()
        self._messages.clear()
        self._append_system_msg(
            "🗑 Keys eliminadas.\n\n"
            "Pega tus keys arriba para reconectarte:\n"
            "  Principal →  Groq: gsk_...        (console.groq.com/keys)\n"
            "  Respaldo  →  OpenRouter: sk-or-... (openrouter.ai → Keys)")

    # ─── Botón Explicar resultado ────────────────────────────────────────────────
    def _on_explain_click(self):
        if not self._pending_explain:
            return
        prompt = self._pending_explain
        self._pending_explain = ""
        self.btn_explain.setEnabled(False)
        self.btn_explain.setText("📊 Explicar último resultado")
        self._send_message(prompt)

    # ─── Botón dinámico Comparar métodos ────────────────────────────────────────
    def _on_compare_click(self):
        """
        Genera un prompt de comparación basado en los métodos y función
        que hay en la sesión actual del optimizador.
        """
        import re as _re_ctx
        ctx = self._context.strip()

        metodos   = _re_ctx.findall(r'Método activo:\s*(.+)', ctx)
        funciones = _re_ctx.findall(r'Función:\s*f\(.*?\)\s*=\s*(.+)', ctx)
        resultado = _re_ctx.search(r'Resultado:\s*(.+)', ctx)
        iters     = _re_ctx.search(r'Iteraciones ejecutadas:\s*(\d+)', ctx)

        metodo_actual = metodos[0].strip()   if metodos   else None
        fx_actual     = funciones[0].strip() if funciones else None
        res_str       = resultado.group(1).strip() if resultado else None
        n_iter        = iters.group(1)               if iters     else None

        session_methods = getattr(self, '_session_methods', [])
        session_results = getattr(self, '_session_results', {})

        if session_methods and len(session_methods) >= 2:
            lista = ', '.join(f'**{m}**' for m in session_methods)
            resultados_detalle = ""
            for m in session_methods:
                r = session_results.get(m)
                if r:
                    resultados_detalle += f"\n  • {m}: {r}"
            detalle_bloque = (
                f"\n\n**Resultados obtenidos en esta sesión:**{resultados_detalle}"
                if resultados_detalle else ""
            )
            prompt = (
                f"En esta sesión apliqué ÚNICAMENTE los siguientes métodos "
                f"a f(x) = {fx_actual or 'la función actual'}:\n"
                f"{lista}{detalle_bloque}\n\n"
                f"IMPORTANTE: Compara EXCLUSIVAMENTE estos {len(session_methods)} métodos "
                f"entre sí. NO menciones ni introduzcas otros métodos durante la comparación.\n\n"
                f"Compara SOLO {lista} en:\n"
                f"1. Velocidad de convergencia y número de iteraciones de cada uno\n"
                f"2. Requisitos matemáticos de cada uno (gradiente, Hessiano, sin derivadas)\n"
                f"3. Cuál obtuvo el mejor resultado numérico y por qué\n"
                f"4. En qué tipo de funciones cada uno de estos métodos es superior al otro\n"
                f"5. Conclusión: cuál recomendarías para esta función específica y por qué\n\n"
                f"Solo al final, en un párrafo breve separado titulado "
                f"'¿Explorar otros métodos?', puedes mencionar si existe algún otro "
                f"método del optimizador que podría dar mejores resultados, y por qué."
            )
        elif metodo_actual and fx_actual:
            extra = ""
            if res_str: extra += f"\nResultado obtenido: {res_str}"
            if n_iter:  extra += f"\nIteraciones: {n_iter}"
            prompt = (
                f"Acabo de usar el método **{metodo_actual}** para optimizar "
                f"f(x) = {fx_actual}.{extra}\n\n"
                f"Primero explícame cómo se desempeñó **{metodo_actual}** con esta función.\n\n"
                f"Luego, dime qué otros métodos del optimizador serían comparables "
                f"o podrían dar mejores resultados para esta función específica, "
                f"explicando en cada caso por qué serían una mejora o alternativa."
            )
        elif metodo_actual:
            prompt = (
                f"Estoy usando el método **{metodo_actual}** en el optimizador. "
                f"¿Con qué otros métodos lo compararías? Explica ventajas, "
                f"desventajas y en qué funciones cada uno funciona mejor."
            )
        else:
            prompt = (
                "Aún no he ejecutado ningún cálculo. ¿Puedes hacer un resumen "
                "comparativo de todos los métodos disponibles en el optimizador "
                "(Búsqueda Local, Fibonacci, Armijo, Wolfe, Goldstein, Newton-Raphson, "
                "PSO, GA, SA), indicando cuándo usar cada uno y sus ventajas?"
            )
        self._send_message(prompt)

    def update_session_methods(self, methods: list):
        """
        Llamado por interfaz_qt.py para registrar todos los métodos
        ejecutados en la sesión actual. Permite que el botón 'Comparar'
        construya un prompt con los métodos reales usados.
        Ejemplo: panel_ai.update_session_methods(["Armijo", "Wolfe", "Newton"])
        """
        self._session_methods = list(dict.fromkeys(methods))

    # ─── Enviar mensaje ──────────────────────────────────────────────────────────
    def _on_send(self):
        text = self.edt_msg.text().strip()
        if not text:
            return
        self.edt_msg.clear()
        self._send_message(text)

    # ─── Modo de respuesta (Escrita / Voz / Ambas) ─────────────────────────────
    def _ask_response_mode(self):
        self._append_system_msg(
            "Antes de responder — ¿prefieres que te conteste por escrito, "
            "en voz, o de ambas formas? Elige una opción junto al cuadro de "
            "texto, abajo (puedes cambiarla cuando quieras).")

    def _set_response_mode(self, mode: str):
        self._response_mode = mode
        if self._pending_first_message:
            pending = self._pending_first_message
            self._pending_first_message = ""
            self._send_message(pending)

    def _speak_response(self, text: str) -> None:
        """Lee la respuesta en voz alta (SAPI5/pyttsx3) si el modo actual
        incluye voz. No superpone dos lecturas si ya hay una en curso."""
        if self._tts_worker and self._tts_worker.isRunning():
            return
        plain = _strip_markdown_for_speech(text)
        if not plain:
            return
        self._tts_worker = _TTSWorker(plain, parent=self)
        self._tts_worker.error_occurred.connect(self._on_tts_error)
        self._tts_worker.start()

    def _on_tts_error(self, msg: str) -> None:
        self._append_system_msg(f"🔇 {msg}")

    # ─── Entrada por voz (micrófono → Groq Whisper) ────────────────────────────
    def _toggle_recording(self):
        if self._recording:
            if self._record_worker:
                self._record_worker.stop()
            self.btn_mic.setText("🎤")
            self.btn_mic.setEnabled(False)  # se reactiva al terminar la transcripción
            self._recording = False
            return
        if not self._api_key:
            self._append_system_msg(
                "⚠️ Conecta tu API Key primero (la transcripción de voz usa "
                "el mismo proveedor Groq que el chat).")
            return
        self._recording = True
        self.btn_mic.setText("🔴")
        self._record_worker = _RecordWorker(parent=self)
        self._record_worker.recording_ready.connect(self._on_recording_ready)
        self._record_worker.error_occurred.connect(self._on_recording_error)
        self._record_worker.start()

    def _on_recording_ready(self, wav_path: str):
        self.btn_mic.setText("🎤")
        self.btn_mic.setEnabled(True)
        self.edt_msg.setPlaceholderText("Transcribiendo tu voz…")
        self._stt_worker = _STTWorker(self._api_key, wav_path, parent=self)
        self._stt_worker.text_ready.connect(self._on_transcription_ready)
        self._stt_worker.error_occurred.connect(self._on_transcription_error)
        self._stt_worker.start()

    def _on_recording_error(self, msg: str):
        self.btn_mic.setText("🎤")
        self.btn_mic.setEnabled(_REC_OK)
        self._recording = False
        self._append_system_msg(f"🎤 {msg}")

    def _on_transcription_ready(self, text: str):
        self.edt_msg.setPlaceholderText("Escribe tu pregunta sobre el optimizador…")
        self.edt_msg.setText(text)
        self.edt_msg.setFocus()

    def _on_transcription_error(self, msg: str):
        self.edt_msg.setPlaceholderText("Escribe tu pregunta sobre el optimizador…")
        self._append_system_msg(f"🎤 {msg}")

    def _send_message(self, text: str, is_auto: bool = False):
        if not self._api_key:
            self._append_system_msg("⚠️ Conecta tu API Key primero.")
            return
        if self._response_mode is None and not is_auto:
            self._pending_first_message = text
            self._ask_response_mode()
            return
        if self._busy:
            # Encolar el mensaje — se procesará cuando llegue la respuesta
            self._msg_queue.append((text, is_auto))
            if not is_auto:
                self._append_system_msg(f"⏳ En cola: “{text[:60]}…”" if len(text) > 60 else f"⏳ En cola: “{text}”")
            return
        # Borrar el mensaje de bienvenida al primer mensaje real del usuario
        if self._welcome_shown:
            self.chat_view.clear()
            self._welcome_shown = False
        if not is_auto:
            self._append_user_msg(text)
        self._messages.append({"role": "user", "content": text})
        self._retry_msg = text
        self._busy = True
        self._append_system_msg("⏳ Analizando…")
        self.btn_send.setEnabled(False)
        self.btn_send.setText("…")
        system = _SYSTEM_PROMPT.replace("{context}", self._context)
        self._worker = _AIWorker(
            self._api_key, self._api_key_backup,
            list(self._messages), system)
        self._worker.response_ready.connect(self._on_response)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.provider_used.connect(self._on_provider_used)
        self._worker.start()

    def _on_response(self, text: str):
        self._remove_last_system_msg()
        self._retry_count = 0
        self._messages.append({"role": "assistant", "content": text})
        if len(self._messages) > 20:
            self._messages = self._messages[-20:]
        self._busy = False
        self.btn_send.setEnabled(True)
        self.btn_send.setText("➤")

        if self._response_mode in ("voz", "ambas"):
            self._speak_response(text)

        # Procesar siguiente mensaje en cola con pequeño delay (evita rate limit)
        if self._msg_queue:
            next_text, next_auto = self._msg_queue.pop(0)
            QTimer.singleShot(800, lambda: self._send_message(next_text, next_auto))

        # Encabezado del asistente (inmediato, sin matplotlib)
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
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

        # Renderizar bloques en hilo separado (no bloquea Qt)
        blocks = _parse_response_blocks(text)
        _BLOCK_LIMIT = 35
        if len(blocks) > _BLOCK_LIMIT:
            self._continuation_blocks = blocks[_BLOCK_LIMIT:]
            blocks_to_render = blocks[:_BLOCK_LIMIT]
            self.btn_continue.setVisible(True)
        else:
            self._continuation_blocks = []
            blocks_to_render = blocks
            self.btn_continue.setVisible(False)
        if _MPL_OK:
            self._math_worker = _MathRenderWorker(blocks_to_render, parent=self)
            self._math_worker.done.connect(self._on_render_done)
            self._math_worker.start()
        else:
            self._on_render_done(
                [('line', '', b[2]) if b[0] in ('line','display_math')
                 else b for b in blocks_to_render])

    def _on_render_done(self, results: list):
        """
        Slot llamado cuando _MathRenderWorker termina.
        Inserta los bloques ya renderizados en el chat desde el hilo principal.
        """
        cursor = self.chat_view.textCursor()
        cursor.movePosition(
            QtGui.QTextCursor.MoveOperation.End if _QT=="PyQt6"
            else QtGui.QTextCursor.End)
        doc   = self.chat_view.document()
        blank = QtGui.QTextCharFormat()
        idx   = getattr(self, '_math_img_idx', 0)

        def _insert_png(png_bytes: bytes, centered: bool = False):
            nonlocal idx
            qimg = QtGui.QImage()
            qimg.loadFromData(png_bytes)
            key   = f"aimath_{idx}"; idx += 1
            rtype = (QtGui.QTextDocument.ResourceType.ImageResource
                     if _QT == "PyQt6" else QtGui.QTextDocument.ImageResource)
            doc.addResource(rtype, QtCore.QUrl(key), qimg)
            ifmt = QtGui.QTextImageFormat()
            ifmt.setName(key)
            if centered:
                bfmt = QtGui.QTextBlockFormat()
                bfmt.setAlignment(
                    QtCore.Qt.AlignmentFlag.AlignCenter if _QT == "PyQt6"
                    else QtCore.Qt.AlignCenter)
                cursor.insertBlock(bfmt)
                cursor.insertImage(ifmt)
                cursor.insertBlock(QtGui.QTextBlockFormat())
            else:
                cursor.insertImage(ifmt)

        for btype, extra, content in results:

            # ── Bloque de código ──────────────────────────────────────────────
            if btype == 'code':
                cursor.insertHtml(
                    '<span style="font-size:9pt;color:#7eb8ff;'
                    'font-family:Consolas,monospace;">◆ código</span>')
                cursor.insertText('\n', blank)
                for cline in content.split('\n'):
                    cursor.insertHtml(
                        f'<span style="font-family:Consolas,\'Courier New\','
                        f'monospace;font-size:10.5pt;color:#ffd080;'
                        f'background-color:rgba(20,30,70,220);">'
                        f'&nbsp;&nbsp;{cline.replace(" ","&nbsp;")}</span>')
                    cursor.insertText('\n', blank)
                cursor.insertText('\n', blank)
                continue

            # ── Tabla markdown ─────────────────────────────────────────────────
            if btype == 'table':
                cursor.insertHtml(_table_md_to_html(content))
                cursor.insertText('\n', blank)
                continue

            # ── Display math (PNG ya renderizado) ─────────────────────────────
            if btype == 'display_math':
                if isinstance(content, bytes):
                    _insert_png(content, centered=True)
                    cursor.insertText('\n', blank)
                else:
                    # Fallback texto si matplotlib falló
                    cursor.insertHtml(
                        f'<span style="font-family:Consolas;font-size:11pt;'
                        f'color:#ffd080;">{content}</span>')
                    cursor.insertText('\n', blank)
                continue

            # ── Math inline / bullet con imagen ───────────────────────────────
            if btype == 'math_line':
                if extra == 'bullet':
                    fmt = QtGui.QTextCharFormat()
                    fmt.setForeground(QtGui.QColor("#c8f0d8"))
                    fmt.setFontPointSize(11)
                    cursor.insertText('  •  ', fmt)
                _insert_png(content, centered=False)
                cursor.insertText('\n', blank)
                continue

            # ── Línea de texto normal ──────────────────────────────────────────
            html = (
                '<span style="font-family:\'Segoe UI\',Consolas;font-size:11pt;">'
                + _md_to_html(content)
                + '</span>'
            )
            cursor.insertHtml(html)
            cursor.insertText('\n', blank)

        self._math_img_idx = idx
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

    def _on_continue_click(self):
        """Muestra la segunda parte de una respuesta larga."""
        self.btn_continue.setVisible(False)
        if not self._continuation_blocks:
            return
        blocks = self._continuation_blocks
        self._continuation_blocks = []
        if _MPL_OK:
            self._cont_worker = _MathRenderWorker(blocks, parent=self)
            self._cont_worker.done.connect(self._on_render_done)
            self._cont_worker.start()
        else:
            self._on_render_done(
                [('line', '', b[2]) if b[0] in ('line', 'display_math')
                 else b for b in blocks])

    def _on_provider_used(self, provider: str):
        """Actualiza el tooltip del status si respondió el proveedor de respaldo."""
        names = {"groq": "Groq", "openrouter": "OpenRouter"}
        self.lbl_status.setToolTip(f"Último en responder: {names.get(provider, provider)}")

    def _on_error(self, error: str):
        """
        Reintentos silenciosos — el usuario nunca ve mensajes de error técnicos.
        Solo muestra aviso si la API Key es inválida (error del usuario, no del servicio).
        """
        # Error de autenticación: sí avisar, es algo que el usuario debe corregir
        if any(x in error for x in ("inválida", "401", "403 Forbidden")):
            self._remove_last_system_msg()
            if self._messages and self._messages[-1]["role"] == "user":
                self._messages.pop()
            self._append_system_msg("⚠️ API Key inválida. Revisa tu clave.")
            self._busy = False
            self.btn_send.setEnabled(bool(self._api_key))
            self.btn_send.setText("➤")
            self._retry_count = 0
            self._msg_queue.clear()
            return

        # Cualquier otro error: reintentar en silencio con delay corto
        if self._retry_count < self._MAX_RETRIES:
            self._retry_count += 1
            delay_ms = self._retry_count * 600   # 600ms, 1.2s, 1.8s — mucho más rápido
            self._remove_last_system_msg()
            self._append_system_msg("⏳ Analizando…")
            if self._worker:
                try:
                    self._worker.response_ready.disconnect()
                    self._worker.error_occurred.disconnect()
                    self._worker.provider_used.disconnect()
                except Exception:
                    pass
            system = _SYSTEM_PROMPT.replace("{context}", self._context)
            msgs_snapshot = list(self._messages)
            k1, k2 = self._api_key, self._api_key_backup

            def _do_retry():
                self._worker = _AIWorker(k1, k2, msgs_snapshot, system)
                self._worker.response_ready.connect(self._on_response)
                self._worker.error_occurred.connect(self._on_error)
                self._worker.provider_used.connect(self._on_provider_used)
                self._worker.start()

            QTimer.singleShot(delay_ms, _do_retry)
            return

        # Se agotaron reintentos — mostrar siempre un mensaje al usuario
        self._remove_last_system_msg()
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages.pop()
        self._busy = False
        self._retry_count = 0
        self.btn_send.setEnabled(bool(self._api_key))
        self.btn_send.setText("➤")
        self._msg_queue.clear()
        self._append_ai_msg(
            "⚠️ No pude conectarme a ningún proveedor de IA en este momento.\n\n"
            "Posibles causas:\n"
            "  • Sin conexión a internet\n"
            "  • Límite de uso alcanzado en ambos proveedores\n"
            "  • Los servidores están sobrecargados\n\n"
            "💡 Puedes intentarlo de nuevo en unos segundos — "
            "simplemente vuelve a enviar tu mensaje."
        )

    # _process_queue kept for QTimer compatibility (no-op now)
    def _process_queue(self):
        pass

    # ─── Actualizar contexto desde la app principal ─────────────────────────────
    def update_context(self, metodo: str = "", fx: str = "",
                       result_summary: str = "", n_iters: int = 0,
                       vars_: list = None, x0: list = None,
                       lo: float = 0.0, hi: float = 5.0):
        """
        Llamado por la ventana principal tras cada cálculo.
        Actualiza el contexto Y activa el botón Explicar si hay key y datos.
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

        # Auto-registrar el método en la lista de sesión para el botón "Comparar"
        if metodo:
            if not hasattr(self, '_session_methods'):
                self._session_methods = []
            if not hasattr(self, '_session_results'):
                self._session_results = {}   # {metodo: result_summary}
            if metodo not in self._session_methods:
                self._session_methods.append(metodo)
            if result_summary:
                self._session_results[metodo] = result_summary

        # Si hay key y datos de un cálculo, activar botón con prompt genérico
        # (fallback por si explain_calculation no fue llamado por la app principal)
        if self._api_key and metodo and fx and not self._pending_explain:
            self._pending_explain = (
                f"Se ejecutó el método **{metodo}** sobre f = {fx}.\n"
                f"{self._context}\n\n"
                f"Explica de forma clara:\n"
                f"1. ¿Qué hizo {metodo} y cómo funciona?\n"
                f"2. ¿Qué significan los resultados obtenidos?\n"
                f"3. ¿Convergió bien? ¿Por qué?\n"
                f"4. ¿Alguna sugerencia para mejorar el resultado?"
            )
            self.btn_explain.setText(f"📊 Explicar: {metodo}")
            self.btn_explain.setEnabled(True)

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

        # ── Guardar resultado numérico real en _session_results ───────────────
        # Sobrescribe el resultado vago que pudo haber guardado update_context
        if not hasattr(self, '_session_results'):
            self._session_results = {}
        real_summary = f"x*={x_fin}, f*={f_fin}"
        if has_grad:
            real_summary += f", ‖∇f‖={grad_fin}"
        real_summary += f", iters={n}"
        self._session_results[metodo] = real_summary
        if not hasattr(self, '_session_methods'):
            self._session_methods = []
        if metodo not in self._session_methods:
            self._session_methods.append(metodo)

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
        """
        Usado solo para mensajes de bienvenida/conexión (sin math).
        Las respuestas del AI van por _on_response → _MathRenderWorker → _on_render_done.
        """
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
        blank = QtGui.QTextCharFormat()
        for raw in text.split('\n'):
            html = (
                '<span style="font-family:\'Segoe UI\',Consolas;font-size:11pt;">'
                + _md_to_html(raw)
                + '</span>'
            )
            cursor.insertHtml(html)
            cursor.insertText('\n', blank)
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
        """Elimina el último indicador de carga del chat."""
        html = self.chat_view.toPlainText()
        lines = html.split("\n")
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

    def _export_chat_pdf(self):
        """
        Exporta la conversación completa a PDF leyendo directamente de
        self._messages (lista de dicts con 'role'/'content'), lo que garantiza
        que el contenido sea 100 % fiel al historial real, sin depender del
        texto plano del QTextEdit que puede estar truncado o mal formateado.

        Características del PDF generado:
          • Portada con título, fecha/hora y estadísticas de la sesión
          • Burbuja de usuario (fondo azul claro) y burbuja de asistente
            (fondo verde claro) para distinción visual clara
          • Soporte de markdown básico: **negrita**, *cursiva*, `código inline`,
            bloques ```código```, listas con - / *, listas numeradas, encabezados #
          • Bloques de código con fondo oscuro y fuente monoespaciada
          • Separador visual entre cada par de turnos
          • Pie de página con número de página y nombre del asistente
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm, mm
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
                KeepTogether, Table, TableStyle, Preformatted
            )
            from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
            from reportlab.platypus.flowables import Flowable
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self, "Dependencia faltante",
                "Se necesita 'reportlab' para exportar el PDF.\n\n"
                "Instálalo con:\n    pip install reportlab"
            )
            return

        import datetime, re, html as _hl

        # ── Verificar que hay mensajes reales ────────────────────────────────
        real_msgs = [m for m in self._messages if m.get("role") in ("user", "assistant")]
        if not real_msgs:
            QtWidgets.QMessageBox.information(
                self, "Sin conversación",
                "Aún no hay mensajes en el historial para exportar.\n"
                "Envía al menos un mensaje primero."
            )
            return

        # ── Diálogo de destino ───────────────────────────────────────────────
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"chat_optimizador_{ts}.pdf"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Guardar conversación como PDF",
            default_name, "Archivos PDF (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"

        # ════════════════════════════════════════════════════════════════════
        #  PALETA Y CONSTANTES
        # ════════════════════════════════════════════════════════════════════
        PAGE_W, PAGE_H = A4
        M        = 1.8 * cm          # margen general
        INNER_W  = PAGE_W - 2 * M    # ancho útil

        C_HEADER_BG   = colors.HexColor("#0d1b3e")
        C_USER_BG     = colors.HexColor("#dceeff")
        C_USER_BORDER = colors.HexColor("#4a9eff")
        C_USER_TEXT   = colors.HexColor("#0a1e40")
        C_USER_LABEL  = colors.HexColor("#1a5bbf")
        C_AI_BG       = colors.HexColor("#d6f5e3")
        C_AI_BORDER   = colors.HexColor("#3cff8a")
        C_AI_TEXT     = colors.HexColor("#0a2a18")
        C_AI_LABEL    = colors.HexColor("#0d6e35")
        C_CODE_BG     = colors.HexColor("#1a1e2e")
        C_CODE_TEXT   = colors.HexColor("#ffd080")
        C_DIVIDER     = colors.HexColor("#c8d8e8")
        C_FOOTER      = colors.HexColor("#8090a8")
        C_HEADER_TEXT = colors.white
        C_SUBTITLE    = colors.HexColor("#9aabcc")

        base = getSampleStyleSheet()

        def _sty(name, **kw):
            s = ParagraphStyle(name, parent=base["Normal"], **kw)
            return s

        sty_title    = _sty("PdfTitle",    fontSize=18, leading=22,
                             textColor=C_HEADER_TEXT, backColor=C_HEADER_BG,
                             alignment=TA_CENTER, spaceAfter=0,
                             fontName="Helvetica-Bold",
                             leftPadding=12, rightPadding=12,
                             topPadding=10, bottomPadding=2)
        sty_subtitle = _sty("PdfSub",      fontSize=9,  leading=13,
                             textColor=C_SUBTITLE, backColor=C_HEADER_BG,
                             alignment=TA_CENTER, spaceAfter=0,
                             leftPadding=12, rightPadding=12,
                             topPadding=2, bottomPadding=10)
        sty_stats    = _sty("PdfStats",    fontSize=8,  leading=12,
                             textColor=C_SUBTITLE, backColor=C_HEADER_BG,
                             alignment=TA_CENTER, spaceAfter=16,
                             leftPadding=12, rightPadding=12, bottomPadding=12)

        # Estilos de cuerpo para usuario y asistente
        def _body_sty(name, text_color):
            return _sty(name, fontSize=10, leading=15,
                        textColor=text_color, leftIndent=0, rightIndent=0,
                        spaceAfter=3)

        sty_body_user  = _body_sty("BodyU",  C_USER_TEXT)
        sty_body_ai    = _body_sty("BodyAI", C_AI_TEXT)
        sty_label_user = _sty("LblU", fontSize=9, leading=11,
                               textColor=C_USER_LABEL, fontName="Helvetica-Bold",
                               spaceAfter=4)
        sty_label_ai   = _sty("LblAI", fontSize=9, leading=11,
                               textColor=C_AI_LABEL, fontName="Helvetica-Bold",
                               spaceAfter=4)
        sty_code       = ParagraphStyle("Code", parent=base["Code"],
                                        fontSize=8.5, leading=12,
                                        textColor=C_CODE_TEXT,
                                        backColor=C_CODE_BG,
                                        fontName="Courier",
                                        leftIndent=8, rightIndent=8,
                                        spaceAfter=4, spaceBefore=2,
                                        leftPadding=6, rightPadding=6,
                                        topPadding=4, bottomPadding=4)
        sty_heading    = _sty("MdH", fontSize=11, leading=15,
                               fontName="Helvetica-Bold",
                               spaceBefore=6, spaceAfter=2)
        sty_bullet_u   = _sty("BulU", fontSize=10, leading=15,
                               textColor=C_USER_TEXT,
                               leftIndent=14, firstLineIndent=-8, spaceAfter=2)
        sty_bullet_ai  = _sty("BulAI", fontSize=10, leading=15,
                               textColor=C_AI_TEXT,
                               leftIndent=14, firstLineIndent=-8, spaceAfter=2)
        sty_footer     = _sty("Footer", fontSize=7.5, textColor=C_FOOTER,
                               alignment=TA_CENTER)

        # ════════════════════════════════════════════════════════════════════
        #  HELPERS DE MARKDOWN → reportlab-HTML
        # ════════════════════════════════════════════════════════════════════
        def _esc(t: str) -> str:
            """Escapa para Paragraph de reportlab (no para bloques de código)."""
            return (_hl.escape(t)
                    .replace("&amp;amp;", "&amp;")   # evitar doble escape
                    )

        def _inline_md(line: str) -> str:
            """Convierte markdown inline a tags HTML de reportlab."""
            # Negrita+cursiva ***texto***
            line = re.sub(r'\*\*\*(.+?)\*\*\*',
                          lambda m: f'<b><i>{_esc(m.group(1))}</i></b>', line)
            # Negrita **texto**
            line = re.sub(r'\*\*(.+?)\*\*',
                          lambda m: f'<b>{_esc(m.group(1))}</b>', line)
            # Cursiva *texto* (no confundir con listas)
            line = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)',
                          lambda m: f'<i>{_esc(m.group(1))}</i>', line)
            # Código inline `código`
            line = re.sub(r'`([^`]+)`',
                          lambda m: (f'<font name="Courier" size="9">'
                                     f'<b>{_esc(m.group(1))}</b></font>'), line)
            # Si la línea no tiene etiquetas HTML ya, escapar el resto
            # (ya se escapó arriba; sólo las partes fuera de las etiquetas)
            return line

        def _parse_content(content: str, is_user: bool):
            """
            Convierte el contenido markdown de un mensaje en una lista de
            flowables de reportlab listos para insertar.
            """
            body_sty   = sty_body_user  if is_user else sty_body_ai
            bullet_sty = sty_bullet_u   if is_user else sty_bullet_ai
            items      = []
            lines      = content.split("\n")
            i          = 0

            while i < len(lines):
                line = lines[i]

                # ── Bloque de código ```...``` ────────────────────────────
                if line.strip().startswith("```"):
                    code_lines = []
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("```"):
                        code_lines.append(lines[i])
                        i += 1
                    i += 1  # saltar línea de cierre ```
                    code_text = "\n".join(code_lines)
                    # Usar Preformatted para preservar espacios
                    items.append(Preformatted(code_text, sty_code))
                    continue

                stripped = line.rstrip()

                # ── Encabezado Markdown # / ## / ### ─────────────────────
                m_hdr = re.match(r'^(#{1,3})\s+(.*)', stripped)
                if m_hdr:
                    level   = len(m_hdr.group(1))
                    htext   = m_hdr.group(2)
                    h_sty   = _sty(f"Hdr{level}_{id(htext)}",
                                   fontSize=13 - level,
                                   leading=17 - level,
                                   fontName="Helvetica-Bold",
                                   textColor=C_USER_LABEL if is_user else C_AI_LABEL,
                                   spaceBefore=8, spaceAfter=3)
                    items.append(Paragraph(_inline_md(htext), h_sty))
                    i += 1
                    continue

                # ── Lista con viñeta (- texto  o  * texto) ────────────────
                m_bul = re.match(r'^(\s*)[-*]\s+(.*)', stripped)
                if m_bul:
                    indent_extra = len(m_bul.group(1)) * 6
                    btext = m_bul.group(2)
                    bsty  = _sty(f"Bul_{i}",
                                 fontSize=10, leading=15,
                                 textColor=C_USER_TEXT if is_user else C_AI_TEXT,
                                 leftIndent=14 + indent_extra,
                                 firstLineIndent=-8, spaceAfter=2)
                    items.append(Paragraph("• " + _inline_md(btext), bsty))
                    i += 1
                    continue

                # ── Lista numerada  1. texto ──────────────────────────────
                m_num = re.match(r'^(\s*)(\d+)\.\s+(.*)', stripped)
                if m_num:
                    indent_extra = len(m_num.group(1)) * 6
                    num   = m_num.group(2)
                    ntext = m_num.group(3)
                    nsty  = _sty(f"Num_{i}",
                                 fontSize=10, leading=15,
                                 textColor=C_USER_TEXT if is_user else C_AI_TEXT,
                                 leftIndent=18 + indent_extra,
                                 firstLineIndent=-12, spaceAfter=2)
                    items.append(Paragraph(f"{num}.  {_inline_md(ntext)}", nsty))
                    i += 1
                    continue

                # ── Línea vacía → pequeño espacio ─────────────────────────
                if not stripped:
                    items.append(Spacer(1, 3))
                    i += 1
                    continue

                # ── Texto normal ──────────────────────────────────────────
                try:
                    items.append(Paragraph(_inline_md(stripped), body_sty))
                except Exception:
                    # Si el HTML inline da error, insertar texto plano seguro
                    items.append(Paragraph(_esc(stripped), body_sty))
                i += 1

            return items

        # ════════════════════════════════════════════════════════════════════
        #  BURBUJA DE MENSAJE  (tabla de 1 celda con fondo y borde)
        # ════════════════════════════════════════════════════════════════════
        def _bubble(inner_flowables: list, label_para, is_user: bool) -> Table:
            bg     = C_USER_BG     if is_user else C_AI_BG
            border = C_USER_BORDER if is_user else C_AI_BORDER

            content_cell = [label_para] + inner_flowables
            tbl = Table([[content_cell]], colWidths=[INNER_W])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, -1), bg),
                ("BOX",         (0, 0), (-1, -1), 0.8, border),
                ("LEFTPADDING",  (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING",   (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
                ("VALIGN",      (0, 0), (-1, -1), "TOP"),
            ]))
            return tbl

        # ════════════════════════════════════════════════════════════════════
        #  CONSTRUIR EL STORY COMPLETO
        # ════════════════════════════════════════════════════════════════════
        def _build_story() -> list:
            story = []
            now_str   = datetime.datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
            n_user    = sum(1 for m in real_msgs if m["role"] == "user")
            n_ai      = sum(1 for m in real_msgs if m["role"] == "assistant")

            # ── Portada / encabezado ─────────────────────────────────────
            story.append(Paragraph("🤖 Asistente de Optimización", sty_title))
            story.append(Paragraph(f"Conversación exportada el {now_str}", sty_subtitle))
            story.append(Paragraph(
                f"Turnos del usuario: {n_user}  •  Respuestas del asistente: {n_ai}  "
                f"•  Total de mensajes: {len(real_msgs)}",
                sty_stats))
            story.append(HRFlowable(width="100%", thickness=1.2,
                                    color=C_DIVIDER, spaceAfter=14))

            # ── Mensajes ─────────────────────────────────────────────────
            for idx, msg in enumerate(real_msgs):
                role    = msg.get("role", "")
                content = msg.get("content", "").strip()
                if not content:
                    continue

                is_user   = (role == "user")
                label_txt = "👤  Tú" if is_user else "🤖  Asistente"
                label_sty = sty_label_user if is_user else sty_label_ai

                label_para    = Paragraph(label_txt, label_sty)
                body_flowables = _parse_content(content, is_user)

                bubble = _bubble(body_flowables, label_para, is_user)
                story.append(bubble)
                story.append(Spacer(1, 6))

                # Separador fino entre pares usuario-asistente
                if not is_user and idx < len(real_msgs) - 1:
                    story.append(HRFlowable(
                        width="60%", thickness=0.4,
                        color=C_DIVIDER, spaceAfter=6,
                        hAlign="CENTER"))

            # ── Pie del documento ────────────────────────────────────────
            story.append(Spacer(1, 0.6 * cm))
            story.append(HRFlowable(width="100%", thickness=0.8,
                                    color=C_DIVIDER, spaceAfter=6))
            story.append(Paragraph(
                f"Fin de la conversación  •  Asistente de Optimización  •  {now_str}",
                sty_footer))
            return story

        # ── Pie de página con número ─────────────────────────────────────
        def _draw_page(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(C_FOOTER)
            # Línea divisoria
            canvas.setStrokeColor(C_DIVIDER)
            canvas.setLineWidth(0.4)
            canvas.line(M, M * 0.85, PAGE_W - M, M * 0.85)
            # Número de página
            canvas.drawRightString(PAGE_W - M, M * 0.45, f"Página {doc.page}")
            # Nombre del asistente a la izquierda
            canvas.drawString(M, M * 0.45, "Asistente de Optimización")
            canvas.restoreState()

        # ── Generar el PDF ───────────────────────────────────────────────
        try:
            self.btn_export_pdf.setEnabled(False)
            self.btn_export_pdf.setText("⏳ Generando…")
            QtWidgets.QApplication.processEvents()

            doc = SimpleDocTemplate(
                path, pagesize=A4,
                leftMargin=M, rightMargin=M,
                topMargin=M, bottomMargin=M * 1.5,
                title="Chat — Asistente de Optimización",
                author="Asistente de Optimización",
            )
            doc.build(_build_story(),
                      onFirstPage=_draw_page,
                      onLaterPages=_draw_page)

            self.btn_export_pdf.setEnabled(True)
            self.btn_export_pdf.setText("📄 Exportar PDF")
            QtWidgets.QMessageBox.information(
                self, "PDF exportado correctamente",
                f"✅ Conversación guardada en:\n\n{path}\n\n"
                f"Mensajes exportados: {len(real_msgs)}"
            )
        except Exception as exc:
            self.btn_export_pdf.setEnabled(True)
            self.btn_export_pdf.setText("📄 Exportar PDF")
            QtWidgets.QMessageBox.critical(
                self, "Error al exportar",
                f"No se pudo generar el PDF:\n\n{exc}"
            )

    def _clear_chat(self):
        self.chat_view.clear()
        self._messages.clear()
        self._busy = False
        self._retry_count = 0
        self._session_methods = []
        self._session_results = {}
        self._msg_queue.clear()
        self._continuation_blocks = []
        self.btn_continue.setVisible(False)
        self._btn_compare.setText("Comparar métodos utilizados")
        self.btn_send.setEnabled(bool(self._api_key))
        self.btn_send.setText("➤")
        self._append_system_msg("🗑 Chat limpiado. ¿En qué puedo ayudarte?")

    # ══════════════════════════════════════════════════════════════════════════
    #  API PÚBLICA — llamar desde interfaz_qt.py antes de ejecutar el cálculo
    # ══════════════════════════════════════════════════════════════════════════
    def check_method_compatibility(self,
                                   metodo:           str,
                                   fx:               str,
                                   n_vars:           int,
                                   tiene_restriccion: bool) -> bool:
        """
        Verifica si el método es compatible con la configuración actual.
        Si no lo es, muestra un aviso del asistente de forma INMEDIATA
        (sin llamada a la API) y retorna False.
        Retorna True si todo es compatible (la ejecución puede continuar).

        Llamar así desde interfaz_qt.py, justo ANTES de lanzar el cálculo:
        ────────────────────────────────────────────────────────────────────
            ok = panel_ai.check_method_compatibility(
                    metodo=nombre_metodo,        # str, ej. "Fibonacci"
                    fx=texto_funcion,            # str, ej. "x1**2+x2**2"
                    n_vars=len(variables),       # int
                    tiene_restriccion=bool(restriccion.strip())
                 )
            if not ok:
                return   # bloquear la ejecución
        ────────────────────────────────────────────────────────────────────
        """
        regla = _METHOD_RULES.get(metodo)
        if regla is None:
            return True   # método desconocido → no bloquear

        problemas  = []
        sugeridos  = []

        # ── Verificar número de variables ─────────────────────────────────
        max_v = regla.get("max_vars")
        if max_v is not None and n_vars > max_v:
            problemas.append(
                f"**{metodo}** solo funciona con funciones de **{max_v} variable** "
                f"(1D). Tu función tiene **{n_vars} variables**."
            )
            sugeridos = regla.get("alternativas_nd", [])

        # ── Verificar restricción requerida ───────────────────────────────
        elif regla.get("necesita_restriccion") and not tiene_restriccion:
            problemas.append(
                f"**{metodo}** está diseñado para problemas **con restricción** "
                f"activa (g(x) = ...). No tienes ninguna restricción definida."
            )
            sugeridos = regla.get("alternativas_sin_restriccion", [])

        # ── Verificar restricción no soportada ────────────────────────────
        elif not regla.get("acepta_restriccion", True) and tiene_restriccion:
            problemas.append(
                f"**{metodo}** es un método **sin restricciones**. "
                f"No puede manejar directamente la restricción g(x) que ingresaste."
            )
            sugeridos = regla.get("alternativas_restriccion", [])

        if not problemas:
            return True   # todo compatible

        # ── Construir y mostrar el aviso del asistente ────────────────────
        lista_sug = ""
        if sugeridos:
            items = "".join(f"\n  • {s}" for s in sugeridos)
            lista_sug = f"\n\n**Métodos recomendados para esta configuración:**{items}"

        aviso = (
            f"⚠️ **Incompatibilidad detectada**\n\n"
            + "\n".join(problemas)
            + lista_sug
            + "\n\nPor favor elige un método compatible antes de ejecutar."
        )
        self._append_ai_msg(aviso)
        return False
