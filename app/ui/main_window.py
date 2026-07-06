# interfaz_qt.py — Optimizador de Funciones
# ─────────────────────────────────────────────────────────────────────────────
# CORRECCIONES v4:
#  1. Módulos (localsearch, line_search, wrappers) se cargan LAZY, usando
#     resource_path para agregar el directorio del proyecto a sys.path ANTES
#     de cualquier importación.
#  2. Fondo con paintEvent + WA_OpaquePaintEvent (funciona en Windows/PyQt5/6).
#  3. Los widgets hijos NO son transparentes — tienen fondo propio semiopaco.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import csv, importlib, math, os, sys, tempfile, traceback
from typing import Any, Dict, List, Optional

# ── PyInstaller: agregar _MEIPASS a sys.path ANTES de cualquier import dinámico ──
# Sin esto, __import__("localsearch") falla en el EXE aunque el .py esté en datas.
if getattr(sys, 'frozen', False):
    _meipass = getattr(sys, '_MEIPASS', '')
    if _meipass and _meipass not in sys.path:
        sys.path.insert(0, _meipass)

# ── Qt ───────────────────────────────────────────────────────────────────────
_QT = None
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import Qt, QPoint, QRect
    from PyQt6.QtGui  import (QColor, QFont, QLinearGradient, QPainter,
                               QPen, QBrush, QPixmap)
    _QT = "PyQt6"
except Exception:
    from PyQt5 import QtCore, QtGui, QtWidgets            # type: ignore
    from PyQt5.QtCore import Qt, QPoint, QRect            # type: ignore
    from PyQt5.QtGui  import (QColor, QFont, QLinearGradient, QPainter,  # type: ignore
                               QPen, QBrush, QPixmap)
    _QT = "PyQt5"

# Helpers de atributos Qt (compatibilidad PyQt5/6)
def _WA(name):
    return getattr(Qt.WidgetAttribute, name) if _QT=="PyQt6" else getattr(Qt, name)
def _AA(name):
    return getattr(QPainter.RenderHint, name) if _QT=="PyQt6" else getattr(QPainter, name)
def _AL(*names):
    if _QT=="PyQt6":
        f=Qt.AlignmentFlag; r=getattr(f,names[0])
        for n in names[1:]: r|=getattr(f,n)
    else:
        r=getattr(Qt,names[0])
        for n in names[1:]: r|=getattr(Qt,n)
    return r

QApp   = QtWidgets.QApplication;  QMW   = QtWidgets.QMainWindow
QW     = QtWidgets.QWidget;       QVBox = QtWidgets.QVBoxLayout
QHBox  = QtWidgets.QHBoxLayout;   QGrid = QtWidgets.QGridLayout
QLbl   = QtWidgets.QLabel;        QLE   = QtWidgets.QLineEdit
QCmb   = QtWidgets.QComboBox;     QBtn  = QtWidgets.QPushButton
QFrame = QtWidgets.QFrame;        QTbl  = QtWidgets.QTableWidget
QTI    = QtWidgets.QTableWidgetItem; QHV = QtWidgets.QHeaderView
QDSpin = QtWidgets.QDoubleSpinBox; QMsgBox = QtWidgets.QMessageBox
QSplit = QtWidgets.QSplitter;     QFD  = QtWidgets.QFileDialog
QScroll = QtWidgets.QScrollArea
QMenu  = QtWidgets.QMenu
try:    QAction = QtWidgets.QAction
except: QAction = QtGui.QAction                           # type: ignore

import numpy as np
try:    import sympy as sp; _SYMPY = True
except: sp = None; _SYMPY = False                         # type: ignore


# ── QLineEdit con menú contextual en español ──────────────────────────────────
class SpanishLineEdit(QLE):
    """QLineEdit con menú contextual completamente en español."""
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#141e36;color:#d8e8f8;border:1px solid #2a3e6e;"
            "border-radius:6px;padding:4px;font-size:13px;}"
            "QMenu::item{padding:6px 22px;border-radius:4px;}"
            "QMenu::item:selected{background:#1a50c0;}"
            "QMenu::item:disabled{color:#4a6080;}"
            "QMenu::separator{height:1px;background:#2a3e6e;margin:3px 8px;}"
        )

        act_undo   = menu.addAction("Deshacer")
        act_redo   = menu.addAction("Rehacer")
        menu.addSeparator()
        act_cut    = menu.addAction("Cortar")
        act_copy   = menu.addAction("Copiar")
        act_paste  = menu.addAction("Pegar")
        act_delete = menu.addAction("Eliminar")
        menu.addSeparator()
        act_selall = menu.addAction("Seleccionar todo")

        # Habilitar/deshabilitar según estado
        act_undo.setEnabled(self.isUndoAvailable())
        act_redo.setEnabled(self.isRedoAvailable())
        has_sel = bool(self.selectedText())
        act_cut.setEnabled(has_sel and not self.isReadOnly())
        act_copy.setEnabled(has_sel)
        act_delete.setEnabled(has_sel and not self.isReadOnly())
        try:
            cb = QtWidgets.QApplication.clipboard()
            act_paste.setEnabled(bool(cb.text()) and not self.isReadOnly())
        except Exception:
            act_paste.setEnabled(not self.isReadOnly())
        act_selall.setEnabled(bool(self.text()))

        chosen = menu.exec(event.globalPos()) if _QT == "PyQt6" else menu.exec_(event.globalPos())
        if chosen is None:
            return
        if chosen == act_undo:    self.undo()
        elif chosen == act_redo:  self.redo()
        elif chosen == act_cut:   self.cut()
        elif chosen == act_copy:  self.copy()
        elif chosen == act_paste: self.paste()
        elif chosen == act_delete:
            cur = self.cursorPosition()
            sel_start = self.selectionStart()
            self.del_()  # borra selección o char siguiente
        elif chosen == act_selall: self.selectAll()
try:    import matplotlib; matplotlib.use("QtAgg")
except: pass

# NavigationToolbar2QT — para zoom/pan/home sin barra visible
_NavToolbar = None
try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as _NavToolbar
except Exception:
    try:
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as _NavToolbar  # type: ignore
    except Exception:
        _NavToolbar = None

# ── Canvases (importación silenciosa) ────────────────────────────────────────
Function2DCanvas = Rotating3DCanvas = None
def _load_canvases():
    global Function2DCanvas, Rotating3DCanvas
    for m in ("app.ui.canvas_2d_estilo_fixed","app.ui.canvas_2d"):
        try: Function2DCanvas=__import__(m,fromlist=["Function2DCanvas"]).Function2DCanvas; break
        except: pass
    for m in ("app.ui.rotacion_3d_superficie","app.ui.rotacion_3d"):
        try: Rotating3DCanvas=__import__(m,fromlist=["Rotating3DCanvas"]).Rotating3DCanvas; break
        except: pass

# ── Módulos de métodos (se cargan LAZY en _init_modules) ─────────────────────
newton_armijo_fn = newton_wolfe_fn = None
fibonacci_fn     = local_fn        = None
_wrappers_mod    = None
_heuristics_mod  = None   # heuristics.py
_metaheur_mod    = None   # metaheuristics.py
_modules_ready   = False

def _add_to_path(path: str):
    """Agrega una ruta a sys.path si no está ya incluida."""
    path = os.path.normpath(path)
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

def _try_import(name):
    try:    return importlib.import_module(name)
    except: return None

def _init_modules(resource_path: str = ""):
    """Carga todos los módulos de métodos. Llamado desde __init__ con resource_path."""
    global newton_armijo_fn, newton_wolfe_fn, fibonacci_fn, local_fn
    global _wrappers_mod, _heuristics_mod, _metaheur_mod, _modules_ready
    if _modules_ready:
        return

    here = os.path.dirname(os.path.abspath(__file__))
    cwd  = os.getcwd()

    # Agregar TODOS los directorios candidatos a sys.path
    dirs_to_add = []
    if resource_path:
        dirs_to_add.append(resource_path)
    dirs_to_add += [here, cwd]
    for base in list(dirs_to_add):          # subcarpetas comunes
        for sub in ("multi-dimensionals", "multidimensionals",
                    "metodos", "methods"):
            dirs_to_add.append(os.path.join(base, sub))
    for d in dirs_to_add:
        _add_to_path(d)

    # Importar módulos 1D clásicos
    ls  = _try_import("app.optimization.line_search.nd_symbolic")
    loc = _try_import("app.optimization.local_search")

    newton_armijo_fn = getattr(ls,  "newton_armijo",            None) if ls  else None
    newton_wolfe_fn  = getattr(ls,  "newton_wolfe_step",         None) if ls  else None
    fibonacci_fn     = getattr(loc, "fibonacci_search",          None) if loc else None
    local_fn         = getattr(loc, "local_neighborhood_search", None) if loc else None

    # Importar wrappers MD
    _wrappers_mod   = _try_import("app.optimization.constrained.penalty_barrier")

    # Importar heurísticos y metaheurísticos
    _heuristics_mod = _try_import("app.optimization.heuristics")
    _metaheur_mod   = _try_import("app.optimization.metaheuristics")

    _modules_ready = True
    print(f"[módulos] line_search={'OK' if ls else 'NO'}  "
          f"localsearch={'OK' if loc else 'NO'}  "
          f"wrappers={'OK' if _wrappers_mod else 'NO'}  "
          f"heuristics={'OK' if _heuristics_mod else 'NO'}  "
          f"metaheuristics={'OK' if _metaheur_mod else 'NO'}")
    print(f"[sys.path] {sys.path[:5]}")

# ── Módulo de compatibilidad de funciones (carga lazy) ───────────────────────
_func_compat = None
def _get_fc():
    global _func_compat
    if _func_compat is None:
        try:
            import app.application.func_compat as _fc
            _func_compat = _fc
        except ImportError:
            pass
    return _func_compat

# ── Módulo del Asistente de IA (carga lazy) ───────────────────────────────────
_ai_panel_cls = None
def _get_ai_panel_cls():
    global _ai_panel_cls
    if _ai_panel_cls is None:
        try:
            from app.infrastructure.ai_assistant import AIAssistantPanel as _cls
            _ai_panel_cls = _cls
        except ImportError:
            pass
    return _ai_panel_cls

# ── Métodos ───────────────────────────────────────────────────────────────────
METHODS_1D = ["Búsqueda Local", "Fibonacci", "Armijo", "Wolfe"]
METHODS_MD = [
    "MD: Penalización (Newton)", "MD: Barreras (Newton)",
    "MD: Pen. (BFGS)",           "MD: Pes. (BFGS)",
    "MD: Pes. (Nelder-Mead)",    "MD: Sum. (BFGS)",
]
METHODS_HEU = [
    "Sección Dorada",
    "Sección Áurea",
    "Goldstein",
    "Newton-Raphson",
    "Grad. Conjugado",
]
METHODS_META = [
    "SA: Recocido Simulado",
    "PSO: Enjambre",
    "GA: Algoritmo Genético",
]
# ── Multiobjetivo ─────────────────────────────────────────────────────────────
# El usuario ingresa f(x) mono-objetivo → el sistema la convierte en MO
# aplicando Escalarización con α → luego resuelve con el algoritmo elegido.
# La tabla de α es un RESULTADO (referencia), no un método seleccionable.
METHODS_MULTIOBJ = [
    "Escalarización (Suma Ponderada)", # ← ESENCIAL: convierte F(x)=[f1,f2] en φ(x,α)
    "MO — Bisección",
    "MO — Sección Dorada",
    "MO — Frente de Pareto",
    "MO — Análisis Jacobiano",
]

# Métodos de búsqueda lineal que aceptan funciones multivariable
METHODS_LS   = ["Armijo", "Wolfe"]
# Métodos que usan vars+x0 pero no restricción
METHODS_ND   = METHODS_HEU[1:] + METHODS_META   # Goldstein, NR, GC + meta
# Métodos que usan bounds [lo,hi] + vars (no necesitan x0 manual)
METHODS_BOUNDS = ["PSO: Enjambre", "GA: Algoritmo Genético"]

ALL_METHODS = METHODS_1D + METHODS_MD + METHODS_HEU + METHODS_META + METHODS_MULTIOBJ

# ── Agrupación por categoría para el ComboBox ─────────────────────────────────
# Orden y clasificación basados en la naturaleza del método (1D → ND → MD → Meta → MO)
GROUPED_METHODS = [
    ("Unidimensionales (1D)", [
        "Búsqueda Local",
        "Fibonacci",
        "Sección Áurea",
        "Sección Dorada",
        "Goldstein",
        "Armijo",
        "Wolfe",
        "Newton-Raphson",
    ]),
    ("Multidimensionales (ND)", [
        "Grad. Conjugado",
    ]),
    ("Restringidos (MD)", [
        "MD: Penalización (Newton)",
        "MD: Barreras (Newton)",
        "MD: Pen. (BFGS)",
        "MD: Pes. (BFGS)",
        "MD: Pes. (Nelder-Mead)",
        "MD: Sum. (BFGS)",
    ]),
    ("Global / Metaheurística", [
        "SA: Recocido Simulado",
        "PSO: Enjambre",
        "GA: Algoritmo Genético",
    ]),
    ("Multiobjetivo (MO)", [
        "Escalarización (Suma Ponderada)",   # ← convierte F(x)=[f1,f2] en φ(x,α)
        "MO — Bisección",
        "MO — Sección Dorada",
        "MO — Frente de Pareto",
        "MO — Análisis Jacobiano",
    ]),
]

# Rol interno para marcar items de encabezado de categoría en el ComboBox
_CMB_HEADER_ROLE = (Qt.ItemDataRole.UserRole + 10
                    if _QT == "PyQt6" else Qt.UserRole + 10)


# ── Delegado para encabezados de grupo en el ComboBox ────────────────────────
class _GroupHeaderDelegate(QtWidgets.QStyledItemDelegate):
    """Pinta los encabezados de categoría con fondo y tipografía distintos;
    los ítems normales se delegan al comportamiento estándar."""

    _BG   = QColor("#0b1628")
    _FG   = QColor("#5ba3ff")
    _LINE = QColor("#1e3a6e")

    def paint(self, painter, option, index):
        if index.data(_CMB_HEADER_ROLE):
            painter.save()
            r = option.rect
            painter.fillRect(r, self._BG)
            # línea divisoria superior
            painter.setPen(QPen(self._LINE, 1))
            painter.drawLine(r.left(), r.top(), r.right(), r.top())
            # texto
            painter.setPen(self._FG)
            font = QFont(option.font)
            font.setBold(True)
            font.setPointSizeF(max(7.5, font.pointSizeF() - 0.5))
            painter.setFont(font)
            text_r = r.adjusted(10, 0, -6, 0)
            flags = (_AL("AlignVCenter", "AlignLeft")
                     if _QT == "PyQt6" else _AL("AlignVCenter", "AlignLeft"))
            painter.drawText(text_r, flags, index.data())
            painter.restore()
        else:
            super().paint(painter, option, index)

    def sizeHint(self, option, index):
        sh = super().sizeHint(option, index)
        if index.data(_CMB_HEADER_ROLE):
            return sh.__class__(sh.width(), max(sh.height(), 24))
        return sh


def _populate_grouped_combo(cmb: "QCmb") -> None:
    """Pobla el QComboBox con métodos agrupados por categoría.
    Los encabezados son visualmente distintos y no seleccionables."""
    cmb.setItemDelegate(_GroupHeaderDelegate(cmb))
    cmb.addItem("")                    # ítem vacío inicial (placeholder)

    for category, methods in GROUPED_METHODS:
        # ── Encabezado de categoría ──────────────────────────────────────────
        cmb.addItem(category)
        hdr_idx = cmb.count() - 1
        hdr_item = cmb.model().item(hdr_idx)
        hdr_item.setData(True, _CMB_HEADER_ROLE)
        if _QT == "PyQt6":
            hdr_item.setFlags(
                hdr_item.flags()
                & ~Qt.ItemFlag.ItemIsSelectable
                & ~Qt.ItemFlag.ItemIsEnabled)
        else:
            hdr_item.setFlags(
                hdr_item.flags()
                & ~Qt.ItemIsSelectable
                & ~Qt.ItemIsEnabled)

        # ── Métodos de la categoría ──────────────────────────────────────────
        for method in methods:
            cmb.addItem(method)

def _is_md(m):        return m in METHODS_MD
def _is_ls(m):        return m in METHODS_LS
def _is_fib(m):       return m == "Fibonacci"
def _is_heu(m):       return m in METHODS_HEU
def _is_meta(m):      return m in METHODS_META
def _is_multiobj(m):  return m in METHODS_MULTIOBJ
def _is_1d_only(m):   return m in ("Búsqueda Local", "Fibonacci", "Sección Áurea",
                                     "Bisección", "Sección Dorada")
def _needs_x0(m):     return m in (METHODS_LS + METHODS_ND)
def _needs_alpha(m):  return m in (
    "MO — Bisección",
    "MO — Sección Dorada",
)
# Ningún método MO tiene función predefinida — el usuario siempre ingresa f(x)
_MO_SCHAFFER_FIXED: set = set()   # vacío: todos requieren función del usuario

# ── Módulo multiobjetivo (carga lazy) ─────────────────────────────────────────
_multiobj_mod = None
def _get_multiobj():
    global _multiobj_mod
    if _multiobj_mod is None:
        try:
            import app.optimization.multiobjective.schaffer as _mo
            _multiobj_mod = _mo
        except ImportError:
            pass
    return _multiobj_mod

# ── Ruta de imagen de fondo ───────────────────────────────────────────────────
def _find_bg(extra: list = None) -> str:
    # __file__ vive en app/ui/, la carpeta Menu/ sigue en la raíz del proyecto
    # (o en _MEIPASS cuando corre congelado con PyInstaller).
    here = getattr(sys, '_MEIPASS', None) or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cands = (extra or []) + [
        os.path.join(here, "Menu", "Fondo", "Imgen de fondo.jpg"),
        os.path.join(here, "Menu", "Fondo", "Imgen de fondo.png"),
        os.path.join(here, "Menu", "Fondo", "Imgen de fondo.jpg"),
        os.path.join(here, "Menu", "Fondo", "Imgen de fondo.png"),

        os.path.join(here, "Imgen_de_fondo.jpg"),
        os.path.join(here, "Imgen_de_fondo.png"),
        os.path.join(here, "background.jpg"),
        "/mnt/user-data/uploads/Imgen_de_fondo.jpg",
        "/mnt/user-data/outputs/Imgen_de_fondo.jpg",
        "/mnt/user-data/outputs/Imgen_de_fondo.jpg",
        "/mnt/user-data/outputs/Imgen_de_fondo.jpg",
    ]
    return next((p for p in cands if p and os.path.exists(p)), "")


# ══════════════════════════════════════════════════════════════════════════════
#  LegendResizeOverlay — handles de redimensionado + mover leyenda
# ══════════════════════════════════════════════════════════════════════════════
class LegendResizeOverlay(QW):
    """
    Comportamiento de interacción con la leyenda:

    CLIC IZQUIERDO en el centro de la leyenda
        → Selecciona la leyenda y muestra los 8 handles estilo Excel.
        → Al arrastrar un handle se expande o contrae el recuadro.
        → Clic fuera de la leyenda → deselecciona y oculta los handles.

    CLIC DERECHO sobre la leyenda
        → Abre un pequeño menú contextual con la opción "Mover leyenda".
        → Al seleccionarla la leyenda queda "pegada" al cursor (modo mover).
        → La leyenda sigue al cursor en tiempo real.
        → Dos clics izquierdos consecutivos sueltan la leyenda en esa posición.
    """
    HANDLE = 8   # tamaño del cuadradito en px

    def __init__(self, canvas, ax, parent=None):
        super().__init__(canvas)
        self.canvas = canvas
        self.ax     = ax

        # ── estado resize ────────────────────────────────────────────────────
        self._active     = False   # handles visibles
        self._dragging   = False
        self._drag_hdl   = None
        self._drag_start = None
        self._init_fs    = 10.0
        self._handles    = []

        # ── estado mover ─────────────────────────────────────────────────────
        self._move_mode        = False
        self._move_click_count = 0   # 2 clics izq para soltar
        self._motion_cid       = None
        # offset en fracción de ejes entre cursor y esquina inf-izq de la leyenda
        self._move_offset_ax   = (0.0, 0.0)

        self.setAttribute(_WA("WA_TranslucentBackground"), True)
        self.setMouseTracking(True)
        self.resize(canvas.size())
        canvas.installEventFilter(self)

        # Eventos matplotlib
        canvas.fig.canvas.mpl_connect("button_press_event",   self._on_mpl_click)
        canvas.fig.canvas.mpl_connect("motion_notify_event",  self._on_mpl_motion)

        self.raise_()
        self.hide()   # inicia oculto

    # ══════════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _in_legend(self, event):
        """True si el evento matplotlib cae dentro del bbox de la leyenda."""
        leg = self.ax.get_legend()
        if leg is None or not leg.get_visible():
            return False
        try:
            renderer = self.canvas.fig.canvas.get_renderer()
            bb = leg.get_window_extent(renderer)
            return bb.contains(event.x, event.y)
        except Exception:
            return False

    def _display_to_axes(self, x, y):
        """Convierte coordenadas display (mpl) a fracción de ejes."""
        try:
            return self.ax.transAxes.inverted().transform((x, y))
        except Exception:
            return (0.5, 0.5)

    # ══════════════════════════════════════════════════════════════════════════
    #  Eventos matplotlib
    # ══════════════════════════════════════════════════════════════════════════
    def _on_mpl_click(self, event):
        # ── En modo mover: contar clics izquierdos para soltar ────────────────
        if self._move_mode:
            if event.button == 1:
                self._move_click_count += 1
                if self._move_click_count >= 2:
                    self._stop_move_mode()
            return

        # ── Clic derecho sobre la leyenda → menú contextual ──────────────────
        if event.button == 3 and self._in_legend(event):
            self._show_legend_menu(event)
            return

        # ── Clic izquierdo → activar/desactivar handles ───────────────────────
        if event.button == 1:
            if self._in_legend(event):
                self._activate()
            else:
                self._deactivate()

    def _on_mpl_motion(self, event):
        """En modo mover: la leyenda sigue al cursor en tiempo real."""
        if not self._move_mode:
            return
        try:
            ax_pos = self._display_to_axes(event.x, event.y)
            ox, oy = self._move_offset_ax
            new_x = ax_pos[0] + ox
            new_y = ax_pos[1] + oy
            leg = self.ax.get_legend()
            if leg:
                leg.set_bbox_to_anchor((new_x, new_y), transform=self.ax.transAxes)
                self.canvas.draw_idle()
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    #  Menú contextual de la leyenda (clic derecho)
    # ══════════════════════════════════════════════════════════════════════════
    def _show_legend_menu(self, event):
        menu = QMenu(self.canvas)
        menu.setStyleSheet(
            "QMenu{background:#141e36;color:#d8e8f8;border:1px solid #2a3e6e;"
            "border-radius:6px;padding:4px;font-size:13px;}"
            "QMenu::item{padding:6px 22px;border-radius:4px;}"
            "QMenu::item:selected{background:#1a50c0;}"
        )
        act_move = menu.addAction("✋  Mover leyenda")
        act_move.setToolTip("La leyenda seguirá al cursor. Da 2 clics para soltarla.")

        try:
            gpos = self.canvas.mapToGlobal(
                QtCore.QPoint(int(event.x), int(self.canvas.height() - event.y)))
        except Exception:
            gpos = QtGui.QCursor.pos()

        chosen = menu.exec(gpos) if _QT == "PyQt6" else menu.exec_(gpos)
        if chosen == act_move:
            self._start_move_mode(event)

    # ══════════════════════════════════════════════════════════════════════════
    #  Modo mover
    # ══════════════════════════════════════════════════════════════════════════
    def _start_move_mode(self, event):
        """Activa el modo mover: la leyenda sigue al cursor."""
        self._deactivate()          # ocultar handles si estaban visibles
        self._move_mode        = True
        self._move_click_count = 0

        # Calcular offset para que la leyenda no "salte" al activar el modo
        try:
            leg = self.ax.get_legend()
            renderer = self.canvas.fig.canvas.get_renderer()
            bb = leg.get_window_extent(renderer)
            # esquina inf-izq de la leyenda en coords de ejes
            leg_ax = self._display_to_axes(bb.x0, bb.y0)
            # cursor en coords de ejes
            cur_ax = self._display_to_axes(event.x, event.y)
            self._move_offset_ax = (leg_ax[0] - cur_ax[0],
                                    leg_ax[1] - cur_ax[1])
            # Desactivar el draggable nativo para evitar conflictos
            leg.set_draggable(False)
        except Exception:
            self._move_offset_ax = (0.0, 0.0)

        # Cursor SizeAll
        cur = (Qt.CursorShape.SizeAllCursor if _QT == "PyQt6"
               else Qt.SizeAllCursor)
        self.canvas.setCursor(QtGui.QCursor(cur))

        # Mostrar tooltip de instrucción
        try:
            parent_win = self.canvas.parent()
            while parent_win and not hasattr(parent_win, "lbl_st"):
                parent_win = parent_win.parent()
            if parent_win:
                parent_win.lbl_st.setText(
                    "✋ Modo mover activo — mueve el cursor y da 2 clics para soltar la leyenda.")
        except Exception:
            pass

    def _stop_move_mode(self):
        """Desactiva el modo mover y restaura el estado normal."""
        self._move_mode        = False
        self._move_click_count = 0
        self.canvas.unsetCursor()
        # Restaurar draggable
        try:
            leg = self.ax.get_legend()
            if leg:
                leg.set_draggable(True, use_blit=False)
        except Exception:
            pass
        self.canvas.draw_idle()
        # Limpiar tooltip
        try:
            parent_win = self.canvas.parent()
            while parent_win and not hasattr(parent_win, "lbl_st"):
                parent_win = parent_win.parent()
            if parent_win:
                parent_win.lbl_st.setText(
                    "✅ Leyenda colocada. Clic derecho sobre ella para volver a moverla.")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    #  Activación / desactivación de handles
    # ══════════════════════════════════════════════════════════════════════════
    def _activate(self):
        if not self._active:
            self._active = True
            self.raise_()
            self.show()
        self.update()

    def _deactivate(self):
        if self._active:
            self._active   = False
            self._dragging = False
            self.hide()

    # ══════════════════════════════════════════════════════════════════════════
    #  Geometría de handles
    # ══════════════════════════════════════════════════════════════════════════
    def _leg_bbox(self):
        """QRect de la leyenda en coordenadas Qt (y invertida)."""
        leg = self.ax.get_legend()
        if leg is None or not leg.get_visible():
            return None
        try:
            renderer = self.canvas.fig.canvas.get_renderer()
            bb = leg.get_window_extent(renderer)
            h  = self.canvas.height()
            return QtCore.QRect(int(bb.x0), int(h - bb.y1),
                                int(bb.x1 - bb.x0), int(bb.y1 - bb.y0))
        except Exception:
            return None

    def _build_handles(self, rect):
        hs = self.HANDLE
        h2 = hs // 2
        x0, y0 = rect.x(), rect.y()
        xm, ym = x0 + rect.width() // 2, y0 + rect.height() // 2
        x1, y1 = x0 + rect.width(),      y0 + rect.height()

        def _r(cx, cy):
            return QtCore.QRect(cx - h2, cy - h2, hs, hs)

        self._handles = [
            (0, 0, _r(x0, y0)), (0, 1, _r(xm, y0)), (0, 2, _r(x1, y0)),
            (1, 0, _r(x0, ym)),                       (1, 2, _r(x1, ym)),
            (2, 0, _r(x0, y1)), (2, 1, _r(xm, y1)), (2, 2, _r(x1, y1)),
        ]

    def _handle_at(self, pos):
        for row, col, rect in self._handles:
            if rect.contains(pos):
                return row, col
        return None

    @staticmethod
    def _cursor_for_handle(row, col):
        mapping = {
            (0, 0): "SizeFDiagCursor", (0, 2): "SizeBDiagCursor",
            (2, 0): "SizeBDiagCursor", (2, 2): "SizeFDiagCursor",
            (0, 1): "SizeVerCursor",   (2, 1): "SizeVerCursor",
            (1, 0): "SizeHorCursor",   (1, 2): "SizeHorCursor",
        }
        name = mapping.get((row, col), "ArrowCursor")
        return (QtGui.QCursor(getattr(Qt.CursorShape, name)) if _QT == "PyQt6"
                else QtGui.QCursor(getattr(Qt, name)))

    # ══════════════════════════════════════════════════════════════════════════
    #  Paint
    # ══════════════════════════════════════════════════════════════════════════
    def paintEvent(self, event):
        if not self._active:
            return
        rect = self._leg_bbox()
        if rect is None:
            return
        self._build_handles(rect)

        p    = QPainter(self)
        hint = QPainter.RenderHint.Antialiasing if _QT == "PyQt6" else QPainter.Antialiasing
        p.setRenderHint(hint)

        # borde de selección punteado azul
        dash     = Qt.PenStyle.DashLine  if _QT == "PyQt6" else Qt.DashLine
        no_brush = Qt.BrushStyle.NoBrush if _QT == "PyQt6" else Qt.NoBrush
        p.setPen(QPen(QColor(80, 160, 255, 210), 1.5, dash))
        p.setBrush(QBrush(no_brush))
        p.drawRect(rect)

        # handles: cuadrado blanco con borde azul
        for _, _, hrect in self._handles:
            p.setPen(QPen(QColor(60, 130, 240), 1.2))
            p.setBrush(QBrush(QColor(255, 255, 255, 230)))
            p.drawRect(hrect)

        p.end()

    # ══════════════════════════════════════════════════════════════════════════
    #  Eventos Qt del overlay (redimensionado con handles)
    # ══════════════════════════════════════════════════════════════════════════
    def mousePressEvent(self, event):
        left = Qt.MouseButton.LeftButton if _QT == "PyQt6" else Qt.LeftButton
        if event.button() == left:
            hdl = self._handle_at(event.pos())
            if hdl:
                self._dragging   = True
                self._drag_hdl   = hdl
                self._drag_start = event.pos()
                leg = self.ax.get_legend()
                self._init_fs = (leg.get_texts()[0].get_fontsize()
                                 if leg and leg.get_texts() else 10.0)
                event.accept()
                return
            # Clic en el overlay fuera de un handle → deseleccionar
            self._deactivate()
        event.ignore()

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self._dragging and self._drag_start is not None:
            delta = pos - self._drag_start
            row, col = self._drag_hdl
            if   col == 2: d =  delta.x()
            elif col == 0: d = -delta.x()
            elif row == 2: d =  delta.y()
            elif row == 0: d = -delta.y()
            else:          d = (abs(delta.x()) + abs(delta.y())) * 0.5

            new_fs = max(6.0, min(22.0, self._init_fs + d * 0.07))
            leg = self.ax.get_legend()
            if leg:
                for txt in leg.get_texts():
                    txt.set_fontsize(new_fs)
                try:
                    leg.prop.set_size(new_fs)
                except Exception:
                    pass
                self.canvas.draw_idle()
            self.update()
            event.accept()
        else:
            hdl = self._handle_at(pos)
            if hdl:
                self.setCursor(self._cursor_for_handle(*hdl))
                event.accept()
            else:
                self.unsetCursor()
                event.ignore()

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging   = False
            self._drag_hdl   = None
            self._drag_start = None
            event.accept()
        else:
            event.ignore()

    # ══════════════════════════════════════════════════════════════════════════
    #  Resize del canvas
    # ══════════════════════════════════════════════════════════════════════════
    def eventFilter(self, obj, event):
        resize_t = QtCore.QEvent.Type.Resize if _QT == "PyQt6" else QtCore.QEvent.Resize
        if obj is self.canvas and event.type() == resize_t:
            self.resize(self.canvas.size())
            if self._active:
                self.update()
        return False

    def refresh(self):
        """Llamar tras draw_idle para repintar los handles si están activos."""
        if self._active:
            self.update()


# ══════════════════════════════════════════════════════════════════════════════
#  BgPanel — imagen de fondo directa, sin capas oscuras
# ══════════════════════════════════════════════════════════════════════════════
class BgPanel(QW):
    """
    Panel con imagen de fondo usando QPalette + paintEvent limpio.
    Sin velos ni capas oscuras — muestra la imagen tal cual.
    """
    def __init__(self, bg_path: str, overlay_alpha: int = 0, parent=None):
        super().__init__(parent)
        self._pix: Optional[QPixmap] = None
        self._bg_path = bg_path
        self._scaled_pix: Optional[QPixmap] = None
        self._last_size = (-1, -1)

        # Cargar imagen
        if bg_path and os.path.exists(bg_path):
            tmp = QPixmap(bg_path)
            if not tmp.isNull():
                self._pix = tmp
                print(f"[BgPanel] Imagen cargada: {bg_path}")
            else:
                print(f"[BgPanel] ERROR: QPixmap nulo para {bg_path}")
        else:
            print(f"[BgPanel] No se encontró imagen en: {bg_path!r}")

        # Necesario para que paintEvent funcione correctamente
        self.setAttribute(_WA("WA_OpaquePaintEvent"), True)
        self.setAutoFillBackground(False)

    def paintEvent(self, event):
        W, H = self.width(), self.height()
        if W <= 0 or H <= 0:
            return
        painter = QPainter(self)
        if self._pix:
            # Escalar al tamaño del panel cubriendo todo (sin deformar)
            if (W, H) != self._last_size:
                mode = (Qt.AspectRatioMode.KeepAspectRatioByExpanding if _QT=="PyQt6"
                        else Qt.KeepAspectRatioByExpanding)
                hint = (Qt.TransformationMode.SmoothTransformation if _QT=="PyQt6"
                        else Qt.SmoothTransformation)
                scaled = self._pix.scaled(W, H, mode, hint)
                ox = (scaled.width()  - W) // 2
                oy = (scaled.height() - H) // 2
                self._scaled_pix = scaled.copy(ox, oy, W, H)
                self._last_size = (W, H)
            painter.drawPixmap(0, 0, self._scaled_pix)
        else:
            # Fallback: fondo oscuro si no hay imagen
            painter.fillRect(0, 0, W, H, QColor(8, 14, 35))
        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._last_size = (-1, -1)
        self.update()


# ══════════════════════════════════════════════════════════════════════════════
#  LeftPanel — BgPanel + cubo 3D wireframe
# ══════════════════════════════════════════════════════════════════════════════
class LeftPanel(BgPanel):
    def __init__(self, bg_path, parent=None):
        super().__init__(bg_path, overlay_alpha=0, parent=parent)

    def paintEvent(self, event):
        # 1) Fondo (heredado)
        super().paintEvent(event)
        # Capa semiopaca sobre el fondo para mejorar legibilidad de los controles
        p0 = QPainter(self)
        p0.fillRect(0, 0, self.width(), self.height(), QColor(4, 10, 32, 175))
        p0.end()

        # 2) Cubo wireframe isométrico
        p = QPainter(self)
        p.setRenderHint(_AA("Antialiasing"))
        W, H = self.width(), self.height()

        cx, cy = W*0.50, H*0.67
        s = min(W, H)*0.28
        c30, s30 = math.cos(math.radians(30)), math.sin(math.radians(30))

        def iso(x, y, z):
            return QPoint(int(cx+(x-z)*c30*s), int(cy-y*s+(x+z)*s30*s*0.5))

        verts = {
            'A':(-1,-1,-1),'B':(1,-1,-1),'C':(1,1,-1),'D':(-1,1,-1),
            'E':(-1,-1, 1),'F':(1,-1, 1),'G':(1,1, 1),'H':(-1,1, 1),
        }
        V = {k: iso(*v) for k,v in verts.items()}

        p.setPen(QPen(QColor(55,110,220,50), 1.2))
        for a,b in [('A','B'),('B','C'),('C','D'),('D','A'),
                    ('A','E'),('B','F'),('C','G'),('D','H')]:
            p.drawLine(V[a], V[b])
        p.setPen(QPen(QColor(70,140,255,80), 2.2))
        for a,b in [('E','F'),('F','G'),('G','H'),('H','E')]:
            p.drawLine(V[a], V[b])

        # Cubo interior
        s2 = s*0.46
        def iso2(x,y,z):
            return QPoint(int(cx+(x-z)*c30*s2), int(cy-y*s2+(x+z)*s30*s2*0.5))
        V2 = {k: iso2(*v) for k,v in verts.items()}
        p.setPen(QPen(QColor(60,110,200,30), 1.0))
        for a,b in [('A','B'),('B','C'),('C','D'),('D','A'),
                    ('E','F'),('F','G'),('G','H'),('H','E'),
                    ('A','E'),('B','F'),('C','G'),('D','H')]:
            p.drawLine(V2[a], V2[b])

        # Grid suelo
        p.setPen(QPen(QColor(40,80,180,20), 0.8))
        gy0, gy1 = int(H*0.82), int(H*0.95)
        for i in range(11):
            gx = int(W*0.05 + i*W*0.90/10)
            p.drawLine(gx, gy0, gx, gy1)
        for j in range(5):
            gy = int(gy0 + j*(gy1-gy0)/4)
            p.drawLine(int(W*0.05), gy, int(W*0.95), gy)

        # Textos fantasma
        AH = _AL("AlignHCenter"); AR = _AL("AlignRight")
        p.setFont(QFont("Segoe UI",11,QFont.Weight.Bold if _QT=="PyQt6" else QFont.Bold))
        p.setPen(QPen(QColor(255,255,255,28)))
        p.drawText(QRect(int(W*.05),int(H*.81),int(W*.9),28), AH, "Initial Search Space")
        p.setFont(QFont("Segoe UI",8))
        p.setPen(QPen(QColor(255,255,255,18)))
        p.drawText(QRect(int(W*.05),int(H*.86),int(W*.9),22), AH, "Interval N")
        p.setPen(QPen(QColor(255,255,255,20)))
        p.drawText(QRect(int(W*.30),int(H*.91),int(W*.65),20), AR, "Optimal Solution")
        p.end()


# ══════════════════════════════════════════════════════════════════════════════
#  QSS — Los hijos del panel tienen fondos opacos propios (no transparentes)
#        para que el pintado del padre sea visible.
# ══════════════════════════════════════════════════════════════════════════════
QSS = """
/* Base: sin background para que BgPanel.paintEvent sea visible */
QMainWindow { background: #080e22; }
QWidget      { background: transparent; color: #d8e8f8;
               font-family: 'Segoe UI', Consolas, Arial; font-size: 13px; }

/* Títulos */
QLabel#panelTitle { font-size:19px; font-weight:bold; color:#fff;
    padding:6px 0 2px 0; background:transparent; }
QLabel#dashTitle  { font-size:19px; font-weight:bold; color:#fff;
    padding:6px 0 2px 0; background:transparent; }
QLabel#statusLbl  { color:#5ab4ff; font-size:12px; padding:3px 0;
    background:transparent; }
QLabel { color:#b8cce0; font-size:12px; background:transparent; }

/* Inputs con fondo semiopaco para legibilidad sobre imagen */
QLineEdit {
    background: rgba(10,20,55,230); border:1px solid #2a3e6e;
    border-radius:5px; color:#dce8f8; padding:5px 10px; min-height:26px;
    selection-background-color:#1e4ea0;
}
QLineEdit:focus { border:1.5px solid #3a6ee8; }

QDoubleSpinBox {
    background: rgba(10,20,55,230); border:1px solid #2a3e6e;
    border-radius:5px; color:#dce8f8; padding:3px 8px; min-height:25px;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width:18px; background:rgba(26,44,90,220);
    border:none; border-left:1px solid #2a3e6e;
}

QComboBox {
    background: rgba(10,20,55,230); border:1px solid #2a3e6e;
    border-radius:5px; color:#dce8f8; padding:5px 10px; min-height:26px;
}
QComboBox::drop-down { border:none; width:20px; }
QComboBox QAbstractItemView {
    background:#141e36; color:#dce8f8; border:1px solid #2a3e6e;
    selection-background-color:#1e4ea0; outline:none;
}

QPushButton {
    background:#1a50c0; color:#fff; border:none; border-radius:7px;
    padding:7px 16px; font-weight:bold; font-size:13px; min-height:33px;
}
QPushButton:hover   { background:#2462d8; }
QPushButton:pressed { background:#133a90; }

/* Tabla con fondo semiopaco */
QTableWidget {
    background: rgba(8,16,42,230); color:#c0d4ee;
    gridline-color:#1a2e58; border:1px solid #1a2e58;
    border-radius:4px; alternate-background-color:rgba(14,26,58,210);
    selection-background-color:#1a3e78;
}
QHeaderView::section {
    background: rgba(20,36,86,240); color:#ffffff;
    border:none; border-right:1px solid #243060;
    border-bottom:1px solid #243060;
    padding:5px 6px; font-weight:bold; font-size:12px;
}
QTableWidget::item           { background:transparent; }
QTableWidget::item:alternate { background:rgba(14,26,58,120); }
QTableCornerButton::section  { background:rgba(20,36,86,240); border:none; }

/* Scrollbars */
QScrollBar:vertical   { background:rgba(8,16,42,180); width:8px; }
QScrollBar:horizontal { background:rgba(8,16,42,180); height:8px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background:rgba(50,80,150,200); border-radius:4px; min-height:20px;
}
QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }

/* Frames gráficas */
QFrame#frame2d, QFrame#frame3d {
    background: rgba(8,16,42,210); border:1px solid #1a2e58; border-radius:6px;
}
QSplitter::handle { background:rgba(30,50,100,180); }

/* Menú */
QMenu { background:#141e36; color:#d8e8f8; border:1px solid #2a3e6e;
    border-radius:6px; padding:4px; }
QMenu::item            { padding:6px 20px; border-radius:4px; background:transparent; }
QMenu::item:selected   { background:#1a50c0; }

/* Divisor */
QFrame#hline { background:rgba(80,120,200,90); max-height:1px; border:none; }

/* Sección MD */
QFrame#mdSec {
    background: rgba(16,30,72,190); border:1px solid rgba(60,100,200,140);
    border-radius:6px;
}
"""

# ── Utilidades ────────────────────────────────────────────────────────────────
def _err(parent, title, msg, detail=""):
    b = QMsgBox(parent)
    b.setIcon(QMsgBox.Icon.Critical if _QT=="PyQt6" else QMsgBox.Critical)
    b.setWindowTitle(title); b.setText(msg)
    if detail: b.setDetailedText(detail)
    ok = QMsgBox.StandardButton.Ok if _QT=="PyQt6" else QMsgBox.Ok
    b.setStandardButtons(ok)
    b.exec() if _QT=="PyQt6" else b.exec_()

def _pvars(t): return [v.strip() for v in (t or "").replace(","," ").split() if v.strip()]
def _px0(t):
    t2 = (t or "").replace(";"," ").replace(","," ")
    return [float(x) for x in t2.split() if x.strip()]

def _callable(expr, vnames):
    """Crea callable numpy para una expresión con las variables dadas.
    Normaliza la expresión primero (^→**, ln→log, etc.) y maneja
    funciones especiales de sympy (DiracDelta, Heaviside)."""
    if not _SYMPY: raise RuntimeError("SymPy no disponible.")
    fc = _get_fc()
    norm = fc.normalize_expr(expr) if fc else expr
    # Construir símbolos
    lmap = {v: sp.Symbol(v) for v in vnames}
    f_sym = sp.sympify(norm, locals=lmap)
    modules = [
        {
            "DiracDelta"  : lambda *a: 0.0,
            "Heaviside"   : lambda x, *a: np.heaviside(np.asarray(x, dtype=float), 0.5),
            "conjugate"   : np.conj,
            "re"          : np.real,
            "im"          : np.imag,
        },
        "numpy",
    ]
    return sp.lambdify([lmap[v] for v in vnames], f_sym, modules=modules)

def _e1d(expr, vnames, xs):
    return np.asarray(_callable(expr, vnames)(xs), dtype=float)

def _grid(lo, hi, n=400):
    if lo==hi: lo-=1; hi+=1
    if lo>hi:  lo,hi=hi,lo
    return np.linspace(lo, hi, max(50,int(n)))


# ── Validación numérica ───────────────────────────────────────────────────────
class _NumericalWarning(Exception):
    """Excepción para resultados numéricos problemáticos (inf, nan, overflow)."""
    pass

def _validate_hist(hist: list):
    """Lanza _NumericalWarning si el historial contiene valores problemáticos."""
    if not hist:
        return
    last = hist[-1]
    for key, val in last.items():
        if isinstance(val, float):
            if math.isinf(val):
                raise _NumericalWarning(
                    f"El resultado diverge hacia el infinito en '{key}'.\n"
                    "Verifica la función, el rango o el punto inicial.")
            if math.isnan(val):
                raise _NumericalWarning(
                    f"El cálculo produjo un valor indefinido (NaN) en '{key}'.\n"
                    "La función puede ser discontinua o el punto inicial es inválido.")
        elif isinstance(val, np.ndarray):
            if np.any(np.isinf(val)):
                raise _NumericalWarning(
                    f"El vector '{key}' contiene valores infinitos.\n"
                    "El método divergió. Prueba un punto inicial diferente.")
            if np.any(np.isnan(val)):
                raise _NumericalWarning(
                    f"El vector '{key}' contiene valores NaN.\n"
                    "La función puede no estar definida en ese punto.")
    # Comprobar si el valor óptimo es extremadamente grande
    for kf in ("f_k", "f_lambda", "f_mu"):
        if kf in last and isinstance(last[kf], float):
            if abs(last[kf]) > 1e12:
                raise _NumericalWarning(
                    f"El valor óptimo f* = {last[kf]:.4g} es extremadamente grande.\n"
                    "La función puede no tener mínimo en el rango dado, "
                    "o el punto inicial está muy lejos del óptimo.")


# ══════════════════════════════════════════════════════════════════════════════
#  Ventana principal
# ══════════════════════════════════════════════════════════════════════════════
class InterfazOptimizacion(QMW):

    def __init__(self, resource_path=None, **kwargs):
        super().__init__()

        # ── Inicializar módulos de métodos con resource_path ──────────────────
        rp = str(resource_path) if resource_path else ""
        _init_modules(rp)
        _load_canvases()

        # ── Buscar imagen de fondo ────────────────────────────────────────────
        extra_bg = []
        if rp:
            extra_bg = [os.path.join(rp, "Imgen_de_fondo.jpg"),
                        os.path.join(rp, "Imgen_de_fondo.png")]
        bg = _find_bg(extra_bg)
        print(f"[fondo] path='{bg}' existe={os.path.exists(bg) if bg else False}")

        # ── Estado interno ────────────────────────────────────────────────────
        self._hist: List[Dict[str,Any]] = []
        self._metodo = self._fx = ""
        self._vars: List[str]   = []
        self._x0:   List[float] = []
        self._lo: float = 0.0;  self._hi: float = 5.0
        self._gx: str   = "";   self._tol_L: float = 0.1
        self._click_cid = None   # matplotlib event connection id
        # Historial de sesión: lista de dicts con datos de cada ejecución
        self._session: List[Dict[str,Any]] = []
        # Métodos ejecutados en la sesión actual (para botón "Comparar" del AI)
        self._session_methods: List[str] = []
        # Historial persistente de funciones y métodos usados (no se borra con Limpiar)
        self._history_log: List[Dict[str,Any]] = []


        # ── Ventana ───────────────────────────────────────────────────────────
        self.setWindowTitle("Optimizador de Funciones")
        self.setMinimumSize(1120, 700)
        self.setStyleSheet(QSS)

        central = QW(); self.setCentralWidget(central)
        root = QHBox(central)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # Panel izquierdo con scroll (los campos MD no se cortan)
        self.left = LeftPanel(bg)
        self.left.setFixedWidth(375)

        self._left_inner = QW()
        self._left_inner.setStyleSheet("background: transparent;")
        self._LL = QVBox(self._left_inner)
        self._LL.setContentsMargins(22,14,22,14); self._LL.setSpacing(7)

        self._left_scroll = QScroll()
        self._left_scroll.setWidget(self._left_inner)
        self._left_scroll.setWidgetResizable(True)
        self._left_scroll.setFrameShape(QFrame.Shape.NoFrame if _QT=="PyQt6" else QFrame.NoFrame)
        self._left_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff if _QT=="PyQt6"
            else QtCore.Qt.ScrollBarAlwaysOff)
        self._left_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded if _QT=="PyQt6"
            else QtCore.Qt.ScrollBarAsNeeded)
        self._left_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { background: rgba(10,20,60,120); width: 6px;"
            "  border-radius: 3px; margin: 0; }"
            "QScrollBar::handle:vertical { background: rgba(60,120,220,180);"
            "  border-radius: 3px; min-height: 20px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }")

        _left_outer_layout = QVBox(self.left)
        _left_outer_layout.setContentsMargins(0,0,0,0)
        _left_outer_layout.addWidget(self._left_scroll)

        # Separador vertical
        div = QFrame()
        div.setFrameShape(QFrame.Shape.VLine if _QT=="PyQt6" else QFrame.VLine)
        div.setFixedWidth(1)
        div.setStyleSheet("background:rgba(60,100,200,120); border:none;")

        # Panel derecho
        self.right = BgPanel(bg, overlay_alpha=0)
        self._RL = QVBox(self.right)
        self._RL.setContentsMargins(16,14,16,14); self._RL.setSpacing(10)

        root.addWidget(self.left)
        root.addWidget(div)
        root.addWidget(self.right, 1)

        # ── Panel de IA (se muestra/oculta con btn_ai) ────────────────────────
        self.ai_panel = None
        AICls = _get_ai_panel_cls()
        if AICls:
            try:
                self.ai_panel = AICls(central)
                self.ai_panel.setVisible(False)
                root.addWidget(self.ai_panel)
            except Exception as _ai_err:
                print(f"[AI] No se pudo inicializar el panel: {_ai_err}")
                self.ai_panel = None

        self._build_left()
        self._build_right()
        self._wire()
        self._on_method()
        self._reset()

    # ── BUILD LEFT ────────────────────────────────────────────────────────────
    def _build_left(self):
        L = self._LL

        t = QLbl("Panel de control"); t.setObjectName("panelTitle")
        L.addWidget(t); L.addWidget(self._hl())

        # 1 — Método (primero)
        L.addWidget(QLbl("Método:"))
        self.cmb = QCmb()
        _populate_grouped_combo(self.cmb)
        L.addWidget(self.cmb); L.addSpacing(3)

        # 2 — Función
        L.addWidget(QLbl("Función f(x):"))
        self.edt_fx = SpanishLineEdit(); self.edt_fx.setPlaceholderText("Ej: -(x-3)**2 + 10")
        L.addWidget(self.edt_fx)

        # 3 — Rango
        L.addSpacing(3)
        rng = QW(); rl = QGrid(rng)
        rl.setContentsMargins(0,0,0,0); rl.setSpacing(5)
        self.smin = QDSpin(); self.smax = QDSpin()
        for s in (self.smin,self.smax): s.setDecimals(6); s.setRange(-1e9,1e9)
        self.smin.setValue(0.0); self.smax.setValue(5.0)
        rl.addWidget(QLbl("mínimo:"),0,0); rl.addWidget(self.smin,0,1)
        rl.addWidget(QLbl("máximo:"),1,0); rl.addWidget(self.smax,1,1)
        L.addWidget(rng)

        # 4 — Tolerancia L (solo Fibonacci)
        L.addSpacing(3)
        self.row_tol = QW(); rt = QHBox(self.row_tol)
        rt.setContentsMargins(0,0,0,0); rt.setSpacing(6)
        rt.addWidget(QLbl("Tolerancia L:"))
        self.edt_tol = SpanishLineEdit(); self.edt_tol.setPlaceholderText("Ej: 0.1")
        self.edt_tol.setText("0.1"); rt.addWidget(self.edt_tol)
        L.addWidget(self.row_tol); self.row_tol.setVisible(False)

        # 4b — Alpha α (solo métodos multiobjetivo con escalarización)
        L.addSpacing(3)
        self.row_alpha = QW(); ra = QHBox(self.row_alpha)
        ra.setContentsMargins(0,0,0,0); ra.setSpacing(6)
        ra.addWidget(QLbl("Peso α ∈ [0,1]:"))
        self.edt_alpha = QDSpin()
        self.edt_alpha.setDecimals(4); self.edt_alpha.setRange(0.0, 1.0)
        self.edt_alpha.setSingleStep(0.05); self.edt_alpha.setValue(0.5)
        self.edt_alpha.setToolTip(
            "α controla el peso de cada objetivo.\n"
            "α=0 → minimiza solo f2=(x-2)²\n"
            "α=1 → minimiza solo f1=x²\n"
            "α=0.5 → balance igual entre ambos objetivos")
        ra.addWidget(self.edt_alpha)
        L.addWidget(self.row_alpha); self.row_alpha.setVisible(False)

        # 4c — Punto inicial x0 para métodos MO generales (solo 1 valor)
        self.row_x0_mo = QW(); rx0 = QHBox(self.row_x0_mo)
        rx0.setContentsMargins(0,0,0,0); rx0.setSpacing(6)
        rx0.addWidget(QLbl("x₀ inicial:"))
        self.edt_x0_mo = SpanishLineEdit(); self.edt_x0_mo.setPlaceholderText("Ej: 1.0")
        self.edt_x0_mo.setText("1.0")
        rx0.addWidget(self.edt_x0_mo)
        L.addWidget(self.row_x0_mo); self.row_x0_mo.setVisible(False)
        L.addSpacing(3)
        self.md_sec = QFrame(); self.md_sec.setObjectName("mdSec")
        ml = QVBox(self.md_sec); ml.setContentsMargins(8,6,8,6); ml.setSpacing(5)
        self.lbl_gx = QLbl("Restricción g(x):")
        ml.addWidget(self.lbl_gx)
        self.edt_gx = SpanishLineEdit(); self.edt_gx.setPlaceholderText("Ej: x1**2+x2**2-1  (opcional)")
        ml.addWidget(self.edt_gx)
        ml.addWidget(QLbl("Variables:"))
        self.edt_vars = SpanishLineEdit()
        self.edt_vars.setPlaceholderText("Ej: x1 x2  (opcional: se autodetectan)")
        ml.addWidget(self.edt_vars)
        ml.addWidget(QLbl("Punto inicial x0:"))
        self.edt_x0 = SpanishLineEdit(); self.edt_x0.setPlaceholderText("Ej: 0.5, 0.5  (opcional)")
        ml.addWidget(self.edt_x0)
        L.addWidget(self.md_sec); self.md_sec.setVisible(False)

        # 6 — Botones (en recuadro con borde rojo-acento)
        L.addSpacing(8)
        self._btn_frame = QFrame()
        self._btn_frame.setObjectName("btnFrame")
        self._btn_frame.setStyleSheet(
            "QFrame#btnFrame {"
            "  background: rgba(10,18,50,180);"
            "  border: 1px solid rgba(30,110,232,180);"
            "  border-radius: 12px;"
            "  padding: 6px;"
            "}")
        bf = QVBox(self._btn_frame)
        bf.setContentsMargins(8, 6, 8, 6); bf.setSpacing(6)

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

        r1 = QW(); b1 = QHBox(r1); b1.setContentsMargins(0,0,0,0); b1.setSpacing(8)
        self.btn_calc  = QBtn("Calcular");  self.btn_calc.setStyleSheet(_BTN_STYLE)
        self.btn_clear = QBtn("Limpiar");   self.btn_clear.setStyleSheet(_BTN_STYLE)
        b1.addWidget(self.btn_calc); b1.addWidget(self.btn_clear)
        bf.addWidget(r1)

        r2 = QW(); b2 = QHBox(r2); b2.setContentsMargins(0,0,0,0); b2.setSpacing(8)
        self.btn_export = QBtn("Exportar  ▾"); self.btn_export.setStyleSheet(_BTN_STYLE)
        self.btn_exit   = QBtn("Salir");       self.btn_exit.setStyleSheet(_BTN_STYLE)
        b2.addWidget(self.btn_export); b2.addWidget(self.btn_exit)
        bf.addWidget(r2)

        L.addWidget(self._btn_frame)

        # Botón Asistente IA
        L.addSpacing(4)
        self.btn_ai = QBtn("🤖  Asistente de IA")
        self.btn_ai.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #1a3a8c, stop:1 #0e2060);"
            " color:#a8d0ff; border:1px solid #2a4ea0; border-radius:7px;"
            " font-weight:bold; font-size:12px; min-height:33px; }"
            "QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #2a50c0, stop:1 #1a3898); color:#d0e8ff; }"
            "QPushButton:checked { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #0e4090, stop:1 #081a50); color:#3cff8a;"
            " border:1px solid #3cff8a; }")
        self.btn_ai.setCheckable(True)
        L.addWidget(self.btn_ai)

        # Botón Historial
        L.addSpacing(4)
        self.btn_hist = QBtn("📋  Historial de funciones")
        self.btn_hist.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #1a3a8c, stop:1 #0e2060);"
            " color:#a8d0ff; border:1px solid #2a4ea0; border-radius:7px;"
            " font-weight:bold; font-size:12px; min-height:33px; }"
            "QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #2a50c0, stop:1 #1a3898); color:#d0e8ff; }"
        )
        L.addWidget(self.btn_hist)

        L.addSpacing(5)
        self.lbl_st = QLbl(""); self.lbl_st.setObjectName("statusLbl")
        self.lbl_st.setWordWrap(True)
        self.lbl_st.setVisible(True)
        self.lbl_st.setStyleSheet(
            "color: #00e5ff; font-size: 12px; font-weight: bold;"
            "background: rgba(8,20,55,180); border-radius: 4px; padding: 4px 6px;"
        )
        L.addWidget(self.lbl_st); L.addStretch(1)

    def _hl(self):
        f = QFrame(); f.setObjectName("hline")
        f.setFrameShape(QFrame.Shape.HLine if _QT=="PyQt6" else QFrame.HLine)
        f.setFixedHeight(1); return f

    # ── BUILD RIGHT ───────────────────────────────────────────────────────────
    def _build_right(self):
        R = self._RL
        t = QLbl("Dashboard"); t.setObjectName("dashTitle"); R.addWidget(t)

        # ── Splitter principal vertical: tabla / 2D / 3D ─────────────────────
        main_sp = QSplit(Qt.Orientation.Vertical if _QT=="PyQt6" else Qt.Vertical)
        main_sp.setChildrenCollapsible(True)
        main_sp.setHandleWidth(6)
        main_sp.setStyleSheet(
            "QSplitter::handle { background: rgba(60,100,200,160); border-radius:3px; }"
            "QSplitter::handle:hover { background: rgba(80,140,255,220); }"
        )

        # -- Tabla --
        tbl_wrap = QFrame(); tbl_wrap.setObjectName("frame2d")
        tw_l = QVBox(tbl_wrap); tw_l.setContentsMargins(4,4,4,4)
        self.table = QTbl(0,0)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)   # Ocultar numeración duplicada
        tw_l.addWidget(self.table)

        # -- Frame 2D --
        self.frame_2d = QFrame(); self.frame_2d.setObjectName("frame2d")
        f2l = QVBox(self.frame_2d); f2l.setContentsMargins(4,4,4,4)
        self.lbl_2d = QLbl("Ejecute un método para ver la gráfica 2D.")
        self.lbl_2d.setAlignment(_AL("AlignCenter")); f2l.addWidget(self.lbl_2d)

        # -- Frame 3D --
        self.frame_3d = QFrame(); self.frame_3d.setObjectName("frame3d")
        f3l = QVBox(self.frame_3d); f3l.setContentsMargins(4,4,4,4)
        self.lbl_3d = QLbl("Ejecute un método para ver la gráfica 3D.")
        self.lbl_3d.setAlignment(_AL("AlignCenter")); f3l.addWidget(self.lbl_3d)

        main_sp.addWidget(tbl_wrap)
        main_sp.addWidget(self.frame_2d)
        main_sp.addWidget(self.frame_3d)
        main_sp.setSizes([260, 280, 280])
        R.addWidget(main_sp, 1)

        self.canvas2d = self.canvas3d = None
        self._toolbar2d = None          # NavigationToolbar oculta para zoom/pan/home
        if Function2DCanvas:
            try:
                self.canvas2d = Function2DCanvas(self.frame_2d)
                self.frame_2d.layout().addWidget(self.canvas2d)
                self.canvas2d.setVisible(False)
                self.canvas2d.setToolTip("💡 Haz clic en la gráfica para recalcular desde ese punto")
                # Crear toolbar oculta (necesaria para zoom/pan/home)
                if _NavToolbar is not None:
                    self._toolbar2d = _NavToolbar(self.canvas2d, self.frame_2d)
                    self._toolbar2d.setVisible(False)   # invisible, solo para funcionalidad
            except: pass
        if Rotating3DCanvas:
            try:
                self.canvas3d = Rotating3DCanvas(self.frame_3d)
                self.frame_3d.layout().addWidget(self.canvas3d)
                self.canvas3d.setVisible(False)
            except: pass

    # ── WIRE ──────────────────────────────────────────────────────────────────
    def _wire(self):
        self.btn_calc.clicked.connect(self.ejecutar)
        self.btn_clear.clicked.connect(self.limpiar)
        self.btn_exit.clicked.connect(self.salir)
        self.btn_export.clicked.connect(self._exp_menu)
        self.cmb.currentTextChanged.connect(self._on_method)
        self.btn_ai.toggled.connect(self._toggle_ai_panel)
        self.btn_hist.clicked.connect(self._show_historial)

    def _toggle_ai_panel(self, checked: bool):
        """Muestra u oculta el panel del asistente de IA."""
        if self.ai_panel:
            self.ai_panel.setVisible(checked)
            # Si se abre el panel Y ya hay un cálculo previo → explicarlo
            if checked and self._hist and self._metodo:
                self._update_ai_context()
        elif checked:
            self.btn_ai.setChecked(False)
            QMsgBox.information(self, "Asistente de IA",
                "El módulo 'ai_assistant.py' no se encontró en la carpeta del proyecto.\n"
                "Asegúrate de colocar ai_assistant.py junto a interfaz_qt.py.")

    def _update_ai_context(self):
        """
        Actualiza el contexto del asistente y dispara la explicación
        automática del cálculo recién ejecutado.
        """
        if not self.ai_panel:
            return
        try:
            # 1. Actualizar contexto (siempre, aunque el panel no sea visible)
            self.ai_panel.update_context(
                metodo         = self._metodo,
                fx             = self._fx,
                result_summary = self._rmsg(self._hist) if self._hist else "",
                n_iters        = len(self._hist),
                vars_          = self._vars,
                x0             = self._x0,
                lo             = self._lo,
                hi             = self._hi,
            )

            # 2. Registrar método en la sesión para el botón "Comparar métodos"
            if self._metodo and self._metodo not in self._session_methods:
                self._session_methods.append(self._metodo)
            if hasattr(self.ai_panel, 'update_session_methods'):
                self.ai_panel.update_session_methods(self._session_methods)

            # 3. Si el panel está visible Y hay API Key → explicación automática
            if self.ai_panel.isVisible() and self.ai_panel._api_key:
                self.ai_panel.explain_calculation(
                    metodo = self._metodo,
                    fx     = self._fx,
                    hist   = self._hist,
                    vars_  = self._vars,
                    x0     = self._x0,
                    lo     = self._lo,
                    hi     = self._hi,
                    gx     = self._gx,
                )
        except Exception:
            pass

    def _on_method(self):
        m = self.cmb.currentText()
        is_md    = _is_md(m)
        is_nd    = _needs_x0(m)
        is_mo    = _is_multiobj(m)
        is_alpha = _needs_alpha(m)

        show_panel = is_md or is_nd
        self.md_sec.setVisible(show_panel)
        self.lbl_gx.setVisible(is_md)
        self.edt_gx.setVisible(is_md)
        self.row_tol.setVisible(_is_fib(m))

        # Controles α y x0 solo para métodos MO con escalarización
        self.row_alpha.setVisible(is_alpha)
        self.row_x0_mo.setVisible(is_mo)

        # Placeholder del campo función
        if is_mo:
            self.edt_fx.setEnabled(True)
            self.edt_fx.setPlaceholderText(
                "Ingresa f(x) mono-objetivo  →  se convierte en problema MO\n"
                "Ej: x**2    o    (x-3)**2 + 1    o    x**2 - 4*x + 3")
        else:
            self.edt_fx.setEnabled(True)
            self.edt_fx.setPlaceholderText("Ej: -(x-3)**2 + 10")

        if _is_meta(m) or m in ("PSO: Enjambre", "GA: Algoritmo Genético"):
            self.edt_x0.setPlaceholderText("Opcional (el método genera población aleatoria)")
        elif _is_heu(m) or _is_ls(m):
            self.edt_x0.setPlaceholderText("Ej: 1.0  o  0.5, 0.5")

    # ── RESET ─────────────────────────────────────────────────────────────────
    def _reset(self):
        self.table.setRowCount(0); self.table.setColumnCount(0)
        self.lbl_2d.setVisible(True); self.lbl_3d.setVisible(True)
        for c in (self.canvas2d, self.canvas3d):
            if c:
                try: c.clear()
                except: pass
                c.setVisible(False)

    # ── LEYENDA COLAPSABLE ────────────────────────────────────────────────────
    def _add_collapsible_legend(self, canvas, ax):
        """
        Instala el overlay de redimensionado estilo Excel sobre la leyenda.
        Los handles aparecen solo al hacer clic en la leyenda y se ocultan
        al hacer clic fuera de ella.
        """
        # Limpiar overlay anterior si existe
        overlay_attr = f"_leg_resize_{id(canvas)}"
        old_ov = getattr(self, overlay_attr, None)
        if old_ov is not None:
            try: old_ov.deleteLater()
            except: pass

        leg = ax.get_legend()
        if leg is None:
            return

        try:
            overlay = LegendResizeOverlay(canvas, ax)
            # Refrescar handles cada vez que matplotlib redibuje
            canvas.fig.canvas.mpl_connect(
                "draw_event", lambda _ev, ov=overlay: ov.refresh())
            setattr(self, overlay_attr, overlay)
        except Exception:
            pass

    # ── CLICK EN GRÁFICA → RE-CALCULAR / MENÚ CONTEXTUAL ────────────────────
    def _connect_click(self):
        """
        Conecta eventos de clic en el canvas 2D:
          · Clic izquierdo  → re-ejecutar método desde el punto seleccionado
          · Clic derecho    → menú contextual en español
        También activa la leyenda arrastrable.
        """
        if not self.canvas2d or not hasattr(self.canvas2d, "fig"):
            return

        # ── Desconectar evento anterior ──────────────────────────────────────
        if self._click_cid is not None:
            try: self.canvas2d.fig.canvas.mpl_disconnect(self._click_cid)
            except: pass
            self._click_cid = None

        # ── Hacer la leyenda 2D arrastrable ──────────────────────────────────
        try:
            fig = self.canvas2d.fig
            for ax in fig.axes:
                leg = ax.get_legend()
                if leg is not None:
                    leg.set_draggable(True, use_blit=False)
        except Exception:
            pass

        # ── Helper: ¿el evento cae sobre la leyenda del canvas 2D? ────────────
        def _click_on_legend(ev):
            try:
                fig2 = self.canvas2d.fig
                if not fig2.axes:
                    return False
                leg = fig2.axes[0].get_legend()
                if leg is None or not leg.get_visible():
                    return False
                renderer = self.canvas2d.fig.canvas.get_renderer()
                bb = leg.get_window_extent(renderer)
                return bb.contains(ev.x, ev.y)
            except Exception:
                return False

        # ── Manejador principal de eventos ───────────────────────────────────
        def _on_click(event):
            # Si el clic (izq o der) cae sobre la leyenda → el overlay lo maneja
            if _click_on_legend(event):
                return

            # ── Clic DERECHO fuera de la leyenda → menú contextual de la gráfica
            if event.button == 3:
                menu = QMenu(self.canvas2d)
                menu.setStyleSheet(
                    "QMenu{background:#141e36;color:#d8e8f8;border:1px solid #2a3e6e;"
                    "border-radius:6px;padding:4px;font-size:13px;}"
                    "QMenu::item{padding:6px 22px;border-radius:4px;}"
                    "QMenu::item:selected{background:#1a50c0;}"
                )
                # Acciones
                act_home   = menu.addAction("🏠  Restablecer gráfica")
                act_zoom   = menu.addAction("🔍  Zoom +")
                act_zoomout= menu.addAction("🔎  Zoom −")
                act_pan    = menu.addAction("✋  Mover gráfica (pan)")
                menu.addSeparator()
                act_save   = menu.addAction("💾  Guardar imagen…")
                act_copy   = menu.addAction("📋  Copiar imagen al portapapeles")
                menu.addSeparator()
                act_grid   = menu.addAction("⊞   Alternar cuadrícula")
                menu.addSeparator()
                act_calc   = menu.addAction("📍  Calcular desde este punto")
                act_info   = menu.addAction("ℹ️   Coordenadas del cursor")

                # Calcular posición global desde el widget
                try:
                    gpos = self.canvas2d.mapToGlobal(
                        QtCore.QPoint(int(event.x), int(self.canvas2d.height() - event.y)))
                except Exception:
                    gpos = QtGui.QCursor.pos()

                chosen = menu.exec(gpos) if _QT == "PyQt6" else menu.exec_(gpos)
                if chosen is None:
                    return

                try:
                    tb   = getattr(self, "_toolbar2d", None)   # toolbar oculta
                    fig2 = self.canvas2d.fig
                    ax2  = fig2.axes[0] if fig2.axes else None

                    if chosen == act_home:
                        if tb:
                            try:
                                tb.home()
                            except Exception:
                                for ax_ in fig2.axes:
                                    ax_.autoscale()
                                self.canvas2d.draw_idle()
                        else:
                            for ax_ in fig2.axes:
                                ax_.autoscale()
                            self.canvas2d.draw_idle()

                    elif chosen == act_zoom:
                        if tb:
                            try:
                                tb.zoom()
                                self.lbl_st.setText("🔍 Zoom + activo — dibuja un rectángulo para acercar.")
                            except Exception as _ze:
                                self.lbl_st.setText(f"⚠ Zoom no disponible: {_ze}")
                        else:
                            self.lbl_st.setText("⚠ Zoom no disponible sin NavigationToolbar.")

                    elif chosen == act_zoomout:
                        if ax2:
                            try:
                                xl = ax2.get_xlim(); yl = ax2.get_ylim()
                                cx = (xl[0]+xl[1])/2; cy = (yl[0]+yl[1])/2
                                rx = (xl[1]-xl[0])*0.65; ry = (yl[1]-yl[0])*0.65
                                ax2.set_xlim(cx-rx, cx+rx); ax2.set_ylim(cy-ry, cy+ry)
                                self.canvas2d.draw_idle()
                                self.lbl_st.setText("🔎 Zoom − aplicado.")
                            except Exception as _ze:
                                self.lbl_st.setText(f"⚠ Zoom − error: {_ze}")
                        else:
                            self.lbl_st.setText("⚠ Zoom − no disponible.")

                    elif chosen == act_pan:
                        if tb:
                            try:
                                tb.pan()
                                self.lbl_st.setText("✋ Modo pan activo — arrastra la gráfica con el ratón.")
                            except Exception as _pe:
                                self.lbl_st.setText(f"⚠ Pan no disponible: {_pe}")
                        else:
                            self.lbl_st.setText("⚠ Pan no disponible sin NavigationToolbar.")

                    elif chosen == act_save:
                        p, _ = QFD.getSaveFileName(
                            self, "Guardar imagen", "grafica_2d.png",
                            "Imágenes PNG (*.png);;Imágenes SVG (*.svg);;PDF (*.pdf)")
                        if p:
                            fig2.savefig(p, dpi=150, bbox_inches="tight",
                                         facecolor="#0a1228")
                            QMsgBox.information(self, "Imagen guardada",
                                f"✅ La imagen se guardó correctamente en:\n{p}")

                    elif chosen == act_copy:
                        import io
                        buf = io.BytesIO()
                        fig2.savefig(buf, format="png", dpi=130,
                                     bbox_inches="tight", facecolor="#0a1228")
                        buf.seek(0)
                        img = QtGui.QImage.fromData(buf.read())
                        QtWidgets.QApplication.clipboard().setImage(img)
                        self.lbl_st.setText("📋 Imagen copiada al portapapeles.")

                    elif chosen == act_grid:
                        if ax2:
                            ax2.grid(not ax2.xaxis.get_gridlines()[0].get_visible()
                                     if ax2.xaxis.get_gridlines() else True,
                                     color="#1a2e58", alpha=0.4, linewidth=0.6)
                            self.canvas2d.draw_idle()

                    elif chosen == act_calc:
                        if hasattr(event,"xdata") and event.xdata is not None:
                            xc = float(event.xdata)
                            lo_, hi_ = self._lo, self._hi
                            xc = max(lo_, min(hi_, xc))
                            if self._metodo and self._fx:
                                try:
                                    self.lbl_st.setText(f"⏳ Calculando desde x = {xc:.4f}…")
                                    QApp.processEvents()
                                    if _is_md(self._metodo) or _is_ls(self._metodo):
                                        new_x0 = list(self._x0)
                                        if new_x0: new_x0[0] = xc
                                        else: new_x0 = [xc, xc]
                                        self.edt_x0.setText(", ".join(f"{v:.6g}" for v in new_x0))
                                        x0_ = new_x0
                                    else:
                                        self.smin.setValue(xc); lo_ = xc; x0_ = [xc]
                                    hist_ = self._run(self._metodo, self._fx, self._gx,
                                                      self._vars, x0_, lo_, hi_, self._tol_L)
                                    _validate_hist(hist_)
                                    self._hist = hist_; self._x0 = x0_; self._lo = lo_
                                    self._render_table(hist_)
                                    self._render_plots(self._metodo, self._fx, self._vars,
                                                       hist_, x0_, lo_, hi_)
                                    self._connect_click()
                                    self._session.append({
                                        "metodo": self._metodo, "fx": self._fx, "gx": self._gx,
                                        "vars": list(self._vars), "x0": list(x0_),
                                        "lo": lo_, "hi": hi_, "hist": hist_,
                                        "traj": self._traj(self._vars, hist_,
                                                           x0_ if (_is_md(self._metodo) or _is_ls(self._metodo))
                                                              else [lo_]),
                                    })
                                    self._log_history(self._metodo, self._fx, hist_)
                                    self.lbl_st.setText(f"✓ Calculado desde x={xc:.4f} → {self._rmsg(hist_)}")
                                except _NumericalWarning as nw:
                                    self.lbl_st.setText(f"⚠ {nw}")
                                except Exception as ex_:
                                    self.lbl_st.setText(f"✗ Error: {ex_}")
                            else:
                                self.lbl_st.setText("⚠ No hay método activo. Ejecuta un cálculo primero.")
                        else:
                            self.lbl_st.setText("ℹ️ Haz clic dentro del área de la gráfica.")

                    elif chosen == act_info:
                        if hasattr(event,"xdata") and event.xdata is not None and event.ydata is not None:
                            QMsgBox.information(self, "Coordenadas",
                                f"x = {event.xdata:.6g}\nf(x) ≈ {event.ydata:.6g}")
                        else:
                            QMsgBox.information(self, "Coordenadas",
                                "Haz clic dentro del área de la gráfica para ver coordenadas.")
                except Exception as ex:
                    self.lbl_st.setText(f"⚠ Menú: {ex}")
                return

            # ── Clic IZQUIERDO → recalcular desde nuevo x₀ ──────────────────
            if event.button != 1:
                return
            if event.inaxes is None:
                return
            xc = float(event.xdata)
            lo, hi = self._lo, self._hi
            xc = max(lo, min(hi, xc))
            if not self._metodo or not self._fx:
                return
            try:
                self.lbl_st.setText(f"⏳ Recalculando desde x₀ = {xc:.4f}…")
                QApp.processEvents()

                if _is_md(self._metodo) or _is_ls(self._metodo):
                    new_x0 = list(self._x0)
                    if new_x0: new_x0[0] = xc
                    else: new_x0 = [xc, xc]
                    self.edt_x0.setText(", ".join(f"{v:.6g}" for v in new_x0))
                    x0 = new_x0
                else:
                    self.smin.setValue(xc)
                    lo = xc
                    x0 = [xc]

                gx    = self._gx
                vars_ = self._vars
                tol_L = self._tol_L

                hist = self._run(self._metodo, self._fx, gx, vars_,
                                 x0, lo, hi, tol_L)

                # Validar resultado numérico
                _validate_hist(hist)

                self._hist = hist
                self._x0   = x0
                self._lo   = lo

                self._render_table(hist)
                self._render_plots(self._metodo, self._fx, vars_, hist,
                                   x0 if (_is_md(self._metodo) or _is_ls(self._metodo)) else [lo],
                                   lo, hi)
                self._connect_click()

                self._session.append({
                    "metodo": self._metodo, "fx": self._fx, "gx": gx,
                    "vars": list(vars_), "x0": list(x0),
                    "lo": lo, "hi": hi, "hist": hist,
                    "traj": self._traj(vars_, hist,
                                       x0 if (_is_md(self._metodo) or _is_ls(self._metodo))
                                          else [lo]),
                })
                self._log_history(self._metodo, self._fx, hist)
                self.lbl_st.setText(
                    f"✓ {self._metodo} (x₀={xc:.4f}) → {self._rmsg(hist)}")
            except _NumericalWarning as nw:
                QMsgBox.warning(self, "⚠ Advertencia numérica", str(nw))
                self.lbl_st.setText(f"⚠ {nw}")
                # Mostrar tabla y gráfica incluso con advertencia numérica
                try:
                    self._hist = hist
                    self._render_table(hist)
                    self._render_plots(self._metodo, self._fx, self._vars, hist,
                                       x0 if _is_md(self._metodo) else [lo], lo, hi)
                    self._connect_click()
                except Exception:
                    pass
            except Exception as e:
                self.lbl_st.setText(f"✗ Error al recalcular: {e}")

        self._click_cid = self.canvas2d.fig.canvas.mpl_connect(
            "button_press_event", _on_click)

        # Cursor en cruz para indicar interactividad
        try:
            self.canvas2d.setCursor(QtGui.QCursor(
                Qt.CursorShape.CrossCursor if _QT == "PyQt6" else Qt.CrossCursor))
        except Exception:
            pass

    # ── CALCULAR ──────────────────────────────────────────────────────────────
    def ejecutar(self):
        try:
            m = self.cmb.currentText().strip()
            if not m:
                QMsgBox.warning(self, "⚠ Dato faltante",
                    "Por favor selecciona un método de optimización antes de calcular.")
                return

            fx = self.edt_fx.text().strip()
            if not fx and m not in _MO_SCHAFFER_FIXED:
                QMsgBox.warning(self, "⚠ Dato faltante",
                    "Por favor ingresa la función f(x).\nEjemplo: -(x-3)**2 + 10")
                return
            # Para métodos Schaffer fijos, definir función interna
            if m in _MO_SCHAFFER_FIXED:
                fx = "schaffer_interno"

            lo, hi = self.smin.value(), self.smax.value()
            if lo >= hi:
                QMsgBox.warning(self, "⚠ Rango inválido",
                    f"El valor mínimo ({lo}) debe ser menor que el máximo ({hi}).\n"
                    "Corrige el rango de búsqueda.")
                return

            # Verificar que la función sea evaluable en el rango
            if _SYMPY and not _is_multiobj(m):
                try:
                    fc = _get_fc()
                    if fc:
                        _norm_fx = fc.normalize_expr(fx)
                        # SIEMPRE detectar variables desde la expresión — no desde el campo UI.
                        # El campo de variables solo se usa en _run para indicar el orden de vars
                        # en métodos MD; para la validación lo que importa es que la expresión
                        # sea sintácticamente correcta y evaluable con sus propias variables.
                        _vars_det = fc.detect_variables(_norm_fx)
                        if not _vars_det:
                            _vars_det = ["x"]
                        # Punto de prueba: centro del rango para la primera variable,
                        # 0.0 para las demás (evita puntos donde f puede ser indefinida)
                        mid = (lo + hi) / 2.0
                        _test_vals = [mid] + [0.0] * (len(_vars_det) - 1)
                        _test_y = fc.safe_test_point(_norm_fx, _vars_det, _test_vals)
                    else:
                        # Fallback sin func_compat
                        _test_x = (lo + hi) / 2
                        _xs = sp.Symbol("x")
                        _fl = sp.lambdify(_xs, sp.sympify(fx, locals={"x": _xs}), modules=["numpy"])
                        _test_y = float(_fl(_test_x))

                    if not (isinstance(_test_y, (int, float)) and
                            not math.isinf(_test_y) and not math.isnan(_test_y)):
                        if math.isinf(_test_y):
                            QMsgBox.warning(self, "⚠ Función divergente",
                                f"La función tiende a infinito en el punto de prueba.\n"
                                "Verifica la expresión o ajusta el rango.")
                            return
                        if math.isnan(_test_y):
                            QMsgBox.warning(self, "⚠ Función indefinida",
                                f"La función no está definida en el punto de prueba.\n"
                                "Verifica la expresión o ajusta el rango.")
                            return
                except ValueError as eval_err:
                    # Error de sintaxis real — mostrar mensaje claro
                    QMsgBox.warning(self, "⚠ Error en la función",
                        f"No se puede evaluar  f(x) = {fx}\n\n"
                        f"Causa: {eval_err}\n\n"
                        "Revisa la sintaxis. Ejemplos correctos:\n"
                        "  • x**2 - 4*x + 5\n"
                        "  • (1-x)**2 + 100*(y - x**2)**2\n"
                        "  • cos(x)*exp(-x**2) + sin(y)\n"
                        "  • 2*(x1-3)**2 + x1*x2**3")
                    return
                except Exception:
                    # No bloquear el cálculo por errores de validación inesperados
                    pass

            gx    = self.edt_gx.text().strip()
            vars_ = _pvars(self.edt_vars.text()) or ["x1","x2"]
            x0    = _px0(self.edt_x0.text()) or [0.0,0.0]
            tol_L = 0.1
            if _is_fib(m):
                try:
                    tol_L = float(self.edt_tol.text().strip())
                    if tol_L <= 0:
                        QMsgBox.warning(self, "⚠ Tolerancia inválida",
                            "La tolerancia L debe ser un número positivo mayor que cero.")
                        return
                except ValueError:
                    QMsgBox.warning(self, "⚠ Tolerancia inválida",
                        "El valor de tolerancia L no es un número válido.\nEjemplo: 0.1")
                    return

            self.lbl_st.setText("⏳ Calculando…"); QApp.processEvents()

            hist = self._run(m, fx, gx, vars_, x0, lo, hi, tol_L)

            # Validar resultados numéricos
            _validate_hist(hist)

            self._hist=hist; self._metodo=m; self._fx=fx
            # Detectar variables correctamente para gráficas
            if _is_md(m):
                # Usar las mismas variables reconciliadas que usó _run
                fc2 = _get_fc()
                if fc2:
                    _fx_vars2 = fc2.detect_variables(fc2.normalize_expr(fx))
                    _gx_vars2 = fc2.detect_variables(fc2.normalize_expr(gx)) if gx else []
                    _ui_vars2 = [v for v in vars_ if v.strip()]
                    _merged2  = list(_fx_vars2)
                    for v in _gx_vars2 + _ui_vars2:
                        if v not in _merged2: _merged2.append(v)
                    self._vars = _merged2 if _merged2 else (_ui_vars2 or ["x1", "x2"])
                else:
                    self._vars = vars_
            elif _is_1d_only(m):
                # Métodos estrictamente 1D — usar la única variable de la expresión
                fc2 = _get_fc()
                self._vars = (fc2.detect_variables(fc2.normalize_expr(fx)) or ["x"])[:1] if fc2 else ["x"]
            else:
                # Armijo, Wolfe, heurísticos, metaheurísticos — auto-detectar desde expr
                fc2 = _get_fc()
                if fc2:
                    self._vars = fc2.detect_variables(fc2.normalize_expr(fx)) or ["x"]
                else:
                    self._vars = ["x"]
            self._x0   = x0
            self._lo   = lo;  self._hi  = hi
            self._gx   = gx;  self._tol_L = tol_L

            self._render_table(hist)
            self._render_plots(m, fx, self._vars, hist,
                               x0 if (_is_md(m) or _is_ls(m)) else [lo], lo, hi)
            self._connect_click()

            # Guardar sesión para exportación completa
            self._session.append({
                "metodo": m, "fx": fx, "gx": gx,
                "vars": list(self._vars), "x0": list(x0),
                "lo": lo, "hi": hi, "hist": hist,
                "traj": self._traj(self._vars, hist,
                                   x0 if (_is_md(m) or _is_ls(m)) else [lo]),
            })
            self._log_history(m, fx, hist)
            self.lbl_st.setText(f"✓ {m} → {self._rmsg(hist)}")
            self._update_ai_context()

        except _NumericalWarning as nw:
            QMsgBox.warning(self, "⚠ Advertencia numérica", str(nw))
            self.lbl_st.setText(f"⚠ Resultado numérico problemático.")
            # Intentar renderizar la gráfica igualmente para que el usuario
            # pueda visualizar el comportamiento de la función (incluso si tiende a inf/NaN)
            try:
                self._hist   = hist
                self._metodo = m
                self._fx     = fx
                self._vars   = vars_ if _is_md(m) else ["x"]
                self._x0     = x0
                self._lo     = lo;  self._hi   = hi
                self._gx     = gx;  self._tol_L = tol_L
                self._render_table(hist)
                self._render_plots(m, fx, self._vars, hist,
                                   x0 if _is_md(m) else [lo], lo, hi)
                self._connect_click()
                self._session.append({
                    "metodo": m, "fx": fx, "gx": gx,
                    "vars": list(self._vars), "x0": list(x0),
                    "lo": lo, "hi": hi, "hist": hist,
                    "traj": self._traj(self._vars, hist,
                                       x0 if _is_md(m) else [lo]),
                })
                self._log_history(m, fx, hist)
            except Exception:
                pass  # Si la gráfica también falla, no pasa nada
        except Exception as e:
            self.lbl_st.setText("✗ Error.")
            _err(self, "Error", str(e), traceback.format_exc())

    def _rmsg(self, hist):
        if not hist: return "sin resultado"
        last = hist[-1]

        # ── Multiobjetivo: claves especiales ──────────────────────────────────
        if "f1(x*)=x*²" in last or "f1(x*)=x*²" in last:
            xv  = last.get("x*(Bisección)", last.get("x*(Sec.Dorada)", last.get("x_k","?")))
            f1v = last.get("f1(x*)=x*²", last.get("f1(x*)", "?"))
            f2v = last.get("f2(x*)=(x*-2)²", last.get("f2(x*)", "?"))
            alv = last.get("α", "?")
            try: return (f"α={alv}  x*={float(xv):.4f}  "
                         f"f1={float(f1v):.4f}  f2={float(f2v):.4f}")
            except: return f"α={alv}  f1={f1v}  f2={f2v}"
        if "f1(x*)" in last and "f2(x*)" in last:
            xv  = last.get("x*(α)", last.get("x_k","?"))
            f1v = last.get("f1(x*)","?"); f2v = last.get("f2(x*)","?")
            try: return (f"x*={float(xv):.4f}  f1={float(f1v):.4f}  f2={float(f2v):.4f}")
            except: return f"f1={f1v}  f2={f2v}"
        if "J[f1]=2x" in last:
            xv = last.get("x","?"); nj = last.get("‖J‖","?")
            pareto = last.get("¿Pareto-opt?","")
            try: return f"x={float(xv):.4f}  ‖J‖={float(nj):.4f}  {pareto}"
            except: return f"x={xv}  {pareto}"

        def _safe_get(rec, *keys):
            for k in keys:
                v = rec.get(k)
                if v is None: continue
                if isinstance(v, np.ndarray): return v
                if isinstance(v, (int, float, np.number)): return v
                if isinstance(v, (list, tuple)) and len(v) > 0: return v
            return None

        v = _safe_get(last, "x_k", "x", "lambda_k")
        if v is None: return "ver tabla"

        for kf in ("f_k", "f_lambda", "f_mu"):
            if kf in last and isinstance(last[kf], (int, float)):
                fv = float(last[kf])
                if isinstance(v, np.ndarray):
                    xs = "[" + ", ".join(f"{x:.4g}" for x in v.flat) + "]"
                elif isinstance(v, (list, tuple)):
                    xs = "[" + ", ".join(f"{x:.4g}" for x in v) + "]"
                else:
                    xs = f"{float(v):.6g}"
                return f"x*={xs}, f*={fv:.6g}"

        if isinstance(v, np.ndarray):
            return "[" + ", ".join(f"{x:.4g}" for x in v.flat) + "]"
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(f"{x:.4g}" for x in v) + "]"
        return f"x*={float(v):.6g}"

    # ── LIMPIAR ───────────────────────────────────────────────────────────────
    def limpiar(self):
        self.cmb.blockSignals(True); self.cmb.setCurrentIndex(0); self.cmb.blockSignals(False)
        for w in (self.edt_fx,self.edt_gx,self.edt_vars,self.edt_x0): w.clear()
        self.edt_tol.setText("0.1")
        self.smin.setValue(0.0); self.smax.setValue(5.0)
        self.lbl_st.setText("")
        self._hist=[]; self._metodo=self._fx=""; self._vars=[]; self._x0=[]
        self._session=[]; self._session_methods=[]
        if self.ai_panel and hasattr(self.ai_panel, 'update_session_methods'):
            self.ai_panel.update_session_methods([])
        self.md_sec.setVisible(False); self.row_tol.setVisible(False)
        self._reset()

    # ── HISTORIAL ─────────────────────────────────────────────────────────────
    def _log_history(self, metodo: str, fx: str, hist: list):
        """Registra una entrada en el historial persistente."""
        import datetime
        # resultado final
        xopt = fopt = float("nan")
        try:
            last = hist[-1]
            xopt = float(last.get("x", float("nan")))
            fopt = float(last.get("f_mu", last.get("mu_k", last.get("f_lambda",
                         last.get("fx", float("nan"))))))
        except Exception:
            pass
        self._history_log.append({
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "metodo": metodo,
            "fx": fx,
            "iters": len(hist),
            "xopt": xopt,
            "fopt": fopt,
        })

    def _show_historial(self):
        """Muestra el historial de funciones y métodos en un diálogo."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("📋  Historial de funciones")
        dlg.setMinimumSize(820, 460)
        dlg.setStyleSheet(
            "QDialog { background:#0d1830; }"
            "QLabel#titulo { color:#7ab0ff; font-size:15px; font-weight:bold; padding:6px 0; }"
            "QLabel#hint { color:#4a7090; font-size:11px; padding:0 0 4px 0; }"
            "QTableWidget { background:#0a1525; color:#d8e8f8; gridline-color:#1a3060;"
            "  border:1px solid #2a4080; border-radius:6px; font-size:12px; }"
            "QTableWidget::item { padding:4px 8px; }"
            "QTableWidget::item:selected { background:#1a50c0; color:#fff; }"
            "QHeaderView::section { background:#0e1e40; color:#7ab0ff; font-weight:bold;"
            "  font-size:11px; padding:5px 8px; border:none; border-bottom:1px solid #2a4080; }"
            "QPushButton { background:#0e2060; color:#a8d0ff; border:1px solid #2a4ea0;"
            "  border-radius:6px; font-size:12px; padding:5px 18px; font-weight:bold; }"
            "QPushButton:hover { background:#1a3898; color:#d0e8ff; }"
            "QPushButton:disabled { background:#0a1530; color:#3a5070; border-color:#1a2a50; }"
            "QPushButton#btnRecalc { background:#0e3a1a; color:#3cff8a; border-color:#1a6040; }"
            "QPushButton#btnRecalc:hover { background:#1a5a2a; color:#80ffb0; }"
            "QPushButton#btnRecalc:disabled { background:#0a1510; color:#1a3020; border-color:#0a2010; }"
            "QPushButton#btnCargar { background:#0e2a50; color:#60c0ff; border-color:#1a4a80; }"
            "QPushButton#btnCargar:hover { background:#1a4a80; color:#a0d8ff; }"
            "QPushButton#btnCargar:disabled { background:#0a1525; color:#1a3050; border-color:#0a1a35; }"
            "QPushButton#btnBorrar { background:#3a0a0a; color:#ff8080; border-color:#6a2020; }"
            "QPushButton#btnBorrar:hover { background:#5a1010; }"
        )
        vl = QVBox(dlg); vl.setContentsMargins(14, 10, 14, 12); vl.setSpacing(6)

        titulo = QLbl("Historial de funciones y métodos"); titulo.setObjectName("titulo")
        vl.addWidget(titulo)
        hint = QLbl("Selecciona una fila y usa los botones, o haz doble clic para cargar la función y recalcular.")
        hint.setObjectName("hint"); vl.addWidget(hint)

        tbl = QTbl(0, 6)
        tbl.setHorizontalHeaderLabels(["Hora", "Función f(x)", "Método", "Iteraciones", "x*", "f(x*)"])
        tbl.horizontalHeader().setStretchLastSection(False)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
                            if _QT=="PyQt6" else QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
                                 if _QT=="PyQt6" else QtWidgets.QAbstractItemView.SelectRows)
        tbl.setAlternatingRowColors(True)
        hh = tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QHV.ResizeMode.ResizeToContents if _QT=="PyQt6" else QHV.ResizeToContents)
        hh.setSectionResizeMode(1, QHV.ResizeMode.Stretch if _QT=="PyQt6" else QHV.Stretch)
        hh.setSectionResizeMode(2, QHV.ResizeMode.ResizeToContents if _QT=="PyQt6" else QHV.ResizeToContents)
        hh.setSectionResizeMode(3, QHV.ResizeMode.ResizeToContents if _QT=="PyQt6" else QHV.ResizeToContents)
        hh.setSectionResizeMode(4, QHV.ResizeMode.ResizeToContents if _QT=="PyQt6" else QHV.ResizeToContents)
        hh.setSectionResizeMode(5, QHV.ResizeMode.ResizeToContents if _QT=="PyQt6" else QHV.ResizeToContents)

        def _fill():
            tbl.setRowCount(0)
            for i, e in enumerate(self._history_log):
                tbl.insertRow(i)
                xv = f"{e['xopt']:.5f}" if not (e['xopt'] != e['xopt']) else "—"
                fv = f"{e['fopt']:.5f}" if not (e['fopt'] != e['fopt']) else "—"
                for j, val in enumerate([e["timestamp"], e["fx"], e["metodo"],
                                          str(e["iters"]), xv, fv]):
                    it = QTI(val)
                    it.setTextAlignment(_AL("AlignVCenter", "AlignHCenter")
                                        if j in (0,3,4,5) else _AL("AlignVCenter", "AlignLeft"))
                    tbl.setItem(i, j, it)

        _fill()
        vl.addWidget(tbl, 1)

        # ── Botones de acción ──────────────────────────────────────────────────
        brow = QW(); brl = QHBox(brow); brl.setContentsMargins(0,4,0,0); brl.setSpacing(8)

        btn_recalc = QBtn("🔄  Cargar y recalcular"); btn_recalc.setObjectName("btnRecalc")
        btn_cargar = QBtn("📥  Cargar función");      btn_cargar.setObjectName("btnCargar")
        btn_borrar = QBtn("🗑  Limpiar historial");   btn_borrar.setObjectName("btnBorrar")
        btn_close  = QBtn("Cerrar")

        btn_recalc.setEnabled(False); btn_recalc.setToolTip("Carga la función y el método del historial y ejecuta el cálculo")
        btn_cargar.setEnabled(False); btn_cargar.setToolTip("Carga solo la función en el panel (puedes cambiar el método antes de calcular)")

        brl.addWidget(btn_recalc)
        brl.addWidget(btn_cargar)
        brl.addSpacing(12)
        brl.addWidget(btn_borrar)
        brl.addStretch()
        brl.addWidget(btn_close)
        vl.addWidget(brow)

        # Habilitar botones solo cuando hay selección
        def _on_selection():
            has = bool(tbl.selectedItems())
            btn_recalc.setEnabled(has)
            btn_cargar.setEnabled(has)

        tbl.itemSelectionChanged.connect(_on_selection)

        def _selected_entry():
            rows = tbl.selectedItems()
            if not rows:
                return None
            row = tbl.currentRow()
            if 0 <= row < len(self._history_log):
                return self._history_log[row]
            return None

        def _load_entry(entry, recalcular: bool):
            """Carga función y opcionalmente el método en el panel principal."""
            if entry is None:
                return
            # Cargar función
            self.edt_fx.setText(entry["fx"])
            # Cargar método si corresponde
            metodo = entry["metodo"]
            idx = self.cmb.findText(metodo)
            if idx >= 0:
                self.cmb.setCurrentIndex(idx)
            # Cerrar diálogo
            dlg.accept()
            # Recalcular si se pidió
            if recalcular:
                QtCore.QTimer.singleShot(80, self.ejecutar)

        def _do_recalc():
            _load_entry(_selected_entry(), recalcular=True)

        def _do_cargar():
            _load_entry(_selected_entry(), recalcular=False)

        def _on_double_click(row, _col):
            if 0 <= row < len(self._history_log):
                _load_entry(self._history_log[row], recalcular=True)

        btn_recalc.clicked.connect(_do_recalc)
        btn_cargar.clicked.connect(_do_cargar)
        tbl.cellDoubleClicked.connect(_on_double_click)

        def _borrar():
            r = QMsgBox.question(dlg, "Confirmar",
                                 "¿Desea borrar todo el historial?",
                                 QMsgBox.StandardButton.Yes | QMsgBox.StandardButton.No
                                 if _QT=="PyQt6" else QMsgBox.Yes | QMsgBox.No)
            yes_btn = QMsgBox.StandardButton.Yes if _QT=="PyQt6" else QMsgBox.Yes
            if r == yes_btn:
                self._history_log.clear()
                _fill()
                btn_recalc.setEnabled(False)
                btn_cargar.setEnabled(False)

        btn_borrar.clicked.connect(_borrar)
        btn_close.clicked.connect(dlg.accept)
        dlg.exec() if _QT=="PyQt6" else dlg.exec_()

    def salir(self):
        yes = QMsgBox.StandardButton.Yes if _QT=="PyQt6" else QMsgBox.Yes
        no  = QMsgBox.StandardButton.No  if _QT=="PyQt6" else QMsgBox.No
        dlg = QMsgBox(self)
        dlg.setWindowTitle("Salir")
        dlg.setText("¿Desea salir de la aplicación?")
        dlg.setIcon(QMsgBox.Icon.Question if _QT=="PyQt6" else QMsgBox.Question)
        dlg.setStandardButtons(yes | no)
        dlg.setDefaultButton(no)
        # Texto más grande
        dlg.setStyleSheet("QLabel { font-size: 15px; font-weight: bold; min-width: 280px; }"
                          "QPushButton { font-size: 13px; padding: 6px 24px; }")
        try:
            dlg.button(yes).setText("Sí")
            dlg.button(no).setText("No")
        except Exception:
            pass
        result = dlg.exec() if _QT=="PyQt6" else dlg.exec_()
        if result == yes:
            for c in (self.canvas2d,self.canvas3d):
                if not c: continue
                try: c.stop_animation()
                except: pass
                try: c.stop_rotation()
                except: pass
            self.close()

    def _exp_menu(self):
        m = QMenu(self)
        m.setStyleSheet(
            "QMenu{background:#0d1830;color:#d8e8f8;border:1px solid #2a4080;"
            "border-radius:8px;padding:6px;font-size:13px;}"
            "QMenu::item{padding:8px 24px;border-radius:5px;}"
            "QMenu::item:selected{background:#1a50c0;}"
            "QMenu::separator{height:1px;background:#2a3e6e;margin:4px 10px;}"
        )
        m.addAction("📄 Exportar como CSV").triggered.connect(self.export_csv_db)
        m.addAction("📊 Exportar como Excel (.xlsx)").triggered.connect(self.export_xlsx)
        m.addAction("📑 Exportar como PDF").triggered.connect(self.export_pdf)
        m.exec(self.btn_export.mapToGlobal(self.btn_export.rect().bottomLeft()))

    def _show_export_success(self, tipo: str, path: str, extra: str = ""):
        """Muestra diálogo de éxito con botón Aceptar tras exportar."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Exportación exitosa")
        dlg.setFixedSize(460, 260)
        dlg.setStyleSheet(
            "QDialog{background:#0d1830;border:2px solid #1a50c0;border-radius:10px;}"
            "QLabel{color:#d8e8f8;background:transparent;}"
            "QPushButton{background:#1a50c0;color:#fff;border:none;border-radius:6px;"
            "padding:10px 36px;font-size:14px;font-weight:bold;}"
            "QPushButton:hover{background:#2462d8;}"
        )
        lay = QVBox(dlg); lay.setContentsMargins(28, 22, 28, 22); lay.setSpacing(12)

        icon_lbl = QLbl("✅")
        icon_lbl.setAlignment(_AL("AlignHCenter"))
        font_big = QFont("Segoe UI", 32)
        icon_lbl.setFont(font_big)
        lay.addWidget(icon_lbl)

        title = QLbl(f"¡Archivo {tipo} exportado exitosamente!")
        title.setAlignment(_AL("AlignHCenter"))
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold if _QT=="PyQt6" else QFont.Bold))
        title.setStyleSheet("color:#ffffff;")
        lay.addWidget(title)

        fname = QLbl(f"📄 {os.path.basename(path)}")
        fname.setAlignment(_AL("AlignHCenter"))
        fname.setStyleSheet("color:#7aaaea;font-size:12px;")
        lay.addWidget(fname)

        if extra:
            ex_lbl = QLbl(extra)
            ex_lbl.setAlignment(_AL("AlignHCenter"))
            ex_lbl.setStyleSheet("color:#7aaaea;font-size:11px;")
            ex_lbl.setWordWrap(True)
            lay.addWidget(ex_lbl)

        lay.addStretch()

        btn = QBtn("  Aceptar  ")
        btn.clicked.connect(dlg.accept)
        btn.setFixedHeight(38)
        btn_row = QHBox(); btn_row.addStretch(); btn_row.addWidget(btn); btn_row.addStretch()
        lay.addLayout(btn_row)

        dlg.exec() if _QT == "PyQt6" else dlg.exec_()

    # ── EJECUCIÓN ─────────────────────────────────────────────────────────────
    def _run(self, metodo, fx, gx, vars_, x0, lo, hi, tol_L):

        def coerce(res):
            if res is None: return []
            if isinstance(res,list):
                return res if (res and isinstance(res[0],dict)) \
                       else [{"k":i,"val":r} for i,r in enumerate(res)]
            if isinstance(res,tuple) and len(res)==3: return coerce(res[2])
            if isinstance(res,dict):
                return coerce(res["history"]) if "history" in res else [res]
            try:
                import pandas as pd
                if isinstance(res,pd.DataFrame): return res.to_dict(orient="records")
            except: pass
            return [{"val":res}]

        # ── Helper: normalizar expresión y detectar/resolver variables ────────
        fc = _get_fc()

        def _norm(expr):
            return fc.normalize_expr(expr) if fc else expr

        def _detect_vars(expr):
            """Detecta variables, priorizando las que el usuario especificó."""
            if vars_ and vars_ != ["x1", "x2"] and any(v.strip() for v in vars_):
                return [v for v in vars_ if v.strip()]
            if fc:
                detected = fc.detect_variables(_norm(expr))
                return detected if detected else ["x"]
            return ["x"]

        def _get_x0_for(vnames):
            """Devuelve punto inicial: usa UI si fue especificado, si no usa centro del rango."""
            mid = (lo + hi) / 2
            if x0 and len(x0) == len(vnames):
                return list(x0)
            # Si solo hay una variable, usar el punto medio del rango
            if len(vnames) == 1:
                return [mid]
            # Multivariable sin x0: intentar x0 de UI truncado/ampliado
            if x0:
                base = list(x0)
                while len(base) < len(vnames): base.append(mid)
                return base[:len(vnames)]
            return [mid] * len(vnames)

        # ── Función 1D con detección automática de variable ──────────────────
        def f1d():
            if not _SYMPY: raise RuntimeError("SymPy no disponible.")
            if fc:
                norm = _norm(fx)
                fn_callable, var_name = fc.parse_function_1d(norm)
                return fn_callable
            else:
                # Fallback clásico
                xs = sp.Symbol("x")
                fl = sp.lambdify(xs, sp.sympify(fx, locals={"x": xs}), modules=["numpy"])
                return lambda v: float(fl(v))

        def _f1d_callable(norm_expr: str) -> object:
            """Alias para f1d que acepta expresión ya normalizada."""
            if fc:
                fn_c, _ = fc.parse_function_1d(norm_expr)
                return fn_c
            return f1d()

        # ── 1D ────────────────────────────────────────────────────────────────
        if metodo == "Búsqueda Local":
            if not local_fn: raise RuntimeError(
                "localsearch.py no encontrado.\nVerifica que el archivo esté en la carpeta del proyecto.")
            xo,fm,fa,fb,log = local_fn(f1d(), lo, hi, (hi-lo)*0.1, 20)
            for r in log: r.setdefault("x", r.get("lambda_k",0.))
            return log

        if metodo == "Fibonacci":
            if not fibonacci_fn: raise RuntimeError(
                "localsearch.py no encontrado.\nVerifica que el archivo esté en la carpeta del proyecto.")
            xo,fm,fa,fb,log = fibonacci_fn(f1d(), lo, hi, max(tol_L,1e-10))
            for r in log: r.setdefault("x", r.get("lambda_k",0.))
            return log

        if metodo == "Armijo":
            if not newton_armijo_fn: raise RuntimeError(
                "line_search.py no encontrado.\nVerifica que el archivo esté en la carpeta del proyecto.")
            norm_fx  = _norm(fx)
            v_names  = _detect_vars(norm_fx)
            var_str  = " ".join(v_names)
            x0_use   = _get_x0_for(v_names)
            xo, yo, log = newton_armijo_fn(
                func_str=norm_fx, var_str=var_str, x0=x0_use,
                max_iter=50, tolerance=1e-6)
            return log or []

        if metodo == "Wolfe":
            if not newton_wolfe_fn: raise RuntimeError(
                "line_search.py no encontrado.\nVerifica que el archivo esté en la carpeta del proyecto.")
            norm_fx  = _norm(fx)
            v_names  = _detect_vars(norm_fx)
            x0_use   = _get_x0_for(v_names)
            xo, yo, log = newton_wolfe_fn(
                func_str=norm_fx, var_str=" ".join(v_names),
                x_k_list=x0_use, u_in=lo, v_in=hi, mu=0.6)
            return log or []

        # ── BISECCIÓN y SECCIÓN DORADA (autónomas, sin módulos externos) ───────
        if metodo == "Bisección":
            fn1d   = _f1d_callable(_norm(fx))
            h      = max((hi - lo) * 1e-5, 1e-9)
            fprime = lambda xv: (fn1d(xv + h) - fn1d(xv - h)) / (2.0 * h)
            a_b, b_b = float(lo), float(hi)
            tol_b, max_b = 1e-8, 300

            # Buscar subintervalo con cambio de signo en f'
            if fprime(a_b) * fprime(b_b) > 0:
                n_sub  = 400
                xs_sub = np.linspace(a_b, b_b, n_sub)
                found  = False
                for _i in range(n_sub - 1):
                    if fprime(xs_sub[_i]) * fprime(xs_sub[_i+1]) <= 0:
                        a_b, b_b = float(xs_sub[_i]), float(xs_sub[_i+1])
                        found = True; break
                if not found:
                    ys_sub = np.array([fn1d(xv) for xv in xs_sub])
                    xopt   = float(xs_sub[np.argmin(ys_sub)])
                    return [{"k": 0, "x_k": round(xopt, 8),
                             "f_k": round(fn1d(xopt), 8),
                             "f'(x_k)": round(fprime(xopt), 8),
                             "b-a": round(b_b - a_b, 8),
                             "nota": "sin cambio signo; retorna mín. en grid"}]

            hist_b: list = []
            for k in range(max_b):
                xm  = (a_b + b_b) / 2.0
                fpm = fprime(xm)
                largo = b_b - a_b
                hist_b.append({
                    "k":        k,
                    "a_k":      round(a_b, 8),
                    "b_k":      round(b_b, 8),
                    "x_k":      round(xm,  8),
                    "f_k":      round(fn1d(xm), 8),
                    "f'(x_k)":  round(fpm,  8),
                    "b-a":      round(largo, 8),
                    "converged": largo < tol_b or abs(fpm) < tol_b,
                })
                if largo < tol_b or abs(fpm) < tol_b:
                    break
                if fprime(a_b) * fpm < 0:
                    b_b = xm
                else:
                    a_b = xm
            return hist_b

        if metodo == "Sección Dorada":
            fn1d = _f1d_callable(_norm(fx))
            phi  = (math.sqrt(5.0) - 1.0) / 2.0   # ≈ 0.61803…
            a_g, b_g = float(lo), float(hi)
            tol_g, max_g = 1e-8, 300

            c_g  = b_g - phi * (b_g - a_g)
            d_g  = a_g + phi * (b_g - a_g)
            fc_g = fn1d(c_g); fd_g = fn1d(d_g)

            hist_g: list = []
            for k in range(max_g):
                largo = b_g - a_g
                xm    = (a_g + b_g) / 2.0
                hist_g.append({
                    "k":      k,
                    "a_k":    round(a_g,  8),
                    "b_k":    round(b_g,  8),
                    "c_k":    round(c_g,  8),
                    "f(c_k)": round(fc_g, 8),
                    "d_k":    round(d_g,  8),
                    "f(d_k)": round(fd_g, 8),
                    "x_k":    round(xm,   8),
                    "f_k":    round(fn1d(xm), 8),
                    "b-a":    round(largo, 8),
                    "r_φ":    round(phi,   6),
                    "converged": largo < tol_g,
                })
                if largo < tol_g:
                    break
                if fc_g < fd_g:
                    b_g = d_g; d_g = c_g; fd_g = fc_g
                    c_g = b_g - phi * (b_g - a_g); fc_g = fn1d(c_g)
                else:
                    a_g = c_g; c_g = d_g; fc_g = fd_g
                    d_g = a_g + phi * (b_g - a_g); fd_g = fn1d(d_g)

            xopt_g = (a_g + b_g) / 2.0
            hist_g[-1]["x_k"] = round(xopt_g, 8)
            hist_g[-1]["f_k"] = round(fn1d(xopt_g), 8)
            return hist_g

        # ── HEURÍSTICOS ───────────────────────────────────────────────────────
        if metodo == "Sección Áurea":
            if not _heuristics_mod: raise RuntimeError(
                "heuristics.py no encontrado.")
            fn1d = _f1d_callable(_norm(fx))
            return _heuristics_mod.seccion_aurea(fn1d, lo, hi, tol=1e-6, max_iter=200)

        if metodo == "Goldstein":
            if not _heuristics_mod: raise RuntimeError(
                "heuristics.py no encontrado.")
            norm_fx = _norm(fx)
            v_names = _detect_vars(norm_fx)
            x0_use  = _get_x0_for(v_names)
            return _heuristics_mod.goldstein_search(
                norm_fx, " ".join(v_names), x0_use,
                mu=0.2, alpha0=1.0, rho=0.5, max_iter=60)

        if metodo == "Newton-Raphson":
            if not _heuristics_mod: raise RuntimeError(
                "heuristics.py no encontrado.")
            norm_fx = _norm(fx)
            v_names = _detect_vars(norm_fx)
            x0_use  = _get_x0_for(v_names)
            return _heuristics_mod.newton_raphson(
                norm_fx, " ".join(v_names), x0_use,
                tol=1e-7, max_iter=60)

        if metodo == "Grad. Conjugado":
            if not _heuristics_mod: raise RuntimeError(
                "heuristics.py no encontrado.")
            norm_fx = _norm(fx)
            v_names = _detect_vars(norm_fx)
            x0_use  = _get_x0_for(v_names)
            return _heuristics_mod.gradient_conjugate(
                norm_fx, " ".join(v_names), x0_use,
                tol=1e-7, max_iter=150)

        # ── METAHEURÍSTICOS ───────────────────────────────────────────────────
        if metodo == "SA: Recocido Simulado":
            if not _metaheur_mod: raise RuntimeError(
                "metaheuristics.py no encontrado.")
            norm_fx = _norm(fx)
            v_names = _detect_vars(norm_fx)
            x0_use  = _get_x0_for(v_names)
            return _metaheur_mod.simulated_annealing(
                norm_fx, " ".join(v_names), x0_use,
                T_init=100.0, T_min=1e-5, cooling=0.95,
                max_iter=600, step_size=(hi - lo) * 0.1)

        if metodo == "PSO: Enjambre":
            if not _metaheur_mod: raise RuntimeError(
                "metaheuristics.py no encontrado.")
            norm_fx = _norm(fx)
            v_names = _detect_vars(norm_fx)
            return _metaheur_mod.pso(
                norm_fx, " ".join(v_names),
                bounds=[lo, hi],
                n_particles=30, max_iter=150,
                w=0.7, c1=1.5, c2=1.5)

        if metodo == "GA: Algoritmo Genético":
            if not _metaheur_mod: raise RuntimeError(
                "metaheuristics.py no encontrado.")
            norm_fx = _norm(fx)
            v_names = _detect_vars(norm_fx)
            return _metaheur_mod.genetic_algorithm(
                norm_fx, " ".join(v_names),
                bounds=[lo, hi],
                pop_size=50, max_gen=150,
                p_cross=0.8, p_mut=0.15, elite=3)

        # ── MD ────────────────────────────────────────────────────────────────
        if metodo.startswith("MD:"):
            w = _wrappers_mod
            if not w: raise RuntimeError(
                "wrappers.py no encontrado.\nVerifica que el archivo esté en la carpeta del proyecto.")
            norm_fx = _norm(fx)
            norm_gx = _norm(gx) if gx else ""

            # ── Reconciliar variables ─────────────────────────────────────────
            # wrappers.py usa sympify con los símbolos dados; si la expresión
            # tiene variables distintas a las del campo UI (ej. usuario pone 'x1 x2'
            # pero la función usa 'x'), la evaluación falla con TypeError.
            # Solución: construir la lista efectiva de variables tomando la unión
            # de las detectadas en la expresión + las de la restricción + las del campo UI,
            # dando prioridad a las detectadas en la expresión para que coincidan.
            if fc:
                _fx_vars  = fc.detect_variables(norm_fx)
                _gx_vars  = fc.detect_variables(norm_gx) if norm_gx else []
                _ui_vars  = [v for v in vars_ if v.strip()]

                # Unión en orden: vars de fx primero, luego gx, luego UI
                _merged: list = list(_fx_vars)
                for v in _gx_vars + _ui_vars:
                    if v not in _merged:
                        _merged.append(v)

                # Si la lista efectiva está vacía, usar las del UI o fallback
                effective_vars = _merged if _merged else (_ui_vars if _ui_vars else ["x1", "x2"])
            else:
                effective_vars = [v for v in vars_ if v.strip()] or ["x1", "x2"]

            var_str = " ".join(effective_vars)

            # Ajustar x0 al número de variables efectivas
            mid = (lo + hi) / 2.0
            x0_md = list(x0) if x0 else []
            while len(x0_md) < len(effective_vars):
                x0_md.append(mid)
            x0_md = x0_md[:len(effective_vars)]

            fn_map = {
                "MD: Penalización (Newton)": "penalty_method_newton",
                "MD: Barreras (Newton)":     "barrier_method_newton",
                "MD: Pen. (BFGS)":           "penalty_bfgs",
                "MD: Pes. (BFGS)":           "steepest_bfgs",
                "MD: Pes. (Nelder-Mead)":    "nelder_mead",
                "MD: Sum. (BFGS)":           "sum_bfgs",
            }
            fn_name = fn_map.get(metodo)
            fn = getattr(w, fn_name, None) if fn_name else None
            if fn is None:
                if "Penalización" in metodo or "Pen." in metodo:
                    fn = getattr(w, "penalty_method_newton", None)
                elif "Barrera" in metodo:
                    fn = getattr(w, "barrier_method_newton", None)
                else:
                    fn = getattr(w, "penalty_method_newton", None)
            if fn is None:
                raise RuntimeError(f"Método '{metodo}' no implementado en wrappers.py.")

            # Llamada principal con firma correcta
            _,_,hist = fn(norm_fx, norm_gx, var_str, x0_md)
            return hist

        # ── MULTIOBJETIVO ─────────────────────────────────────────────────────
        # Flujo: f(x) mono-objetivo del usuario
        #        → Escalarización: F(x)=(f1,f2) donde f1=f(x), f2=(x-x_c)²
        #        → φ(x,α) = α·f1(x) + (1-α)·f2(x)   para α dado
        #        → Resolver φ con Bisección o Sección Dorada
        #        → La tabla de α es un RESULTADO de referencia (no método)
        if _is_multiobj(metodo):
            mo = _get_multiobj()
            if mo is None:
                raise RuntimeError(
                    "multiobj_schaffer.py no encontrado.\n"
                    "Coloca el archivo junto a interfaz_qt.py.")
            if not fx or fx == "schaffer_interno":
                raise RuntimeError(
                    "Ingresa la función f(x) mono-objetivo en el campo superior.\n"
                    "El optimizador la convertirá en problema multiobjetivo.")

            import math as _math
            norm_fx_mo = _norm(fx)
            v_names    = _detect_vars(norm_fx_mo)
            var1       = v_names[0] if v_names else "x"

            # Punto inicial
            try:
                x0_mo_txt = getattr(self, 'edt_x0_mo', None)
                x0_start  = float(x0_mo_txt.text().strip()) if x0_mo_txt and x0_mo_txt.text().strip() else (lo+hi)/2
            except Exception:
                x0_start = (lo + hi) / 2

            # α del spinner
            a_val = getattr(self, 'edt_alpha', None)
            a_val = float(a_val.value()) if a_val else 0.5

            # Construir f1 y f2 callables
            if not _SYMPY:
                raise RuntimeError("SymPy es necesario para métodos multiobjetivo.")
            sym_x   = sp.Symbol(var1)
            sym_f1  = sp.sympify(norm_fx_mo)
            sym_df1 = sp.diff(sym_f1, sym_x)
            sym_d2f1= sp.diff(sym_df1, sym_x)
            f1_call  = sp.lambdify(sym_x, sym_f1,   modules=["numpy"])
            df1_call = sp.lambdify(sym_x, sym_df1,  modules=["numpy"])
            d2f1_call= sp.lambdify(sym_x, sym_d2f1, modules=["numpy"])

            x_c  = (lo + hi) / 2.0          # centro del rango → f2 centrada
            f2   = lambda xv: (float(xv) - x_c)**2
            df2  = lambda xv: 2.0*(float(xv) - x_c)
            d2f2 = lambda _xv: 2.0

            # φ(x,α) escalarizada y su derivada
            phi   = lambda xv, a: a*float(f1_call(xv)) + (1-a)*float(f2(xv))
            dphi  = lambda xv, a: a*float(df1_call(xv)) + (1-a)*float(df2(xv))
            d2phi = lambda xv, a: a*float(d2f1_call(xv)) + (1-a)*float(d2f2(xv))

            # ── Tabla de referencia α (siempre se calcula, se guarda en hist) ──
            # Resuelve φ con Sección Dorada para α = {0,1/5,1/4,1/3,1/2,1}
            _ALPHAS_REF = [
                (0.0,   "0"),
                (1/5,   "1/5"),
                (1/4,   "1/4"),
                (1/3,   "1/3"),
                (1/2,   "1/2"),
                (1.0,   "1"),
            ]
            tabla_ref = []
            for a_r, a_str in _ALPHAS_REF:
                # Sección Dorada rápida para cada α
                phi_r = lambda xv, a=a_r: a*float(f1_call(xv)) + (1-a)*float(f2(xv))
                p_g = (_math.sqrt(5)-1)/2
                ag_, bg_ = float(lo), float(hi)
                cg_ = bg_-p_g*(bg_-ag_); dg_ = ag_+p_g*(bg_-ag_)
                fcg_=phi_r(cg_); fdg_=phi_r(dg_)
                for _ in range(200):
                    if bg_-ag_ < 1e-8: break
                    if fcg_<fdg_: bg_=dg_; dg_=cg_; fdg_=fcg_; cg_=bg_-p_g*(bg_-ag_); fcg_=phi_r(cg_)
                    else:         ag_=cg_; cg_=dg_; fcg_=fdg_; dg_=ag_+p_g*(bg_-ag_); fdg_=phi_r(dg_)
                xopt_r = (ag_+bg_)/2
                try:
                    f1r = float(f1_call(xopt_r)); f2r = float(f2(xopt_r))
                    phi_r_val = float(phi_r(xopt_r))
                except Exception:
                    f1r = f2r = phi_r_val = float("nan")
                tabla_ref.append({
                    "α": a_str,
                    "f(x,α)": f"α·f₁+(1-α)·(x-{x_c:.2f})²",
                    "x*(α)":  round(xopt_r, 6),
                    "f₁(x*)": round(f1r, 6),
                    "f₂(x*)": round(f2r, 6),
                    "φ(x*,α)": round(phi_r_val, 6),
                })

            # ═══════════════════════════════════════════════════════════════════
            # MO — BISECCIÓN
            # ═══════════════════════════════════════════════════════════════════
            if metodo == "MO — Bisección":
                h = max((hi-lo)*1e-5, 1e-9)
                fp_num = lambda xv: (phi(xv+h,a_val) - phi(xv-h,a_val))/(2*h)
                a_b, b_b = float(lo), float(hi)
                # buscar cambio de signo en φ'
                fa_s, fb_s = fp_num(a_b), fp_num(b_b)
                if fa_s * fb_s > 0:
                    xs_ = np.linspace(a_b, b_b, 500)
                    for _i in range(len(xs_)-1):
                        if fp_num(xs_[_i])*fp_num(xs_[_i+1]) <= 0:
                            a_b, b_b = float(xs_[_i]), float(xs_[_i+1]); break
                hist_b = []
                for k in range(300):
                    xm = (a_b+b_b)/2.0; fpm = fp_num(xm); largo = b_b-a_b
                    try:
                        f1v = float(f1_call(xm)); f2v = float(f2(xm))
                        phiv= float(phi(xm, a_val))
                    except Exception:
                        f1v = f2v = phiv = float("nan")
                    hist_b.append({
                        "k":          k,
                        "a_k":        round(a_b,   8),
                        "b_k":        round(b_b,   8),
                        "x_k":        round(xm,    8),
                        "φ(x_k,α)":  round(phiv,  8),
                        "φ'(x_k,α)": round(fpm,   8),
                        "b−a":        round(largo,  8),
                        "f₁(x_k)":   round(f1v,   6),
                        "f₂(x_k)":   round(f2v,   6),
                        "α":          round(a_val, 4),
                        "converged":  largo < 1e-8 or abs(fpm) < 1e-8,
                    })
                    if largo < 1e-8 or abs(fpm) < 1e-8: break
                    if fp_num(a_b)*fpm < 0: b_b = xm
                    else:                   a_b = xm
                # Adjuntar tabla de referencia α al final como filas separadas
                hist_b.append({"k": "──", "a_k": "── TABLA DE REFERENCIA α ──",
                                "b_k":"", "x_k":"", "φ(x_k,α)":"",
                                "φ'(x_k,α)":"","b−a":"","f₁(x_k)":"","f₂(x_k)":"",
                                "α":"","converged":""})
                for row in tabla_ref:
                    hist_b.append({
                        "k":          "ref",
                        "a_k":        row["α"],
                        "b_k":        row["f(x,α)"],
                        "x_k":        row["x*(α)"],
                        "φ(x_k,α)":  row["φ(x*,α)"],
                        "φ'(x_k,α)": "",
                        "b−a":        "",
                        "f₁(x_k)":   row["f₁(x*)"],
                        "f₂(x_k)":   row["f₂(x*)"],
                        "α":          row["α"],
                        "converged":  "",
                    })
                # Guardar tabla ref para gráficas
                self._mo_tabla_ref   = tabla_ref
                self._mo_hist_iters  = [r for r in hist_b if r.get("k") not in ("──","ref")]
                self._mo_alpha       = a_val
                self._mo_x_c         = x_c
                self._mo_f1_call     = f1_call
                self._mo_f2_call     = f2
                self._mo_var1        = var1
                return hist_b

            # ═══════════════════════════════════════════════════════════════════
            # MO — SECCIÓN DORADA
            # ═══════════════════════════════════════════════════════════════════
            if metodo == "MO — Sección Dorada":
                p_g = (_math.sqrt(5)-1)/2
                ag_, bg_ = float(lo), float(hi)
                cg_ = bg_-p_g*(bg_-ag_); dg_ = ag_+p_g*(bg_-ag_)
                fcg_ = phi(cg_,a_val); fdg_ = phi(dg_,a_val)
                hist_g = []
                for k in range(300):
                    largo = bg_-ag_; xm = (ag_+bg_)/2
                    try:
                        f1v = float(f1_call(xm)); f2v = float(f2(xm))
                        phiv= float(phi(xm, a_val))
                    except Exception:
                        f1v = f2v = phiv = float("nan")
                    hist_g.append({
                        "k":         k,
                        "a_k":       round(ag_,  8),
                        "b_k":       round(bg_,  8),
                        "c_k":       round(cg_,  8),
                        "φ(c)":      round(fcg_, 8),
                        "d_k":       round(dg_,  8),
                        "φ(d)":      round(fdg_, 8),
                        "x_k":       round(xm,   8),
                        "φ(x_k,α)": round(phiv, 8),
                        "b−a":       round(largo, 8),
                        "φ":         round(p_g,  6),
                        "f₁(x_k)":  round(f1v,  6),
                        "f₂(x_k)":  round(f2v,  6),
                        "α":         round(a_val, 4),
                        "converged": largo < 1e-8,
                    })
                    if largo < 1e-8: break
                    if fcg_ < fdg_:
                        bg_=dg_; dg_=cg_; fdg_=fcg_
                        cg_=bg_-p_g*(bg_-ag_); fcg_=phi(cg_,a_val)
                    else:
                        ag_=cg_; cg_=dg_; fcg_=fdg_
                        dg_=ag_+p_g*(bg_-ag_); fdg_=phi(dg_,a_val)
                xopt_g = (ag_+bg_)/2
                try:
                    hist_g[-1]["x_k"]       = round(xopt_g, 8)
                    hist_g[-1]["φ(x_k,α)"] = round(phi(xopt_g,a_val), 8)
                    hist_g[-1]["f₁(x_k)"]  = round(float(f1_call(xopt_g)), 6)
                    hist_g[-1]["f₂(x_k)"]  = round(float(f2(xopt_g)), 6)
                except Exception:
                    pass
                # Adjuntar tabla de referencia α
                hist_g.append({"k":"──","a_k":"── TABLA DE REFERENCIA α ──",
                                "b_k":"","c_k":"","φ(c)":"","d_k":"","φ(d)":"",
                                "x_k":"","φ(x_k,α)":"","b−a":"","φ":"",
                                "f₁(x_k)":"","f₂(x_k)":"","α":"","converged":""})
                for row in tabla_ref:
                    hist_g.append({
                        "k":         "ref",
                        "a_k":       row["α"],
                        "b_k":       row["f(x,α)"],
                        "c_k": "", "φ(c)":"","d_k":"","φ(d)":"",
                        "x_k":       row["x*(α)"],
                        "φ(x_k,α)": row["φ(x*,α)"],
                        "b−a": "", "φ":"",
                        "f₁(x_k)":  row["f₁(x*)"],
                        "f₂(x_k)":  row["f₂(x*)"],
                        "α":         row["α"],
                        "converged": "",
                    })
                self._mo_tabla_ref   = tabla_ref
                self._mo_hist_iters  = [r for r in hist_g if r.get("k") not in ("──","ref")]
                self._mo_alpha       = a_val
                self._mo_x_c         = x_c
                self._mo_f1_call     = f1_call
                self._mo_f2_call     = f2
                self._mo_var1        = var1
                return hist_g

            # ═══════════════════════════════════════════════════════════════════
            # MO — FRENTE DE PARETO
            # ═══════════════════════════════════════════════════════════════════
            if metodo == "MO — Frente de Pareto":
                result = mo.pareto_front_generic(
                    func_expr=norm_fx_mo, var_name=var1,
                    x0=x0_start, lo=lo, hi=hi, n_alpha=80)
                self._mo_tabla_ref  = tabla_ref
                self._mo_f1_call    = f1_call
                self._mo_f2_call    = f2
                self._mo_x_c        = x_c
                self._mo_var1       = var1
                return result["history"]

            # ═══════════════════════════════════════════════════════════════════
            # MO — ANÁLISIS JACOBIANO
            # ═══════════════════════════════════════════════════════════════════
            if metodo == "MO — Análisis Jacobiano":
                rows_j = mo.jacobian_generic(
                    func_expr=norm_fx_mo, var_name=var1,
                    x0=x0_start, lo=lo, hi=hi, n_points=15)
                self._mo_tabla_ref = tabla_ref
                self._mo_f1_call   = f1_call
                self._mo_f2_call   = f2
                self._mo_x_c       = x_c
                self._mo_var1      = var1
                return rows_j

        return []

    # ── TABLA ─────────────────────────────────────────────────────────────────
    def _render_table(self, hist):
        if not hist: self.table.setRowCount(0); self.table.setColumnCount(0); return
        cols: List[str] = []
        for r in hist:
            for k in r:
                if k not in cols: cols.append(k)
        if "x_k_str" in cols and "x_k" in cols: cols.remove("x_k")
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(hist))
        for i,row in enumerate(hist):
            for j,col in enumerate(cols):
                v = row.get(col,"")
                if isinstance(v,np.ndarray): v="["+", ".join(f"{x:.5g}" for x in v.flat)+"]"
                elif isinstance(v,float):    v=f"{v:.6g}"
                self.table.setItem(i,j,QTI(str(v)))
        try:
            self.table.horizontalHeader().setSectionResizeMode(
                QHV.ResizeMode.Stretch if _QT=="PyQt6" else QHV.Stretch)
        except: pass

    # ── GRÁFICAS ──────────────────────────────────────────────────────────────
    def _render_plots(self, metodo, fx, vars_, hist, x0, lo, hi):
        """Dibuja directamente sobre matplotlib fig del canvas, con trayectoria completa."""
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # ── DESVÍO: métodos multiobjetivo → render especial ───────────────────
        if _is_multiobj(metodo):
            self._render_plots_multiobj(metodo, fx, hist, lo, hi)
            return

        # ── Reconciliar variables con la expresión ────────────────────────────
        # Garantizar que vars_ coincide exactamente con las variables que usa fx.
        # Si el usuario tenía "x1 x2" en el campo UI pero la función usa "x",
        # _callable fallaría. Aquí lo corregimos antes de graficar.
        fc_plot = _get_fc()
        if fc_plot:
            _expr_vars = fc_plot.detect_variables(fc_plot.normalize_expr(fx))
            if _expr_vars:
                # Si las vars pasadas no están en la expresión → usar las detectadas
                _passed_set = set(vars_)
                _expr_set   = set(_expr_vars)
                if not _passed_set.intersection(_expr_set):
                    # Ninguna variable pasada existe en la expresión → reemplazar
                    vars_ = _expr_vars
                elif len(_expr_vars) != len(vars_):
                    # Diferente cantidad → usar detectadas
                    vars_ = _expr_vars
        # Limitar a 2 variables para las gráficas (siempre mostramos 1D o 2D)
        if len(vars_) > 2:
            vars_ = vars_[:2]


        traj = self._traj(vars_, hist, x0)
        BG    = "#0a1228"
        CC    = "#4a9eff"   # curva principal
        CT    = "#ff6b35"   # trayectoria
        CM    = "#ffd700"   # marcador óptimo
        CGRID = "#1a2e58"

        def _style_ax2d(ax):
            ax.set_facecolor(BG)
            for sp in ax.spines.values(): sp.set_color("#2a3e6e")
            ax.tick_params(colors="#7aaaea", labelsize=8)
            ax.xaxis.label.set_color("#7aaaea")
            ax.yaxis.label.set_color("#7aaaea")
            ax.title.set_color("#d8e8f8")
            ax.grid(True, color=CGRID, alpha=0.4, linewidth=0.6)

        def _style_ax3d(ax):
            ax.set_facecolor(BG)
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False; pane.set_edgecolor("#1a2e58")
            ax.tick_params(colors="#7aaaea", labelsize=7)
            ax.xaxis.label.set_color("#7aaaea")
            ax.yaxis.label.set_color("#7aaaea")
            ax.zaxis.label.set_color("#7aaaea")

        # ── 2D ────────────────────────────────────────────────────────────────
        if self.canvas2d and hasattr(self.canvas2d, "fig"):
            try:
                self.lbl_2d.setVisible(False); self.canvas2d.setVisible(True)
                fig = self.canvas2d.fig; fig.clear()
                fig.patch.set_facecolor(BG)
                ax = fig.add_subplot(111); _style_ax2d(ax)

                if len(vars_) == 1:
                    xg = _grid(lo, hi, 500)
                    yg = _e1d(fx, vars_, xg)
                    ax.plot(xg, yg, color=CC, lw=2.2, label=f"f(x) = {fx}", zorder=2)

                    if traj and traj[0].size > 0:
                        tx = traj[0]; ty = _e1d(fx, vars_, tx)
                        # Línea de trayectoria degradada
                        for i in range(len(tx)-1):
                            alpha = 0.3 + 0.7*(i/(max(len(tx)-1,1)))
                            ax.plot(tx[i:i+2], ty[i:i+2], '-', color=CT,
                                    lw=2, alpha=alpha, zorder=4)
                        ax.scatter(tx, ty, c=range(len(tx)), cmap='YlOrRd',
                                   s=40, zorder=5, edgecolors='none', label="Iteraciones")
                        # Punto inicial
                        ax.plot(tx[0], ty[0], 'o', color="#00e5ff",
                                markersize=9, label=f"x₀ = {tx[0]:.4f}", zorder=6)
                        # Punto óptimo
                        ax.plot(tx[-1], ty[-1], 'D', color=CM, markersize=7,
                                label=f"x* = {tx[-1]:.4f}, f* = {ty[-1]:.4f}", zorder=7)
                        # Línea vertical al óptimo
                        ax.axvline(tx[-1], color=CM, lw=0.8, linestyle='--', alpha=0.5)
                        ax.axhline(ty[-1], color=CM, lw=0.8, linestyle='--', alpha=0.5)
                        # Anotación
                        ax.annotate(f"Óptimo ({tx[-1]:.3f}, {ty[-1]:.3f})",
                                    xy=(tx[-1], ty[-1]),
                                    xytext=(tx[-1]+0.05*(hi-lo), ty[-1]+0.05*(max(yg)-min(yg))),
                                    color="#ffd700", fontsize=8,
                                    arrowprops=dict(arrowstyle='->', color='#ffd700', lw=1.2))
                    ax.set_xlabel(vars_[0]); ax.set_ylabel("f(x)")
                else:
                    g  = np.linspace(min(lo,hi), max(lo,hi), 120)
                    X1, X2 = np.meshgrid(g, g)
                    Z  = np.asarray(_callable(fx, vars_[:2])(X1, X2), dtype=float)
                    cf = ax.contourf(X1, X2, Z, levels=30, cmap='coolwarm', alpha=0.88)
                    ax.contour(X1, X2, Z, levels=12, colors='white', alpha=0.2, linewidths=0.5)
                    fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.04).ax.tick_params(colors="#7aaaea")
                    if traj and len(traj) >= 2 and traj[0].size > 0:
                        px, py = traj[0], traj[1]
                        ax.plot(px, py, '-', color=CT, lw=2, alpha=0.8, zorder=4)
                        ax.scatter(px, py, c=range(len(px)), cmap='YlOrRd',
                                   s=45, zorder=5, edgecolors='none', label="Iteraciones")
                        ax.plot(px[0],  py[0],  'o', color="#00e5ff",
                                markersize=9, label="x₀", zorder=6)
                        ax.plot(px[-1], py[-1], 'D', color=CM,
                                markersize=7, label="x*", zorder=7)
                        ax.annotate(f"x*=({px[-1]:.3f},{py[-1]:.3f})",
                                    xy=(px[-1], py[-1]),
                                    xytext=(px[-1]+0.05*(hi-lo), py[-1]+0.05*(hi-lo)),
                                    color="#ffd700", fontsize=8,
                                    arrowprops=dict(arrowstyle='->', color='#ffd700', lw=1.2))
                    ax.set_xlabel(vars_[0]); ax.set_ylabel(vars_[1])

                ax.set_title(f"2D — {metodo}", fontsize=10, pad=8)
                leg2d = ax.legend(facecolor="#141e36", edgecolor="#2a3e6e",
                          labelcolor="#d8e8f8", fontsize=8,
                          loc="upper left",
                          ncol=1, framealpha=0.88,
                          borderaxespad=0.5)
                try: leg2d.set_draggable(True)   # leyenda arrastrable
                except: pass
                fig.tight_layout()
                try: self.canvas2d.draw()
                except: pass
                # Botón colapsar/expandir leyenda 2D
                try: self._add_collapsible_legend(self.canvas2d, ax)
                except: pass
            except Exception as e:
                self.lbl_2d.setText(f"⚠ 2D: {e}"); self.lbl_2d.setVisible(True)
                self.canvas2d.setVisible(False)
        elif self.canvas2d:
            # fallback a métodos del canvas
            try:
                self.lbl_2d.setVisible(False); self.canvas2d.setVisible(True)
                if len(vars_)==1:
                    xg=_grid(lo,hi,400); yg=_e1d(fx,vars_,xg)
                    tx=traj[0] if traj else np.array([])
                    ty=_e1d(fx,vars_,tx) if tx.size else np.array([])
                    if hasattr(self.canvas2d,"animate_1d") and tx.size>1:
                        self.canvas2d.animate_1d(xg,yg,tx,ty,title=f"2D – {metodo}")
                    else:
                        self.canvas2d.plot_1d(xg,yg,tx,ty,title=f"2D – {metodo}")
                else:
                    g=np.linspace(min(lo,hi),max(lo,hi),110); X1,X2=np.meshgrid(g,g)
                    Z=np.asarray(_callable(fx,vars_[:2])(X1,X2),dtype=float)
                    px=traj[0] if len(traj)>0 else np.array([])
                    py=traj[1] if len(traj)>1 else np.array([])
                    if hasattr(self.canvas2d,"animate_2d_contour") and px.size>1:
                        self.canvas2d.animate_2d_contour(X1,X2,Z,px,py,title=f"2D – {metodo}")
                    else:
                        self.canvas2d.plot_2d_contour(X1,X2,Z,px,py,title=f"2D – {metodo}")
            except Exception as e:
                self.lbl_2d.setText(f"⚠ 2D: {e}"); self.lbl_2d.setVisible(True)
                self.canvas2d.setVisible(False)

        # ── 3D ────────────────────────────────────────────────────────────────
        if self.canvas3d and hasattr(self.canvas3d, "fig"):
            try:
                self.lbl_3d.setVisible(False); self.canvas3d.setVisible(True)
                fig3 = self.canvas3d.fig; fig3.clear()
                fig3.patch.set_facecolor(BG)
                ax3 = fig3.add_subplot(111, projection='3d')
                _style_ax3d(ax3)

                if len(vars_) == 1:
                    xg = _grid(lo, hi, 150)
                    yg = _e1d(fx, vars_, xg)
                    it = np.arange(xg.size, dtype=float)
                    ax3.plot(it, xg, yg, color=CC, lw=1.8, label="f(x)", alpha=0.9)
                    if traj and traj[0].size > 0:
                        tx = traj[0]; ty = _e1d(fx, vars_, tx)
                        pit = np.arange(tx.size, dtype=float)
                        ax3.plot(pit, tx, ty, 'o-', color=CT, lw=2.5,
                                 markersize=6, label="Trayectoria", zorder=5)
                        ax3.plot([pit[0]],  [tx[0]],  [ty[0]],  'o',
                                 color="#00e5ff", markersize=9, label="x₀", zorder=6)
                        ax3.plot([pit[-1]], [tx[-1]], [ty[-1]], 'D',
                                 color=CM, markersize=7, label="x*", zorder=7)
                    ax3.set_xlabel("iter."); ax3.set_ylabel(vars_[0]); ax3.set_zlabel("f(x)")
                else:
                    g  = np.linspace(min(lo,hi), max(lo,hi), 60)
                    X1, X2 = np.meshgrid(g, g)
                    f2c = _callable(fx, vars_[:2])
                    Z   = np.asarray(f2c(X1, X2), dtype=float)
                    ax3.plot_surface(X1, X2, Z, cmap='coolwarm', alpha=0.80,
                                     linewidth=0, antialiased=True, rcount=60, ccount=60)
                    if traj and len(traj)>=2 and traj[0].size>0:
                        px, py = traj[0], traj[1]
                        pz = np.asarray(f2c(px, py), dtype=float)
                        # Sombra en el suelo z=min
                        zmin = float(Z.min())
                        ax3.plot(px, py, [zmin]*len(px), '--',
                                 color=CT, lw=1.2, alpha=0.35, zorder=3)
                        ax3.plot(px, py, pz, 'o-', color=CT, lw=2.5,
                                 markersize=6, label="Trayectoria", zorder=5)
                        ax3.plot([px[0]],  [py[0]],  [pz[0]],  'o',
                                 color="#00e5ff", markersize=9, label="x₀", zorder=6)
                        ax3.plot([px[-1]], [py[-1]], [pz[-1]], 'D',
                                 color=CM, markersize=7, label="x*", zorder=7)
                    ax3.set_xlabel(vars_[0]); ax3.set_ylabel(vars_[1]); ax3.set_zlabel("f(x)")

                ax3.set_title(f"3D — {metodo}", fontsize=10, color="#d8e8f8", pad=10)
                _h3, _l3 = ax3.get_legend_handles_labels()
                if _h3:
                    leg3d = fig3.legend(_h3, _l3,
                               facecolor="#141e36", edgecolor="#2a3e6e",
                               labelcolor="#d8e8f8", fontsize=8,
                               loc="upper left",
                               bbox_to_anchor=(0.02, 0.97),
                               bbox_transform=fig3.transFigure,
                               ncol=1, framealpha=0.88,
                               borderaxespad=0.5)
                    try: leg3d.set_draggable(True)
                    except: pass
                ax3.view_init(elev=28, azim=-55)
                fig3.tight_layout()
                try: self.canvas3d.draw()
                except: pass
                # Botón colapsar/expandir leyenda 3D
                try: self._add_collapsible_legend(self.canvas3d, ax3)
                except: pass

                # Rotación automática via QTimer
                self._rot_angle = -55
                if hasattr(self, "_rot_timer") and self._rot_timer:
                    try: self._rot_timer.stop()
                    except: pass
                self._rot_ax3   = ax3
                self._rot_canvas = self.canvas3d
                self._rot_timer = QtCore.QTimer(self)
                def _rotate():
                    self._rot_angle = (self._rot_angle + 0.8) % 360
                    try:
                        self._rot_ax3.view_init(elev=28, azim=self._rot_angle)
                        self._rot_canvas.draw_idle()
                    except: self._rot_timer.stop()
                self._rot_timer.timeout.connect(_rotate)
                self._rot_timer.start(40)   # ~25 fps

            except Exception as e:
                self.lbl_3d.setText(f"⚠ 3D: {e}"); self.lbl_3d.setVisible(True)
                self.canvas3d.setVisible(False)
        elif self.canvas3d:
            # fallback
            try:
                self.lbl_3d.setVisible(False); self.canvas3d.setVisible(True)
                if len(vars_)==1:
                    xg=_grid(lo,hi,200); yg=_e1d(fx,vars_,xg)
                    tx=traj[0] if traj else np.array([]); it=np.arange(xg.size,dtype=float)
                    if hasattr(self.canvas3d,"set_curve3d"):
                        tx2=traj[0] if traj else np.array([]); ty2=_e1d(fx,vars_,tx2) if tx2.size else np.array([])
                        self.canvas3d.set_curve3d(it,xg,yg,title=f"3D – {metodo}",
                            path_it=np.arange(tx2.size,dtype=float),path_x=tx2,path_z=ty2)
                else:
                    g=np.linspace(min(lo,hi),max(lo,hi),55); X1,X2=np.meshgrid(g,g)
                    f2=_callable(fx,vars_[:2]); Z=np.asarray(f2(X1,X2),dtype=float)
                    px=traj[0] if len(traj)>0 else np.array([])
                    py=traj[1] if len(traj)>1 else np.array([])
                    pz=np.asarray(f2(px,py),dtype=float) if px.size and py.size else np.array([])
                    self.canvas3d.set_surface_and_path(X1,X2,Z,px,py,pz)
                try: self.canvas3d.start_rotation()
                except: pass
            except Exception as e:
                self.lbl_3d.setText(f"⚠ 3D: {e}"); self.lbl_3d.setVisible(True)
                self.canvas3d.setVisible(False)



    # ── GRÁFICAS MULTIOBJETIVO ────────────────────────────────────────────────
    # ── GRÁFICAS MULTIOBJETIVO ───────────────────────────────────────────────
    def _render_plots_multiobj(self, metodo, fx, hist, lo, hi):
        """
        Renderiza gráficas para problemas multiobjetivo.
        Flujo: f(x) mono-objetivo → Escalarización φ(x,α) → Algoritmo → Resultados MO
        La tabla de α se muestra como referencia dentro de la gráfica.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        BG = "#0a1228"; CC = "#4a9eff"; CT = "#ff6b35"; CM = "#ffd700"
        C2 = "#3cff8a"; CGRID = "#1a2e58"

        def _sax(ax):
            ax.set_facecolor(BG)
            for sp in ax.spines.values(): sp.set_color("#2a3e6e")
            ax.tick_params(colors="#7aaaea", labelsize=8)
            ax.xaxis.label.set_color("#7aaaea"); ax.yaxis.label.set_color("#7aaaea")
            ax.title.set_color("#d8e8f8")
            ax.grid(True, color=CGRID, alpha=0.4, linewidth=0.6)

        if not hist:
            return

        # Separar filas de iteración vs filas de tabla de referencia
        iter_rows = [r for r in hist if r.get("k") not in ("──", "ref", None) and
                     isinstance(r.get("k"), int)]
        tabla_ref = getattr(self, '_mo_tabla_ref', [])
        a_val     = getattr(self, '_mo_alpha', 0.5)
        x_c       = getattr(self, '_mo_x_c', (lo+hi)/2)
        f1_call   = getattr(self, '_mo_f1_call', None)
        f2_call   = getattr(self, '_mo_f2_call', None)

        # ── CANVAS 2D ─────────────────────────────────────────────────────────
        if self.canvas2d and hasattr(self.canvas2d, "fig"):
            try:
                self.lbl_2d.setVisible(False); self.canvas2d.setVisible(True)
                fig = self.canvas2d.fig; fig.clear()
                fig.patch.set_facecolor(BG)

                # ── BISECCIÓN o SECCIÓN DORADA ────────────────────────────────
                if metodo in ("MO — Bisección", "MO — Sección Dorada"):
                    x_vals = np.array([r.get("x_k", float("nan")) for r in iter_rows])
                    f_vals = np.array([r.get("φ(x_k,α)", float("nan")) for r in iter_rows])
                    f1_vals= np.array([r.get("f₁(x_k)", float("nan")) for r in iter_rows])
                    f2_vals= np.array([r.get("f₂(x_k)", float("nan")) for r in iter_rows])
                    b_a    = np.array([r.get("b−a",     float("nan")) for r in iter_rows])
                    algo   = "Bisección" if "Bisección" in metodo else "Sección Dorada"

                    # Sub-gráfica izq: φ(x,α) + trayectoria del algoritmo
                    ax1 = fig.add_subplot(121); _sax(ax1)
                    if f1_call is not None:
                        xg  = np.linspace(lo, hi, 400)
                        try:
                            f1g = np.array([float(f1_call(xv)) for xv in xg])
                            f2g = np.array([(float(xv)-x_c)**2 for xv in xg])
                            yg  = a_val*f1g + (1-a_val)*f2g
                            ax1.plot(xg, yg, color=CC, lw=2.2, zorder=2,
                                     label=f"φ(x,α={a_val:.2f}) = α·f₁+(1-α)·f₂")
                            ax1.plot(xg, f1g, color="#8080ff", lw=1.2,
                                     linestyle="--", alpha=0.6, zorder=2,
                                     label="f₁(x)  (mono-objetivo original)")
                            ax1.plot(xg, f2g, color="#ff80a0", lw=1.2,
                                     linestyle=":", alpha=0.6, zorder=2,
                                     label=f"f₂(x)=(x−{x_c:.1f})²  (objetivo auxiliar)")
                        except Exception:
                            pass

                    if x_vals.size > 0 and not np.all(np.isnan(x_vals)):
                        ax1.plot(x_vals, f_vals, "o-", color=CT, lw=1.8, markersize=5,
                                 label=f"Iteraciones {algo}", zorder=4)
                        ax1.plot(x_vals[0],  f_vals[0],  "o",  color="#00e5ff",
                                 markersize=10, label=f"x₀={x_vals[0]:.3f}", zorder=6)
                        ax1.plot(x_vals[-1], f_vals[-1], "D",  color=CM,
                                 markersize=10, label=f"x*={x_vals[-1]:.5f}", zorder=7)
                        ax1.axvline(x_vals[-1], color=CM, lw=0.8,
                                    linestyle=":", alpha=0.5)
                    ax1.set_xlabel("x"); ax1.set_ylabel("f(x)")
                    ax1.set_title(
                        f"Función Escalarizada  φ(x,α={a_val:.2f})\n"
                        f"f(x) mono-obj → Problema MO → {algo}",
                        fontsize=8, pad=6)
                    ax1.legend(facecolor="#141e36", edgecolor="#2a3e6e",
                               labelcolor="#d8e8f8", fontsize=6.5, framealpha=0.88)

                    # Sub-gráfica der: espacio objetivo f₁ vs f₂ + frente Pareto ref
                    ax2 = fig.add_subplot(122); _sax(ax2)
                    # Frente de Pareto de referencia (tabla)
                    if tabla_ref:
                        f1r = [r.get("f₁(x*)","") for r in tabla_ref]
                        f2r = [r.get("f₂(x*)","") for r in tabla_ref]
                        alr = [r.get("α","") for r in tabla_ref]
                        try:
                            f1r_f = [float(v) for v in f1r]
                            f2r_f = [float(v) for v in f2r]
                            ax2.plot(f1r_f, f2r_f, "s--", color=C2, lw=1.5,
                                     markersize=7, zorder=3,
                                     label="Tabla ref. α (Frente Pareto)")
                            for i, albl in enumerate(alr):
                                ax2.annotate(f"α={albl}",
                                             xy=(f1r_f[i], f2r_f[i]),
                                             xytext=(f1r_f[i]+0.02, f2r_f[i]+0.02),
                                             color="#90ff90", fontsize=6.5)
                        except Exception:
                            pass
                    # Punto óptimo actual (α fijo)
                    if x_vals.size > 0 and f1_call is not None:
                        try:
                            xopt = float(x_vals[-1])
                            f1opt= float(f1_call(xopt))
                            f2opt= float((xopt-x_c)**2)
                            ax2.scatter([f1opt], [f2opt], color=CM, s=100,
                                        zorder=7,
                                        label=f"Sol. α={a_val:.2f}  x*={xopt:.4f}")
                        except Exception:
                            pass
                    ax2.set_xlabel("f₁(x)"); ax2.set_ylabel("f₂(x)")
                    ax2.set_title(
                        "Espacio Objetivo\n(Tabla α = referencia Frente Pareto)",
                        fontsize=8, pad=6)
                    ax2.legend(facecolor="#141e36", edgecolor="#2a3e6e",
                               labelcolor="#d8e8f8", fontsize=6.5, framealpha=0.88)

                # ── FRENTE DE PARETO ──────────────────────────────────────────
                elif metodo == "MO — Frente de Pareto":
                    f1v = np.array([r.get("f1(x*)", float("nan")) for r in hist])
                    f2v = np.array([r.get("f2(x*)", float("nan")) for r in hist])
                    alp = np.array([r.get("α",      float("nan")) for r in hist])
                    xv  = np.array([r.get("x*(α)",  float("nan")) for r in hist])

                    ax1 = fig.add_subplot(121); _sax(ax1)
                    valid = ~(np.isnan(f1v)|np.isnan(f2v))
                    if valid.any():
                        sc = ax1.scatter(f1v[valid], f2v[valid], c=alp[valid],
                                         cmap="plasma", s=24, zorder=4,
                                         edgecolors="none")
                        cbar = fig.colorbar(sc, ax=ax1, fraction=0.04, pad=0.04)
                        cbar.ax.tick_params(colors="#7aaaea")
                        cbar.set_label("α", color="#7aaaea")
                    ax1.set_xlabel("f₁(x)  [mono-objetivo original]")
                    ax1.set_ylabel("f₂(x)  [objetivo auxiliar centrado]")
                    ax1.set_title(
                        "Frente de Pareto\n"

                        "Cada punto = solución óptima para un α distinto",
                        fontsize=8, pad=6)

                    ax2 = fig.add_subplot(122); _sax(ax2)
                    if valid.any():
                        ax2.plot(alp[valid], f1v[valid], "-", color=CC, lw=2,
                                 label="f₁(x*(α))")
                        ax2.plot(alp[valid], f2v[valid], "-", color=CT, lw=2,
                                 label="f₂(x*(α))")
                        ax2.plot(alp[valid], xv[valid],  "--", color=CM, lw=1.8,
                                 label="x*(α)")
                    ax2.set_xlabel("α"); ax2.set_ylabel("Valor")
                    ax2.set_title("Variación con α", fontsize=8, pad=6)
                    ax2.legend(facecolor="#141e36", edgecolor="#2a3e6e",
                               labelcolor="#d8e8f8", fontsize=7, framealpha=0.88)

                # ── JACOBIANO ─────────────────────────────────────────────────
                elif metodo == "MO — Análisis Jacobiano":
                    xs  = [r.get("x",  0.0) for r in hist]
                    jf  = [r.get("J_f=f'(x)",  0.0) for r in hist]
                    jg  = [r.get("J_g=2(x-c)", 0.0) for r in hist]
                    nJ  = [r.get("‖J‖",  0.0) for r in hist]
                    p_ok= [r.get("¿Pareto-opt?","") == "✓ Sí" for r in hist]

                    ax1 = fig.add_subplot(121); _sax(ax1)
                    ax1.plot(xs, jf, "o-", color=CC, lw=2, markersize=6,
                             label="J[f₁]=f'(x)  (gradiente mono-obj)", zorder=4)
                    ax1.plot(xs, jg, "s-", color=CT, lw=2, markersize=6,
                             label="J[f₂]=2(x−c)  (gradiente aux)", zorder=4)
                    ax1.axhline(0, color="#7aaaea", lw=0.7, linestyle="--", alpha=0.5)
                    ax1.set_xlabel("x"); ax1.set_ylabel("Componente Jacobiana")
                    ax1.set_title("Jacobiano J(x) = [f'(x), 2(x−c)]", fontsize=9, pad=6)
                    ax1.legend(facecolor="#141e36", edgecolor="#2a3e6e",
                               labelcolor="#d8e8f8", fontsize=7, framealpha=0.88)

                    ax2 = fig.add_subplot(122); _sax(ax2)
                    w_ = (xs[-1]-xs[0])/(len(xs)+1)*0.85 if len(xs)>1 else 0.15
                    colors_bar = [CM if p else "#3060a0" for p in p_ok]
                    ax2.bar(xs, nJ, width=w_, color=colors_bar, edgecolor="none", zorder=4)
                    ax2.set_xlabel("x"); ax2.set_ylabel("‖J(x)‖")
                    ax2.set_title("Norma del Jacobiano  (dorado = Pareto-óptimo)",
                                  fontsize=9, pad=6)
                    ax2.legend(handles=[
                        Patch(facecolor=CM,        label="Pareto-óptimo"),
                        Patch(facecolor="#3060a0",  label="No Pareto-óptimo"),
                    ], facecolor="#141e36", edgecolor="#2a3e6e",
                       labelcolor="#d8e8f8", fontsize=7, framealpha=0.88)

                fig.tight_layout()
                try: self.canvas2d.draw()
                except: pass
            except Exception as e:
                self.lbl_2d.setText(f"⚠ MO 2D: {e}"); self.lbl_2d.setVisible(True)
                self.canvas2d.setVisible(False)

        # ── CANVAS 3D: superficie φ(x,α) con curva de Pareto ─────────────────
        if self.canvas3d and hasattr(self.canvas3d, "fig"):
            try:
                self.lbl_3d.setVisible(False); self.canvas3d.setVisible(True)
                fig3 = self.canvas3d.fig; fig3.clear()
                fig3.patch.set_facecolor(BG)
                ax3  = fig3.add_subplot(111, projection="3d")
                ax3.set_facecolor(BG)
                for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
                    pane.fill = False; pane.set_edgecolor("#1a2e58")
                ax3.tick_params(colors="#7aaaea", labelsize=7)

                xg = np.linspace(lo, hi, 50)
                ag = np.linspace(0.0, 1.0, 50)
                X3, A3 = np.meshgrid(xg, ag)
                if f1_call is not None:
                    try:
                        F1_3 = np.vectorize(lambda xv: float(f1_call(xv)))(X3)
                        F2_3 = (X3 - x_c)**2
                        Z3   = A3*F1_3 + (1-A3)*F2_3
                        ax3.plot_surface(X3, A3, Z3, cmap="coolwarm", alpha=0.78,
                                         linewidth=0, antialiased=True,
                                         rcount=40, ccount=40)
                        # Curva Pareto (mínimo de φ para cada α)
                        a_pf = np.linspace(0,1,120)
                        # Encontrar x*(α) numéricamente via min en grid
                        xg_pf = np.linspace(lo, hi, 400)
                        f1_pf = np.array([float(f1_call(xv)) for xv in xg_pf])
                        f2_pf = (xg_pf - x_c)**2
                        x_pf  = []
                        f_pf  = []
                        for a_p in a_pf:
                            z_  = a_p*f1_pf + (1-a_p)*f2_pf
                            idx = np.argmin(z_)
                            x_pf.append(xg_pf[idx]); f_pf.append(z_[idx])
                        ax3.plot(x_pf, a_pf, f_pf, color=CM, lw=3,
                                 label="Frente de Pareto  x*(α)", zorder=6)
                        # Marcar punto actual si hay iteraciones
                        if iter_rows:
                            xopt = iter_rows[-1].get("x_k", float("nan"))
                            try:
                                fopt = a_val*float(f1_call(xopt)) + (1-a_val)*(float(xopt)-x_c)**2
                                ax3.scatter([xopt],[a_val],[fopt],
                                            color=C2, s=80, zorder=7,
                                            label=f"x*(α={a_val:.2f})={xopt:.4f}")
                            except Exception:
                                pass
                    except Exception as e3:
                        ax3.set_title(f"Error 3D: {e3}", color="#ff8080")
                ax3.set_xlabel("x"); ax3.set_ylabel("α")
                ax3.set_zlabel("φ(x,α)")
                ax3.set_title(
                    "Superficie φ(x,α) = α·f₁(x)+(1−α)·f₂(x)\n"

                    "(Escalarización del problema MO)",
                    fontsize=8, color="#d8e8f8", pad=10)
                _h3, _l3 = ax3.get_legend_handles_labels()
                if _h3:
                    ax3.legend(_h3, _l3, facecolor="#141e36", edgecolor="#2a3e6e",
                               labelcolor="#d8e8f8", fontsize=7,
                               loc="upper left", framealpha=0.88)
                ax3.view_init(elev=28, azim=-55)
                fig3.tight_layout()
                try: self.canvas3d.draw()
                except: pass

                self._rot_angle = -55
                if hasattr(self,"_rot_timer") and self._rot_timer:
                    try: self._rot_timer.stop()
                    except: pass
                self._rot_ax3 = ax3; self._rot_canvas = self.canvas3d
                self._rot_timer = QtCore.QTimer(self)
                def _rotate():
                    self._rot_angle = (self._rot_angle+0.8)%360
                    try: self._rot_ax3.view_init(elev=28,azim=self._rot_angle); self._rot_canvas.draw_idle()
                    except: self._rot_timer.stop()
                self._rot_timer.timeout.connect(_rotate)
                self._rot_timer.start(40)
            except Exception as e:
                self.lbl_3d.setText(f"⚠ MO 3D: {e}"); self.lbl_3d.setVisible(True)
                self.canvas3d.setVisible(False)

    def _traj(self, vars_, hist, x0):
        """Extrae trayectoria de puntos del historial, compatible con arrays numpy."""

        def _first_val(rec, *keys):
            """Devuelve el primer valor no-None de las claves dadas, seguro con arrays."""
            for k in keys:
                v = rec.get(k)
                if v is None:
                    continue
                # numpy array: no se puede evaluar como bool directamente
                if isinstance(v, np.ndarray):
                    return v
                # escalar Python / numpy scalar
                if isinstance(v, (int, float, np.number)):
                    return v
                # lista o tupla
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return v
            return None

        if not hist:
            if x0:
                if len(vars_) == 1:
                    return [np.asarray([x0[0]], dtype=float)]
                if len(vars_) >= 2 and len(x0) >= 2:
                    return [np.asarray([x0[0]]), np.asarray([x0[1]])]
            return []

        if len(vars_) == 1:
            xs = []
            for r in hist:
                v = _first_val(r, "x_k", "x", "lambda_k")
                if v is None:
                    continue
                if isinstance(v, (int, float, np.number)):
                    xs.append(float(v))
                elif isinstance(v, np.ndarray):
                    if v.size == 1:
                        xs.append(float(v.flat[0]))
                    elif v.size > 1:
                        xs.append(float(v.flat[0]))   # tomar primera coordenada
                elif isinstance(v, (list, tuple)) and len(v) >= 1:
                    try: xs.append(float(v[0]))
                    except: pass
            return [np.asarray(xs, dtype=float)] if xs else []

        # Multivariable: extraer x1, x2
        x1s, x2s = [], []
        for r in hist:
            v = _first_val(r, "x_k", "x")
            if v is None:
                continue
            if isinstance(v, np.ndarray) and v.size >= 2:
                try: x1s.append(float(v.flat[0])); x2s.append(float(v.flat[1]))
                except: pass
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                try: x1s.append(float(v[0])); x2s.append(float(v[1]))
                except: pass
        if x1s and x2s:
            return [np.asarray(x1s, dtype=float), np.asarray(x2s, dtype=float)]
        if x0 and len(x0) >= 2:
            return [np.asarray([x0[0]]), np.asarray([x0[1]])]
        return []

    # ── EXPORTAR ──────────────────────────────────────────────────────────────
    def _tdata(self):
        cols=[self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
        rows=[[("" if (it:=self.table.item(r,c)) is None else it.text())
               for c in range(self.table.columnCount())] for r in range(self.table.rowCount())]
        return cols,rows

    def _splots(self):
        """Guarda las gráficas actuales (canvas) como archivos temporales PNG."""
        imgs=[]
        for c,lb in [(self.canvas2d,"2D"),(self.canvas3d,"3D")]:
            if c and c.isVisible():
                try:
                    f=tempfile.NamedTemporaryFile(suffix=".png",delete=False); f.close()
                    c.fig.savefig(f.name,dpi=110,bbox_inches="tight",facecolor="#0a1228")
                    imgs.append((f.name,f"Gráfica {lb}"))
                except: pass
        return imgs

    def _generate_plot_png(self, entry: dict) -> list:
        """Genera PNGs 2D y 3D de alta calidad para exportación, con misma estética que pantalla."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        imgs  = []
        fx    = entry["fx"]
        vrs   = entry["vars"]
        lo    = entry["lo"]; hi = entry["hi"]
        traj  = entry["traj"]
        metod = entry["metodo"]

        BG = "#0a1228"; CC = "#4a9eff"; CT = "#ff6b35"; CM = "#ffd700"; CG = "#1a2e58"

        def _sax2(ax):
            ax.set_facecolor(BG)
            for sp in ax.spines.values(): sp.set_color("#2a3e6e")
            ax.tick_params(colors="#7aaaea", labelsize=8)
            ax.xaxis.label.set_color("#7aaaea"); ax.yaxis.label.set_color("#7aaaea")
            ax.title.set_color("#d8e8f8"); ax.grid(True, color=CG, alpha=0.4, lw=0.6)

        def _sax3(ax):
            ax.set_facecolor(BG)
            for pn in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pn.fill=False; pn.set_edgecolor("#1a2e58")
            ax.tick_params(colors="#7aaaea", labelsize=7)
            for lbl in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
                lbl.set_color("#7aaaea")

        # ── 2D ────────────────────────────────────────────────────────────────
        try:
            fig2, ax = plt.subplots(figsize=(8,5), facecolor=BG)
            _sax2(ax)

            if len(vrs) == 1:
                xg = _grid(lo, hi, 500); yg = _e1d(fx, vrs, xg)
                ax.plot(xg, yg, color=CC, lw=2.2, label=f"f(x) = {fx}", zorder=2)
                if traj and traj[0].size > 0:
                    tx = traj[0]; ty = _e1d(fx, vrs, tx)
                    for i in range(len(tx)-1):
                        al = 0.3 + 0.7*(i/max(len(tx)-1,1))
                        ax.plot(tx[i:i+2], ty[i:i+2], '-', color=CT, lw=2, alpha=al, zorder=4)
                    ax.scatter(tx, ty, c=range(len(tx)), cmap='YlOrRd',
                               s=45, zorder=5, edgecolors='none', label="Iteraciones")
                    ax.plot(tx[0],  ty[0],  'o', color="#00e5ff", ms=9,
                            label=f"x₀={tx[0]:.4f}", zorder=6)
                    ax.plot(tx[-1], ty[-1], '*', color=CM, ms=16,
                            label=f"x*={tx[-1]:.4f}, f*={ty[-1]:.4f}", zorder=7)
                    ax.axvline(tx[-1], color=CM, lw=0.8, ls='--', alpha=0.5)
                    ax.axhline(ty[-1], color=CM, lw=0.8, ls='--', alpha=0.5)
                    yrng = max(yg)-min(yg) if max(yg)!=min(yg) else 1
                    ax.annotate(f"x*={tx[-1]:.4f}  f*={ty[-1]:.4f}",
                                xy=(tx[-1], ty[-1]),
                                xytext=(tx[-1]+0.06*(hi-lo), ty[-1]+0.06*yrng),
                                color=CM, fontsize=8,
                                arrowprops=dict(arrowstyle='->', color=CM, lw=1.2))
                ax.set_xlabel(vrs[0]); ax.set_ylabel("f(x)")
            else:
                g = np.linspace(min(lo,hi), max(lo,hi), 130)
                X1, X2 = np.meshgrid(g, g)
                Z = np.asarray(_callable(fx, vrs[:2])(X1, X2), dtype=float)
                cf = ax.contourf(X1, X2, Z, levels=30, cmap='coolwarm', alpha=0.88)
                ax.contour(X1, X2, Z, levels=12, colors='white', alpha=0.2, lw=0.5)
                fig2.colorbar(cf, ax=ax, fraction=0.035, pad=0.04).ax.tick_params(colors="#7aaaea")
                if traj and len(traj)>=2 and traj[0].size>0:
                    px, py = traj[0], traj[1]
                    ax.plot(px, py, '-', color=CT, lw=2, alpha=0.8, zorder=4)
                    ax.scatter(px, py, c=range(len(px)), cmap='YlOrRd',
                               s=50, zorder=5, edgecolors='none', label="Iteraciones")
                    ax.plot(px[0],  py[0],  'o', color="#00e5ff", ms=9, label="x₀", zorder=6)
                    ax.plot(px[-1], py[-1], '*', color=CM, ms=16, label="x*", zorder=7)
                    rng = hi-lo if hi!=lo else 1
                    ax.annotate(f"x*=({px[-1]:.3f},{py[-1]:.3f})",
                                xy=(px[-1], py[-1]),
                                xytext=(px[-1]+0.06*rng, py[-1]+0.06*rng),
                                color=CM, fontsize=8,
                                arrowprops=dict(arrowstyle='->', color=CM, lw=1.2))
                ax.set_xlabel(vrs[0]); ax.set_ylabel(vrs[1])

            ax.set_title(f"2D — {metod}", fontsize=11, pad=8)
            ax.legend(facecolor="#141e36", edgecolor="#2a3e6e",
                      labelcolor="#d8e8f8", fontsize=8,
                      loc="upper left",
                      ncol=1, framealpha=0.88,
                      borderaxespad=0.5)
            fig2.tight_layout()
            f2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False); f2.close()
            fig2.savefig(f2.name, dpi=130, bbox_inches="tight", facecolor=BG)
            plt.close(fig2); imgs.append((f2.name, "Gráfica 2D"))
        except Exception as e:
            print(f"[plot2d export err] {e}")

        # ── 3D ────────────────────────────────────────────────────────────────
        try:
            for azim in (-55, 30, 110):   # 3 vistas estáticas del 3D
                fig3 = plt.figure(figsize=(7, 5), facecolor=BG)
                ax3  = fig3.add_subplot(111, projection='3d'); _sax3(ax3)

                if len(vrs) == 1:
                    xg = _grid(lo, hi, 130); yg = _e1d(fx, vrs, xg)
                    it = np.arange(xg.size, dtype=float)
                    ax3.plot(it, xg, yg, color=CC, lw=1.8, label="f(x)", alpha=0.9)
                    if traj and traj[0].size > 0:
                        tx = traj[0]; ty = _e1d(fx, vrs, tx)
                        pit = np.arange(tx.size, dtype=float)
                        ax3.plot(pit, tx, ty, 'o-', color=CT, lw=2.5,
                                 markersize=6, label="Trayectoria", zorder=5)
                        ax3.plot([pit[0]],  [tx[0]],  [ty[0]],  'o', color="#00e5ff", ms=9, label="x₀")
                        ax3.plot([pit[-1]], [tx[-1]], [ty[-1]], '*', color=CM, ms=14, label="x*")
                    ax3.set_xlabel("iter."); ax3.set_ylabel(vrs[0]); ax3.set_zlabel("f(x)")
                else:
                    g = np.linspace(min(lo,hi), max(lo,hi), 60)
                    X1, X2 = np.meshgrid(g, g)
                    f2c = _callable(fx, vrs[:2])
                    Z   = np.asarray(f2c(X1, X2), dtype=float)
                    ax3.plot_surface(X1, X2, Z, cmap='coolwarm', alpha=0.80,
                                     linewidth=0, antialiased=True, rcount=60, ccount=60)
                    if traj and len(traj)>=2 and traj[0].size>0:
                        px, py = traj[0], traj[1]
                        pz = np.asarray(f2c(px, py), dtype=float)
                        zmin = float(Z.min())
                        ax3.plot(px, py, [zmin]*len(px), '--', color=CT, lw=1.2, alpha=0.35)
                        ax3.plot(px, py, pz, 'o-', color=CT, lw=2.5,
                                 markersize=6, label="Trayectoria", zorder=5)
                        ax3.plot([px[0]],  [py[0]],  [pz[0]],  'o', color="#00e5ff", ms=9, label="x₀")
                        ax3.plot([px[-1]], [py[-1]], [pz[-1]], '*', color=CM, ms=14, label="x*")
                    ax3.set_xlabel(vrs[0]); ax3.set_ylabel(vrs[1]); ax3.set_zlabel("f(x)")

                lbl_azim = {-55:"Vista Frontal", 30:"Vista Lateral", 110:"Vista Superior"}
                ax3.set_title(f"3D — {metod}  ({lbl_azim.get(azim,'')})",
                              fontsize=10, color="#d8e8f8", pad=10)
                _h3, _l3 = ax3.get_legend_handles_labels()
                if _h3:
                    fig3.legend(_h3, _l3,
                               facecolor="#141e36", edgecolor="#2a3e6e",
                               labelcolor="#d8e8f8", fontsize=8,
                               loc="upper left",
                               bbox_to_anchor=(0.02, 0.97),
                               bbox_transform=fig3.transFigure,
                               ncol=1, framealpha=0.88,
                               borderaxespad=0.5)
                ax3.view_init(elev=28, azim=azim)
                fig3.tight_layout()
                f3 = tempfile.NamedTemporaryFile(suffix=".png", delete=False); f3.close()
                fig3.savefig(f3.name, dpi=130, bbox_inches="tight", facecolor=BG)
                plt.close(fig3)
                imgs.append((f3.name, f"Gráfica 3D ({lbl_azim.get(azim,'')})"))
        except Exception as e:
            print(f"[plot3d export err] {e}")

        return imgs


    # ══════════════════════════════════════════════════════════════════════════
    # MÉTODOS DE DESCRIPCIÓN PARA PDF / EXCEL
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _describe_function(fx: str) -> str:
        """Descripción breve y matemática de la función."""
        fx_l = fx.lower().replace(" ", "")
        if any(k in fx_l for k in ("sin", "cos", "tan")):
            return (f"La función f(x) = {fx} es de tipo trigonométrico. "
                    "Este tipo de funciones presenta comportamiento oscilatorio periódico, "
                    "con múltiples máximos y mínimos locales distribuidos uniformemente. "
                    "Su análisis requiere considerar el intervalo de búsqueda con cuidado "
                    "para evitar convergencia a óptimos locales indeseados.")
        if any(k in fx_l for k in ("exp", "log", "ln")):
            return (f"La función f(x) = {fx} es de tipo exponencial o logarítmico. "
                    "Estas funciones presentan crecimiento/decrecimiento monótono o "
                    "con un único punto crítico, lo que generalmente facilita la convergencia "
                    "de los métodos de optimización. Su derivada es continua y bien definida "
                    "en el dominio de interés.")
        if "**" in fx_l or "sqrt" in fx_l:
            import re
            degs = [int(m) for m in re.findall(r"\*\*(\d+)", fx_l)]
            grado = max(degs) if degs else 2
            return (f"La función f(x) = {fx} es un polinomio de grado {grado}. "
                    f"{'Al ser de grado par, posee un mínimo o máximo global y posiblemente mínimos locales. ' if grado % 2 == 0 else 'Al ser de grado impar, la función no está acotada en ambos extremos del dominio. '}"
                    "La derivada es un polinomio de grado inferior, lo que permite calcular "
                    "analíticamente el gradiente necesario para los métodos de descenso.")
        if "/" in fx_l:
            return (f"La función f(x) = {fx} es de tipo racional. "
                    "Este tipo de funciones puede presentar discontinuidades (polos) en los "
                    "puntos donde el denominador se anula. Es fundamental verificar que el "
                    "intervalo de búsqueda no contenga dichos puntos para garantizar la "
                    "correcta aplicación de los métodos numéricos.")
        return (f"La función f(x) = {fx} es una función polinomial o algebraica. "
                "Presenta un comportamiento suave y diferenciable en todo su dominio, "
                "lo que la hace adecuada para la aplicación de métodos de optimización "
                "basados en gradiente. Su análisis permite identificar claramente "
                "puntos críticos mediante condiciones de primer y segundo orden.")

    @staticmethod
    def _describe_applicable_studies(fx: str) -> str:
        """Explica qué tipos de estudio son aplicables a la función y por qué."""
        fx_l = fx.lower().replace(" ", "")
        es_trig = any(k in fx_l for k in ("sin", "cos", "tan"))
        es_exp  = any(k in fx_l for k in ("exp", "log", "ln"))
        es_poly = "**" in fx_l or "sqrt" in fx_l or (not es_trig and not es_exp and "/" not in fx_l)

        partes = []
        partes.append(
            "Búsqueda Unidimensional: Aplicable cuando la función depende de una sola "
            "variable. Los métodos de sección áurea (Fibonacci) reducen iterativamente "
            "el intervalo de búsqueda sin necesitar derivadas, siendo robustos ante "
            "funciones no diferenciables o con ruido.")
        partes.append(
            "Búsqueda de Línea (Armijo / Wolfe): Adecuada para funciones diferenciables. "
            "Determina el tamaño de paso óptimo en la dirección de descenso. "
            "La condición de Armijo garantiza reducción suficiente del valor de la función, "
            "mientras que Wolfe añade la condición de curvatura para paso óptimo.")
        if es_trig or es_poly:
            partes.append(
                "Búsqueda Local: Explora el entorno de la solución actual mediante "
                "perturbaciones del punto corriente. Útil para funciones con múltiples "
                "extremos locales donde métodos de gradiente podrían quedar atrapados.")
        partes.append(
            "Optimización Multidimensional (MD): Cuando la función depende de varias "
            "variables. Los métodos de penalización y barrera transforman el problema "
            "con restricciones en uno sin restricciones. BFGS y Nelder-Mead son "
            "eficientes para funciones no lineales con múltiples variables.")
        return "  ".join(partes)

    @staticmethod
    def _justify_method(metodo: str, fx: str) -> str:
        """Explica por qué se seleccionó este método para esta función."""
        fx_l = fx.lower().replace(" ", "")
        justificaciones = {
            "Armijo": (
                "El método de Newton con condición de Armijo fue seleccionado porque la "
                f"función f(x) = {fx} es diferenciable en el dominio de análisis. "
                "Este método calcula la dirección de Newton usando el gradiente y la "
                "segunda derivada, luego aplica la condición de Armijo (condición de "
                "descenso suficiente) para elegir un tamaño de paso α que garantice "
                "una reducción real del valor de la función. La condición exige que "
                "f(x + α·d) ≤ f(x) + c·α·∇f(x)ᵀ·d, con c ∈ (0,1), lo que asegura "
                "convergencia evitando pasos demasiado grandes o pequeños."),
            "Wolfe": (
                "El método de Newton con condiciones de Wolfe fue seleccionado para "
                f"f(x) = {fx} por su mayor robustez comparado con Armijo. "
                "Además de la condición de descenso suficiente (Armijo), impone la "
                "condición de curvatura: |∇f(x+α·d)ᵀ·d| ≤ c₂·|∇f(x)ᵀ·d| con c₂ ∈ (c₁,1). "
                "Esto garantiza que el paso elegido no sea excesivamente corto, "
                "mejorando la velocidad de convergencia en funciones con curvatura variable."),
            "Fibonacci": (
                "La búsqueda de Fibonacci fue seleccionada porque es el método óptimo "
                "para minimizar una función unimodal en un intervalo cerrado sin "
                f"necesitar derivadas de f(x) = {fx}. "
                "Usando los números de Fibonacci para determinar los puntos de evaluación, "
                "reduce el intervalo de incertidumbre de forma óptima con el mínimo "
                "número de evaluaciones funcionales."),
            "Búsqueda Local": (
                "La búsqueda local por vecindad fue seleccionada como método exploratorio "
                f"sobre f(x) = {fx}. "
                "A partir del punto inicial, evalúa candidatos en el entorno inmediato "
                "(x ± paso) y se mueve al mejor vecino encontrado, repitiendo el proceso "
                "iterativamente."),
            "Sección Áurea": (
                f"La Sección Áurea fue seleccionada para f(x) = {fx} por ser el método "
                "de búsqueda unidimensional con convergencia garantizada en funciones "
                "unimodales. Utiliza la proporción τ = (√5−1)/2 ≈ 0.618 para reducir "
                "el intervalo de búsqueda en cada paso, reutilizando una evaluación "
                "de función por iteración. Es más eficiente que la bisección clásica."),
            "Goldstein": (
                f"Las condiciones de Goldstein fueron aplicadas a f(x) = {fx} para "
                "controlar el tamaño de paso con dos restricciones simultáneas: "
                "1) Sufficient decrease (evita pasos demasiado grandes), y "
                "2) Cota inferior (evita pasos demasiado pequeños). "
                "Con μ ∈ (0, 0.5), garantiza progreso real en cada iteración."),
            "Newton-Raphson": (
                f"El método de Newton-Raphson fue elegido para f(x) = {fx} por su "
                "convergencia cuadrática cerca del mínimo. Calcula la dirección de "
                "Newton d = −H⁻¹·∇f usando el gradiente y el Hessiano exactos "
                "(via diferenciación simbólica), lo que proporciona la actualización "
                "x_{k+1} = x_k − H⁻¹·∇f(x_k) con precisión máxima."),
            "Grad. Conjugado": (
                f"El Gradiente Conjugado (Fletcher-Reeves) fue seleccionado para "
                f"f(x) = {fx} por su eficiencia en funciones cuadráticas y no lineales. "
                "Genera direcciones conjugadas d_{k+1} = −∇f_{k+1} + β_k·d_k, donde "
                "β_k = ‖∇f_{k+1}‖²/‖∇f_k‖², logrando convergencia superlineal con "
                "solo información de primer orden."),
            "SA: Recocido Simulado": (
                f"El Recocido Simulado fue elegido para f(x) = {fx} por su capacidad "
                "de escapar mínimos locales. Inspirado en el enfriamiento de metales, "
                "acepta soluciones peores con probabilidad P=exp(−ΔE/T), donde T "
                "decrece gradualmente. Esto permite explorar el espacio de búsqueda "
                "globalmente antes de converger al mínimo."),
            "PSO: Enjambre": (
                f"PSO fue seleccionado para f(x) = {fx} por su eficacia en funciones "
                "multimodales sin derivadas. Un enjambre de partículas actualiza su "
                "velocidad combinando inercia propia, atracción a su mejor posición "
                "personal (c₁) y al mejor global del enjambre (c₂), explorando "
                "el espacio de forma paralela y cooperativa."),
            "GA: Algoritmo Genético": (
                f"El Algoritmo Genético fue aplicado a f(x) = {fx} por su robustez "
                "en espacios de búsqueda complejos y multimodales. Opera sobre una "
                "población de soluciones mediante selección por torneo, cruce aritmético "
                "y mutación gaussiana, con elitismo para preservar los mejores individuos. "
                "No requiere derivadas y puede encontrar el óptimo global."),
        }
        # MD methods
        if "Penalización" in metodo or "Pen." in metodo:
            return (f"El método de penalización fue seleccionado para resolver el problema "
                    f"con restricciones sobre f(x) = {fx}. "
                    "Transforma el problema restringido en una secuencia de problemas sin "
                    "restricciones añadiendo un término de penalización μ·P(x) al objetivo, "
                    "donde μ → ∞ fuerza el cumplimiento de las restricciones. "
                    "Newton o BFGS resuelven cada subproblema, garantizando convergencia "
                    "cuadrática o superlineal respectivamente.")
        if "Barrera" in metodo:
            return (f"El método de barrera (punto interior) fue seleccionado para f(x) = {fx} "
                    "con restricciones de desigualdad. "
                    "Añade términos logarítmicos −μ·ln(−gᵢ(x)) que tienden a infinito "
                    "cuando x se acerca al borde factible, manteniendo la solución en el "
                    "interior. A medida que μ → 0, la solución converge al óptimo restringido.")
        if "BFGS" in metodo:
            return (f"BFGS (Broyden–Fletcher–Goldfarb–Shanno) fue seleccionado para f(x) = {fx} "
                    "como método cuasi-Newton eficiente. "
                    "Aproxima la inversa del Hessiano usando solo gradientes de primer orden, "
                    "logrando convergencia superlineal sin el costo computacional de calcular "
                    "el Hessiano exacto. Es altamente efectivo para funciones suaves con "
                    "múltiples variables.")
        if "Nelder-Mead" in metodo:
            return (f"Nelder-Mead fue seleccionado para f(x) = {fx} porque no requiere "
                    "derivadas. Opera con un simplex que se deforma (reflexión, expansión, "
                    "contracción) para explorar el espacio de búsqueda. Adecuado cuando "
                    "la función no es diferenciable o el gradiente es costoso de evaluar.")
        return justificaciones.get(metodo,
            f"El método {metodo} fue aplicado a f(x) = {fx} como técnica de optimización "
            "numérica para encontrar el mínimo o máximo de la función en el dominio dado.")

    @staticmethod
    def _math_to_unicode(text: str) -> str:
        """
        Convierte notación matemática a superíndices/subíndices Unicode.
        Usado para celdas de Excel donde no hay HTML.
        """
        SUP_MAP = str.maketrans(
            "0123456789+-=()",
            "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
        )
        SUB_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

        def _sup(m):
            exp = m.group(2).strip("{}")
            return m.group(1) + exp.translate(SUP_MAP)

        def _sub(m):
            idx = m.group(2).strip("{}")
            return m.group(1) + idx.translate(SUB_MAP)

        import re as _re
        # x**{k+1}, x**2
        text = _re.sub(r'([a-zA-Z\u03b1-\u03c9\)\]])\*\*(\{[^}]+\}|-?\d+)', _sup, text)
        # x^{n}, x^2
        text = _re.sub(r'([a-zA-Z\u03b1-\u03c9\)\]])\^(\{[^}]+\}|-?\d+)', _sup, text)
        # x_{k+1}, x_k  (solo dígitos para subscript unicode)
        text = _re.sub(r'([a-zA-Z\u03b1-\u03c9])_\{(\d+)\}', _sub, text)
        text = _re.sub(r'([a-zA-Z\u03b1-\u03c9])_(\d+)',     _sub, text)
        # Fracciones comunes
        for raw, uni in [("1/2","½"),("1/3","⅓"),("1/4","¼"),
                         ("2/3","⅔"),("3/4","¾"),("1/8","⅛")]:
            text = text.replace(raw, uni)
        return text

    @staticmethod
    def _math_to_reportlab(text: str) -> str:
        """
        Convierte notación matemática a etiquetas XML de ReportLab
        (<super>, <sub>) para usar dentro de Paragraph().
        Aplica html.escape primero para proteger < y >.
        """
        import html as _hl
        import re as _re

        s = _hl.escape(text)

        # Superíndices: x**{k+1}, x**2, x^{n}, x^2
        s = _re.sub(r'([a-zA-Z\u03b1-\u03c9\u0391-\u03a9\u2207\)\]])\*\*\{([^}]+)\}',
                    lambda m: f"{m.group(1)}<super>{m.group(2)}</super>", s)
        s = _re.sub(r'([a-zA-Z\u03b1-\u03c9\u0391-\u03a9\u2207\)\]])\*\*(-?\d+(?:\.\d+)?)',
                    lambda m: f"{m.group(1)}<super>{m.group(2)}</super>", s)
        s = _re.sub(r'([a-zA-Z\u03b1-\u03c9\u0391-\u03a9\)\]])\^\{([^}]+)\}',
                    lambda m: f"{m.group(1)}<super>{m.group(2)}</super>", s)
        s = _re.sub(r'([a-zA-Z\u03b1-\u03c9\u0391-\u03a9\)\]])\^(-?\d+)',
                    lambda m: f"{m.group(1)}<super>{m.group(2)}</super>", s)
        # Subíndices: x_{k+1}, x_k
        s = _re.sub(r'([a-zA-Z\u03b1-\u03c9\u0391-\u03a9])_\{([^}]+)\}',
                    lambda m: f"{m.group(1)}<sub>{m.group(2)}</sub>", s)
        s = _re.sub(r'([a-zA-Z\u03b1-\u03c9\u0391-\u03a9])_([a-zA-Z0-9\+\-\*]+)',
                    lambda m: f"{m.group(1)}<sub>{m.group(2)}</sub>", s)
        # Fracciones comunes
        for raw, uni in [("1/2","½"),("1/3","⅓"),("1/4","¼"),
                         ("2/3","⅔"),("3/4","¾"),("1/8","⅛")]:
            s = s.replace(raw, uni)
        return s

    @staticmethod
    def _render_math_img_file(expr: str,
                               fontsize: float = 11,
                               fg: str = '#1a1a3a',
                               dpi: int = 150) -> 'str | None':
        """
        Renderiza una expresión mathtext a un archivo PNG temporal.
        Retorna la ruta del archivo o None si matplotlib no está disponible.
        Usado para incrustar fórmulas en PDF y Excel.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import tempfile, os
            import re as _re2

            # ── Convertir a mathtext ($...$) ──────────────────────────────
            s = expr.strip()
            UNI = {
                'α':r'\alpha','β':r'\beta','γ':r'\gamma','δ':r'\delta',
                'λ':r'\lambda','μ':r'\mu','σ':r'\sigma','ω':r'\omega',
                '∇':r'\nabla','∂':r'\partial','∞':r'\infty',
                '≤':r'\leq','≥':r'\geq','≠':r'\neq',
                '·':r'\cdot','×':r'\times','±':r'\pm','‖':r'\|',
            }
            for uc, lt in UNI.items():
                s = s.replace(uc, lt)
            # Potencias
            s = _re2.sub(r'([a-zA-Z0-9\)\|])\*\*\{([^}]+)\}', r'\1^{\2}', s)
            s = _re2.sub(r'([a-zA-Z0-9\)\|])\*\*(-?\d+)',      r'\1^{\2}', s)
            # Multiplicación
            s = _re2.sub(r'(?<!\^)(?<!\{)\*(?!\*)(?!\{)', r' \\cdot ', s)
            # exp / sqrt / fracciones
            s = _re2.sub(r'exp\(([^)]+)\)',  r'e^{\1}',        s)
            s = _re2.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}',    s)
            s = _re2.sub(r'\b(\d+)/(\d+)\b', r'\\frac{\1}{\2}', s)

            mathtext = f'${s}$'

            fig = plt.figure(figsize=(0.01, 0.01))
            fig.patch.set_facecolor('#e8f2ff')
            fig.text(0.5, 0.5, mathtext, fontsize=fontsize,
                     color=fg, ha='center', va='center')
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp.close()
            fig.savefig(tmp.name, format='png', dpi=dpi,
                        bbox_inches='tight', pad_inches=0.12)
            plt.close(fig)
            return tmp.name
        except Exception:
            return None

    @staticmethod
    def _build_step_by_step(entry: dict) -> list:
        """
        Genera lista de strings con los cálculos paso a paso del método,
        derivados directamente del historial de iteraciones.
        Retorna: lista de (título, descripción) por iteración.
        """
        hist   = entry.get("hist", [])
        metodo = entry.get("metodo", "")
        fx     = entry.get("fx", "")
        lo     = entry.get("lo", 0)
        hi     = entry.get("hi", 1)
        x0     = entry.get("x0", [])
        steps  = []

        if not hist:
            return [("Sin datos", "No se registraron iteraciones para este método.")]

        # ── Paso 0: Condiciones iniciales ─────────────────────────────────────
        x0_str = (f"[{', '.join(f'{v:.4g}' for v in x0)}]" if x0
                  else "no especificado")
        steps.append((
            "Condiciones iniciales",
            f"Función objetivo: f(x) = {fx}\n"
            f"Intervalo de búsqueda: [{lo:.4g}, {hi:.4g}]\n"
            f"Punto inicial: x₀ = {x0_str}\n"
            f"Método aplicado: {metodo}"
        ))

        # ── Pasos por iteración ───────────────────────────────────────────────
        for i, rec in enumerate(hist):
            titulo = f"Iteración {i + 1}"

            # Extraer campos comunes
            k      = rec.get("k",    rec.get("iter", i))
            x_k    = rec.get("x_k",  rec.get("x",    rec.get("lambda_k", None)))
            f_k    = rec.get("f_k",  rec.get("f(x)", rec.get("f_new",    rec.get("f_lambda", None))))
            alpha  = rec.get("alpha_k", rec.get("alpha", rec.get("paso", None)))
            grad   = rec.get("grad_norm", None)
            d_k    = rec.get("d_k_str",   rec.get("d_k", None))

            def _fmt(v):
                if v is None: return "N/A"
                if isinstance(v, np.ndarray):
                    return "[" + ", ".join(f"{x:.5g}" for x in v.flat) + "]"
                try:    return f"{float(v):.6g}"
                except: return str(v)

            lines = []

            # Armijo / Wolfe
            if metodo in ("Armijo", "Wolfe"):
                armijo_ok = rec.get("armijo_ok", None)
                curv_ok   = rec.get("curv_ok",   None)
                x_new     = rec.get("x_new",     x_k)
                f_new     = rec.get("f_new",      f_k)
                lines.append(f"  Punto actual:       x_k = {_fmt(x_k)}")
                lines.append(f"  Norma del gradiente: ‖∇f(x_k)‖ = {_fmt(grad)}")
                if d_k is not None:
                    lines.append(f"  Dirección de descenso: d_k = {_fmt(d_k)}")
                lines.append(f"  Tamaño de paso:      α = {_fmt(alpha)}")
                lines.append(f"  Nuevo punto:         x_{{k+1}} = x_k + α·d_k = {_fmt(x_new)}")
                lines.append(f"  Valor de la función: f(x_{{k+1}}) = {_fmt(f_new)}")
                if armijo_ok is not None:
                    lines.append(f"  Condición Armijo cumplida: {'✓ Sí' if armijo_ok else '✗ No'}")
                if curv_ok is not None:
                    lines.append(f"  Condición curvatura (Wolfe) cumplida: {'✓ Sí' if curv_ok else '✗ No'}")

            # Fibonacci / Búsqueda Local
            elif metodo in ("Fibonacci", "Búsqueda Local"):
                a_i   = rec.get("a", rec.get("a_n", None))
                b_i   = rec.get("b", rec.get("b_n", None))
                x_c   = rec.get("x_candidate", rec.get("x1", rec.get("x2", None)))
                f_c   = rec.get("f_candidate", rec.get("f1", rec.get("f2", None)))
                improved = rec.get("improved", None)
                lines.append(f"  Punto actual:      x = {_fmt(x_k)},  f(x) = {_fmt(f_k)}")
                if a_i is not None and b_i is not None:
                    lines.append(f"  Intervalo activo:  [{_fmt(a_i)},  {_fmt(b_i)}]")
                    lines.append(f"  Longitud intervalo: Δ = {abs(float(b_i)-float(a_i)):.5g}" if
                                 b_i is not None and a_i is not None else "")
                if x_c is not None:
                    lines.append(f"  Candidato evaluado: x_c = {_fmt(x_c)},  f(x_c) = {_fmt(f_c)}")
                if alpha is not None:
                    lines.append(f"  Paso utilizado:    δ = {_fmt(alpha)}")
                if improved is not None:
                    lines.append(f"  Mejora encontrada: {'✓ Sí → actualizar punto' if improved else '✗ No → mantener punto'}")

            # MD methods
            else:
                x_str = rec.get("x_k_str", rec.get("x_str", None))
                f_mu  = rec.get("f_mu",  rec.get("f_lambda", f_k))
                mu    = rec.get("mu",    rec.get("lambda",  None))
                lines.append(f"  Iteración global k = {_fmt(k)}")
                lines.append(f"  Punto x_k = {_fmt(x_k)}")
                if x_str: lines.append(f"  Representación: {x_str}")
                lines.append(f"  f(x_k) = {_fmt(f_k)}")
                if f_mu is not None and f_mu != f_k:
                    lines.append(f"  f aumentada/penalizada = {_fmt(f_mu)}")
                if mu is not None:
                    lines.append(f"  Parámetro μ/λ = {_fmt(mu)}")
                if grad is not None:
                    lines.append(f"  ‖∇f(x_k)‖ = {_fmt(grad)}")
                # Resto de campos no estándar
                for kk, vv in rec.items():
                    if kk not in ("k","iter","x_k","x","x_k_str","f_k","f_mu",
                                  "f_lambda","mu","lambda","grad_norm","x_str"):
                        lines.append(f"  {kk} = {_fmt(vv)}")

            steps.append((titulo, "\n".join(l for l in lines if l.strip())))

        # ── Paso final: conclusión ────────────────────────────────────────────
        last = hist[-1]
        x_final = last.get("x_k", last.get("x", last.get("lambda_k", None)))
        f_final = last.get("f_k", last.get("f(x)", last.get("f_new",
                  last.get("f_lambda", None))))
        def _fv(v):
            if v is None: return "N/D"
            if isinstance(v, np.ndarray):
                return "[" + ", ".join(f"{x:.6g}" for x in v.flat) + "]"
            try:    return f"{float(v):.6g}"
            except: return str(v)

        steps.append((
            "Resultado final",
            f"Tras {len(hist)} iteración(es) el método convergió a:\n"
            f"  x* = {_fv(x_final)}\n"
            f"  f(x*) = {_fv(f_final)}\n"
            f"El valor de x* representa el {'mínimo' if f_final is not None and float(f_final if isinstance(f_final,(int,float)) else 0) < 0 else 'óptimo'} "
            f"encontrado por {metodo} en el dominio [{lo:.4g}, {hi:.4g}]."
        ))
        return steps

    @staticmethod
    def _classify_study(metodo: str) -> str:
        """Clasifica el tipo de estudio según el método."""
        md_keywords = ("MD","Penalización","Pen.","Barrera","BFGS","Nelder","Newton MD","Grad. Conj.")
        if any(k in metodo for k in md_keywords):
            return "Optimización Multidimensional (MD)"
        if metodo == "Fibonacci":
            return "Búsqueda Unidimensional — Sección Áurea / Fibonacci"
        if metodo == "Sección Áurea":
            return "Heurístico — Búsqueda por Proporción Áurea"
        if metodo == "Goldstein":
            return "Heurístico — Búsqueda de Línea (Condiciones de Goldstein)"
        if metodo == "Newton-Raphson":
            return "Heurístico — Método de Newton-Raphson (2ᵃ derivada)"
        if metodo == "Grad. Conjugado":
            return "Heurístico — Gradiente Conjugado (Fletcher-Reeves)"
        if metodo in ("Armijo", "Wolfe"):
            return "Búsqueda de Línea — Condiciones de Wolfe/Armijo"
        if "Recocido" in metodo or "SA:" in metodo:
            return "Metaheurístico — Recocido Simulado (Simulated Annealing)"
        if "PSO" in metodo or "Enjambre" in metodo:
            return "Metaheurístico — Optimización por Enjambre de Partículas (PSO)"
        if "GA:" in metodo or "Genético" in metodo:
            return "Metaheurístico — Algoritmo Genético (GA)"
        return "Búsqueda Unidimensional Local"

    @staticmethod
    def _classify_function(fx: str) -> str:
        """Clasifica la función por tipo matemático, incluyendo funciones de prueba conocidas."""
        fx_l = fx.lower().replace(" ", "")
        # Funciones de prueba conocidas del PDF
        if "rosenbrock" in fx_l or ("(1-x)**2" in fx_l and "100" in fx_l):
            return "No Convexa (Rosenbrock)"
        if "himmelblau" in fx_l or ("x**2+y-11" in fx_l):
            return "Multimodal (Himmelblau)"
        if "rastrigin" in fx_l or ("10*cos(2*pi*x)" in fx_l):
            return "Multimodal (Rastrigin)"
        if "griewank" in fx_l or ("4000" in fx_l and "cos(x)" in fx_l):
            return "Multimodal (Griewank)"
        if "ackley" in fx_l or ("20*exp" in fx_l and "cos(2*pi" in fx_l):
            return "Multimodal (Ackley)"
        if "beale" in fx_l or ("1.5-x+x*y" in fx_l.replace(" ", "")):
            return "No Convexa (Beale)"
        # Por características matemáticas
        fc = _get_fc()
        if fc:
            vars_ = fc.detect_variables(fx)
            n = len(vars_)
        else:
            n = 1
        dim_str = f" ({n}D)" if n > 1 else ""
        if any(k in fx_l for k in ("sin", "cos", "tan", "pi")):
            return f"Trigonométrica{dim_str}"
        if any(k in fx_l for k in ("exp", "log", "ln")):
            return f"Exponencial/Logarítmica{dim_str}"
        if "**" in fx_l or "sqrt" in fx_l:
            import re
            degs = [int(m) for m in re.findall(r"\*\*(\d+)", fx_l)]
            mx = max(degs) if degs else 2
            return f"Polinomial grado {mx}{dim_str}"
        if "/" in fx_l:
            return f"Racional{dim_str}"
        return f"Polinomial/Algebraica{dim_str}"

    @staticmethod
    def _build_conclusion(session: list, fx: str) -> str:
        """Genera conclusión general basada en los métodos ejecutados."""
        if not session: return "No se ejecutaron métodos."
        n = len(session)
        metodos = [s["metodo"] for s in session]
        resultados = []
        for s in session:
            h = s["hist"]
            if h:
                last = h[-1]
                for kx in ("x_k","x","lambda_k"):
                    v = last.get(kx)
                    if v is None: continue
                    # safe extraction regardless of type
                    try:
                        if isinstance(v, np.ndarray):
                            xs = float(v.flat[0])
                        elif isinstance(v, (list, tuple)) and len(v) > 0:
                            xs = float(v[0])
                        else:
                            xs = float(v)
                        resultados.append(f"{s['metodo']}: x*≈{xs:.4f}")
                    except Exception:
                        pass
                    break
        lines = [
            f"Se aplicaron {n} método(s) de optimización a la función f(x) = {fx}.",
            "Métodos utilizados: " + ", ".join(metodos) + ".",
        ]
        if resultados:
            lines.append("Resultados obtenidos — " + ";  ".join(resultados) + ".")
        lines.append(
            "Los métodos de búsqueda de línea (Armijo, Wolfe) garantizan "
            "descenso suficiente en cada iteración. Los métodos de región de "
            "confianza como Fibonacci son robustos ante funciones no convexas. "
            "Los métodos MD con penalización/barrera manejan restricciones de "
            "igualdad o desigualdad."
        )
        return "  ".join(lines)

    def _nodata(self): QMsgBox.information(self,"Exportar","No hay datos.")

    def export_csv_db(self):
        """
        Exporta 3 CSV normalizados + script SQL compatibles con:
        PostgreSQL, MySQL, SQLite, MariaDB, SQL Server, Oracle,
        DBeaver, pgAdmin, DataGrip, TablePlus, etc.

        Archivos generados (en la carpeta que elija el usuario):
          reporte_sesion.csv       — una fila por método ejecutado
          reporte_iteraciones.csv  — una fila por iteración
          reporte_historial.csv    — historial de funciones de la sesión
          crear_tablas.sql         — script CREATE TABLE + comandos COPY
        """
        if self.table.rowCount() == 0:
            return self._nodata()

        folder = QFD.getExistingDirectory(self, "Seleccionar carpeta de destino para los CSV")
        if not folder:
            return

        try:
            import datetime, math, re, os

            session = self._session if self._session else [{
                "metodo": self._metodo, "fx": self._fx, "gx": "",
                "vars": list(self._vars), "x0": list(self._x0),
                "lo": self.smin.value(), "hi": self.smax.value(),
                "hist": self._hist,
                "traj": self._traj(self._vars, self._hist, self._x0),
            }]

            def _sql_col(name: str) -> str:
                """Nombre SQL-safe: minúsculas, sin tildes, solo a-z0-9_."""
                t = name.lower().strip()
                t = t.replace("á","a").replace("é","e").replace("í","i")\
                     .replace("ó","o").replace("ú","u").replace("ñ","n")
                t = re.sub(r"[^a-z0-9_]", "_", t).strip("_")
                return t or "col"

            timestamp_export = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            def _num(v):
                """Devuelve float limpio como string, o None — nunca NaN/Inf/string vacío con comillas."""
                if v is None:
                    return None
                if isinstance(v, np.ndarray):
                    flat = v.flatten()
                    if flat.size == 1:
                        v = float(flat[0])
                    else:
                        return None
                try:
                    f = float(v)
                    if math.isnan(f) or math.isinf(f):
                        return None
                    return f"{f:.10g}"
                except Exception:
                    return None

            def _txt(v):
                """Devuelve texto limpio o cadena vacía."""
                if v is None:
                    return ""
                return str(v)

            def _csv_writer(path):
                """CSV con comillas solo en campos de texto, numéricos sin comillas — compatible PostgreSQL."""
                fh = open(path, "w", newline="", encoding="utf-8")
                w = csv.writer(fh, delimiter=",",
                               quoting=csv.QUOTE_MINIMAL,
                               lineterminator="\r\n")
                return fh, w

            # ═══════════════════════════════════════════════════════════════════
            # TABLA 1 — reporte_sesion
            # ═══════════════════════════════════════════════════════════════════
            p_sesion = os.path.join(folder, "reporte_sesion.csv")
            fh1, w1 = _csv_writer(p_sesion)
            w1.writerow([
                "metodo_id", "metodo", "funcion_fx", "variables",
                "intervalo_min", "intervalo_max", "punto_inicial",
                "total_iteraciones", "x_optimo", "f_optimo",
                "tipo_estudio", "exportado_en"
            ])
            for idx, entry in enumerate(session, 1):
                hist  = entry.get("hist") or []
                vars_ = entry.get("vars", [])
                x0    = entry.get("x0", [])
                x_opt = f_opt = None
                if hist:
                    last = hist[-1]
                    for kx in ("x_k","x","lambda_k","mu_k","x1"):
                        if kx in last:
                            x_opt = _num(last[kx]); break
                    for kf in ("f_k","f(x)","f_new","f_lambda","f_mu","fx"):
                        if kf in last:
                            f_opt = _num(last[kf]); break
                x0_str = ";".join(str(v) for v in x0) if x0 else ""
                w1.writerow([
                    idx,
                    _txt(entry["metodo"]),
                    _txt(entry["fx"]),
                    _txt(";".join(vars_)),
                    _num(entry.get("lo")),
                    _num(entry.get("hi")),
                    x0_str,
                    len(hist),
                    x_opt,
                    f_opt,
                    _txt(self._classify_study(entry["metodo"])),
                    timestamp_export,
                ])
            fh1.close()

            # ═══════════════════════════════════════════════════════════════════
            # TABLA 2 — reporte_iteraciones
            # ═══════════════════════════════════════════════════════════════════
            # Recolectar columnas únicas de todos los registros de todas las iteraciones
            orig_cols: list = []       # nombres originales
            for entry in session:
                for rec in (entry.get("hist") or []):
                    for k in rec:
                        if k not in orig_cols:
                            orig_cols.append(k)

            sql_cols = [_sql_col(k) for k in orig_cols]

            p_iter = os.path.join(folder, "reporte_iteraciones.csv")
            fh2, w2 = _csv_writer(p_iter)
            w2.writerow(["metodo_id", "metodo", "funcion_fx", "iteracion"] + sql_cols)
            for idx, entry in enumerate(session, 1):
                for i, rec in enumerate(entry.get("hist") or []):
                    row = [idx, _txt(entry["metodo"]), _txt(entry["fx"]), i]
                    for ok in orig_cols:
                        row.append(_num(rec.get(ok)))
                    w2.writerow(row)
            fh2.close()

            # ═══════════════════════════════════════════════════════════════════
            # TABLA 3 — reporte_historial
            # ═══════════════════════════════════════════════════════════════════
            p_hist = os.path.join(folder, "reporte_historial.csv")
            fh3, w3 = _csv_writer(p_hist)
            w3.writerow(["hora","funcion_fx","metodo","iteraciones","x_optimo","f_optimo"])
            for i, e in enumerate(self._history_log, 1):
                w3.writerow([
                    _txt(e.get("timestamp","")),
                    _txt(e.get("fx","")),
                    _txt(e.get("metodo","")),
                    e.get("iters") or None,
                    _num(e.get("xopt")),
                    _num(e.get("fopt")),
                ])
            fh3.close()

            # ═══════════════════════════════════════════════════════════════════
            # ═══════════════════════════════════════════════════════════════════
            # SCRIPT SQL — CREATE TABLE
            # ═══════════════════════════════════════════════════════════════════
            p_sql = os.path.join(folder, "crear_tablas.sql")
            folder_sql = folder.replace("\\", "/")
            iter_cols_ddl = "\n".join(
                f"    {c:<24} DOUBLE PRECISION," for c in sql_cols
            )
            with open(p_sql, "w", encoding="utf-8") as fs:
                fs.write(f"-- Generado por Optimizador de Funciones  |  {timestamp_export}\n")
                fs.write("-- PASO 1: Ejecuta este archivo en pgAdmin Query Tool para crear las tablas.\n")
                fs.write("-- PASO 2: Usa importar_datos.sql para cargar los CSV.\n\n")
                fs.write(
                    "CREATE TABLE IF NOT EXISTS reporte_sesion (\n"
                    "    metodo_id         SERIAL          PRIMARY KEY,\n"
                    "    metodo            VARCHAR(120),\n"
                    "    funcion_fx        TEXT,\n"
                    "    variables         TEXT,\n"
                    "    intervalo_min     DOUBLE PRECISION,\n"
                    "    intervalo_max     DOUBLE PRECISION,\n"
                    "    punto_inicial     TEXT,\n"
                    "    total_iteraciones INTEGER,\n"
                    "    x_optimo          DOUBLE PRECISION,\n"
                    "    f_optimo          DOUBLE PRECISION,\n"
                    "    tipo_estudio      VARCHAR(120),\n"
                    "    exportado_en      TIMESTAMP\n"
                    ");\n\n"
                )
                fs.write(
                    "CREATE TABLE IF NOT EXISTS reporte_iteraciones (\n"
                    "    id                SERIAL          PRIMARY KEY,\n"
                    "    metodo_id         INTEGER         REFERENCES reporte_sesion(metodo_id),\n"
                    "    metodo            VARCHAR(120),\n"
                    "    funcion_fx        TEXT,\n"
                    "    iteracion         INTEGER,\n"
                )
                fs.write(iter_cols_ddl + "\n")
                fs.write(
                    "    _dummy            BOOLEAN DEFAULT NULL\n"
                    ");\n\n"
                )
                fs.write(
                    "CREATE TABLE IF NOT EXISTS reporte_historial (\n"
                    "    id                SERIAL          PRIMARY KEY,\n"
                    "    hora              VARCHAR(20),\n"
                    "    funcion_fx        TEXT,\n"
                    "    metodo            VARCHAR(120),\n"
                    "    iteraciones       INTEGER,\n"
                    "    x_optimo          DOUBLE PRECISION,\n"
                    "    f_optimo          DOUBLE PRECISION\n"
                    ");\n"
                )

            # SCRIPT SEPARADO — solo importación (COPY server-side para pgAdmin)
            p_import = os.path.join(folder, "importar_datos.sql")
            with open(p_import, "w", encoding="utf-8") as fi:
                fi.write(f"-- Importar datos  |  {timestamp_export}\n")
                fi.write("-- Ejecuta cada linea POR SEPARADO en pgAdmin Query Tool.\n")
                fi.write("-- El servidor PostgreSQL debe poder leer la ruta indicada.\n")
                fi.write("-- Si usas PostgreSQL local en Windows, la ruta debe ser con barras /\n\n")
                # Columnas explícitas para reporte_sesion (excluye 'id' SERIAL)
                sesion_cols = (
                    "metodo_id, metodo, funcion_fx, variables, intervalo_min, "
                    "intervalo_max, punto_inicial, total_iteraciones, x_optimo, "
                    "f_optimo, tipo_estudio, exportado_en"
                )
                # Columnas explícitas para reporte_iteraciones (excluye 'id' SERIAL y '_dummy')
                iter_copy_cols = ", ".join(
                    ["metodo_id", "metodo", "funcion_fx", "iteracion"] + sql_cols
                )
                fi.write(
                    f"COPY reporte_sesion({sesion_cols})\n"
                    f"  FROM '{folder_sql}/reporte_sesion.csv'\n"
                    f"  WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', ENCODING 'UTF8');\n\n"
                )
                fi.write(
                    f"COPY reporte_iteraciones({iter_copy_cols})\n"
                    f"  FROM '{folder_sql}/reporte_iteraciones.csv'\n"
                    f"  WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', ENCODING 'UTF8');\n\n"
                )
                fi.write(
                    f"COPY reporte_historial(hora, funcion_fx, metodo, iteraciones, x_optimo, f_optimo)\n"
                    f"  FROM '{folder_sql}/reporte_historial.csv'\n"
                    f"  WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', ENCODING 'UTF8');\n\n"
                    "-- ALTERNATIVA: si prefieres usar la interfaz grafica de pgAdmin:\n"
                    "--   Click derecho en la tabla > Import/Export Data\n"
                    "--   Format: csv  |  Header: ON  |  Delimiter: ,  |  Encoding: UTF8\n"
                )

            n_iter = sum(len(s.get("hist") or []) for s in session)
            self._show_export_success(
                "CSV", folder,
                f"3 CSV + crear_tablas.sql + importar_datos.sql  ·  {len(session)} método(s)  ·  {n_iter} iteración(es)"
            )

        except Exception as e:
            _err(self, "Error al exportar CSV", str(e), traceback.format_exc())

    def export_xlsx(self):
        """
        Exporta a Excel con estructura multi-pestaña:
          1) "RESUMEN" — función insertada, tipo de estudio, métodos aplicados
          2+) Una pestaña por método — resumen de resultados, tabla completa de
              iteraciones + gráficas 2D y 3D
        """
        if self.table.rowCount() == 0:
            return self._nodata()
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter
            from openpyxl.drawing.image import Image as XLImg
        except ImportError:
            QMsgBox.warning(self, "Dependencia", "pip install openpyxl")
            return

        p, _ = QFD.getSaveFileName(self, "Guardar Excel", "reporte.xlsx", "Excel (*.xlsx)")
        if not p:
            return

        try:
            # ── Construir sesión ──────────────────────────────────────────────
            session = self._session if self._session else [{
                "metodo": self._metodo, "fx": self._fx, "gx": "",
                "vars": list(self._vars), "x0": list(self._x0),
                "lo": self.smin.value(), "hi": self.smax.value(),
                "hist": self._hist,
                "traj": self._traj(self._vars, self._hist, self._x0),
            }]
            fx_global = session[0]["fx"] if session else self._fx

            # ── Paleta de colores ─────────────────────────────────────────────
            C = {
                "bg_dark":  "080E22",
                "bg_mid":   "101E3A",
                "bg_head":  "162D5A",
                "bg_head2": "0D1E44",
                "bg_resumen":"0A1830",
                "bg_alt":   "E8F2FF",
                "fg_white": "FFFFFF",
                "fg_blue":  "7AAAEA",
                "fg_cyan":  "5DC8D0",
                "fg_gold":  "FFD700",
                "border":   "2A50A0",
                "accent":   "FF6B35",
            }
            def _font(bold=False, size=11, color=None, italic=False):
                return Font(bold=bold, size=size, italic=italic,
                            name="Segoe UI",
                            color=color or C["fg_white"])
            def _fill(hex_color):
                return PatternFill("solid", fgColor=hex_color)
            def _border(color=None):
                s = Side(style="thin", color=color or C["border"])
                return Border(left=s, right=s, top=s, bottom=s)
            def _center(wrap=False):
                return Alignment(horizontal="center", vertical="center", wrap_text=wrap)
            def _left(wrap=False):
                return Alignment(horizontal="left", vertical="center", wrap_text=wrap)
            def _set_col_width(ws, col_idx, width):
                ws.column_dimensions[get_column_letter(col_idx)].width = width
            def _merge_write(ws, cell_range, value, font=None, fill=None,
                             align=None, height=None, border=None):
                ws.merge_cells(cell_range)
                c = ws[cell_range.split(":")[0]]
                c.value = value
                if font:   c.font      = font
                if fill:   c.fill      = fill
                if align:  c.alignment = align
                if border: c.border    = border
                if height:
                    rn = int(''.join(filter(str.isdigit, cell_range.split(":")[0])))
                    ws.row_dimensions[rn].height = height

            def _classify_study(metodo: str) -> str:
                if _is_md(metodo): return "Optimización Multidimensional (MD)"
                if metodo == "Fibonacci": return "Búsqueda Unidimensional — Sección Áurea / Fibonacci"
                if metodo in ("Armijo","Wolfe"): return "Búsqueda de Línea — Condiciones de Wolfe/Armijo"
                return "Búsqueda Unidimensional Local"

            wb = openpyxl.Workbook()
            wb.remove(wb.active)

            # ══════════════════════════════════════════════════════════════════
            # PESTAÑA 1: RESUMEN
            # ══════════════════════════════════════════════════════════════════
            ws_r = wb.create_sheet("RESUMEN")
            ws_r.sheet_view.showGridLines = False
            for ci, w in [(1,3),(2,26),(3,52),(4,3)]:
                _set_col_width(ws_r, ci, w)

            # Título principal
            _merge_write(ws_r, "B1:C1",
                "Optimizador de Funciones — Reporte de Análisis",
                font=_font(bold=True, size=16),
                fill=_fill(C["bg_dark"]), align=_center(), height=40)
            ws_r.row_dimensions[2].height = 6

            r = 3
            # ── Bloque: Función analizada ─────────────────────────────────────
            _merge_write(ws_r, f"B{r}:C{r}", "FUNCIÓN ANALIZADA",
                font=_font(bold=True, size=12, color=C["fg_cyan"]),
                fill=_fill(C["bg_head2"]), align=_center(), height=24)
            r += 1

            fn_rows = [
                ("Función:",         fx_global),
                ("Variables:",       ", ".join(session[0]["vars"])),
                ("Intervalo / Dom.", f"[{session[0]['lo']:.4g},  {session[0]['hi']:.4g}]"),
                ("Tipo de función:", self._classify_function(fx_global)),
            ]
            for lbl, val in fn_rows:
                ws_r.row_dimensions[r].height = 20
                cl = ws_r.cell(row=r, column=2, value=lbl)
                cl.font = _font(bold=True, size=11, color=C["fg_blue"])
                cl.fill = _fill(C["bg_mid"]); cl.alignment = _left()
                cl.border = _border()
                cv = ws_r.cell(row=r, column=3, value=val)
                cv.font = _font(size=11); cv.fill = _fill(C["bg_mid"])
                cv.alignment = _left(); cv.border = _border()
                r += 1

            ws_r.row_dimensions[r].height = 8; r += 1

            # ── Bloque: Tipo de estudio ───────────────────────────────────────
            _merge_write(ws_r, f"B{r}:C{r}", "TIPO DE ESTUDIO APLICADO",
                font=_font(bold=True, size=12, color=C["fg_cyan"]),
                fill=_fill(C["bg_head2"]), align=_center(), height=24)
            r += 1

            all_types = list(dict.fromkeys(_classify_study(s["metodo"]) for s in session))
            for t in all_types:
                ws_r.row_dimensions[r].height = 20
                cl = ws_r.cell(row=r, column=2, value="Tipo:")
                cl.font = _font(bold=True, size=11, color=C["fg_blue"])
                cl.fill = _fill(C["bg_mid"]); cl.alignment = _left(); cl.border = _border()
                cv = ws_r.cell(row=r, column=3, value=t)
                cv.font = _font(size=11); cv.fill = _fill(C["bg_mid"])
                cv.alignment = _left(); cv.border = _border()
                r += 1

            ws_r.row_dimensions[r].height = 8; r += 1

            # ── Bloque: Métodos aplicados ─────────────────────────────────────
            _merge_write(ws_r, f"B{r}:C{r}", "MÉTODOS APLICADOS",
                font=_font(bold=True, size=12, color=C["fg_cyan"]),
                fill=_fill(C["bg_head2"]), align=_center(), height=24)
            r += 1

            for idx_s, s in enumerate(session, 1):
                ws_r.row_dimensions[r].height = 20
                cl = ws_r.cell(row=r, column=2, value=f"  {idx_s}.")
                cl.font = _font(bold=True, size=11, color=C["fg_gold"])
                cl.fill = _fill(C["bg_mid"]); cl.alignment = _center(); cl.border = _border()
                cv = ws_r.cell(row=r, column=3, value=s["metodo"])
                cv.font = _font(size=11); cv.fill = _fill(C["bg_mid"])
                cv.alignment = _left(); cv.border = _border()
                r += 1

            ws_r.row_dimensions[r].height = 8; r += 1

            # ── Pie ───────────────────────────────────────────────────────────
            _merge_write(ws_r, f"B{r}:C{r}",
                f"Generado automáticamente  ·  {len(session)} método(s) ejecutado(s)",
                font=_font(italic=True, size=9, color=C["fg_blue"]),
                fill=_fill(C["bg_dark"]), align=_center(), height=16)

            # ══════════════════════════════════════════════════════════════════
            # PESTAÑAS POR MÉTODO
            # ══════════════════════════════════════════════════════════════════
            all_tmp_imgs: list = []   # rutas de PNGs temporales — borrar tras wb.save
            _tab_counts: dict = {}    # contador de nombres de pestaña usados

            def _safe_tab_name(metodo: str) -> str:
                """Genera nombre de pestaña único: 'Armijo', 'Armijo (2)', 'Armijo (3)'..."""
                base = metodo[:28]
                for ch in (chr(92), "/", "[", "]", "*", "?", ":"):
                    base = base.replace(ch, "-")
                base = base.strip()
                if base not in _tab_counts:
                    _tab_counts[base] = 1
                    return base
                else:
                    _tab_counts[base] += 1
                    return f"{base} ({_tab_counts[base]})"

            for idx_s, entry in enumerate(session, 1):
                metodo = entry["metodo"]
                hist   = entry["hist"]

                tab_name = _safe_tab_name(metodo)
                ws_m = wb.create_sheet(tab_name)
                ws_m.sheet_view.showGridLines = False

                # Columnas fijas: A=margen, B=etiqueta, C=valor (ancho generoso)
                _set_col_width(ws_m, 1, 3)    # A — margen
                _set_col_width(ws_m, 2, 22)   # B — etiquetas
                _set_col_width(ws_m, 3, 52)   # C — valores

                # Banner
                ws_m.row_dimensions[1].height = 34
                _merge_write(ws_m, "B1:C1",
                    f"Optimizador de Funciones  —  {metodo}",
                    font=_font(bold=True, size=14),
                    fill=_fill(C["bg_dark"]), align=_center(), height=34)

                ws_m.row_dimensions[2].height = 4
                r_m = 3

                # Info básica
                info_rows = [
                    ("Método:",          metodo),
                    ("Función:",         entry["fx"]),
                    ("Tipo de estudio:", _classify_study(metodo)),
                    ("Variables:",       ", ".join(entry["vars"])),
                    ("Intervalo:",       f"[{entry['lo']:.4g},  {entry['hi']:.4g}]"),
                ]
                for lbl, val in info_rows:
                    ws_m.row_dimensions[r_m].height = 20
                    cl = ws_m.cell(row=r_m, column=2, value=lbl)
                    cl.font = _font(bold=True, size=11, color=C["fg_blue"])
                    cl.fill = _fill(C["bg_mid"]); cl.alignment = _left(); cl.border = _border()
                    cv = ws_m.cell(row=r_m, column=3, value=val)
                    cv.font = _font(size=11); cv.fill = _fill(C["bg_mid"])
                    cv.alignment = Alignment(horizontal="left", vertical="center",
                                             wrap_text=True)
                    cv.border = _border()
                    r_m += 1

                ws_m.row_dimensions[r_m].height = 6; r_m += 1

                # ── Bloque RESUMEN (resultados óptimos) ───────────────────────
                _merge_write(ws_m, f"B{r_m}:C{r_m}", "Resumen de Resultados",
                    font=_font(bold=True, size=11, color=C["fg_cyan"]),
                    fill=_fill(C["bg_head2"]), align=_center(), height=22)
                r_m += 1

                if hist:
                    last = hist[-1]
                    # x_opt
                    for kx in ("x_k","x","lambda_k"):
                        if kx in last:
                            v = last[kx]
                            if isinstance(v, np.ndarray):
                                xstr = "[" + ", ".join(f"{x:.6g}" for x in v.flat) + "]"
                            else:
                                xstr = f"{float(v):.8g}"
                            ws_m.row_dimensions[r_m].height = 18
                            cl = ws_m.cell(row=r_m, column=2, value="x_opt")
                            cl.font = _font(bold=True, size=11, color=C["fg_gold"])
                            cl.fill = _fill(C["bg_resumen"]); cl.alignment = _center(); cl.border = _border()
                            cv = ws_m.cell(row=r_m, column=3, value=xstr)
                            cv.font = _font(size=11); cv.fill = _fill(C["bg_resumen"])
                            cv.alignment = _left(); cv.border = _border()
                            r_m += 1
                            break
                    # f_opt
                    for kf in ("f_k","f_lambda","f_mu","f(x)"):
                        if kf in last and isinstance(last[kf],(int,float,np.number)):
                            ws_m.row_dimensions[r_m].height = 18
                            cl = ws_m.cell(row=r_m, column=2, value="f_opt")
                            cl.font = _font(bold=True, size=11, color=C["fg_gold"])
                            cl.fill = _fill(C["bg_resumen"]); cl.alignment = _center(); cl.border = _border()
                            cv = ws_m.cell(row=r_m, column=3, value=f"{float(last[kf]):.8g}")
                            cv.font = _font(size=11); cv.fill = _fill(C["bg_resumen"])
                            cv.alignment = _left(); cv.border = _border()
                            r_m += 1
                            break
                    # iter count
                    ws_m.row_dimensions[r_m].height = 18
                    cl = ws_m.cell(row=r_m, column=2, value="iter")
                    cl.font = _font(bold=True, size=11, color=C["fg_gold"])
                    cl.fill = _fill(C["bg_resumen"]); cl.alignment = _center(); cl.border = _border()
                    cv = ws_m.cell(row=r_m, column=3, value=len(hist))
                    cv.font = _font(size=11); cv.fill = _fill(C["bg_resumen"])
                    cv.alignment = _left(); cv.border = _border()
                    r_m += 1

                ws_m.row_dimensions[r_m].height = 6; r_m += 1

                if hist:
                    h_cols: list = []
                    for rec in hist:
                        for k in rec:
                            if k not in h_cols: h_cols.append(k)

                    h_rows = []
                    for rec in hist:
                        row_vals = []
                        for col in h_cols:
                            v = rec.get(col, "")
                            if isinstance(v, np.ndarray):
                                v = "[" + ", ".join(f"{x:.5g}" for x in v.flat) + "]"
                            elif isinstance(v, float):
                                v = f"{v:.6g}"
                            row_vals.append(v if isinstance(v, (int, bool)) else str(v))
                        h_rows.append(row_vals)

                    brd = _border()
                    n_data_cols = len(h_cols)

                    # Ajustar anchos SOLO de columnas D en adelante (B y C ya tienen
                    # ancho fijo del bloque de info — 22 y 52 respectivamente)
                    for j, col in enumerate(h_cols, 2):
                        if j <= 3:
                            continue   # B y C ya son 22/52 — no sobreescribir
                        cw = max(len(str(col)) + 2,
                                 max((len(str(rv[j-2])) for rv in h_rows), default=0) + 2)
                        _set_col_width(ws_m, j, min(cw, 24))

                    # ── Sección header "Tabla de iteraciones" ─────────────────
                    last_col_letter = get_column_letter(1 + n_data_cols)
                    _merge_write(ws_m, f"B{r_m}:{last_col_letter}{r_m}",
                        "Tabla de iteraciones",
                        font=_font(bold=True, size=11, color=C["fg_cyan"]),
                        fill=_fill(C["bg_head2"]), align=_center(), height=22)
                    r_m += 1

                    # ── Encabezados de columna ────────────────────────────────
                    col_header_row = r_m   # fila que queremos congelar
                    ws_m.row_dimensions[r_m].height = 20
                    for j, col in enumerate(h_cols, 2):
                        c_cell = ws_m.cell(row=r_m, column=j, value=col)
                        c_cell.font      = _font(bold=True, size=10)
                        c_cell.fill      = _fill(C["bg_head"])
                        c_cell.alignment = _center()
                        c_cell.border    = brd
                    r_m += 1

                    # ── Filas de datos ────────────────────────────────────────
                    for i_r, row_vals in enumerate(h_rows):
                        ws_m.row_dimensions[r_m].height = 15
                        use_alt = i_r % 2 == 1
                        row_fill = _fill(C["bg_alt"]) if use_alt else _fill(C["bg_mid"])
                        row_font_color = "111830" if use_alt else C["fg_white"]
                        for j, val in enumerate(row_vals, 2):
                            c_cell = ws_m.cell(row=r_m, column=j, value=val)
                            c_cell.fill      = row_fill
                            c_cell.font      = _font(size=9, color=row_font_color)
                            c_cell.alignment = _center()
                            c_cell.border    = brd
                        r_m += 1

                    # Sin freeze_panes — todo el documento desplaza normalmente
                else:
                    _merge_write(ws_m, f"B{r_m}:C{r_m}", "Sin datos de iteración.",
                        font=_font(size=10), fill=_fill(C["bg_mid"]),
                        align=_left(), height=18)
                    r_m += 1

                ws_m.row_dimensions[r_m].height = 10; r_m += 1

                # ── Cálculos paso a paso (ANTES de las gráficas) ─────────────
                _merge_write(ws_m, f"B{r_m}:C{r_m}",
                    "Cálculos Paso a Paso",
                    font=_font(bold=True, size=11, color=C["fg_cyan"]),
                    fill=_fill(C["bg_head2"]), align=_center(), height=22)
                r_m += 1

                steps = self._build_step_by_step(entry)
                for step_title, step_body in steps:
                    ws_m.row_dimensions[r_m].height = 16
                    ct = ws_m.cell(row=r_m, column=2, value=self._math_to_unicode(step_title))
                    ct.font = _font(bold=True, size=10, color=C["fg_gold"])
                    ct.fill = _fill(C["bg_head"]); ct.alignment = _left()
                    ct.border = _border()
                    r_m += 1
                    lines = step_body.split("\n")
                    for line in lines:
                        if not line.strip():
                            continue
                        ws_m.row_dimensions[r_m].height = 14
                        cb = ws_m.cell(row=r_m, column=2,
                                       value=self._math_to_unicode(line.strip()))
                        cb.font = _font(size=9)
                        cb.fill = _fill(C["bg_resumen"])
                        cb.alignment = Alignment(horizontal="left", vertical="center",
                                                 wrap_text=True)
                        cb.border = _border()
                        ws_m.merge_cells(f"B{r_m}:C{r_m}")
                        r_m += 1
                    ws_m.row_dimensions[r_m].height = 4; r_m += 1

                ws_m.row_dimensions[r_m].height = 10; r_m += 1

                # ── Gráficas 2D y 3D ─────────────────────────────────────────
                _merge_write(ws_m, f"B{r_m}:C{r_m}", "Gráficas",
                    font=_font(bold=True, size=11, color=C["fg_cyan"]),
                    fill=_fill(C["bg_head2"]), align=_center(), height=22)
                r_m += 1

                imgs = self._generate_plot_png(entry)
                all_tmp_imgs.extend(imgs)

                IMG_ROW_HEIGHT = 15
                IMG_ROWS       = 23

                for path, lbl in imgs:
                    ws_m.row_dimensions[r_m].height = 16
                    _merge_write(ws_m, f"B{r_m}:C{r_m}", lbl,
                        font=_font(bold=True, size=10, color=C["fg_blue"]),
                        fill=_fill(C["bg_mid"]), align=_center(), height=16)
                    r_m += 1

                    img_start_row = r_m
                    for i in range(IMG_ROWS):
                        ws_m.row_dimensions[img_start_row + i].height = IMG_ROW_HEIGHT

                    try:
                        xl_img = XLImg(path)
                        xl_img.width  = 520
                        xl_img.height = IMG_ROWS * IMG_ROW_HEIGHT
                        ws_m.add_image(xl_img, f"{get_column_letter(2)}{img_start_row}")
                    except Exception as ie:
                        ws_m.cell(row=r_m, column=2,
                                  value=f"[imagen no disponible: {ie}]")

                    r_m += IMG_ROWS + 1

            # ── Guardar (imágenes aún existen en disco) ───────────────────────
            wb.save(p)

            # ── Ahora sí borrar temporales ────────────────────────────────────
            for path, _ in all_tmp_imgs:
                try: os.unlink(path)
                except: pass

            self._show_export_success(
                "Excel (.xlsx)", p,
                f"Pestañas: RESUMEN + {len(session)} método(s)")
        except Exception as e:
            # Limpiar temporales aunque falle
            for path, _ in all_tmp_imgs:
                try: os.unlink(path)
                except: pass
            _err(self, "Error Excel", str(e), traceback.format_exc())

    def export_pdf(self):
        """
        PDF profesional en A4 vertical:
          Pág 1: RESUMEN — descripción función, tipo, estudios aplicables,
                           métodos aplicados y justificación de cada uno.
          Pág N: Por cada método — info detallada, tabla de iteraciones,
                 cálculos paso a paso y gráficas 2D / 3D.
        Todo ajustado a los márgenes del documento.
        """
        if self.table.rowCount() == 0:
            return self._nodata()
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                            Table, TableStyle, Image as RLImg,
                                            HRFlowable, PageBreak, KeepTogether,
                                            FrameBreak)
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
            from reportlab.platypus.flowables import HRFlowable
        except ImportError:
            QMsgBox.warning(self, "Dependencia", "pip install reportlab"); return

        p, _ = QFD.getSaveFileName(self, "Guardar PDF", "reporte.pdf", "PDF (*.pdf)")
        if not p: return

        try:
            session = self._session if self._session else [{
                "metodo": self._metodo, "fx": self._fx, "gx": "",
                "vars": list(self._vars), "x0": list(self._x0),
                "lo": self.smin.value(), "hi": self.smax.value(),
                "hist": self._hist,
                "traj": self._traj(self._vars, self._hist, self._x0),
            }]
            fx_global = session[0]["fx"] if session else self._fx

            # A4 vertical con márgenes estándar
            ML = MR = 2.0 * cm
            MT = MB = 2.0 * cm
            doc = SimpleDocTemplate(p, pagesize=A4,
                                    leftMargin=ML, rightMargin=MR,
                                    topMargin=MT, bottomMargin=MB)
            PW = A4[0] - ML - MR   # ancho útil ≈ 17.0 cm

            # ── Paleta ─────────────────────────────────────────────────────────
            cDark  = colors.HexColor("#0A1428")
            cMid   = colors.HexColor("#101E38")
            cHead  = colors.HexColor("#1A3060")
            cHead2 = colors.HexColor("#0D1E44")
            cBlue  = colors.HexColor("#2A50A0")
            cAqua  = colors.HexColor("#5DC8D0")
            cWhite = colors.white
            cAlt   = colors.HexColor("#E8F2FF")
            cGold  = colors.HexColor("#FFD700")
            cGray  = colors.HexColor("#B0C4D8")
            cDkTxt = colors.HexColor("#111830")

            # ── Estilos de párrafo ─────────────────────────────────────────────
            def _PS(name, **kw):
                defaults = dict(fontName="Helvetica", fontSize=10,
                                textColor=cWhite, leading=14)
                defaults.update(kw)
                return ParagraphStyle(name, **defaults)

            sBanner  = _PS("Ban", fontSize=16, fontName="Helvetica-Bold",
                           alignment=TA_CENTER, leading=20, textColor=cWhite)
            sSub     = _PS("Sub", fontSize=10, textColor=cAqua,
                           alignment=TA_CENTER, leading=13)
            sSecHdr  = _PS("SHd", fontSize=12, fontName="Helvetica-Bold",
                           textColor=cAqua, leading=15, spaceBefore=6, spaceAfter=3)
            sBody    = _PS("Bod", fontSize=9, textColor=cGray,
                           alignment=TA_JUSTIFY, leading=13, spaceAfter=3)
            sBodyDk  = _PS("BDk", fontSize=9, textColor=cDkTxt,
                           alignment=TA_JUSTIFY, leading=13)
            sLabel   = _PS("Lbl", fontSize=9, fontName="Helvetica-Bold",
                           textColor=cAqua, leading=12)
            sVal     = _PS("Val", fontSize=9, textColor=cWhite, leading=12)
            sStep    = _PS("Stp", fontSize=9, fontName="Helvetica-Bold",
                           textColor=cGold, leading=12)
            sStepBd  = _PS("StB", fontSize=8, textColor=cGray,
                           leading=11, leftIndent=10)
            sSmall   = _PS("Sml", fontSize=7, textColor=cGray,
                           alignment=TA_CENTER, leading=10)
            sItalic  = _PS("Itl", fontSize=9, fontName="Helvetica-Oblique",
                           textColor=cGray, leading=12, alignment=TA_JUSTIFY)

            def _HR(thick=0.8, color=None):
                return HRFlowable(width="100%", thickness=thick,
                                  color=color or cBlue,
                                  spaceBefore=4, spaceAfter=4)

            def _banner(title, subtitle=""):
                rows = [[Paragraph(title, sBanner)]]
                if subtitle:
                    rows.append([Paragraph(subtitle, sSub)])
                t = Table(rows, colWidths=[PW])
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,-1), cDark),
                    ("TOPPADDING",    (0,0), (-1,-1), 12),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 10),
                    ("LEFTPADDING",   (0,0), (-1,-1), 14),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 14),
                ]))
                return t

            def _section_box(content_rows, col_w=None):
                """Tabla de dos columnas etiqueta/valor con fondo cMid."""
                cw = col_w or [PW * 0.30, PW * 0.70]
                t = Table(content_rows, colWidths=cw)
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,-1), cMid),
                    ("GRID",          (0,0), (-1,-1), 0.3, cBlue),
                    ("TOPPADDING",    (0,0), (-1,-1), 5),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                    ("LEFTPADDING",   (0,0), (-1,-1), 8),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 8),
                    ("VALIGN",        (0,0), (-1,-1), "TOP"),
                ]))
                return t

            def _full_box(text_para):
                """Caja de ancho completo con fondo cMid."""
                t = Table([[text_para]], colWidths=[PW])
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,-1), cMid),
                    ("TOPPADDING",    (0,0), (-1,-1), 8),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                    ("LEFTPADDING",   (0,0), (-1,-1), 10),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 10),
                ]))
                return t

            def _step_box(title, body_lines):
                """Caja de paso con super/subíndices via ReportLab (sin imágenes)."""
                rows = [[Paragraph(self._math_to_reportlab(title), sStep)]]
                for line in body_lines:
                    if line.strip():
                        rows.append([Paragraph(
                            self._math_to_reportlab(line.strip()), sStepBd)])
                t = Table(rows, colWidths=[PW])
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (0,0),  cHead2),
                    ("BACKGROUND",    (0,1), (-1,-1), cMid),
                    ("GRID",          (0,0), (-1,-1), 0.3, cBlue),
                    ("TOPPADDING",    (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                    ("LEFTPADDING",   (0,0), (-1,-1), 8),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 8),
                    ("VALIGN",        (0,0), (-1,-1), "TOP"),
                ]))
                return t

            # ── Estilos BLANCOS para Resumen y Conclusión ─────────────────────
            cBlack   = colors.HexColor("#111111")
            cDkBlue  = colors.HexColor("#1A3A6A")
            cLightBg = colors.HexColor("#F4F7FB")
            cBorder  = colors.HexColor("#B0C4D8")
            cAccent  = colors.HexColor("#1A3A6A")

            sWTitle  = ParagraphStyle("WTi", fontName="Helvetica-Bold", fontSize=18,
                           textColor=colors.white, alignment=TA_CENTER, leading=22)
            sWSub    = ParagraphStyle("WSu", fontName="Helvetica", fontSize=10,
                           textColor=colors.HexColor("#DDECFF"), alignment=TA_CENTER, leading=13)
            sWSecHdr = ParagraphStyle("WSH", fontName="Helvetica-Bold", fontSize=12,
                           textColor=cDkBlue, leading=15, spaceBefore=6, spaceAfter=3)
            sWLabel  = ParagraphStyle("WLb", fontName="Helvetica-Bold", fontSize=9,
                           textColor=cDkBlue, leading=12)
            sWVal    = ParagraphStyle("WVa", fontName="Helvetica", fontSize=9,
                           textColor=cBlack, leading=12)
            sWBody   = ParagraphStyle("WBo", fontName="Helvetica", fontSize=9,
                           textColor=cBlack, alignment=TA_JUSTIFY, leading=13, spaceAfter=3)
            sWStep   = ParagraphStyle("WSt", fontName="Helvetica-Bold", fontSize=9,
                           textColor=colors.white, leading=12)
            sWStepBd = ParagraphStyle("WSb", fontName="Helvetica", fontSize=9,
                           textColor=cBlack, leading=12, leftIndent=10,
                           alignment=TA_JUSTIFY)
            sWSmall  = ParagraphStyle("WSm", fontName="Helvetica", fontSize=7,
                           textColor=colors.HexColor("#666666"), alignment=TA_CENTER, leading=10)
            sWConc   = ParagraphStyle("WCo", fontName="Helvetica", fontSize=9,
                           textColor=cBlack, alignment=TA_JUSTIFY, leading=14, spaceAfter=4)
            sWConcH  = ParagraphStyle("WCH", fontName="Helvetica-Bold", fontSize=10,
                           textColor=cDkBlue, leading=14, spaceBefore=8, spaceAfter=3)

            def _HR_w(thick=0.8):
                return HRFlowable(width="100%", thickness=thick,
                                  color=cBorder, spaceBefore=4, spaceAfter=4)

            def _banner_w(title, subtitle=""):
                """Banner con fondo azul oscuro para la página de resumen."""
                rows = [[Paragraph(title, sWTitle)]]
                if subtitle:
                    rows.append([Paragraph(subtitle, sWSub)])
                t = Table(rows, colWidths=[PW])
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,-1), cDkBlue),
                    ("TOPPADDING",    (0,0), (-1,-1), 14),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 12),
                    ("LEFTPADDING",   (0,0), (-1,-1), 14),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 14),
                ]))
                return t

            def _section_box_w(content_rows, col_w=None):
                """Tabla etiqueta/valor con fondo blanco."""
                cw = col_w or [PW * 0.30, PW * 0.70]
                t = Table(content_rows, colWidths=cw)
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,-1), colors.white),
                    ("GRID",          (0,0), (-1,-1), 0.5, cBorder),
                    ("TOPPADDING",    (0,0), (-1,-1), 5),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                    ("LEFTPADDING",   (0,0), (-1,-1), 8),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 8),
                    ("VALIGN",        (0,0), (-1,-1), "TOP"),
                    ("BACKGROUND",    (0,0), (0,-1),  cLightBg),
                ]))
                return t

            def _full_box_w(text_para):
                """Caja de ancho completo con fondo blanco."""
                t = Table([[text_para]], colWidths=[PW])
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,-1), cLightBg),
                    ("BOX",           (0,0), (-1,-1), 0.5, cBorder),
                    ("TOPPADDING",    (0,0), (-1,-1), 8),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                    ("LEFTPADDING",   (0,0), (-1,-1), 10),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 10),
                ]))
                return t

            def _method_box_w(number, name, body_text):
                """Caja de método con encabezado azul y cuerpo blanco."""
                rows = [
                    [Paragraph(f"{number}. {name}", sWStep)],
                    [Paragraph(body_text, sWStepBd)],
                ]
                t = Table(rows, colWidths=[PW])
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (0,0),  cDkBlue),
                    ("BACKGROUND",    (0,1), (-1,-1), colors.white),
                    ("BOX",           (0,0), (-1,-1), 0.5, cBorder),
                    ("LINEBELOW",     (0,0), (-1,0),  0.5, cBorder),
                    ("TOPPADDING",    (0,0), (-1,-1), 6),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                    ("LEFTPADDING",   (0,0), (-1,-1), 10),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 10),
                ]))
                return t

            elems = []

            # ══════════════════════════════════════════════════════════════════
            # PÁGINA 1-2: RESUMEN GENERAL  (fondo blanco, letras negras)
            # ══════════════════════════════════════════════════════════════════
            elems.append(_banner_w(
                "Optimizador de Funciones",
                f"Reporte de Análisis  ·  {len(session)} método(s) ejecutado(s)"))
            elems.append(Spacer(1, 0.3*cm))

            # ── 1. Función analizada ───────────────────────────────────────────
            elems.append(_HR_w(1.5))
            elems.append(Paragraph("1. Función Analizada", sWSecHdr))
            tipo_fn = self._classify_function(fx_global)
            elems.append(_section_box_w([
                [Paragraph("Expresión:", sWLabel),  Paragraph(fx_global, sWVal)],
                [Paragraph("Tipo:", sWLabel),        Paragraph(tipo_fn, sWVal)],
                [Paragraph("Variables:", sWLabel),   Paragraph(", ".join(session[0]["vars"]), sWVal)],
                [Paragraph("Dominio:", sWLabel),     Paragraph(f"[{session[0]['lo']:.4g},  {session[0]['hi']:.4g}]", sWVal)],
            ]))
            elems.append(Spacer(1, 0.2*cm))
            desc_fn = self._describe_function(fx_global)
            elems.append(_full_box_w(Paragraph(desc_fn, sWBody)))
            elems.append(Spacer(1, 0.25*cm))

            # ── 2. Tipos de estudio aplicables ────────────────────────────────
            elems.append(_HR_w(1.5))
            elems.append(Paragraph("2. Tipos de Estudio Aplicables a la Función", sWSecHdr))
            desc_estudios = self._describe_applicable_studies(fx_global)
            elems.append(_full_box_w(Paragraph(desc_estudios, sWBody)))
            elems.append(Spacer(1, 0.25*cm))

            # ── 3. Métodos aplicados y justificación ──────────────────────────
            elems.append(_HR_w(1.5))
            elems.append(Paragraph("3. Métodos Aplicados", sWSecHdr))
            for i, s in enumerate(session, 1):
                justif = self._justify_method(s["metodo"], fx_global)
                elems.append(Spacer(1, 0.15*cm))
                # KeepTogether: el encabezado azul y el cuerpo blanco
                # nunca se separan entre páginas
                elems.append(KeepTogether([_method_box_w(i, s["metodo"], justif)]))

            elems.append(Spacer(1, 0.15*cm))
            elems.append(_HR_w(0.5))
            elems.append(Paragraph("Generado automáticamente · Optimizador de Funciones", sWSmall))

            # ══════════════════════════════════════════════════════════════════
            # PÁGINAS POR MÉTODO
            # ══════════════════════════════════════════════════════════════════
            all_pdf_imgs: list = []   # acumula (path, lbl) — se borran tras doc.build()
            for idx_s, entry in enumerate(session, 1):
                metodo = entry["metodo"]
                hist   = entry["hist"]
                elems.append(PageBreak())

                # Banner del método + Info: se mantienen juntos en la misma página
                tipo_est = self._classify_study(metodo)
                elems.append(KeepTogether([
                    _banner(
                        f"Método {idx_s}: {metodo}",
                        f"f(x) = {entry['fx']}"),
                    Spacer(1, 0.2*cm),
                    _HR(1.0, cBlue),
                    Paragraph("Información del Método", sSecHdr),
                    _section_box([
                        [Paragraph("Método:", sLabel),          Paragraph(metodo, sVal)],
                        [Paragraph("Función:", sLabel),         Paragraph(entry["fx"], sVal)],
                        [Paragraph("Tipo de estudio:", sLabel), Paragraph(tipo_est, sVal)],
                        [Paragraph("Variables:", sLabel),       Paragraph(", ".join(entry["vars"]), sVal)],
                        [Paragraph("Intervalo:", sLabel),       Paragraph(f"[{entry['lo']:.4g},  {entry['hi']:.4g}]", sVal)],
                    ]),
                    Spacer(1, 0.2*cm),
                ]))

                # ── Tabla de iteraciones ───────────────────────────────────────
                if hist:
                    h_cols: list = []
                    for rec in hist:
                        for k in rec:
                            if k not in h_cols: h_cols.append(k)

                    h_rows_raw = []
                    for rec in hist:
                        row_v = []
                        for col in h_cols:
                            v = rec.get(col, "")
                            if isinstance(v, np.ndarray):
                                v = "[" + ", ".join(f"{x:.4g}" for x in v.flat) + "]"
                            elif isinstance(v, float):
                                v = f"{v:.5g}"
                            row_v.append(str(v))
                        h_rows_raw.append(row_v)

                    nc = len(h_cols)

                    # Calcular ancho proporcional al contenido de cada columna
                    col_max_len = []
                    for j, col in enumerate(h_cols):
                        max_len = len(col)
                        for rv in h_rows_raw:
                            max_len = max(max_len, len(rv[j]))
                        col_max_len.append(max_len)
                    total_chars = sum(col_max_len) or 1
                    col_widths = [PW * (cl / total_chars) for cl in col_max_len]
                    # Mínimo 1.0 cm por columna
                    from reportlab.lib.units import cm as _cm
                    col_widths = [max(w, 1.0*_cm) for w in col_widths]
                    # Si la suma excede PW, escalar hacia abajo
                    total_w = sum(col_widths)
                    if total_w > PW:
                        scale = PW / total_w
                        col_widths = [w * scale for w in col_widths]

                    hdr_st = _PS(f"th{idx_s}", fontSize=7, fontName="Helvetica-Bold",
                                 textColor=cWhite, alignment=TA_CENTER)
                    cel_st = _PS(f"td{idx_s}", fontSize=7, textColor=cWhite,
                                 alignment=TA_CENTER, leading=10)
                    cel_alt= _PS(f"ta{idx_s}", fontSize=7, textColor=cWhite,
                                 alignment=TA_CENTER, leading=10)

                    tbl_data = [[Paragraph(c, hdr_st) for c in h_cols]]
                    for i_r, row_v in enumerate(h_rows_raw):
                        st = cel_alt if i_r % 2 == 1 else cel_st
                        tbl_data.append([Paragraph(v, st) for v in row_v])

                    # Dos tonos oscuros para filas alternas — texto blanco siempre legible
                    cRowA = colors.HexColor("#101E38")   # azul oscuro (igual que cMid)
                    cRowB = colors.HexColor("#162848")   # azul medio-oscuro

                    # Estilo base reutilizable para la tabla de iteraciones
                    def _make_tbl_style(bg0=cRowA, bg1=cRowB):
                        return TableStyle([
                            ("BACKGROUND",    (0,0), (-1,0),  cHead),
                            ("ROWBACKGROUNDS",(0,1), (-1,-1), [bg0, bg1]),
                            ("GRID",          (0,0), (-1,-1), 0.4, cBlue),
                            ("TOPPADDING",    (0,0), (-1,-1), 3),
                            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                            ("LEFTPADDING",   (0,0), (-1,-1), 2),
                            ("RIGHTPADDING",  (0,0), (-1,-1), 2),
                            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                        ])

                    # ── Ancla: título de sección + encabezado + primeras filas
                    # (se mantienen juntos para que el encabezado nunca quede
                    #  huérfano al final de una página)
                    N_ANCHOR = min(4, len(tbl_data))   # encabezado + hasta 3 filas de datos
                    anchor_tbl = Table(tbl_data[:N_ANCHOR], colWidths=col_widths)
                    anchor_tbl.setStyle(_make_tbl_style())
                    elems.append(KeepTogether([
                        _HR(1.0, cBlue),
                        Paragraph("Tabla de Iteraciones", sSecHdr),
                        anchor_tbl,
                    ]))

                    # ── Filas restantes: encabezado repetido al inicio de cada
                    # página nueva, con colores alternados en continuidad
                    if len(tbl_data) > N_ANCHOR:
                        n_anchor_data = N_ANCHOR - 1        # filas de datos ya mostradas
                        bg0 = cRowB if (n_anchor_data % 2 == 1) else cRowA
                        bg1 = cRowA if bg0 == cRowB else cRowB
                        remaining_data = [tbl_data[0]] + tbl_data[N_ANCHOR:]
                        remaining_tbl = Table(
                            remaining_data, colWidths=col_widths, repeatRows=1)
                        remaining_tbl.setStyle(_make_tbl_style(bg0, bg1))
                        elems.append(remaining_tbl)
                else:
                    elems.append(_HR(1.0, cBlue))
                    elems.append(Paragraph("Tabla de Iteraciones", sSecHdr))
                    elems.append(Paragraph("Sin datos de iteración.", sBody))

                # ── Cálculos paso a paso ───────────────────────────────────────
                elems.append(Spacer(1, 0.4*cm))
                steps = self._build_step_by_step(entry)
                if steps:
                    # Anclar el título de la sección con el primer paso para que
                    # el encabezado nunca quede solo al final de una página
                    first_title, first_body = steps[0]
                    first_lines = [l for l in first_body.split("\n") if l.strip()]
                    elems.append(KeepTogether([
                        _HR(1.0, cBlue),
                        Paragraph("Cálculos Paso a Paso", sSecHdr),
                        Spacer(1, 0.15*cm),
                        _step_box(first_title, first_lines),
                    ]))
                    for step_title, step_body in steps[1:]:
                        body_lines = [l for l in step_body.split("\n") if l.strip()]
                        elems.append(Spacer(1, 0.12*cm))
                        # Cada caja se mantiene unida: el título nunca queda
                        # solo al fondo de una página separado de su contenido
                        elems.append(KeepTogether([_step_box(step_title, body_lines)]))
                    elems.append(Spacer(1, 0.12*cm))
                else:
                    elems.append(_HR(1.0, cBlue))
                    elems.append(Paragraph("Cálculos Paso a Paso", sSecHdr))

                # ── Gráficas (siempre en página nueva) ────────────────────────
                elems.append(PageBreak())
                imgs = self._generate_plot_png(entry)
                all_pdf_imgs.extend(imgs)

                if imgs:
                    img_w = (PW - 0.4*cm) / 2
                    img_h = img_w * 0.55

                    def _img_cell(path, lbl):
                        try:
                            return [
                                Paragraph(lbl, _PS(f"il{lbl}",
                                    fontSize=8, textColor=cAqua,
                                    alignment=TA_CENTER,
                                    fontName="Helvetica-Bold")),
                                RLImg(path, width=img_w, height=img_h)
                            ]
                        except:
                            return [Paragraph(lbl, sSmall),
                                    Paragraph("[imagen no disponible]", sSmall)]

                    pairs = [imgs[i:i+2] for i in range(0, len(imgs), 2)]
                    img_tables = []
                    for pair in pairs:
                        row_cells = [_img_cell(path, lbl) for path, lbl in pair]
                        while len(row_cells) < 2:
                            row_cells.append([Spacer(1,1), Spacer(1,1)])
                        rt = Table([row_cells], colWidths=[PW/2, PW/2])
                        rt.setStyle(TableStyle([
                            ("VALIGN",        (0,0), (-1,-1), "TOP"),
                            ("LEFTPADDING",   (0,0), (-1,-1), 2),
                            ("RIGHTPADDING",  (0,0), (-1,-1), 2),
                            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                        ]))
                        img_tables.append(rt)
                        img_tables.append(Spacer(1, 0.2*cm))

                    # Banner de gráficas + primera fila juntos (no se parte)
                    graf_banner = _banner(f"Gráficas — {metodo}", f"f(x) = {entry['fx']}")
                    first_block = [graf_banner, Spacer(1, 0.25*cm)]
                    if img_tables:
                        first_block.append(img_tables[0])
                    elems.append(KeepTogether(first_block))
                    # Resto de filas de imágenes fluyen libremente
                    for tbl in img_tables[1:]:
                        elems.append(tbl)


            # ══════════════════════════════════════════════════════════════════
            # PÁGINA FINAL: CONCLUSIÓN GENERALIZADA  (fondo blanco, letra negra)
            # ══════════════════════════════════════════════════════════════════
            elems.append(PageBreak())
            elems.append(_banner_w(
                "Conclusión General",
                f"Análisis de f(x) = {fx_global}  ·  {len(session)} método(s)"))
            elems.append(Spacer(1, 0.35*cm))

            # ── Qué se hizo ────────────────────────────────────────────────────
            elems.append(_HR_w(1.5))
            elems.append(Paragraph("¿Qué se realizó?", sWSecHdr))
            metodos_lista = ", ".join(s["metodo"] for s in session)
            n_iter_total  = sum(len(s.get("hist") or []) for s in session)
            qué_texto = (
                f"Se llevó a cabo el análisis y optimización numérica de la función "
                f"f(x) = {fx_global}, clasificada como función de tipo "
                f"{self._classify_function(fx_global)}. "
                f"Para ello se aplicaron {len(session)} método(s) de optimización: "
                f"{metodos_lista}. "
                f"En total se ejecutaron {n_iter_total} iteración(es) computacionales, "
                f"registrando en cada paso los valores del punto actual, el valor de la "
                f"función, el gradiente, el tamaño de paso y los criterios de convergencia. "
                f"Los resultados se presentan en tablas de iteraciones completas y se "
                f"visualizan mediante gráficas 2D y 3D del comportamiento de la función "
                f"y la trayectoria de convergencia de cada método."
            )
            elems.append(_full_box_w(Paragraph(qué_texto, sWConc)))
            elems.append(Spacer(1, 0.25*cm))

            # ── Ventajas y beneficios ──────────────────────────────────────────
            elems.append(_HR_w(1.5))
            elems.append(Paragraph("Ventajas y Beneficios", sWSecHdr))

            ventajas = [
                ("Diversidad metodológica",
                 "La aplicación de múltiples métodos sobre la misma función permite "
                 "comparar su eficiencia, velocidad de convergencia y robustez, "
                 "eligiendo el más adecuado según las características del problema."),
                ("Trazabilidad completa",
                 "El registro iteración por iteración de todos los parámetros "
                 "(x_k, f(x_k), ‖∇f‖, α, dirección de descenso) garantiza total "
                 "transparencia del proceso numérico y facilita la detección de "
                 "anomalías o comportamientos inesperados."),
                ("Visualización clara",
                 "Las gráficas 2D y 3D permiten interpretar visualmente el "
                 "comportamiento de la función y la trayectoria de convergencia, "
                 "facilitando la comprensión sin necesidad de interpretar solo datos numéricos."),
                ("Exportación profesional",
                 "La generación automática de reportes en Excel y PDF con todos los "
                 "cálculos paso a paso reduce el tiempo de documentación y garantiza "
                 "reproducibilidad y presentación formal de los resultados."),
                ("Criterios de convergencia garantizados",
                 "Los métodos Armijo y Wolfe incluyen condiciones matemáticas "
                 "verificadas en cada iteración, asegurando que el algoritmo avance "
                 "hacia el óptimo sin pasos excesivamente grandes o pequeños."),
            ]
            for titulo_v, texto_v in ventajas:
                elems.append(Spacer(1, 0.1*cm))
                v_rows = [
                    [Paragraph(f"✦  {titulo_v}", sWLabel)],
                    [Paragraph(texto_v, sWConc)],
                ]
                vt = Table(v_rows, colWidths=[PW])
                vt.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (0,0),  cLightBg),
                    ("BACKGROUND",    (0,1), (-1,-1), colors.white),
                    ("BOX",           (0,0), (-1,-1), 0.4, cBorder),
                    ("LINEBELOW",     (0,0), (-1,0),  0.4, cBorder),
                    ("TOPPADDING",    (0,0), (-1,-1), 5),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                    ("LEFTPADDING",   (0,0), (-1,-1), 10),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 10),
                ]))
                elems.append(vt)

            elems.append(Spacer(1, 0.25*cm))

            # ── Utilidad futura ────────────────────────────────────────────────
            elems.append(_HR_w(1.5))
            elems.append(Paragraph("Utilidad en Diversas Áreas a Futuro", sWSecHdr))

            areas = [
                ("Ingeniería y Diseño",
                 "Los métodos de optimización son fundamentales en el diseño de "
                 "estructuras, circuitos eléctricos, sistemas de control y rutas "
                 "logísticas. La capacidad de encontrar mínimos de funciones de costo "
                 "o máximos de funciones de rendimiento es clave en cualquier proceso "
                 "de diseño óptimo."),
                ("Inteligencia Artificial y Machine Learning",
                 "El entrenamiento de redes neuronales es en esencia un problema de "
                 "minimización de una función de pérdida en miles de dimensiones. "
                 "Los métodos de descenso de gradiente con condiciones de Wolfe/Armijo "
                 "son la base de optimizadores como Adam, SGD y L-BFGS usados en "
                 "frameworks como TensorFlow y PyTorch."),
                ("Economía y Finanzas",
                 "La optimización de portafolios de inversión, la minimización de "
                 "riesgo financiero y la maximización de utilidad en modelos "
                 "econométricos dependen directamente de métodos numéricos robustos "
                 "como los implementados en esta herramienta."),
                ("Ciencias Naturales e Investigación",
                 "En física, química y biología computacional, la minimización de "
                 "energía de sistemas moleculares, la calibración de modelos y la "
                 "estimación de parámetros de ecuaciones diferenciales son problemas "
                 "de optimización donde estos métodos encuentran aplicación directa."),
                ("Robótica y Control Automático",
                 "La planificación de trayectorias de robots, el control predictivo "
                 "de procesos industriales (MPC) y la calibración de controladores "
                 "PID óptimos son aplicaciones directas de los métodos de optimización "
                 "con restricciones estudiados en este reporte."),
            ]
            for area, desc_area in areas:
                elems.append(Spacer(1, 0.1*cm))
                elems.append(Paragraph(f"▸  {area}", sWConcH))
                elems.append(_full_box_w(Paragraph(desc_area, sWConc)))

            elems.append(Spacer(1, 0.3*cm))
            elems.append(_HR_w(1.0))
            elems.append(Paragraph(
                f"Reporte generado automáticamente por el Optimizador de Funciones  "
                f"·  {len(session)} método(s) analizado(s)  ·  f(x) = {fx_global}",
                sWSmall))

            # ── Construir PDF (imágenes aún existen en disco) ─────────────────
            doc.build(elems)

            # ── Ahora sí borrar temporales ────────────────────────────────────
            for path, _ in all_pdf_imgs:
                try: os.unlink(path)
                except: pass

            self._show_export_success(
                "PDF", p,
                f"Páginas: Resumen + {len(session)} método(s) con cálculos y gráficas")
        except Exception as e:
            # Limpiar temporales aunque falle
            for path, _ in all_pdf_imgs:
                try: os.unlink(path)
                except: pass
            _err(self, "Error PDF", str(e), traceback.format_exc())



# ══════════════════════════════════════════════════════════════════════════════
def main():
    app = QApp(sys.argv)
    app.setApplicationName("Optimizador de Funciones")
    win = InterfazOptimizacion()
    win.showMaximized()
    sys.exit(app.exec() if _QT=="PyQt6" else app.exec_())

if __name__ == "__main__":
    main()
