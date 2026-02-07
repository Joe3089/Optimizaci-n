# canvas_2d.py
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Function2DCanvas(FigureCanvas):
    """
    Canvas 2D (Matplotlib + Qt)
    - Dibuja contornos de f(x1,x2) y la trayectoria de puntos (x1,x2)
    - Puntos de trayectoria en color distinto de la línea
    - Interacción: click/drag para mover un punto (callback opcional)
    """

    def __init__(self, parent=None, title: str = "Gráfica 2D"):
        self.fig = Figure()
        super().__init__(self.fig)
        if parent is not None:
            self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self._title = title

        # data
        self._X = None
        self._Y = None
        self._Z = None
        self._path_xy: List[Tuple[float, float]] = []

        # interaction
        self._sel_idx: Optional[int] = None
        self._dragging = False
        self._on_point_changed: Optional[Callable[[int, float, float], None]] = None

        self._cid_press = self.mpl_connect("button_press_event", self._on_press)
        self._cid_release = self.mpl_connect("button_release_event", self._on_release)
        self._cid_move = self.mpl_connect("motion_notify_event", self._on_move)

        self._redraw()

    def set_point_changed_callback(self, fn: Callable[[int, float, float], None]):
        """fn(idx, x, y) será llamado al hacer click o arrastrar el punto."""
        self._on_point_changed = fn

    def set_title(self, title: str):
        self._title = title
        try:
            self.ax.set_title(self._title)
            self.draw_idle()
        except Exception:
            pass

    def clear(self):
        self._X = self._Y = self._Z = None
        self._path_xy = []
        self._redraw()

    def set_contour(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
        self._X, self._Y, self._Z = X, Y, Z
        self._redraw()

    def set_path(self, path_xy: Sequence[Sequence[float]]):
        self._path_xy = [(float(p[0]), float(p[1])) for p in path_xy] if path_xy else []
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax.set_title(self._title)
        self.ax.grid(True, alpha=0.3)

        # Contours
        if self._X is not None and self._Y is not None and self._Z is not None:
            try:
                self.ax.contour(self._X, self._Y, self._Z, levels=20, linewidths=1.0)
            except Exception:
                # fallback
                pass

        # Path: line + points (different styles)
        if self._path_xy:
            xs = [p[0] for p in self._path_xy]
            ys = [p[1] for p in self._path_xy]
            self.ax.plot(xs, ys, linewidth=2.0)          # line
            self.ax.scatter(xs, ys, s=35)                # points

        self.fig.tight_layout()
        self.draw_idle()

    # ---------------- interaction ----------------
    def _nearest_point(self, x: float, y: float, tol_px: float = 12.0) -> Optional[int]:
        if not self._path_xy:
            return None
        # transform data coords to display coords
        pts = np.array(self._path_xy, dtype=float)
        disp = self.ax.transData.transform(pts)
        q = self.ax.transData.transform(np.array([[x, y]], dtype=float))[0]
        d = np.sqrt(((disp - q) ** 2).sum(axis=1))
        i = int(np.argmin(d))
        return i if d[i] <= tol_px else None

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        idx = self._nearest_point(event.xdata, event.ydata)
        if idx is None:
            return
        self._sel_idx = idx
        self._dragging = True
        if self._on_point_changed:
            self._on_point_changed(idx, float(event.xdata), float(event.ydata))

    def _on_release(self, event):
        self._dragging = False
        self._sel_idx = None

    def _on_move(self, event):
        if not self._dragging or self._sel_idx is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        # update point
        idx = self._sel_idx
        self._path_xy[idx] = (float(event.xdata), float(event.ydata))
        if self._on_point_changed:
            self._on_point_changed(idx, float(event.xdata), float(event.ydata))
        self._redraw()
