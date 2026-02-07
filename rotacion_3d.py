# rotacion_3d.py
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Rotating3DCanvas(FigureCanvas):
    """
    Canvas 3D (Matplotlib + Qt)
    - Superficie z=f(x1,x2) (wireframe/ surface) y trayectoria 3D (x1,x2,f)
    - Rotación automática con timer (start_rotation/stop_rotation)
    """

    def __init__(self, parent=None, title: str = "Gráfica 3D"):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setSizePolicy(self.sizePolicy().Expanding, self.sizePolicy().Expanding)
        if parent is not None:
            self.setParent(parent)

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
        self._title = title

        self._X = self._Y = self._Z = None
        self._path = None  # Nx3

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._azim = 45
        self._elev = 25

        self._redraw()

    def set_title(self, title: str):
        self._title = title
        try:
            self.ax.set_title(self._title)
            self.draw_idle()
        except Exception:
            pass

    def clear(self):
        self._X = self._Y = self._Z = None
        self._path = None
        self._redraw()

    def set_surface(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
        self._X, self._Y, self._Z = X, Y, Z
        self._redraw()

    def set_path_xyz(self, path_xyz: Sequence[Sequence[float]]):
        self._path = np.array(path_xyz, dtype=float) if path_xyz is not None else None
        self._redraw()

    def start_rotation(self, interval_ms: int = 50):
        if not self._timer.isActive():
            self._timer.start(interval_ms)

    def stop_rotation(self):
        if self._timer.isActive():
            self._timer.stop()

    def _tick(self):
        self._azim = (self._azim + 1) % 360
        self.ax.view_init(elev=self._elev, azim=self._azim)
        self.draw_idle()

    def _redraw(self):
        self.ax.clear()
        self.ax.set_title(self._title)

        if self._X is not None and self._Y is not None and self._Z is not None:
            try:
                # lightweight surface
                self.ax.plot_surface(self._X, self._Y, self._Z, rstride=2, cstride=2, linewidth=0, antialiased=True, alpha=0.65)
            except Exception:
                pass

        if self._path is not None and len(self._path) >= 1:
            xs, ys, zs = self._path[:, 0], self._path[:, 1], self._path[:, 2]
            self.ax.plot(xs, ys, zs, linewidth=2.0)
            self.ax.scatter(xs, ys, zs, s=20)

        self.fig.tight_layout()
        self.draw_idle()
