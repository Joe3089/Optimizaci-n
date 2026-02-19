# canvas_2d.py
# Canvas 2D (Matplotlib + Qt) con soporte de animación de trayectoria.
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# QtCore para QTimer (PyQt6/PyQt5)
try:
    from PyQt6 import QtCore
except Exception:  # PyQt5
    from PyQt5 import QtCore  # type: ignore


class Function2DCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        if parent is not None:
            self.setParent(parent)

        self.ax = self.fig.add_subplot(111)

        # animación
        self._anim_timer: Optional[QtCore.QTimer] = None
        self._anim_idx: int = 0
        self._anim_path: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._traj_line = None
        self._traj_scatter = None

        self._style_axes()

    def _style_axes(self):
        self.ax.grid(True, alpha=0.25)

    def clear(self):
        self.stop_animation()
        self.ax.clear()
        self._style_axes()
        self.draw_idle()

    def stop_animation(self):
        if self._anim_timer is not None:
            try:
                self._anim_timer.stop()
            except Exception:
                pass
            self._anim_timer = None

    def set_title(self, title: str):
        self.ax.set_title(title or "")
        self.draw_idle()

    # ---------------- 1D: f(x) + trayectoria ----------------
    def plot_1d(self, x: np.ndarray, y: np.ndarray,
                traj_x: Optional[np.ndarray] = None,
                traj_y: Optional[np.ndarray] = None,
                title: str = "Gráfica 2D"):
        self.stop_animation()
        self.ax.clear()
        self._style_axes()

        # función (un color)
        self.ax.plot(x, y, color="tab:blue", linewidth=2)

        # trayectoria (otro color)
        if traj_x is not None and traj_x.size:
            if traj_y is None or not traj_y.size:
                traj_y = np.zeros_like(traj_x)
            self.ax.plot(traj_x, traj_y, linestyle="--", color="tab:orange", linewidth=2)
            self.ax.scatter(traj_x, traj_y, s=26, color="tab:red", zorder=3)

        self.ax.set_title(title)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.draw_idle()

    def animate_1d(self, x: np.ndarray, y: np.ndarray,
                   traj_x: np.ndarray, traj_y: np.ndarray,
                   title: str = "Gráfica 2D",
                   interval_ms: int = 60):
        """Dibuja f(x) y anima la trayectoria punto a punto."""
        self.stop_animation()
        self.ax.clear()
        self._style_axes()

        # función
        self.ax.plot(x, y, color="tab:blue", linewidth=2)

        # artistas trayectoria (inician vacíos)
        self._traj_line, = self.ax.plot([], [], linestyle="--", color="tab:orange", linewidth=2)
        self._traj_scatter = self.ax.scatter([], [], s=26, color="tab:red", zorder=3)

        self.ax.set_title(title)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")

        self._anim_path = (np.asarray(traj_x, dtype=float), np.asarray(traj_y, dtype=float))
        self._anim_idx = 0

        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.timeout.connect(self._step_anim_1d)  # type: ignore
        self._anim_timer.start(max(10, int(interval_ms)))
        self.draw_idle()

    def _step_anim_1d(self):
        if self._anim_path is None:
            self.stop_animation()
            return
        tx, ty = self._anim_path
        if tx.size == 0:
            self.stop_animation()
            return

        self._anim_idx = min(self._anim_idx + 1, tx.size)
        xs = tx[:self._anim_idx]
        ys = ty[:self._anim_idx]

        if self._traj_line is not None:
            self._traj_line.set_data(xs, ys)
        if self._traj_scatter is not None:
            self._traj_scatter.set_offsets(np.c_[xs, ys])

        self.draw_idle()
        if self._anim_idx >= tx.size:
            self.stop_animation()

    # ---------------- 2D: contornos + trayectoria ----------------
    def plot_2d_contour(self, X1: np.ndarray, X2: np.ndarray, Z: np.ndarray,
                        path_x: Optional[np.ndarray] = None,
                        path_y: Optional[np.ndarray] = None,
                        title: str = "Gráfica 2D"):
        self.stop_animation()
        self.ax.clear()
        self._style_axes()

        self.ax.contour(X1, X2, Z, levels=20, colors="tab:blue", linewidths=1.0)

        if path_x is not None and path_x.size and path_y is not None and path_y.size:
            self.ax.plot(path_x, path_y, linestyle="--", color="tab:orange", linewidth=2)
            self.ax.scatter(path_x, path_y, s=26, color="tab:red", zorder=3)

        self.ax.set_title(title)
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.draw_idle()

    def animate_2d_contour(self, X1: np.ndarray, X2: np.ndarray, Z: np.ndarray,
                           path_x: np.ndarray, path_y: np.ndarray,
                           title: str = "Gráfica 2D",
                           interval_ms: int = 60):
        self.stop_animation()
        self.ax.clear()
        self._style_axes()

        self.ax.contour(X1, X2, Z, levels=20, colors="tab:blue", linewidths=1.0)

        self._traj_line, = self.ax.plot([], [], linestyle="--", color="tab:orange", linewidth=2)
        self._traj_scatter = self.ax.scatter([], [], s=26, color="tab:red", zorder=3)

        self.ax.set_title(title)
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")

        self._anim_path = (np.asarray(path_x, dtype=float), np.asarray(path_y, dtype=float))
        self._anim_idx = 0

        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.timeout.connect(self._step_anim_2d)  # type: ignore
        self._anim_timer.start(max(10, int(interval_ms)))
        self.draw_idle()

    def _step_anim_2d(self):
        if self._anim_path is None:
            self.stop_animation()
            return
        px, py = self._anim_path
        if px.size == 0:
            self.stop_animation()
            return
        self._anim_idx = min(self._anim_idx + 1, px.size)
        xs = px[:self._anim_idx]
        ys = py[:self._anim_idx]

        if self._traj_line is not None:
            self._traj_line.set_data(xs, ys)
        if self._traj_scatter is not None:
            self._traj_scatter.set_offsets(np.c_[xs, ys])

        self.draw_idle()
        if self._anim_idx >= px.size:
            self.stop_animation()
