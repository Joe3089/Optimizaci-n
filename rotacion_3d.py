# rotacion_3d.py
# Canvas 3D (Matplotlib + Qt) con superficie y trayectoria animada + rotación.
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

try:
    from PyQt6 import QtCore
except Exception:
    from PyQt5 import QtCore  # type: ignore


class Rotating3DCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        if parent is not None:
            self.setParent(parent)

        self.ax = self.fig.add_subplot(111, projection="3d")

        # timers
        self._rot_timer: Optional[QtCore.QTimer] = None
        self._anim_timer: Optional[QtCore.QTimer] = None

        # rotation state
        self._azim = -60.0
        self._elev = 25.0

        # animation state
        self._anim_idx: int = 0
        self._anim_path: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._traj_line = None
        self._traj_scatter = None

        self._set_default_view()

    def _set_default_view(self):
        try:
            self.ax.view_init(elev=self._elev, azim=self._azim)
        except Exception:
            pass

    def clear(self):
        self.stop_rotation()
        self.stop_animation()
        self.ax.clear()
        self._set_default_view()
        self.draw_idle()

    def set_title(self, title: str):
        self.ax.set_title(title or "")
        self.draw_idle()

    # -------- Rotación --------
    def start_rotation(self, interval_ms: int = 60, delta_azim: float = 1.5):
        self.stop_rotation()
        self._rot_timer = QtCore.QTimer(self)
        def _tick():
            self._azim = (self._azim + float(delta_azim)) % 360.0
            try:
                self.ax.view_init(elev=self._elev, azim=self._azim)
            except Exception:
                pass
            self.draw_idle()
        self._rot_timer.timeout.connect(_tick)  # type: ignore
        self._rot_timer.start(max(10, int(interval_ms)))

    def stop_rotation(self):
        if self._rot_timer is not None:
            try:
                self._rot_timer.stop()
            except Exception:
                pass
            self._rot_timer = None

    # -------- Animación trayectoria --------
    def stop_animation(self):
        if self._anim_timer is not None:
            try:
                self._anim_timer.stop()
            except Exception:
                pass
            self._anim_timer = None
        self._anim_path = None

    # -------- Surface + trayectoria (static) --------
    def set_surface_and_path(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                             path_x: Optional[np.ndarray] = None,
                             path_y: Optional[np.ndarray] = None,
                             path_z: Optional[np.ndarray] = None,
                             title: str = "Gráfica 3D"):
        self.stop_animation()
        self.ax.clear()
        self._set_default_view()

        self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0,
                             antialiased=True, alpha=0.6, color="tab:blue")

        if path_x is not None and path_x.size and path_y is not None and path_y.size:
            if path_z is None or not path_z.size:
                path_z = np.zeros_like(path_x)
            self.ax.plot(path_x, path_y, path_z, linestyle="--", color="tab:orange", linewidth=2)
            self.ax.scatter(path_x, path_y, path_z, s=26, color="tab:red")

        self.ax.set_title(title)
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.ax.set_zlabel("f(x)")
        self.draw_idle()

    def animate_surface_and_path(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                 path_x: np.ndarray, path_y: np.ndarray, path_z: np.ndarray,
                                 title: str = "Gráfica 3D",
                                 interval_ms: int = 60,
                                 rotate: bool = True):
        """Dibuja superficie y anima trayectoria. Opcionalmente rota la cámara."""
        self.stop_animation()
        self.ax.clear()
        self._set_default_view()

        self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0,
                             antialiased=True, alpha=0.6, color="tab:blue")

        self._traj_line, = self.ax.plot([], [], [], linestyle="--", color="tab:orange", linewidth=2)
        self._traj_scatter = self.ax.scatter([], [], [], s=26, color="tab:red")

        self.ax.set_title(title)
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.ax.set_zlabel("f(x)")

        self._anim_path = (np.asarray(path_x, dtype=float), np.asarray(path_y, dtype=float), np.asarray(path_z, dtype=float))
        self._anim_idx = 0

        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.timeout.connect(self._step_anim_3d)  # type: ignore
        self._anim_timer.start(max(10, int(interval_ms)))

        if rotate:
            self.start_rotation()

        self.draw_idle()

    def _step_anim_3d(self):
        if self._anim_path is None:
            self.stop_animation()
            return
        px, py, pz = self._anim_path
        if px.size == 0:
            self.stop_animation()
            return

        self._anim_idx = min(self._anim_idx + 1, px.size)
        xs = px[:self._anim_idx]
        ys = py[:self._anim_idx]
        zs = pz[:self._anim_idx]

        if self._traj_line is not None:
            self._traj_line.set_data(xs, ys)
            self._traj_line.set_3d_properties(zs)
        if self._traj_scatter is not None:
            # matplotlib 3d scatter update
            try:
                self._traj_scatter._offsets3d = (xs, ys, zs)
            except Exception:
                pass

        self.draw_idle()
        if self._anim_idx >= px.size:
            self.stop_animation()
