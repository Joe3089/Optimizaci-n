# -*- coding: utf-8 -*-
"""
rotacion_3d.py
Canvas 3D (matplotlib + PyQt5) que rota automáticamente la vista.
Expone: Rotating3DCanvas (alias Rotating3dCanvas por compatibilidad)
"""

from PyQt5.QtCore import QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Rotating3DCanvas(FigureCanvas):
    def __init__(self, parent=None, interval_ms: int = 50):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111, projection="3d")
        self._angle = 30
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(interval_ms)

    def _tick(self):
        self._angle = (self._angle + 1) % 360
        try:
            self.ax.view_init(elev=25, azim=self._angle)
            self.draw_idle()
        except Exception:
            pass


    def set_data(self, iters, xs, zs):
        """Actualiza la curva 3D (iteración, x, f(x)) en el dashboard."""
        self.ax.cla()
        self.ax.plot(iters, xs, zs, linewidth=2)
        if len(iters) > 0:
            self.ax.scatter([iters[-1]], [xs[-1]], [zs[-1]], s=35)
        self.ax.set_title("Gráfica 3D")
        self.ax.set_xlabel("Iteración")
        self.ax.set_ylabel("x")
        self.ax.set_zlabel("f(x)")
        self.draw_idle()

    def start_rotation(self):
        """Inicia la rotación automática (si está detenida)."""
        if not self._timer.isActive():
            self._timer.start()

    def stop_rotation(self):
        """Detiene la rotación automática."""
        if self._timer.isActive():
            self._timer.stop()

    def clear(self):
        self.ax.cla()
        self.draw_idle()

    def plot_curve(self, xs, ys, zs=None, title="Gráfica 3D"):
        self.ax.cla()
        if zs is None:
            zs = ys
            ys = [0 for _ in xs]
        self.ax.plot(xs, ys, zs)
        self.ax.set_title(title)
        self.draw_idle()


# compat alias
Rotating3dCanvas = Rotating3DCanvas
