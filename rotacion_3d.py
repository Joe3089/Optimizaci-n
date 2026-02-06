from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class Rotating3DCanvas(FigureCanvas):
   """
Canvas 3D giratorio (API compatible).

Soporta:
- set_title(str)
- set_surface_and_path(X,Y,Z, path_x,path_y,path_z, labels=...)
- set_curves(x,y,z, labels=...)
- set_data(d1,d2,d3, labels=...)
- start_rotation() / stop_rotation()
"""


def __init__(self, parent=None, interval_ms=50, title="Gráfica 3D"):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._title = title
        self._angle = 30

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(interval_ms)
        self._timer.start()

def _tick(self):
        self._angle = (self._angle + 1) % 360
        self.ax.view_init(elev=30, azim=self._angle)
        self.ax.set_title(self._title)
        self.draw_idle()

def start_rotation(self):
        if not self._timer.isActive():
            self._timer.start()

def stop_rotation(self):
        if self._timer.isActive():
            self._timer.stop()

def set_title(self, title: str):
        self._title = title
        self.ax.set_title(self._title)
        self.draw_idle()

def clear(self, msg: str = None):
        self.ax.cla()
        self.ax.set_title(self._title)
        if msg:
            self.ax.text2D(0.5, 0.5, msg, transform=self.ax.transAxes, ha="center", va="center")
        self.draw_idle()

def set_surface_and_path(self, X, Y, Z, path_x=None, path_y=None, path_z=None,
                             labels=("x1", "x2", "f(x)"), surface_alpha=0.55):
        self.ax.cla()
        Zm = np.ma.masked_invalid(Z)
        try:
            self.ax.plot_surface(X, Y, Zm, rstride=1, cstride=1, linewidth=0,
                                 antialiased=True, alpha=surface_alpha)
        except Exception:
            pass

        if path_x is not None and len(path_x) > 0:
            self.ax.plot(path_x, path_y, path_z, color="tab:orange", linewidth=2.5)
            self.ax.scatter(path_x, path_y, path_z, color="tab:red", s=32)
            self.ax.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], color="black", s=70)

        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        self.ax.set_zlabel(labels[2])
        self.ax.set_title(self._title)
        self.draw_idle()

def set_curve3d(self, x, y, z, labels=("k", "x", "f(x)")):
        self.ax.cla()
        if len(x) > 0:
            self.ax.plot(x, y, z, color="tab:orange", linewidth=2.5)
            self.ax.scatter(x, y, z, color="tab:red", s=32)
            self.ax.scatter([x[-1]], [y[-1]], [z[-1]], color="black", s=70)
        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        self.ax.set_zlabel(labels[2])
        self.ax.set_title(self._title)
        self.draw_idle()

def set_data(self, d1, d2, d3, labels=("x1", "x2", "f(x)")):
        self.set_curve3d(d1, d2, d3, labels=labels)
