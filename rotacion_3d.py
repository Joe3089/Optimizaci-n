from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class Rotating3DCanvas(FigureCanvas):
    """
    Canvas 3D:
    - Superficie f(x1,x2)
    - Trayectoria del método
    - Rotación automática
    """

    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.angle = 30

        self.timer = QTimer()
        self.timer.timeout.connect(self._rotate)
        self.timer.start(50)

    def _rotate(self):
        self.angle = (self.angle + 1) % 360
        self.ax.view_init(30, self.angle)
        self.draw_idle()

    def plot_surface_and_path(self, X, Y, Z, xk, yk, fk):
        self.ax.clear()
        self.ax.plot_surface(X, Y, Z, alpha=0.5, cmap="viridis")
        self.ax.plot(xk, yk, fk, color="orange", lw=2)
        self.ax.scatter(xk, yk, fk, color="red")
        self.ax.scatter([xk[-1]], [yk[-1]], [fk[-1]], color="black", s=80)
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.ax.set_zlabel("f(x)")
        self.draw_idle()
