from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class Function2DCanvas(FigureCanvas):
    """
    Canvas 2D:
    - 1D: curva f(x) + recorrido
    - MD: contornos f(x1,x2) + recorrido
    - Click + drag para seleccionar punto
    """

    def __init__(self, parent=None, title="Gráfica 2D"):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._title = title

        self._callback = None
        self._selected = None
        self._dragging = False

        self._style()
        self._connect_events()

    def _style(self):
        self.ax.set_title(self._title)
        self.ax.grid(True, alpha=0.3)
        for s in self.ax.spines.values():
            s.set_linewidth(2)

    def set_point_changed_callback(self, fn):
        self._callback = fn

    def _connect_events(self):
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_move)

    def _on_press(self, e):
        if e.inaxes != self.ax:
            return
        if e.xdata is None or e.ydata is None:
            return
        x, y = float(e.xdata), float(e.ydata)

        if self._selected is None:
            self._selected = self.ax.scatter([x], [y], s=90, color="black", zorder=10)
        else:
            self._selected.set_offsets([[x, y]])

        self._dragging = True
        self.draw_idle()
        if self._callback:
            self._callback(x, y)

    def _on_move(self, e):
        if not self._dragging or e.inaxes != self.ax:
            return
        if e.xdata is None or e.ydata is None:
            return
        x, y = float(e.xdata), float(e.ydata)
        self._selected.set_offsets([[x, y]])
        self.draw_idle()
        if self._callback:
            self._callback(x, y)

    def _on_release(self, e):
        self._dragging = False

    # ---------- PLOTS ----------

    def plot_1d(self, x, fx, xk, fk):
        self.ax.clear()
        self._style()
        self.ax.plot(x, fx, color="blue", lw=2, label="f(x)")
        self.ax.plot(xk, fk, color="orange", lw=2, label="Recorrido")
        self.ax.scatter(xk, fk, color="red")
        self.ax.scatter([xk[-1]], [fk[-1]], color="black", s=80)
        self.ax.legend()
        self.draw_idle()

    def plot_contours(self, X, Y, Z, xk, yk):
        self.ax.clear()
        self._style()
        self.ax.contour(X, Y, Z, 20, colors="gray")
        self.ax.plot(xk, yk, color="orange", lw=2)
        self.ax.scatter(xk, yk, color="red")
        self.ax.scatter([xk[-1]], [yk[-1]], color="black", s=80)
        self.draw_idle()
