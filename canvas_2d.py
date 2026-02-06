from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class Function2DCanvas(FigureCanvas):
   """
Canvas 2D:
- 1D: f(x) + trayectoria (x_k, f(x_k))
- 2D: contornos de f(x1, x2) + trayectoria (x1_k, x2_k)
- Interacción: click/drag para mover un punto (callback opcional)
"""

def __init__(self, parent=None):
        fig = Figure()
        super().__init__(fig)      
        if parent is not None:
            self.setParent(parent) 

        self.fig = fig
        self.ax = self.fig.add_subplot(111)

        self._title = "Gráfica 2D"
        self._sel = None
        self._dragging = False
        self._on_point_changed = None

def _style_axes(self):
        self.ax.set_title(self._title)
        self.ax.grid(True, alpha=0.35)
        for s in self.ax.spines.values():
            s.set_linewidth(2.0)
        self.fig.tight_layout()

def set_title(self, title: str):
        self._title = title
        self.ax.set_title(self._title)
        self.draw_idle()

def clear(self, msg: str = None):
        self.ax.cla()
        self._sel = None
        self.ax.set_title(self._title)
        self.ax.grid(True, alpha=0.35)
        for s in self.ax.spines.values():
            s.set_linewidth(2.0)
        if msg:
            self.ax.text(0.5, 0.5, msg, ha="center", va="center", transform=self.ax.transAxes)
        self.fig.tight_layout()
        self.draw_idle()

    # ---------- Interacción ----------
def set_point_changed_callback(self, fn):
       """fn(x, y) será llamado al hacer click o arrastrar el punto."""
       self._on_point_changed = fn

def _connect_events(self):
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

def _ensure_sel(self, x, y):
        if self._sel is None:
            self._sel = self.ax.scatter([x], [y], color="black", s=90, zorder=10)
        else:
            self._sel.set_offsets([[x, y]])

def _on_press(self, e):
        if e.inaxes != self.ax or e.xdata is None or e.ydata is None:
            return
        self._dragging = True
        x, y = float(e.xdata), float(e.ydata)
        self._ensure_sel(x, y)
        self.draw_idle()
        if callable(self._on_point_changed):
            self._on_point_changed(x, y)

def _on_motion(self, e):
        if not self._dragging:
            return
        if e.inaxes != self.ax or e.xdata is None or e.ydata is None:
            return
        x, y = float(e.xdata), float(e.ydata)
        self._ensure_sel(x, y)
        self.draw_idle()
        if callable(self._on_point_changed):
            self._on_point_changed(x, y)

def _on_release(self, e):
        self._dragging = False

    # ---------- Plots ----------
def plot_1d(self, x_curve, y_curve, x_path=None, y_path=None,
                func_color="tab:blue", path_color="tab:orange",
                points_color="tab:red", last_color="black"):
        self.ax.cla()
        self._sel = None
        self.ax.plot(x_curve, y_curve, color=func_color, linewidth=2.5, label="f(x)")
        if x_path is not None and y_path is not None and len(x_path) > 0:
            self.ax.plot(x_path, y_path, color=path_color, linewidth=2.5, label="Recorrido")
            self.ax.scatter(x_path, y_path, color=points_color, s=45, label="Puntos")
            self.ax.scatter([x_path[-1]], [y_path[-1]], color=last_color, s=90, zorder=6, label="Último")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.legend(loc="best")
        self._style_axes()
        self.draw_idle()

def plot_contours(self, X, Y, Z, path_x=None, path_y=None, gZ=None,
                      contour_levels=18, contour_color="0.25",
                      g0_color="black", g0_ls="--",
                      path_color="tab:orange", points_color="tab:red",
                      last_color="black"):
        self.ax.cla()
        self._sel = None

        finite = np.isfinite(Z)
        if np.any(finite):
            zmin = np.nanmin(Z)
            zmax = np.nanmax(Z)
            if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
                levels = np.linspace(zmin, zmax, contour_levels)
                self.ax.contour(X, Y, Z, levels=levels, colors=contour_color, linewidths=1.2)

        if gZ is not None:
            try:
                self.ax.contour(X, Y, gZ, levels=[0.0], colors=g0_color, linestyles=g0_ls, linewidths=2.0)
            except Exception:
                pass

        if path_x is not None and path_y is not None and len(path_x) > 0:
            self.ax.plot(path_x, path_y, color=path_color, linewidth=2.5, label="Recorrido")
            self.ax.scatter(path_x, path_y, color=points_color, s=45, label="Puntos")
            self.ax.scatter([path_x[-1]], [path_y[-1]], color=last_color, s=90, zorder=6, label="Último")

        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.ax.legend(loc="best")
        self._style_axes()
        self.draw_idle()
