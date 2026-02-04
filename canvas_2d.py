from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class Function2DCanvas(FigureCanvas):
    """Canvas 2D con:
    - Estilo cartesiano (grilla + ejes gruesos)
    - Curva 1D f(x) + recorrido (línea y puntos con colores distintos)
    - Contornos MD f(x1,x2) + recorrido + g(x)=0 (opcional)
    - Click + drag para seleccionar un punto (dispara callback(x,y))
    """

    def __init__(self, parent=None, title="Gráfica 2D"):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self._title = title

        self._on_point_changed = None  # callback(x, y)
        self._selected_artist = None
        self._dragging = False

        self._apply_style()
        self._connect_events()

    # ---------- estilo ----------
    def _apply_style(self):
        self.ax.set_title(self._title)
        self.ax.grid(True, alpha=0.35)
        for s in self.ax.spines.values():
            s.set_linewidth(2.0)
        self.fig.tight_layout()

    def set_title(self, title: str):
        self._title = title
        self.ax.set_title(self._title)
        self.draw_idle()

    def _finalize(self, legend=True):
        self.ax.set_title(self._title)
        self.ax.grid(True, alpha=0.35)
        for s in self.ax.spines.values():
            s.set_linewidth(2.0)
        if legend:
            self.ax.legend(loc="best")
        self.fig.tight_layout()
        self.draw_idle()

    # ---------- interacción ----------
    def set_point_changed_callback(self, fn):
        self._on_point_changed = fn

    def _connect_events(self):
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def _ensure_selected(self, x, y):
        if self._selected_artist is None:
            self._selected_artist = self.ax.scatter([x], [y], color="black", s=95, zorder=6, label="_sel")
        else:
            self._selected_artist.set_offsets([[x, y]])

    def _near_selected(self, x, y):
        if self._selected_artist is None:
            return False
        pts = self._selected_artist.get_offsets()
        if len(pts) != 1:
            return False
        sx, sy = float(pts[0, 0]), float(pts[0, 1])
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        tolx = 0.02 * (xlim[1] - xlim[0] + 1e-12)
        toly = 0.02 * (ylim[1] - ylim[0] + 1e-12)
        return abs(x - sx) <= tolx and abs(y - sy) <= toly

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if self._near_selected(x, y):
            self._dragging = True
            return

        self._ensure_selected(x, y)
        self._dragging = True
        self.draw_idle()
        if callable(self._on_point_changed):
            self._on_point_changed(x, y)

    def _on_motion(self, event):
        if not self._dragging:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        self._ensure_selected(x, y)
        self.draw_idle()
        if callable(self._on_point_changed):
            self._on_point_changed(x, y)

    def _on_release(self, event):
        self._dragging = False

    # ---------- plots ----------
    def plot_1d_function(self, x_curve, y_curve, x_path, y_path,
                         func_color="tab:blue",
                         path_color="tab:orange",
                         points_color="tab:red",
                         last_color="black"):
        self.ax.cla()
        self._selected_artist = None

        self.ax.plot(x_curve, y_curve, color=func_color, linewidth=2.5, label="f(x)")

        if len(x_path) > 0:
            self.ax.plot(x_path, y_path, color=path_color, linewidth=2.5, label="Recorrido")
            self.ax.scatter(x_path, y_path, color=points_color, s=45, label="Puntos")
            self.ax.scatter([x_path[-1]], [y_path[-1]], color=last_color, s=90, zorder=5, label="Último")

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self._finalize()

    def plot_contour_with_path(self, X, Y, Z, path_x, path_y,
                               gZ=None,
                               contour_levels=18,
                               contour_color="0.25",
                               g0_color="black",
                               g0_ls="--",
                               path_color="tab:orange",
                               points_color="tab:red",
                               last_color="black"):
        self.ax.cla()
        self._selected_artist = None

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

        if len(path_x) > 0:
            self.ax.plot(path_x, path_y, color=path_color, linewidth=2.5, label="Recorrido")
            self.ax.scatter(path_x, path_y, color=points_color, s=45, label="Puntos")
            self.ax.scatter([path_x[-1]], [path_y[-1]], color=last_color, s=90, zorder=5, label="Último")

        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self._finalize()

    def plot_convergence(self, it, fx,
                         line_color="tab:orange",
                         points_color="tab:red",
                         last_color="black"):
        self.ax.cla()
        self._selected_artist = None

        self.ax.plot(it, fx, color=line_color, linewidth=2.5, label="f(x_k)")
        self.ax.scatter(it, fx, color=points_color, s=45, label="Puntos")
        if len(it) > 0:
            self.ax.scatter([it[-1]], [fx[-1]], color=last_color, s=90, zorder=5, label="Último")

        self.ax.set_xlabel("Iteración k")
        self.ax.set_ylabel("f(x)")
        self._finalize()
