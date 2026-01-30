from PyQt5.QtCore import QTimer, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d


class Rotating3DCanvas(FigureCanvas):
    point_moved = pyqtSignal(int, float)
    def __init__(self, parent=None, interval_ms=50, title="Gráfica 3D", **kwargs):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111, projection="3d")
        self._angle = 30
        self._title = title

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(interval_ms)
        self._connect_interaction()

    def set_title(self, title: str):
        self._title = title
        try:
            self.ax.set_title(self._title)
            self.draw_idle()
        except Exception:
            pass

    def _tick(self):
        self._angle = (self._angle + 1) % 360
        self.ax.view_init(elev=30, azim=self._angle)
        self.ax.set_title(self._title)
        self.draw_idle()

    def set_data(self, iters, xs, zs, row_indices=None):
        self._iters = list(iters)
        self._xs = list(xs)
        self._zs = list(zs)
        self._row_indices = list(row_indices) if row_indices is not None else list(range(len(self._xs)))
        self.ax.cla()
        self.ax.plot(self._iters, self._xs, self._zs, linewidth=2)
        self._pts = self.ax.scatter(self._iters, self._xs, self._zs, s=32)
        self._sel = self.ax.scatter([], [], [], s=90)
        self.ax.set_xlabel("Iteración")
        self.ax.set_ylabel("x")
        self.ax.set_zlabel("f(x)")
        self.ax.set_title(self._title)
        self.draw_idle()

    def start_rotation(self):
        if not self._timer.isActive():
            self._timer.start()

    def stop_rotation(self):
        if self._timer.isActive():
            self._timer.stop()



    def _nearest_point(self, event):
        # Devuelve índice del punto más cercano en coordenadas de pantalla (pixeles)
        if not self._iters:
            return None
        xs2d, ys2d = [], []
        for itv, xv, zv in zip(self._iters, self._xs, self._zs):
            x2, y2, _ = proj3d.proj_transform(itv, xv, zv, self.ax.get_proj())
            xd, yd = self.ax.transData.transform((x2, y2))
            xs2d.append(xd); ys2d.append(yd)
        ex, ey = event.x, event.y
        best_i, best_d = None, None
        for i, (xd, yd) in enumerate(zip(xs2d, ys2d)):
            d = (xd - ex) ** 2 + (yd - ey) ** 2
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        if best_d is not None and best_d <= (18 ** 2):
            return best_i
        return None

    def _set_selected(self, idx):
        if idx is None or idx < 0 or idx >= len(self._iters):
            self._sel._offsets3d = ([], [], [])
            self.draw_idle()
            return
        self._sel._offsets3d = ([self._iters[idx]], [self._xs[idx]], [self._zs[idx]])
        self.draw_idle()

    def _connect_interaction(self):
        self._dragging = False
        self._drag_idx = None
        self._iters = []
        self._xs = []
        self._zs = []
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        idx = self._nearest_point(event)
        if idx is None:
            return
        self._dragging = True
        self._drag_idx = idx
        self._set_selected(idx)
        self.stop_rotation()

    def _on_motion(self, event):
        if not getattr(self, "_dragging", False):
            return
        if event.inaxes != self.ax:
            return
        if self._drag_idx is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Actualiza x del punto seleccionado según el movimiento vertical del mouse (event.ydata)
        i = self._drag_idx
        new_x = float(event.ydata)
        self._xs[i] = new_x
        try:
            hid = int(self._row_indices[i])
            self.point_moved.emit(hid, new_x)
        except Exception:
            pass
        self.set_data(self._iters, self._xs, self._zs, row_indices=self._row_indices)
        self._set_selected(i)

    def _on_release(self, event):
        if not getattr(self, "_dragging", False):
            return
        self._dragging = False
        self._drag_idx = None
        self.start_rotation()
    def clear(self):
        self.ax.cla()
        self.draw_idle()
