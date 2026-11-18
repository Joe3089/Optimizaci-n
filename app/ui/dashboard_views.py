# app/ui/dashboard_views.py
# Utilidades del dashboard: cálculo en segundo plano (QThread), lectura de los
# puntos reales del historial del motor, columnas legibles de la tabla y las
# vistas 3D de los métodos del catálogo (solo cuando tienen significado
# matemático; si no, se devuelve el motivo para informarlo al usuario).
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PyQt5 import QtCore, QtWidgets
except ImportError:  # pragma: no cover
    from PyQt6 import QtCore, QtWidgets  # type: ignore


# ── Cálculo sin congelar la interfaz ─────────────────────────────────────────
class _Job(QtCore.QThread):
    def __init__(self, fn, args, kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs
        self.result = None
        self.error: Optional[BaseException] = None

    def run(self):
        try:
            self.result = self._fn(*self._args, **self._kwargs)
        except BaseException as ex:  # se relanza en el hilo de la interfaz
            self.error = ex


def run_in_thread(fn: Callable, *args, **kwargs):
    """Ejecuta fn en un QThread y espera con un bucle de eventos local: la
    ventana sigue repintándose y respondiendo mientras el motor calcula. Para
    quien llama es síncrono (devuelve el resultado o relanza la excepción)."""
    if QtWidgets.QApplication.instance() is None:
        return fn(*args, **kwargs)
    job = _Job(fn, args, kwargs)
    loop = QtCore.QEventLoop()
    job.finished.connect(loop.quit)
    job.start()
    if not job.isFinished():
        loop.exec_() if hasattr(loop, "exec_") else loop.exec()
    job.wait()
    if job.error is not None:
        raise job.error
    return job.result


# ── Puntos reales del historial ──────────────────────────────────────────────
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan", re.I)
POINT_KEYS = ("x_k", "x", "lambda_k", "x_opt", "x_k_str")


def parse_point(v) -> Optional[np.ndarray]:
    """Convierte el punto de una fila del historial (número, arreglo, lista o
    el texto «[1.5 0.5]» que guardan algunos métodos) en un vector float."""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.astype(float).ravel() if v.size else None
    if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
        return np.asarray([float(v)])
    if isinstance(v, (list, tuple)):
        try:
            arr = np.asarray(v, dtype=float).ravel()
            return arr if arr.size else None
        except (TypeError, ValueError):
            return None
    if isinstance(v, str):
        nums = _NUM.findall(v)
        if not nums:
            return None
        try:
            return np.asarray([float(n) for n in nums])
        except ValueError:
            return None
    return None


def row_point(row: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in POINT_KEYS:
        p = parse_point(row.get(key))
        if p is not None and p.size:
            return p
    return None


def row_interval(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Intervalo de búsqueda [a, b] de la iteración, si el método lo reporta."""
    for ka, kb in (("a", "b"), ("a_k", "b_k")):
        a, b = row.get(ka), row.get(kb)
        if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
            lo, hi = float(min(a, b)), float(max(a, b))
            if np.isfinite(lo) and np.isfinite(hi):
                return lo, hi
    return None


# ── Tabla: encabezados legibles sin columnas repetidas ───────────────────────
_LABELS = {
    "k": "Iteración k", "iter": "Iteración", "gen": "Generación",
    "lambda_k": "λₖ (punto)", "x": "xₖ", "x_k": "xₖ", "x_k_str": "xₖ", "x_opt": "x óptimo (iter.)",
    "f_lambda": "f(λₖ)", "f_k": "f(xₖ)", "f(x)": "f(x)", "mu_k": "μₖ (punto)", "f_mu": "f(μₖ)",
    "a": "aₖ (lím. inf.)", "b": "bₖ (lím. sup.)", "a_k": "aₖ (lím. inf.)", "b_k": "bₖ (lím. sup.)",
    "c_k": "cₖ", "d_k": "dₖ", "b-a": "Ancho bₖ − aₖ", "L": "Ancho L", "step": "Paso",
    "grad_norm": "‖∇f(xₖ)‖", "alpha_k": "Tamaño de paso αₖ", "d_k_str": "Dirección dₖ",
    "direccion": "Tipo de dirección", "beta_k": "βₖ (conjugación)", "hess_det": "det(Hessiano)",
    "cond1_ok": "Cond. Goldstein 1", "cond2_ok": "Cond. Goldstein 2", "converged": "¿Convergió?",
    "f_best": "Mejor f", "f_media": "f media", "T": "Temperatura T", "delta_E": "ΔE",
    "prob_acept": "P(aceptar)", "aceptado": "¿Aceptado?", "diversidad": "Diversidad",
    "vel_media": "Velocidad media", "sigma": "σ mutación", "mu": "μ (penalización)",
    "t": "t (barrera)", "w": "w (peso)",
}
# Columnas que el motor repite con otro nombre (mismo dato): se muestra la primera.
_ALIASES = [("k", "iter"), ("lambda_k", "x"), ("f_lambda", "f_k", "f(x)"), ("x_k", "x_k_str")]


def table_layout(hist: Sequence[Dict[str, Any]], constrained: bool = False,
                 readable: bool = True) -> Tuple[List[str], List[str]]:
    """Devuelve (claves, encabezados) de la tabla de iteraciones. Una columna
    alias se omite solo si sus valores coinciden fila a fila con la anterior."""
    keys: List[str] = []
    for r in hist:
        for k in r:
            if k not in keys:
                keys.append(k)
    if "x_k_str" in keys and "x_k" in keys:
        keys.remove("x_k")
    for group in _ALIASES:
        present = [k for k in group if k in keys]
        for extra in present[1:]:
            if all(_same(r.get(present[0]), r.get(extra)) for r in hist):
                keys.remove(extra)
    heads = []
    for k in keys:
        h = _LABELS.get(k, k) if readable else k
        if constrained and k == "f_k":
            h = "Φ(xₖ) del subproblema"
        heads.append(h)
    return keys, heads


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                                rtol=0, atol=0, equal_nan=True))
    except (TypeError, ValueError):
        return str(a) == str(b)


def cell_text(v) -> str:
    if isinstance(v, np.ndarray):
        return "[" + ", ".join(f"{x:.5g}" for x in v.flat) + "]"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


# ── Vistas 3D de los métodos del catálogo ────────────────────────────────────
_NO3D: Dict[str, str] = {}
_NO3D_GROUP = {
    "Programación Dinámica": "la recursión por etapas se lee en la tabla y en la gráfica 2D; no hay una superficie con significado.",
    "Reemplazo y Mantenimiento": "las decisiones por periodo son una secuencia (2D), no una superficie.",
    "Inventario": "cada producto se analiza por separado; la tabla y la gráfica 2D muestran su EOQ y su clase ABC.",
    "Optimización Combinatoria": "las soluciones son combinaciones discretas (rutas, conjuntos, particiones), sin una superficie continua.",
    "Optimización Cuántica": "los resultados son probabilidades o energías por iteración (2D).",
}
_LP_LABELS = ("PL: Método Gráfico", "PL: Simplex", "PL: Simplex No Acotado", "PL: Degenerado",
              "PL: Big M", "PL: Dos Fases (variables artificiales)", "PL: Método de Punto Interior",
              "PL: Simplex Acotado (cotas en variables)")


def _style3d(ax, title):
    ax.set_facecolor("#0a1228")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#1a2e58")
    ax.tick_params(colors="#7aaaea", labelsize=7)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.label.set_color("#7aaaea")
    ax.set_title(title, fontsize=10, color="#d8e8f8")


def problem_plot3d(label: str, group: str, values: Dict[str, str], res: Dict[str, Any]
                   ) -> Tuple[Optional[Callable], str]:
    """(dibujar(fig), explicación de los ejes) o (None, motivo de no tener 3D)."""
    try:
        if label in _LP_LABELS:
            return _lp3d(label, values, res)
        if label.startswith("Transporte:"):
            return _transport3d(res)
        if label == "Asignación: Método Húngaro":
            return _assign3d(values, res)
        if label == "Cuántica: Bit y Qubit":
            return _bloch3d(res)
        if label in _STATS3D:
            return _STATS3D[label](values, res)
    except Exception as ex:  # datos que no permiten la vista: se informa, no se inventa
        return None, f"no se pudo construir la vista 3D con estos datos ({ex})."
    if label in _NO3D:
        return None, _NO3D[label]
    if group in _NO3D_GROUP:
        return None, _NO3D_GROUP[group]
    return None, "sus resultados no forman una superficie con significado; consulta la tabla y la gráfica 2D."


def problem_plot2d(label: str, values: Dict[str, str], res: Dict[str, Any]) -> Optional[Callable]:
    """Vista 2D para métodos del catálogo que no traen la suya (Simplex y variantes)."""
    if res.get("plot") is not None or label not in _LP_LABELS:
        return res.get("plot")
    from app.application import problem_catalog as cat
    from app.operations_research import linear_programming as lp
    c, A, signs, b, _sense = cat._lp_read(values)
    x_opt = res.get("x")
    optimal = res.get("status") == lp.OPTIMAL and x_opt is not None
    if len(c) != 2:
        if not optimal:
            return None

        def bars(fig):
            ax = fig.add_subplot(111)
            xs = np.asarray(x_opt, float)
            ax.bar([f"x{i + 1}" for i in range(xs.size)], xs, color="#4a9eff")
            ax.set_ylabel("Valor óptimo")
            ax.set_title(f"Solución óptima ({xs.size} variables), Z = {float(res['z']):.6g}")
        return bars
    A_, b_, c_ = np.asarray(A, float), np.asarray(b, float), np.asarray(c, float)
    pts = [p for p in lp._intersections(np.vstack([A_, np.eye(2)]), np.concatenate([b_, [0, 0]]))
           if lp._is_feasible(p, A_, signs, b_)]
    span = 1.4 * max([1.0] + [float(v) for p in pts for v in p]
                     + ([float(np.max(x_opt))] if optimal else []))

    def draw(fig):
        ax = fig.add_subplot(111)
        grid = np.array([0.0, span])
        for i, (row, rhs) in enumerate(zip(A_, b_)):
            if abs(row[1]) > 1e-12:
                ax.plot(grid, (rhs - row[0] * grid) / row[1], lw=1, label=f"R{i + 1}")
            elif abs(row[0]) > 1e-12:
                ax.axvline(rhs / row[0], lw=1, label=f"R{i + 1}")
        poly = lp._polygon(A_, signs, b_, span, span)
        if len(poly) >= 3:
            ax.fill(*zip(*poly), color="#4a9eff", alpha=0.3, label="Región factible")
        if pts:
            V = np.asarray(pts, float)
            ax.plot(V[:, 0], V[:, 1], "o", color="#c8d8f0", ms=4, label="Vértices factibles")
        if optimal:
            xo = np.asarray(x_opt, float)
            ax.plot(xo[0], xo[1], "D", color="#ffd700", ms=9,
                    label=f"Óptimo del Simplex, Z = {float(res['z']):.6g}")
            if abs(c_[1]) > 1e-12:
                ax.plot(grid, (float(res["z"]) - c_[0] * grid) / c_[1], "--", color="#ffd700", lw=1)
        else:
            ax.set_title(f"Estado: {res.get('status')}")
        ax.set_xlim(0, span); ax.set_ylim(0, span)
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        if optimal:
            ax.set_title("Región factible y óptimo encontrado por el Simplex")
        ax.legend(fontsize=7)
    return draw


def _lp3d(label, values, res):
    from app.application import problem_catalog as cat
    from app.operations_research import linear_programming as lp
    c, A, signs, b, _sense = cat._lp_read(values)
    if len(c) != 2:
        return None, f"el problema tiene {len(c)} variables; la región factible solo se puede dibujar con 2."
    A = [list(r) for r in A]
    signs, b = list(signs), list(b)
    if label.startswith("PL: Simplex Acotado") and values.get("cotas"):
        lb, ub = cat.bounds(values["cotas"], 2)
        for i in range(2):
            row = [1.0 if j == i else 0.0 for j in range(2)]
            if lb[i] > 0:
                A.append(row); signs.append(">="); b.append(lb[i])
            if np.isfinite(ub[i]):
                A.append(row); signs.append("<="); b.append(ub[i])
    A_, b_, c_ = np.asarray(A, float), np.asarray(b, float), np.asarray(c, float)
    x_opt = res.get("x")
    pts = [p for p in lp._intersections(np.vstack([A_, np.eye(2)]), np.concatenate([b_, [0, 0]]))
           if lp._is_feasible(p, A_, signs, b_)]
    span = max([1.0] + [float(v) for p in pts for v in p])
    if x_opt is not None and res.get("status") == lp.OPTIMAL:
        span = max(span, float(np.max(x_opt)))
    span *= 1.4
    poly = lp._polygon(A_, signs, b_, span, span)
    if len(poly) < 3:
        why = ("hay restricciones de igualdad: la región factible es un segmento o un punto, "
               "no un área sobre la cual elevar el plano Z (ver la gráfica 2D)."
               if "=" in signs else "la región factible está vacía; no hay superficie que dibujar.")
        return None, why
    path = res.get("path") if label == "PL: Método de Punto Interior" else None
    unbounded = res.get("status") not in (None, lp.OPTIMAL)

    def draw(fig):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax = fig.add_subplot(111, projection="3d")
        P = np.asarray(poly, float)
        Z = P @ c_
        ax.add_collection3d(Poly3DCollection([list(zip(P[:, 0], P[:, 1], Z))], alpha=0.45,
                                             facecolor="#4a9eff", edgecolor="#9cc8ff"))
        ax.add_collection3d(Poly3DCollection([list(zip(P[:, 0], P[:, 1], np.zeros(len(P))))],
                                             alpha=0.15, facecolor="#3cff8a", edgecolor="#3cff8a"))
        V = np.asarray(pts, float) if pts else np.zeros((0, 2))
        if V.size:
            ax.scatter(V[:, 0], V[:, 1], V @ c_, color="#c8d8f0", s=18, label="Vértices factibles")
        if path:
            Pp = np.asarray(path, float)
            ax.plot(Pp[:, 0], Pp[:, 1], Pp @ c_, "o-", color="#ff6b35", label="Trayectoria del punto interior")
        if not unbounded and x_opt is not None:
            xo = np.asarray(x_opt, float)
            ax.scatter([xo[0]], [xo[1]], [float(xo @ c_)], color="#ffd700", s=70, marker="D",
                       label=f"Óptimo Z = {float(xo @ c_):.4g}")
        ax.set_xlim(0, span); ax.set_ylim(0, span)
        zs = np.concatenate([Z, [0.0]])
        ax.set_zlim(float(zs.min()), float(zs.max()) if zs.max() > zs.min() else float(zs.min()) + 1)
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("Z = c·x")
        _style3d(ax, "3D — Z sobre la región factible" + (" (recortada: Z no acotada)" if unbounded else ""))
        ax.legend(fontsize=7, loc="upper left")

    return draw, ("Ejes: x1, x2 (variables de decisión) y Z = c·x. La región factible (verde, z = 0) "
                  "se eleva al plano del objetivo (azul); los vértices y el óptimo son los calculados.")


def _transport3d(res):
    M = np.asarray(res["allocation"], float)

    def draw(fig):
        ax = fig.add_subplot(111, projection="3d")
        ii, jj = np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[1]), indexing="ij")
        h = M.ravel()
        colors = ["#ffd700" if v > 0 else "#2a3e6e" for v in h]
        ax.bar3d(ii.ravel() - 0.3, jj.ravel() - 0.3, np.zeros_like(h), 0.6, 0.6, h,
                 color=colors, alpha=0.85, shade=True)
        ax.set_xticks(range(M.shape[0])); ax.set_xticklabels([f"O{i + 1}" for i in range(M.shape[0])])
        ax.set_yticks(range(M.shape[1])); ax.set_yticklabels([f"D{j + 1}" for j in range(M.shape[1])])
        ax.set_xlabel("Origen"); ax.set_ylabel("Destino"); ax.set_zlabel("Unidades enviadas")
        _style3d(ax, f"3D — Asignación de envíos (costo total {res.get('total', 0):.6g})")

    return draw, ("Ejes: origen, destino y unidades enviadas x_ij de la solución calculada "
                  "(incluye origen/destino ficticio si hubo que balancear).")


def _assign3d(values, res):
    from app.application import problem_catalog as cat
    C = np.asarray(cat.matrix(values["costos"]), float)
    chosen = set(tuple(p) for p in res["pairs"])

    def draw(fig):
        ax = fig.add_subplot(111, projection="3d")
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                ax.bar3d(i - 0.3, j - 0.3, 0, 0.6, 0.6, C[i, j],
                         color="#ffd700" if (i, j) in chosen else "#2a4ea0", alpha=0.85)
        ax.set_xticks(range(C.shape[0])); ax.set_xticklabels([f"A{i + 1}" for i in range(C.shape[0])])
        ax.set_yticks(range(C.shape[1])); ax.set_yticklabels([f"T{j + 1}" for j in range(C.shape[1])])
        ax.set_xlabel("Agente"); ax.set_ylabel("Tarea"); ax.set_zlabel("Costo")
        _style3d(ax, f"3D — Matriz de costos (dorado = asignación, total {res.get('total', 0):.6g})")

    return draw, "Ejes: agente, tarea y costo c_ij ingresado; en dorado, los pares asignados por el método húngaro."


def _regression3d(values, res):
    from app.application import problem_catalog as cat
    M = np.asarray(cat.matrix(values["datos"]), float)
    if M.shape[1] != 3:
        n = M.shape[1] - 1
        return None, (f"la regresión tiene {n} variable(s) explicativa(s); el plano ajustado solo se "
                      "puede dibujar en 3D con exactamente 2 (con 1, la recta 2D ya es la vista completa).")
    beta = np.asarray(res["beta"], float)

    def draw(fig):
        ax = fig.add_subplot(111, projection="3d")
        g1 = np.linspace(M[:, 0].min(), M[:, 0].max(), 20)
        g2 = np.linspace(M[:, 1].min(), M[:, 1].max(), 20)
        G1, G2 = np.meshgrid(g1, g2)
        ax.plot_surface(G1, G2, beta[0] + beta[1] * G1 + beta[2] * G2, alpha=0.45, cmap="coolwarm")
        ax.scatter(M[:, 0], M[:, 1], M[:, 2], color="#ffd700", s=25, label="Observaciones")
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
        _style3d(ax, "3D — Plano de mínimos cuadrados")
        ax.legend(fontsize=7, loc="upper left")

    return draw, "Ejes: x1, x2 (datos) e y; el plano usa los coeficientes β estimados."


def _bloch3d(res):
    v = np.asarray(res["bloch"], float)

    def draw(fig):
        ax = fig.add_subplot(111, projection="3d")
        u, w = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        ax.plot_wireframe(np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w),
                          color="#2a4ea0", linewidth=0.4, alpha=0.6)
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color="#ffd700", linewidth=2.5)
        ax.scatter([v[0]], [v[1]], [v[2]], color="#ffd700", s=40, label="Estado |ψ⟩")
        ax.text(0, 0, 1.15, "|0⟩", color="#c8d8f0"); ax.text(0, 0, -1.25, "|1⟩", color="#c8d8f0")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        _style3d(ax, "3D — Esfera de Bloch")
        ax.legend(fontsize=7, loc="upper left")

    return draw, "Ejes: componentes (x, y, z) del vector de Bloch calculado a partir de θ y φ."


# ── Estadística: vistas 3D con variables reales del problema ────────────────
def _data(values, key="datos"):
    from app.application import problem_catalog as cat
    return np.asarray(cat.floats(values[key]), float)


def _descriptive3d(values, res):
    x = _data(values)
    n, s = x.size, float(x.std(ddof=1))
    h0 = 1.06 * s * n ** (-1 / 5) if s > 0 else 1.0     # regla de Silverman
    hs = np.linspace(0.4 * h0, 2.5 * h0, 30)
    pad = 3 * h0
    xs = np.linspace(x.min() - pad, x.max() + pad, 120)
    H, X = np.meshgrid(hs, xs, indexing="ij")
    D = (np.exp(-0.5 * ((X[..., None] - x) / H[..., None]) ** 2).sum(-1)
         / (n * H * np.sqrt(2 * np.pi)))

    def draw(fig):
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, H, D, cmap="viridis", alpha=0.8, linewidth=0)
        d0 = np.exp(-0.5 * ((xs[:, None] - x) / h0) ** 2).sum(-1) / (n * h0 * np.sqrt(2 * np.pi))
        ax.plot(xs, np.full_like(xs, h0), d0, color="#ffd700", lw=2.5,
                label=f"h de Silverman = {h0:.4g}")
        ax.scatter(x, np.full(n, hs[0]), np.zeros(n), color="#ff6b35", s=12, label="Datos")
        ax.set_xlabel("x (valor)"); ax.set_ylabel("h (ancho de banda)"); ax.set_zlabel("densidad")
        _style3d(ax, "3D — Densidad estimada f̂ₕ(x) de la muestra")
        ax.legend(fontsize=7, loc="upper left")

    return draw, ("Ejes: valor x, ancho de banda h y densidad estimada por núcleo gaussiano "
                  "f̂ₕ(x) = Σ φ((x − xᵢ)/h)/(n·h) calculada con tus datos; la curva dorada usa "
                  "la h de Silverman.")


def _ci3d(values, res):
    from scipy import stats as sps
    x = _data(values)
    n, s = x.size, float(x.std(ddof=1))
    level = float(values.get("nivel", "0.95"))
    ns = np.arange(2, max(4 * n, 30) + 1)
    lv = np.linspace(0.80, 0.995, 30)
    N, L = np.meshgrid(ns, lv, indexing="ij")
    E = sps.t.ppf((1 + L) / 2, N - 1) * s / np.sqrt(N)

    def draw(fig):
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(N, L, E, cmap="coolwarm", alpha=0.75, linewidth=0)
        e0 = float(res["high"] - res["mean"])
        ax.scatter([n], [level], [e0], color="#ffd700", s=70, marker="D",
                   label=f"Tu muestra: n = {n}, {level:.0%} → E = {e0:.4g}")
        ax.set_xlabel("n (tamaño de muestra)"); ax.set_ylabel("nivel de confianza")
        ax.set_zlabel("margen de error E")
        _style3d(ax, "3D — Margen de error E = t·s/√n")
        ax.legend(fontsize=7, loc="upper left")

    return draw, (f"Ejes: n, nivel de confianza y margen de error E = t((1+nivel)/2, n−1)·s/√n "
                  f"con la s = {s:.4g} de tus datos; el punto es tu intervalo calculado.")


def _ttest3d(values, res):
    from scipy import stats as sps
    x = _data(values)
    df = x.size - 1
    t_obs = float(res["t"])
    lim = max(4.0, abs(t_obs) + 1)
    ts = np.linspace(-lim, lim, 120)
    dfs = np.arange(1, max(2 * df, 30) + 1)
    T, G = np.meshgrid(ts, dfs, indexing="ij")
    P = sps.t.pdf(T, G)

    def draw(fig):
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(T, G, P, cmap="viridis", alpha=0.6, linewidth=0)
        ax.plot(ts, np.full_like(ts, df), sps.t.pdf(ts, df), color="#4a9eff", lw=2.5,
                label=f"t con {df} g.l. (tu prueba)")
        ax.scatter([t_obs], [df], [float(sps.t.pdf(t_obs, df))], color="#ffd700", s=70,
                   marker="D", label=f"t observado = {t_obs:.4g}, p = {float(res['p']):.4g}")
        ax.set_xlabel("t"); ax.set_ylabel("grados de libertad"); ax.set_zlabel("densidad")
        _style3d(ax, "3D — Distribución t según los grados de libertad")
        ax.legend(fontsize=7, loc="upper left")

    return draw, ("Ejes: estadístico t, grados de libertad y densidad de la distribución t; la "
                  "curva azul es la de tu prueba y el punto, el t calculado con tus datos.")


def _regression_any3d(values, res):
    from app.application import problem_catalog as cat
    M = np.asarray(cat.matrix(values["datos"]), float)
    if M.shape[1] == 3:
        return _regression3d(values, res)
    if M.shape[1] != 2:
        return None, (f"con {M.shape[1] - 1} variables explicativas no hay una superficie "
                      "3D directa; consulta ŷ frente a y en la gráfica 2D.")
    x, y = M[:, 0], M[:, 1]
    b0, b1 = (float(v) for v in res["beta"])
    r0 = 3 * max(abs(b0), float(np.std(y)), 1e-3)
    r1 = 3 * max(abs(b1), 1e-3)
    B0, B1 = np.meshgrid(np.linspace(b0 - r0, b0 + r0, 60), np.linspace(b1 - r1, b1 + r1, 60))
    S = ((y[None, None, :] - B0[..., None] - B1[..., None] * x[None, None, :]) ** 2).sum(-1)
    sse = float(((y - b0 - b1 * x) ** 2).sum())

    def draw(fig):
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(B0, B1, S, cmap="coolwarm", alpha=0.75, linewidth=0)
        ax.scatter([b0], [b1], [sse], color="#ffd700", s=80, marker="D",
                   label=f"Mínimo: β₀ = {b0:.4g}, β₁ = {b1:.4g}, SSE = {sse:.4g}")
        ax.set_xlabel("β₀"); ax.set_ylabel("β₁"); ax.set_zlabel("SSE")
        _style3d(ax, "3D — Suma de errores al cuadrado SSE(β₀, β₁)")
        ax.legend(fontsize=7, loc="upper left")

    return draw, ("Ejes: intercepto β₀, pendiente β₁ y SSE = Σ(yᵢ − β₀ − β₁xᵢ)² con tus datos; "
                  "mínimos cuadrados elige el punto más bajo de esta superficie.")


def _montecarlo3d(values, res):
    sv = res.get("sample_vars") or {}
    y = np.asarray(res.get("sample_y", []), float)
    names = list(sv)
    if len(names) >= 2:
        a, b = np.asarray(sv[names[0]]), np.asarray(sv[names[1]])

        def draw(fig):
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(a, b, y, c=y, cmap="viridis", s=4, alpha=0.6)
            fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08)
            ax.set_xlabel(names[0]); ax.set_ylabel(names[1]); ax.set_zlabel("valor simulado")
            _style3d(ax, f"3D — {y.size} simulaciones reales de la expresión")

        return draw, (f"Ejes: valores simulados de {names[0]} y {names[1]} y el valor de la "
                      f"expresión en cada simulación (primeras {y.size} del cálculo).")
    if y.size < 200:
        return None, "hay muy pocas simulaciones para una vista 3D por tramos."
    chunks = np.array_split(y, 10)
    edges = np.histogram_bin_edges(y, bins=20)

    def draw1(fig):
        ax = fig.add_subplot(111, projection="3d")
        w = edges[1] - edges[0] if edges.size > 1 else 1.0
        for i, c in enumerate(chunks):
            h, _ = np.histogram(c, bins=edges)
            ax.bar(edges[:-1], h / c.size, zs=i, zdir="y", width=w, alpha=0.75, color="#4a9eff")
        ax.set_xlabel("valor simulado"); ax.set_ylabel("tramo de simulaciones")
        ax.set_zlabel("frecuencia relativa")
        _style3d(ax, "3D — Distribución por tramos de simulaciones")

    return draw1, ("Ejes: valor simulado, tramo (10 bloques consecutivos de las primeras "
                   f"{y.size} simulaciones) y frecuencia relativa: muestra si la distribución "
                   "se estabiliza.")


def _chebyshev3d(values, res):
    mu, sigma = float(res["mu"]), float(res["sigma"])
    k = float(res["k"])
    d = np.linspace(1.05 * sigma, max(4.0, k + 1.5) * sigma, 50)
    A, B = np.meshgrid(mu - d, mu + d)
    Dm = np.minimum(mu - A, B - mu)
    P = 100 * (1 - sigma ** 2 / Dm ** 2)

    def draw(fig):
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(A, B, P, cmap="viridis", alpha=0.75, linewidth=0)
        ax.scatter([float(res["low"])], [float(res["high"])], [float(res["percent"])],
                   color="#ffd700", s=80, marker="D",
                   label=f"[{res['low']:.4g}, {res['high']:.4g}] → ≥ {res['percent']:.4g} %")
        ax.set_xlabel("límite inferior a"); ax.set_ylabel("límite superior b")
        ax.set_zlabel("% mínimo garantizado")
        _style3d(ax, "3D — Cota de Chebyshev según el intervalo (a, b)")
        ax.legend(fontsize=7, loc="upper left")

    return draw, (f"Ejes: límites a < μ < b y % mínimo = 100·(1 − σ²/d²), con d = mín(μ − a, "
                  f"b − μ), μ = {mu:.4g} y σ = {sigma:.4g}. Es una cota mínima para cualquier "
                  "distribución, no una probabilidad exacta.")


_STATS3D = {
    "Estadística Descriptiva": _descriptive3d,
    "Intervalo de Confianza": _ci3d,
    "Prueba t de Hipótesis": _ttest3d,
    "Regresión Lineal": _regression_any3d,
    "Simulación de Monte Carlo": _montecarlo3d,
    "Desigualdad de Chebyshev": _chebyshev3d,
}


# ── Leyendas: mover (arrastrar) y contraer/expandir (doble clic) ────────────
LEGEND_HINT = ("Leyendas: arrástralas para moverlas; doble clic sobre una leyenda para "
               "contraerla o expandirla.")


def _legends(fig):
    out = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    return out + list(getattr(fig, "legends", []))


def _toggle_legend(fig, leg):
    """Contrae la leyenda a solo su título («▸ Leyenda») o la expande de nuevo,
    en el mismo lugar donde el usuario la dejó."""
    parent = leg.axes if leg.axes is not None and leg in [a.get_legend() for a in fig.axes] \
        else fig
    renderer = fig.canvas.get_renderer()
    bb = leg.get_window_extent(renderer)
    x0, y1 = fig.transFigure.inverted().transform((bb.x0, bb.y1))
    frame = leg.get_frame()
    texts = leg.get_texts()
    style = dict(loc="upper left", bbox_to_anchor=(x0, y1), bbox_transform=fig.transFigure,
                 facecolor=frame.get_facecolor(), edgecolor=frame.get_edgecolor(),
                 framealpha=frame.get_alpha() if frame.get_alpha() is not None else 0.88,
                 fontsize=texts[0].get_fontsize() if texts else 8,
                 labelcolor=texts[0].get_color() if texts else "#d8e8f8")
    stored = getattr(leg, "_opti_entries", None)
    if parent is fig:
        leg.remove()
    if stored is None:                       # contraer
        entries = (list(leg.legend_handles), [t.get_text() for t in texts])
        new = parent.legend([], [], title="▸ Leyenda", **style)
        new._opti_entries = entries
    else:                                    # expandir
        new = parent.legend(stored[0], stored[1], **style)
    if new.get_title() is not None:
        new.get_title().set_color(style["labelcolor"])
        new.get_title().set_fontsize(style["fontsize"])
    try:
        new.set_draggable(True)
    except Exception:
        pass
    return new


def install_legend_tools(fig, canvas=None) -> int:
    """Hace arrastrables todas las leyendas de la figura y conecta (una sola vez
    por figura) el doble clic que las contrae/expande. Devuelve cuántas hay."""
    legs = _legends(fig)
    for leg in legs:
        try:
            leg.set_draggable(True)
        except Exception:
            pass
    if not getattr(fig, "_opti_legend_cid", None):
        def on_press(ev):
            if not ev.dblclick:
                return
            for leg in _legends(fig):
                try:
                    if leg.get_visible() and leg.contains(ev)[0]:
                        ev._opti_legend_handled = True   # que nadie más use este clic
                        _toggle_legend(fig, leg)
                        fig.canvas.draw_idle()
                        return
                except Exception:
                    continue
        fig._opti_legend_cid = fig.canvas.mpl_connect("button_press_event", on_press)
    return len(legs)
