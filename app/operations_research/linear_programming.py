# app/operations_research/linear_programming.py
# Programación lineal: Simplex tabular (holguras, Big M, Dos Fases, variables
# acotadas, detección de degeneración/no acotamiento) y método gráfico.
#
# La Big M se maneja de forma simbólica (dos filas de costo: parte en M y
# parte real, comparadas lexicográficamente) en vez de usar un número enorme,
# que degradaría la precisión del punto flotante.
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

EPS = 1e-9
FEAS_TOL = 1e-7
MAX_TABLEAU_ITERS = 200

OPTIMAL, UNBOUNDED, INFEASIBLE = "Óptimo", "No acotado", "Infactible"


def _fmt_big_m(m: float, r: float) -> str:
    m, r = round(m, 6), round(r, 6)
    if abs(m) < 1e-6:
        return f"{r:g}"
    if abs(r) < 1e-6:
        return f"{m:g}M"
    return f"{m:g}M{r:+g}"


class _Simplex:
    """Estado del tableau en forma min c·x, A·x = b, 0 ≤ x ≤ ub."""

    def __init__(self, T, rhs, basis, names, ub, keep, max_iter, sign, offset):
        self.T, self.rhs, self.basis, self.names = T, rhs, basis, names
        self.ub = ub
        self.flipped = np.zeros(T.shape[1], dtype=bool)
        self.keep, self.max_iter = keep, max_iter
        self.sign, self.offset = sign, offset  # para mostrar Z en el sentido original
        self.iters = 0
        self.pivots = 0
        self.degenerate_pivots = 0
        self.snapshots: List[Dict[str, Any]] = []
        self.ray: Optional[np.ndarray] = None

    def flip(self, j, costs):
        """Sustituye x_j = u_j − x'_j: la variable pasa a medirse desde su cota."""
        self.rhs -= self.T[:, j] * self.ub[j]
        self.T[:, j] *= -1
        for c in costs:
            if c is not None:
                c[j] *= -1
        self.flipped[j] = not self.flipped[j]

    def pivot(self, r, j):
        p = self.T[r, j]
        self.T[r] /= p
        self.rhs[r] /= p
        for i in range(self.T.shape[0]):
            if i != r and self.T[i, j] != 0:
                f = self.T[i, j]
                self.T[i] -= f * self.T[r]
                self.rhs[i] -= f * self.rhs[r]
        self.basis[r] = j

    def reduced(self, c):
        return c - c[self.basis] @ self.T

    def values(self):
        v = np.zeros(self.T.shape[1])
        v[self.basis] = self.rhs
        for j in np.where(self.flipped)[0]:
            v[j] = self.ub[j] - v[j]
        return v

    def snapshot(self, phase, c0, c1, enter="", leave="", theta=None, note=""):
        if not self.keep:
            self.snapshots.append({"Iter": self.iters, "Fase": phase, "Entra": enter,
                                   "Sale": leave, "θ": theta, "Nota": note})
            return
        r0 = self.reduced(c0)
        r1 = self.reduced(c1) if c1 is not None else None
        self.snapshots.append({
            "iter": self.iters, "fase": phase, "T": self.T.copy(), "rhs": self.rhs.copy(),
            "basis": list(self.basis), "r0": r0, "r1": r1, "names": list(self.names),
            "z": self.current_z(), "enter": enter, "leave": leave, "theta": theta,
            "note": note,
        })

    def current_z(self):
        return self.sign * float(self.c_real @ self.values()) + self.offset

    def oriented(self, c):
        """Copia del costo con el signo de las variables medidas desde su cota."""
        out = c.copy()
        out[self.flipped] *= -1
        return out

    def run(self, c0, c1, phase, forbidden=()):
        c0 = self.oriented(c0)
        c1 = self.oriented(c1) if c1 is not None else None
        forbidden = set(forbidden)
        use_bland = False
        n = self.T.shape[1]
        self.snapshot(phase, c0, c1, note="Tableau inicial")
        while True:
            if self.iters >= self.max_iter:
                raise RuntimeError(
                    f"El Simplex superó el límite de {self.max_iter} iteraciones.")
            r0 = self.reduced(c0)
            r1 = self.reduced(c1) if c1 is not None else None
            allowed = np.ones(n, dtype=bool)
            allowed[list(self.basis)] = False
            if forbidden:
                allowed[list(forbidden)] = False

            cand = np.zeros(n, dtype=bool)
            score = r0
            if r1 is not None:
                cand = allowed & (r1 < -EPS)
                score = r1
            if not cand.any():
                tie = allowed if r1 is None else allowed & (np.abs(r1) <= EPS)
                cand = tie & (r0 < -EPS)
                score = r0
            if not cand.any():
                return OPTIMAL

            idx = np.where(cand)[0]
            j = int(idx[0]) if use_bland else int(idx[np.argmin(score[idx])])

            col = self.T[:, j]
            theta, row, to_upper = np.inf, -1, False
            for i in range(self.T.shape[0]):
                b = self.basis[i]
                if col[i] > EPS:
                    t, up = max(self.rhs[i], 0.0) / col[i], False
                elif col[i] < -EPS and np.isfinite(self.ub[b]):
                    t, up = max(self.ub[b] - self.rhs[i], 0.0) / (-col[i]), True
                else:
                    continue
                if t < theta - EPS or (abs(t - theta) <= EPS and b < self.basis[row]):
                    theta, row, to_upper = t, i, up

            self.iters += 1
            if self.ub[j] < theta - EPS:
                # La propia variable llega a su cota: no hay cambio de base.
                self.flip(j, (c0, c1))
                self.snapshot(phase, c0, c1, self.names[j], "—", float(self.ub[j]),
                              f"{self.names[j]} llega a su cota superior")
                use_bland = False
                continue
            if row < 0:
                ray = np.zeros(n)
                ray[j] = -1.0 if self.flipped[j] else 1.0
                for i, b in enumerate(self.basis):
                    ray[b] = col[i] if self.flipped[b] else -col[i]
                self.ray = ray
                self.snapshot(phase, c0, c1, self.names[j], "—", None,
                              f"Ninguna fila limita a {self.names[j]}: problema no acotado")
                return UNBOUNDED

            leaving = self.names[self.basis[row]]
            if to_upper:
                self.flip(self.basis[row], (c0, c1))
            self.pivot(row, j)
            self.pivots += 1
            degenerate = theta <= EPS
            if degenerate:
                self.degenerate_pivots += 1
            # Tras un pivote degenerado se aplica Bland para evitar ciclos; un
            # pivote con avance real mejora Z y permite volver a Dantzig.
            use_bland = degenerate
            self.snapshot(phase, c0, c1, self.names[j], leaving, float(theta),
                          "Pivote degenerado (θ = 0)" if degenerate else "")


def _standard_form(c, A, signs, b, lb, ub, sense):
    n = len(c)
    cost = -c if sense == "max" else c.copy()
    rhs = b - A @ lb
    span = ub - lb
    if np.any(span < -EPS):
        raise ValueError("Hay una variable con cota inferior mayor que su cota superior.")

    m = len(rhs)
    A2 = A.copy()
    sg = list(signs)
    for i in range(m):
        if rhs[i] < 0:
            rhs[i] *= -1
            A2[i] *= -1
            sg[i] = {"<=": ">=", ">=": "<=", "=": "="}[sg[i]]

    extra: List[np.ndarray] = []
    names = [f"x{j + 1}" for j in range(n)]
    art_idx: List[int] = []
    basis = [-1] * m

    def add(row, val, name):
        col = np.zeros(m)
        col[row] = val
        extra.append(col)
        names.append(name)
        return n + len(extra) - 1

    for i, s in enumerate(sg):
        if s == "<=":
            basis[i] = add(i, 1.0, f"s{i + 1}")
        elif s == ">=":
            add(i, -1.0, f"e{i + 1}")
            basis[i] = add(i, 1.0, f"a{i + 1}")
            art_idx.append(basis[i])
        else:
            basis[i] = add(i, 1.0, f"a{i + 1}")
            art_idx.append(basis[i])

    T = np.column_stack([A2] + extra) if extra else A2
    N = T.shape[1]
    c0 = np.zeros(N)
    c0[:n] = cost
    bounds = np.full(N, np.inf)
    bounds[:n] = span
    return T, rhs, basis, names, c0, bounds, art_idx


def solve_lp(c: Sequence[float], A: Sequence[Sequence[float]], signs: Sequence[str],
             b: Sequence[float], sense: str = "max", method: str = "big_m",
             lb: Optional[Sequence[float]] = None, ub: Optional[Sequence[float]] = None,
             keep_tableaus: bool = True, max_iter: int = 50_000) -> Dict[str, Any]:
    """Resuelve el PL con Simplex tabular.

    method: "simplex" (exige base inicial de holguras), "big_m" o "two_phase".
    """
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[1] != len(c):
        raise ValueError(f"Cada restricción debe tener {len(c)} coeficientes, "
                         "uno por variable.")
    n = len(c)
    lb_a = np.zeros(n) if lb is None else np.asarray(lb, dtype=float)
    ub_a = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=float)
    if len(lb_a) != n or len(ub_a) != n:
        raise ValueError(f"Debe indicar cotas para las {n} variables.")
    if not np.all(np.isfinite(lb_a)):
        raise ValueError("Las cotas inferiores deben ser finitas.")

    T, rhs, basis, names, c0, bounds, art = _standard_form(c, A, signs, b, lb_a, ub_a, sense)
    if method == "simplex" and art:
        raise ValueError(
            "El Simplex estándar necesita una base inicial de holguras, es decir, "
            "solo restricciones ≤ con lado derecho no negativo.\n\n"
            "Para restricciones ≥ o =, usa «PL: Big M» o «PL: Dos Fases».")

    sign = -1.0 if sense == "max" else 1.0
    offset = float(c @ lb_a)
    S = _Simplex(T, rhs, basis, names, bounds, keep_tableaus, max_iter, sign, offset)
    S.c_real = c0.copy()

    c1 = None
    if art:
        c1 = np.zeros(T.shape[1])
        c1[art] = 1.0

    removed_rows = 0
    if method == "two_phase" and art:
        status = S.run(c1, None, "Fase 1", ())
        vals = S.values()
        if status != OPTIMAL or vals[art].sum() > FEAS_TOL:
            return _finish(S, c, n, c0, INFEASIBLE, art, lb_a, removed_rows)
        # Saca de la base las artificiales que quedaron en cero.
        art_set = set(art)
        keep_rows = []
        for r, bv in enumerate(list(S.basis)):
            if bv in art_set:
                nz = [j for j in range(T.shape[1])
                      if j not in art_set and j not in S.basis and abs(S.T[r, j]) > EPS]
                if nz:
                    S.pivot(r, nz[0])
                    keep_rows.append(r)
                else:
                    removed_rows += 1  # fila redundante
            else:
                keep_rows.append(r)
        S.T, S.rhs = S.T[keep_rows], S.rhs[keep_rows]
        S.basis = [S.basis[r] for r in keep_rows]
        status = S.run(c0, None, "Fase 2", art)
    else:
        phase = "Simplex" if not art else "Big M"
        status = S.run(c0, c1, phase, ())
        if art and status == OPTIMAL and S.values()[art].sum() > FEAS_TOL:
            status = INFEASIBLE
        if art and status == UNBOUNDED and c1 is not None \
                and S.values()[art].sum() > FEAS_TOL:
            status = INFEASIBLE

    return _finish(S, c, n, c0, status, art, lb_a, removed_rows)


def _tableau_rows(S: _Simplex) -> List[Dict[str, Any]]:
    if not S.keep or S.iters > MAX_TABLEAU_ITERS:
        out = []
        for sn in S.snapshots:
            if "T" in sn:
                out.append({"Iter": sn["iter"], "Fase": sn["fase"], "Entra": sn["enter"],
                            "Sale": sn["leave"], "θ": sn["theta"], "Nota": sn["note"]})
            else:
                out.append(sn)
        return out
    rows: List[Dict[str, Any]] = []
    for sn in S.snapshots:
        for i, bv in enumerate(sn["basis"]):
            row: Dict[str, Any] = {"Iter": sn["iter"], "Fase": sn["fase"],
                                   "Base": sn["names"][bv]}
            for j, nm in enumerate(sn["names"]):
                row[nm] = round(float(sn["T"][i, j]), 6)
            row["LD"] = round(float(sn["rhs"][i]), 6)
            rows.append(row)
        z_row: Dict[str, Any] = {"Iter": sn["iter"], "Fase": sn["fase"], "Base": "Cj−Zj"}
        for j, nm in enumerate(sn["names"]):
            z_row[nm] = (_fmt_big_m(sn["r1"][j], sn["r0"][j]) if sn["r1"] is not None
                         else round(float(sn["r0"][j]), 6))
        note = sn["note"]
        if sn["enter"]:
            note = f"Entra {sn['enter']}, sale {sn['leave']}. {note}".strip()
        z_row["Nota"] = note
        rows.append(z_row)
    return rows


def _finish(S, c, n, c0, status, art, lb, removed_rows):
    vals = S.values()
    x = lb + vals[:n]
    z = float(c @ x)
    basic_vals = np.array([vals[b] for b in S.basis])
    degenerate = status == OPTIMAL and bool(np.any(np.abs(basic_vals) <= 1e-8))

    alternate = False
    if status == OPTIMAL:
        r = S.reduced(S.oriented(c0))
        nonbasic = np.ones(len(r), dtype=bool)
        nonbasic[list(S.basis)] = False
        if art:
            nonbasic[art] = False
        alternate = bool(np.any(np.abs(r[nonbasic]) <= 1e-8))

    rows = _tableau_rows(S)
    notes: List[str] = []
    if S.degenerate_pivots:
        notes.append(f"{S.degenerate_pivots} pivote(s) degenerado(s)")
    if degenerate:
        notes.append("solución degenerada (variable básica en cero)")
    if alternate:
        notes.append("existen óptimos alternativos")
    if removed_rows:
        notes.append(f"{removed_rows} restricción(es) redundante(s)")

    ray = None
    if status == UNBOUNDED and S.ray is not None:
        ray = (S.ray[:n] + 0.0).tolist()  # + 0.0 convierte −0.0 en 0.0

    if status == OPTIMAL:
        rows.append({"Iter": "Final", "Base": "Z*", "LD": round(z, 6),
                     "Nota": "; ".join(notes) or "Óptimo único y no degenerado"})
        for j in range(n):
            rows.append({"Iter": "Final", "Base": f"x{j + 1}", "LD": round(float(x[j]), 6)})
        x_txt = ", ".join(f"x{j + 1}={x[j]:.4g}" for j in range(n))
        resumen = f"Óptimo: {x_txt}, Z* = {z:.6g}"
        if notes:
            resumen += f" ({'; '.join(notes)})"
    elif status == UNBOUNDED:
        ray_txt = ", ".join(f"{v:.4g}" for v in ray) if ray else ""
        rows.append({"Iter": "Final", "Base": "Z", "LD": "±∞",
                     "Nota": f"Problema no acotado. Dirección de crecimiento: ({ray_txt})"})
        resumen = f"Problema NO ACOTADO: Z crece sin límite en la dirección ({ray_txt})"
    else:
        rows.append({"Iter": "Final", "Base": "—", "LD": "—",
                     "Nota": "Infactible: una variable artificial sigue positiva"})
        resumen = "Problema INFACTIBLE: ningún punto cumple todas las restricciones"

    return {
        "status": status, "x": x.tolist(), "z": z if status == OPTIMAL else None,
        "pivots": S.pivots, "iters": S.iters, "degenerate_pivots": S.degenerate_pivots,
        "degenerate": degenerate, "alternate": alternate, "ray": ray,
        "hist": rows, "resumen": resumen, "plot": None,
    }


# ── Método gráfico ───────────────────────────────────────────────────────────
def _intersections(A, b):
    pts = []
    for i in range(len(b)):
        for k in range(i + 1, len(b)):
            M = np.array([A[i], A[k]])
            if abs(np.linalg.det(M)) > 1e-10:
                pts.append(np.linalg.solve(M, [b[i], b[k]]))
    return pts


def _is_feasible(p, A, signs, b, tol=1e-7):
    if p[0] < -tol or p[1] < -tol:
        return False
    for row, s, rhs in zip(A, signs, b):
        v = row @ p
        if (s == "<=" and v > rhs + tol) or (s == ">=" and v < rhs - tol) \
                or (s == "=" and abs(v - rhs) > tol):
            return False
    return True


def _polygon(A, signs, b, xmax, ymax):
    """Vértices de la región factible recortada a la caja de dibujo."""
    extra_A = np.vstack([A, [[1, 0], [0, 1], [1, 0], [0, 1]]])
    extra_b = np.concatenate([b, [0, 0, xmax, ymax]])
    pts = [p for p in _intersections(extra_A, extra_b)
           if _is_feasible(p, A, signs, b) and p[0] <= xmax + 1e-7 and p[1] <= ymax + 1e-7]
    if len(pts) < 2:
        return pts
    P = np.array(pts)
    center = P.mean(axis=0)
    order = np.argsort(np.arctan2(P[:, 1] - center[1], P[:, 0] - center[0]))
    return [tuple(v) for v in P[order]]


def graphical(c, A, signs, b, sense="max") -> Dict[str, Any]:
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(c) != 2 or A.shape[1] != 2:
        raise ValueError("El método gráfico solo aplica a problemas de 2 variables.")

    res = solve_lp(c, A, signs, b, sense, "big_m", keep_tableaus=False)
    axes_A = np.vstack([A, [[1, 0], [0, 1]]])
    axes_b = np.concatenate([b, [0, 0]])
    vertices = []
    for p in _intersections(axes_A, axes_b):
        if _is_feasible(p, A, signs, b) and not any(np.allclose(p, v, atol=1e-7) for v in vertices):
            vertices.append(p)
    vertices.sort(key=lambda p: (p[0], p[1]))

    best = None
    if res["status"] == OPTIMAL:
        best = float(res["z"])
    rows = []
    for p in vertices:
        z = float(c @ p)
        is_best = best is not None and abs(z - best) <= 1e-7 * max(1.0, abs(best))
        rows.append({"Vértice": f"({p[0]:.4g}, {p[1]:.4g})", "x1": float(p[0]), "x2": float(p[1]),
                     "Z": z, "Estado": "Óptimo" if is_best else ""})
    if not rows:
        rows.append({"Vértice": "—", "Estado": "Región factible vacía"})

    span = max([1.0] + [v for p in vertices for v in p])
    xmax = ymax = span * 1.6

    def plot(fig):
        ax = fig.add_subplot(111)
        grid = np.array([0.0, xmax])
        for i, (row, rhs) in enumerate(zip(A, b)):
            lbl = f"R{i + 1}"
            if abs(row[1]) > 1e-12:
                ax.plot(grid, (rhs - row[0] * grid) / row[1], label=lbl)
            elif abs(row[0]) > 1e-12:
                ax.axvline(rhs / row[0], label=lbl)
        poly = _polygon(A, signs, b, xmax, ymax)
        if len(poly) >= 3:
            ax.fill(*zip(*poly), color="#4a9eff", alpha=0.3, label="Región factible")
        if res["status"] == OPTIMAL:
            ax.plot(res["x"][0], res["x"][1], "o", color="#ffd700", markersize=10,
                    label=f"Óptimo, Z = {res['z']:.4g}")
            if abs(c[1]) > 1e-12:
                ax.plot(grid, (res["z"] - c[0] * grid) / c[1], "--", color="#ffd700")
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_title("Método gráfico"); ax.legend()

    return {**res, "hist": rows, "vertices": [p.tolist() for p in vertices], "plot": plot}
