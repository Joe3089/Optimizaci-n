# app/operations_research/transportation.py
# Problema de transporte: esquina noroeste, Vogel y Simplex de transporte (MODI).
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

EPS = 1e-9
Cell = Tuple[int, int]


def matrix_plot(fig, mat, row_names, col_names, title, labels=None):
    ax = fig.add_subplot(111)
    M = np.asarray(mat, dtype=float)
    ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(col_names)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(row_names)
    mid = (M.max() + M.min()) / 2 if M.size else 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = labels[i][j] if labels is not None else f"{M[i, j]:g}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="#ffffff" if M[i, j] > mid else "#0a1228")
    ax.set_title(title)


def _balance(supply, demand, cost):
    s, d = np.array(supply, float), np.array(demand, float)
    c = np.array(cost, float)
    if c.shape != (len(s), len(d)):
        raise ValueError(f"La matriz de costos debe ser de {len(s)}×{len(d)} "
                         "(una fila por origen, una columna por destino).")
    if np.any(s < 0) or np.any(d < 0):
        raise ValueError("La oferta y la demanda no pueden ser negativas.")
    rows = [f"O{i + 1}" for i in range(len(s))]
    cols = [f"D{j + 1}" for j in range(len(d))]
    gap = float(s.sum() - d.sum())
    note = "Problema balanceado"
    if gap > EPS:
        d = np.append(d, gap)
        c = np.column_stack([c, np.zeros(len(s))])
        cols.append("D-Ficticio")
        note = f"Se agregó un destino ficticio con demanda {gap:g}"
    elif gap < -EPS:
        s = np.append(s, -gap)
        c = np.vstack([c, np.zeros(len(d))])
        rows.append("O-Ficticio")
        note = f"Se agregó un origen ficticio con oferta {-gap:g}"
    return s, d, c, rows, cols, note


def _northwest(s, d, c, rows, cols):
    s, d = s.copy(), d.copy()
    alloc: Dict[Cell, float] = {}
    steps = []
    i = j = 0
    while i < len(s) and j < len(d):
        q = min(s[i], d[j])
        alloc[(i, j)] = float(q)
        s[i] -= q
        d[j] -= q
        steps.append({"Paso": len(steps) + 1, "Celda": f"{rows[i]}→{cols[j]}", "Cantidad": float(q),
                      "Costo unitario": float(c[i, j]), "Oferta restante": float(s[i]),
                      "Demanda restante": float(d[j])})
        if s[i] <= EPS and i < len(s) - 1:
            i += 1
        elif d[j] <= EPS:
            j += 1
        else:
            i += 1
    return alloc, steps


def _penalty(values):
    vals = sorted(values)
    return vals[1] - vals[0] if len(vals) > 1 else (vals[0] if vals else -1.0)


def _vogel(s, d, c, rows, cols):
    s, d = s.copy(), d.copy()
    alloc: Dict[Cell, float] = {}
    steps = []
    active_r, active_c = set(range(len(s))), set(range(len(d)))
    while active_r and active_c:
        rp = {i: _penalty([c[i, j] for j in active_c]) for i in active_r}
        cp = {j: _penalty([c[i, j] for i in active_r]) for j in active_c}
        best_r = max(rp, key=lambda k: (rp[k], -k))
        best_c = max(cp, key=lambda k: (cp[k], -k))
        if rp[best_r] >= cp[best_c]:
            i = best_r
            j = min(active_c, key=lambda k: (c[i, k], k))
            pen_txt = f"Fila {rows[i]}: {rp[best_r]:g}"
        else:
            j = best_c
            i = min(active_r, key=lambda k: (c[k, j], k))
            pen_txt = f"Columna {cols[j]}: {cp[best_c]:g}"
        q = float(min(s[i], d[j]))
        if q > EPS:
            alloc[(i, j)] = q
        s[i] -= q
        d[j] -= q
        steps.append({"Paso": len(steps) + 1, "Mayor penalización": pen_txt,
                      "Celda": f"{rows[i]}→{cols[j]}", "Cantidad": q,
                      "Costo unitario": float(c[i, j])})
        if s[i] <= EPS:
            active_r.discard(i)
        if d[j] <= EPS:
            active_c.discard(j)
    return alloc, steps


def _complete_basis(alloc, c):
    """Añade celdas en cero hasta tener un árbol de m+n−1 celdas básicas."""
    m, n = c.shape
    parent = list(range(m + n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in alloc:
        parent[find(i)] = find(m + j)
    cells = sorted(((c[i, j], i, j) for i in range(m) for j in range(n) if (i, j) not in alloc))
    for _, i, j in cells:
        if len(alloc) >= m + n - 1:
            break
        a, b = find(i), find(m + j)
        if a != b:
            parent[a] = b
            alloc[(i, j)] = 0.0
    return alloc


def _potentials(basis, c):
    m, n = c.shape
    u: List[Any] = [None] * m
    v: List[Any] = [None] * n
    u[0] = 0.0
    pending = list(basis)
    while pending:
        rest = []
        for i, j in pending:
            if u[i] is not None and v[j] is None:
                v[j] = c[i, j] - u[i]
            elif v[j] is not None and u[i] is None:
                u[i] = c[i, j] - v[j]
            elif u[i] is None and v[j] is None:
                rest.append((i, j))
        if len(rest) == len(pending):
            raise RuntimeError("La base de transporte no es un árbol conexo.")
        pending = rest
    return u, v


def _cycle(basis, enter, m):
    """Celdas básicas del camino que cierra el ciclo con la celda entrante."""
    adj: Dict[int, List[Tuple[int, Cell]]] = {}
    for i, j in basis:
        adj.setdefault(i, []).append((m + j, (i, j)))
        adj.setdefault(m + j, []).append((i, (i, j)))
    start, goal = m + enter[1], enter[0]
    stack = [(start, -1, [])]
    while stack:
        node, prev, path = stack.pop()
        if node == goal:
            return path
        for nxt, cell in adj.get(node, []):
            if nxt != prev:
                stack.append((nxt, node, path + [cell]))
    raise RuntimeError("No se encontró el ciclo de transporte.")


def _modi(alloc, c, rows, cols):
    m, n = c.shape
    steps = []
    use_bland = False
    for it in range(1, 20001):
        u, v = _potentials(alloc, c)
        entering, best = None, -1e-9
        for i in range(m):
            for j in range(n):
                if (i, j) in alloc:
                    continue
                r = c[i, j] - u[i] - v[j]
                if r < best:
                    best, entering = r, (i, j)
                    if use_bland:
                        break
            if use_bland and entering:
                break
        total = sum(q * c[k] for k, q in alloc.items())
        pot = ("u=[" + ", ".join(f"{x:g}" for x in u) + "]  v=["
               + ", ".join(f"{x:g}" for x in v) + "]")
        if entering is None:
            steps.append({"Iteración": it, "Entra": "—", "Costo reducido": 0.0, "θ": "—",
                          "Sale": "—", "Costo total": float(total), "Potenciales": pot,
                          "Nota": "Todos los costos reducidos ≥ 0: óptimo"})
            return alloc, steps
        path = _cycle(alloc, entering, m)
        minus = path[0::2]
        plus = path[1::2]
        theta = min(alloc[k] for k in minus)
        leave = min((k for k in minus if alloc[k] <= theta + EPS), key=lambda k: k)
        for k in minus:
            alloc[k] -= theta
        for k in plus:
            alloc[k] += theta
        alloc[entering] = theta
        del alloc[leave]
        use_bland = theta <= EPS
        steps.append({
            "Iteración": it, "Entra": f"{rows[entering[0]]}→{cols[entering[1]]}",
            "Costo reducido": float(best), "θ": float(theta),
            "Sale": f"{rows[leave[0]]}→{cols[leave[1]]}", "Costo total": float(total),
            "Potenciales": pot, "Nota": "Pivote degenerado" if theta <= EPS else "",
        })
    raise RuntimeError("El Simplex de transporte superó el límite de iteraciones.")


def solve_transport(supply, demand, cost, method: str) -> Dict[str, Any]:
    s, d, c, rows, cols, note = _balance(supply, demand, cost)
    if method == "northwest":
        alloc, steps = _northwest(s, d, c, rows, cols)
    else:
        alloc, steps = _vogel(s, d, c, rows, cols)
    initial = float(sum(q * c[k] for k, q in alloc.items()))
    if method == "modi":
        alloc = _complete_basis(alloc, c)
        alloc, steps2 = _modi(alloc, c, rows, cols)
        steps = steps2
    total = float(sum(q * c[k] for k, q in alloc.items()))

    X = np.zeros(c.shape)
    for k, q in alloc.items():
        X[k] = q
    hist: List[Dict[str, Any]] = list(steps)
    for (i, j), q in sorted(alloc.items()):
        if q > EPS:
            hist.append({"Paso": "Final", "Celda": f"{rows[i]}→{cols[j]}", "Cantidad": float(q),
                         "Costo unitario": float(c[i, j]), "Costo": float(q * c[i, j])})
    hist.append({"Paso": "Final", "Celda": "Costo total", "Costo": total, "Nota": note})

    labels = [[f"{X[i, j]:g}\n(c={c[i, j]:g})" for j in range(c.shape[1])] for i in range(c.shape[0])]

    def plot(fig):
        matrix_plot(fig, X, rows, cols, f"Asignación final (costo total = {total:g})", labels)

    return {"allocation": X.tolist(), "total": total, "initial": initial, "note": note,
            "hist": hist, "plot": plot, "resumen": f"Costo total = {total:g}. {note}"}
