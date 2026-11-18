# app/operations_research/assignment.py
# Modelo de asignación resuelto con el método húngaro.
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from app.operations_research.transportation import matrix_plot

EPS = 1e-9


def _matrix_text(M) -> str:
    return " | ".join(" ".join(f"{v:g}" for v in row) for row in np.round(M, 6))


def _max_matching(Z):
    n = Z.shape[0]
    match_col = [-1] * n

    def augment(r, seen):
        for c in range(n):
            if Z[r, c] and not seen[c]:
                seen[c] = True
                if match_col[c] < 0 or augment(match_col[c], seen):
                    match_col[c] = r
                    return True
        return False

    for r in range(n):
        augment(r, [False] * n)
    return match_col


def _min_cover(Z, match_col):
    """Cubrimiento mínimo de ceros por el teorema de König."""
    n = Z.shape[0]
    match_row = [-1] * n
    for c, r in enumerate(match_col):
        if r >= 0:
            match_row[r] = c
    vis_r = [match_row[r] < 0 for r in range(n)]
    vis_c = [False] * n
    changed = True
    while changed:
        changed = False
        for r in range(n):
            if vis_r[r]:
                for c in range(n):
                    if Z[r, c] and not vis_c[c]:
                        vis_c[c] = True
                        changed = True
        for c in range(n):
            if vis_c[c] and match_col[c] >= 0 and not vis_r[match_col[c]]:
                vis_r[match_col[c]] = True
                changed = True
    return [not v for v in vis_r], vis_c


def hungarian(cost, sense: str = "min") -> Dict[str, Any]:
    C = np.asarray(cost, dtype=float)
    if C.ndim != 2 or C.size == 0:
        raise ValueError("Ingresa una matriz de costos, una fila por agente.")
    m, n0 = C.shape
    size = max(m, n0)
    P = np.zeros((size, size))
    P[:m, :n0] = C
    note = "Matriz cuadrada" if m == n0 else "Se agregaron filas o columnas ficticias con costo 0"

    W = (P.max() - P) if sense == "max" else P.copy()
    steps: List[Dict[str, Any]] = [{"Paso": "Matriz inicial", "Matriz": _matrix_text(W), "Nota": note}]
    W -= W.min(axis=1, keepdims=True)
    steps.append({"Paso": "Reducción de filas", "Matriz": _matrix_text(W)})
    W -= W.min(axis=0, keepdims=True)
    steps.append({"Paso": "Reducción de columnas", "Matriz": _matrix_text(W)})

    for it in range(1, 10 * size * size + 2):
        Z = np.abs(W) <= EPS
        match_col = _max_matching(Z)
        lines = sum(1 for r in match_col if r >= 0)
        if lines == size:
            break
        cov_r, cov_c = _min_cover(Z, match_col)
        cr, cc = np.array(cov_r), np.array(cov_c)
        delta = W[np.ix_(~cr, ~cc)].min()
        W[np.ix_(~cr, ~cc)] -= delta
        W[np.ix_(cr, cc)] += delta
        steps.append({"Paso": f"Ajuste {it}", "Matriz": _matrix_text(W), "Líneas": lines,
                      "Nota": f"Menor valor no cubierto: {delta:g}"})
    else:
        raise RuntimeError("El método húngaro no convergió.")

    pairs = []
    total = 0.0
    for c, r in enumerate(match_col):
        if r < m and c < n0:
            pairs.append((r, c))
            total += float(C[r, c])
    pairs.sort()
    hist = steps + [{"Paso": "Asignación", "Matriz": f"Agente {r + 1} → Tarea {c + 1}",
                     "Costo": float(C[r, c])} for r, c in pairs]
    hist.append({"Paso": "Total", "Costo": total})

    labels = [[("✔ " if (i, j) in pairs else "") + f"{C[i, j]:g}" for j in range(n0)] for i in range(m)]

    def plot(fig):
        matrix_plot(fig, C, [f"A{i + 1}" for i in range(m)], [f"T{j + 1}" for j in range(n0)],
                    f"Asignación óptima (total = {total:g})", labels)

    txt = ", ".join(f"A{r + 1}→T{c + 1}" for r, c in pairs)
    return {"pairs": pairs, "total": total, "hist": hist, "plot": plot,
            "resumen": f"{txt}. Total = {total:g}"}
