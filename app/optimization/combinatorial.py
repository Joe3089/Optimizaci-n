# app/optimization/combinatorial.py
# Optimización combinatoria: mochila 0/1 (ramificación y acotamiento), viajante,
# Máximo Corte y comparación de complejidad computacional.
from __future__ import annotations

import heapq
import math
from itertools import count
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int, float]


# ── Mochila 0/1 ──────────────────────────────────────────────────────────────
def knapsack_bb(values: Sequence[float], weights: Sequence[float], capacity: float) -> Dict[str, Any]:
    n = len(values)
    if n == 0 or n != len(weights):
        raise ValueError("Valores y pesos deben tener la misma cantidad de elementos.")
    if any(w <= 0 for w in weights) or capacity <= 0:
        raise ValueError("Los pesos y la capacidad deben ser positivos.")
    order = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    v = [float(values[i]) for i in order]
    w = [float(weights[i]) for i in order]

    def bound(level, val, wt):
        room = capacity - wt
        for i in range(level, n):
            if w[i] <= room:
                room -= w[i]
                val += v[i]
            else:
                return val + v[i] * room / w[i]
        return val

    best_val, best_set = 0.0, []
    tie = count()
    heap = [(-bound(0, 0.0, 0.0), next(tie), 0, 0.0, 0.0, [])]
    rows: List[Dict[str, Any]] = []
    nodes = 0
    while heap:
        neg_b, _, level, val, wt, chosen = heapq.heappop(heap)
        nodes += 1
        ub = -neg_b
        state = "Ramificado"
        if ub <= best_val + 1e-12:
            state = "Podado por cota"
        elif level == n:
            state = "Hoja"
        else:
            if wt + w[level] <= capacity + 1e-12:
                nv, nw = val + v[level], wt + w[level]
                pick = chosen + [order[level] + 1]
                if nv > best_val:
                    best_val, best_set = nv, pick
                heapq.heappush(heap, (-bound(level + 1, nv, nw), next(tie), level + 1, nv, nw, pick))
            heapq.heappush(heap, (-bound(level + 1, val, wt), next(tie), level + 1, val, wt, chosen))
        if nodes <= 200:
            rows.append({"Nodo": nodes, "Nivel": level, "Valor": val, "Peso": wt, "Cota": ub,
                         "Objetos": str(sorted(chosen)), "Estado": state,
                         "Mejor valor": best_val})
    best_set = sorted(best_set)
    if nodes > 200:
        rows.append({"Nodo": "…", "Estado": f"Se muestran 200 de {nodes} nodos"})
    rows.append({"Nodo": "Final", "Objetos": str(best_set), "Mejor valor": best_val,
                 "Estado": f"{nodes} nodos explorados de {2 ** (n + 1) - 1} posibles"})

    def plot(fig):
        ax = fig.add_subplot(111)
        seq = [r["Mejor valor"] for r in rows if isinstance(r["Nodo"], int)]
        ax.step(range(1, len(seq) + 1), seq, where="post", color="#ffd700")
        ax.set_xlabel("Nodo explorado"); ax.set_ylabel("Mejor valor")
        ax.set_title("Ramificación y acotamiento")

    return {"chosen": best_set, "total": best_val, "nodes": nodes, "hist": rows, "plot": plot,
            "resumen": f"Objetos {best_set}: valor = {best_val:g} ({nodes} nodos explorados)"}


# ── Viajante ─────────────────────────────────────────────────────────────────
def _tour_len(tour, D):
    return float(sum(D[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))))


def _held_karp(D):
    n = len(D)
    table = {(1 << k, k): (D[0, k], 0) for k in range(1, n)}
    for size in range(2, n):
        for mask in range(1, 1 << n, 2):
            sub = mask >> 1
            if bin(sub).count("1") != size:
                continue
            bits = sub << 1
            for k in range(1, n):
                if not bits >> k & 1:
                    continue
                prev = bits ^ (1 << k)
                table[(bits, k)] = min(
                    (table[(prev, j)][0] + D[j, k], j)
                    for j in range(1, n) if prev >> j & 1)
    full = (1 << n) - 2
    best, last = min((table[(full, k)][0] + D[k, 0], k) for k in range(1, n))
    tour = []
    mask = full
    while last != 0:
        tour.append(last)
        mask, last = mask ^ (1 << last), table[(mask, last)][1]
    return float(best), [0] + tour[::-1]


def _nearest_neighbor(D):
    n = len(D)
    tour, left = [0], set(range(1, n))
    while left:
        nxt = min(left, key=lambda j: D[tour[-1], j])
        tour.append(nxt)
        left.remove(nxt)
    return tour


def _two_opt(tour, D):
    tour = tour[:]
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 2, n):
                if i == 0 and k == n - 1:
                    continue
                a, b, c, d = tour[i], tour[i + 1], tour[k], tour[(k + 1) % n]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-12:
                    tour[i + 1:k + 1] = reversed(tour[i + 1:k + 1])
                    improved = True
    return tour


def tsp(coords: Sequence[Sequence[float]]) -> Dict[str, Any]:
    P = np.asarray(coords, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2 or len(P) < 3:
        raise ValueError("Ingresa al menos 3 ciudades, una por línea: x, y.")
    n = len(P)
    if n > 60:
        raise ValueError("El máximo admitido es 60 ciudades.")
    D = np.linalg.norm(P[:, None] - P[None, :], axis=2)
    nn = _nearest_neighbor(D)
    opt2 = _two_opt(nn, D)
    fmt = lambda t: " → ".join(str(v + 1) for v in t + [t[0]])
    hist = [{"Método": "Vecino más cercano", "Longitud": _tour_len(nn, D), "Ruta": fmt(nn)},
            {"Método": "Mejora 2-opt", "Longitud": _tour_len(opt2, D), "Ruta": fmt(opt2)}]
    best_tour, best = opt2, _tour_len(opt2, D)
    exact = None
    if n <= 13:
        exact, tour = _held_karp(D)
        hist.append({"Método": "Exacto (Held-Karp)", "Longitud": exact, "Ruta": fmt(tour)})
        best_tour, best = tour, exact
    else:
        hist.append({"Método": "Exacto (Held-Karp)", "Ruta": "Se omite con más de 13 ciudades"})

    def plot(fig):
        ax = fig.add_subplot(111)
        loop = best_tour + [best_tour[0]]
        ax.plot(P[loop, 0], P[loop, 1], "o-", color="#4a9eff")
        for i, (x, y) in enumerate(P):
            ax.annotate(str(i + 1), (x, y), color="#ffd700")
        ax.set_title(f"Mejor ruta (longitud = {best:.4g})")

    return {"best": best, "tour": best_tour, "exact": exact, "hist": hist, "plot": plot,
            "resumen": f"Longitud = {best:.4f}. Ruta: {fmt(best_tour)}"}


# ── Máximo Corte ─────────────────────────────────────────────────────────────
def check_graph(n: int, edges: Sequence[Edge], max_nodes: int) -> None:
    if n < 2 or not edges:
        raise ValueError("El grafo necesita al menos 2 nodos y 1 arista.")
    if n > max_nodes:
        raise ValueError(f"Este método admite hasta {max_nodes} nodos.")
    for u, v, _ in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise ValueError(f"Arista inválida: {u} {v}. Los nodos van de 0 a {n - 1}.")


def cut_values(n: int, edges: Sequence[Edge]) -> np.ndarray:
    """Valor del corte para las 2^n particiones (bit q = lado del nodo q)."""
    idx = np.arange(1 << n)
    total = np.zeros(1 << n)
    for u, v, w in edges:
        total += w * (((idx >> u) ^ (idx >> v)) & 1)
    return total


def bits(i: int, n: int) -> str:
    """Cadena de bits con el nodo/qubit 0 a la izquierda."""
    return "".join(str((i >> q) & 1) for q in range(n))


def _gain_cut(side, edges):
    return sum(w for u, v, w in edges if side[u] != side[v])


def _greedy(n, edges):
    side = [0] * n
    for node in range(1, n):
        score = {0: 0.0, 1: 0.0}
        for u, v, w in edges:
            other = v if u == node else (u if v == node else None)
            if other is not None and other < node:
                score[1 - side[other]] += w
        side[node] = 1 if score[1] > score[0] else 0
    return side


def _local_search(side, edges):
    side = side[:]
    improved = True
    while improved:
        improved = False
        base = _gain_cut(side, edges)
        for q in range(len(side)):
            side[q] ^= 1
            cand = _gain_cut(side, edges)
            if cand > base + 1e-12:
                base = cand
                improved = True
            else:
                side[q] ^= 1
    return side


def graph_plot(fig, n, edges, side, title):
    ax = fig.add_subplot(111)
    ang = 2 * np.pi * np.arange(n) / n
    xy = np.column_stack([np.cos(ang), np.sin(ang)])
    for u, v, _ in edges:
        cut = side[u] != side[v]
        ax.plot(xy[[u, v], 0], xy[[u, v], 1], color="#ff6b35" if cut else "#35507f",
                linewidth=2.5 if cut else 1, linestyle="-" if cut else "--")
    ax.scatter(xy[:, 0], xy[:, 1], s=320, zorder=3,
               c=["#ffd700" if s else "#4a9eff" for s in side])
    for i, (x, y) in enumerate(xy):
        ax.annotate(str(i), (x, y), ha="center", va="center", color="#0a1228", zorder=4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)


def max_cut(n: int, edges: Sequence[Edge]) -> Dict[str, Any]:
    check_graph(n, edges, 22)
    vals = cut_values(n, edges)
    best_i = int(np.argmax(vals))
    best = float(vals[best_i])
    side_opt = [(best_i >> q) & 1 for q in range(n)]
    g = _greedy(n, edges)
    ls = _local_search(g, edges)

    def row(name, side, note=""):
        v = float(_gain_cut(side, edges))
        return {"Método": name, "Partición": "".join(map(str, side)), "Valor del corte": v,
                "Razón vs. óptimo": v / best if best else 1.0, "Nota": note}

    hist = [row("Voraz", g), row("Búsqueda local (1 cambio)", ls),
            row("Exacto (fuerza bruta)", side_opt, f"{2 ** n} particiones evaluadas")]

    def plot(fig):
        graph_plot(fig, n, edges, side_opt, f"Máximo Corte = {best:g} (aristas cortadas en naranja)")

    return {"best": best, "side": side_opt, "greedy": hist[0]["Valor del corte"],
            "local": hist[1]["Valor del corte"], "hist": hist, "plot": plot,
            "resumen": f"Corte máximo = {best:g}, partición {''.join(map(str, side_opt))}"}


# ── Complejidad computacional ────────────────────────────────────────────────
_GROWTH = [
    ("O(log n)", lambda n: math.log10(max(math.log2(n), 1e-9))),
    ("O(n)", lambda n: math.log10(n)),
    ("O(n log n)", lambda n: math.log10(n * max(math.log2(n), 1e-9))),
    ("O(n²)", lambda n: 2 * math.log10(n)),
    ("O(n³)", lambda n: 3 * math.log10(n)),
    ("O(2ⁿ)", lambda n: n * math.log10(2)),
    ("O(n!)", lambda n: math.lgamma(n + 1) / math.log(10)),
]


def human_time(log10_ops: float, speed: float) -> str:
    log_s = log10_ops - math.log10(speed)
    if log_s > 300:
        return f"≈ 10^{log_s - 7.5:.0f} años"
    sec = 10 ** log_s
    for limit, div, unit in [(1e-6, 1e-9, "ns"), (1e-3, 1e-6, "µs"), (1, 1e-3, "ms"),
                             (60, 1, "s"), (3600, 60, "min"), (86400, 3600, "h"),
                             (3.15e7, 86400, "días"), (3.15e9, 3.15e7, "años")]:
        if sec < limit:
            return f"{sec / div:.3g} {unit}"
    return f"≈ 10^{log_s - 7.5:.0f} años"


def complexity(sizes: Sequence[int], speed: float) -> Dict[str, Any]:
    if not sizes or any(n < 1 or int(n) != n for n in sizes):
        raise ValueError("Los tamaños n deben ser enteros positivos.")
    if speed <= 0:
        raise ValueError("La velocidad debe ser positiva.")
    sizes = sorted(int(n) for n in sizes)
    hist = []
    for name, fn in _GROWTH:
        row: Dict[str, Any] = {"Complejidad": name}
        for n in sizes:
            row[f"n = {n}"] = human_time(fn(n), speed)
        hist.append(row)
    limit = next((n for n in range(1, 400)
                  if _GROWTH[5][1](n) - math.log10(speed) > math.log10(3.15e7)), None)

    def plot(fig):
        ax = fig.add_subplot(111)
        for name, fn in _GROWTH:
            ax.plot(sizes, [fn(n) for n in sizes], "o-", label=name)
        ax.set_xlabel("n"); ax.set_ylabel("log₁₀(operaciones)")
        ax.set_title("Crecimiento de las operaciones"); ax.legend()

    return {"hist": hist, "plot": plot, "limit_2n": limit,
            "resumen": f"A {speed:.3g} op/s, O(2ⁿ) supera un año de cálculo desde n = {limit}"}
