# app/optimization/meta_extra.py
# Metaheurísticas para problemas complejos que no son «una f(x)»:
#   • Multiobjetivo: NSGA-II con dominancia restringida (frente de Pareto)
#   • Ruteo de vehículos con capacidad (CVRP): ruta gigante + Split de Prins,
#     GA memético y Recocido Simulado
#   • Programación de tareas: flow shop (NEH + Iterated Greedy, SA, GA) y
#     job shop (GA con cruce POX, SA), con diagrama de Gantt
from __future__ import annotations

import itertools
import math
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from app.optimization import advanced_metaheuristics as am

MO_ALGORITHMS = ("NSGA-II",)
VRP_ALGORITHMS = ("Algoritmo Genético (Split de Prins)", "Recocido Simulado", "Comparar todos")
FLOW_ALGORITHMS = ("Iterated Greedy (NEH)", "Recocido Simulado", "Algoritmo Genético", "Comparar todos")
JOB_ALGORITHMS = ("Algoritmo Genético (POX)", "Recocido Simulado", "Comparar todos")
_PAL = ["#ffd700", "#4a9eff", "#3cff8a", "#ff6b35", "#c77dff", "#ff5c8a", "#5ce1e6",
        "#f4a261", "#9be15d", "#e76f51"]


def _check_sizes(iters, pop):
    if not 5 <= iters <= 5000 or not 4 <= pop <= 1000:
        raise ValueError("Usa entre 5 y 5 000 iteraciones y una población entre 4 y 1 000.")


# ══════════════════════════════════════════════════════════════════════════════
#  1. Multiobjetivo — NSGA-II
# ══════════════════════════════════════════════════════════════════════════════
def _parse_objectives(text: str) -> List[Tuple[int, str]]:
    objs = []
    for ln in (text or "").splitlines():
        if not ln.strip():
            continue
        m = re.match(r"^\s*(min(?:imizar)?|max(?:imizar)?)\s*[:=]?\s*(.+)$", ln.strip(), re.I)
        sense, expr = (1, ln.strip()) if not m else (-1 if m.group(1).lower().startswith("max") else 1,
                                                    m.group(2))
        objs.append((sense, expr))
    if len(objs) < 2:
        raise ValueError("Un problema multiobjetivo necesita al menos 2 objetivos, uno por "
                         "línea (ej.: «min: x**2» y «min: (x-2)**2»).")
    if len(objs) > 6:
        raise ValueError("El máximo admitido es 6 objetivos.")
    return objs


def _nondominated_sort(F: np.ndarray, viol: np.ndarray) -> List[np.ndarray]:
    """Frentes de Pareto con dominancia restringida (Deb): factible domina a
    infactible; entre infactibles domina la menor violación."""
    n = len(F)
    feas = viol <= 0
    le = (F[:, None, :] <= F[None, :, :]).all(-1)
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    dom = le & lt & feas[:, None] & feas[None, :]
    dom |= feas[:, None] & ~feas[None, :]
    dom |= (~feas[:, None]) & (~feas[None, :]) & (viol[:, None] < viol[None, :])
    count = dom.sum(0)
    fronts, assigned = [], np.zeros(n, bool)
    while not assigned.all():
        cur = np.where((count == 0) & ~assigned)[0]
        if cur.size == 0:
            cur = np.where(~assigned)[0]
        fronts.append(cur)
        assigned[cur] = True
        count = count - dom[cur].sum(0)
    return fronts


def _crowding(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        o = np.argsort(F[:, j])
        rng_ = F[o[-1], j] - F[o[0], j]
        d[o[0]] = d[o[-1]] = np.inf
        if rng_ > 0:
            d[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / rng_
    return d


def _sbx_mutation(P, lo, hi, rng, pc=0.9, eta_c=15, eta_m=20):
    n, d = P.shape
    kids = P.copy()
    for i in range(0, n - 1, 2):
        if rng.random() < pc:
            a, b = P[i], P[i + 1]
            u = rng.random(d)
            beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)),
                            (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
            swap = rng.random(d) < 0.5
            c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
            c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
            kids[i] = np.where(swap, c2, c1); kids[i + 1] = np.where(swap, c1, c2)
    pm = 1.0 / d
    mask = rng.random((n, d)) < pm
    u = rng.random((n, d))
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1,
                     1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
    kids = kids + mask * delta * (hi - lo)
    return kids


def hypervolume_2d(F: np.ndarray, ref: Sequence[float]) -> float:
    pts = F[np.argsort(F[:, 0])]
    hv, prev = 0.0, ref[1]
    for f1, f2 in pts:
        if f1 < ref[0] and f2 < prev:
            hv += (ref[0] - f1) * (prev - f2)
            prev = f2
    return hv


def solve_multiobjective(objectives: str, variables: str, constraints: str, generations: int,
                         population: int, seed: int) -> Dict[str, Any]:
    _check_sizes(generations, population)
    objs = _parse_objectives(objectives)
    joined = " + ".join(f"({e})" for _, e in objs)
    prob = am.parse_problem(joined, "Minimizar", variables, constraints)
    fs = [am._vector_fn(am._check_expr(e, f"el objetivo {k + 1}"), prob.names)
          for k, (_, e) in enumerate(objs)]
    sg = np.array([s for s, _ in objs], dtype=float)
    pop = population + population % 2
    rng = np.random.default_rng(seed)

    def evaluate(X):
        F = np.column_stack([f(X) for f in fs]) * sg          # todo a minimizar
        _, _, viol = prob.evaluate(X)
        F[~np.isfinite(F)] = 1e300
        return F, viol

    X = am._init_pop(prob, pop, rng)
    F, V = evaluate(X)
    snapshots, rows = [], []
    for g in range(1, generations + 1):
        fronts = _nondominated_sort(F, V)
        rank = np.empty(len(X), int)
        crowd = np.empty(len(X))
        for r, fr in enumerate(fronts):
            rank[fr] = r
            crowd[fr] = _crowding(F[fr])
        a, b = rng.integers(0, pop, (2, pop))
        win = np.where((rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b])), a, b)
        Q = prob.repair(_sbx_mutation(X[win], prob.lo, prob.hi, rng))
        FQ, VQ = evaluate(Q)
        R, FR, VR = np.vstack([X, Q]), np.vstack([F, FQ]), np.concatenate([V, VQ])
        keep: List[int] = []
        for fr in _nondominated_sort(FR, VR):
            if len(keep) + len(fr) <= pop:
                keep += fr.tolist()
            else:
                cd = _crowding(FR[fr])
                keep += fr[np.argsort(-cd)][:pop - len(keep)].tolist()
                break
        keep = np.array(keep)
        X, F, V = R[keep], FR[keep], VR[keep]
        f0 = _nondominated_sort(F, V)[0]
        nfeas = int((V[f0] <= 0).sum())
        if g == 1 or g % max(1, generations // 12) == 0 or g == generations:
            snapshots.append((g, (F[f0] * sg).copy()))
        if g == 1 or g % max(1, generations // 60) == 0 or g == generations:
            rows.append({"Generación": g, "Puntos en el frente": len(f0),
                         "Factibles en el frente": nfeas,
                         **{f"Mín f{k + 1}": float((F[f0] * sg)[:, k].min()) for k in range(len(objs))},
                         **{f"Máx f{k + 1}": float((F[f0] * sg)[:, k].max()) for k in range(len(objs))}})
    f0 = _nondominated_sort(F, V)[0]
    f0 = f0[V[f0] <= 0] if (V[f0] <= 0).any() else f0
    order = f0[np.argsort(F[f0, 0])]
    front, Xf = F[order] * sg, X[order]
    hv = None
    if len(objs) == 2:
        Fm = F[order]
        ref = Fm.max(0) + 0.1 * np.maximum(np.abs(Fm.max(0)), 1e-9)
        hv = hypervolume_2d(Fm, ref)
    for k, (fv, xv) in enumerate(zip(front, Xf)):
        rows.append({"Generación": "frente", "Puntos en el frente": k + 1,
                     "Factibles en el frente": "Sí" if V[order[k]] <= 0 else "No",
                     **{f"Mín f{j + 1}": float(fv[j]) for j in range(len(objs))},
                     **{f"Máx f{j + 1}": am._fmt_x(prob, xv) if j == 0 else ""
                        for j in range(len(objs))}})
    m = len(objs)
    labels = [f"f{k + 1} ({'máx' if s < 0 else 'mín'})" for k, (s, _) in enumerate(objs)]

    def plot(fig):
        ax = fig.add_subplot(111)
        sc = ax.scatter(front[:, 0], front[:, 1], c=(front[:, 2] if m >= 3 else "#ffd700"),
                        cmap="viridis" if m >= 3 else None, s=22, label="Frente de Pareto final")
        if m >= 3:
            fig.colorbar(sc, ax=ax, label=labels[2])
        for g_, S in snapshots[:-1]:
            ax.scatter(S[:, 0], S[:, 1], s=6, alpha=0.25, color="#4a9eff")
        ax.scatter([], [], s=6, color="#4a9eff", alpha=0.5, label="Frentes de generaciones previas")
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
        ax.set_title(f"NSGA-II: frente de Pareto ({len(front)} soluciones)"
                     + (f", hipervolumen {hv:.4g}" if hv is not None else ""))
        ax.legend(fontsize=7)

    def plot3d(fig):
        ax = fig.add_subplot(111, projection="3d")
        if m >= 3:
            ax.scatter(front[:, 0], front[:, 1], front[:, 2], c=front[:, 2], cmap="viridis", s=20,
                       label="Frente de Pareto")
            ax.set_zlabel(labels[2])
        else:
            for g_, S in snapshots:
                ax.scatter(S[:, 0], S[:, 1], np.full(len(S), g_), s=8,
                           color="#ffd700" if g_ == snapshots[-1][0] else "#4a9eff",
                           alpha=1.0 if g_ == snapshots[-1][0] else 0.4)
            ax.set_zlabel("generación")
            ax.scatter([], [], [], color="#ffd700", label="Frente final")
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
        ax.set_title("3D — NSGA-II", fontsize=10, color="#d8e8f8")
        ax.legend(fontsize=7, loc="upper left")

    note = ("Ejes: f1, f2 y f3 de cada solución no dominada final." if m >= 3 else
            "Ejes: f1, f2 y la generación: evolución real del frente no dominado.")
    return {"front": front.tolist(), "x": Xf.tolist(), "hypervolume": hv, "names": prob.names,
            "hist": rows, "plot": plot, "plot3d": plot3d, "note3d": note,
            "resumen": (f"NSGA-II: {len(front)} soluciones no dominadas"
                        + (f", hipervolumen {hv:.6g}" if hv is not None else ""))}


# ══════════════════════════════════════════════════════════════════════════════
#  2. Ruteo de vehículos con capacidad (CVRP)
# ══════════════════════════════════════════════════════════════════════════════
def _split(order, D, q, Q):
    """Split de Prins: partición óptima de la ruta gigante en viajes con carga ≤ Q."""
    n = len(order)
    V = np.full(n + 1, np.inf); V[0] = 0.0
    P = np.zeros(n + 1, int)
    for i in range(1, n + 1):
        load, cost, j = 0.0, 0.0, i
        while j <= n:
            c = order[j - 1]
            load += q[c]
            if load > Q:
                break
            cost = D[0, c] + D[c, 0] if j == i else cost - D[order[j - 2], 0] + D[order[j - 2], c] + D[c, 0]
            if V[i - 1] + cost < V[j]:
                V[j], P[j] = V[i - 1] + cost, i - 1
            j += 1
    routes, j = [], n
    while j > 0:
        routes.append([int(c) for c in order[P[j]:j]]); j = P[j]
    return V[n], routes[::-1]


def _route_len(D, r):
    return float(D[0, r[0]] + D[r[:-1], r[1:]].sum() + D[r[-1], 0]) if r else 0.0


def _improve_routes(D, routes):
    out = []
    for r in routes:
        if len(r) >= 3:
            t = am.two_opt(D, np.array([0] + r))
            k = int(np.where(t == 0)[0][0])
            r = [int(c) for c in np.roll(t, -k)[1:]]
        out.append(r)
    return out


def _vrp_cost(D, routes, vmax):
    c = sum(_route_len(D, r) for r in routes)
    return c + (1e6 * (len(routes) - vmax) if vmax and len(routes) > vmax else 0.0)


def _vrp_ga(D, q, Q, vmax, iters, pop, rng):
    n = len(q) - 1
    cust = np.arange(1, n + 1)

    def decode(perm):
        _, routes = _split(perm, D, q, Q)
        routes = _improve_routes(D, routes)
        return _vrp_cost(D, routes, vmax), routes
    Dc = D[1:, 1:]

    def polish_giant(perm):                       # 2-opt sobre la ruta gigante
        t = am.two_opt(Dc, np.asarray(perm) - 1, max_rounds=n)
        return t + 1

    nn = [0]                                      # vecino más cercano desde el depósito
    free = set(range(1, n + 1))
    while free:
        c = min(free, key=lambda x: D[nn[-1], x]); nn.append(c); free.remove(c)
    P = [polish_giant(nn[1:])] + [rng.permutation(cust) for _ in range(pop - 1)]
    sols = [decode(p) for p in P]
    rows, best = [], min(sols, key=lambda s: s[0])
    for it in range(1, iters + 1):
        def pick():
            a, b = rng.integers(0, pop, 2)
            return P[a] if sols[a][0] < sols[b][0] else P[b]
        order = np.argsort([s[0] for s in sols])
        newP = [P[order[0]].copy()]
        while len(newP) < pop:
            a, b = pick(), pick()
            i, k = sorted(rng.choice(n, 2, replace=False))
            child = np.zeros(n, int); child[i:k + 1] = a[i:k + 1]
            seg = set(a[i:k + 1].tolist())
            child[[x for x in range(n) if not (i <= x <= k)]] = [g for g in b if g not in seg]
            if rng.random() < 0.3:
                x, y = rng.choice(n, 2, replace=False); child[[x, y]] = child[[y, x]]
            newP.append(child)
        P = newP
        sols = [decode(p) for p in P]
        k = int(np.argsort([s_[0] for s_ in sols])[min(1, pop - 1)])   # mejor hijo nuevo
        P[k] = polish_giant(P[k]); sols[k] = decode(P[k])
        cur = min(sols, key=lambda s: s[0])
        if cur[0] < best[0]:
            best = cur
        rows.append((it, best[0], float(np.mean([s[0] for s in sols]))))
    return best[1], best[0], rows


def _vrp_sa(D, q, Q, vmax, iters, moves, rng):
    n = len(q) - 1
    perm = rng.permutation(np.arange(1, n + 1))
    cost = _vrp_cost(D, _split(perm, D, q, Q)[1], vmax)
    best_p, best_c, rows = perm.copy(), cost, []
    T0 = float(D[0, 1:].mean())
    moves = max(moves, 2 * n)
    for it in range(1, iters + 1):
        T = T0 * (1e-3 ** (it / iters))
        for _ in range(moves):
            y = perm.copy()
            i, k = sorted(rng.choice(n, 2, replace=False))
            r = rng.random()
            if r < 0.4:
                y[i:k + 1] = y[i:k + 1][::-1]
            elif r < 0.7:
                y[[i, k]] = y[[k, i]]
            else:
                y = np.insert(np.delete(y, i), k, y[i])
            cy = _vrp_cost(D, _split(y, D, q, Q)[1], vmax)
            if cy <= cost or rng.random() < math.exp(-(cy - cost) / max(T, 1e-12)):
                perm, cost = y, cy
                if cy < best_c:
                    best_p, best_c = y.copy(), cy
        rows.append((it, best_c, cost))
    routes = _improve_routes(D, _split(best_p, D, q, Q)[1])
    return routes, _vrp_cost(D, routes, vmax), rows


def solve_vrp(depot: Sequence[float], customers: Sequence[Sequence[float]], capacity: float,
              vehicles: int, algorithm: str, iterations: int, population: int,
              seed: int) -> Dict[str, Any]:
    _check_sizes(iterations, population)
    C = np.asarray(customers, dtype=float)
    if C.ndim != 2 or C.shape[1] != 3 or len(C) < 2:
        raise ValueError("Ingresa al menos 2 clientes, uno por línea: x, y, demanda.")
    if len(C) > 300:
        raise ValueError("El máximo admitido es 300 clientes.")
    if len(depot) != 2:
        raise ValueError("El depósito debe ser «x, y».")
    if capacity <= 0 or (C[:, 2] <= 0).any():
        raise ValueError("La capacidad y las demandas deben ser positivas.")
    if (C[:, 2] > capacity).any():
        raise ValueError("Hay clientes cuya demanda supera la capacidad de un vehículo.")
    if vehicles < 0:
        raise ValueError("El número de vehículos no puede ser negativo (0 = sin límite).")
    if vehicles and C[:, 2].sum() > vehicles * capacity:
        raise ValueError(f"La demanda total ({C[:, 2].sum():g}) supera la capacidad de la flota "
                         f"({vehicles} × {capacity:g}).")
    if algorithm not in VRP_ALGORITHMS:
        raise ValueError(f"Algoritmo desconocido: «{algorithm}».")
    pts = np.vstack([np.asarray(depot, float), C[:, :2]])
    D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    q = np.concatenate([[0.0], C[:, 2]])
    runners = {"Algoritmo Genético (Split de Prins)": _vrp_ga, "Recocido Simulado": _vrp_sa}
    names = list(runners) if algorithm == "Comparar todos" else [algorithm]
    res = {n: runners[n](D, q, capacity, vehicles, iterations, population,
                         np.random.default_rng(seed)) for n in names}
    best_name = min(names, key=lambda n: res[n][1])
    routes, cost, _ = res[best_name]
    feasible = not vehicles or len(routes) <= vehicles
    rows = []
    for n in names:
        r = res[n][2]
        step = max(1, len(r) // 60)
        for k, (it, b, c) in enumerate(r):
            if k % step == 0 or k == len(r) - 1:
                rows.append({"Iteración": it, **({"Algoritmo": n} if len(names) > 1 else {}),
                             "Mejor distancia total": b, "Distancia de la iteración": c})
    for k, r in enumerate(routes, 1):
        rows.append({"Iteración": f"Ruta {k}", **({"Algoritmo": best_name} if len(names) > 1 else {}),
                     "Mejor distancia total": _route_len(D, r),
                     "Distancia de la iteración": "Depósito → " + " → ".join(str(c) for c in r)
                     + f" → Depósito (carga {q[r].sum():g}/{capacity:g})"})

    def plot(fig):
        ax1, ax2 = fig.subplots(1, 2)
        for k, r in enumerate(routes):
            t = [0] + r + [0]
            ax1.plot(pts[t, 0], pts[t, 1], "-o", color=_PAL[k % len(_PAL)], ms=4, lw=1.4,
                     label=f"Ruta {k + 1} ({q[r].sum():g})")
        ax1.plot(*pts[0], "s", color="#ffffff", ms=10, label="Depósito")
        ax1.set_title(f"{len(routes)} rutas, distancia {cost:.5g}")
        ax1.legend(fontsize=6)
        for n in names:
            rr = np.array(res[n][2])
            ax2.plot(rr[:, 0], rr[:, 1], lw=1.6, label=n)
        ax2.set_xlabel("Iteración"); ax2.set_ylabel("Mejor distancia")
        ax2.set_title("Convergencia"); ax2.legend(fontsize=7)

    def plot3d(fig):
        ax = fig.add_subplot(111, projection="3d")
        for k, r in enumerate(routes):
            t = [0] + r + [0]
            load = np.concatenate([[0], np.cumsum(q[r]), [q[r].sum()]])
            ax.plot(pts[t, 0], pts[t, 1], load, "-o", color=_PAL[k % len(_PAL)], ms=3,
                    label=f"Ruta {k + 1}")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("carga acumulada")
        ax.set_title("3D — Rutas y carga entregada", fontsize=10, color="#d8e8f8")
        ax.legend(fontsize=6, loc="upper left")

    return {"routes": routes, "cost": cost, "feasible": feasible, "algorithm": best_name,
            "results": {n: res[n][1] for n in names}, "hist": rows, "plot": plot,
            "plot3d": plot3d,
            "note3d": "Ejes: coordenadas x, y del recorrido de cada vehículo y la carga "
                      "acumulada que lleva entregada en cada cliente.",
            "resumen": f"{best_name}: {len(routes)} rutas, distancia total {cost:.6g}"}


def brute_force_vrp(depot, customers, capacity) -> float:
    """Óptimo exacto (solo instancias pequeñas): todas las rutas gigantes + Split."""
    C = np.asarray(customers, float)
    pts = np.vstack([np.asarray(depot, float), C[:, :2]])
    D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    q = np.concatenate([[0.0], C[:, 2]])
    return min(_split(np.array(p), D, q, capacity)[0]
               for p in itertools.permutations(range(1, len(C) + 1)))


# ══════════════════════════════════════════════════════════════════════════════
#  3. Programación de tareas: flow shop y job shop
# ══════════════════════════════════════════════════════════════════════════════
def flow_makespan(P: np.ndarray, perm) -> float:
    c = np.zeros(P.shape[1])
    for j in perm:
        c[0] += P[j, 0]
        for m in range(1, P.shape[1]):
            c[m] = max(c[m], c[m - 1]) + P[j, m]
    return float(c[-1])


def _neh(P):
    order = list(np.argsort(-P.sum(1)))
    seq = [order[0]]
    for j in order[1:]:
        best = min(range(len(seq) + 1), key=lambda k: flow_makespan(P, seq[:k] + [j] + seq[k:]))
        seq.insert(best, j)
    return seq


def _flow_ig(P, iters, pop, rng, d=None):
    """Iterated Greedy (Ruiz & Stützle, 2007) partiendo de NEH."""
    n = len(P)
    d = d or max(2, min(4, n // 2))
    seq = _neh(P); cur = flow_makespan(P, seq)
    best, best_c, rows = seq[:], cur, []
    T = 0.4 * P.sum() / (n * P.shape[1] * 10)
    for it in range(1, iters + 1):
        s2 = seq[:]
        removed = [s2.pop(int(rng.integers(len(s2)))) for _ in range(min(d, n - 1))]
        for j in removed:
            k = min(range(len(s2) + 1), key=lambda k: flow_makespan(P, s2[:k] + [j] + s2[k:]))
            s2.insert(k, j)
        c2 = flow_makespan(P, s2)
        if c2 < cur or rng.random() < math.exp(-(c2 - cur) / max(T, 1e-9)):
            seq, cur = s2, c2
            if c2 < best_c:
                best, best_c = s2[:], c2
        rows.append((it, best_c, cur))
    return best, best_c, rows


def _flow_sa(P, iters, pop, rng):
    n = len(P)
    seq = list(rng.permutation(n)); cur = flow_makespan(P, seq)
    best, best_c, rows = seq[:], cur, []
    T0 = float(P.mean())
    for it in range(1, iters + 1):
        T = T0 * (1e-3 ** (it / iters))
        for _ in range(max(pop, n)):
            i, k = rng.choice(n, 2, replace=False)
            s2 = seq[:]; s2.insert(k, s2.pop(i))
            c2 = flow_makespan(P, s2)
            if c2 <= cur or rng.random() < math.exp(-(c2 - cur) / max(T, 1e-12)):
                seq, cur = s2, c2
                if c2 < best_c:
                    best, best_c = s2[:], c2
        rows.append((it, best_c, cur))
    return best, best_c, rows


def _flow_ga(P, iters, pop, rng):
    n = len(P)
    popl = [list(rng.permutation(n)) for _ in range(pop - 1)] + [_neh(P)]
    fit = [flow_makespan(P, s) for s in popl]
    rows = []
    for it in range(1, iters + 1):
        order = np.argsort(fit)
        new = [popl[order[0]][:]]
        while len(new) < pop:
            a = popl[min(rng.integers(0, pop, 2), key=lambda j: fit[j])]
            b = popl[min(rng.integers(0, pop, 2), key=lambda j: fit[j])]
            i, k = sorted(rng.choice(n, 2, replace=False))
            mid = a[i:k + 1]
            rest = [g for g in b if g not in mid]
            child = rest[:i] + mid + rest[i:]
            if rng.random() < 0.3:
                x, y = rng.choice(n, 2, replace=False); child.insert(y, child.pop(x))
            new.append(child)
        popl = new
        fit = [flow_makespan(P, s) for s in popl]
        rows.append((it, float(min(fit)), float(np.mean(fit))))
    i = int(np.argmin(fit))
    return popl[i], fit[i], rows


def _flow_schedule(P, seq):
    m = P.shape[1]
    c = np.zeros(m); ops = []
    for j in seq:
        prev = 0.0
        for k in range(m):
            start = max(c[k], prev)
            c[k] = prev = start + P[j, k]
            ops.append((int(j), k, start, c[k]))
    return ops


def _job_decode(jobs, seq):
    """Decodifica una secuencia con repetición en un programa semiactivo."""
    nm = 1 + max(m for job in jobs for m, _ in job)
    mready, jready = np.zeros(nm), np.zeros(len(jobs))
    nxt = [0] * len(jobs); ops = []
    for j in seq:
        m, t = jobs[j][nxt[j]]
        s = max(mready[m], jready[j])
        mready[m] = jready[j] = s + t
        ops.append((int(j), int(m), float(s), float(s + t)))
        nxt[j] += 1
    return float(jready.max()), ops


def _job_ga(jobs, iters, pop, rng):
    base = np.array([j for j, job in enumerate(jobs) for _ in job])
    n_j = len(jobs)
    P = [rng.permutation(base) for _ in range(pop)]
    fit = [_job_decode(jobs, s)[0] for s in P]
    rows = []

    def pox(a, b):
        keep = set(rng.choice(n_j, max(1, n_j // 2), replace=False).tolist())
        fill = iter([g for g in b if g not in keep])
        return np.array([g if g in keep else next(fill) for g in a])
    for it in range(1, iters + 1):
        order = np.argsort(fit)
        new = [P[order[0]].copy(), P[order[1]].copy()]
        while len(new) < pop:
            a = P[min(rng.integers(0, pop, 3), key=lambda j: fit[j])]
            b = P[min(rng.integers(0, pop, 3), key=lambda j: fit[j])]
            c = pox(a, b)
            if rng.random() < 0.4:
                x, y = rng.choice(len(c), 2, replace=False); c[[x, y]] = c[[y, x]]
            new.append(c)
        P = new
        fit = [_job_decode(jobs, s)[0] for s in P]
        rows.append((it, float(min(fit)), float(np.mean(fit))))
    i = int(np.argmin(fit))
    return P[i], fit[i], rows


def _job_sa(jobs, iters, pop, rng):
    seq = rng.permutation(np.array([j for j, job in enumerate(jobs) for _ in job]))
    cur = _job_decode(jobs, seq)[0]
    best, best_c, rows = seq.copy(), cur, []
    T0 = float(np.mean([t for job in jobs for _, t in job]))
    L = len(seq)
    for it in range(1, iters + 1):
        T = T0 * (1e-3 ** (it / iters))
        for _ in range(max(pop, L)):
            y = seq.copy()
            i, k = rng.choice(L, 2, replace=False)
            if rng.random() < 0.5:
                y[[i, k]] = y[[k, i]]
            else:
                y = np.insert(np.delete(y, i), k, y[i])
            c2 = _job_decode(jobs, y)[0]
            if c2 <= cur or rng.random() < math.exp(-(c2 - cur) / max(T, 1e-12)):
                seq, cur = y, c2
                if c2 < best_c:
                    best, best_c = y.copy(), c2
        rows.append((it, best_c, cur))
    return best, best_c, rows


def _parse_job_shop(text: str) -> List[List[Tuple[int, float]]]:
    jobs = []
    for ln in (text or "").splitlines():
        if not ln.strip():
            continue
        nums = [x for x in re.split(r"[,;\s]+", ln.strip()) if x]
        if len(nums) % 2:
            raise ValueError(f"Trabajo inválido: «{ln.strip()}». Usa pares «máquina tiempo» "
                             "(máquinas numeradas desde 1), ej.: 1 3, 2 5, 3 2.")
        try:
            pairs = [(int(float(nums[i])) - 1, float(nums[i + 1])) for i in range(0, len(nums), 2)]
        except ValueError as ex:
            raise ValueError(f"Hay un valor no numérico en «{ln.strip()}».") from ex
        if any(m < 0 or t < 0 for m, t in pairs):
            raise ValueError("Las máquinas se numeran desde 1 y los tiempos no pueden ser negativos.")
        jobs.append(pairs)
    if len(jobs) < 2:
        raise ValueError("Ingresa al menos 2 trabajos, uno por línea.")
    return jobs


def solve_scheduling(kind: str, data: str, algorithm: str, iterations: int, population: int,
                     seed: int) -> Dict[str, Any]:
    _check_sizes(iterations, population)
    flow = kind.lower().startswith("flow")
    if flow:
        rows_ = [[float(x) for x in re.split(r"[,;\s]+", ln.strip()) if x]
                 for ln in (data or "").splitlines() if ln.strip()]
        if len(rows_) < 2 or len({len(r) for r in rows_}) != 1:
            raise ValueError("Flow shop: una fila por trabajo con el tiempo en cada máquina "
                             "(todas las filas con la misma cantidad de máquinas).")
        P = np.asarray(rows_)
        if (P < 0).any():
            raise ValueError("Los tiempos de proceso no pueden ser negativos.")
        algos = {"Iterated Greedy (NEH)": _flow_ig, "Recocido Simulado": _flow_sa,
                 "Algoritmo Genético": _flow_ga}
        allowed = FLOW_ALGORITHMS
    else:
        jobs = _parse_job_shop(data)
        algos = {"Algoritmo Genético (POX)": _job_ga, "Recocido Simulado": _job_sa}
        allowed = JOB_ALGORITHMS
    if algorithm not in allowed:
        raise ValueError(f"«{algorithm}» no aplica a {'flow shop' if flow else 'job shop'}; "
                         f"usa uno de: {', '.join(allowed)}.")
    names = list(algos) if algorithm == "Comparar todos" else [algorithm]
    res = {n: algos[n](P if flow else jobs, iterations, population,
                       np.random.default_rng(seed)) for n in names}
    best_name = min(names, key=lambda n: res[n][1])
    seq, mk, _ = res[best_name]
    ops = _flow_schedule(P, seq) if flow else _job_decode(jobs, seq)[1]
    nm = (P.shape[1] if flow else 1 + max(m for job in jobs for m, _ in job))
    rows = []
    for n in names:
        r = res[n][2]
        step = max(1, len(r) // 60)
        for k, (it, b, c) in enumerate(r):
            if k % step == 0 or k == len(r) - 1:
                rows.append({"Iteración": it, **({"Algoritmo": n} if len(names) > 1 else {}),
                             "Mejor makespan": b, "Makespan de la iteración": c})
    for j, m, s, e in sorted(ops, key=lambda o: (o[1], o[2])):
        rows.append({"Iteración": f"Máquina {m + 1}", **({"Algoritmo": best_name} if len(names) > 1 else {}),
                     "Mejor makespan": f"Trabajo {j + 1}",
                     "Makespan de la iteración": f"{s:g} → {e:g}"})

    def plot(fig):
        ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
        for j, m, s, e in ops:
            ax1.barh(m, e - s, left=s, color=_PAL[j % len(_PAL)], edgecolor="#0a1228")
            if e - s > 0:
                ax1.text((s + e) / 2, m, str(j + 1), ha="center", va="center", fontsize=7,
                         color="#0a1228")
        ax1.set_yticks(range(nm)); ax1.set_yticklabels([f"M{m + 1}" for m in range(nm)])
        ax1.invert_yaxis(); ax1.set_xlabel("tiempo")
        ax1.set_title(f"Gantt ({best_name}) — makespan {mk:g}")
        for n in names:
            rr = np.array(res[n][2])
            ax2.plot(rr[:, 0], rr[:, 1], lw=1.6, label=n)
        ax2.set_xlabel("Iteración"); ax2.set_ylabel("Mejor makespan")
        ax2.set_title("Convergencia"); ax2.legend(fontsize=7)

    def plot3d(fig):
        ax = fig.add_subplot(111, projection="3d")
        for j, m, s, e in ops:
            ax.bar3d(s, m - 0.3, j - 0.3, max(e - s, 1e-9), 0.6, 0.6,
                     color=_PAL[j % len(_PAL)], alpha=0.85)
        ax.set_xlabel("tiempo"); ax.set_ylabel("máquina"); ax.set_zlabel("trabajo")
        ax.set_title("3D — Programa de tareas", fontsize=10, color="#d8e8f8")

    return {"sequence": [int(x) for x in seq], "makespan": mk, "algorithm": best_name,
            "operations": ops, "results": {n: res[n][1] for n in names}, "hist": rows,
            "plot": plot, "plot3d": plot3d,
            "note3d": "Ejes: tiempo, máquina y trabajo; cada bloque es una operación con su "
                      "inicio y su fin reales en el programa encontrado.",
            "resumen": f"{best_name}: makespan {mk:g} ({'flow shop' if flow else 'job shop'})"}


FT06 = ("3 1, 1 3, 2 6, 4 7, 6 3, 5 6\n2 8, 3 5, 5 10, 6 10, 1 10, 4 4\n"
        "3 5, 4 4, 6 8, 1 9, 2 1, 5 7\n2 5, 1 5, 3 5, 4 3, 5 8, 6 9\n"
        "3 9, 2 3, 5 5, 6 4, 1 3, 4 1\n2 3, 4 3, 6 9, 1 10, 5 4, 3 1")
