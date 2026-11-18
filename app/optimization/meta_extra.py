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
#  2. Ruteo de vehículos: capacidad, ventanas horarias y flota heterogénea
# ══════════════════════════════════════════════════════════════════════════════
class VRPModel:
    """Datos del VRP. Nodo 0 = depósito. Tiempo de viaje = distancia (velocidad 1).
    types: [(capacidad, costo fijo, costo por distancia, cantidad (0 = ilimitada))]."""

    def __init__(self, pts, q, tw, svc, types):
        self.pts = pts
        self.D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
        self.q, self.tw, self.svc, self.types = q, tw, svc, types
        self.maxcap = max(t[0] for t in types)
        self.has_tw = bool(np.isfinite(tw[:, 1]).any() or (tw[:, 0] > 0).any())

    def route_eval(self, r):
        """(distancia, carga, factible) de una ruta depósito → r → depósito."""
        D, tw, svc = self.D, self.tw, self.svc
        t, prev, dist, load = tw[0, 0], 0, 0.0, 0.0
        for c in r:
            dist += D[prev, c]; t += D[prev, c]
            if t > tw[c, 1] + 1e-9:
                return dist, load, False
            t = max(t, tw[c, 0]) + svc[c]
            load += self.q[c]; prev = c
        dist += D[prev, 0]; t += D[prev, 0]
        return dist, load, (load <= self.maxcap + 1e-9 and t <= tw[0, 1] + 1e-9)

    def route_cost(self, dist, load):
        fits = [f + v * dist for cap, f, v, _ in self.types if cap + 1e-9 >= load]
        return min(fits) if fits else np.inf

    def split(self, order):
        """Split de Prins con ventanas horarias y costo por tipo de vehículo."""
        n = len(order)
        D, tw, svc, q = self.D, self.tw, self.svc, self.q
        V = np.full(n + 1, np.inf); V[0] = 0.0
        P = np.zeros(n + 1, int)
        for i in range(1, n + 1):
            load = dist = 0.0
            t = tw[0, 0]
            prev = 0
            for j in range(i, n + 1):
                c = order[j - 1]
                load += q[c]
                if load > self.maxcap + 1e-9:
                    break
                dist += D[prev, c]; t += D[prev, c]
                if t > tw[c, 1] + 1e-9:
                    break
                t = max(t, tw[c, 0]) + svc[c]; prev = c
                if t + D[c, 0] > tw[0, 1] + 1e-9:
                    continue
                cost = self.route_cost(dist + D[c, 0], load)
                if V[i - 1] + cost < V[j]:
                    V[j], P[j] = V[i - 1] + cost, i - 1
        if not np.isfinite(V[n]):
            return np.inf, []
        routes, j = [], n
        while j > 0:
            routes.append([int(c) for c in order[P[j]:j]]); j = P[j]
        return V[n], routes[::-1]

    def assign(self, routes):
        """Asigna tipos respetando las cantidades (mayores cargas primero, tipo
        más barato que cabe). Devuelve (costo total, [tipo por ruta], rutas sin vehículo)."""
        left = [cnt if cnt > 0 else 10 ** 9 for *_, cnt in self.types]
        out, total, missing = [None] * len(routes), 0.0, 0
        info = [self.route_eval(r) for r in routes]
        for k in sorted(range(len(routes)), key=lambda k: -info[k][1]):
            dist, load, _ = info[k]
            opts = [(f + v * dist, ti) for ti, (cap, f, v, _) in enumerate(self.types)
                    if cap + 1e-9 >= load and left[ti] > 0]
            if not opts:
                missing += 1
                total += 1e6 + dist
                continue
            c, ti = min(opts)
            left[ti] -= 1; out[k] = ti; total += c
        return total, out, missing

    def improve(self, routes):
        """2-opt dentro de cada ruta; se acepta solo si sigue siendo factible y mejora."""
        out = []
        for r in routes:
            if len(r) >= 3:
                t = am.two_opt(self.D, np.array([0] + r))
                k = int(np.where(t == 0)[0][0])
                r2 = [int(c) for c in np.roll(t, -k)[1:]]
                d1, l1, ok1 = self.route_eval(r)
                d2, l2, ok2 = self.route_eval(r2)
                if ok2 and d2 < d1 - 1e-12:
                    r = r2
            out.append(r)
        return out

    def decode(self, perm):
        c, routes = self.split(perm)
        if not routes:
            return 1e12, []
        routes = self.improve(routes)
        return self.assign(routes)[0], routes


def _vrp_ga(M: VRPModel, iters, pop, rng):
    n = len(M.q) - 1
    cust = np.arange(1, n + 1)
    Dc = M.D[1:, 1:]

    def polish_giant(perm):
        return am.two_opt(Dc, np.asarray(perm) - 1, max_rounds=n) + 1

    starts = []
    nn, free = [0], set(range(1, n + 1))
    while free:
        c = min(free, key=lambda x: M.D[nn[-1], x]); nn.append(c); free.remove(c)
    starts.append(polish_giant(nn[1:]))
    if M.has_tw:                                   # orden por cierre de ventana (EDD)
        starts.append(np.array(sorted(cust, key=lambda c: (M.tw[c, 1], M.tw[c, 0]))))
    P = starts + [rng.permutation(cust) for _ in range(pop - len(starts))]
    sols = [M.decode(p) for p in P]
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
        sols = [M.decode(p) for p in P]
        if not M.has_tw:
            k = int(np.argsort([s_[0] for s_ in sols])[min(1, pop - 1)])
            P[k] = polish_giant(P[k]); sols[k] = M.decode(P[k])
        cur = min(sols, key=lambda s: s[0])
        if cur[0] < best[0]:
            best = cur
        rows.append((it, best[0], float(np.mean([min(s[0], 1e12) for s in sols]))))
    return best[1], best[0], rows


def _vrp_sa(M: VRPModel, iters, moves, rng):
    n = len(M.q) - 1
    cust = np.arange(1, n + 1)
    perm = (np.array(sorted(cust, key=lambda c: M.tw[c, 1])) if M.has_tw
            else rng.permutation(cust))
    cost = M.decode(perm)[0]
    best_p, best_c, rows = perm.copy(), cost, []
    T0 = float(M.D[0, 1:].mean())
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
            cy = M.decode(y)[0]
            if cy <= cost or rng.random() < math.exp(-min(cy - cost, 1e6) / max(T, 1e-12)):
                perm, cost = y, cy
                if cy < best_c:
                    best_p, best_c = y.copy(), cy
        rows.append((it, best_c, min(cost, 1e12)))
    c, routes = M.decode(best_p)
    return routes, c, rows


def _parse_vehicle_types(text: str, capacity: float, vehicles: int):
    types = []
    for ln in (text or "").splitlines():
        if not ln.strip():
            continue
        nums = [x for x in re.split(r"[,;\s]+", ln.strip()) if x]
        try:
            vals = [float(x) for x in nums]
        except ValueError as ex:
            raise ValueError(f"Tipo de vehículo inválido: «{ln.strip()}». Formato: capacidad, "
                             "costo fijo, costo por distancia, cantidad (0 = ilimitada).") from ex
        if len(vals) not in (1, 2, 3, 4):
            raise ValueError(f"Tipo de vehículo inválido: «{ln.strip()}».")
        cap = vals[0]
        fixed = vals[1] if len(vals) > 1 else 0.0
        var = vals[2] if len(vals) > 2 else 1.0
        cnt = vals[3] if len(vals) > 3 else 0.0
        if cap <= 0 or fixed < 0 or var < 0 or cnt < 0 or int(cnt) != cnt:
            raise ValueError(f"Valores no válidos en el tipo de vehículo «{ln.strip()}».")
        types.append((cap, fixed, var, int(cnt)))
    if not types:
        if capacity <= 0:
            raise ValueError("La capacidad debe ser positiva.")
        types = [(float(capacity), 0.0, 1.0, int(vehicles))]
    return types


def solve_vrp(depot: Sequence[float], customers: Sequence[Sequence[float]], capacity: float,
              vehicles: int, algorithm: str, iterations: int, population: int,
              seed: int, vehicle_types: str = "") -> Dict[str, Any]:
    _check_sizes(iterations, population)
    C = np.asarray(customers, dtype=float)
    if C.ndim != 2 or C.shape[1] not in (3, 6) or len(C) < 2:
        raise ValueError("Ingresa al menos 2 clientes, uno por línea: «x, y, demanda» o, con "
                         "ventanas horarias, «x, y, demanda, inicio, fin, servicio».")
    if len(C) > 300:
        raise ValueError("El máximo admitido es 300 clientes.")
    if len(depot) not in (2, 4):
        raise ValueError("El depósito debe ser «x, y» o «x, y, apertura, cierre».")
    if vehicles < 0:
        raise ValueError("El número de vehículos no puede ser negativo (0 = sin límite).")
    types = _parse_vehicle_types(vehicle_types, capacity, vehicles)
    if (C[:, 2] <= 0).any():
        raise ValueError("Las demandas deben ser positivas.")
    maxcap = max(t[0] for t in types)
    if (C[:, 2] > maxcap).any():
        raise ValueError("Hay clientes cuya demanda supera la capacidad de un vehículo.")
    if all(t[3] > 0 for t in types) and C[:, 2].sum() > sum(t[0] * t[3] for t in types):
        raise ValueError(f"La demanda total ({C[:, 2].sum():g}) supera la capacidad de la flota "
                         f"({sum(t[0] * t[3] for t in types):g}).")
    if algorithm not in VRP_ALGORITHMS:
        raise ValueError(f"Algoritmo desconocido: «{algorithm}».")
    n = len(C)
    pts = np.vstack([np.asarray(depot[:2], float), C[:, :2]])
    q = np.concatenate([[0.0], C[:, 2]])
    tw = np.zeros((n + 1, 2)); tw[:, 1] = np.inf
    svc = np.zeros(n + 1)
    if len(depot) == 4:
        tw[0] = depot[2:]
    if C.shape[1] == 6:
        tw[1:] = C[:, 3:5]; svc[1:] = C[:, 5]
        if (tw[1:, 0] > tw[1:, 1]).any() or (svc < 0).any():
            raise ValueError("Cada ventana debe tener inicio ≤ fin y el servicio no puede ser negativo.")
    M = VRPModel(pts, q, tw, svc, types)
    unreachable = [c for c in range(1, n + 1) if not M.route_eval([c])[2]]
    if unreachable:
        raise ValueError("Ningún vehículo puede atender a tiempo a los clientes "
                         + ", ".join(map(str, unreachable)) + " (revisa sus ventanas horarias "
                         "y el horario del depósito).")
    runners = {"Algoritmo Genético (Split de Prins)": _vrp_ga, "Recocido Simulado": _vrp_sa}
    names = list(runners) if algorithm == "Comparar todos" else [algorithm]
    res = {nm: runners[nm](M, iterations, population, np.random.default_rng(seed)) for nm in names}
    best_name = min(names, key=lambda nm: res[nm][1])
    routes, cost, _ = res[best_name]
    total, assigned, missing = M.assign(routes)
    feasible = missing == 0 and np.isfinite(total)
    dist_total = sum(M.route_eval(r)[0] for r in routes)
    rows = []
    for nm in names:
        r = res[nm][2]
        step = max(1, len(r) // 60)
        for k, (it, b, c) in enumerate(r):
            if k % step == 0 or k == len(r) - 1:
                rows.append({"Iteración": it, **({"Algoritmo": nm} if len(names) > 1 else {}),
                             "Mejor costo total": b, "Costo de la iteración": c})
    for k, r in enumerate(routes, 1):
        dist, load, ok = M.route_eval(r)
        ti = assigned[k - 1]
        veh = (f"vehículo tipo {ti + 1} (cap. {types[ti][0]:g})" if ti is not None
               else "SIN VEHÍCULO DISPONIBLE")
        rows.append({"Iteración": f"Ruta {k}", **({"Algoritmo": best_name} if len(names) > 1 else {}),
                     "Mejor costo total": dist,
                     "Costo de la iteración": "Depósito → " + " → ".join(map(str, r))
                     + f" → Depósito · carga {load:g} · {veh}"
                     + ("" if ok else " · INFACTIBLE")})

    def plot(fig):
        ax1, ax2 = fig.subplots(1, 2)
        for k, r in enumerate(routes):
            t = [0] + r + [0]
            ax1.plot(pts[t, 0], pts[t, 1], "-o", color=_PAL[k % len(_PAL)], ms=4, lw=1.4,
                     label=f"Ruta {k + 1} ({q[r].sum():g})")
        ax1.plot(*pts[0], "s", color="#ffffff", ms=10, label="Depósito")
        ax1.set_title(f"{len(routes)} rutas, costo {total:.5g}")
        ax1.legend(fontsize=6)
        for nm in names:
            rr = np.array(res[nm][2])
            ax2.plot(rr[:, 0], np.minimum(rr[:, 1], 1e12), lw=1.6, label=nm)
        ax2.set_xlabel("Iteración"); ax2.set_ylabel("Mejor costo")
        ax2.set_title("Convergencia"); ax2.legend(fontsize=7)

    def plot3d(fig):
        ax = fig.add_subplot(111, projection="3d")
        for k, r in enumerate(routes):
            t = [0] + r + [0]
            if M.has_tw:                                     # z = hora de llegada real
                z, tt, prev = [tw[0, 0]], tw[0, 0], 0
                for c in r:
                    tt += M.D[prev, c]; z.append(tt); tt = max(tt, tw[c, 0]) + svc[c]; prev = c
                z.append(tt + M.D[prev, 0])
            else:
                z = np.concatenate([[0], np.cumsum(q[r]), [q[r].sum()]])
            ax.plot(pts[t, 0], pts[t, 1], z, "-o", color=_PAL[k % len(_PAL)], ms=3,
                    label=f"Ruta {k + 1}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_zlabel("hora de llegada" if M.has_tw else "carga acumulada")
        ax.set_title("3D — Rutas", fontsize=10, color="#d8e8f8")
        ax.legend(fontsize=6, loc="upper left")

    return {"routes": routes, "cost": total, "distance": dist_total, "feasible": feasible,
            "vehicle_types": assigned, "algorithm": best_name,
            "results": {nm: res[nm][1] for nm in names}, "hist": rows, "plot": plot,
            "plot3d": plot3d,
            "note3d": ("Ejes: coordenadas x, y del recorrido y la hora real de llegada a cada "
                       "cliente." if M.has_tw else "Ejes: coordenadas x, y del recorrido y la "
                       "carga acumulada entregada."),
            "resumen": (f"{best_name}: {len(routes)} rutas, costo {total:.6g} "
                        f"(distancia {dist_total:.6g})" + ("" if feasible else " — FLOTA INSUFICIENTE"))}


def brute_force_vrp(depot, customers, capacity, vehicle_types: str = "") -> float:
    """Óptimo exacto (instancias pequeñas, flota ilimitada): todas las rutas gigantes + Split."""
    C = np.asarray(customers, float)
    n = len(C)
    pts = np.vstack([np.asarray(depot[:2], float), C[:, :2]])
    q = np.concatenate([[0.0], C[:, 2]])
    tw = np.zeros((n + 1, 2)); tw[:, 1] = np.inf
    svc = np.zeros(n + 1)
    if len(depot) == 4:
        tw[0] = depot[2:]
    if C.shape[1] == 6:
        tw[1:] = C[:, 3:5]; svc[1:] = C[:, 5]
    M = VRPModel(pts, q, tw, svc, _parse_vehicle_types(vehicle_types, capacity, 0))
    return min(M.split(np.array(p))[0] for p in itertools.permutations(range(1, n + 1)))


# ══════════════════════════════════════════════════════════════════════════════
#  3. Programación de tareas: flow shop y job shop
#     (preparación dependiente de la secuencia, fechas de entrega y pesos)
# ══════════════════════════════════════════════════════════════════════════════
OBJECTIVES = ("Makespan (Cmax)", "Tardanza total", "Tardanza ponderada", "Retraso máximo (Lmax)")


class ShopModel:
    def __init__(self, flow: bool, P=None, jobs=None, setup=None, due=None, weights=None,
                 objective="Makespan (Cmax)"):
        self.flow, self.P, self.jobs = flow, P, jobs
        self.n = len(P) if flow else len(jobs)
        self.nm = P.shape[1] if flow else 1 + max(m for job in jobs for m, _ in job)
        self.S = setup
        self.due = due
        self.w = weights if weights is not None else np.ones(self.n)
        if objective not in OBJECTIVES:
            raise ValueError(f"Objetivo desconocido: «{objective}».")
        if objective != OBJECTIVES[0] and due is None:
            raise ValueError("Para minimizar tardanza o retraso ingresa las fechas de entrega "
                             "(una por trabajo).")
        self.objective = objective

    def schedule(self, seq):
        """(operaciones [(trabajo, máquina, inicio, fin)], tiempos de término por trabajo)."""
        ops, comp = [], np.zeros(self.n)
        S = self.S
        if self.flow:
            c = np.zeros(self.nm); last = -1
            for j in seq:
                prev = 0.0
                for m in range(self.nm):
                    su = S[last, j] if (S is not None and last >= 0) else 0.0
                    start = max(c[m] + su, prev)
                    c[m] = prev = start + self.P[j, m]
                    ops.append((int(j), m, start, c[m]))
                comp[j] = prev; last = j
        else:
            mready = np.zeros(self.nm); jready = np.zeros(self.n)
            mlast = [-1] * self.nm; nxt = [0] * self.n
            for j in seq:
                m, t = self.jobs[j][nxt[j]]
                su = S[mlast[m], j] if (S is not None and mlast[m] >= 0) else 0.0
                s = max(mready[m] + su, jready[j])
                mready[m] = jready[j] = s + t
                mlast[m] = j; nxt[j] += 1
                ops.append((int(j), int(m), float(s), float(s + t)))
            comp = jready.copy()
        return ops, comp

    def value(self, seq) -> float:
        _, comp = self.schedule(seq)
        if self.objective == OBJECTIVES[0]:
            return float(comp.max())
        late = comp - self.due
        if self.objective == OBJECTIVES[1]:
            return float(np.maximum(late, 0).sum())
        if self.objective == OBJECTIVES[2]:
            return float((self.w * np.maximum(late, 0)).sum())
        return float(late.max())

    def base_sequence(self):
        return (np.arange(self.n) if self.flow else
                np.array([j for j, job in enumerate(self.jobs) for _ in job]))


def flow_makespan(P: np.ndarray, perm) -> float:
    return ShopModel(True, P=np.asarray(P, float)).value(list(perm))


def _neh(M: ShopModel):
    if M.due is not None and M.objective != OBJECTIVES[0]:
        order = list(np.argsort(M.due))                       # EDD
    else:
        order = list(np.argsort(-M.P.sum(1)))
    seq = [order[0]]
    for j in order[1:]:
        best = min(range(len(seq) + 1), key=lambda k: M.value(seq[:k] + [j] + seq[k:]))
        seq.insert(best, j)
    return seq


def _flow_ig(M, iters, pop, rng):
    """Iterated Greedy (Ruiz & Stützle, 2007) partiendo de NEH / EDD."""
    n = M.n
    d = max(2, min(4, n // 2))
    seq = _neh(M); cur = M.value(seq)
    best, best_c, rows = seq[:], cur, []
    T = 0.4 * M.P.sum() / (n * M.nm * 10)
    for it in range(1, iters + 1):
        s2 = seq[:]
        removed = [s2.pop(int(rng.integers(len(s2)))) for _ in range(min(d, n - 1))]
        for j in removed:
            k = min(range(len(s2) + 1), key=lambda k: M.value(s2[:k] + [j] + s2[k:]))
            s2.insert(k, j)
        c2 = M.value(s2)
        if c2 < cur or rng.random() < math.exp(-(c2 - cur) / max(T, 1e-9)):
            seq, cur = s2, c2
            if c2 < best_c:
                best, best_c = s2[:], c2
        rows.append((it, best_c, cur))
    return best, best_c, rows


def _perm_sa(M, iters, pop, rng):
    seq = list(rng.permutation(M.base_sequence()))
    L = len(seq)
    cur = M.value(seq)
    best, best_c, rows = seq[:], cur, []
    T0 = float(np.mean(M.P) if M.flow else np.mean([t for job in M.jobs for _, t in job])) or 1.0
    for it in range(1, iters + 1):
        T = T0 * (1e-3 ** (it / iters))
        for _ in range(max(pop, L)):
            s2 = seq[:]
            i, k = rng.choice(L, 2, replace=False)
            if rng.random() < 0.5:
                s2[i], s2[k] = s2[k], s2[i]
            else:
                s2.insert(k, s2.pop(i))
            c2 = M.value(s2)
            if c2 <= cur or rng.random() < math.exp(-(c2 - cur) / max(T, 1e-12)):
                seq, cur = s2, c2
                if c2 < best_c:
                    best, best_c = s2[:], c2
        rows.append((it, best_c, cur))
    return best, best_c, rows


def _perm_ga(M, iters, pop, rng):
    """GA: cruce de orden (flow shop) o POX (job shop, secuencias con repetición)."""
    base = M.base_sequence()
    L = len(base)
    P = [list(rng.permutation(base)) for _ in range(pop - 1)]
    P.append(_neh(M) if M.flow else list(rng.permutation(base)))
    fit = [M.value(s) for s in P]
    rows = []

    def cross(a, b):
        if M.flow:
            i, k = sorted(rng.choice(L, 2, replace=False))
            mid = a[i:k + 1]
            rest = [g for g in b if g not in mid]
            return rest[:i] + mid + rest[i:]
        keep = set(rng.choice(M.n, max(1, M.n // 2), replace=False).tolist())
        fill = iter([g for g in b if g not in keep])
        return [g if g in keep else next(fill) for g in a]
    for it in range(1, iters + 1):
        order = np.argsort(fit)
        new = [P[order[0]][:], P[order[min(1, pop - 1)]][:]]
        while len(new) < pop:
            a = P[min(rng.integers(0, pop, 3), key=lambda j: fit[j])]
            b = P[min(rng.integers(0, pop, 3), key=lambda j: fit[j])]
            c = cross(a, b)
            if rng.random() < 0.35:
                x, y = rng.choice(L, 2, replace=False)
                if M.flow:
                    c.insert(y, c.pop(x))
                else:
                    c[x], c[y] = c[y], c[x]
            new.append(c)
        # diversidad: sin duplicados e inmigrantes aleatorios (~15 % de la población)
        seen, uniq = set(), []
        for c in new:
            key = tuple(c)
            if key not in seen:
                seen.add(key); uniq.append(c)
        n_imm = max(1, pop // 7)
        while len(uniq) < pop:
            uniq.append(list(rng.permutation(base)))
        for k in range(1, n_imm + 1):
            uniq[-k] = list(rng.permutation(base))
        P = uniq[:pop]
        fit = [M.value(s) for s in P]
        srt = np.argsort(fit)
        for k in {int(srt[0]), int(srt[min(2, pop - 1)])}:  # élite y mejor hijo → descenso local
            P[k], fit[k] = _local_descent(M, P[k], fit[k], rng, 2 * L)
        rows.append((it, float(min(fit)), float(np.mean(fit))))
    i = int(np.argmin(fit))
    return P[i], fit[i], rows


def _local_descent(M, seq, val, rng, tries):
    """Descenso local: en flow shop recorre todo el vecindario de inserción hasta
    un óptimo local; en job shop, intercambios/inserciones aleatorios (primera mejora)."""
    seq = list(seq)
    L = len(seq)
    if M.flow and L <= 60:
        improved = True
        while improved:
            improved = False
            for i in range(L):
                for k in range(L):
                    if i == k:
                        continue
                    for move in ("insert", "swap") if i < k else ("insert",):
                        s2 = seq[:]
                        if move == "insert":
                            s2.insert(k, s2.pop(i))
                        else:
                            s2[i], s2[k] = s2[k], s2[i]
                        v2 = M.value(s2)
                        if v2 < val - 1e-12:
                            seq, val, improved = s2, v2, True
        return seq, val
    for _ in range(tries):
        s2 = seq[:]
        i, k = rng.choice(L, 2, replace=False)
        if rng.random() < 0.5:
            s2[i], s2[k] = s2[k], s2[i]
        else:
            s2.insert(k, s2.pop(i))
        v2 = M.value(s2)
        if v2 < val:
            seq, val = s2, v2
    return seq, val


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


def _vector(text: str, n: int, name: str):
    t = (text or "").strip()
    if not t:
        return None
    vals = [x for x in re.split(r"[,;\s]+", t) if x]
    try:
        v = np.array([float(x) for x in vals])
    except ValueError as ex:
        raise ValueError(f"Hay un valor no numérico en {name}.") from ex
    if len(v) != n:
        raise ValueError(f"{name[0].upper() + name[1:]} debe tener {n} valores (uno por trabajo); "
                         f"se leyeron {len(v)}.")
    return v


def _setup_matrix(text: str, n: int):
    t = (text or "").strip()
    if not t:
        return None
    rows = [[float(x) for x in re.split(r"[,;\s]+", ln.strip()) if x]
            for ln in t.splitlines() if ln.strip()]
    if len(rows) != n or any(len(r) != n for r in rows):
        raise ValueError(f"La matriz de preparación debe ser {n}×{n} (fila i = preparación al "
                         "pasar del trabajo i a cada trabajo j).")
    S = np.asarray(rows)
    if (S < 0).any():
        raise ValueError("Los tiempos de preparación no pueden ser negativos.")
    return S


ALL_SCHED_ALGS = {"Iterated Greedy (NEH)": _flow_ig, "Recocido Simulado": _perm_sa,
                  "Algoritmo Genético": _perm_ga, "Algoritmo Genético (POX)": _perm_ga}


def solve_scheduling(kind: str, data: str, algorithm: str, iterations: int, population: int,
                     seed: int, objective: str = "Makespan (Cmax)", due: str = "",
                     weights: str = "", setup: str = "") -> Dict[str, Any]:
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
        n, jobs = len(P), None
        allowed = FLOW_ALGORITHMS
    else:
        jobs = _parse_job_shop(data)
        n, P = len(jobs), None
        allowed = JOB_ALGORITHMS
    if algorithm not in allowed:
        raise ValueError(f"«{algorithm}» no aplica a {'flow shop' if flow else 'job shop'}; "
                         f"usa uno de: {', '.join(allowed)}.")
    M = ShopModel(flow, P=P, jobs=jobs, setup=_setup_matrix(setup, n),
                  due=_vector(due, n, "las fechas de entrega"),
                  weights=_vector(weights, n, "los pesos"), objective=objective)
    names = [a for a in allowed if a != "Comparar todos"] if algorithm == "Comparar todos" else [algorithm]
    res = {nm: ALL_SCHED_ALGS[nm](M, iterations, population, np.random.default_rng(seed))
           for nm in names}
    best_name = min(names, key=lambda nm: res[nm][1])
    seq, val, _ = res[best_name]
    ops, comp = M.schedule(seq)
    mk = float(comp.max())
    rows = []
    for nm in names:
        r = res[nm][2]
        step = max(1, len(r) // 60)
        for k, (it, b, c) in enumerate(r):
            if k % step == 0 or k == len(r) - 1:
                rows.append({"Iteración": it, **({"Algoritmo": nm} if len(names) > 1 else {}),
                             f"Mejor {objective}": b, "Valor de la iteración": c})
    for j in range(n):
        extra = ""
        if M.due is not None:
            extra = f" · entrega {M.due[j]:g} · tardanza {max(0.0, comp[j] - M.due[j]):g}"
        rows.append({"Iteración": f"Trabajo {j + 1}", **({"Algoritmo": best_name} if len(names) > 1 else {}),
                     f"Mejor {objective}": comp[j], "Valor de la iteración": f"termina en {comp[j]:g}{extra}"})
    nm_ = M.nm

    def plot(fig):
        ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
        for j, m, s, e in ops:
            ax1.barh(m, e - s, left=s, color=_PAL[j % len(_PAL)], edgecolor="#0a1228")
            if e - s > 0:
                ax1.text((s + e) / 2, m, str(j + 1), ha="center", va="center", fontsize=7,
                         color="#0a1228")
        if M.due is not None:
            for j in range(n):
                ax1.axvline(M.due[j], color=_PAL[j % len(_PAL)], lw=0.8, ls=":")
        ax1.set_yticks(range(nm_)); ax1.set_yticklabels([f"M{m + 1}" for m in range(nm_)])
        ax1.invert_yaxis(); ax1.set_xlabel("tiempo")
        ax1.set_title(f"Gantt ({best_name}) — {objective} = {val:g}")
        for nm in names:
            rr = np.array(res[nm][2])
            ax2.plot(rr[:, 0], rr[:, 1], lw=1.6, label=nm)
        ax2.set_xlabel("Iteración"); ax2.set_ylabel(objective)
        ax2.set_title("Convergencia"); ax2.legend(fontsize=7)

    def plot3d(fig):
        ax = fig.add_subplot(111, projection="3d")
        for j, m, s, e in ops:
            ax.bar3d(s, m - 0.3, j - 0.3, max(e - s, 1e-9), 0.6, 0.6,
                     color=_PAL[j % len(_PAL)], alpha=0.85)
        ax.set_xlabel("tiempo"); ax.set_ylabel("máquina"); ax.set_zlabel("trabajo")
        ax.set_title("3D — Programa de tareas", fontsize=10, color="#d8e8f8")

    return {"sequence": [int(x) for x in seq], "makespan": mk, "value": val,
            "objective": objective, "completion": comp.tolist(), "algorithm": best_name,
            "operations": ops, "results": {nm: res[nm][1] for nm in names}, "hist": rows,
            "plot": plot, "plot3d": plot3d,
            "note3d": "Ejes: tiempo, máquina y trabajo; cada bloque es una operación con su "
                      "inicio y su fin reales en el programa encontrado.",
            "resumen": f"{best_name}: {objective} = {val:g}, makespan {mk:g} "
                       f"({'flow shop' if flow else 'job shop'})"}


def brute_force_scheduling(kind: str, data: str, objective="Makespan (Cmax)", due="",
                           weights="", setup="") -> float:
    """Óptimo exacto por enumeración (instancias pequeñas)."""
    flow = kind.lower().startswith("flow")
    if flow:
        P = np.asarray([[float(x) for x in re.split(r"[,;\s]+", ln.strip()) if x]
                        for ln in data.splitlines() if ln.strip()])
        M = ShopModel(True, P=P, setup=_setup_matrix(setup, len(P)), due=_vector(due, len(P), "x"),
                      weights=_vector(weights, len(P), "x"), objective=objective)
        return min(M.value(list(p)) for p in itertools.permutations(range(M.n)))
    jobs = _parse_job_shop(data)
    M = ShopModel(False, jobs=jobs, setup=_setup_matrix(setup, len(jobs)),
                  due=_vector(due, len(jobs), "x"), weights=_vector(weights, len(jobs), "x"),
                  objective=objective)
    base = tuple(M.base_sequence())
    return min(M.value(list(p)) for p in set(itertools.permutations(base)))


FT06 = ("3 1, 1 3, 2 6, 4 7, 6 3, 5 6\n2 8, 3 5, 5 10, 6 10, 1 10, 4 4\n"
        "3 5, 4 4, 6 8, 1 9, 2 1, 5 7\n2 5, 1 5, 3 5, 4 3, 5 8, 6 9\n"
        "3 9, 2 3, 5 5, 6 4, 1 3, 4 1\n2 3, 4 3, 6 9, 1 10, 5 4, 3 1")
