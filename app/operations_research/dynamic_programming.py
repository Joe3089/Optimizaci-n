# app/operations_research/dynamic_programming.py
# Programación dinámica: ruta más corta por etapas, mochila, reemplazo de
# equipos y política de mantenimiento preventivo.
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def shortest_path(edges: List[Tuple[str, str, float]], source: str, target: str) -> Dict[str, Any]:
    if not edges:
        raise ValueError("Ingresa al menos un arco con el formato: A B 4.")
    succ: Dict[str, List[Tuple[str, float]]] = {}
    nodes = set()
    for u, v, w in edges:
        succ.setdefault(u, []).append((v, w))
        nodes.update((u, v))
    if source not in nodes or target not in nodes:
        raise ValueError("El nodo inicial o final no aparece en los arcos.")

    order: List[str] = []
    state: Dict[str, int] = {}

    def visit(node):
        if state.get(node) == 1:
            raise ValueError("La red tiene un ciclo; la recursión por etapas requiere una red acíclica.")
        if state.get(node) == 2:
            return
        state[node] = 1
        for nxt, _ in succ.get(node, []):
            visit(nxt)
        state[node] = 2
        order.append(node)

    for node in sorted(nodes):
        visit(node)

    INF = float("inf")
    f: Dict[str, float] = {target: 0.0}
    best: Dict[str, str] = {}
    hist = []
    for node in order:
        if node == target:
            hist.append({"Nodo": node, "Opciones": "—", "f*(nodo)": 0.0, "Siguiente": "—"})
            continue
        opts = [(w + f.get(v, INF), v) for v, w in succ.get(node, [])]
        reach = [o for o in opts if o[0] < INF]
        if reach:
            val, nxt = min(reach)
            f[node], best[node] = val, nxt
        txt = ", ".join(f"{v}: {c:g}" if c < INF else f"{v}: ∞" for c, v in opts) or "—"
        hist.append({"Nodo": node, "Opciones": txt,
                     "f*(nodo)": f.get(node, "∞"), "Siguiente": best.get(node, "—")})
    if source not in f:
        raise ValueError(f"No existe un camino de {source} a {target}.")

    path = [source]
    while path[-1] != target:
        path.append(best[path[-1]])
    total = f[source]
    route = " → ".join(path)
    hist.append({"Nodo": "Ruta óptima", "Opciones": route, "f*(nodo)": total})

    def plot(fig):
        ax = fig.add_subplot(111)
        names = [n for n in reversed(order) if n in f]
        ax.bar(names, [f[n] for n in names],
               color=["#ffd700" if n in path else "#4a9eff" for n in names])
        ax.set_title("Costo óptimo hasta el destino (ruta en dorado)")

    return {"path": path, "total": total, "hist": hist, "plot": plot,
            "resumen": f"Ruta: {route}. Costo = {total:g}"}


def knapsack_dp(values: List[float], weights: List[int], capacity: int) -> Dict[str, Any]:
    n = len(values)
    if n == 0 or n != len(weights):
        raise ValueError("Valores y pesos deben tener la misma cantidad de elementos.")
    if any(w <= 0 or int(w) != w for w in weights) or capacity <= 0 or int(capacity) != capacity:
        raise ValueError("Los pesos y la capacidad deben ser enteros positivos.")
    if capacity > 200_000:
        raise ValueError("La capacidad máxima admitida es 200 000.")
    capacity = int(capacity)
    w = [int(v) for v in weights]
    F = np.zeros((n + 1, capacity + 1))
    for i in range(1, n + 1):
        F[i] = F[i - 1]
        if w[i - 1] <= capacity:
            F[i, w[i - 1]:] = np.maximum(F[i - 1, w[i - 1]:], F[i - 1, :capacity + 1 - w[i - 1]] + values[i - 1])

    chosen, rows, c = [], [], capacity
    for i in range(n, 0, -1):
        take = abs(F[i, c] - F[i - 1, c]) > 1e-12
        rows.append({"Etapa": i, "Objeto": i, "Peso": w[i - 1], "Valor": values[i - 1],
                     "Capacidad disponible": c, "Decisión": "Llevar" if take else "No llevar",
                     "f(etapa, capacidad)": float(F[i, c])})
        if take:
            chosen.append(i)
            c -= w[i - 1]
    chosen.reverse()
    total = float(F[n, capacity])

    def plot(fig):
        ax = fig.add_subplot(111)
        ax.step(range(capacity + 1), F[n], where="post", color="#4a9eff")
        ax.set_xlabel("Capacidad"); ax.set_ylabel("Valor óptimo")
        ax.set_title("Valor óptimo según la capacidad")

    return {"chosen": chosen, "total": total, "hist": rows, "plot": plot,
            "resumen": f"Objetos {chosen}: valor = {total:g}, peso = {capacity - c}"}


def equipment_replacement(horizon: int, price: float, maintenance: List[float],
                          salvage: List[float]) -> Dict[str, Any]:
    """f(t) = min_k { P + Σ_{a<k} m_a − R_k + f(t+k) }, con f(T) = 0.

    maintenance[a]: costo durante el año de vida a+1; salvage[k-1]: valor de
    recuperación al vender con k años de uso.
    """
    L = len(maintenance)
    if horizon <= 0 or int(horizon) != horizon:
        raise ValueError("El horizonte debe ser un entero positivo.")
    if L == 0 or L != len(salvage):
        raise ValueError("Mantenimiento y valor de salvamento deben tener igual longitud.")
    T = int(horizon)
    cycle = [price + sum(maintenance[:k]) - salvage[k - 1] for k in range(1, L + 1)]
    f = [0.0] * (T + 1)
    best = [0] * (T + 1)
    hist = []
    for t in range(T - 1, -1, -1):
        opts = {k: cycle[k - 1] + f[t + k] for k in range(1, min(L, T - t) + 1)}
        k = min(opts, key=lambda a: (opts[a], a))
        f[t], best[t] = opts[k], k
        hist.append({"Año t": t, "Opciones (años de uso: costo)":
                     ", ".join(f"{a}: {v:g}" for a, v in opts.items()),
                     "f(t)": float(f[t]), "Usar por (años)": k, "Reemplazar en": t + k})
    plan, t = [0], 0
    while t < T:
        t += best[t]
        plan.append(t)
    route = " → ".join(str(v) for v in plan)
    hist.append({"Año t": "Plan", "Opciones (años de uso: costo)": f"Comprar/reemplazar en los años: {route}",
                 "f(t)": float(f[0])})

    def plot(fig):
        ax = fig.add_subplot(111)
        ks = list(range(1, L + 1))
        ax.bar(ks, [c / k for c, k in zip(cycle, ks)], color="#4a9eff")
        ax.set_xlabel("Años de uso antes de reemplazar"); ax.set_ylabel("Costo promedio anual")
        ax.set_title("Costo promedio anual según la vida útil")

    return {"plan": plan, "total": float(f[0]), "cycle": cycle, "hist": hist, "plot": plot,
            "resumen": f"Plan de reemplazo: años {route}. Costo total = {f[0]:g}"}


def maintenance_policy(machines: int, probs: List[float], c_prev: float, c_fail: float) -> Dict[str, Any]:
    """Compara mantenimiento preventivo cada n periodos contra solo correctivo.

    B(n) = N·Σ_{k≤n} p_k + Σ_{k<n} B(n−k)·p_k  (fallas esperadas en n periodos)
    """
    if machines <= 0:
        raise ValueError("El número de máquinas debe ser positivo.")
    p = np.asarray(probs, dtype=float)
    if p.size == 0 or np.any(p < 0) or abs(p.sum() - 1) > 1e-6:
        raise ValueError("Las probabilidades de falla deben ser no negativas y sumar 1.")
    if c_prev < 0 or c_fail < 0:
        raise ValueError("Los costos no pueden ser negativos.")
    B = [0.0]
    hist = []
    for n in range(1, p.size + 1):
        val = machines * p[:n].sum() + sum(B[n - k] * p[k - 1] for k in range(1, n))
        B.append(float(val))
    costs = [(c_prev * machines + c_fail * B[n]) / n for n in range(1, p.size + 1)]
    mtbf = float(sum((k + 1) * v for k, v in enumerate(p)))
    corrective = c_fail * machines / mtbf
    for n, cost in enumerate(costs, 1):
        hist.append({"Periodo n": n, "p(n)": float(p[n - 1]), "Fallas esperadas B(n)": B[n],
                     "Costo por periodo": float(cost)})
    best_n = int(np.argmin(costs)) + 1
    best = float(costs[best_n - 1])
    use_prev = best < corrective
    decision = (f"Mantenimiento preventivo cada {best_n} periodo(s)"
                if use_prev else "Solo mantenimiento correctivo")
    hist.append({"Periodo n": "Correctivo", "p(n)": f"Tiempo medio a la falla = {mtbf:g}",
                 "Costo por periodo": corrective})
    hist.append({"Periodo n": "Decisión", "p(n)": decision,
                 "Costo por periodo": min(best, corrective)})

    def plot(fig):
        ax = fig.add_subplot(111)
        ax.plot(range(1, p.size + 1), costs, "o-", color="#4a9eff", label="Preventivo")
        ax.axhline(corrective, color="#ff6b35", linestyle="--", label="Solo correctivo")
        ax.set_xlabel("Periodos entre mantenimientos"); ax.set_ylabel("Costo por periodo")
        ax.set_title("Política de mantenimiento"); ax.legend()

    return {"best_n": best_n, "best_cost": best, "corrective": corrective, "B": B,
            "use_preventive": use_prev, "hist": hist, "plot": plot,
            "resumen": f"{decision}. Costo por periodo = {min(best, corrective):g}"}
