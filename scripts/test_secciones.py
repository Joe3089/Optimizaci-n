# -*- coding: utf-8 -*-
"""
test_secciones.py — Verifica los métodos de Estadística, Investigación de
Operaciones y Optimización contra resultados conocidos y contra SciPy.

Uso:
    python scripts/test_secciones.py
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment, linprog  # noqa: E402

from app.application import problem_catalog as cat  # noqa: E402
from app.domain.inventory import loader  # noqa: E402
from app.operations_research import assignment as asg  # noqa: E402
from app.operations_research import dynamic_programming as dp  # noqa: E402
from app.operations_research import linear_programming as lp  # noqa: E402
from app.operations_research import transportation as tr  # noqa: E402
from app.optimization import combinatorial as comb  # noqa: E402
from app.optimization import interior_point as ip  # noqa: E402
from app.optimization import quantum as qc  # noqa: E402
from app.statistics import descriptive as desc  # noqa: E402
from app.statistics import inferential as inf  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(f"{name}  {detail}" if not cond else name)


def near(a, b, tol=1e-6):
    return a is not None and abs(a - b) <= tol * max(1.0, abs(b))


def raises(name, fn):
    try:
        fn()
    except ValueError:
        check(name, True)
    except Exception as ex:
        check(name, False, f"excepción no esperada {type(ex).__name__}: {ex}")
    else:
        check(name, False, "no rechazó la entrada")


def test_statistics():
    r = desc.descriptive([2, 4, 4, 4, 5, 5, 7, 9])
    check("descriptiva media", near(r["mean"], 5))
    check("descriptiva mediana", near(r["median"], 4.5))
    check("descriptiva moda", r["modes"] == [4.0])
    check("descriptiva desv", near(r["std"], 2.138089935))
    check("descriptiva atípicos", desc.descriptive([1, 2, 3, 4, 100])["outliers"] == [100.0])
    raises("descriptiva 1 dato", lambda: desc.descriptive([1]))

    data = [12, 15, 14, 10, 18, 20, 13, 16, 15, 17, 22, 14, 15, 19, 11]
    r = inf.confidence_interval(data, 0.95)
    check("IC 95", near(r["low"], 13.54197, 1e-4) and near(r["high"], 17.25803, 1e-4), str((r["low"], r["high"])))
    r = inf.t_test(data, 14, "Bilateral (≠)", 0.05)
    check("prueba t", near(r["t"], 1.61607, 1e-3) and not r["reject"], str(r["t"]))

    r = inf.regression([[1, 3], [2, 5], [3, 7], [4, 9.0001]])
    check("regresión exacta", near(r["beta"][1], 2, 1e-3) and near(r["beta"][0], 1, 1e-3))
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    y = 3 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.2, size=60)
    r = inf.regression(np.column_stack([X, y]).tolist())
    ref = np.linalg.lstsq(np.column_stack([np.ones(60), X]), y, rcond=None)[0]
    check("regresión múltiple", np.allclose(r["beta"], ref))
    raises("regresión colineal", lambda: inf.regression([[1, 2, 3], [2, 4, 5], [3, 6, 7], [4, 8, 9], [5, 10, 12]]))

    r = inf.monte_carlo("4*(x**2 + y**2 <= 1)", "x ~ uniforme(0, 1)\ny ~ uniforme(0, 1)", 500000, 42)
    check("Monte Carlo π", abs(r["mean"] - math.pi) < 0.01, str(r["mean"]))
    r2 = inf.monte_carlo("x", "x ~ normal(10, 2)", 200000, 1)
    check("Monte Carlo normal", abs(r2["mean"] - 10) < 0.03 and abs(r2["std"] - 2) < 0.03)
    raises("Monte Carlo código malicioso", lambda: inf.monte_carlo("__import__('os')", "x ~ normal(0, 1)", 1000, 1))
    raises("Monte Carlo atributo", lambda: inf.monte_carlo("x.__class__", "x ~ normal(0, 1)", 1000, 1))


def _scipy_lp(c, A, signs, b, sense, lb=None, ub=None):
    c = np.array(c, float)
    A = np.array(A, float)
    b = np.array(b, float)
    Aub, bub, Aeq, beq = [], [], [], []
    for row, s, r in zip(A, signs, b):
        if s == "<=":
            Aub.append(row); bub.append(r)
        elif s == ">=":
            Aub.append(-row); bub.append(-r)
        else:
            Aeq.append(row); beq.append(r)
    n = len(c)
    lb = [0] * n if lb is None else lb
    ub = [None] * n if ub is None else [None if math.isinf(v) else v for v in ub]
    res = linprog(-c if sense == "max" else c, A_ub=Aub or None, b_ub=bub or None,
                  A_eq=Aeq or None, b_eq=beq or None, bounds=list(zip(lb, ub)), method="highs")
    return res


def test_linear_programming():
    r = lp.solve_lp([3, 5], [[1, 0], [0, 2], [3, 2]], ["<="] * 3, [4, 12, 18], "max", "simplex")
    check("Simplex Wyndor", near(r["z"], 36) and np.allclose(r["x"], [2, 6]), str(r["x"]))
    A, s, b = [[3, 1], [4, 3], [1, 2]], ["=", ">=", "<="], [3, 6, 4]
    r = lp.solve_lp([4, 1], A, s, b, "min", "big_m")
    check("Big M", near(r["z"], 3.4) and np.allclose(r["x"], [0.4, 1.8]), str(r["x"]))
    check("Big M muestra M", any("M" in str(v) for row in r["hist"] for v in row.values() if isinstance(v, str)))
    r = lp.solve_lp([0.4, 0.5], [[0.3, 0.1], [0.5, 0.5], [0.6, 0.4]], ["<=", "=", ">="], [2.7, 6, 6], "min", "two_phase")
    check("Dos Fases", near(r["z"], 5.25) and np.allclose(r["x"], [7.5, 4.5]), str(r["x"]))
    r = lp.solve_lp([3, 9], [[1, 4], [1, 2]], ["<="] * 2, [8, 4], "max", "big_m")
    check("Degenerado", near(r["z"], 18) and r["degenerate"], str(r["status"]))
    r = lp.solve_lp([0.75, -20, 0.5, -6], [[0.25, -8, -1, 9], [0.5, -12, -0.5, 3], [0, 0, 1, 0]],
                    ["<="] * 3, [0, 0, 1], "max", "simplex")
    check("Degenerado Beale sin ciclar", near(r["z"], 1.25), str(r["z"]))
    r = lp.solve_lp([2, 1], [[1, -1], [2, 0]], ["<="] * 2, [10, 40], "max", "big_m")
    check("No acotado", r["status"] == lp.UNBOUNDED and r["ray"] is not None, r["status"])
    r = lp.solve_lp([1, 1], [[1, 1], [1, 1]], ["<=", ">="], [1, 2], "max", "big_m")
    check("Infactible Big M", r["status"] == lp.INFEASIBLE, r["status"])
    r = lp.solve_lp([1, 1], [[1, 1], [1, 1]], ["<=", ">="], [1, 2], "max", "two_phase")
    check("Infactible Dos Fases", r["status"] == lp.INFEASIBLE, r["status"])
    r = lp.solve_lp([3, 2], [[1, 1]], ["<="], [4], "max", "big_m", [0, 0], [3, 2])
    check("Simplex acotado", near(r["z"], 11) and np.allclose(r["x"], [3, 1]), str(r["x"]))
    r = lp.solve_lp([1, 1], [[1, 1]], ["<="], [4], "max", "big_m")
    check("Óptimos alternativos", r["alternate"])
    r = lp.solve_lp([1, 1], [[1, 1], [2, 2], [1, -1]], ["=", "=", "="], [2, 4, 0], "min", "two_phase")
    check("Fila redundante", near(r["z"], 2), str(r["status"]))
    raises("Simplex exige holguras", lambda: lp.solve_lp([1], [[1]], [">="], [1], "max", "simplex"))

    r = lp.graphical([3, 5], np.array([[1, 0], [0, 2], [3, 2]]), ["<="] * 3, [4, 12, 18], "max")
    check("Gráfico", near(r["z"], 36) and len(r["vertices"]) == 5, str(r["vertices"]))
    raises("Gráfico 3 variables", lambda: lp.graphical([1, 1, 1], np.array([[1, 1, 1]]), ["<="], [1], "max"))

    rng = np.random.default_rng(7)
    mism = {"big_m": 0, "two_phase": 0, "ip": 0}
    n_ok = n_inf = n_unb = 0
    for _ in range(400):
        n, m = int(rng.integers(2, 7)), int(rng.integers(1, 7))
        c = rng.integers(-9, 10, n).astype(float).tolist()
        A = rng.integers(-5, 10, (m, n)).astype(float).tolist()
        b = rng.integers(-5, 30, m).astype(float).tolist()
        signs = [["<=", ">=", "="][int(i)] for i in rng.choice(3, m, p=[0.6, 0.3, 0.1])]
        sense = "max" if rng.random() < 0.5 else "min"
        lb = ub = None
        if rng.random() < 0.5:
            lb = rng.integers(0, 3, n).astype(float).tolist()
            ub = [l + float(rng.integers(1, 12)) if rng.random() < 0.7 else float("inf") for l in lb]
        ref = _scipy_lp(c, A, signs, b, sense, lb, ub)
        expected = {0: lp.OPTIMAL, 2: lp.INFEASIBLE, 3: lp.UNBOUNDED}.get(ref.status)
        if expected is None:
            continue
        n_ok += expected == lp.OPTIMAL; n_inf += expected == lp.INFEASIBLE; n_unb += expected == lp.UNBOUNDED
        z_ref = None if ref.status else (-ref.fun if sense == "max" else ref.fun)
        for method in ("big_m", "two_phase"):
            r = lp.solve_lp(c, A, signs, b, sense, method, lb, ub, keep_tableaus=False)
            if r["status"] != expected or (z_ref is not None and not near(r["z"], z_ref, 1e-6)):
                mism[method] += 1
                if mism[method] <= 3:
                    print("DIF", method, c, A, signs, b, sense, lb, ub, r["status"], r["z"], expected, z_ref)
        if lb is None and ref.status == 0:
            r = ip.interior_point(c, A, signs, b, sense)
            if r["status"] != lp.OPTIMAL or not near(r["z"], z_ref, 1e-4):
                mism["ip"] += 1
                if mism["ip"] <= 3:
                    print("DIF ip", c, A, signs, b, sense, r["status"], r["z"], z_ref)
    print(f"PL aleatorios: {n_ok} óptimos, {n_inf} infactibles, {n_unb} no acotados")
    for k, v in mism.items():
        check(f"PL aleatorio vs SciPy ({k})", v == 0, f"{v} diferencias")


def test_interior_klee():
    r = ip.interior_point([3, 5], [[1, 0], [0, 2], [3, 2]], ["<="] * 3, [4, 12, 18], "max")
    check("Punto interior", near(r["z"], 36, 1e-6) and np.allclose(r["x"], [2, 6], atol=1e-5), str(r["x"]))
    r = ip.klee_minty(10)
    check("Klee-Minty 2^n−1", r["pivots"] == [2 ** n - 1 for n in range(2, 11)], str(r["pivots"]))
    check("Klee-Minty Z", all(near(h["Z* Simplex"], 5.0 ** h["n"]) for h in r["hist"]))
    check("Klee-Minty punto interior", all(isinstance(h["Z* punto interior"], float)
                                           and near(h["Z* punto interior"], 5.0 ** h["n"], 1e-4) for h in r["hist"]),
          str([h["Z* punto interior"] for h in r["hist"]]))
    check("Klee-Minty IP pocas iter", max(r["ipm"]) < 60, str(r["ipm"]))
    raises("Klee-Minty n inválido", lambda: ip.klee_minty(20))


def _transport_ref(s, d, c):
    C = np.array(c, float)
    m, n = C.shape
    Aeq, beq = [], []
    Aub, bub = [], []
    for i in range(m):
        row = np.zeros(m * n); row[i * n:(i + 1) * n] = 1
        Aub.append(row); bub.append(s[i])
    for j in range(n):
        row = np.zeros(m * n); row[j::n] = 1
        Aeq.append(row); beq.append(d[j])
    if sum(s) < sum(d):
        Aeq, beq, Aub2, bub2 = [], [], Aub, bub
        Aub = [r for r in Aub2]; bub = list(bub2)
        for j in range(n):
            row = np.zeros(m * n); row[j::n] = -1
            Aub.append(row); bub.append(-d[j])
        # con faltante se surte todo lo ofertado: costo del dummy = 0
        Aeq = [r for r in Aub[:m]]; beq = list(bub[:m])
        Aub = [-r for r in Aub[m:]]; bub = [d[j] for j in range(n)]
    return linprog(C.ravel(), A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, method="highs").fun


def test_transport_assignment():
    s, d, c = [15, 25, 10], [5, 15, 15, 15], [[10, 2, 20, 11], [12, 7, 9, 20], [4, 14, 16, 18]]
    check("Noroeste", near(tr.solve_transport(s, d, c, "northwest")["total"], 520))
    check("Vogel", near(tr.solve_transport(s, d, c, "vogel")["total"], 475),
          str(tr.solve_transport(s, d, c, "vogel")["total"]))
    check("MODI", near(tr.solve_transport(s, d, c, "modi")["total"], 435),
          str(tr.solve_transport(s, d, c, "modi")["total"]))
    r = tr.solve_transport([30, 20], [10, 10], [[1, 2], [3, 1]], "modi")
    check("Transporte desbalanceado", "ficticio" in r["note"] and near(r["total"], 20), str(r["total"]))
    raises("Transporte dimensiones", lambda: tr.solve_transport([1, 2], [3], [[1, 2]], "modi"))

    rng = np.random.default_rng(3)
    bad = 0
    for _ in range(200):
        m, n = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        C = rng.integers(1, 30, (m, n)).astype(float)
        dem = rng.integers(0, 30, n).astype(float)
        sup = rng.multinomial(int(dem.sum()), [1 / m] * m).astype(float)
        if rng.random() < 0.4:
            sup += rng.integers(0, 10, m)
        ref = _transport_ref(sup.tolist(), dem.tolist(), C.tolist())
        got = tr.solve_transport(sup.tolist(), dem.tolist(), C.tolist(), "modi")["total"]
        if not near(got, ref, 1e-6):
            bad += 1
    check("Transporte aleatorio vs SciPy", bad == 0, f"{bad} diferencias")

    r = asg.hungarian([[15, 10, 9], [9, 15, 10], [10, 12, 8]])
    check("Húngaro", near(r["total"], 27), str(r["total"]))
    bad = 0
    for _ in range(300):
        m, n = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        C = rng.integers(0, 50, (m, n)).astype(float)
        sense = "max" if rng.random() < 0.5 else "min"
        i, j = linear_sum_assignment(C, maximize=sense == "max")
        if not near(asg.hungarian(C.tolist(), sense)["total"], C[i, j].sum()):
            bad += 1
    check("Húngaro aleatorio vs SciPy", bad == 0, f"{bad} diferencias")


def test_dp():
    edges = cat.arcs(cat.defaults("PD: Ruta Más Corta por Etapas")["arcos"])
    r = dp.shortest_path(edges, "A", "G")
    check("Ruta por etapas", near(r["total"], 8) and r["path"] == ["A", "C", "E", "G"], str(r["path"]))
    raises("Ruta con ciclo", lambda: dp.shortest_path([("A", "B", 1), ("B", "A", 1)], "A", "B"))
    r = dp.knapsack_dp([60, 100, 120], [10, 20, 30], 50)
    check("Mochila PD", near(r["total"], 220) and r["chosen"] == [2, 3], str(r["chosen"]))
    r = dp.equipment_replacement(4, 1000, [60, 80, 120, 200], [800, 600, 500, 400])
    # Costos de ciclo 260, 540, 760, 1060. Mejor combinación: 260 + 760 = 1020.
    check("Reemplazo", near(r["total"], 1020) and r["plan"] == [0, 1, 4], str(r["plan"]))
    r = dp.maintenance_policy(50, [0.05, 0.10, 0.15, 0.30, 0.40], 20, 100)
    check("Mantenimiento B(1)", near(r["B"][1], 2.5) and near(r["B"][2], 7.625), str(r["B"]))
    raises("Mantenimiento prob", lambda: dp.maintenance_policy(5, [0.5, 0.2], 1, 1))


def test_inventory_file():
    r = loader.load_and_analyze(text=loader.SAMPLE)
    row = next(x for x in r["hist"] if x["Producto"] == "Tornillo")
    check("Inventario EOQ", near(row["Q* (EOQ)"], math.sqrt(2 * 12000 * 40 / 0.5)))
    check("Inventario ROP", near(row["ROP"], 33 * 5 + 1.644854 * 6 * math.sqrt(5), 1e-4), str(row["ROP"]))
    check("Inventario ABC", r["hist"][0]["Producto"] == "Motor" and r["hist"][0]["Clase ABC"] == "A")

    d = tempfile.mkdtemp()
    p = os.path.join(d, "inv.csv")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("Artículo,Demanda Anual,Costo Pedido,Costo Mantener\nA1,1000,50,2\n")
    check("Inventario CSV", near(loader.load_and_analyze(p)["hist"][0]["Q* (EOQ)"], 223.606798, 1e-5))
    from openpyxl import Workbook
    wb = Workbook(); ws = wb.active
    ws.append(["SKU", "Demanda", "Costo de pedido", "Costo de mantener"]); ws.append(["B1", 1000, 50, 2])
    px = os.path.join(d, "inv.xlsx"); wb.save(px)
    check("Inventario Excel", near(loader.load_and_analyze(px)["hist"][0]["Q* (EOQ)"], 223.606798, 1e-5))
    raises("Inventario sin columnas", lambda: loader.load_and_analyze(text="a;b\n1;2\n"))
    raises("Inventario negativo", lambda: loader.load_and_analyze(text="producto;demanda;costo_pedido;costo_mantener\nX;-1;2;3\n"))
    raises("Inventario archivo inexistente", lambda: loader.load_and_analyze("no_existe.csv"))


def test_combinatorial():
    r = comb.knapsack_bb([60, 100, 120], [10, 20, 30], 50)
    check("Mochila B&B", near(r["total"], 220) and r["chosen"] == [2, 3], str(r["chosen"]))
    rng = np.random.default_rng(5)
    bad = 0
    for _ in range(150):
        n = int(rng.integers(1, 12))
        v, w = rng.integers(1, 50, n).tolist(), rng.integers(1, 20, n).tolist()
        cap = int(rng.integers(1, 60))
        if not near(comb.knapsack_bb(v, w, cap)["total"], dp.knapsack_dp(v, w, cap)["total"]):
            bad += 1
    check("Mochila B&B vs PD", bad == 0, f"{bad} diferencias")

    bad = 0
    for _ in range(40):
        P = rng.uniform(0, 10, (int(rng.integers(3, 8)), 2))
        r = comb.tsp(P.tolist())
        D = np.linalg.norm(P[:, None] - P[None, :], axis=2)
        best = min(D[0, t[0]] + sum(D[t[i], t[i + 1]] for i in range(len(t) - 1)) + D[t[-1], 0]
                   for t in permutations(range(1, len(P))))
        if not near(r["exact"], best) or not near(comb._tour_len(r["tour"], D), best):
            bad += 1
    check("TSP Held-Karp vs fuerza bruta", bad == 0, f"{bad} diferencias")

    n, e = cat.graph(cat.defaults("Problema de Máximo Corte")["aristas"])
    r = comb.max_cut(n, e)
    check("Máximo Corte", near(r["best"], 4), str(r["best"]))
    check("Máximo Corte heurística ≤ óptimo", r["greedy"] <= r["best"] and r["local"] <= r["best"])
    r = comb.complexity([10, 20, 30], 1e9)
    check("Complejidad", len(r["hist"]) == 7 and r["limit_2n"] == 55, str(r["limit_2n"]))
    check("Complejidad n grande", "años" in comb.complexity([1000], 1e9)["hist"][6]["n = 1000"])


def test_quantum():
    r = qc.phenomenon("Superposición", 10000, 1)
    check("Superposición", np.allclose(r["probs"], [0.5, 0.5]))
    r = qc.phenomenon("Entrelazamiento", 10000, 1)
    check("Entrelazamiento", np.allclose(r["probs"], [0.5, 0, 0, 0.5]) and r["counts"][1] == r["counts"][2] == 0)
    r = qc.phenomenon("Interferencia", 10000, 1)
    check("Interferencia", np.allclose(r["probs"], [1, 0]))
    r = qc.bit_vs_qubit(60, 45, 20000, 1)
    check("Qubit", near(r["p0"], 0.75) and near(r["p1"], 0.25) and abs(r["ones"] / 20000 - 0.25) < 0.02)
    check("Bloch normalizado", near(sum(v ** 2 for v in r["bloch"]), 1))

    r = qc.grover(3, ["101"])
    check("Grover", r["k_opt"] == 2 and near(r["p_success"], 0.9453125, 1e-6), str(r["p_success"]))
    check("Grover simulada = teórica", all(near(h["P(éxito) simulada"], h["P(éxito) teórica"]) for h in r["hist"]))
    raises("Grover estado inválido", lambda: qc.grover(3, ["12"]))

    r = qc.vqe("-1 ZZ\n-0.5 XI\n-0.5 IX", 2, 42)
    check("VQE", abs(r["energy"] - r["exact"]) < 1e-4 and near(r["exact"], -math.sqrt(2), 1e-6), str((r["energy"], r["exact"])))
    raises("VQE formato", lambda: qc.vqe("hola", 1, 1))

    n, e = cat.graph(cat.defaults("Cuántica: QAOA (Máximo Corte)")["aristas"])
    r = qc.qaoa(n, e, 3, 42)
    check("QAOA razón", r["ratio"] > 0.85 and r["top_cut"] == 4, str(r["ratio"]))
    r = qc.qaoa(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)], 1, 42)
    check("QAOA anillo p=1 (teoría 0.75)", near(r["ratio"], 0.75, 1e-3), str(r["ratio"]))
    r = qc.quantum_annealing(n, e, 50, 1000)
    check("Recocido cuántico", r["p_opt"] > 0.9, str(r["p_opt"]))
    raises("QAOA demasiados nodos", lambda: qc.qaoa(20, [(0, 19, 1)], 1, 1))


def test_catalog_defaults():
    """Todos los ejemplos precargados deben resolverse y tener tabla no vacía."""
    for spec in cat.SPECS:
        try:
            r = cat.solve(spec.label, cat.defaults(spec.label))
            check(f"Ejemplo: {spec.label}", bool(r["hist"]) and bool(r["resumen"]))
        except Exception as ex:
            check(f"Ejemplo: {spec.label}", False, f"{type(ex).__name__}: {ex}")


def test_fixed_limitations():
    from app.application.optimization_service import run_method
    from app.infrastructure.ai_assistant import _SYSTEM_PROMPT, detect_method_label

    h, _ = run_method("Wolfe", "(1-x)**2 + 100*(y-x**2)**2", "", [], [-1.2, 1.0], -5, 5, 0.1)
    check("Wolfe iterativo Rosenbrock", np.allclose(h[-1]["x_k"], [1, 1], atol=1e-6) and h[-1]["grad_norm"] < 1e-6,
          h[-1]["x_k_str"])
    h, _ = run_method("Wolfe", "x**2 - 4*x + 5", "", [], [1.0], 0, 5, 0.1)
    check("Wolfe 1D", near(float(h[-1]["x_k"][0]), 2) and near(h[-1]["f_k"], 1))
    h, _ = run_method("Wolfe", "(x-1)**4 + (y+2)**2", "", [], [3.0, 3.0], -5, 5, 0.1)
    check("Wolfe Hessiano casi singular", h[-1]["grad_norm"] < 1e-6, str(h[-1]["grad_norm"]))

    check("IA conoce el catálogo", all(s.label in _SYSTEM_PROMPT for s in cat.SPECS))
    check("IA detecta nombre largo", detect_method_label(
        "Usa **PL: Simplex Acotado (cotas en variables)**.") == "PL: Simplex Acotado (cotas en variables)")
    check("IA detecta inventario", detect_method_label("Activa Inventario: EOQ Clásico") == "Inventario: EOQ Clásico")
    check("IA sin método", detect_method_label("Hola") is None)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from section_docs_data import SECTION_METHODS, section_tests
    check("Documentación de todo el catálogo", len(SECTION_METHODS) == len(cat.SPECS))
    check("Casos del documento de pruebas", all(t["resultado"] for t in section_tests()))


def main() -> int:
    for fn in (test_fixed_limitations,test_statistics, test_linear_programming, test_interior_klee, test_transport_assignment,
               test_dp, test_inventory_file, test_combinatorial, test_quantum, test_catalog_defaults):
        try:
            fn()
        except Exception as ex:
            import traceback
            FAIL.append(f"{fn.__name__}: {type(ex).__name__}: {ex}\n{traceback.format_exc()}")
    print(f"\n{'=' * 70}\nRESULTADO: {len(PASS)} OK, {len(FAIL)} FALLOS\n{'=' * 70}")
    for f in FAIL:
        print(f"  FAIL {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
