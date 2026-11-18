# -*- coding: utf-8 -*-
"""
Motor metaheurístico avanzado: óptimos conocidos, restricciones, enteros,
binarios, maximización, TSP contra el exacto (Held-Karp) y seguridad.

    venv\\Scripts\\python -m pytest tests/test_metaheuristics.py -v
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from app.optimization import advanced_metaheuristics as am  # noqa: E402
from app.optimization import combinatorial as comb  # noqa: E402

CONT = [n for n in am.ALGORITHMS if n != "Comparar todos"]


@pytest.mark.parametrize("alg", CONT)
def test_constrained_mixed_integer(alg):
    # mín (x1−2)² + (x2−3)² + x3², x1 + x2 ≤ 4, x3 entera → f* = 0.5 en (1.5, 2.5, 0)
    r = am.solve_general("Minimizar", "(x1-2)**2 + (x2-3)**2 + x3**2",
                         "x1: -5, 5\nx2: -5, 5\nx3: entera -3, 3", "x1 + x2 <= 4",
                         alg, 300, 40, 42)
    assert r["feasible"]
    assert abs(r["f"] - 0.5) < 0.02, r["resumen"]
    assert r["x"][2] == 0 and float(r["x"][2]).is_integer()
    assert r["x"][0] + r["x"][1] <= 4 + 1e-9


def test_rastrigin_10d_global_optimum():
    expr = "+".join(f"(x{i}**2 - 10*cos(2*pi*x{i}) + 10)" for i in range(1, 11))
    r = am.solve_general("Minimizar", expr, "todas: -5.12, 5.12", "",
                         "Evolución Diferencial", 600, 60, 1)
    assert r["f"] < 1e-6 and np.allclose(r["x"], 0, atol=1e-3), r["resumen"]


def test_binary_knapsack_maximization():
    r = am.solve_general("Maximizar", "60*x1 + 100*x2 + 120*x3", "todas: binaria",
                         "10*x1 + 20*x2 + 30*x3 <= 50", "Comparar todos", 100, 20, 3)
    assert r["f"] == 220 and r["x"] == [0, 1, 1]
    assert set(r["results"]) == set(CONT)


def test_equality_constraint():
    r = am.solve_general("Minimizar", "x**2 + y**2", "todas: -5, 5", "x + y = 2",
                         "Evolución Diferencial", 300, 30, 0)
    assert abs(r["x"][0] + r["x"][1] - 2) <= 1e-4 and abs(r["f"] - 2) < 1e-3


def test_greater_equal_and_history_consistency():
    r = am.solve_general("Minimizar", "x1 + x2", "todas: 0, 10", "x1 * x2 >= 4",
                         "PSO: Enjambre de Partículas", 200, 30, 5)
    assert r["feasible"] and abs(r["f"] - 4) < 0.05          # óptimo x1 = x2 = 2
    last = [h for h in r["hist"] if h["Iteración"] != "final"][-1]
    assert last["Mejor f"] == pytest.approx(r["f"]) and last["Factible"] == "Sí"


@pytest.mark.parametrize("alg", [a for a in am.TSP_ALGORITHMS if a != "Comparar todos"])
def test_tsp_matches_exact(alg):
    cities = am.random_cities(10, 7)
    exact = comb.tsp(cities)["best"]
    r = am.solve_tsp(cities, alg, 60, 15, 5)
    assert r["length"] == pytest.approx(exact, rel=1e-9)
    assert sorted(r["tour"]) == list(range(10))


def test_tsp_large_instance():
    r = am.solve_tsp(am.random_cities(120, 3), "Comparar todos", 60, 20, 3)
    assert len(r["tour"]) == 120 and sorted(r["tour"]) == list(range(120))
    assert set(r["results"]) == {a for a in am.TSP_ALGORITHMS if a != "Comparar todos"}


@pytest.mark.parametrize("bad", ["__import__('os').system('x')", "open('f')", "x; y",
                                 "lambda: 1", "exec('1')"])
def test_rejects_code(bad):
    with pytest.raises(ValueError, match="no permitidos"):
        am.solve_general("Minimizar", bad, "todas: 0, 1", "", "Evolución Diferencial", 10, 4, 0)


@pytest.mark.parametrize("vars_, cons, msg", [
    ("x1: -5, 5", "", "Faltan las cotas de la variable «x2»"),
    ("x1: 5, -5\nx2: 0, 1", "", "debe ser menor"),
    ("todas: 0, 1", "x1 + x2 <> 3", "Restricción inválida"),
    ("todas: 0, 1\nz9: 0, 1", "", "no aparecen"),
])
def test_input_validation(vars_, cons, msg):
    with pytest.raises(ValueError, match=msg):
        am.solve_general("Minimizar", "x1 + x2", vars_, cons, "Evolución Diferencial", 20, 6, 0)


def test_catalog_integration():
    from app.application.problem_catalog import SPEC_BY_LABEL, defaults, solve
    for label in ("Metaheurística: Problema General", "Metaheurística: Viajante (TSP grande)"):
        assert label in SPEC_BY_LABEL
        res = solve(label, defaults(label))
        assert res["hist"] and res["resumen"] and callable(res["plot3d"])


# ── Limitaciones resueltas ────────────────────────────────────────────────────
from app.optimization import meta_extra as mx  # noqa: E402

RASTRIGIN10 = "+".join(f"(x{i}**2 - 10*cos(2*pi*x{i}) + 10)" for i in range(1, 11))
ACKLEY10 = ("-20*exp(-0.2*sqrt((" + "+".join(f"x{i}**2" for i in range(1, 11)) + ")/10)) - exp(("
            + "+".join(f"cos(2*pi*x{i})" for i in range(1, 11)) + ")/10) + 20 + E")


def test_tabu_reaches_rastrigin_global():
    r = am.solve_general("Minimizar", RASTRIGIN10, "todas: -5.12, 5.12", "",
                         "Búsqueda Tabú", 600, 60, 1)
    assert r["f"] < 1e-6, r["resumen"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_annealing_reaches_rastrigin_global(seed):
    # basin hopping: antes del rediseño el Recocido Simulado quedaba en ~32
    r = am.solve_general("Minimizar", RASTRIGIN10, "todas: -5.12, 5.12", "",
                         "Recocido Simulado", 600, 30, seed)
    assert r["f"] < 1e-6, r["resumen"]


ROSEN10 = "+".join(f"(100*(x{i + 1}-x{i}**2)**2 + (1-x{i})**2)" for i in range(1, 10))


@pytest.mark.parametrize("fn,bounds", [(RASTRIGIN10, "-5.12, 5.12"), (ROSEN10, "-5, 10")])
def test_cma_es_hard_functions(fn, bounds):
    r = am.solve_general("Minimizar", fn, f"todas: {bounds}", "", "CMA-ES", 600, 30, 1)
    assert r["f"] < 1e-6, r["resumen"]


@pytest.mark.parametrize("alg", CONT)
def test_ackley_10d_all_algorithms(alg):
    r = am.solve_general("Minimizar", ACKLEY10, "todas: -32, 32", "", alg, 600, 60, 2)
    assert r["f"] < 1e-6, r["resumen"]


@pytest.mark.parametrize("alg", CONT)
def test_discrete_moves_all_algorithms(alg):
    r = am.solve_general("Maximizar", "60*x1 + 100*x2 + 120*x3", "todas: binaria",
                         "10*x1 + 20*x2 + 30*x3 <= 50", alg, 100, 20, 3)
    q = am.solve_general("Minimizar", "(x1-7)**2 + (x2+4)**2", "todas: entera -10, 10", "",
                         alg, 100, 20, 3)
    assert r["f"] == 220 and q["x"] == [7, -4]


def test_independent_runs_statistics():
    r = am.solve_general("Minimizar", "(x1-2)**2 + (x2-3)**2 + x3**2",
                         "x1: -5, 5\nx2: -5, 5\nx3: entera -3, 3", "x1 + x2 <= 4",
                         "Evolución Diferencial", 150, 30, 42, runs=5)
    st = r["results"]["Evolución Diferencial"]
    assert st["runs"] == 5 and st["factibles"] == 5 and abs(st["media"] - 0.5) < 0.01
    assert any("Ejecuciones" in h for h in r["hist"])


def test_nsga2_zdt1_true_front():
    g = "(1 + 9*(" + "+".join(f"x{i}" for i in range(2, 31)) + ")/29)"
    r = mx.solve_multiobjective(f"min: x1\nmin: {g}*(1 - sqrt(x1/{g}))", "todas: 0, 1", "",
                                250, 100, 1)
    F = np.array(r["front"])
    assert np.abs(F[:, 1] - (1 - np.sqrt(F[:, 0]))).mean() < 0.02
    assert F[:, 0].min() < 0.01 and F[:, 0].max() > 0.99 and r["hypervolume"] > 0.85


def test_nsga2_schaffer_and_constraints():
    r = mx.solve_multiobjective("min: x**2\nmin: (x-2)**2", "x: -10, 10", "", 100, 60, 1)
    X = np.array(r["x"]).ravel()
    assert X.min() > -1e-2 and X.max() < 2.01 and X.max() - X.min() > 1.9
    r = mx.solve_multiobjective("max: x1 + x2\nmin: x1**2 + x2**2", "todas: 0, 5",
                                "x1 + 2*x2 <= 6", 100, 60, 1)
    assert all(a + 2 * b <= 6 + 1e-9 for a, b in r["x"])


@pytest.mark.parametrize("alg", mx.VRP_ALGORITHMS[:-1])
def test_cvrp_matches_brute_force(alg):
    rng = np.random.default_rng(3)
    cust = [[*np.round(rng.random(2) * 100, 1), int(rng.integers(1, 10))] for _ in range(7)]
    exact = mx.brute_force_vrp([50, 50], cust, 15)
    r = mx.solve_vrp([50, 50], cust, 15, 0, alg, 80, 20, 1)
    assert r["cost"] == pytest.approx(exact, rel=1e-9)
    assert all(sum(cust[c - 1][2] for c in route) <= 15 for route in r["routes"])
    assert sorted(c for route in r["routes"] for c in route) == list(range(1, 8))


def test_cvrp_fleet_limit_and_validation():
    cust = [[10, 10, 5], [20, 20, 5], [80, 80, 5], [90, 90, 5]]
    r = mx.solve_vrp([50, 50], cust, 10, 2, "Comparar todos", 40, 10, 1)
    assert len(r["routes"]) <= 2 and r["feasible"]
    with pytest.raises(ValueError, match="supera la capacidad de la flota"):
        mx.solve_vrp([50, 50], cust, 10, 1, "Recocido Simulado", 10, 5, 1)
    with pytest.raises(ValueError, match="supera la capacidad de un vehículo"):
        mx.solve_vrp([50, 50], [[1, 1, 50], [2, 2, 1]], 10, 0, "Recocido Simulado", 10, 5, 1)


@pytest.mark.parametrize("alg", mx.FLOW_ALGORITHMS[:-1])
def test_flow_shop_matches_brute_force(alg):
    import itertools
    P = np.random.default_rng(5).integers(1, 20, (6, 4)).astype(float)
    exact = min(mx.flow_makespan(P, p) for p in itertools.permutations(range(6)))
    data = "\n".join(", ".join(f"{v:g}" for v in row) for row in P)
    assert mx.solve_scheduling("Flow shop", data, alg, 60, 20, 1)["makespan"] == exact


@pytest.mark.parametrize("alg", mx.JOB_ALGORITHMS[:-1])
def test_job_shop_ft06_optimum(alg):
    r = mx.solve_scheduling("Job shop", mx.FT06, alg, 300, 40, 1)
    assert r["makespan"] == 55                      # óptimo conocido de FT06
    ops = r["operations"]
    for m in {o[1] for o in ops}:                   # sin solapes por máquina
        iv = sorted((s, e) for _, mm, s, e in ops if mm == m)
        assert all(a[1] <= b[0] + 1e-9 for a, b in zip(iv, iv[1:]))


def test_new_catalog_entries():
    from app.application.problem_catalog import SPEC_BY_LABEL, defaults, solve
    for label in ("Metaheurística: Multiobjetivo (NSGA-II)",
                  "Metaheurística: Ruteo de Vehículos (CVRP)",
                  "Metaheurística: Programación de Tareas"):
        res = solve(label, defaults(label))
        assert res["hist"] and res["resumen"] and callable(res["plot3d"])


# ── VRP con ventanas horarias y flota heterogénea ────────────────────────────
def _tw_instance():
    r2 = np.random.default_rng(11)
    out = []
    for _ in range(7):
        x, y = np.round(r2.random(2) * 100, 1)
        a = float(r2.integers(0, 150))
        out.append([x, y, int(r2.integers(1, 8)), a, a + 60, 5])
    return out


@pytest.mark.parametrize("alg", mx.VRP_ALGORITHMS[:-1])
def test_vrptw_matches_brute_force(alg):
    cust, dep = _tw_instance(), [50, 50, 0, 400]
    exact = mx.brute_force_vrp(dep, cust, 15)
    r = mx.solve_vrp(dep, cust, 15, 0, alg, 80, 20, 1)
    assert r["cost"] == pytest.approx(exact, rel=1e-9) and r["feasible"]


def test_vrptw_routes_respect_windows():
    cust, dep = _tw_instance(), [50, 50, 0, 400]
    r = mx.solve_vrp(dep, cust, 15, 0, "Comparar todos", 60, 20, 1)
    pts = np.vstack([dep[:2], np.array(cust)[:, :2]])
    for route in r["routes"]:
        t, prev = 0.0, 0
        for c in route:
            t += float(np.hypot(*(pts[c] - pts[prev])))
            assert t <= cust[c - 1][4] + 1e-9                 # llega antes del cierre
            t = max(t, cust[c - 1][3]) + cust[c - 1][5]; prev = c
        assert t + float(np.hypot(*(pts[0] - pts[prev]))) <= 400 + 1e-9


@pytest.mark.parametrize("alg", mx.VRP_ALGORITHMS[:-1])
def test_heterogeneous_fleet_matches_brute_force(alg):
    rng = np.random.default_rng(3)
    cust = [[*np.round(rng.random(2) * 100, 1), int(rng.integers(1, 10))] for _ in range(7)]
    types = "10, 30, 1, 0\n25, 60, 1.2, 0"
    exact = mx.brute_force_vrp([50, 50], cust, 0, types)
    r = mx.solve_vrp([50, 50], cust, 0, 0, alg, 80, 20, 1, types)
    assert r["cost"] == pytest.approx(exact, rel=1e-9)


def test_vrp_unreachable_window_is_reported():
    with pytest.raises(ValueError, match="a tiempo"):
        mx.solve_vrp([0, 0, 0, 100], [[90, 90, 1, 0, 10, 0], [1, 1, 1, 0, 100, 0]], 10, 0,
                     "Recocido Simulado", 10, 5, 1)


# ── Programación con preparación, fechas de entrega y pesos ──────────────────
@pytest.mark.parametrize("objective", mx.OBJECTIVES)
@pytest.mark.parametrize("inst", [5, 7, 8])
def test_flow_shop_objectives_exact(objective, inst):
    P = np.random.default_rng(inst).integers(1, 20, (6, 4)).astype(float)
    data = "\n".join(", ".join(f"{v:g}" for v in row) for row in P)
    S = np.random.default_rng(inst + 1).integers(0, 6, (6, 6))
    setup = "\n".join(" ".join(map(str, r)) for r in S)
    due, w = "40, 55, 60, 70, 80, 95", "1, 2, 1, 3, 1, 2"
    exact = mx.brute_force_scheduling("Flow shop", data, objective, due, w, setup)
    for alg in mx.FLOW_ALGORITHMS[:-1]:
        r = mx.solve_scheduling("Flow shop", data, alg, 80, 20, 1, objective, due, w, setup)
        assert r["value"] == exact, (alg, r["value"], exact)


@pytest.mark.parametrize("objective", mx.OBJECTIVES)
def test_job_shop_setups_objectives_exact(objective):
    js = "1 3, 2 2, 3 2\n1 2, 3 1, 2 4\n2 4, 3 3, 1 1"
    args = ("6, 8, 9", "2, 1, 3", "0 1 2\n1 0 1\n2 1 0")
    exact = mx.brute_force_scheduling("Job shop", js, objective, *args)
    for alg in mx.JOB_ALGORITHMS[:-1]:
        assert mx.solve_scheduling("Job shop", js, alg, 80, 20, 1, objective, *args)["value"] == exact


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ft06_optimum_every_seed(seed):
    for alg in mx.JOB_ALGORITHMS[:-1]:
        assert mx.solve_scheduling("Job shop", mx.FT06, alg, 300, 40, seed)["makespan"] == 55


def test_tardiness_requires_due_dates():
    with pytest.raises(ValueError, match="fechas de entrega"):
        mx.solve_scheduling("Flow shop", "1, 2\n3, 4", "Recocido Simulado", 10, 5, 1,
                            "Tardanza total")
