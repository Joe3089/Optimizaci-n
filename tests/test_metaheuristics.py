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
