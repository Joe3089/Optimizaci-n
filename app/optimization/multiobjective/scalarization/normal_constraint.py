# app/optimization/multiobjective/scalarization/normal_constraint.py
from __future__ import annotations

from typing import Any, Sequence

from .base import ObjectiveFn, ScalarizationResult, ScalarizationStrategy, _normalize_weights


class NormalConstraintStrategy(ScalarizationStrategy):
    """Normal Constraint (NC), variante de NBI con restricción de
    desigualdad en vez de igualdad: minimiza un objetivo representativo
    sujeto a que F(x) quede del lado del punto ideal respecto al hiperplano
    que pasa por z_bar (punto de la CHIM) con normal n = z*-z_bar:

        phi(x) = f_rep(x) + penalty * max(0, dot(F(x)-z_bar, n))^2

    Igual que NBIStrategy, recibe `payoff_table` (fila i = F(x_i*)) y
    `beta` (coeficientes convexos que ubican z_bar en la CHIM).
    """

    name = "normal_constraint"

    def scalarize(self, objectives: Sequence[ObjectiveFn], *,
                  payoff_table: Sequence[Sequence[float]],
                  beta: Sequence[float] = None,
                  representative_index: int = 0,
                  penalty: float = 1e6, **kwargs: Any) -> ScalarizationResult:
        n = len(objectives)
        if payoff_table is None or len(payoff_table) != n or any(len(row) != n for row in payoff_table):
            raise ValueError(f"`payoff_table` debe ser una matriz {n}x{n} (fila i = F(x_i*)).")
        if not (0 <= representative_index < n):
            raise ValueError(f"`representative_index` fuera de rango [0,{n-1}].")
        b = _normalize_weights(beta, n)
        z_bar = [sum(b[i] * payoff_table[i][j] for i in range(n)) for j in range(n)]
        z_star = [min(payoff_table[i][j] for i in range(n)) for j in range(n)]
        normal = [z_star[j] - z_bar[j] for j in range(n)]
        f_rep = objectives[representative_index]

        def phi(x):
            F = [f(x) for f in objectives]
            diff = [F[j] - z_bar[j] for j in range(n)]
            violation = sum(d * c for d, c in zip(diff, normal))
            return f_rep(x) + penalty * max(0.0, violation) ** 2

        return ScalarizationResult(phi, {
            "beta": b, "z_bar": z_bar, "z_star": z_star, "normal": normal,
            "representative_index": representative_index, "penalty": penalty,
        })
