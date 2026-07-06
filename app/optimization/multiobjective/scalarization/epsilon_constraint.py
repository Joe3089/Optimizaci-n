# app/optimization/multiobjective/scalarization/epsilon_constraint.py
from __future__ import annotations

from typing import Any, Sequence

from .base import ObjectiveFn, ScalarizationResult, ScalarizationStrategy


class EpsilonConstraintStrategy(ScalarizationStrategy):
    """Optimiza un objetivo primario y convierte los demás en restricciones
    f_i(x) <= eps_i, vía penalización cuadrática exterior:

        phi(x) = f_primary(x) + penalty * sum_i max(0, f_i(x) - eps_i)^2

    Variar `epsilons` traza distintos puntos del frente de Pareto (incluso
    en frentes no convexos, a diferencia de Weighted Sum).
    """

    name = "epsilon_constraint"

    def scalarize(self, objectives: Sequence[ObjectiveFn], *,
                  primary_index: int = 0, epsilons: Sequence[float] = None,
                  penalty: float = 1e6, **kwargs: Any) -> ScalarizationResult:
        n = len(objectives)
        if not (0 <= primary_index < n):
            raise ValueError(f"`primary_index` fuera de rango [0,{n-1}].")
        others = [(i, f) for i, f in enumerate(objectives) if i != primary_index]
        if epsilons is None:
            raise ValueError("`epsilons` es obligatorio: un límite por cada objetivo "
                              "que no sea el primario.")
        if len(epsilons) != len(others):
            raise ValueError(f"Se esperaban {len(others)} épsilons (uno por objetivo "
                              "no primario), se recibieron {len(epsilons)}.")
        f_primary = objectives[primary_index]

        def phi(x):
            base = f_primary(x)
            violation = sum(max(0.0, f(x) - eps) ** 2
                            for (_, f), eps in zip(others, epsilons))
            return base + penalty * violation

        return ScalarizationResult(phi, {
            "primary_index": primary_index,
            "epsilons": list(epsilons),
            "penalty": penalty,
        })
