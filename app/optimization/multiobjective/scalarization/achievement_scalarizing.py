# app/optimization/multiobjective/scalarization/achievement_scalarizing.py
from __future__ import annotations

from typing import Any, Sequence

from .base import ObjectiveFn, ScalarizationResult, ScalarizationStrategy, _normalize_weights


class AchievementScalarizingStrategy(ScalarizationStrategy):
    """Función de logro de Wierzbicki (ASF):

        phi(x) = max_i [ w_i * (f_i(x) - z_i) ] + rho * sum_i (f_i(x) - z_i)

    `z` es el punto de referencia (típicamente el punto utópico/ideal:
    z_i = min_x f_i(x)). El término de aumento `rho` (pequeño, >0) rompe
    empates y garantiza soluciones Pareto-óptimas (no solo débilmente
    Pareto-óptimas) incluso cuando el máximo por sí solo sería insuficiente.
    """

    name = "achievement_scalarizing"

    def scalarize(self, objectives: Sequence[ObjectiveFn], *,
                  reference_point: Sequence[float], weights: Sequence[float] = None,
                  rho: float = 1e-4, **kwargs: Any) -> ScalarizationResult:
        n = len(objectives)
        if reference_point is None or len(reference_point) != n:
            raise ValueError(f"`reference_point` debe tener {n} componentes.")
        w = _normalize_weights(weights, n)
        z = list(reference_point)

        def phi(x):
            diffs = [f(x) - zi for f, zi in zip(objectives, z)]
            return max(wi * d for wi, d in zip(w, diffs)) + rho * sum(diffs)

        return ScalarizationResult(phi, {
            "reference_point": z, "weights": w, "rho": rho,
        })
