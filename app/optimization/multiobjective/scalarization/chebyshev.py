# app/optimization/multiobjective/scalarization/chebyshev.py
from __future__ import annotations

from typing import Any, Sequence

from .base import ObjectiveFn, ScalarizationResult, ScalarizationStrategy, _normalize_weights


class ChebyshevStrategy(ScalarizationStrategy):
    """Tchebycheff ponderado:

        phi(x) = max_i [ w_i * |f_i(x) - z_i*| ]

    `z*` es el punto ideal/utópico (z_i* = min_x f_i(x)). A diferencia de
    Weighted Sum, puede alcanzar cualquier punto Pareto-óptimo (incluso en
    frentes no convexos), variando los pesos `w`.
    """

    name = "chebyshev"

    def scalarize(self, objectives: Sequence[ObjectiveFn], *,
                  ideal_point: Sequence[float], weights: Sequence[float] = None,
                  **kwargs: Any) -> ScalarizationResult:
        n = len(objectives)
        if ideal_point is None or len(ideal_point) != n:
            raise ValueError(f"`ideal_point` debe tener {n} componentes.")
        w = _normalize_weights(weights, n)
        z = list(ideal_point)

        def phi(x):
            return max(wi * abs(f(x) - zi)
                       for wi, f, zi in zip(w, objectives, z))

        return ScalarizationResult(phi, {"ideal_point": z, "weights": w})
