# app/optimization/multiobjective/scalarization/reference_point.py
from __future__ import annotations

from typing import Any, Sequence

from .base import ObjectiveFn, ScalarizationResult, ScalarizationStrategy, _normalize_weights


class ReferencePointStrategy(ScalarizationStrategy):
    """Método del punto de referencia (Wierzbicki): igual formulación que
    la función de logro (ASF), pero `reference_point` representa una
    aspiración del usuario (deseada, no necesariamente alcanzable ni el
    punto ideal) que normalmente se ajusta de forma iterativa entre
    corridas según el resultado obtenido:

        phi(x) = max_i [ w_i*(f_i(x) - z_ref_i) ] + rho * sum_i (f_i(x) - z_ref_i)

    Si `z_ref` es alcanzable, el óptimo de phi domina a `z_ref`; si no lo
    es, el óptimo es el punto Pareto más cercano a `z_ref` en el sentido de
    esta métrica.
    """

    name = "reference_point"

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

        return ScalarizationResult(phi, {"reference_point": z, "weights": w, "rho": rho})
