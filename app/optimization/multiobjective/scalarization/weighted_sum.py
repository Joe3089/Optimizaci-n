# app/optimization/multiobjective/scalarization/weighted_sum.py
from __future__ import annotations

from typing import Any, Sequence

from .base import ObjectiveFn, ScalarizationResult, ScalarizationStrategy, _normalize_weights


class WeightedSumStrategy(ScalarizationStrategy):
    """phi(x) = sum_i w_i * f_i(x), con pesos w_i >= 0 normalizados a suma 1.

    La técnica más simple: solo garantiza cubrir el frente de Pareto completo
    cuando este es convexo.
    """

    name = "weighted_sum"

    def scalarize(self, objectives: Sequence[ObjectiveFn], *,
                  weights: Sequence[float] = None, **kwargs: Any) -> ScalarizationResult:
        n = len(objectives)
        w = _normalize_weights(weights, n)

        def phi(x):
            return sum(wi * f(x) for wi, f in zip(w, objectives))

        return ScalarizationResult(phi, {"weights": w})
