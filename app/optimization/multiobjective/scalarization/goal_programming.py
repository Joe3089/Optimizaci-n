# app/optimization/multiobjective/scalarization/goal_programming.py
from __future__ import annotations

from typing import Any, Sequence

from .base import ObjectiveFn, ScalarizationResult, ScalarizationStrategy, _normalize_weights


class GoalProgrammingStrategy(ScalarizationStrategy):
    """phi(x) = sum_i w_i * |f_i(x) - goal_i|^p

    Minimiza la desviación ponderada respecto a metas (`goals`) fijadas por
    el usuario para cada objetivo, en vez de minimizar los objetivos
    directamente. `p=1` (por defecto) da desviación absoluta; `p=2` da
    desviación cuadrática (penaliza más las desviaciones grandes).
    """

    name = "goal_programming"

    def scalarize(self, objectives: Sequence[ObjectiveFn], *,
                  goals: Sequence[float], weights: Sequence[float] = None,
                  p: int = 1, **kwargs: Any) -> ScalarizationResult:
        n = len(objectives)
        if goals is None or len(goals) != n:
            raise ValueError(f"Se esperaban {n} metas (`goals`).")
        w = _normalize_weights(weights, n)

        def phi(x):
            return sum(wi * abs(f(x) - gi) ** p
                       for wi, f, gi in zip(w, objectives, goals))

        return ScalarizationResult(phi, {"weights": w, "goals": list(goals), "p": p})
