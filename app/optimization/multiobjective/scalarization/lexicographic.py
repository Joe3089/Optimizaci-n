# app/optimization/multiobjective/scalarization/lexicographic.py
from __future__ import annotations

from typing import Any, Sequence

from .base import ObjectiveFn, ScalarizationResult, ScalarizationStrategy


class LexicographicStrategy(ScalarizationStrategy):
    """Ordena los objetivos por prioridad y optimiza uno a la vez: el
    objetivo de mayor prioridad domina; los siguientes solo desempatan.

    Se aproxima con una suma ponderada de potencias decrecientes de `eps`
    (eps -> 0 recupera el orden lexicográfico exacto):
        phi(x) = f_(1)(x) + eps * f_(2)(x) + eps^2 * f_(3)(x) + ...
    donde (1),(2),... es el orden de `priority` (o el orden de `objectives`
    si no se especifica). Es la formulación estándar para convertir un
    orden lexicográfico en un único escalar continuo optimizable con
    métodos de descenso, evitando resolver m subproblemas anidados con
    restricciones de igualdad.
    """

    name = "lexicographic"

    def scalarize(self, objectives: Sequence[ObjectiveFn], *,
                  priority: Sequence[int] = None, eps: float = 1e-4,
                  **kwargs: Any) -> ScalarizationResult:
        n = len(objectives)
        order = list(priority) if priority is not None else list(range(n))
        if sorted(order) != list(range(n)):
            raise ValueError(f"`priority` debe ser una permutación de 0..{n-1}.")
        ordered = [objectives[i] for i in order]

        def phi(x):
            total = 0.0
            factor = 1.0
            for f in ordered:
                total += factor * f(x)
                factor *= eps
            return total

        return ScalarizationResult(phi, {"priority": order, "eps": eps})
