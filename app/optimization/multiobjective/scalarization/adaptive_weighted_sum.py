# app/optimization/multiobjective/scalarization/adaptive_weighted_sum.py
from __future__ import annotations

from typing import Any, Sequence

from .base import ObjectiveFn, ScalarizationResult, ScalarizationStrategy, _normalize_weights


class AdaptiveWeightedSumStrategy(ScalarizationStrategy):
    """Weighted Sum adaptativo (Kim & de Weck): parte de una suma ponderada
    normal y, cuando se dispone de soluciones ya encontradas del frente de
    Pareto (`neighbor_points`, vectores F(x) previos), añade una
    recompensa por alejarse de ellas — favoreciendo que la siguiente
    llamada explore huecos del frente en vez de re-muestrear puntos ya
    cubiertos:

        phi(x) = sum_i w_i*f_i(x)  -  curvature_penalty * min_j ||F(x)-p_j||^2

    El refinamiento adaptativo completo (múltiples pasadas que subdividen
    la región del hueco más grande) es responsabilidad del llamador: cada
    llamada a `scalarize` resuelve un único paso dada la información de
    vecinos disponible hasta el momento.
    """

    name = "adaptive_weighted_sum"

    def scalarize(self, objectives: Sequence[ObjectiveFn], *,
                  weights: Sequence[float] = None,
                  neighbor_points: Sequence[Sequence[float]] = None,
                  curvature_penalty: float = 0.0, **kwargs: Any) -> ScalarizationResult:
        n = len(objectives)
        w = _normalize_weights(weights, n)
        neighbors = [list(p) for p in neighbor_points] if neighbor_points else []

        def phi(x):
            base = sum(wi * f(x) for wi, f in zip(w, objectives))
            if neighbors and curvature_penalty:
                F = [f(x) for f in objectives]
                min_dist_sq = min(
                    sum((F[j] - p[j]) ** 2 for j in range(n)) for p in neighbors
                )
                base -= curvature_penalty * min_dist_sq
            return base

        return ScalarizationResult(phi, {
            "weights": w, "n_neighbors": len(neighbors), "curvature_penalty": curvature_penalty,
        })
