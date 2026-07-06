# app/optimization/multiobjective/scalarization/nbi.py
from __future__ import annotations

from typing import Any, Sequence

from .base import ObjectiveFn, ScalarizationResult, ScalarizationStrategy, _normalize_weights


class NBIStrategy(ScalarizationStrategy):
    """Normal Boundary Intersection (formulación escalar simplificada).

    El NBI clásico resuelve, para cada punto z_bar de la CHIM (envolvente
    convexa de los mínimos individuales), un subproblema con restricción de
    igualdad: maximizar t tal que F(x) = z_bar + t*n̂. Para exponerlo con la
    misma interfaz `scalarize(objectives) -> phi(x)` que el resto de
    estrategias (sin un solver bi-nivel), se resuelve t analíticamente como
    la proyección de F(x)-z_bar sobre la normal n̂, y phi(x) combina la
    distancia perpendicular a la recta normal (que se anula en la
    intersección buscada) con un término que favorece avanzar hacia el
    punto ideal a lo largo de esa recta:

        t(x)    = dot(F(x)-z_bar, n) / ||n||^2
        phi(x)  = ||(F(x)-z_bar) - t(x)*n||^2  -  eta * t(x)

    `payoff_table[i]` debe ser F(x_i*), el vector objetivo en el minimizador
    del objetivo i-ésimo (la tabla de pagos). `beta` son los coeficientes
    convexos que ubican z_bar sobre la CHIM (por defecto, equidistante).
    """

    name = "nbi"

    def scalarize(self, objectives: Sequence[ObjectiveFn], *,
                  payoff_table: Sequence[Sequence[float]],
                  beta: Sequence[float] = None, eta: float = 0.1,
                  **kwargs: Any) -> ScalarizationResult:
        n = len(objectives)
        if payoff_table is None or len(payoff_table) != n or any(len(row) != n for row in payoff_table):
            raise ValueError(f"`payoff_table` debe ser una matriz {n}x{n} (fila i = F(x_i*)).")
        b = _normalize_weights(beta, n)
        z_bar = [sum(b[i] * payoff_table[i][j] for i in range(n)) for j in range(n)]
        z_star = [min(payoff_table[i][j] for i in range(n)) for j in range(n)]
        normal = [z_star[j] - z_bar[j] for j in range(n)]
        norm_sq = sum(c * c for c in normal) or 1.0

        def phi(x):
            F = [f(x) for f in objectives]
            diff = [F[j] - z_bar[j] for j in range(n)]
            t = sum(d * c for d, c in zip(diff, normal)) / norm_sq
            perp_sq = sum((diff[j] - t * normal[j]) ** 2 for j in range(n))
            return perp_sq - eta * t

        return ScalarizationResult(phi, {
            "beta": b, "z_bar": z_bar, "z_star": z_star, "normal": normal, "eta": eta,
        })
