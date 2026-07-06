# app/optimization/multiobjective/scalarization/base.py
# Interfaz común del motor de escalarización multiobjetivo (Fase 6-8).
#
# Cada estrategia transforma un problema multiobjetivo
#     min  F(x) = (f_1(x), ..., f_m(x))
# en un problema monoobjetivo
#     min  phi(x)
# devolviendo un callable escalar `phi` listo para pasar a cualquier
# optimizador monoobjetivo ya existente en app.optimization (heurísticos,
# metaheurísticos, line search, etc.) — el usuario elige la técnica por
# nombre (ver registry.py) sin modificar nada del resto del sistema.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Sequence

Vector = Sequence[float]
ObjectiveFn = Callable[[Vector], float]


@dataclass
class ScalarizationResult:
    """Resultado de escalarizar un problema multiobjetivo."""
    scalarized: Callable[[Vector], float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, x: Vector) -> float:
        return self.scalarized(x)


class ScalarizationStrategy(ABC):
    """Interfaz común de toda técnica de escalarización.

    Implementaciones concretas: WeightedSumStrategy, LexicographicStrategy,
    GoalProgrammingStrategy, EpsilonConstraintStrategy,
    AchievementScalarizingStrategy, ChebyshevStrategy, NBIStrategy,
    NormalConstraintStrategy, AdaptiveWeightedSumStrategy,
    ReferencePointStrategy.
    """

    name: str = "base"

    @abstractmethod
    def scalarize(self, objectives: Sequence[ObjectiveFn], **kwargs: Any) -> ScalarizationResult:
        """Construye phi(x) a partir de las funciones objetivo `objectives`.

        `kwargs` son los parámetros propios de cada técnica (pesos, metas,
        épsilons, punto de referencia, etc. — ver cada estrategia concreta).
        """
        raise NotImplementedError


def _normalize_weights(weights: Vector, n: int) -> list:
    if weights is None:
        return [1.0 / n] * n
    weights = list(weights)
    if len(weights) != n:
        raise ValueError(f"Se esperaban {n} pesos, se recibieron {len(weights)}.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("La suma de los pesos debe ser positiva.")
    return [w / total for w in weights]


def _evaluate_all(objectives: Sequence[ObjectiveFn], x: Vector) -> list:
    return [f(x) for f in objectives]
