# app/optimization/multiobjective/scalarization/registry.py
# Factory: selecciona una estrategia de escalarización por nombre sin que
# el resto del sistema conozca las clases concretas (Fase 6-8).
from __future__ import annotations

from typing import Dict, Type

from .base import ScalarizationStrategy
from .weighted_sum import WeightedSumStrategy
from .lexicographic import LexicographicStrategy
from .goal_programming import GoalProgrammingStrategy
from .epsilon_constraint import EpsilonConstraintStrategy
from .achievement_scalarizing import AchievementScalarizingStrategy
from .chebyshev import ChebyshevStrategy
from .nbi import NBIStrategy
from .normal_constraint import NormalConstraintStrategy
from .adaptive_weighted_sum import AdaptiveWeightedSumStrategy
from .reference_point import ReferencePointStrategy

STRATEGIES: Dict[str, Type[ScalarizationStrategy]] = {
    "weighted_sum":             WeightedSumStrategy,
    "lexicographic":            LexicographicStrategy,
    "goal_programming":         GoalProgrammingStrategy,
    "epsilon_constraint":       EpsilonConstraintStrategy,
    "achievement_scalarizing":  AchievementScalarizingStrategy,
    "chebyshev":                ChebyshevStrategy,
    "nbi":                      NBIStrategy,
    "normal_constraint":        NormalConstraintStrategy,
    "adaptive_weighted_sum":    AdaptiveWeightedSumStrategy,
    "reference_point":          ReferencePointStrategy,
}


def get_strategy(name: str) -> ScalarizationStrategy:
    """Devuelve una instancia de la estrategia registrada bajo `name`.

    Añadir una técnica nueva no requiere modificar esta función más que
    para registrarla aquí: implementar ScalarizationStrategy en su propio
    módulo y agregar una entrada a STRATEGIES.
    """
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(
            f"Estrategia de escalarización desconocida: {name!r}. "
            f"Disponibles: {sorted(STRATEGIES)}"
        )
    return cls()
