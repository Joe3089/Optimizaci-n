# app/optimization/multiobjective/scalarization/
# Motor de escalarización multiobjetivo: transforma min F(x)=(f_1..f_m)
# en min phi(x) mediante una técnica seleccionable por nombre (Fase 6-8).
from .base import ScalarizationStrategy, ScalarizationResult
from .registry import STRATEGIES, get_strategy

__all__ = ["ScalarizationStrategy", "ScalarizationResult", "STRATEGIES", "get_strategy"]
