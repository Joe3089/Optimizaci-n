"""canvas_2d_estilo_fixed.py
Puente de compatibilidad.

interfaz_qt.py importa este módulo.
Redirige a la implementación real en canvas_2d.py sin modificar nada visual.
"""

from .canvas_2d import Function2DCanvas  # noqa: F401

__all__ = ["Function2DCanvas"]
