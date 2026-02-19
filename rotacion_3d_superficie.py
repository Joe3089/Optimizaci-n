"""rotacion_3d_superficie.py
Puente de compatibilidad.

interfaz_qt.py importa este módulo.
Redirige a la implementación real en rotacion_3d.py sin modificar nada visual.
"""

from rotacion_3d import Rotating3DCanvas  # noqa: F401

__all__ = ["Rotating3DCanvas"]
