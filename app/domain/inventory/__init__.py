# app/domain/inventory/ — Módulo de Inventario del Optimizador.
#
# No es una app aparte: genera expresiones simbólicas (str) que se parsean
# y resuelven con el MISMO motor que cualquier otra función del Optimizador
# (app.application.func_compat + app.application.optimization_service).
# Aquí solo vive la lógica de dominio: qué función de costo corresponde a
# cada modelo de inventario, qué parámetros son válidos y qué métodos del
# optimizador aplican a cada uno.
from __future__ import annotations

from app.domain.inventory.models import (
    eoq_clasico,
    eoq_backorders,
    eoq_descuentos,
    rop_probabilistico,
    MODEL_BUILDERS,
)
from app.domain.inventory.validation import validate_inventory_inputs
from app.domain.inventory.catalog import (
    INVENTORY_FUNCTIONS, INVENTORY_CATEGORY, INVENTORY_REPORT_FIELDS,
    INVENTORY_MODEL_LABELS, INVENTORY_LABEL_TO_KEY, INVENTORY_METHOD_LABELS,
)

__all__ = [
    "eoq_clasico", "eoq_backorders", "eoq_descuentos", "rop_probabilistico",
    "MODEL_BUILDERS", "validate_inventory_inputs",
    "INVENTORY_FUNCTIONS", "INVENTORY_CATEGORY", "INVENTORY_REPORT_FIELDS",
    "INVENTORY_MODEL_LABELS", "INVENTORY_LABEL_TO_KEY", "INVENTORY_METHOD_LABELS",
]
