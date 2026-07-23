# app/domain/inventory/validation.py
# Validación de parámetros de entrada para los modelos de inventario.
#
# Mismo estilo que app.application.func_compat: ValueError con mensaje en
# español + causa + recomendación, para que el `except Exception as e` que
# ya existe en la UI (app/ui/main_window.py, método `ejecutar`) lo muestre
# sin necesitar un widget de error nuevo.
from __future__ import annotations

from typing import List, Tuple


def _require_positive(value: float, name: str) -> None:
    if value is None or value <= 0:
        raise ValueError(
            f"El parámetro '{name}' debe ser un número mayor que 0 "
            f"(se recibió: {value}).\n\n"
            f"En los modelos de inventario, demandas y costos negativos o "
            f"nulos no tienen sentido físico y no se pueden optimizar."
        )


def validate_eoq_clasico(D: float, S: float, H: float) -> None:
    _require_positive(D, "Demanda anual (D)")
    _require_positive(S, "Costo de pedido (S)")
    _require_positive(H, "Costo de mantenimiento (H)")


def validate_eoq_backorders(D: float, S: float, H: float, B: float) -> None:
    validate_eoq_clasico(D, S, H)
    _require_positive(B, "Costo de faltante (B)")


def validate_eoq_descuentos(D: float, S: float, holding_cost_rate: float,
                            price_breaks: List[Tuple[float, float]]) -> None:
    _require_positive(D, "Demanda anual (D)")
    _require_positive(S, "Costo de pedido (S)")
    if holding_cost_rate is None or not (0 < holding_cost_rate < 1):
        raise ValueError(
            f"La tasa de costo de mantenimiento debe estar entre 0 y 1 "
            f"(p.ej. 0.2 = 20% del precio unitario por año). "
            f"Se recibió: {holding_cost_rate}.\n\n"
            f"Verifica que el valor esté expresado como fracción, no como "
            f"porcentaje entero (usa 0.2, no 20)."
        )
    if not price_breaks or len(price_breaks) < 2:
        raise ValueError(
            "Se necesitan al menos 2 tramos de precio (cantidad_mínima, "
            "precio_unitario) para un modelo de descuentos por cantidad.\n\n"
            "Si solo hay un precio fijo, usa el modelo 'EOQ Clásico' en su lugar."
        )
    qtys = [q for q, _ in price_breaks]
    prices = [p for _, p in price_breaks]
    if qtys != sorted(qtys) or len(set(qtys)) != len(qtys):
        raise ValueError(
            "Las cantidades mínimas de los tramos de precio deben ser "
            "estrictamente crecientes y sin repetir.\n\n"
            f"Se recibió: {qtys}. Ordénalas de menor a mayor."
        )
    if qtys[0] < 0:
        raise ValueError(
            "El primer tramo debe iniciar en una cantidad mínima ≥ 0.")
    if any(p <= 0 for p in prices):
        raise ValueError(
            "Todos los precios unitarios por tramo deben ser mayores que 0.")
    if prices != sorted(prices, reverse=True):
        raise ValueError(
            "Los precios unitarios deben ser decrecientes a medida que "
            "aumenta la cantidad mínima del tramo (a mayor volumen, menor "
            f"precio). Se recibió: {prices}.\n\n"
            "Verifica el orden de los tramos de descuento."
        )


def validate_rop_probabilistico(d_diario: float, sigma_d_diario: float,
                                lead_time_dias: float, nivel_servicio: float,
                                D_anual: float, S: float, H: float) -> None:
    validate_eoq_clasico(D_anual, S, H)
    _require_positive(d_diario, "Demanda diaria promedio (d̄)")
    _require_positive(lead_time_dias, "Tiempo de entrega / lead time (L)")
    if sigma_d_diario is None or sigma_d_diario < 0:
        raise ValueError(
            f"La desviación estándar de la demanda diaria (σ_d) no puede "
            f"ser negativa (se recibió: {sigma_d_diario})."
        )
    if nivel_servicio is None or not (0 < nivel_servicio < 1):
        raise ValueError(
            f"El nivel de servicio debe estar estrictamente entre 0 y 1 "
            f"(p.ej. 0.95 = 95%). Se recibió: {nivel_servicio}.\n\n"
            f"Valores de 0, 1 o fuera de ese rango no representan una "
            f"probabilidad válida y la inversa de la normal no está definida."
        )


def validate_inventory_inputs(modelo: str, **kwargs) -> None:
    """
    Punto de entrada único: valida los parámetros de `modelo` antes de
    construir la función de costo. Lanza ValueError con mensaje en español
    si algo es inválido (mismo patrón que func_compat.parse_function_1d).
    """
    validators = {
        "eoq_clasico": validate_eoq_clasico,
        "eoq_backorders": validate_eoq_backorders,
        "eoq_descuentos": validate_eoq_descuentos,
        "rop_probabilistico": validate_rop_probabilistico,
    }
    fn = validators.get(modelo)
    if fn is None:
        raise ValueError(f"Modelo de inventario desconocido: '{modelo}'.")
    fn(**kwargs)
