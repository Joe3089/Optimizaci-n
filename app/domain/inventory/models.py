# app/domain/inventory/models.py
# Generadores de problema para los 4 modelos de inventario soportados.
#
# Cada función devuelve un dict con una expresión simbólica (`expr`, string
# parseable por app.application.func_compat, exactamente igual que una
# función escrita a mano por el usuario) más metadatos de dominio. De ahí
# en adelante el problema fluye SIN CAMBIOS por el mismo
# app.application.optimization_service que resuelve cualquier otra función
# del Optimizador — no hay un solver de inventario aparte.
#
# NOTA sobre nombres de variable: la cantidad de pedido se representa con
# el símbolo "q" (minúscula), NO "Q". Varios solvers nD del proyecto
# (Armijo/Wolfe en app/optimization/line_search/nd_symbolic.py y los
# wrappers de MD) llaman sympy.sympify(func_str) SIN pasar un diccionario
# `locals` que reemplace los nombres globales de sympy — y `sympy.Q` es el
# objeto real de aserciones de sympy (Q.positive, etc.), no un símbolo
# libre. Usar "Q" como variable ahí colisiona con ese global y revienta
# con `TypeError: unsupported operand type(s) for /: 'Integer' and
# 'AssumptionKeys'`. "q" minúscula no choca con ningún global de sympy.
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from app.domain.inventory.validation import (
    validate_eoq_clasico, validate_eoq_backorders,
    validate_eoq_descuentos, validate_rop_probabilistico,
)


def _z_from_service_level(nivel_servicio: float) -> float:
    """
    z tal que Φ(z) = nivel_servicio (inversa de la normal estándar).
    Usa scipy (ya es dependencia del proyecto — la usan los métodos MD/BFGS
    en app/optimization/constrained/penalty_barrier.py) y cae a la
    aproximación racional de Acklam si scipy no está disponible en tiempo
    de ejecución.
    """
    try:
        from scipy.stats import norm
        return float(norm.ppf(nivel_servicio))
    except Exception:
        # Aproximación de Acklam (error < 1.15e-9), sin dependencias.
        p = nivel_servicio
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        p_low, p_high = 0.02425, 1 - 0.02425
        if p < p_low:
            qq = math.sqrt(-2 * math.log(p))
            return (((((c[0]*qq+c[1])*qq+c[2])*qq+c[3])*qq+c[4])*qq+c[5]) / \
                   ((((d[0]*qq+d[1])*qq+d[2])*qq+d[3])*qq+1)
        if p > p_high:
            qq = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*qq+c[1])*qq+c[2])*qq+c[3])*qq+c[4])*qq+c[5]) / \
                    ((((d[0]*qq+d[1])*qq+d[2])*qq+d[3])*qq+1)
        qq = p - 0.5
        r = qq * qq
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*qq / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ─── 1. EOQ clásico (determinista) ───────────────────────────────────────
def eoq_clasico(D: float, S: float, H: float) -> Dict[str, Any]:
    """
    Cantidad Económica de Pedido clásica.
    C(q) = (D/q)·S + (q/2)·H
    q* = √(2DS/H)  (fórmula cerrada, usada solo para fijar límites/validar
    la convergencia numérica — el óptimo real lo calcula el método elegido).
    """
    validate_eoq_clasico(D, S, H)
    expr = f"({D}/q)*{S} + (q/2)*{H}"
    q_analitico = math.sqrt(2 * D * S / H)
    costo_analitico = (D / q_analitico) * S + (q_analitico / 2) * H
    return {
        "modelo": "eoq_clasico",
        "titulo": "EOQ Clásico (Cantidad Económica de Pedido)",
        "expr": expr,
        "gx": "",
        "var_names": ["q"],
        "bounds": {"q": (max(q_analitico * 0.05, 1e-6), q_analitico * 4)},
        "x0": [max(q_analitico * 0.5, 1e-3)],
        "restricciones": ("q > 0. Demanda D, costo de pedido S y costo de "
                          "mantenimiento H constantes y conocidos (modelo "
                          "determinista, sin faltantes)."),
        "metodos_recomendados": [
            "Búsqueda Local", "Fibonacci", "Sección Áurea", "Sección Dorada",
            "Goldstein", "Newton-Raphson", "Grad. Conjugado", "Armijo", "Wolfe",
        ],
        "metodos_incompatibles": [
            "Los 6 métodos MD (requieren restricción g(x) explícita; "
            "este modelo no tiene restricciones activas)",
            "Los 14 métodos multiobjetivo (requieren 2 funciones objetivo; "
            "este modelo tiene una sola función de costo)",
        ],
        "extra_results": {
            "q* (analítico)": round(q_analitico, 6),
            "Costo total (analítico)": round(costo_analitico, 6),
        },
        "dashboard_labels": {"q": "Cantidad óptima de pedido (q*)"},
    }


# ─── 2. EOQ con faltantes planeados (backorders) ─────────────────────────
def eoq_backorders(D: float, S: float, H: float, B: float) -> Dict[str, Any]:
    """
    EOQ con déficit planeado.
    C(q,b) = (D/q)·S + H·(q−b)²/(2q) + B·b²/(2q)
    q, b > 0 con b ≤ q (el faltante no puede superar el lote pedido).
    Óptimo analítico: q*=√(2DS/H · (H+B)/B), b*=q*·H/(H+B).
    """
    validate_eoq_backorders(D, S, H, B)
    expr = (f"({D}/q)*{S} + {H}*(q-b)**2/(2*q) + {B}*b**2/(2*q)")
    q_analitico = math.sqrt((2 * D * S / H) * (H + B) / B)
    b_analitico = q_analitico * (H / (H + B))
    costo_analitico = (D / q_analitico) * S + H * (q_analitico - b_analitico) ** 2 / (2 * q_analitico) \
        + B * b_analitico ** 2 / (2 * q_analitico)
    return {
        "modelo": "eoq_backorders",
        "titulo": "EOQ con Faltantes Planeados (Backorders)",
        "expr": expr,
        "gx": "b - q",   # restricción g(x) <= 0  →  b <= q, usada por los métodos MD
        "var_names": ["q", "b"],
        "bounds": {"q": (max(q_analitico * 0.05, 1e-6), q_analitico * 4),
                   "b": (0.0, q_analitico * 1.2)},
        "x0": [max(q_analitico * 0.5, 1e-3), max(b_analitico * 0.5, 1e-3)],
        "restricciones": ("q > 0, 0 ≤ b ≤ q (el faltante planeado no puede "
                          "superar la cantidad pedida). 2 variables de "
                          "decisión: q (lote) y b (faltante máximo)."),
        "metodos_recomendados": [
            "Armijo", "Wolfe", "MD: Penalización (Newton)", "MD: Barreras (Newton)",
            "MD: Pen. (BFGS)", "MD: Pes. (BFGS)", "MD: Pes. (Nelder-Mead)",
            "MD: Sum. (BFGS)",
        ],
        "metodos_incompatibles": [
            "Búsqueda Local, Fibonacci, Sección Áurea/Dorada (métodos "
            "estrictamente 1D — este modelo tiene 2 variables: q y b)",
            "Los 14 métodos multiobjetivo (requieren 2 funciones objetivo; "
            "este modelo tiene una sola función de costo)",
        ],
        "extra_results": {
            "q* (analítico)": round(q_analitico, 6),
            "b* (analítico)": round(b_analitico, 6),
            "Costo total (analítico)": round(costo_analitico, 6),
        },
        "dashboard_labels": {"q": "Cantidad óptima de pedido (q*)",
                             "b": "Faltante máximo permitido (b*)"},
    }


# ─── 3. EOQ con descuentos por cantidad ───────────────────────────────────
def eoq_descuentos(D: float, S: float, holding_cost_rate: float,
                   price_breaks: List[Tuple[float, float]]) -> Dict[str, Any]:
    """
    EOQ con descuentos por volumen (función de costo por tramos).
    Para cada tramo i con precio unitario p_i válido en [q_i, q_{i+1}):
        C_i(q) = (D/q)·S + (q/2)·(holding_cost_rate·p_i) + D·p_i
    `price_breaks`: lista [(cantidad_mínima, precio_unitario), ...] ordenada
    ascendentemente por cantidad mínima (la primera debe empezar en 0 o en
    el pedido mínimo permitido).
    """
    validate_eoq_descuentos(D, S, holding_cost_rate, price_breaks)
    breaks = sorted(price_breaks, key=lambda t: t[0])

    pieces = []
    for i, (min_q, price) in enumerate(breaks):
        upper = breaks[i + 1][0] if i + 1 < len(breaks) else None
        cost_i = f"({D}/q)*{S} + (q/2)*({holding_cost_rate}*{price}) + {D}*{price}"
        if upper is None:
            cond = f"(q >= {min_q})"
        else:
            cond = f"((q >= {min_q}) & (q < {upper}))"
        pieces.append(f"({cost_i}, {cond})")
    expr = "Piecewise(" + ", ".join(pieces) + ")"

    # Óptimo de referencia: mejor q entre los q-EOQ "sin restricción" de cada
    # tramo (factible dentro de su propio rango), usado solo para acotar
    # bounds/x0 razonables — el método elegido calcula el óptimo real.
    candidatos = []
    for i, (min_q, price) in enumerate(breaks):
        h_i = holding_cost_rate * price
        q_i = math.sqrt(2 * D * S / h_i) if h_i > 0 else min_q
        upper = breaks[i + 1][0] if i + 1 < len(breaks) else q_i * 4
        q_factible = min(max(q_i, min_q), upper if upper > min_q else q_i)
        costo_i = (D / q_factible) * S + (q_factible / 2) * h_i + D * price
        candidatos.append((q_factible, costo_i))
    q_ref, _ = min(candidatos, key=lambda t: t[1])

    return {
        "modelo": "eoq_descuentos",
        "titulo": "EOQ con Descuentos por Cantidad",
        "expr": expr,
        "gx": "",
        "var_names": ["q"],
        "bounds": {"q": (max(breaks[0][0], 1e-6), breaks[-1][0] * 3 + q_ref)},
        "x0": [max(q_ref, breaks[0][0] + 1e-3)],
        "restricciones": (f"q > 0, función por tramos con {len(breaks)} "
                          "niveles de precio (no derivable en los puntos "
                          "de quiebre entre tramos)."),
        "metodos_recomendados": [
            "Búsqueda Local", "SA: Recocido Simulado", "PSO: Enjambre",
            "GA: Algoritmo Genético",
        ],
        "metodos_incompatibles": [
            "Newton-Raphson, Goldstein, Armijo, Wolfe, Grad. Conjugado "
            "(requieren derivadas continuas; la función tiene "
            "discontinuidades de pendiente en los quiebres de tramo)",
            "Fibonacci, Sección Áurea/Dorada (asumen unimodalidad; los "
            "saltos de precio entre tramos pueden romper esa suposición)",
            "Los 6 métodos MD y los 14 multiobjetivo (misma razón que EOQ clásico)",
        ],
        "extra_results": {"q de referencia (heurístico por tramo)": round(q_ref, 6)},
        "dashboard_labels": {"q": "Cantidad óptima de pedido (q*)"},
    }


# ─── 4. Revisión periódica / punto de reorden (demanda probabilística) ────
def rop_probabilistico(d_diario: float, sigma_d_diario: float, lead_time_dias: float,
                       nivel_servicio: float, D_anual: float, S: float,
                       H: float) -> Dict[str, Any]:
    """
    Punto de Reorden (ROP) con demanda probabilística durante el lead time.
    ROP = μ_L + z·σ_L,  μ_L = d̄·L,  σ_L = σ_d·√L,  Φ(z) = nivel_servicio.
    El ROP se calcula analíticamente (no requiere optimizador); q* se sigue
    optimizando con la misma función de costo del EOQ clásico sobre D_anual.
    """
    validate_rop_probabilistico(d_diario, sigma_d_diario, lead_time_dias,
                                nivel_servicio, D_anual, S, H)
    z = _z_from_service_level(nivel_servicio)
    mu_L = d_diario * lead_time_dias
    sigma_L = sigma_d_diario * math.sqrt(lead_time_dias)
    stock_seguridad = z * sigma_L
    rop = mu_L + stock_seguridad

    base = eoq_clasico(D_anual, S, H)
    base.update({
        "modelo": "rop_probabilistico",
        "titulo": "Revisión Periódica / Punto de Reorden (ROP) Probabilístico",
        "restricciones": (base["restricciones"] + " Demanda diaria con media "
                          "d̄ y desviación σ_d; nivel de servicio en (0,1) "
                          "determina z (inversa de la normal estándar)."),
        "extra_results": {
            **base["extra_results"],
            "z (nivel de servicio)": round(z, 4),
            "Demanda esperada en lead time (μ_L)": round(mu_L, 4),
            "Desviación en lead time (σ_L)": round(sigma_L, 4),
            "Stock de seguridad (SS)": round(stock_seguridad, 4),
            "Punto de reorden (ROP)": round(rop, 4),
        },
        "dashboard_labels": {**base["dashboard_labels"], "ROP": "Punto de reorden"},
    })
    return base


MODEL_BUILDERS = {
    "eoq_clasico": eoq_clasico,
    "eoq_backorders": eoq_backorders,
    "eoq_descuentos": eoq_descuentos,
    "rop_probabilistico": rop_probabilistico,
}
