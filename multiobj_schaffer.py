"""
multiobj_schaffer.py
====================
Módulo de optimización multiobjetivo.

Implementa:
  1. Problema de Schaffer clásico:  min F(x) = (f1(x)=x², f2(x)=(x-2)²)
  2. Método de Escalarización (Suma Ponderada):
         f(x, α) = α·x² + (1-α)·(x-2)²     α ∈ [0,1]
  3. Jacobiano de F(x):  J(x) = [2x,  2(x-2)]
  4. Método de Newton-Jacobiano para la función escalarizada
  5. Cálculo del Frente de Pareto completo
  6. Wrapper genérico:  función mono-objetivo → multi-objetivo
     (descompone  f(x) = α·f(x) + (1-α)·g(x)  donde  g(x) = (x-x*_approx)²)

Dependencias: numpy (sympy opcional para funciones generales)
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# ── Sympy opcional ────────────────────────────────────────────────────────────
try:
    import sympy as sp
    _SYMPY = True
except ImportError:
    sp = None
    _SYMPY = False


# ══════════════════════════════════════════════════════════════════════════════
#  1.  PROBLEMA DE SCHAFFER CLÁSICO
# ══════════════════════════════════════════════════════════════════════════════

def f1_schaffer(x: float) -> float:
    """Primera función objetivo de Schaffer: f1(x) = x²"""
    return float(x) ** 2


def f2_schaffer(x: float) -> float:
    """Segunda función objetivo de Schaffer: f2(x) = (x-2)²"""
    return (float(x) - 2.0) ** 2


def schaffer_vector(x: float) -> Tuple[float, float]:
    """Evalúa el vector F(x) = (f1(x), f2(x))"""
    return f1_schaffer(x), f2_schaffer(x)


# ── Jacobiano ─────────────────────────────────────────────────────────────────

def jacobian_schaffer(x: float) -> np.ndarray:
    """
    Jacobiano de F(x) = [f1(x), f2(x)]ᵀ respecto a x.

        J(x) = [df1/dx]  =  [ 2x      ]
               [df2/dx]     [ 2(x-2)  ]

    Devuelve array de forma (2,).
    """
    x = float(x)
    return np.array([2.0 * x, 2.0 * (x - 2.0)])


# ── Escalarización ────────────────────────────────────────────────────────────

def scalarize_schaffer(x: float, alpha: float) -> float:
    """
    Función escalarizada de Schaffer:
        f(x, α) = α·x² + (1-α)·(x-2)²
    """
    a = float(alpha)
    return a * x ** 2 + (1.0 - a) * (x - 2.0) ** 2


def scalarize_deriv(x: float, alpha: float) -> float:
    """
    Derivada de f(x,α) respecto a x:
        f'(x,α) = 2α·x + 2(1-α)·(x-2)
    """
    a = float(alpha)
    return 2.0 * a * x + 2.0 * (1.0 - a) * (x - 2.0)


def scalarize_deriv2(_x: float, alpha: float) -> float:
    """
    Segunda derivada (constante):
        f''(x,α) = 2α + 2(1-α) = 2
    """
    return 2.0


def exact_minimum_scalarized(alpha: float) -> float:
    """
    Mínimo exacto de la función escalarizada:
        x*(α) = 2(1-α)
    (solución analítica, converge en 1 iteración Newton)
    """
    return 2.0 * (1.0 - float(alpha))


# ══════════════════════════════════════════════════════════════════════════════
#  2.  NEWTON-JACOBIANO (Newton 1D aplicado a la escalarización)
# ══════════════════════════════════════════════════════════════════════════════

def newton_jacobiano_schaffer(
    alpha: float,
    x0: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> List[Dict[str, Any]]:
    """
    Aplica el método de Newton usando el Jacobiano de la función escalarizada
    de Schaffer: minimiza f(x,α) = α·x² + (1-α)·(x-2)²

    Iteración:
        x_{k+1} = x_k - f'(x_k,α) / f''(x_k,α)

    El Jacobiano de F(x) en cada iteración se registra en el historial.

    Parámetros
    ----------
    alpha : peso α ∈ [0,1]
    x0    : punto inicial
    tol   : tolerancia en |f'|
    max_iter : iteraciones máximas

    Retorna
    -------
    List[dict] con claves:
        k, x_k, f_k, grad_k, step_k, J_f1, J_f2, converged
    """
    history: List[Dict[str, Any]] = []
    x = float(x0)
    a = float(alpha)

    for k in range(max_iter):
        fval  = scalarize_schaffer(x, a)
        grad  = scalarize_deriv(x, a)
        hess  = scalarize_deriv2(x, a)          # = 2.0 siempre
        J     = jacobian_schaffer(x)            # [2x, 2(x-2)]

        step  = -grad / hess if abs(hess) > 1e-14 else 0.0
        x_new = x + step

        history.append({
            "k":         k,
            "x_k":       round(x, 8),
            "f_k":       round(fval, 8),
            "grad_k":    round(grad, 8),
            "step_k":    round(step, 8),
            "J_f1":      round(float(J[0]), 6),
            "J_f2":      round(float(J[1]), 6),
            "f1(x_k)":   round(f1_schaffer(x), 6),
            "f2(x_k)":   round(f2_schaffer(x), 6),
            "converged": abs(grad) < tol,
        })

        if abs(grad) < tol:
            break
        x = x_new

    return history


# ══════════════════════════════════════════════════════════════════════════════
#  3.  FRENTE DE PARETO
# ══════════════════════════════════════════════════════════════════════════════

def pareto_front_schaffer(n_alpha: int = 200) -> Dict[str, Any]:
    """
    Calcula el Frente de Pareto del problema de Schaffer barriendo α ∈ [0,1].

    El Frente de Pareto exacto corresponde a x* ∈ [0, 2]:
        Pareto: {(x², (x-2)²) : x ∈ [0,2]}

    Parámetros
    ----------
    n_alpha : número de puntos en el frente

    Retorna
    -------
    dict con:
        alphas    : array de α usados
        x_star    : array de x*(α) = 2(1-α)
        f1_vals   : f1(x*(α)) = (2(1-α))²  = 4(1-α)²
        f2_vals   : f2(x*(α)) = (2(1-α)-2)² = 4α²
        history   : lista de dicts (una fila por α)
    """
    alphas  = np.linspace(0.0, 1.0, n_alpha)
    x_star  = 2.0 * (1.0 - alphas)               # x*(α) = 2(1-α)
    f1_vals = x_star ** 2                         # f1 = x²
    f2_vals = (x_star - 2.0) ** 2                 # f2 = (x-2)²

    history = []
    for i, a in enumerate(alphas):
        history.append({
            "k":      i,
            "α":      round(float(a), 4),
            "x*(α)":  round(float(x_star[i]), 6),
            "f1(x*)": round(float(f1_vals[i]), 6),
            "f2(x*)": round(float(f2_vals[i]), 6),
        })

    return {
        "alphas":  alphas,
        "x_star":  x_star,
        "f1_vals": f1_vals,
        "f2_vals": f2_vals,
        "history": history,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4.  TABLA DE CASOS (α = 0, 1/5, 1/4, 1/3, 1/2, 1)
# ══════════════════════════════════════════════════════════════════════════════

def tabla_alphas() -> List[Dict[str, Any]]:
    """
    Genera la tabla de casos de estudio del problema de Schaffer para
    α ∈ {0, 1/5, 1/4, 1/3, 1/2, 1}.

    Devuelve lista de dicts con columnas:
        α,  función escalarizada,  x*(α),  f1(x*),  f2(x*)
    """
    from fractions import Fraction

    casos = [
        (0,    "0",   "(x-2)²"),
        (1/5,  "1/5", "⅕x² + ⅘(x-2)²"),
        (1/4,  "1/4", "¼x² + ¾(x-2)²"),
        (1/3,  "1/3", "⅓x² + ⅔(x-2)²"),
        (1/2,  "1/2", "½x² + ½(x-2)²"),
        (1,    "1",   "x²"),
    ]

    rows = []
    for a_val, a_str, expr in casos:
        xopt = exact_minimum_scalarized(a_val)
        rows.append({
            "α":             a_str,
            "f(x,α)":        expr,
            "x*(α) = 2(1-α)": round(xopt, 6),
            "f1(x*) = x*²":  round(f1_schaffer(xopt), 6),
            "f2(x*)=(x*-2)²": round(f2_schaffer(xopt), 6),
            "f_escalar":     round(scalarize_schaffer(xopt, a_val), 8),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  5.  WRAPPER GENÉRICO: MONO → MULTI (Escalarización generalizada)
# ══════════════════════════════════════════════════════════════════════════════

def mono_to_multi_newton(
    func_expr: str,
    var_name: str,
    alpha: float,
    x0: float = 0.0,
    lo: float = -10.0,
    hi: float = 10.0,
    tol: float = 1e-8,
    max_iter: int = 80,
) -> List[Dict[str, Any]]:
    """
    Convierte una función mono-objetivo f(x) en problema multi-objetivo
    usando escalarización:

        F(x) = (f1(x), f2(x))  donde:
            f1(x) = f(x)                     (función original)
            f2(x) = (x - x_centro)²          (función cuadrática complementaria)

        Función escalarizada:
            φ(x, α) = α·f1(x) + (1-α)·f2(x)

    Optimiza φ usando Newton-Raphson con diferencias finitas.

    Parámetros
    ----------
    func_expr : expresión en sympy/numpy (ej. "x**2 - 4*x + 3")
    var_name  : nombre de la variable (ej. "x")
    alpha     : peso α ∈ [0,1]
    x0        : punto inicial
    lo, hi    : rango de búsqueda
    tol       : tolerancia en |φ'|
    max_iter  : iteraciones máximas

    Retorna
    -------
    List[dict] — historial de iteraciones con claves:
        k, x_k, f1_k, f2_k, phi_k, grad_phi, step_k, J_f1, J_f2, converged
    """
    if not _SYMPY:
        raise RuntimeError("SymPy es necesario para funciones generales.")

    # Construir callables
    sym_x   = sp.Symbol(var_name)
    sym_f   = sp.sympify(func_expr)
    sym_df  = sp.diff(sym_f, sym_x)
    sym_d2f = sp.diff(sym_df, sym_x)

    f_call   = sp.lambdify(sym_x, sym_f,  modules=["numpy"])
    df_call  = sp.lambdify(sym_x, sym_df, modules=["numpy"])
    d2f_call = sp.lambdify(sym_x, sym_d2f, modules=["numpy"])

    # Función complementaria centrada en el punto medio del rango
    x_c  = (lo + hi) / 2.0
    f2   = lambda xv: (float(xv) - x_c) ** 2
    df2  = lambda xv: 2.0 * (float(xv) - x_c)
    d2f2 = lambda _xv: 2.0

    a   = float(alpha)
    x   = float(x0)
    hist: List[Dict[str, Any]] = []

    for k in range(max_iter):
        try:
            f1v  = float(f_call(x))
            f2v  = float(f2(x))
            phi  = a * f1v + (1.0 - a) * f2v
            dphi = a * float(df_call(x)) + (1.0 - a) * float(df2(x))
            d2ph = a * float(d2f_call(x)) + (1.0 - a) * float(d2f2(x))
            # Jacobiano del vector F
            J_f1 = float(df_call(x))
            J_f2 = float(df2(x))
        except Exception as e:
            hist.append({"k": k, "error": str(e)})
            break

        step = -dphi / d2ph if abs(d2ph) > 1e-14 else 0.0
        x_new = x + step
        # Proyectar al rango
        x_new = max(lo, min(hi, x_new))

        hist.append({
            "k":         k,
            "x_k":       round(x, 8),
            "f1(x_k)":   round(f1v, 6),
            "f2(x_k)":   round(f2v, 6),
            "φ(x_k,α)":  round(phi, 6),
            "grad_φ":    round(dphi, 8),
            "step_k":    round(step, 8),
            "J_f1(x_k)": round(J_f1, 6),
            "J_f2(x_k)": round(J_f2, 6),
            "converged": abs(dphi) < tol,
        })

        if abs(dphi) < tol:
            break
        x = x_new

    return hist


# ══════════════════════════════════════════════════════════════════════════════
#  6.  PARETO FRONT GENÉRICO (para función arbitraria via escalarización)
# ══════════════════════════════════════════════════════════════════════════════

def pareto_front_generic(
    func_expr: str,
    var_name: str,
    x0: float = 0.0,
    lo: float = -10.0,
    hi: float = 10.0,
    n_alpha: int = 50,
) -> Dict[str, Any]:
    """
    Aproxima el frente de Pareto de (f1(x), f2(x)) barriendo α ∈ [0,1]
    donde f1(x) = func_expr y f2(x) = (x - x_centro)².

    Parámetros
    ----------
    func_expr : expresión de la función original
    var_name  : nombre de la variable
    x0, lo, hi: punto inicial y rango
    n_alpha   : número de puntos α

    Retorna
    -------
    dict con:
        alphas, x_star, f1_vals, f2_vals, history
    """
    if not _SYMPY:
        raise RuntimeError("SymPy es necesario.")

    alphas  = np.linspace(0.0, 1.0, n_alpha)
    x_stars = []
    f1_vals = []
    f2_vals = []
    history = []

    sym_x   = sp.Symbol(var_name)
    sym_f   = sp.sympify(func_expr)
    sym_df  = sp.diff(sym_f, sym_x)
    sym_d2f = sp.diff(sym_df, sym_x)
    f_call   = sp.lambdify(sym_x, sym_f,   modules=["numpy"])
    df_call  = sp.lambdify(sym_x, sym_df,  modules=["numpy"])
    d2f_call = sp.lambdify(sym_x, sym_d2f, modules=["numpy"])
    x_c  = (lo + hi) / 2.0
    f2   = lambda xv: (float(xv) - x_c) ** 2
    df2  = lambda xv: 2.0 * (float(xv) - x_c)
    d2f2 = lambda _xv: 2.0

    for i, a in enumerate(alphas):
        x = float(x0)
        for _ in range(100):
            try:
                dphi = a * float(df_call(x)) + (1.0 - a) * float(df2(x))
                d2ph = a * float(d2f_call(x)) + (1.0 - a) * float(d2f2(x))
                if abs(d2ph) < 1e-14:
                    break
                x_new = x - dphi / d2ph
                x_new = max(lo, min(hi, x_new))
                if abs(x_new - x) < 1e-10:
                    x = x_new
                    break
                x = x_new
            except Exception:
                break

        try:
            f1v = float(f_call(x))
            f2v = float(f2(x))
        except Exception:
            f1v = f2v = float("nan")

        x_stars.append(x)
        f1_vals.append(f1v)
        f2_vals.append(f2v)
        history.append({
            "k":      i,
            "α":      round(float(a), 4),
            "x*(α)":  round(x, 6),
            "f1(x*)": round(f1v, 6),
            "f2(x*)": round(f2v, 6),
        })

    return {
        "alphas":  alphas,
        "x_star":  np.array(x_stars),
        "f1_vals": np.array(f1_vals),
        "f2_vals": np.array(f2_vals),
        "history": history,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  7.  ANÁLISIS JACOBIANO PUNTO A PUNTO
# ══════════════════════════════════════════════════════════════════════════════

def jacobian_analysis_schaffer(
    x_values: Optional[List[float]] = None,
    n_points: int = 11,
    lo: float = -1.0,
    hi: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    Evalúa el Jacobiano de F(x)=(f1,f2) en una lista de puntos.

    Si x_values es None, usa n_points puntos equiespaciados en [lo, hi].

    Retorna lista de dicts con claves:
        k, x, f1(x), f2(x), J_f1=df1/dx, J_f2=df2/dx,
        norma_J, condicion_Pareto
    """
    if x_values is None:
        x_values = list(np.linspace(lo, hi, n_points))

    rows = []
    for i, xv in enumerate(x_values):
        xv = float(xv)
        f1v = f1_schaffer(xv)
        f2v = f2_schaffer(xv)
        J   = jacobian_schaffer(xv)

        # La condición de Pareto stationarity (1D):
        # λ·J[0] + (1-λ)·J[1] = 0  ⟹  λ = J[1]/(J[1]-J[0])
        denom = J[1] - J[0]
        if abs(denom) > 1e-14:
            lam = J[1] / denom
            pareto_cond = 0.0 <= lam <= 1.0
            lam_str = f"{lam:.4f}"
        else:
            pareto_cond = False
            lam_str = "∞"

        rows.append({
            "k":               i,
            "x":               round(xv, 6),
            "f1(x)=x²":        round(f1v, 6),
            "f2(x)=(x-2)²":    round(f2v, 6),
            "J[f1]=2x":        round(float(J[0]), 6),
            "J[f2]=2(x-2)":    round(float(J[1]), 6),
            "‖J‖":             round(float(np.linalg.norm(J)), 6),
            "λ_Pareto":        lam_str,
            "¿Pareto-opt?":    "✓ Sí" if pareto_cond else "✗ No",
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  8.  JACOBIANO GENÉRICO (para función arbitraria en ℝ¹)
# ══════════════════════════════════════════════════════════════════════════════

def jacobian_generic(
    func_expr: str,
    var_name: str,
    x0: float = 0.0,
    lo: float = -10.0,
    hi: float = 10.0,
    n_points: int = 15,
) -> List[Dict[str, Any]]:
    """
    Calcula el Jacobiano de F(x) = (f(x), g(x)) donde
        f(x) = func_expr  (función del usuario)
        g(x) = (x - centro)²  (función auxiliar)

    en n_points puntos del rango [lo, hi].

    Devuelve lista de dicts con claves:
        k, x, f(x), g(x), J_f=f'(x), J_g=2(x-c), ‖J‖, λ_Pareto, ¿Pareto-opt?
    """
    if not _SYMPY:
        raise RuntimeError("SymPy es necesario para funciones generales.")

    sym_x  = sp.Symbol(var_name)
    sym_f  = sp.sympify(func_expr)
    sym_df = sp.diff(sym_f, sym_x)
    f_call  = sp.lambdify(sym_x, sym_f,  modules=["numpy"])
    df_call = sp.lambdify(sym_x, sym_df, modules=["numpy"])

    x_c = (lo + hi) / 2.0
    g   = lambda xv: (xv - x_c) ** 2
    dg  = lambda xv: 2.0 * (xv - x_c)

    xs   = np.linspace(lo, hi, n_points)
    rows = []
    for i, xv in enumerate(xs):
        xv = float(xv)
        try:
            fv  = float(f_call(xv))
            gv  = float(g(xv))
            jf  = float(df_call(xv))
            jg  = float(dg(xv))
            J   = np.array([jf, jg])
            nJ  = float(np.linalg.norm(J))
            denom = jg - jf
            if abs(denom) > 1e-14:
                lam = jg / denom
                pareto = 0.0 <= lam <= 1.0
                ls = f"{lam:.4f}"
            else:
                pareto = False
                ls = "∞"
            rows.append({
                "k":          i,
                "x":          round(xv, 6),
                "f(x)":       round(fv, 6),
                "g(x)=(x-c)²": round(gv, 6),
                "J_f=f'(x)":  round(jf, 6),
                "J_g=2(x-c)": round(jg, 6),
                "‖J‖":        round(nJ, 6),
                "λ_Pareto":   ls,
                "¿Pareto-opt?": "✓ Sí" if pareto else "✗ No",
            })
        except Exception as e:
            rows.append({"k": i, "x": round(xv, 6), "error": str(e)})

    return rows
