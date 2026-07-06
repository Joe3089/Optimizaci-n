"""
heuristics.py — Métodos Heurísticos Clásicos
=============================================
Implementa:
  • seccion_aurea      — Búsqueda de mínimo 1D por proporción áurea
  • goldstein_search   — Búsqueda de línea con condiciones de Goldstein
  • newton_raphson_1d  — Newton-Raphson para minimización 1D
  • gradient_conjugate — Gradiente Conjugado (Fletcher-Reeves) ND

Todos retornan una lista de dicts (historial de iteraciones)
compatible con la tabla e interfaz de la aplicación.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Callable, List, Dict, Any

try:
    import sympy as sp
    _SYMPY = True
except ImportError:
    _SYMPY = False

# ─── utilidades ──────────────────────────────────────────────────────────────
def _approx_grad(f: Callable, x: float, eps: float = 1e-6) -> float:
    return (f(x + eps) - f(x - eps)) / (2 * eps)

def _approx_grad_nd(f: Callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps; xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g

def _make_callable_sympy(func_str: str, var_str: str):
    """Crea callable numpy desde string de función y variables."""
    if not _SYMPY:
        raise RuntimeError("SymPy no disponible.")
    syms = sp.symbols(var_str)
    if not isinstance(syms, (list, tuple)):
        syms = (syms,)
    f_sym = sp.sympify(func_str)
    modules = [{"DiracDelta": lambda *a: 0.0,
                "Heaviside": lambda x, *a: np.heaviside(x, 0.5)}, "numpy"]
    return sp.lambdify(list(syms), f_sym, modules=modules), list(syms)


# ══════════════════════════════════════════════════════════════════════════════
#  1. SECCIÓN ÁUREA
# ══════════════════════════════════════════════════════════════════════════════
def seccion_aurea(func: Callable, a: float, b: float,
                  tol: float = 1e-6, max_iter: int = 200) -> List[Dict[str, Any]]:
    """
    Búsqueda del mínimo de f en [a,b] por la proporción áurea (Golden Section).

    τ = (√5 − 1) / 2 ≈ 0.6180

    En cada iteración se evalúan dos puntos internos c y d, se descarta
    el subintervalo que no contiene el mínimo y se reutiliza una evaluación.
    Convergencia lineal garantizada con factor τ por iteración.
    """
    TAU = (math.sqrt(5) - 1) / 2    # 0.61803…
    a, b = float(a), float(b)
    history: List[Dict[str, Any]] = []

    c = b - TAU * (b - a)
    d = a + TAU * (b - a)
    fc, fd = func(c), func(d)

    for k in range(1, max_iter + 1):
        L = b - a
        history.append({
            "iter"    : k,
            "a"       : round(a, 8),
            "b"       : round(b, 8),
            "c"       : round(c, 8),
            "d"       : round(d, 8),
            "f(c)"    : round(fc, 8),
            "f(d)"    : round(fd, 8),
            "L"       : round(L, 8),
            "x_opt"   : round((a + b) / 2, 8),
        })

        if L < tol:
            break

        if fc < fd:
            b  = d
            d  = c;   fd = fc
            c  = b - TAU * (b - a)
            fc = func(c)
        else:
            a  = c
            c  = d;   fc = fd
            d  = a + TAU * (b - a)
            fd = func(d)

    # Marcar óptimo en última fila
    x_opt = (a + b) / 2
    history[-1]["x_opt"] = round(x_opt, 8)
    history[-1]["f_opt"] = round(func(x_opt), 8)
    return history


# ══════════════════════════════════════════════════════════════════════════════
#  2. CONDICIONES DE GOLDSTEIN
# ══════════════════════════════════════════════════════════════════════════════
def goldstein_search(func_str: str, var_str: str, x0: list,
                     mu: float = 0.2, alpha0: float = 1.0,
                     rho: float = 0.5, max_iter: int = 50) -> List[Dict[str, Any]]:
    """
    Búsqueda de línea con condiciones de Goldstein (Newton + descenso de gradiente).

    Las dos condiciones de Goldstein son:
      1. Sufficient decrease (Armijo):  f(x+αd) ≤ f(x) + μ·α·∇f·d
      2. Cota inferior:                 f(x+αd) ≥ f(x) + (1-μ)·α·∇f·d

    Con μ ∈ (0, 0.5). Esto evita pasos demasiado grandes (cond.1)
    y también pasos demasiado pequeños (cond.2).
    """
    if not _SYMPY:
        raise RuntimeError("SymPy no disponible.")

    syms = sp.symbols(var_str)
    if not isinstance(syms, (list, tuple)):
        syms = (syms,)
    n = len(syms)
    f_sym   = sp.sympify(func_str)
    grad_sym = [f_sym.diff(s) for s in syms]
    modules = [{"DiracDelta": lambda *a: 0.0,
                "Heaviside":  lambda x, *a: np.heaviside(x, 0.5)}, "numpy"]
    f_num    = sp.lambdify(list(syms), f_sym,    modules=modules)
    grad_num = sp.lambdify(list(syms), grad_sym, modules=modules)

    x  = np.array(x0, dtype=float)
    history: List[Dict[str, Any]] = []

    for k in range(max_iter):
        vals = tuple(x.tolist()) if n > 1 else (float(x[0]),)
        f0   = float(f_num(*vals))
        g0   = np.array(grad_num(*vals), dtype=float).reshape(-1)
        gn   = float(np.linalg.norm(g0))

        history.append({
            "k"        : k,
            "x_k_str"  : "[" + " ".join(f"{v:.5g}" for v in x) + "]",
            "f_k"      : round(f0, 8),
            "grad_norm": round(gn, 8),
            "alpha_k"  : 0.0,
            "cond1_ok" : False,
            "cond2_ok" : False,
        })

        if gn < 1e-7:
            history[-1]["alpha_k"] = 0.0
            break

        d = -g0   # dirección de descenso
        dg = float(np.dot(g0, d))   # siempre negativo

        alpha = float(alpha0)
        cond1 = cond2 = False
        for _ in range(60):
            xn  = x + alpha * d
            vn  = tuple(xn.tolist()) if n > 1 else (float(xn[0]),)
            fn  = float(f_num(*vn))
            rhs1 = f0 + mu * alpha * dg
            rhs2 = f0 + (1 - mu) * alpha * dg
            cond1 = fn <= rhs1
            cond2 = fn >= rhs2
            if cond1 and cond2:
                break
            alpha *= rho

        history[-1]["alpha_k"]  = round(alpha, 8)
        history[-1]["cond1_ok"] = bool(cond1)
        history[-1]["cond2_ok"] = bool(cond2)
        x = x + alpha * d

    return history


# ══════════════════════════════════════════════════════════════════════════════
#  3. NEWTON-RAPHSON (minimización)
# ══════════════════════════════════════════════════════════════════════════════
def newton_raphson(func_str: str, var_str: str, x0: list,
                   tol: float = 1e-6, max_iter: int = 50) -> List[Dict[str, Any]]:
    """
    Método de Newton-Raphson para minimización:
        x_{k+1} = x_k − [H(x_k)]⁻¹ · ∇f(x_k)

    Usa diferenciación simbólica (SymPy) para gradiente y Hessiano exactos.
    Convergencia cuadrática cerca del mínimo.
    """
    if not _SYMPY:
        raise RuntimeError("SymPy no disponible.")

    syms = sp.symbols(var_str)
    if not isinstance(syms, (list, tuple)):
        syms = (syms,)
    n = len(syms)
    f_sym    = sp.sympify(func_str)
    grad_sym = [f_sym.diff(s) for s in syms]
    hess_sym = sp.hessian(f_sym, list(syms))
    modules  = [{"DiracDelta": lambda *a: 0.0,
                 "Heaviside":  lambda x, *a: np.heaviside(x, 0.5)}, "numpy"]
    f_num    = sp.lambdify(list(syms), f_sym,    modules=modules)
    grad_num = sp.lambdify(list(syms), grad_sym, modules=modules)
    hess_num = sp.lambdify(list(syms), hess_sym, modules=modules)

    x = np.array(x0, dtype=float)
    history: List[Dict[str, Any]] = []

    for k in range(max_iter):
        vals = tuple(x.tolist()) if n > 1 else (float(x[0]),)
        fk   = float(f_num(*vals))
        gk   = np.array(grad_num(*vals), dtype=float).reshape(-1)
        Hk   = np.array(hess_num(*vals), dtype=float).reshape(n, n)
        gn   = float(np.linalg.norm(gk))

        history.append({
            "k"           : k,
            "x_k_str"     : "[" + " ".join(f"{v:.6g}" for v in x) + "]",
            "f_k"         : round(fk, 8),
            "grad_norm"   : round(gn, 8),
            "alpha_k"     : 1.0,
            "hess_det"    : round(float(np.linalg.det(Hk)), 6),
        })

        if gn < tol:
            break

        # Regularizar Hessiano si es casi singular
        reg = 1e-10
        for _ in range(8):
            try:
                d = np.linalg.solve(Hk + reg * np.eye(n), -gk)
                break
            except np.linalg.LinAlgError:
                reg *= 10
        else:
            d = -gk  # fallback: descenso de gradiente

        # Backtracking simple para estabilidad
        alpha = 1.0
        for _ in range(30):
            xn = x + alpha * d
            vn = tuple(xn.tolist()) if n > 1 else (float(xn[0]),)
            try:
                fn = float(f_num(*vn))
                if fn < fk:
                    break
            except Exception:
                pass
            alpha *= 0.5

        history[-1]["alpha_k"] = round(alpha, 8)
        x = x + alpha * d

    return history


# ══════════════════════════════════════════════════════════════════════════════
#  4. GRADIENTE CONJUGADO (Fletcher-Reeves)
# ══════════════════════════════════════════════════════════════════════════════
def gradient_conjugate(func_str: str, var_str: str, x0: list,
                       tol: float = 1e-6, max_iter: int = 100) -> List[Dict[str, Any]]:
    """
    Método de Gradiente Conjugado (Fletcher-Reeves) para minimización ND.

    Actualización del parámetro β (Fletcher-Reeves):
        β_k = ||∇f(x_{k+1})||² / ||∇f(x_k)||²

    Dirección conjugada:
        d_{k+1} = −∇f(x_{k+1}) + β_k · d_k

    Convergencia superlineal en funciones cuadráticas, buena en no lineales.
    """
    if not _SYMPY:
        raise RuntimeError("SymPy no disponible.")

    syms = sp.symbols(var_str)
    if not isinstance(syms, (list, tuple)):
        syms = (syms,)
    n = len(syms)
    f_sym    = sp.sympify(func_str)
    grad_sym = [f_sym.diff(s) for s in syms]
    modules  = [{"DiracDelta": lambda *a: 0.0,
                 "Heaviside":  lambda x, *a: np.heaviside(x, 0.5)}, "numpy"]
    f_num    = sp.lambdify(list(syms), f_sym,    modules=modules)
    grad_num = sp.lambdify(list(syms), grad_sym, modules=modules)

    def fval(xv):
        v = tuple(xv.tolist()) if n > 1 else (float(xv[0]),)
        return float(f_num(*v))

    def gval(xv):
        v = tuple(xv.tolist()) if n > 1 else (float(xv[0]),)
        return np.array(grad_num(*v), dtype=float).reshape(-1)

    x  = np.array(x0, dtype=float)
    g  = gval(x)
    d  = -g.copy()
    history: List[Dict[str, Any]] = []

    for k in range(max_iter):
        fk = fval(x)
        gn = float(np.linalg.norm(g))

        history.append({
            "k"        : k,
            "x_k_str"  : "[" + " ".join(f"{v:.6g}" for v in x) + "]",
            "f_k"      : round(fk, 8),
            "grad_norm": round(gn, 8),
            "beta_k"   : 0.0,
            "alpha_k"  : 0.0,
        })

        if gn < tol:
            break

        # Búsqueda de línea (Armijo backtracking)
        alpha = 1.0
        dg    = float(np.dot(g, d))
        for _ in range(50):
            xn = x + alpha * d
            if fval(xn) <= fk + 1e-4 * alpha * dg:
                break
            alpha *= 0.5
        else:
            alpha = 1e-4

        history[-1]["alpha_k"] = round(alpha, 8)
        x  = x + alpha * d
        g_new = gval(x)
        gn_new = float(np.linalg.norm(g_new))

        # β Fletcher-Reeves
        beta = (gn_new ** 2) / (gn ** 2) if gn > 1e-15 else 0.0
        history[-1]["beta_k"] = round(beta, 6)

        # Reiniciar dirección cada n pasos para estabilidad
        d = -g_new + beta * d if (k + 1) % max(n, 5) != 0 else -g_new
        g = g_new

    return history
