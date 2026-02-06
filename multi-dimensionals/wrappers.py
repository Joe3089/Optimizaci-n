# wrappers.py
# Métodos wrappers para tu interfaz (Penalización y Barrera con Newton)
# Firma esperada por interfaz_qt.py:
#   penalty_method_newton(func_str, constr_str, var_str, x0) -> (x_opt, f_opt, history)
#   barrier_method_newton(func_str, constr_str, var_str, x0) -> (x_opt, f_opt, history)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any

import numpy as np
import sympy as sp


def _parse_symbols(var_str: str):
    names = [v.strip() for v in var_str.split() if v.strip()]
    if not names:
        raise ValueError("Variables vacías. Ej: 'x1 x2'")
    syms = sp.symbols(names)
    # sympy returns single Symbol if len==1
    if isinstance(syms, sp.Symbol):
        syms = (syms,)
    return names, syms


def _as_np(x) -> np.ndarray:
    arr = np.array(x, dtype=float).reshape(-1)
    return arr


def _safe_eval_lambdify(expr, syms):
    # use numpy backend
    return sp.lambdify(syms, expr, "numpy")


def _format_x(x: np.ndarray) -> str:
    return "[" + " ".join(f"{v:.6g}" for v in x.tolist()) + "]"


def _armijo_backtracking(phi: Callable[[np.ndarray], float],
                         grad: Callable[[np.ndarray], np.ndarray],
                         x: np.ndarray,
                         p: np.ndarray,
                         alpha0: float = 1.0,
                         c1: float = 1e-4,
                         tau: float = 0.5,
                         max_ls: int = 30) -> float:
    f0 = float(phi(x))
    g0 = grad(x)
    dg = float(np.dot(g0, p))
    if dg >= 0:
        # not a descent direction -> fall back to steepest descent
        p = -g0
        dg = float(np.dot(g0, p))
        if dg >= 0:
            return 0.0

    alpha = alpha0
    for _ in range(max_ls):
        xn = x + alpha * p
        fn = float(phi(xn))
        if fn <= f0 + c1 * alpha * dg:
            return alpha
        alpha *= tau
    return alpha


def _newton_unconstrained(phi: Callable[[np.ndarray], float],
                          grad: Callable[[np.ndarray], np.ndarray],
                          hess: Callable[[np.ndarray], np.ndarray],
                          x0: np.ndarray,
                          tol: float = 1e-6,
                          max_iter: int = 60) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    x = _as_np(x0)
    history: List[Dict[str, Any]] = []

    for k in range(max_iter):
        g = grad(x)
        gn = float(np.linalg.norm(g))
        fx = float(phi(x))

        # record (alpha filled later)
        history.append({
            "k": k,
            "x_k": x.copy(),
            "x_k_str": _format_x(x),
            "f_k": fx,
            "grad_norm": gn,
            "alpha_k": 0.0,
        })

        if gn <= tol:
            break

        H = hess(x)
        # Regularización leve si H es singular o indefinida
        reg = 1e-8
        for _ in range(6):
            try:
                # Solve H p = -g
                p = np.linalg.solve(H + reg * np.eye(len(x)), -g)
                break
            except np.linalg.LinAlgError:
                reg *= 10.0
        else:
            p = -g  # fallback

        alpha = _armijo_backtracking(phi, grad, x, p, alpha0=1.0)
        x = x + alpha * p
        history[-1]["alpha_k"] = float(alpha)

    return x, history


def penalty_method_newton(func_str: str,
                          constr_str: str,
                          var_str: str,
                          x0,
                          mu0: float = 1.0,
                          mu_factor: float = 10.0,
                          outer_iters: int = 3) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
    """Penalización cuadrática para una restricción g(x) <= 0.
    phi(x) = f(x) + mu * max(0, g(x))^2
    """
    names, syms = _parse_symbols(var_str)
    x = _as_np(x0)
    n = len(syms)

    f_sym = sp.sympify(func_str)
    f_num = _safe_eval_lambdify(f_sym, syms)

    if constr_str and constr_str.strip() != "0":
        g_sym = sp.sympify(constr_str)
        g_num = _safe_eval_lambdify(g_sym, syms)
    else:
        g_sym = sp.Integer(0)
        g_num = lambda *args: 0.0

    # Build penalty expression with sympy: max(0,g)^2 via Piecewise
    g_pos = sp.Piecewise((g_sym, g_sym > 0), (0, True))
    phi_sym_mu = sp.Symbol("mu")  # parameter
    phi_sym = f_sym + phi_sym_mu * (g_pos ** 2)

    grad_sym = [sp.diff(phi_sym, s) for s in syms]
    hess_sym = sp.hessian(phi_sym, syms)

    history_all: List[Dict[str, Any]] = []
    mu = float(mu0)

    for outer in range(outer_iters):
        # lambdify with mu substituted
        phi_mu = sp.lambdify(syms, sp.simplify(phi_sym.subs({phi_sym_mu: mu})), "numpy")
        grad_mu = sp.lambdify(syms, [gi.subs({phi_sym_mu: mu}) for gi in grad_sym], "numpy")
        hess_mu = sp.lambdify(syms, hess_sym.subs({phi_sym_mu: mu}), "numpy")

        def phi(xv: np.ndarray) -> float:
            vals = tuple(xv.tolist()) if n > 1 else (float(xv[0]),)
            return float(phi_mu(*vals))

        def grad(xv: np.ndarray) -> np.ndarray:
            vals = tuple(xv.tolist()) if n > 1 else (float(xv[0]),)
            g = np.array(grad_mu(*vals), dtype=float).reshape(-1)
            return g

        def hess(xv: np.ndarray) -> np.ndarray:
            vals = tuple(xv.tolist()) if n > 1 else (float(xv[0]),)
            H = np.array(hess_mu(*vals), dtype=float)
            return H

        x, hist_inner = _newton_unconstrained(phi, grad, hess, x, tol=1e-6, max_iter=60)
        # merge history, keep k global
        k0 = len(history_all)
        for rec in hist_inner:
            rec["k"] = k0 + rec["k"]
            rec["mu"] = mu
            history_all.append(rec)

        # increase mu
        mu *= mu_factor

    # Report objective value (original f) at final x
    vals = tuple(x.tolist()) if n > 1 else (float(x[0]),)
    f_opt = float(f_num(*vals))
    return x, f_opt, history_all


def barrier_method_newton(func_str: str,
                          constr_str: str,
                          var_str: str,
                          x0,
                          t0: float = 1.0,
                          t_factor: float = 10.0,
                          outer_iters: int = 3) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
    """Barrera log para g(x) <= 0:
    phi(x) = f(x) - (1/t) * log(-g(x))
    Requiere punto factible (g(x) < 0). Si no lo es, empuja x0 ligeramente.
    """
    names, syms = _parse_symbols(var_str)
    x = _as_np(x0)
    n = len(syms)

    f_sym = sp.sympify(func_str)
    f_num = _safe_eval_lambdify(f_sym, syms)

    if not constr_str or constr_str.strip() == "0":
        # sin restricción: simplemente Newton en f
        grad_sym = [sp.diff(f_sym, s) for s in syms]
        hess_sym = sp.hessian(f_sym, syms)
        grad_f = sp.lambdify(syms, grad_sym, "numpy")
        hess_f = sp.lambdify(syms, hess_sym, "numpy")

        def phi(xv):
            vals = tuple(xv.tolist()) if n > 1 else (float(xv[0]),)
            return float(f_num(*vals))

        def grad(xv):
            vals = tuple(xv.tolist()) if n > 1 else (float(xv[0]),)
            return np.array(grad_f(*vals), dtype=float).reshape(-1)

        def hess(xv):
            vals = tuple(xv.tolist()) if n > 1 else (float(xv[0]),)
            return np.array(hess_f(*vals), dtype=float)

        x, hist = _newton_unconstrained(phi, grad, hess, x, tol=1e-6, max_iter=80)
        f_opt = float(phi(x))
        return x, f_opt, hist

    g_sym = sp.sympify(constr_str)
    g_num = _safe_eval_lambdify(g_sym, syms)

    # Try to make x feasible if needed
    def g_val(xv):
        vals = tuple(xv.tolist()) if n > 1 else (float(xv[0]),)
        return float(g_num(*vals))

    if g_val(x) >= -1e-8:
        # push along negative gradient of g
        grad_g_sym = [sp.diff(g_sym, s) for s in syms]
        grad_g = sp.lambdify(syms, grad_g_sym, "numpy")
        for _ in range(50):
            vals = tuple(x.tolist()) if n > 1 else (float(x[0]),)
            gg = float(g_num(*vals))
            if gg < -1e-6:
                break
            dg = np.array(grad_g(*vals), dtype=float).reshape(-1)
            step = 0.05 / (np.linalg.norm(dg) + 1e-9)
            x = x - step * dg
        # if still not feasible, small random jitter
        if g_val(x) >= -1e-8:
            x = x - 0.1 * np.ones_like(x)

    t = float(t0)

    # Barrier expression
    t_sym = sp.Symbol("t")
    phi_sym = f_sym - (1 / t_sym) * sp.log(-g_sym)

    grad_sym = [sp.diff(phi_sym, s) for s in syms]
    hess_sym = sp.hessian(phi_sym, syms)

    history_all: List[Dict[str, Any]] = []

    for outer in range(outer_iters):
        phi_t = sp.lambdify(syms, sp.simplify(phi_sym.subs({t_sym: t})), "numpy")
        grad_t = sp.lambdify(syms, [gi.subs({t_sym: t}) for gi in grad_sym], "numpy")
        hess_t = sp.lambdify(syms, hess_sym.subs({t_sym: t}), "numpy")

        def phi(xv: np.ndarray) -> float:
            # enforce feasibility for evaluation (avoid log domain)
            gv = g_val(xv)
            if gv >= 0:
                return float("inf")
            vals = tuple(xv.tolist()) if n > 1 else (float(xv[0]),)
            return float(phi_t(*vals))

        def grad(xv: np.ndarray) -> np.ndarray:
            vals = tuple(xv.tolist()) if n > 1 else (float(xv[0]),)
            return np.array(grad_t(*vals), dtype=float).reshape(-1)

        def hess(xv: np.ndarray) -> np.ndarray:
            vals = tuple(xv.tolist()) if n > 1 else (float(xv[0]),)
            return np.array(hess_t(*vals), dtype=float)

        # Newton with feasibility-preserving line search (shrink alpha if infeasible)
        xk = x.copy()
        inner_hist: List[Dict[str, Any]] = []
        for k in range(60):
            gk = grad(xk)
            gn = float(np.linalg.norm(gk))
            fk = float(phi(xk))
            inner_hist.append({
                "k": k,
                "x_k": xk.copy(),
                "x_k_str": _format_x(xk),
                "f_k": fk,
                "grad_norm": gn,
                "alpha_k": 0.0,
                "t": t,
            })
            if gn <= 1e-6:
                break
            Hk = hess(xk)
            reg = 1e-8
            for _ in range(6):
                try:
                    pk = np.linalg.solve(Hk + reg * np.eye(len(xk)), -gk)
                    break
                except np.linalg.LinAlgError:
                    reg *= 10.0
            else:
                pk = -gk

            alpha = 1.0
            # shrink until feasible and Armijo
            for _ in range(40):
                xn = xk + alpha * pk
                if g_val(xn) < 0 and float(phi(xn)) < float("inf"):
                    # Armijo on barrier phi
                    f0 = float(phi(xk))
                    dg = float(np.dot(gk, pk))
                    if float(phi(xn)) <= f0 + 1e-4 * alpha * dg:
                        break
                alpha *= 0.5
            inner_hist[-1]["alpha_k"] = float(alpha)
            xk = xk + alpha * pk

        x = xk
        k0 = len(history_all)
        for rec in inner_hist:
            rec["k"] = k0 + rec["k"]
            history_all.append(rec)

        t *= t_factor

    vals = tuple(x.tolist()) if n > 1 else (float(x[0]),)
    f_opt = float(f_num(*vals))
    return x, f_opt, history_all
