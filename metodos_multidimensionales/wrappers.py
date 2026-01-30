# -*- coding: utf-8 -*-
"""
metodos_multidimensional/wrappers.py  (SIN dependencias externas)

Objetivo:
- NO tocar los archivos originales "wrappers (1).py", "line_search (1).py", etc.
- NO requerir pip (sin scipy/sympy).
- Proveer implementaciones numéricas (gradiente/Hessiana por diferencias finitas)
  para que estos métodos funcionen desde tu interfaz:

    - penalty_method_newton
    - barrier_method_newton
    - penalty_method_bfgs
    - penalty_method_nelder

Salida:
    (x_opt (list[float]), f_opt (float), log_data (list[dict]))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple
import numpy as np

_SAFE = {
    "np": np,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": abs, "pi": np.pi, "e": np.e,
}

def _parse_vars(var_str: str) -> List[str]:
    s = (var_str or "").strip()
    if not s:
        raise ValueError("Variables vacías. Ej: x1 x2")
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    if not parts:
        raise ValueError("Variables inválidas. Ej: x1 x2")
    return parts

def _compile_expr(expr: str, vars_: List[str]) -> Callable[[np.ndarray], float]:
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Expresión vacía.")
    code = compile(expr, "<expr>", "eval")

    def f(x: np.ndarray) -> float:
        env = {"__builtins__": {}}
        local = dict(_SAFE)
        for i, name in enumerate(vars_):
            local[name] = float(x[i])
        return float(eval(code, env, local))

    _ = f(np.zeros(len(vars_), dtype=float))
    return f

def _grad(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = x.size
    g = np.zeros(n, dtype=float)
    fx = f(x)
    for i in range(n):
        x2 = x.copy()
        x2[i] += eps
        g[i] = (f(x2) - fx) / eps
    return g

def _hess(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    n = x.size
    H = np.zeros((n, n), dtype=float)
    g0 = _grad(f, x, eps=eps/10)
    for i in range(n):
        x2 = x.copy()
        x2[i] += eps
        gi = _grad(f, x2, eps=eps/10)
        H[:, i] = (gi - g0) / eps
    return 0.5 * (H + H.T)

def _armijo_backtracking(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    p: np.ndarray,
    g: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_ls: int = 30,
) -> float:
    fx = f(x)
    alpha = alpha0
    for _ in range(max_ls):
        xn = x + alpha * p
        if f(xn) <= fx + c1 * alpha * float(g @ p):
            return alpha
        alpha *= rho
    return alpha

@dataclass
class _LogRow:
    iter: int
    x: list
    f: float
    paso: float
    x_candidate: list
    f_candidate: float
    improved: bool

def _coerce_log(log: List[_LogRow]) -> List[Dict]:
    return [r.__dict__ for r in log]

def penalty_method_newton(
    func_str: str,
    constraint_str: str,
    var_str: str,
    x0: List[float],
    mu0: float = 10.0,
    mu_mult: float = 10.0,
    outer_iters: int = 6,
    inner_iters: int = 50,
    tol: float = 1e-6,
) -> Tuple[List[float], float, List[Dict]]:
    vars_ = _parse_vars(var_str)
    f0 = _compile_expr(func_str, vars_)
    g0 = _compile_expr(constraint_str, vars_)

    def phi(x: np.ndarray, mu: float) -> float:
        gx = g0(x)
        pen = max(0.0, gx) ** 2
        return f0(x) + mu * pen

    x = np.array(x0, dtype=float)
    mu = mu0
    log: List[_LogRow] = []
    it = 0

    for _outer in range(outer_iters):
        def fmu(z): return phi(z, mu)

        for _inner in range(inner_iters):
            it += 1
            fx = fmu(x)
            g = _grad(fmu, x)
            if np.linalg.norm(g) < tol:
                break
            H = _hess(fmu, x) + 1e-6 * np.eye(len(vars_))
            try:
                p = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                p = -g
            alpha = _armijo_backtracking(fmu, x, p, g)
            xc = x + alpha * p
            fc = fmu(xc)
            log.append(_LogRow(it, x.tolist(), fx, float(alpha), xc.tolist(), fc, bool(fc < fx)))
            x = xc
        mu *= mu_mult

    return x.tolist(), float(f0(x)), _coerce_log(log)

def barrier_method_newton(
    func_str: str,
    constraint_str: str,
    var_str: str,
    x0: List[float],
    t0: float = 1.0,
    t_mult: float = 10.0,
    outer_iters: int = 6,
    inner_iters: int = 50,
    tol: float = 1e-6,
) -> Tuple[List[float], float, List[Dict]]:
    vars_ = _parse_vars(var_str)
    f0 = _compile_expr(func_str, vars_)
    g0 = _compile_expr(constraint_str, vars_)

    x = np.array(x0, dtype=float)
    if g0(x) >= 0:
        raise ValueError("Para barrera necesitas x0 factible: g(x0) < 0")

    def phi(x: np.ndarray, t: float) -> float:
        gx = g0(x)
        return t * f0(x) - np.log(-gx)

    t = t0
    log: List[_LogRow] = []
    it = 0

    for _outer in range(outer_iters):
        def ft(z): return phi(z, t)

        for _inner in range(inner_iters):
            it += 1
            fx = ft(x)
            g = _grad(ft, x)
            if np.linalg.norm(g) < tol:
                break
            H = _hess(ft, x) + 1e-6 * np.eye(len(vars_))
            try:
                p = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                p = -g

            alpha = 1.0
            for _ in range(30):
                xc = x + alpha * p
                if g0(xc) < 0:
                    break
                alpha *= 0.5

            alpha = _armijo_backtracking(ft, x, p, g, alpha0=alpha)
            xc = x + alpha * p
            fc = ft(xc)
            log.append(_LogRow(it, x.tolist(), fx, float(alpha), xc.tolist(), fc, bool(fc < fx)))
            x = xc

        t *= t_mult

    return x.tolist(), float(f0(x)), _coerce_log(log)

def penalty_method_bfgs(
    func_str: str,
    constraint_str: str,
    var_str: str,
    x0: List[float],
    mu: float = 10.0,
    iters: int = 200,
    tol: float = 1e-6,
) -> Tuple[List[float], float, List[Dict]]:
    vars_ = _parse_vars(var_str)
    f0 = _compile_expr(func_str, vars_)
    g0 = _compile_expr(constraint_str, vars_)

    def fmu(x: np.ndarray) -> float:
        gx = g0(x)
        return f0(x) + mu * max(0.0, gx) ** 2

    x = np.array(x0, dtype=float)
    n = x.size
    H = np.eye(n, dtype=float)
    log: List[_LogRow] = []

    for k in range(1, iters + 1):
        fx = fmu(x)
        g = _grad(fmu, x)
        if np.linalg.norm(g) < tol:
            break
        p = -H @ g
        alpha = _armijo_backtracking(fmu, x, p, g)
        xc = x + alpha * p
        gc = _grad(fmu, xc)
        s = (xc - x)
        y = (gc - g)
        ys = float(y @ s)
        if ys > 1e-10:
            rho = 1.0 / ys
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        log.append(_LogRow(k, x.tolist(), fx, float(alpha), xc.tolist(), float(fmu(xc)), bool(fmu(xc) < fx)))
        x = xc

    return x.tolist(), float(f0(x)), _coerce_log(log)

def penalty_method_nelder(
    func_str: str,
    constraint_str: str,
    var_str: str,
    x0: List[float],
    mu: float = 10.0,
    iters: int = 300,
    step: float = 0.1,
) -> Tuple[List[float], float, List[Dict]]:
    vars_ = _parse_vars(var_str)
    f0 = _compile_expr(func_str, vars_)
    g0 = _compile_expr(constraint_str, vars_)

    def fmu(x: np.ndarray) -> float:
        gx = g0(x)
        return f0(x) + mu * max(0.0, gx) ** 2

    x0v = np.array(x0, dtype=float)
    n = x0v.size

    simplex = [x0v]
    for i in range(n):
        v = x0v.copy()
        v[i] += step
        simplex.append(v)
    simplex = np.array(simplex, dtype=float)

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    log: List[_LogRow] = []

    for k in range(1, iters + 1):
        vals = np.array([fmu(s) for s in simplex])
        idx = np.argsort(vals)
        simplex = simplex[idx]
        vals = vals[idx]

        best = simplex[0]
        worst = simplex[-1]
        centroid = np.mean(simplex[:-1], axis=0)

        xr = centroid + alpha * (centroid - worst)
        fr = fmu(xr)

        if fr < vals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = fmu(xe)
            new = xe if fe < fr else xr
            fnew = min(fe, fr)
        elif fr < vals[-2]:
            new = xr
            fnew = fr
        else:
            if fr < vals[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid + rho * (worst - centroid)
            fc = fmu(xc)
            if fc < vals[-1]:
                new = xc
                fnew = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = best + sigma * (simplex[i] - best)
                continue

        log.append(_LogRow(k, best.tolist(), float(vals[0]), 0.0, new.tolist(), float(fnew), bool(fnew < vals[-1])))
        simplex[-1] = new

        if np.std(vals) < 1e-8:
            break

    vals = np.array([fmu(s) for s in simplex])
    best = simplex[np.argmin(vals)]
    return best.tolist(), float(f0(best)), _coerce_log(log)
