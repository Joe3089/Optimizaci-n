# sympy_api.py (FIX)
from __future__ import annotations
from typing import Callable, List, Sequence, Tuple
import numpy as np
import sympy as sp
import re

def _sanitize_expr(expr: str) -> str:
    if expr is None:
        return ""
    expr = expr.strip()
    expr = expr.replace("^", "**")
    expr = re.sub(r"\bsen\b", "sin", expr)
    return expr

def parse_variables(var_str: str) -> List[str]:
    if not var_str:
        return ["x"]
    raw = var_str.replace(",", " ").split()
    return [t.strip() for t in raw if t.strip()]

def parse_x0(x0_str: str, dim: int) -> np.ndarray:
    if x0_str is None:
        raise ValueError("x0 vacío")
    parts = [p.strip() for p in x0_str.replace(";", ",").split(",") if p.strip()]
    if dim == 1 and len(parts) == 1:
        return np.array([float(parts[0])], dtype=float)
    if len(parts) != dim:
        raise ValueError(f"x0 debe tener {dim} valores separados por coma.")
    return np.array([float(p) for p in parts], dtype=float)

def parse_expression(expr_str: str, var_names: Sequence[str]) -> sp.Expr:
    expr_str = _sanitize_expr(expr_str)
    if not expr_str:
        raise ValueError("Expresión vacía.")
    syms = sp.symbols(" ".join(var_names))
    locals_ = {name: sym for name, sym in zip(var_names, syms)}
    locals_.update({"pi": sp.pi, "e": sp.E})
    try:
        return sp.sympify(expr_str, locals=locals_)
    except Exception as e:
        raise ValueError(f"No se pudo interpretar la expresión: {e}") from e

def build_lambda(expr: sp.Expr, var_names: Sequence[str]) -> Callable:
    syms = sp.symbols(" ".join(var_names))
    return sp.lambdify(syms, expr, modules=["numpy"])

def make_mesh_1d(xmin: float, xmax: float, n: int = 400) -> np.ndarray:
    if xmin == xmax:
        xmax = xmin + 1.0
    return np.linspace(float(xmin), float(xmax), int(n))

def make_mesh_2d(xmin: float, xmax: float, n: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(float(xmin), float(xmax), int(n))
    X, Y = np.meshgrid(xs, xs)
    return X, Y

def eval_f_1d(expr: sp.Expr, xgrid: np.ndarray, var_name: str = "x") -> np.ndarray:
    f = build_lambda(expr, [var_name])
    y = f(xgrid)
    return np.asarray(y, dtype=float)

def eval_f_2d(expr: sp.Expr, X: np.ndarray, Y: np.ndarray, var_names: Sequence[str]) -> np.ndarray:
    f = build_lambda(expr, var_names)
    Z = f(X, Y)
    return np.asarray(Z, dtype=float)
