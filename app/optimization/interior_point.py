# app/optimization/interior_point.py
# Método de punto interior primal-dual (predictor-corrector de Mehrotra) para
# PL, y el problema de Klee-Minty que contrasta Simplex contra punto interior.
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from app.operations_research.linear_programming import OPTIMAL, solve_lp


def _standard(c, A, signs, b, sense):
    c = np.asarray(c, float)
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    if A.ndim != 2 or A.shape[1] != len(c):
        raise ValueError(f"Cada restricción debe tener {len(c)} coeficientes.")
    cost = -c if sense == "max" else c
    slack = []
    for i, s in enumerate(signs):
        if s != "=":
            col = np.zeros(len(b))
            col[i] = 1.0 if s == "<=" else -1.0
            slack.append(col)
    As = np.column_stack([A] + slack) if slack else A
    return np.concatenate([cost, np.zeros(len(slack))]), As, b


def _solve_normal(M, r):
    try:
        return np.linalg.solve(M, r)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, r, rcond=None)[0]


def _max_step(v, dv):
    neg = dv < 0
    return min(1.0, float(np.min(-v[neg] / dv[neg]))) if neg.any() else 1.0


def interior_point(c, A, signs, b, sense="max", tol=1e-8, max_iter=200) -> Dict[str, Any]:
    n_orig = len(c)
    cf, As, bf = _standard(c, A, signs, b, sense)
    # Escalado global para que Klee-Minty (lados derechos enormes) no degrade μ.
    beta = max(1.0, float(np.abs(bf).max()))
    gamma = max(1.0, float(np.abs(cf).max()))
    cs, bs = cf / gamma, bf / beta
    m, n = As.shape

    AAT = As @ As.T
    x = As.T @ _solve_normal(AAT, bs)
    y = _solve_normal(AAT, As @ cs)
    s = cs - As.T @ y
    x += max(-1.5 * x.min(), 0.0)
    s += max(-1.5 * s.min(), 0.0)
    if x.sum() <= 0 or s.sum() <= 0:
        x, s = np.ones(n), np.ones(n)
    xs = x @ s
    x += 0.5 * xs / s.sum()
    s += 0.5 * xs / x.sum()

    hist: List[Dict[str, Any]] = []
    path = []
    status = "No convergió"

    for k in range(max_iter + 1):
        rp = bs - As @ x
        rd = cs - As.T @ y - s
        mu = float(x @ s / n)
        z_user = float((-1 if sense == "max" else 1) * gamma * beta * (cs @ x))
        pres = float(np.linalg.norm(rp) / (1 + np.linalg.norm(bs)))
        dres = float(np.linalg.norm(rd) / (1 + np.linalg.norm(cs)))
        gap = mu * n / (1 + abs(float(cs @ x)))
        path.append((beta * x[:n_orig]).copy())
        hist.append({"Iter": k, "μ": mu, "Residuo primal": pres, "Residuo dual": dres,
                     "Brecha relativa": gap, "Z": z_user,
                     "Punto": "[" + ", ".join(f"{v:.4g}" for v in beta * x[:n_orig]) + "]"})
        if pres < tol and dres < tol and gap < tol:
            status = OPTIMAL
            break
        if not np.isfinite(mu) or mu > 1e25 or np.linalg.norm(x) > 1e25:
            break
        if k == max_iter:
            break

        D = x / s
        M = (As * D[:, None].T) @ As.T

        def direction(rc):
            dy = _solve_normal(M, rp - As @ (rc / s) + As @ (D * rd))
            ds = rd - As.T @ dy
            dx = rc / s - D * ds
            return dx, dy, ds

        dxa, _, dsa = direction(-x * s)
        ap, ad = _max_step(x, dxa), _max_step(s, dsa)
        mu_aff = float((x + ap * dxa) @ (s + ad * dsa) / n)
        sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0
        dx, dy, ds = direction(-x * s - dxa * dsa + sigma * mu)
        ap, ad = 0.995 * _max_step(x, dx), 0.995 * _max_step(s, ds)
        x += ap * dx
        y += ad * dy
        s += ad * ds

    xo = (beta * x[:n_orig]).tolist()
    z = float(np.dot(c, xo))
    iters = len(hist) - 1
    if status == OPTIMAL:
        txt = ", ".join(f"x{i + 1}={v:.4g}" for i, v in enumerate(xo))
        resumen = f"{txt}, Z* = {z:.6g} en {iters} iteraciones"
    else:
        resumen = "No convergió: el problema puede ser infactible o no acotado"
    P = np.array(path)

    def plot(fig):
        ax = fig.add_subplot(111)
        mus = [max(h["μ"], 1e-300) for h in hist]
        ax.semilogy(range(len(mus)), mus, "o-", color="#4a9eff")
        ax.set_xlabel("Iteración"); ax.set_ylabel("μ (brecha de complementariedad)")
        ax.set_title("Convergencia del punto interior")

    return {"status": status, "x": xo, "z": z if status == OPTIMAL else None, "iters": iters,
            "path": P.tolist(), "hist": hist, "plot": plot, "resumen": resumen}


def klee_minty_problem(n: int):
    """max Σ 2^(n−j)·x_j  s.a.  Σ_{j<i} 2^(i−j+1)·x_j + x_i ≤ 5^i,  x ≥ 0."""
    c = [2.0 ** (n - j) for j in range(1, n + 1)]
    A = [[2.0 ** (i - j + 1) if j < i else (1.0 if j == i else 0.0)
          for j in range(1, n + 1)] for i in range(1, n + 1)]
    b = [5.0 ** i for i in range(1, n + 1)]
    return c, A, ["<="] * n, b


def klee_minty(n_max: int) -> Dict[str, Any]:
    if not 2 <= n_max <= 12:
        raise ValueError("La dimensión n debe estar entre 2 y 12.")
    hist = []
    sizes, spx, ipm = [], [], []
    for n in range(2, n_max + 1):
        c, A, signs, b = klee_minty_problem(n)
        r_s = solve_lp(c, A, signs, b, "max", "simplex", keep_tableaus=False, max_iter=10000)
        r_i = interior_point(c, A, signs, b, "max")
        sizes.append(n); spx.append(r_s["pivots"]); ipm.append(r_i["iters"])
        hist.append({"n": n, "Pivotes Simplex": r_s["pivots"], "2^n − 1": 2 ** n - 1,
                     "Iteraciones punto interior": r_i["iters"],
                     "Z* Simplex": float(r_s["z"]),
                     "Z* punto interior": r_i["z"] if r_i["z"] is not None else "No convergió",
                     "Z* teórico (5^n)": float(5.0 ** n)})

    def plot(fig):
        ax = fig.add_subplot(111)
        ax.semilogy(sizes, spx, "o-", color="#ff6b35", label="Simplex (Dantzig)")
        ax.semilogy(sizes, ipm, "s-", color="#4a9eff", label="Punto interior")
        ax.set_xlabel("n"); ax.set_ylabel("Iteraciones (escala log)")
        ax.set_title("Klee-Minty: Simplex exponencial, punto interior casi constante")
        ax.legend()

    return {"hist": hist, "plot": plot, "pivots": spx, "ipm": ipm,
            "resumen": f"n = {n_max}: Simplex {spx[-1]} pivotes, punto interior {ipm[-1]} iteraciones"}
