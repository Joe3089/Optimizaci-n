# -*- coding: utf-8 -*-
"""
localsearch.py
==============
Métodos de búsqueda en línea (1-D) para el Optimizador de Funciones.

Funciones exportadas
--------------------
local_neighborhood_search(f, lo, hi, step, n_iter)
    Búsqueda local por vecindad (grid adaptativo con refinamiento).

fibonacci_search(f, lo, hi, tol)
    Método de Fibonacci para minimización unimodal en [lo, hi].
"""

import math


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad interna
# ─────────────────────────────────────────────────────────────────────────────

def _safe_call(f, x):
    """Evalúa f(x) devolviendo inf si falla."""
    try:
        v = f(x)
        return float(v) if math.isfinite(float(v)) else math.inf
    except Exception:
        return math.inf


# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda Local por Vecindad
# ─────────────────────────────────────────────────────────────────────────────

def local_neighborhood_search(f, lo, hi, step=None, n_iter=20):
    """
    Búsqueda local por vecindad en 1-D.

    Realiza un barrido inicial uniforme con `step` para localizar el mejor
    punto y después refina iterativamente el intervalo alrededor de ese punto,
    reduciendo el paso a la mitad en cada iteración.

    Parámetros
    ----------
    f      : callable  — función objetivo f(x) → float
    lo     : float     — límite inferior del intervalo
    hi     : float     — límite superior del intervalo
    step   : float     — paso inicial (por defecto (hi-lo)*0.1)
    n_iter : int       — número de iteraciones de refinamiento

    Retorna
    -------
    (x_opt, f_opt, f_lo, f_hi, log)
    log : list[dict]  — historial de iteraciones con claves
                        'k', 'lambda_k', 'f_lambda', 'a', 'b', 'step'
    """
    lo, hi = float(lo), float(hi)
    if step is None or step <= 0:
        step = (hi - lo) * 0.1
    else:
        step = float(step)
    n_iter = max(1, int(n_iter))

    log = []
    f_lo = _safe_call(f, lo)
    f_hi = _safe_call(f, hi)

    # ── Barrido inicial ──────────────────────────────────────────────────────
    x_best = lo
    f_best = f_lo
    x = lo
    k = 0
    while x <= hi + 1e-14:
        fx = _safe_call(f, x)
        log.append({
            "k":        k,
            "iter":     k,
            "lambda_k": x,
            "x":        x,
            "f_lambda": fx,
            "f_k":      fx,
            "f(x)":     fx,
            "a":        lo,
            "b":        hi,
            "step":     step,
        })
        if fx < f_best:
            f_best = fx
            x_best = x
        x += step
        k += 1

    # ── Refinamiento iterativo ───────────────────────────────────────────────
    a_ref = max(lo, x_best - step)
    b_ref = min(hi, x_best + step)
    step_ref = step / 2.0

    for _ in range(n_iter):
        if step_ref < 1e-14:
            break
        x = a_ref
        while x <= b_ref + 1e-14:
            fx = _safe_call(f, x)
            log.append({
                "k":        k,
                "iter":     k,
                "lambda_k": x,
                "x":        x,
                "f_lambda": fx,
                "f_k":      fx,
                "f(x)":     fx,
                "a":        a_ref,
                "b":        b_ref,
                "step":     step_ref,
            })
            if fx < f_best:
                f_best = fx
                x_best = x
            x += step_ref
            k += 1

        a_ref = max(lo, x_best - step_ref)
        b_ref = min(hi, x_best + step_ref)
        step_ref /= 2.0

    return x_best, f_best, f_lo, f_hi, log


# ─────────────────────────────────────────────────────────────────────────────
# Método de Fibonacci
# ─────────────────────────────────────────────────────────────────────────────

def fibonacci_search(f, lo, hi, tol=1e-6):
    """
    Minimización unimodal en 1-D por el método de Fibonacci.

    Parámetros
    ----------
    f   : callable — función objetivo f(x) → float
    lo  : float    — límite inferior
    hi  : float    — límite superior
    tol : float    — tolerancia de convergencia (|b-a| < tol para parar)

    Retorna
    -------
    (x_opt, f_opt, f_lo, f_hi, log)
    log : list[dict]  — historial con claves
                        'k', 'lambda_k', 'f_lambda', 'mu_k', 'f_mu', 'a', 'b'
    """
    lo, hi = float(lo), float(hi)
    tol = max(float(tol), 1e-14)

    # ── Número de evaluaciones: F_N ≥ (b − a) / tol ─────────────────────────
    fibs = [1, 1]
    while fibs[-1] < (hi - lo) / tol:
        fibs.append(fibs[-1] + fibs[-2])
    if len(fibs) < 4:                       # garantiza al menos una reducción
        fibs += [fibs[-1] + fibs[-2], fibs[-1] * 2 + fibs[-2]]
    n = len(fibs) - 1

    f_lo = _safe_call(f, lo)
    f_hi = _safe_call(f, hi)

    a, b = lo, hi
    log  = []
    lam = a + fibs[n - 2] / fibs[n] * (b - a)
    mu  = a + fibs[n - 1] / fibs[n] * (b - a)
    f_lam = _safe_call(f, lam)
    f_mu  = _safe_call(f, mu)

    # Recurrencia estándar: λ_k, μ_k ∈ [a_k, b_k] y el intervalo se reduce en
    # la razón F_{n−k}/F_{n−k+1} en cada paso (k = 1 … n−2).
    for k in range(1, n - 1):
        log.append({
            "k":        k,
            "iter":     k,
            "lambda_k": lam,
            "x":        lam,
            "f_lambda": f_lam,
            "f_k":      f_lam,
            "f(x)":     f_lam,
            "mu_k":     mu,
            "f_mu":     f_mu,
            "a":        a,
            "b":        b,
        })
        if f_lam > f_mu:
            a, lam, f_lam = lam, mu, f_mu
            mu   = a + fibs[n - k - 1] / fibs[n - k] * (b - a)
            f_mu = _safe_call(f, mu)
        else:
            b, mu, f_mu = mu, lam, f_lam
            lam   = a + fibs[n - k - 2] / fibs[n - k] * (b - a)
            f_lam = _safe_call(f, lam)

    # Resultado del algoritmo: centro del intervalo final (se registra en el
    # historial para que tabla, gráficas y resumen muestren el mismo x*).
    x_opt = (a + b) / 2.0
    f_opt = _safe_call(f, x_opt)
    k_fin = len(log) + 1
    log.append({
        "k": k_fin, "iter": k_fin, "lambda_k": x_opt, "x": x_opt,
        "f_lambda": f_opt, "f_k": f_opt, "f(x)": f_opt,
        "mu_k": "—", "f_mu": "—", "a": a, "b": b,
    })

    return x_opt, f_opt, f_lo, f_hi, log
