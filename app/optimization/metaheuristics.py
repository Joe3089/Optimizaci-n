"""
metaheuristics.py — Métodos Metaheurísticos
============================================
Implementa:
  • simulated_annealing  — Recocido Simulado (SA)
  • pso                  — Optimización por Enjambre de Partículas (PSO)
  • genetic_algorithm    — Algoritmo Genético (GA)

Todos son estocásticos, no requieren derivadas y pueden escapar
mínimos locales. Retornan historial compatible con la interfaz.
"""

from __future__ import annotations
import math
import random
import numpy as np
from typing import Callable, List, Dict, Any, Tuple

try:
    import sympy as sp
    _SYMPY = True
except ImportError:
    _SYMPY = False


# ─── Utilidad: crear función callable desde string ───────────────────────────
def _make_fn(func_str: str, var_str: str) -> Tuple[Callable, int]:
    """Retorna (f_callable, n_vars) desde string."""
    if not _SYMPY:
        raise RuntimeError("SymPy no disponible.")
    syms = sp.symbols(var_str)
    if not isinstance(syms, (list, tuple)):
        syms = (syms,)
    n = len(syms)
    f_sym   = sp.sympify(func_str)
    modules = [{"DiracDelta": lambda *a: 0.0,
                "Heaviside":  lambda x, *a: np.heaviside(x, 0.5)}, "numpy"]
    f_num = sp.lambdify(list(syms), f_sym, modules=modules)

    def fn(x: np.ndarray) -> float:
        v = tuple(x.tolist()) if n > 1 else (float(x[0]),)
        try:
            r = f_num(*v)
            return float(r)
        except Exception:
            return float("inf")

    return fn, n


# ══════════════════════════════════════════════════════════════════════════════
#  1. SIMULATED ANNEALING (Recocido Simulado)
# ══════════════════════════════════════════════════════════════════════════════
def simulated_annealing(func_str: str, var_str: str, x0: list,
                        T_init: float = 100.0, T_min: float = 1e-4,
                        cooling: float = 0.95, max_iter: int = 500,
                        step_size: float = 0.3,
                        seed: int = 42) -> List[Dict[str, Any]]:
    """
    Recocido Simulado (Simulated Annealing).

    Inspirado en el proceso físico de enfriamiento lento de metales.
    Acepta soluciones peores con probabilidad:
        P(aceptar) = exp(−ΔE / T)

    Esto le permite escapar mínimos locales. A medida que T → 0,
    el algoritmo se vuelve más selectivo y converge al mínimo.

    Parámetros:
      T_init    : Temperatura inicial (controla exploración)
      T_min     : Temperatura final (criterio de parada)
      cooling   : Factor de enfriamiento por iteración (ej: 0.95)
      step_size : Amplitud del vecino aleatorio
      seed      : Semilla para reproducibilidad
    """
    rng = np.random.default_rng(seed)
    fn, n = _make_fn(func_str, var_str)

    x_cur  = np.array(x0, dtype=float)
    f_cur  = fn(x_cur)
    x_best = x_cur.copy()
    f_best = f_cur
    T      = float(T_init)
    history: List[Dict[str, Any]] = []

    for k in range(max_iter):
        if T < T_min:
            break

        # Generar vecino aleatorio
        delta  = rng.normal(0, step_size, size=n)
        x_new  = x_cur + delta
        f_new  = fn(x_new)
        dE     = f_new - f_cur

        # Criterio de aceptación
        if dE < 0:
            accept = True
            prob   = 1.0
        else:
            prob   = math.exp(-dE / T) if T > 1e-15 else 0.0
            accept = rng.random() < prob

        if accept:
            x_cur = x_new
            f_cur = f_new

        if f_cur < f_best:
            x_best = x_cur.copy()
            f_best = f_cur

        history.append({
            "iter"      : k + 1,
            "T"         : round(T, 6),
            "x_k_str"   : "[" + " ".join(f"{v:.5g}" for v in x_cur) + "]",
            "f_k"       : round(f_cur, 8),
            "f_best"    : round(f_best, 8),
            "delta_E"   : round(dE, 6),
            "prob_acept": round(prob, 5),
            "aceptado"  : accept,
        })

        T *= cooling

    return history


# ══════════════════════════════════════════════════════════════════════════════
#  2. PSO — Particle Swarm Optimization (Enjambre de Partículas)
# ══════════════════════════════════════════════════════════════════════════════
def pso(func_str: str, var_str: str, bounds: list,
        n_particles: int = 20, max_iter: int = 100,
        w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
        seed: int = 42) -> List[Dict[str, Any]]:
    """
    Particle Swarm Optimization (PSO).

    Cada partícula mantiene:
      • posición   x_i
      • velocidad  v_i
      • mejor posición personal   pbest_i
      • mejor posición global     gbest

    Actualización de velocidad:
      v_i ← w·v_i + c1·r1·(pbest_i − x_i) + c2·r2·(gbest − x_i)

    Parámetros:
      n_particles : número de partículas (enjambre)
      w           : inercia (balance exploración/explotación)
      c1          : coeficiente cognitivo (atracción a pbest)
      c2          : coeficiente social    (atracción a gbest)
      bounds      : [[lb1,ub1], [lb2,ub2], ...] por variable
      seed        : semilla aleatoria
    """
    rng = np.random.default_rng(seed)
    fn, n = _make_fn(func_str, var_str)

    # Convertir bounds: aceptar [lo, hi] o [[lo1,hi1],[lo2,hi2]]
    if isinstance(bounds[0], (int, float)):
        lb = np.full(n, bounds[0], dtype=float)
        ub = np.full(n, bounds[1], dtype=float)
    else:
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)

    # Inicializar enjambre
    pos  = lb + rng.random((n_particles, n)) * (ub - lb)
    vel  = rng.uniform(-(ub - lb), (ub - lb), (n_particles, n))
    pbest = pos.copy()
    pbest_val = np.array([fn(pos[i]) for i in range(n_particles)])

    gbest_idx = int(np.argmin(pbest_val))
    gbest     = pbest[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])

    history: List[Dict[str, Any]] = []

    for k in range(max_iter):
        r1 = rng.random((n_particles, n))
        r2 = rng.random((n_particles, n))

        vel  = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos  = np.clip(pos + vel, lb, ub)

        for i in range(n_particles):
            fval = fn(pos[i])
            if fval < pbest_val[i]:
                pbest[i]     = pos[i].copy()
                pbest_val[i] = fval
                if fval < gbest_val:
                    gbest     = pos[i].copy()
                    gbest_val = fval

        # Diversidad: std promedio de posiciones
        diversity = float(np.mean(np.std(pos, axis=0)))

        history.append({
            "iter"      : k + 1,
            "x_k_str"   : "[" + " ".join(f"{v:.5g}" for v in gbest) + "]",
            "f_k"       : round(gbest_val, 8),
            "f_best"    : round(gbest_val, 8),
            "diversidad": round(diversity, 5),
            "vel_media" : round(float(np.mean(np.abs(vel))), 5),
        })

        if gbest_val < 1e-10 or diversity < 1e-8:
            break

    return history


# ══════════════════════════════════════════════════════════════════════════════
#  3. ALGORITMO GENÉTICO
# ══════════════════════════════════════════════════════════════════════════════
def genetic_algorithm(func_str: str, var_str: str, bounds: list,
                      pop_size: int = 40, max_gen: int = 100,
                      p_cross: float = 0.8, p_mut: float = 0.15,
                      elite: int = 2, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Algoritmo Genético (GA) para minimización con codificación real.

    Operadores:
      • Selección   : Torneo de 3 individuos
      • Cruce       : Cruce aritmético (blend crossover BLX-α)
      • Mutación    : Mutación gaussiana adaptativa
      • Elitismo    : Los `elite` mejores pasan directamente

    Parámetros:
      pop_size : tamaño de la población
      max_gen  : número de generaciones
      p_cross  : probabilidad de cruce
      p_mut    : probabilidad de mutación por gen
      elite    : número de individuos elite (elitismo)
      bounds   : [lo, hi] o [[lo1,hi1],[lo2,hi2],...]
      seed     : semilla
    """
    rng = np.random.default_rng(seed)
    fn, n = _make_fn(func_str, var_str)

    if isinstance(bounds[0], (int, float)):
        lb = np.full(n, bounds[0], dtype=float)
        ub = np.full(n, bounds[1], dtype=float)
    else:
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)

    rng_range = ub - lb

    # Población inicial
    pop = lb + rng.random((pop_size, n)) * rng_range
    fit = np.array([fn(pop[i]) for i in range(pop_size)])

    def _tournament(k: int = 3) -> int:
        idx = rng.choice(pop_size, k, replace=False)
        return int(idx[np.argmin(fit[idx])])

    def _crossover(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha = 0.5
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
        return np.clip(c1, lb, ub), np.clip(c2, lb, ub)

    def _mutate(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        mask = rng.random(n) < p_mut
        x = x + mask * rng.normal(0, sigma * rng_range, n)
        return np.clip(x, lb, ub)

    history: List[Dict[str, Any]] = []
    best_idx = int(np.argmin(fit))
    best_x   = pop[best_idx].copy()
    best_f   = float(fit[best_idx])

    for g in range(max_gen):
        # Calcular sigma adaptativo (se reduce con generaciones)
        sigma = 0.3 * (1 - g / max_gen) + 0.02

        # Nueva generación
        order  = np.argsort(fit)
        new_pop = pop[order[:elite]].copy()   # elitismo
        new_fit = fit[order[:elite]].copy()

        while len(new_pop) < pop_size:
            p1_idx = _tournament()
            p2_idx = _tournament()
            c1, c2 = _crossover(pop[p1_idx], pop[p2_idx]) \
                     if rng.random() < p_cross \
                     else (pop[p1_idx].copy(), pop[p2_idx].copy())
            c1 = _mutate(c1, sigma)
            c2 = _mutate(c2, sigma)
            for child in (c1, c2):
                if len(new_pop) < pop_size:
                    new_pop = np.vstack([new_pop, child])
                    new_fit = np.append(new_fit, fn(child))

        pop = new_pop[:pop_size]
        fit = new_fit[:pop_size]

        gen_best_idx = int(np.argmin(fit))
        gen_best_f   = float(fit[gen_best_idx])
        if gen_best_f < best_f:
            best_f = gen_best_f
            best_x = pop[gen_best_idx].copy()

        # Diversidad genética: std media de la población
        diversity = float(np.mean(np.std(pop, axis=0)))

        history.append({
            "gen"       : g + 1,
            "x_k_str"   : "[" + " ".join(f"{v:.5g}" for v in best_x) + "]",
            "f_k"       : round(gen_best_f, 8),
            "f_best"    : round(best_f, 8),
            "f_media"   : round(float(np.mean(fit)), 5),
            "diversidad": round(diversity, 5),
            "sigma"     : round(sigma, 4),
        })

        if best_f < 1e-10 or diversity < 1e-9:
            break

    return history
