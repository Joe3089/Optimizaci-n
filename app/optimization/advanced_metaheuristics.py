# app/optimization/advanced_metaheuristics.py
# Motor metaheurístico general para problemas complejos:
#   • f(x) de n variables, minimizar o maximizar
#   • variables reales, enteras o binarias con cotas propias
#   • restricciones ≤, ≥ y = (reglas de factibilidad de Deb)
#   • Evolución Diferencial, PSO, Algoritmo Genético, Recocido Simulado,
#     Búsqueda Tabú, o la comparación de todos
#   • Viajante (TSP) de gran tamaño con Colonia de Hormigas, Recocido
#     Simulado (2-opt) y Algoritmo Genético (cruce OX)
# Las expresiones se convierten con el parser del proyecto (func_compat);
# antes se rechaza todo lo que no sea una expresión matemática.
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.application import func_compat as fc

ALGORITHMS = ("Evolución Diferencial", "CMA-ES", "PSO: Enjambre de Partículas",
              "Algoritmo Genético", "Recocido Simulado", "Búsqueda Tabú", "Comparar todos")
TSP_ALGORITHMS = ("Colonia de Hormigas (ACO)", "Recocido Simulado (2-opt)",
                  "Algoritmo Genético (cruce OX)", "Comparar todos")
EQ_TOL = 1e-4                     # tolerancia de las restricciones de igualdad
_FORBIDDEN = re.compile(r"__|\bimport\b|\blambda\b|\bexec\b|\beval\b|\bopen\b|[\"';`$\\{}\[\]@]")
_COLORS = {"Evolución Diferencial": "#ffd700", "PSO: Enjambre de Partículas": "#4a9eff",
           "Algoritmo Genético": "#3cff8a", "Recocido Simulado": "#ff6b35",
           "Búsqueda Tabú": "#c77dff", "CMA-ES": "#ff5c8a"}


# ── Expresiones seguras ──────────────────────────────────────────────────────
def _check_expr(expr: str, what: str) -> str:
    e = (expr or "").strip()
    if not e:
        raise ValueError(f"Falta {what}.")
    if _FORBIDDEN.search(e):
        raise ValueError(f"{what[0].upper() + what[1:]} contiene caracteres o palabras no "
                         f"permitidos: «{e}». Usa solo una expresión matemática.")
    return fc.normalize_expr(e)


def _vector_fn(expr: str, names: List[str]) -> Callable[[np.ndarray], np.ndarray]:
    """f(X) para una población X (filas = individuos); valores no finitos → inf."""
    try:
        f = fc.make_callable(expr, names)
    except Exception as ex:
        raise ValueError(f"No se pudo interpretar «{expr}»: {ex}") from ex

    def ev(X: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            try:
                out = f(*[X[:, j] for j in range(X.shape[1])])
                out = np.broadcast_to(np.asarray(out, dtype=float), (X.shape[0],)).copy()
            except Exception:
                out = np.array([_scalar(f, x) for x in X], dtype=float)
        out[~np.isfinite(out)] = np.inf
        return out
    return ev


def _scalar(f, x) -> float:
    try:
        return float(f(*x))
    except Exception:
        return float("inf")


# ── Definición del problema ──────────────────────────────────────────────────
@dataclass
class Var:
    name: str
    lo: float
    hi: float
    kind: str = "real"        # real | entera | binaria


@dataclass
class Problem:
    names: List[str]
    vars: List[Var]
    sense: int                # +1 minimizar, −1 maximizar
    f: Callable
    g: List[Tuple[Callable, str]]   # (h(X), "<=" | "=") con h ≤ 0 o h = 0
    objective: str
    constraints_txt: List[str]

    @property
    def lo(self):
        return np.array([v.lo for v in self.vars])

    @property
    def hi(self):
        return np.array([v.hi for v in self.vars])

    def repair(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(X, self.lo, self.hi)
        for j, v in enumerate(self.vars):
            if v.kind != "real":
                X[:, j] = np.clip(np.round(X[:, j]), v.lo, v.hi)
        return X

    def evaluate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(f original, objetivo a minimizar, violación total)."""
        fx = self.f(X)
        viol = np.zeros(X.shape[0])
        for h, op in self.g:
            hv = h(X)
            hv[~np.isfinite(hv)] = 1e12
            viol += np.maximum(hv, 0.0) if op == "<=" else np.maximum(np.abs(hv) - EQ_TOL, 0.0)
        return fx, self.sense * fx, viol


def _better(o1, v1, o2, v2):
    """Reglas de Deb: factible > infactible; entre factibles, menor objetivo;
    entre infactibles, menor violación. Vectorizado: True si (o1,v1) es mejor."""
    f1, f2 = v1 <= 0, v2 <= 0
    return np.where(f1 & f2, o1 < o2, np.where(f1 != f2, f1, v1 < v2))


def _best_index(obj, viol) -> int:
    feas = viol <= 0
    if feas.any():
        idx = np.where(feas)[0]
        return int(idx[np.argmin(obj[idx])])
    return int(np.argmin(viol))


_VAR_LINE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*[:=]\s*(?:(real|continua|entera|entero|binaria|binario)\s*)?"
    r"(?:\[|\()?\s*([-+.\deE]+)?\s*[,;]?\s*([-+.\deE]+)?\s*(?:\]|\))?\s*$", re.IGNORECASE)


def parse_problem(objective: str, sense: str, variables: str, constraints: str) -> Problem:
    obj = _check_expr(objective, "la función objetivo f(x)")
    cons = [ln.strip() for ln in (constraints or "").splitlines() if ln.strip()]
    parsed_c: List[Tuple[str, str, str]] = []
    for c in cons:
        m = re.match(r"^(.*?)(<=|>=|≤|≥|==|=)(.*)$", c)
        if not m or not m.group(1).strip() or not m.group(3).strip():
            raise ValueError(f"Restricción inválida: «{c}». Ejemplo: x1 + x2 <= 10.")
        lhs = _check_expr(m.group(1), "el lado izquierdo de una restricción")
        rhs = _check_expr(m.group(3), "el lado derecho de una restricción")
        op = {"≤": "<=", "≥": ">=", "==": "="}.get(m.group(2), m.group(2))
        parsed_c.append((lhs, op, rhs))

    names: List[str] = []
    for e in [obj] + [p for c in parsed_c for p in (c[0], c[2])]:
        for v in fc.detect_variables(e):
            if v not in names:
                names.append(v)
    if not names:
        raise ValueError("La función objetivo no tiene variables.")
    names = fc.detect_variables(" + ".join(names))      # orden estándar x1, x2, …

    spec: Dict[str, Var] = {}
    default: Optional[Var] = None
    for ln in (variables or "").splitlines():
        if not ln.strip():
            continue
        m = _VAR_LINE.match(ln)
        if not m:
            raise ValueError(f"Línea de variable inválida: «{ln.strip()}». Ejemplos: "
                             "«x1: -5, 5», «x2: entera 0, 10», «x3: binaria», «todas: -10, 10».")
        name, kind, a, b = m.group(1), (m.group(2) or "real").lower(), m.group(3), m.group(4)
        kind = {"entero": "entera", "binario": "binaria", "continua": "real"}.get(kind, kind)
        if kind == "binaria":
            lo, hi = 0.0, 1.0
        else:
            if a is None or b is None:
                raise ValueError(f"Faltan las cotas de «{name}». Escribe «{name}: mínimo, máximo».")
            lo, hi = float(a), float(b)
            if lo >= hi:
                raise ValueError(f"En «{name}» el mínimo ({lo:g}) debe ser menor que el máximo ({hi:g}).")
        var = Var(name, lo, hi, kind)
        if name.lower() in ("todas", "todos", "default", "general"):
            default = var
        else:
            spec[name] = var
    unknown = [n for n in spec if n not in names]
    if unknown:
        raise ValueError(f"Las variables {', '.join(unknown)} no aparecen en la función ni en "
                         "las restricciones.")
    vars_: List[Var] = []
    for n in names:
        if n in spec:
            vars_.append(spec[n])
        elif default is not None:
            vars_.append(Var(n, default.lo, default.hi, default.kind))
        else:
            raise ValueError(f"Faltan las cotas de la variable «{n}». Agrega una línea "
                             f"«{n}: mínimo, máximo» o «todas: mínimo, máximo».")

    f = _vector_fn(obj, names)
    g = []
    for lhs, op, rhs in parsed_c:
        h = _vector_fn(f"({lhs}) - ({rhs})", names) if op != ">=" else \
            _vector_fn(f"({rhs}) - ({lhs})", names)
        g.append((h, "=" if op == "=" else "<="))
    return Problem(names, vars_, -1 if sense.lower().startswith("max") else 1, f, g, obj,
                   [f"{a} {o} {b}" for a, o, b in parsed_c])


# ── Algoritmos (todas usan las mismas reglas y registran el mejor por iteración) ─
class _Recorder:
    def __init__(self, prob: Problem, name: str):
        self.p, self.name = prob, name
        self.best_x = None
        self.best_o, self.best_v = np.inf, np.inf
        self.rows: List[Tuple[int, np.ndarray, float, float]] = []
        self.evals = 0
        self.polished = False

    def offer(self, X, obj, viol):
        i = _best_index(obj, viol)
        if self.best_x is None or _better(obj[i], viol[i], self.best_o, self.best_v):
            self.best_x, self.best_o, self.best_v = X[i].copy(), float(obj[i]), float(viol[i])

    def log(self, it):
        self.rows.append((it, self.best_x.copy(), self.best_o, self.best_v))


def _init_pop(p: Problem, n, rng):
    X = p.lo + rng.random((n, len(p.vars))) * (p.hi - p.lo)
    return p.repair(X)


def _eval(p: Problem, X, rec: _Recorder):
    fx, obj, viol = p.evaluate(X)
    rec.evals += X.shape[0]
    rec.offer(X, obj, viol)
    return obj, viol


def differential_evolution(p, iters, pop, rng):
    """jDE (Brest et al., 2006): cada individuo adapta su F y su CR."""
    rec = _Recorder(p, "Evolución Diferencial")
    d = len(p.vars)
    X = _init_pop(p, pop, rng); o, v = _eval(p, X, rec); rec.log(0)
    F = np.full(pop, 0.5); CR = np.full(pop, 0.9)
    ar = np.arange(pop)
    for it in range(1, iters + 1):
        Fn = np.where(rng.random(pop) < 0.1, 0.1 + 0.9 * rng.random(pop), F)
        CRn = np.where(rng.random(pop) < 0.1, rng.random(pop), CR)
        r = np.argsort(rng.random((pop, pop)), axis=1)
        r = np.array([row[row != k][:3] for k, row in enumerate(r)])     # 3 distintos de i
        mut = X[r[:, 0]] + Fn[:, None] * (X[r[:, 1]] - X[r[:, 2]])
        cross = rng.random((pop, d)) < CRn[:, None]
        cross[ar, rng.integers(0, d, pop)] = True
        T = p.repair(np.where(cross, mut, X))
        ot, vt = _eval(p, T, rec)
        win = _better(ot, vt, o, v) | ((ot == o) & (vt == v))
        X[win], o[win], v[win] = T[win], ot[win], vt[win]
        F[win], CR[win] = Fn[win], CRn[win]
        rec.log(it)
    return rec


def particle_swarm(p, iters, pop, rng):
    rec = _Recorder(p, "PSO: Enjambre de Partículas")
    span = p.hi - p.lo
    X = _init_pop(p, pop, rng); V = (rng.random(X.shape) - 0.5) * span * 0.2
    o, v = _eval(p, X, rec); rec.log(0)
    P, po, pv = X.copy(), o.copy(), v.copy()
    for it in range(1, iters + 1):
        w = 0.9 - 0.5 * it / iters
        g = P[_best_index(po, pv)]
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = w * V + 1.5 * r1 * (P - X) + 1.5 * r2 * (g - X)
        V = np.clip(V, -span * 0.5, span * 0.5)
        X = p.repair(X + V)
        o, v = _eval(p, X, rec)
        win = _better(o, v, po, pv)
        P[win], po[win], pv[win] = X[win], o[win], v[win]
        rec.log(it)
    return rec


def genetic(p, iters, pop, rng, pc=0.9, elite=2):
    rec = _Recorder(p, "Algoritmo Genético")
    d = len(p.vars)
    span = p.hi - p.lo
    X = _init_pop(p, pop, rng); o, v = _eval(p, X, rec); rec.log(0)
    pm = 1.0 / d
    for it in range(1, iters + 1):
        def pick():
            a, b = rng.integers(0, pop, 2)
            return a if _better(o[a], v[a], o[b], v[b]) else b
        order = sorted(range(pop), key=lambda i: (v[i] > 0, v[i] if v[i] > 0 else o[i]))
        kids = [X[i].copy() for i in order[:elite]]
        sigma = 0.1 * (1 - it / (iters + 1)) + 0.01
        while len(kids) < pop:
            a, b = X[pick()], X[pick()]
            if rng.random() < pc:                       # cruce aritmético BLX
                alpha = rng.random(d) * 1.5 - 0.25
                c1, c2 = alpha * a + (1 - alpha) * b, alpha * b + (1 - alpha) * a
            else:
                c1, c2 = a.copy(), b.copy()
            for c in (c1, c2):
                m = rng.random(d) < pm
                c[m] += rng.normal(0, sigma, m.sum()) * span[m]
                kids.append(c)
        X = p.repair(np.array(kids[:pop]))
        o, v = _eval(p, X, rec)
        rec.log(it)
    return rec


def _disc_delta(p: Problem, j: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """En variables enteras/binarias el paso se redondea a ±1 como mínimo (si no,
    el redondeo anularía el movimiento)."""
    disc = np.array([p.vars[int(k)].kind != "real" for k in np.atleast_1d(j)])
    if not disc.any():
        return delta
    d2 = np.array(delta, dtype=float)
    sgn = np.where(d2 == 0, 1.0, np.sign(d2))
    d2[disc] = sgn[disc] * np.maximum(1.0, np.round(np.abs(d2[disc])))
    return d2


def _energy(o, v):
    return o + 1e6 * v                      # energía penalizada (SA / Tabú / pulido)


def annealing(p, iters, pop, rng):
    """Recocido Simulado. Con variables reales y sin restricciones usa basin
    hopping (Metropolis sobre mínimos locales, Wales y Doye 1997); en otro caso,
    intercambio de réplicas con paso adaptativo."""
    if not p.g and all(v.kind == "real" for v in p.vars):
        try:
            import scipy.optimize  # noqa: F401
            return _annealing_bh(p, iters, pop, rng)
        except ImportError:
            pass
    return _annealing_pt(p, iters, pop, rng)


def _local_min(p, x, rec):
    from scipy.optimize import minimize
    lo, hi = p.lo, p.hi

    def fun(z):
        val = float(p.sense * p.f(np.clip(z, lo, hi)[None, :])[0])
        return val if math.isfinite(val) else 1e300
    r = minimize(fun, x, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                 options={"maxiter": 300})
    rec.evals += int(r.nfev)
    y = np.clip(r.x, lo, hi)
    _eval(p, y[None, :], rec)
    return y, float(r.fun)


def _annealing_bh(p, iters, pop, rng):
    rec = _Recorder(p, "Recocido Simulado")
    d = len(p.vars)
    span = p.hi - p.lo
    S = _init_pop(p, max(10, pop), rng)
    so, _ = _eval(p, S, rec)
    x, e = _local_min(p, S[int(np.argmin(so))], rec)
    fin = so[np.isfinite(so)]
    T0 = max(1e-3, 0.02 * float(np.std(fin))) if fin.size > 1 else 1.0
    step = 0.1 * span
    acc, tried = 0, 0
    rec.log(0)
    for it in range(1, iters + 1):
        T = T0 * (0.01 ** (it / iters))
        k = 1 if rng.random() < 0.9 else int(rng.integers(1, max(2, d // 4) + 1))
        j = rng.choice(d, k, replace=False)
        y0 = x.copy()
        y0[j] += rng.choice((-1.0, 1.0), k) * (0.5 + rng.random(k)) * step[j]
        y, ey = _local_min(p, np.clip(y0, p.lo, p.hi), rec)
        tried += 1
        if ey <= e or rng.random() < math.exp(-(ey - e) / max(T, 1e-300)):
            x, e = y, ey
            acc += 1
        rec.log(it)
    return rec


def cma_es(p, iters, pop, rng):
    """CMA-ES (Hansen) en el espacio normalizado [0, 1]^n con reinicios IPOP
    (población doble en cada reinicio) y ranking por reglas de Deb."""
    rec = _Recorder(p, "CMA-ES")
    d = len(p.vars)
    lo, span = p.lo, p.hi - p.lo
    lam0 = max(pop, 4 + int(3 * np.log(d)))
    it, restarts = 0, 0
    _eval(p, p.repair((lo + 0.5 * span)[None, :]), rec)
    rec.log(0)
    while it < iters:
        lam = lam0 * 2 ** restarts
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
        mueff = 1 / (w ** 2).sum()
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        chiN = math.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d))
        m = rng.random(d) if restarts else np.full(d, 0.5)
        sigma, pc, ps = 0.3, np.zeros(d), np.zeros(d)
        B, Dg, C = np.eye(d), np.ones(d), np.eye(d)
        gen, stall, last = 0, 0, np.inf
        while it < iters:
            it += 1; gen += 1
            Z = rng.standard_normal((lam, d))
            Y = Z @ (B * Dg).T
            Xn = np.clip(m + sigma * Y, 0, 1)
            X = p.repair(lo + Xn * span)
            o, v = _eval(p, X, rec)
            key = np.where(v > 0, v, o)
            idx = np.lexsort((key, v > 0))[:mu]
            m_old = m
            ysel = (Xn[idx] - m_old) / sigma
            m = np.clip(m_old + sigma * (w @ ysel), 0, 1)
            inv = B @ np.diag(1 / Dg) @ B.T
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * inv @ (m - m_old) / sigma
            hs = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2 / (d + 1)
            pc = (1 - cc) * pc + hs * math.sqrt(cc * (2 - cc) * mueff) * (m - m_old) / sigma
            C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C)
                 + cmu * (ysel.T * w) @ ysel)
            sigma = min(sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1)), 1.0)
            C = np.triu(C) + np.triu(C, 1).T
            ev, B = np.linalg.eigh(C)
            Dg = np.sqrt(np.maximum(ev, 1e-20))
            rec.log(it)
            cur = float(key[idx[0]])
            stall = stall + 1 if cur >= last - 1e-12 * max(1.0, abs(last)) else 0
            last = min(last, cur)
            if sigma < 1e-11 or Dg.max() / Dg.min() > 1e7 or stall > 10 + 30 * d / lam:
                restarts += 1                         # reinicio IPOP
                break
    return rec


def _annealing_pt(p, iters, pop, rng):
    """Recocido Simulado con intercambio de réplicas (parallel tempering): K
    cadenas a temperaturas geométricas que se enfrían juntas; cada movimiento
    cambia una variable con paso propio adaptado (~44 % de aceptación) y las
    cadenas vecinas intercambian estados con el criterio de Metropolis."""
    rec = _Recorder(p, "Recocido Simulado")
    d = len(p.vars)
    span = p.hi - p.lo
    K = int(min(max(4, pop // 4), 12))
    ar = np.arange(K)
    X = _init_pop(p, K, rng)
    o, v = _eval(p, X, rec)
    E = _energy(o, v)
    fin = E[np.isfinite(E)]
    T0 = float(np.std(fin)) if fin.size > 1 and np.std(fin) > 0 else 1.0
    ladder = T0 * np.geomspace(1.0, 1e-2, K)
    step = np.tile(span * 0.25, (K, 1))
    moves = max(2, pop)
    rec.log(0)
    for it in range(1, iters + 1):
        T = ladder * (1e-3 ** (it / iters))
        acc = np.zeros(K)
        for _ in range(moves):
            j = rng.integers(0, d, K)
            Y = X.copy()
            Y[ar, j] += _disc_delta(p, j, (rng.random(K) * 2 - 1) * step[ar, j])
            Y = p.repair(Y)
            oy, vy = _eval(p, Y, rec)
            ey = _energy(oy, vy)
            with np.errstate(over="ignore", invalid="ignore"):
                ok = (ey <= E) | (rng.random(K) < np.exp(-(ey - E) / np.maximum(T, 1e-300)))
            X[ok], E[ok] = Y[ok], ey[ok]
            acc += ok
        rate = acc / moves
        step = np.clip(step * np.where(rate > 0.44, 1.3, 0.75)[:, None], span * 1e-8, span)
        for k in range(it % 2, K - 1, 2):                 # intercambio de réplicas
            dlt = (1 / T[k] - 1 / T[k + 1]) * (E[k] - E[k + 1])
            if np.isfinite(dlt) and (dlt >= 0 or rng.random() < math.exp(dlt)):
                X[[k, k + 1]] = X[[k + 1, k]]
                E[[k, k + 1]] = E[[k + 1, k]]
        rec.log(it)
    return rec


def tabu_search(p, iters, pop, rng):
    """Búsqueda Tabú por coordenadas: vecinos que mueven una variable ±paso,
    atributo tabú (variable, dirección), aspiración, intensificación (paso más
    fino) y diversificación (reinicio) ante estancamiento."""
    rec = _Recorder(p, "Búsqueda Tabú")
    d = len(p.vars)
    span = p.hi - p.lo
    x = _init_pop(p, 1, rng)[0]
    _eval(p, x[None, :], rec)
    step = span * 0.2
    tenure = max(3, int(round(math.sqrt(2 * d))))
    tabu: Dict[Tuple[int, int], int] = {}
    stall, best_seen = 0, rec.best_o + 1e6 * rec.best_v
    rec.log(0)
    for it in range(1, iters + 1):
        m = max(2, pop)
        js = rng.integers(0, d, m)
        sg = rng.choice((-1, 1), m)
        N = np.repeat(x[None, :], m, axis=0)
        N[np.arange(m), js] += _disc_delta(p, js, sg * step[js] * (0.5 + rng.random(m)))
        N = p.repair(N)
        o, v = _eval(p, N, rec)
        en = _energy(o, v)
        best_e = rec.best_o + 1e6 * rec.best_v
        allowed = np.array([tabu.get((int(j), int(-s_)), 0) < it or en[k] < best_e
                            for k, (j, s_) in enumerate(zip(js, sg))])
        cand = np.where(allowed)[0] if allowed.any() else np.arange(m)
        k = int(cand[np.argmin(en[cand])])
        x = N[k]
        tabu[(int(js[k]), int(sg[k]))] = it + tenure      # prohíbe deshacer el movimiento
        cur = rec.best_o + 1e6 * rec.best_v
        if cur < best_seen - 1e-12:
            best_seen, stall = cur, 0
        else:
            stall += 1
        if stall and stall % max(8, iters // 30) == 0:
            step = np.maximum(step * 0.5, span * 1e-7)       # intensificación
        if stall > max(20, iters // 8):                      # diversificación
            x = p.repair((rec.best_x + (rng.random(d) * 2 - 1) * span * 0.15)[None, :])[0]
            step = span * 0.2; tabu.clear(); stall = 0
        rec.log(it)
    return rec


def polish(p: Problem, x0: np.ndarray, o0: float, v0: float):
    """Pulido local (L-BFGS-B sin restricciones / Powell con ellas) sobre las
    variables reales; se acepta solo si mejora según las reglas de Deb."""
    real = np.array([v.kind == "real" for v in p.vars])
    if not real.any():
        return x0, o0, v0
    try:
        from scipy.optimize import minimize
    except ImportError:
        return x0, o0, v0
    base = x0.copy()

    def full(z):
        y = base.copy(); y[real] = z
        return p.repair(y[None, :])

    def energy(z):
        _, o, v = p.evaluate(full(z))
        val = float(_energy(o, v)[0])
        return val if math.isfinite(val) else 1e300
    bnds = list(zip(p.lo[real], p.hi[real]))
    method = "L-BFGS-B" if not p.g else "Powell"
    try:
        r = minimize(energy, x0[real], method=method, bounds=bnds,
                     options={"maxiter": 2000})
    except Exception:
        return x0, o0, v0
    y = full(r.x)
    _, o1, v1 = p.evaluate(y)
    if _better(o1[0], v1[0], o0, v0):
        return y[0], float(o1[0]), float(v1[0])
    return x0, o0, v0


_RUNNERS = {"Evolución Diferencial": differential_evolution, "CMA-ES": cma_es,
            "PSO: Enjambre de Partículas": particle_swarm,
            "Algoritmo Genético": genetic, "Recocido Simulado": annealing,
            "Búsqueda Tabú": tabu_search}


def _fmt_x(p: Problem, x) -> str:
    return ", ".join(f"{n} = {int(round(val))}" if v.kind != "real" else f"{n} = {val:.6g}"
                     for n, v, val in zip(p.names, p.vars, x))


def solve_general(sense: str, objective: str, variables: str, constraints: str, algorithm: str,
                  iterations: int, population: int, seed: int, runs: int = 1,
                  local_polish: bool = True) -> Dict[str, Any]:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo desconocido: «{algorithm}».")
    if not 10 <= iterations <= 20000:
        raise ValueError("Las iteraciones deben estar entre 10 y 20 000.")
    if not 4 <= population <= 2000:
        raise ValueError("La población debe estar entre 4 y 2 000.")
    p = parse_problem(objective, sense, variables, constraints)
    if len(p.vars) > 200:
        raise ValueError("El máximo admitido es 200 variables.")
    if not 1 <= runs <= 100:
        raise ValueError("Las ejecuciones independientes deben estar entre 1 y 100.")
    names = list(_RUNNERS) if algorithm == "Comparar todos" else [algorithm]
    recs = {}
    stats: Dict[str, Dict[str, float]] = {}
    for name in names:
        outs = []
        for r_ in range(runs):                     # semillas seed, seed+1, …
            rec = _RUNNERS[name](p, iterations, population, np.random.default_rng(seed + r_))
            if local_polish:
                x1, o1, v1 = polish(p, rec.best_x, rec.best_o, rec.best_v)
                if (x1 is not rec.best_x) and (o1, v1) != (rec.best_o, rec.best_v):
                    rec.best_x, rec.best_o, rec.best_v = x1, o1, v1
                    rec.log(rec.rows[-1][0] + 1)     # fila «pulido local»
                    rec.polished = True
            outs.append(rec)
        best_run = min(outs, key=lambda r: (r.best_v > 0, r.best_v if r.best_v > 0 else r.best_o))
        recs[name] = best_run
        vals = np.array([p.sense * r.best_o for r in outs if r.best_v <= 0])
        stats[name] = {"runs": runs, "factibles": int(vals.size),
                       "mejor": float(p.sense * best_run.best_o),
                       "media": float(vals.mean()) if vals.size else float("nan"),
                       "desv": float(vals.std(ddof=1)) if vals.size > 1 else 0.0}
    best_name = min(names, key=lambda n: (recs[n].best_v > 0, recs[n].best_v
                                          if recs[n].best_v > 0 else recs[n].best_o))
    best = recs[best_name]
    f_best = p.sense * best.best_o
    feasible = best.best_v <= 0

    # Tabla: progreso del mejor (máx. ~120 filas por algoritmo) + resumen
    rows: List[Dict[str, Any]] = []
    for name in names:
        r = recs[name]
        step = max(1, len(r.rows) // 120)
        for k, (it, x, o, v) in enumerate(r.rows):
            if k % step and k != len(r.rows) - 1:
                continue
            row = {"Iteración": it}
            if len(names) > 1:
                row["Algoritmo"] = name
            row["Mejor f"] = p.sense * o
            row["Violación"] = v
            row["Factible"] = "Sí" if v <= 0 else "No"
            row["Mejor x"] = _fmt_x(p, x)
            rows.append(row)
    if len(names) > 1 or runs > 1:
        for name in names:
            r = recs[name]
            row = {"Iteración": "final"}
            if len(names) > 1:
                row["Algoritmo"] = name
            row.update({"Mejor f": p.sense * r.best_o, "Violación": r.best_v,
                        "Factible": "Sí" if r.best_v <= 0 else "No",
                        "Mejor x": _fmt_x(p, r.best_x)})
            if runs > 1:
                st = stats[name]
                row["Ejecuciones"] = (f"{st['factibles']}/{runs} factibles · media "
                                      f"{st['media']:.6g} · desv. {st['desv']:.3g}")
            rows.append(row)

    curves = {n: (np.array([r[0] for r in recs[n].rows]),
                  np.array([p.sense * r[2] for r in recs[n].rows]),
                  np.array([r[3] for r in recs[n].rows])) for n in names}
    path = np.array([r[1] for r in best.rows])

    def plot(fig):
        if len(p.vars) == 2:
            ax1, ax2 = fig.subplots(1, 2)
        else:
            ax2 = fig.add_subplot(111); ax1 = None
        for n in names:
            it, fv, vv = curves[n]
            ok = vv <= 0
            fv_plot = np.where(ok, fv, np.nan)
            ax2.plot(it, fv_plot, color=_COLORS[n], lw=1.6, label=n)
        ax2.set_xlabel("Iteración"); ax2.set_ylabel("Mejor f factible")
        ax2.set_title("Convergencia" + (" (comparación)" if len(names) > 1 else ""))
        ax2.legend(fontsize=7)
        if ax1 is not None:
            g1 = np.linspace(p.vars[0].lo, p.vars[0].hi, 120)
            g2 = np.linspace(p.vars[1].lo, p.vars[1].hi, 120)
            G1, G2 = np.meshgrid(g1, g2)
            pts = np.column_stack([G1.ravel(), G2.ravel()])
            Z = p.f(pts).reshape(G1.shape)
            Zc = np.where(np.isfinite(Z), Z, np.nan)
            ax1.contourf(G1, G2, Zc, levels=30, cmap="coolwarm", alpha=0.85)
            if p.g:
                _, _, vg = p.evaluate(pts)
                ax1.contourf(G1, G2, (vg > 0).reshape(G1.shape).astype(float), levels=[0.5, 1.5],
                             colors=["#000000"], alpha=0.45)
            ax1.plot(path[:, 0], path[:, 1], "-", color="#ff6b35", lw=1.2, alpha=0.8,
                     label="Mejor por iteración")
            ax1.plot(best.best_x[0], best.best_x[1], "D", color="#ffd700", ms=9,
                     label=f"x* ({'factible' if feasible else 'infactible'})")
            ax1.set_xlabel(p.names[0]); ax1.set_ylabel(p.names[1])
            ax1.set_title("f y región infactible (sombra)" if p.g else "Curvas de nivel de f")
            ax1.legend(fontsize=7)

    def plot3d(fig):
        ax = fig.add_subplot(111, projection="3d")
        if len(p.vars) == 2:
            g1 = np.linspace(p.vars[0].lo, p.vars[0].hi, 60)
            g2 = np.linspace(p.vars[1].lo, p.vars[1].hi, 60)
            G1, G2 = np.meshgrid(g1, g2)
            Z = p.f(np.column_stack([G1.ravel(), G2.ravel()])).reshape(G1.shape)
            ax.plot_surface(G1, G2, np.where(np.isfinite(Z), Z, np.nan), cmap="coolwarm",
                            alpha=0.6, linewidth=0)
            pz = p.f(path)
            ax.plot(path[:, 0], path[:, 1], pz, "o-", color="#ff6b35", ms=3, label="Mejor por iteración")
            ax.scatter([best.best_x[0]], [best.best_x[1]], [f_best], color="#ffd700", s=80,
                       marker="D", label=f"x*, f* = {f_best:.6g}")
            ax.set_xlabel(p.names[0]); ax.set_ylabel(p.names[1]); ax.set_zlabel("f")
        elif len(p.vars) == 1:
            it = np.arange(len(path))
            ax.plot(it, path[:, 0], p.f(path), "o-", color="#ff6b35", ms=3, label="Mejor por iteración")
            ax.set_xlabel("Iteración"); ax.set_ylabel(p.names[0]); ax.set_zlabel("f")
        else:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], "o-", color="#ff6b35", ms=3,
                    label="Mejor por iteración")
            ax.scatter(*best.best_x[:3], color="#ffd700", s=80, marker="D", label="x*")
            ax.set_xlabel(p.names[0]); ax.set_ylabel(p.names[1]); ax.set_zlabel(p.names[2])
        ax.set_title(f"3D — {best_name}", fontsize=10, color="#d8e8f8")
        ax.legend(fontsize=7, loc="upper left")

    if len(p.vars) == 2:
        note = (f"Ejes: {p.names[0]}, {p.names[1]} y f; la trayectoria es el mejor punto de "
                f"cada iteración de {best_name}.")
    elif len(p.vars) == 1:
        note = f"Ejes: iteración, {p.names[0]} y f del mejor punto de cada iteración de {best_name}."
    else:
        note = (f"Ejes: {p.names[0]}, {p.names[1]}, {p.names[2]} (primeras 3 de {len(p.vars)} "
                f"variables): recorrido real del mejor punto de {best_name}.")
    sense_txt = "máx" if p.sense < 0 else "mín"
    resumen = (f"{best_name}: f {sense_txt} = {f_best:.6g} en {_fmt_x(p, best.best_x)}"
               + ("" if feasible else f" — INFACTIBLE (violación {best.best_v:.3g})"))
    return {"x": best.best_x.tolist(), "f": f_best, "feasible": feasible,
            "violation": best.best_v, "algorithm": best_name, "names": p.names,
            "results": {n: {"f": p.sense * recs[n].best_o, "violation": recs[n].best_v,
                            "x": recs[n].best_x.tolist(), "evals": recs[n].evals,
                            **stats[n]} for n in names},
            "hist": rows, "resumen": resumen, "plot": plot, "plot3d": plot3d, "note3d": note}


# ── Viajante (TSP) de gran tamaño ────────────────────────────────────────────
def _tour_len(D, t) -> float:
    return float(D[t, np.roll(t, -1)].sum())


def _two_opt_delta(D, t, i, k):
    a, b, c, d = t[i - 1], t[i], t[k], t[(k + 1) % len(t)]
    return D[a, c] + D[b, d] - D[a, b] - D[c, d]


def two_opt(D, t, max_rounds=None):
    """Mejora local 2-opt (mejor mejora por ronda), vectorizada con NumPy."""
    t = np.array(t)
    n = len(t)
    if n < 4:
        return t
    I, K = np.triu_indices(n, 1)
    keep = (I >= 1) & (K - I >= 1) & ~((I == 1) & (K == n - 1))
    I, K = I[keep], K[keep]
    for _ in range(max_rounds or 4 * n):
        a, b, c, d = t[I - 1], t[I], t[K], t[(K + 1) % n]
        delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
        m = int(delta.argmin())
        if delta[m] >= -1e-12:
            break
        t[I[m]:K[m] + 1] = t[I[m]:K[m] + 1][::-1]
    return t


def tsp_aco(D, iters, ants, rng, alpha=1.0, beta=3.0, rho=0.3):
    """Sistema de hormigas Max-Min simplificado; rutas construidas en paralelo."""
    n = len(D)
    eta = (1.0 / (D + np.eye(n))) ** beta
    tau = np.ones((n, n))
    best_t, best_l, rows = None, np.inf, []
    ar = np.arange(ants)
    for it in range(1, iters + 1):
        T = np.empty((ants, n), int)
        T[:, 0] = rng.integers(0, n, ants)
        free = np.ones((ants, n), bool); free[ar, T[:, 0]] = False
        attr = (tau ** alpha) * eta
        for s_ in range(1, n):
            w = attr[T[:, s_ - 1]] * free
            w /= w.sum(1, keepdims=True)
            nxt = (w.cumsum(1) < rng.random((ants, 1))).sum(1).clip(0, n - 1)
            nxt = np.where(free[ar, nxt], nxt, free.argmax(1))   # protección numérica
            T[:, s_] = nxt; free[ar, nxt] = False
        lens = D[T, np.roll(T, -1, axis=1)].sum(1)
        k = int(lens.argmin())
        cand = two_opt(D, T[k], max_rounds=n)                   # búsqueda local de la mejor
        cl = _tour_len(D, cand)
        if cl < best_l:
            best_l, best_t = cl, cand
        tau *= (1 - rho)
        tau[best_t, np.roll(best_t, -1)] += 1.0 / best_l
        tau[np.roll(best_t, -1), best_t] += 1.0 / best_l
        tau = np.clip(tau, 1e-3, 1e3)
        rows.append((it, best_l, float(lens.mean())))
    return best_t, best_l, rows


def tsp_annealing(D, iters, moves, rng):
    n = len(D)
    moves = max(moves, 4 * n)                 # movimientos por temperatura según el tamaño
    t = rng.permutation(n); L = _tour_len(D, t)
    best_t, best_l, rows = t.copy(), L, []
    T0 = D.mean() * 0.5
    for it in range(1, iters + 1):
        T = T0 * (0.0005 ** (it / iters))
        pairs = np.sort(rng.integers(1, n, (moves, 2)), axis=1)
        u = rng.random(moves)
        for (i, k), r in zip(pairs, u):
            if i == k:
                continue
            dl = _two_opt_delta(D, t, i, k)
            if dl < 0 or r < math.exp(-dl / max(T, 1e-12)):
                t[i:k + 1] = t[i:k + 1][::-1]; L += dl
                if L < best_l - 1e-12:
                    best_l, best_t = L, t.copy()
        rows.append((it, best_l, L))
    best_t = two_opt(D, best_t)               # pulido final
    best_l = _tour_len(D, best_t)
    rows[-1] = (rows[-1][0], best_l, rows[-1][2])
    return best_t, best_l, rows


def tsp_genetic(D, iters, pop, rng):
    """GA memético: cruce OX, mutación por inversión y 2-opt sobre el mejor hijo."""
    n = len(D)
    P = [two_opt(D, rng.permutation(n), max_rounds=n) if k == 0 else rng.permutation(n)
         for k in range(pop)]
    L = np.array([_tour_len(D, t) for t in P])
    rows = []

    def ox(a, b):
        i, k = sorted(rng.choice(n, 2, replace=False))
        child = -np.ones(n, int); child[i:k + 1] = a[i:k + 1]
        seg = set(a[i:k + 1].tolist())
        child[child < 0] = [g for g in b if g not in seg]
        return child
    for it in range(1, iters + 1):
        order = np.argsort(L)
        new = [P[order[0]].copy(), P[order[1]].copy()]
        while len(new) < pop:
            a = P[min(rng.integers(0, pop, 3), key=lambda j: L[j])]
            b = P[min(rng.integers(0, pop, 3), key=lambda j: L[j])]
            c = ox(a, b)
            if rng.random() < 0.3:
                i, k = sorted(rng.choice(np.arange(1, n), 2, replace=False))
                c[i:k + 1] = c[i:k + 1][::-1]
            new.append(c)
        L = np.array([_tour_len(D, t) for t in new])
        kbest = int(np.argsort(L)[min(2, pop - 1)])               # mejor hijo nuevo
        new[kbest] = two_opt(D, new[kbest], max_rounds=n)
        L[kbest] = _tour_len(D, new[kbest])
        P = new
        rows.append((it, float(L.min()), float(L.mean())))
    i = int(L.argmin())
    return P[i], float(L[i]), rows


_TSP = {"Colonia de Hormigas (ACO)": tsp_aco, "Recocido Simulado (2-opt)": tsp_annealing,
        "Algoritmo Genético (cruce OX)": tsp_genetic}


def solve_tsp(coords: Sequence[Sequence[float]], algorithm: str, iterations: int,
              agents: int, seed: int) -> Dict[str, Any]:
    P = np.asarray(coords, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2 or len(P) < 4:
        raise ValueError("Ingresa al menos 4 ciudades, una por línea: x, y "
                         "(o genera ciudades aleatorias).")
    if len(P) > 300:
        raise ValueError("El máximo admitido es 300 ciudades.")
    if algorithm not in TSP_ALGORITHMS:
        raise ValueError(f"Algoritmo desconocido: «{algorithm}».")
    if not 5 <= iterations <= 5000 or not 2 <= agents <= 500:
        raise ValueError("Usa entre 5 y 5 000 iteraciones y entre 2 y 500 agentes.")
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    names = list(_TSP) if algorithm == "Comparar todos" else [algorithm]
    res = {n: _TSP[n](D, iterations, agents, np.random.default_rng(seed)) for n in names}
    best_name = min(names, key=lambda n: res[n][1])
    tour, length, _ = res[best_name]
    rows: List[Dict[str, Any]] = []
    for n in names:
        r = res[n][2]
        step = max(1, len(r) // 100)
        for k, (it, bl, cur) in enumerate(r):
            if k % step == 0 or k == len(r) - 1:
                row = {"Iteración": it}
                if len(names) > 1:
                    row["Algoritmo"] = n
                row["Mejor longitud"] = bl
                row["Longitud de la iteración"] = cur
                rows.append(row)
    route = " → ".join(str(int(c) + 1) for c in list(tour) + [tour[0]])
    rows.append({"Iteración": "final", **({"Algoritmo": best_name} if len(names) > 1 else {}),
                 "Mejor longitud": length, "Longitud de la iteración": f"Ruta: {route}"})

    def plot(fig):
        ax1, ax2 = fig.subplots(1, 2)
        t = list(tour) + [tour[0]]
        ax1.plot(P[t, 0], P[t, 1], "-", color="#ffd700", lw=1.4)
        ax1.plot(P[:, 0], P[:, 1], "o", color="#4a9eff", ms=5, label="Ciudades")
        ax1.plot(P[tour[0], 0], P[tour[0], 1], "s", color="#3cff8a", ms=8, label="Inicio")
        ax1.set_title(f"Mejor ruta ({best_name}): {length:.4g}")
        ax1.legend(fontsize=7)
        for n in names:
            r = np.array(res[n][2])
            ax2.plot(r[:, 0], r[:, 1], lw=1.6, label=n)
        ax2.set_xlabel("Iteración"); ax2.set_ylabel("Mejor longitud")
        ax2.set_title("Convergencia"); ax2.legend(fontsize=7)

    def plot3d(fig):
        ax = fig.add_subplot(111, projection="3d")
        t = list(tour) + [tour[0]]
        order = np.arange(len(t))
        ax.plot(P[t, 0], P[t, 1], order, "o-", color="#ffd700", ms=3, label="Ruta (z = orden de visita)")
        ax.scatter(P[:, 0], P[:, 1], np.zeros(len(P)), color="#4a9eff", s=12, label="Ciudades")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("orden de visita")
        ax.set_title(f"3D — Recorrido del viajante ({best_name})", fontsize=10, color="#d8e8f8")
        ax.legend(fontsize=7, loc="upper left")

    return {"tour": [int(c) for c in tour], "length": length, "algorithm": best_name,
            "results": {n: res[n][1] for n in names}, "hist": rows,
            "resumen": f"{best_name}: longitud {length:.6g} con {len(P)} ciudades",
            "plot": plot, "plot3d": plot3d,
            "note3d": "Ejes: coordenadas x, y de cada ciudad y el orden en que la visita la "
                      "mejor ruta encontrada."}


def random_cities(n: int, seed: int) -> List[List[float]]:
    if not 4 <= n <= 300:
        raise ValueError("Genera entre 4 y 300 ciudades.")
    rng = np.random.default_rng(seed)
    return np.round(rng.random((n, 2)) * 100, 2).tolist()
