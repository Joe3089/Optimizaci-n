# -*- coding: utf-8 -*-
"""
result_analysis.py — Detector de eventos y analizador contextual.

Capa pura (sin Qt, sin red): recibe lo que produjo el MOTOR MATEMÁTICO
(historial, excepción o un punto de evaluación) y lo convierte en un
`OptimizationEvent` estructurado. El motor es la única fuente de verdad: aquí
nunca se recalcula un óptimo, solo se leen los valores que el motor devolvió
y se diagnostica *por qué* algo falló (raíz de negativo, división entre cero,
logaritmo de no positivo, iteraciones agotadas, sintaxis…).

Además genera una explicación didáctica local (`local_explanation`) que sirve
de respaldo cuando la IA no está disponible, y el prompt que se le envía a la
IA (`build_ai_prompt`), que le pide interpretar SOLO estos datos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

# ─── Tipos de evento ──────────────────────────────────────────────────────────
SUCCESS = "success"                 # Resultado válido y convergente
WARNING = "invalid_result"          # Resultado numérico inválido (inf / NaN / valor enorme)
INVALID_RESULT = WARNING
DOMAIN_ERROR = "domain_error"       # Función no definida en los reales en ese punto
DIVISION_BY_ZERO = "division_by_zero"
CONVERGENCE = "convergence"         # Terminó sin cumplir el criterio de convergencia
MAX_ITERATIONS = "max_iterations"   # Se agotó el máximo de iteraciones sin la precisión pedida
PARAMETER_ERROR = "parameter_error" # Datos/parámetros de entrada incorrectos
AI_COMM_ERROR = "ai_comm_error"     # La IA no respondió (no afecta al resultado matemático)
SYNTAX_ERROR = "syntax_error"       # Expresión mal escrita
EXECUTION_ERROR = "execution_error" # Cualquier otro fallo del motor
INFINITE_RESULT = "infinite_result" # El resultado es +∞ / −∞ (polo, función no acotada)
NAN_RESULT = "nan_result"           # El resultado es NaN (operación indeterminada)
OVERFLOW = "overflow"               # Magnitud fuera del rango de coma flotante

ERROR_KINDS = {DOMAIN_ERROR, DIVISION_BY_ZERO, SYNTAX_ERROR, EXECUTION_ERROR, PARAMETER_ERROR}
NON_CONVERGED = {CONVERGENCE, MAX_ITERATIONS}
INVALID_KINDS = {WARNING, INFINITE_RESULT, NAN_RESULT, OVERFLOW}
# Eventos que impiden mostrar/interpretar el resultado → Opti abre el chat.
# (Los parámetros mal escritos ya tienen su diálogo; un éxito no lo necesita.)
# Casos en que Opti interviene por su cuenta (entrada o resultado incorrecto).
# Un resultado correcto (SUCCESS) NO dispara a la IA: solo responde si el
# usuario le escribe o le habla.
CHAT_KINDS = INVALID_KINDS | NON_CONVERGED | {DOMAIN_ERROR, DIVISION_BY_ZERO,
                                              SYNTAX_ERROR, EXECUTION_ERROR,
                                              PARAMETER_ERROR, WARNING}
FLOAT_MAX = 1.7976931348623157e308

TITLES = {
    SUCCESS: "Resultado satisfactorio",
    WARNING: "Resultado numérico inválido",
    DOMAIN_ERROR: "Error de dominio matemático",
    DIVISION_BY_ZERO: "Error: división entre cero",
    CONVERGENCE: "Problema de convergencia",
    MAX_ITERATIONS: "Máximo de iteraciones alcanzado",
    PARAMETER_ERROR: "Parámetros incorrectos",
    AI_COMM_ERROR: "Error de comunicación con la IA",
    SYNTAX_ERROR: "Error de sintaxis en la función",
    EXECUTION_ERROR: "Error de ejecución",
    INFINITE_RESULT: "Resultado infinito",
    NAN_RESULT: "Resultado NaN (no es un número)",
    OVERFLOW: "Desbordamiento numérico",
}

# Métodos deterministas que arrancan en x₀ y se detienen cuando ‖∇f‖ ≈ 0.
GRADIENT_METHODS = {"Newton-Raphson", "Grad. Conjugado", "Armijo", "Wolfe", "Goldstein"}
INTERVAL_METHODS = {"Bisección", "Sección Dorada", "Sección Áurea", "Fibonacci", "Búsqueda Local"}
META_METHODS = {"SA: Recocido Simulado", "PSO: Enjambre", "GA: Algoritmo Genético"}
# Máximo de iteraciones con que optimization_service invoca cada método.
MAX_ITER = {"Newton-Raphson": 60, "Grad. Conjugado": 150, "Armijo": 50,
            "Wolfe": 100, "Goldstein": 60, "Sección Áurea": 200,
            "SA: Recocido Simulado": 600, "PSO: Enjambre": 150}
# Tolerancia con que optimization_service invoca cada método (Fibonacci usa la L de la UI).
TOLERANCE = {"Newton-Raphson": 1e-7, "Grad. Conjugado": 1e-7, "Armijo": 1e-6,
             "Wolfe": 1e-6, "Goldstein": 1e-7, "Sección Áurea": 1e-6,
             "Bisección": 1e-8, "Sección Dorada": 1e-8}
GRAD_TOL = 1e-3


@dataclass
class OptimizationEvent:
    kind: str
    metodo: str = ""
    fx: str = ""
    vars_: List[str] = field(default_factory=list)
    x0: List[float] = field(default_factory=list)
    lo: Optional[float] = None
    hi: Optional[float] = None
    iterations: int = 0
    x_opt: Optional[str] = None
    f_opt: Optional[float] = None
    grad_norm: Optional[float] = None
    converged: Optional[bool] = None
    result_summary: str = ""
    error_message: str = ""
    cause: str = ""
    recommendation: str = ""
    technical_detail: str = ""
    point: Optional[List[float]] = None   # punto donde se detectó el fallo
    tolerance: Optional[float] = None
    max_iter: Optional[int] = None
    params: dict = field(default_factory=dict)
    duration_s: Optional[float] = None
    exception_type: str = ""

    @property
    def title(self) -> str:
        return TITLES.get(self.kind, "Resultado")

    @property
    def is_error(self) -> bool:
        return self.kind in ERROR_KINDS

    @property
    def convergence_label(self) -> str:
        if self.kind == SUCCESS:
            return "Convergió" if self.converged is not False else "Resultado obtenido"
        if self.kind == MAX_ITERATIONS:
            return "No convergió (se agotó el máximo de iteraciones)"
        if self.kind == CONVERGENCE:
            return "No convergió (criterio de parada no cumplido)"
        if self.kind in INVALID_KINDS:
            return "Resultado numérico no válido"
        return "No aplica (el cálculo no se completó)"

    def facts(self) -> List[tuple]:
        """Pares (etiqueta, valor) para el recuadro visual y el prompt."""
        rows = [("Función", f"f = {self.fx}" if self.fx else "—"),
                ("Método", self.metodo or "—")]
        if self.kind != SUCCESS:
            rows.append(("Tipo de evento", self.title))
        if self.x0 and self.metodo in GRADIENT_METHODS:
            rows.append(("Punto inicial", _fmt_vec(self.x0)))
        if self.lo is not None and self.hi is not None and self.metodo and self.fx:
            rows.append(("Rango", f"[{_fmt_num(self.lo)}, {_fmt_num(self.hi)}]"))
        if self.tolerance is not None:
            rows.append(("Tolerancia", f"{self.tolerance:g}"))
        if self.max_iter:
            rows.append(("Máx. iteraciones", str(self.max_iter)))
        for k, v in (self.params or {}).items():
            rows.append((k, str(v)))
        if self.point is not None and self.kind != SUCCESS:
            rows.append(("Punto problemático", _point_str(self.vars_ or ["x"], self.point)))
        if self.kind == SUCCESS or self.kind in NON_CONVERGED:
            rows.append(("Resultado", self.result_summary or "—"))
            if self.x_opt is not None:
                lbl = "Punto óptimo x*" if self.kind == SUCCESS else "Último punto"
                rows.append((lbl, self.x_opt))
            if self.f_opt is not None:
                lbl = "Valor óptimo f*" if self.kind == SUCCESS else "Último valor f"
                rows.append((lbl, _fmt_num(self.f_opt)))
        if self.iterations:
            rows.append(("Iteraciones", str(self.iterations)))
        if self.grad_norm is not None:
            rows.append(("‖∇f‖ final", _fmt_num(self.grad_norm)))
        rows.append(("Convergencia", self.convergence_label))
        if self.duration_s is not None:
            rows.append(("Tiempo de ejecución", f"{self.duration_s * 1000:.1f} ms"))
        if self.error_message:
            rows.append(("Mensaje", self.error_message))
        if self.cause:
            rows.append(("Posible causa", self.cause))
        return rows


# ─── Utilidades de formato ────────────────────────────────────────────────────
def _fmt_num(v) -> str:
    try:
        f = float(v)
    except Exception:
        return str(v)
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "∞" if f > 0 else "-∞"
    if abs(f - round(f)) < 1e-6 and abs(f) < 1e12:
        return str(int(round(f)))
    return f"{f:.6g}"


def _fmt_vec(v) -> str:
    try:
        vals = [float(x) for x in v]
    except Exception:
        return str(v)
    if len(vals) == 1:
        return _fmt_num(vals[0])
    return "(" + ", ".join(_fmt_num(x) for x in vals) + ")"


def _point_str(names: Sequence[str], point: Sequence[float]) -> str:
    return ", ".join(f"{n} = {_fmt_num(p)}" for n, p in zip(names, point))


# ─── Parseo simbólico (reutiliza func_compat, la misma normalización que el motor)
def _fc():
    try:
        from app.application import func_compat
        return func_compat
    except Exception:
        return None


def parse_expression(fx: str):
    """Devuelve (expr_sympy, [nombres de variables]). Lanza la excepción de
    sympy si la expresión es sintácticamente inválida."""
    import sympy as sp
    fc = _fc()
    norm = fc.normalize_expr(fx) if fc else fx
    names = (fc.detect_variables(norm) if fc else []) or ["x"]
    syms = {n: sp.Symbol(n) for n in names}
    expr = sp.sympify(norm, locals=syms)
    if not isinstance(expr, sp.Basic) or isinstance(expr, (sp.logic.boolalg.Boolean,)):
        raise ValueError("La expresión no es una función numérica.")
    return expr, names


def syntax_problem(fx: str) -> Optional[str]:
    """None si la expresión se puede interpretar; si no, descripción breve."""
    if not fx or not fx.strip():
        return "La función está vacía."
    try:
        parse_expression(fx)
        return None
    except Exception as ex:
        return f"{type(ex).__name__}: {ex}"


def effective_start_point(vars_: Sequence[str], x0: Sequence[float], lo, hi) -> List[float]:
    """Replica `_get_x0_for` de optimization_service: el punto real desde el
    que arranca el método."""
    mid = ((lo or 0.0) + (hi or 0.0)) / 2.0
    n = len(vars_)
    if x0 and len(x0) == n:
        return [float(v) for v in x0]
    if n == 1:
        return [mid]
    if x0:
        base = [float(v) for v in x0]
        while len(base) < n:
            base.append(mid)
        return base[:n]
    return [mid] * n


def check_point(fx: str, point: Sequence[float], var_names: Sequence[str] = None):
    """Analiza si f está definida (en los reales) en `point`.
    Devuelve None si está bien, o (kind, cause, recommendation)."""
    try:
        import sympy as sp
        expr, names = parse_expression(fx)
    except Exception:
        return None
    names = list(var_names or names)
    subs = {sp.Symbol(n): float(p) for n, p in zip(names, point)}
    if len(subs) < len(expr.free_symbols):
        return None
    pstr = _point_str(names, point)

    def num(e):
        try:
            v = complex(e.subs(subs).evalf())
            return v.real if abs(v.imag) < 1e-12 else None
        except Exception:
            return None

    found = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow):
            base, ex = node.args
            bv = num(base)
            if bv is None or not ex.is_number:
                continue
            exv = float(ex)
            if exv < 0 and abs(bv) < 1e-12:
                z = _zero_str(base)
                found.append((0, DIVISION_BY_ZERO,
                    f"En {pstr} el denominador {sp.sstr(base)} vale 0, y la división "
                    f"entre cero no está definida (la función tiene una asíntota vertical ahí).",
                    (f"Evita {z}: " if z else "") +
                    f"cambia el punto inicial o el intervalo para que no incluya el punto "
                    f"donde {sp.sstr(base)} = 0; por ejemplo, empieza a un lado de la discontinuidad."))
            elif bv < 0 and abs(exv - round(exv)) > 1e-12:
                root = "raíz cuadrada" if abs(exv - 0.5) < 1e-12 else "potencia fraccionaria"
                bound = _solve_bound(base, names)
                sug = _suggest_point(base)
                hint = (f"un punto inicial con {bound}" if bound
                        else f"un punto donde {sp.sstr(base)} ≥ 0")
                if sug:
                    hint += f" (por ejemplo, prueba {sug})"
                found.append((1, DOMAIN_ERROR,
                    f"En {pstr} el argumento de la {root}, {sp.sstr(base)}, vale "
                    f"{_fmt_num(bv)}: es negativo y en los números reales no existe esa {root}.",
                    f"Usa {hint}, o define un intervalo que quede dentro del dominio de la función."))
        elif isinstance(node, sp.log):
            av = num(node.args[0])
            if av is not None and av <= 0:
                found.append((1, DOMAIN_ERROR,
                    f"En {pstr} el argumento del logaritmo, {sp.sstr(node.args[0])}, vale "
                    f"{_fmt_num(av)}; el logaritmo solo está definido para valores positivos.",
                    f"Elige un punto inicial y un intervalo donde {sp.sstr(node.args[0])} > 0"
                    + (f" (por ejemplo, prueba {_suggest_point(node.args[0])})."
                       if _suggest_point(node.args[0]) else ".")))
        elif isinstance(node, (sp.asin, sp.acos)):
            av = num(node.args[0])
            if av is not None and abs(av) > 1:
                found.append((1, DOMAIN_ERROR,
                    f"En {pstr} el argumento de {type(node).__name__}, {sp.sstr(node.args[0])}, "
                    f"vale {_fmt_num(av)}; solo está definido entre -1 y 1.",
                    "Ajusta el punto inicial o el intervalo para que el argumento quede en [-1, 1]."))
    if found:
        found.sort(key=lambda t: t[0])
        return found[0][1:]
    # Comprobación numérica final (casos no cubiertos: tan(π/2), etc.)
    v = None
    exact = None
    try:
        exact = expr.subs(subs).evalf()
        if exact.is_real and exact.is_finite and abs(exact) > FLOAT_MAX:
            return (OVERFLOW,
                    f"En {pstr} el valor exacto de la función es aproximadamente "
                    f"{sp.N(exact, 4)}, que supera el mayor número representable en coma "
                    f"flotante (≈ 1.8·10^308): la magnitud del cálculo se desborda.",
                    "Usa valores de entrada más pequeños (un punto inicial o un rango más "
                    "cercano a 0), reescala la función (por ejemplo, minimiza log(f) o "
                    "divide por una constante) o usa una formulación numéricamente más estable.")
        if exact in (sp.oo, -sp.oo, sp.zoo):
            sign = {sp.oo: "+∞", -sp.oo: "−∞"}.get(exact, "∞ (sin signo)")
            return (INFINITE_RESULT,
                    f"En {pstr} la función vale {sign}: no es un número finito, así que no "
                    f"puede tomarse como resultado ni dibujarse.",
                    "Elige un punto inicial o un intervalo que no contenga ese punto.")
        v = complex(exact)
    except Exception:
        pass
    if v is not None and (abs(v.imag) > 1e-12 or math.isnan(v.real) or math.isinf(v.real)):
        return (DOMAIN_ERROR,
                f"La función no produce un número real finito en {pstr}.",
                "Prueba otro punto inicial o un intervalo donde la función esté definida.")
    return None


def _linear_root(base):
    """(símbolo, raíz, pendiente) si `base` es lineal en una variable."""
    import sympy as sp
    if len(base.free_symbols) != 1:
        return None
    s = list(base.free_symbols)[0]
    coef = sp.diff(base, s)
    if not coef.is_number or coef == 0:
        return None
    sol = sp.solve(sp.Eq(base, 0), s)
    if len(sol) == 1 and sol[0].is_real:
        return s, float(sol[0]), float(coef)
    return None


def _zero_str(base) -> str:
    try:
        r = _linear_root(base)
        return f"{r[0]} = {_fmt_num(r[1])}" if r else ""
    except Exception:
        return ""


def _suggest_point(base) -> str:
    """Punto válido concreto: para x-5 ≥ 0 sugiere 'x = 6' (frontera ± 1)."""
    try:
        r = _linear_root(base)
        if not r:
            return ""
        s, root, slope = r
        return f"{s} = {_fmt_num(root + (1 if slope > 0 else -1))}"
    except Exception:
        return ""


def _solve_bound(base, names) -> str:
    """Para bases lineales en una variable (p.ej. x-5) devuelve 'x ≥ 5'."""
    try:
        import sympy as sp
        if len(base.free_symbols) != 1:
            return ""
        s = list(base.free_symbols)[0]
        sol = sp.solve(sp.Eq(base, 0), s)
        if len(sol) == 1 and sol[0].is_real:
            coef = sp.diff(base, s)
            if coef.is_number:
                op = "≥" if coef > 0 else "≤"
                return f"{s} {op} {_fmt_num(float(sol[0]))}"
    except Exception:
        pass
    return ""


# ─── Lectura del historial del motor ─────────────────────────────────────────
_F_KEYS = ("f_k", "f(x)", "f_lambda", "f_best", "f(c_k)", "f_mu", "f(x*)")
_X_KEYS = ("x_k_str", "x_k", "x", "lambda_k", "x_best", "c_k")


def _last_value(row: dict, keys) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _as_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _x_to_str(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            parts = s[1:-1].replace(",", " ").split()
            try:
                return _fmt_vec([float(p) for p in parts])
            except Exception:
                return s
        return s
    if hasattr(v, "__iter__"):
        return _fmt_vec(list(v))
    return _fmt_num(v)


def _has_bad(v) -> bool:
    try:
        import numpy as np
        arr = np.asarray(v, dtype=float)
        return bool(np.any(~np.isfinite(arr)))
    except Exception:
        if isinstance(v, str):
            return "nan" in v.lower() or "inf" in v.lower()
        return False


def _nonfinite_values(row: dict, x_raw=None) -> tuple:
    """(hay_nan, hay_+inf, hay_-inf) entre los valores numéricos de la fila."""
    import numpy as np
    vals = []
    for v in list(row.values()) + ([x_raw] if x_raw is not None else []):
        if isinstance(v, bool):
            continue
        try:
            vals.extend(np.asarray(v, dtype=float).ravel().tolist())
        except Exception:
            if isinstance(v, str):
                lv = v.lower()
                if "nan" in lv:
                    vals.append(float("nan"))
                if "-inf" in lv:
                    vals.append(float("-inf"))
                elif "inf" in lv:
                    vals.append(float("inf"))
    return (any(math.isnan(x) for x in vals),
            any(math.isinf(x) and x > 0 for x in vals),
            any(math.isinf(x) and x < 0 for x in vals))


def _intermediate_overflow(fx: str, point) -> bool:
    """¿Alguna subexpresión de f supera el rango float en `point`? (∞−∞, 0·∞…)"""
    try:
        import sympy as sp
        expr, names = parse_expression(fx)
        subs = {sp.Symbol(n): float(p) for n, p in zip(names, point)}
        for node in sp.preorder_traversal(expr):
            if node.free_symbols:
                val = node.subs(subs).evalf()
                if val.is_real and val.is_finite and abs(val) > FLOAT_MAX:
                    return True
    except Exception:
        pass
    return False


def _last_point(ev: "OptimizationEvent", row: dict):
    """Último punto numérico finito conocido (para diagnosticar), o x₀."""
    import numpy as np
    for key in _X_KEYS:
        if key in row:
            try:
                arr = np.asarray(row[key], dtype=float).ravel()
                if arr.size and np.all(np.isfinite(arr)):
                    return arr.tolist()
            except Exception:
                pass
    return list(ev.x0) if ev.x0 else None


def _base(metodo, fx, vars_, x0, lo, hi) -> dict:
    return dict(metodo=metodo or "", fx=fx or "", vars_=list(vars_ or []),
                x0=[float(v) for v in (x0 or [])], lo=lo, hi=hi)


# ─── Detector: resultado del motor ───────────────────────────────────────────
def classify_result(metodo: str, fx: str, vars_, x0, lo, hi, hist: list,
                    result_summary: str = "", gx: str = "", tol_L: float = None,
                    duration_s: float = None) -> OptimizationEvent:
    ev = OptimizationEvent(kind=SUCCESS, result_summary=result_summary,
                           **_base(metodo, fx, vars_, x0, lo, hi))
    _fill_params(ev, gx, tol_L)
    ev.duration_s = duration_s
    if not hist:
        ev.kind = EXECUTION_ERROR
        ev.error_message = "El método no devolvió ninguna iteración."
        ev.cause = "El motor terminó sin producir resultados para estos datos."
        ev.recommendation = "Revisa la función, el rango y el método seleccionado."
        return ev
    last = hist[-1] if isinstance(hist[-1], dict) else {}
    ev.iterations = len(hist)
    f_raw = _last_value(last, _F_KEYS)
    x_raw = _last_value(last, _X_KEYS)
    ev.f_opt = _as_float(f_raw)
    ev.x_opt = _x_to_str(x_raw)
    g = _as_float(last.get("grad_norm"))
    ev.grad_norm = g
    conv = last.get("converged")
    if isinstance(conv, bool):
        ev.converged = conv

    # 1) Valores no válidos → nunca se presentan como éxito
    bad = [k for k, v in last.items()
           if isinstance(v, (float, int)) and not isinstance(v, bool) and _has_bad(v)]
    if x_raw is not None and _has_bad(x_raw):
        bad.append("x")
    if bad:
        ev.kind = WARNING
        ev.converged = False
        ev.error_message = ("El cálculo produjo valores no válidos (NaN o infinito) en: "
                            + ", ".join(sorted(set(bad))) + ".")
        diag = _diagnose_start(metodo, fx, ev.vars_, ev.x0, lo, hi)
        if diag:
            kind, cause, rec, point = diag
            ev.kind, ev.cause, ev.recommendation, ev.point = kind, cause, rec, point
        else:
            has_nan, pos_inf, neg_inf = _nonfinite_values(last, x_raw)
            start = list(ev.x0) or _last_point(ev, last)
            ovf = bool(start) and _intermediate_overflow(fx, start)
            ev.point = start or None
            if has_nan:
                ev.kind = NAN_RESULT
                ev.error_message = ("El resultado es NaN («no es un número») en: "
                                    + ", ".join(sorted(set(bad))) + ".")
                ev.cause = ("NaN aparece cuando el cálculo realiza una operación indeterminada "
                            "(∞ − ∞, 0·∞, 0/0). "
                            + ("Aquí, partes de la función superan el rango de coma flotante "
                               "(≈ 1.8·10^308) y se convierten en ∞ antes de restarse, aunque el "
                               "valor exacto sea finito." if ovf else
                               "Durante las iteraciones el método llegó a una zona donde la "
                               "función no está definida o diverge."))
                ev.recommendation = ("Empieza desde un punto con valores más moderados (por "
                                     "ejemplo, cerca de 0), reduce el rango o reescribe la función "
                                     "de forma estable (p. ej. cosh(x) - sinh(x) = exp(-x)).")
            elif pos_inf or neg_inf:
                ev.kind = OVERFLOW if ovf else INFINITE_RESULT
                sign = "+∞" if pos_inf and not neg_inf else ("−∞" if neg_inf and not pos_inf
                                                             else "±∞")
                ev.error_message = f"El cálculo produjo un valor {sign} en: " + \
                    ", ".join(sorted(set(bad))) + "."
                if ovf:
                    ev.cause = ("La magnitud de la función supera el mayor número representable "
                                "(≈ 1.8·10^308): el valor se desborda a infinito.")
                    ev.recommendation = ("Usa un punto inicial o un rango con valores más "
                                         "pequeños, o minimiza una versión reescalada de la "
                                         "función (por ejemplo log(f)).")
                else:
                    ev.cause = (f"La función toma el valor {sign} durante el cálculo: tiene un "
                                "polo o no está acotada en esa región, así que no existe un "
                                "mínimo finito que mostrar.")
                    ev.recommendation = ("Acota el rango de búsqueda para excluir el polo o la "
                                         "zona donde la función crece sin límite.")
            else:
                ev.cause = ("Durante las iteraciones el método llegó a una zona donde la función "
                            "no está definida o crece sin límite (diverge).")
                ev.recommendation = ("Prueba un punto inicial más cercano al mínimo, reduce el "
                                     "rango o verifica que la función tenga un mínimo en esa región.")
        return ev
    if ev.f_opt is not None and ev.f_opt < -1e12:
        # Un valor negativo NO es un error; uno que tiende a −∞ sí impide un óptimo
        ev.kind = INFINITE_RESULT
        ev.converged = False
        ev.error_message = (f"El valor f = {_fmt_num(ev.f_opt)} decrece sin límite "
                            f"(tiende a −∞).")
        ev.cause = ("La función no está acotada inferiormente en la región explorada: siempre "
                    "existe un punto con un valor aún menor, así que no hay mínimo finito.")
        ev.recommendation = ("Acota el rango de búsqueda a una región donde la función tenga un "
                             "mínimo, o revisa el signo de la función.")
        return ev
    if ev.f_opt is not None and abs(ev.f_opt) > 1e12:
        ev.kind = WARNING
        ev.converged = False
        ev.error_message = f"El valor f* = {_fmt_num(ev.f_opt)} es extremadamente grande."
        ev.cause = "La función probablemente no tiene mínimo en la región explorada (no está acotada)."
        ev.recommendation = "Acota el rango de búsqueda o revisa que la función tenga un mínimo."
        return ev

    # 2) Convergencia
    not_conv = False
    if ev.converged is False:
        not_conv = True
    elif metodo in GRADIENT_METHODS and not gx and g is not None and g > GRAD_TOL:
        not_conv = True
    if not_conv:
        ev.converged = False
        lim = MAX_ITER.get(metodo)
        reached = lim is not None and ev.iterations >= lim
        ev.kind = MAX_ITERATIONS if reached else CONVERGENCE
        ev.error_message = (
            f"Se alcanzó el máximo de {lim} iteraciones sin llegar a la precisión solicitada."
            if reached else "El método terminó sin cumplir el criterio de convergencia.")
        ev.cause = ("El gradiente final sigue lejos de cero"
                    + (f" (‖∇f‖ = {_fmt_num(g)})" if g is not None else "")
                    + ": el punto obtenido no es un mínimo confirmado. Suele ocurrir cuando "
                      "el punto inicial está en una zona muy plana o lejos del mínimo, o "
                      "cuando la curvatura hace que los pasos de Newton se alejen.")
        ev.recommendation = ("Prueba otro punto inicial más cercano al mínimo, cambia el tamaño "
                             "de paso o la tolerancia, o usa un método más robusto "
                             "(Armijo, Wolfe o Sección Dorada en 1D).")
        return ev
    ev.converged = True if ev.converged is None else ev.converged
    return ev


def _fill_params(ev: OptimizationEvent, gx: str = "", tol_L: float = None):
    m = ev.metodo
    ev.tolerance = tol_L if (m == "Fibonacci" and tol_L) else TOLERANCE.get(m)
    ev.max_iter = MAX_ITER.get(m)
    if gx:
        ev.params["Restricción g(x)"] = gx


def parameter_event(metodo: str, fx: str, message: str, cause: str,
                    recommendation: str, lo=None, hi=None) -> OptimizationEvent:
    """Datos de entrada incorrectos detectados por la validación de la UI."""
    return OptimizationEvent(kind=PARAMETER_ERROR, metodo=metodo or "", fx=fx or "",
                             lo=lo, hi=hi, error_message=message, cause=cause,
                             recommendation=recommendation)


def _diagnose_start(metodo, fx, vars_, x0, lo, hi):
    try:
        _, names = parse_expression(fx)
    except Exception:
        return None
    pts = []
    if metodo in GRADIENT_METHODS:
        pts.append(effective_start_point(names, x0, lo, hi))
    if lo is not None and hi is not None and len(names) == 1:
        pts += [[lo], [hi], [(lo + hi) / 2]]
    for p in pts:
        r = check_point(fx, p, names)
        if r:
            return r + (p,)
    return None


# ─── Detector: comprobación previa en el punto inicial ───────────────────────
def preflight(metodo: str, fx: str, vars_, x0, lo, hi) -> Optional[OptimizationEvent]:
    """Antes de ejecutar un método que arranca en x₀, verifica que f esté
    definida allí. Evita lanzar el motor sobre un punto imposible (p. ej.
    sqrt(x-5) en x=2), que produciría NaN o un bucle de búsqueda de paso."""
    if metodo not in GRADIENT_METHODS:
        return None
    try:
        _, names = parse_expression(fx)
    except Exception:
        return None
    if vars_ and vars_ != ["x1", "x2"] and any(v.strip() for v in vars_):
        names = [v for v in vars_ if v.strip()]
    p = effective_start_point(names, x0, lo, hi)
    r = check_point(fx, p, names)
    if not r:
        return None
    kind, cause, rec = r
    ev = OptimizationEvent(kind=kind, cause=cause, recommendation=rec, point=p,
                           **_base(metodo, fx, names, p, lo, hi))
    ev.error_message = (f"La función no se puede evaluar en el punto inicial "
                        f"{_point_str(names, p)}.")
    return ev


# ─── Detector: excepciones del motor / de la validación ──────────────────────
def classify_exception(metodo: str, fx: str, vars_, x0, lo, hi, exc: BaseException,
                       point: Sequence[float] = None) -> OptimizationEvent:
    ev = OptimizationEvent(kind=EXECUTION_ERROR, **_base(metodo, fx, vars_, x0, lo, hi))
    _fill_params(ev)
    ev.exception_type = type(exc).__name__
    msg = str(exc).strip()
    ev.technical_detail = f"{type(exc).__name__}: {msg}"
    first_line = msg.split("\n")[0] if msg else type(exc).__name__
    ev.error_message = first_line

    syn = syntax_problem(fx)
    if syn:
        ev.kind = SYNTAX_ERROR
        ev.error_message = "La expresión matemática no se pudo interpretar."
        ev.cause = _syntax_cause(fx)
        ev.recommendation = ("Escribe la función con sintaxis de Python/Sympy: usa * para "
                             "multiplicar, ** o ^ para potencias, paréntesis balanceados y "
                             "funciones como sqrt(), exp(), log(), sin(). Ejemplo: (x-3)**2 + 2.")
        return ev

    if isinstance(exc, ZeroDivisionError):
        ev.kind = DIVISION_BY_ZERO
    low0 = msg.lower()
    if isinstance(exc, OverflowError) or any(k in low0 for k in (
            "result too large", "math range error", "overflow")):
        ev.kind = OVERFLOW
        ev.cause = ("Durante las iteraciones los valores crecieron hasta superar el mayor número "
                    "representable (≈ 1.8·10^308): el cálculo se desbordó. Suele indicar que la "
                    "función no está acotada inferiormente en esa dirección o que el paso diverge.")
        ev.recommendation = ("Acota el rango, empieza desde un punto más cercano a un mínimo, o "
                             "revisa si la función realmente tiene un mínimo (por ejemplo, -x**3 "
                             "decrece sin límite).")
        return ev
    diag = None
    if point is not None:
        r = check_point(fx, point)
        diag = r + (list(point),) if r else None
    if diag is None:
        diag = _diagnose_start(metodo, fx, vars_, x0, lo, hi)
    if diag:
        ev.kind, ev.cause, ev.recommendation, ev.point = diag
        return ev
    low = msg.lower()
    if ev.kind == DIVISION_BY_ZERO or "division by zero" in low or "zerodivision" in low:
        ev.kind = DIVISION_BY_ZERO
        ev.cause = "En algún paso el método evaluó un punto donde un denominador vale cero."
        ev.recommendation = "Cambia el punto inicial o el intervalo para evitar la discontinuidad."
    elif "math domain" in low or "complex" in low:
        ev.kind = DOMAIN_ERROR
        ev.cause = "El método evaluó la función fuera de su dominio real."
        ev.recommendation = "Elige un punto inicial e intervalo donde la función esté definida."
    elif any(k in low for k in ("variables", "requiere", "restricci", "tolerancia",
                                "rango", "dimensi", "estrictamente")):
        ev.kind = PARAMETER_ERROR
        ev.cause = ("Los datos de entrada no son compatibles con el método elegido: "
                    + first_line)
        ev.recommendation = ("Revisa el número de variables, el punto inicial, el rango y, si el "
                             "método lo exige, la restricción g(x); o elige un método adecuado "
                             "para este tipo de función.")
    else:
        ev.cause = ("El motor no pudo aplicar el método a esta función con los datos dados "
                    "(por ejemplo, número de variables incompatible, función no derivable o "
                    "falta de una restricción).")
        ev.recommendation = ("Revisa que el método sea adecuado para la función (1D o multivariable, "
                             "con o sin restricción) y que los datos de entrada sean correctos.")
    return ev


def _syntax_cause(fx: str) -> str:
    s = fx or ""
    if s.count("(") != s.count(")"):
        return (f"Los paréntesis no están balanceados: hay {s.count('(')} de apertura "
                f"y {s.count(')')} de cierre.")
    if "\\" in s or "$" in s or "{" in s:
        return "La expresión contiene notación LaTeX (\\, $, { }), que el motor no interpreta."
    stripped = s.replace(" ", "")
    for bad in ("+*", "*+", "-*", "/*", "*/", "++*", "**+", "//"):
        if bad in stripped:
            return f"Hay operadores seguidos sin un número entre ellos ('{bad}')."
    if stripped and stripped[-1] in "+-*/^":
        return "La expresión termina en un operador: le falta el último término."
    return "La expresión tiene un error de escritura que impide interpretarla como fórmula."


def event_from_eval_problem(metodo, fx, vars_, x0, lo, hi, point, value) -> OptimizationEvent:
    """Para la validación previa de la ventana principal (f(punto) = inf/NaN)."""
    r = check_point(fx, point)
    ev = OptimizationEvent(kind=WARNING, point=list(point), **_base(metodo, fx, vars_, x0, lo, hi))
    _fill_params(ev)
    ev.error_message = f"La función vale {_fmt_num(value)} en el punto de prueba {_fmt_vec(point)}."
    if r:
        ev.kind, ev.cause, ev.recommendation = r
    elif isinstance(value, float) and math.isnan(value):
        ev.kind = NAN_RESULT
        ev.cause = "La función produce NaN (operación indeterminada) en el centro del rango."
        ev.recommendation = "Ajusta el rango de búsqueda o reescribe la expresión de forma estable."
    else:
        if isinstance(value, float) and math.isinf(value):
            ev.kind = INFINITE_RESULT
        ev.cause = "La función no produce un valor real finito en el centro del rango."
        ev.recommendation = "Ajusta el rango de búsqueda o revisa la expresión."
    return ev


# ─── Explicación local (respaldo sin IA) ─────────────────────────────────────
def _why_method(ev: OptimizationEvent) -> str:
    m = ev.metodo
    if m in GRADIENT_METHODS:
        g = f" (‖∇f‖ = {_fmt_num(ev.grad_norm)})" if ev.grad_norm is not None else ""
        return (f"{m} avanza usando la pendiente de la función y se detiene cuando el "
                f"gradiente prácticamente se anula{g}: en ese punto la función ya no baja "
                f"en ninguna dirección.")
    if m in INTERVAL_METHODS:
        return (f"{m} va reduciendo el intervalo de búsqueda, descartando en cada paso la "
                f"parte donde la función es mayor, hasta que el intervalo es tan pequeño "
                f"que encierra al mínimo.")
    if m in META_METHODS:
        return (f"{m} explora muchos puntos de forma aleatoria guiada y se queda con el mejor; "
                f"es un buen candidato, aunque no garantiza el óptimo exacto.")
    return f"El método {m} terminó su procedimiento y entregó este resultado."


def local_explanation(ev: OptimizationEvent) -> str:
    """Explicación didáctica construida SOLO con los datos del motor."""
    if ev.kind == SUCCESS:
        parts = []
        if ev.x_opt is not None and ev.f_opt is not None:
            var = ev.vars_[0] if len(ev.vars_) == 1 else "x"
            parts.append(f"El método encontró el mínimo en {var} = {ev.x_opt}, "
                         f"donde la función alcanza su valor mínimo f = {_fmt_num(ev.f_opt)}.")
        elif ev.result_summary:
            parts.append(f"Resultado obtenido: {ev.result_summary}.")
        parts.append(_why_method(ev))
        parts.append(f"Se usaron {ev.iterations} iteraciones y el método convergió.")
        parts.append("Interpretación: ningún punto cercano produce un valor menor de la "
                     "función; si buscas el óptimo global en funciones con varios valles, "
                     "conviene repetir con otros puntos iniciales.")
        parts.append("Para experimentar: cambia el punto inicial o el rango, o compara con "
                     "otro método y observa cuántas iteraciones necesita cada uno.")
        return " ".join(parts)
    if ev.kind in NON_CONVERGED:
        return (f"El método {ev.metodo} no encontró una solución con la precisión solicitada. "
                f"{ev.error_message} {ev.cause} "
                f"Recomendación: {ev.recommendation} "
                f"Después, vuelve a pulsar Calcular para repetir el procedimiento.")
    head = {
        DOMAIN_ERROR: "Se produjo un error de dominio: la función no está definida en los números reales para el punto evaluado.",
        DIVISION_BY_ZERO: "Se produjo una división entre cero.",
        SYNTAX_ERROR: "La función tiene un error de sintaxis y no se pudo interpretar.",
        WARNING: "El resultado numérico no es válido, así que no debe tomarse como óptimo.",
        INFINITE_RESULT: "El resultado es infinito, así que no puede mostrarse como un óptimo ni representarse en la gráfica.",
        NAN_RESULT: "El resultado es NaN, es decir, «no es un número»: no puede interpretarse ni graficarse.",
        OVERFLOW: "Se produjo un desbordamiento numérico: el cálculo superó el rango de números que el ordenador puede representar.",
        PARAMETER_ERROR: "Los parámetros de entrada no son correctos para este método.",
        EXECUTION_ERROR: "El cálculo no se pudo completar.",
    }.get(ev.kind, "Ocurrió un problema.")
    txt = f"{head} {ev.cause}".strip()
    if ev.recommendation:
        txt += f" Recomendación: {ev.recommendation}"
    txt += " Cuando lo corrijas, vuelve a pulsar Calcular para ejecutar de nuevo el método."
    return txt


# ─── Identidad de Opti (saludo e invitación) ─────────────────────────────────
OPTI_NAME = "Opti"

_FIRST_GREETING = [
    "¡Hola! ¿Cómo estás? Soy Opti, tu asistente inteligente de optimización matemática. "
    "Estoy aquí para ayudarte a comprender los resultados y a encontrar soluciones cuando "
    "algo no sale como esperamos. {det} Vamos a revisarlo juntos.",
    "¡Hola! Soy Opti, el asistente inteligente del Optimizador de Funciones. Mi trabajo es "
    "ayudarte a entender cada resultado y a resolver lo que se complique. {det} "
    "Tranquilo, tiene solución; veámoslo juntos.",
]
_NEXT_GREETING = [
    "Hola de nuevo. {det} Vamos a verlo.",
    "Aquí estoy otra vez. {det} Revisémoslo juntos.",
    "De nuevo por aquí. {det} Tiene arreglo; te explico.",
]
_DETECTED = {
    INFINITE_RESULT: "He detectado que la función que acabas de ejecutar dio un resultado infinito.",
    NAN_RESULT: "He detectado que el cálculo que acabas de ejecutar devolvió un valor NaN.",
    OVERFLOW: "He detectado un desbordamiento numérico en la función que acabas de ejecutar.",
    DOMAIN_ERROR: "He detectado que la función se evaluó fuera de su dominio.",
    DIVISION_BY_ZERO: "He detectado una división entre cero en la función que acabas de ejecutar.",
    SYNTAX_ERROR: "He detectado un detalle de escritura en la función.",
    MAX_ITERATIONS: "He detectado que el método agotó sus iteraciones sin llegar a la precisión pedida.",
    CONVERGENCE: "He detectado que el método no logró converger.",
    PARAMETER_ERROR: "He detectado un dato de entrada que no es válido para el método.",
    WARNING: "He detectado un resultado numérico que no es válido.",
    EXECUTION_ERROR: "He detectado que la función no se pudo evaluar con el método elegido.",
}
_INVITATIONS = [
    "Si quieres, puedes preguntarme cualquier duda sobre esta función, los métodos o los "
    "resultados: escríbeme en el chat o háblame con el micrófono.",
    "¿Te quedó alguna duda? Pregúntame por el chat o con el micrófono; estaré encantado de ayudarte.",
    "Cuando quieras, escríbeme o usa el micrófono y seguimos analizándolo juntos.",
]
_INVITE_HINTS = ("pregúnta", "pregunta", "escríbeme", "escribeme", "micrófono", "microfono",
                 "chat", "consúltame", "consultame", "duda")


def opti_greeting(ev: OptimizationEvent, first: bool, n: int = 0) -> str:
    """Saludo de Opti (etapa 1). `first`: primera intervención de la sesión
    (se presenta por su nombre); después, saludos breves que van variando."""
    if ev.kind not in CHAT_KINDS:
        return ""   # Opti se presenta en la primera intervención por un problema (abre el chat)
    det = _DETECTED.get(ev.kind, "He detectado una situación en la función que acabas de ejecutar.")
    pool = _FIRST_GREETING if first else _NEXT_GREETING
    return pool[n % len(pool)].format(det=det)


def opti_invitation(n: int = 0) -> str:
    return _INVITATIONS[n % len(_INVITATIONS)]


def ensure_invitation(text: str, n: int = 0) -> str:
    """Garantiza la etapa 4 (invitación) aunque la IA la haya omitido."""
    t = (text or "").strip()
    if any(h in t.lower()[-260:] for h in _INVITE_HINTS):
        return t
    return f"{t} {opti_invitation(n)}".strip()


# ─── Prompt para la IA (solo interpreta, no calcula) ─────────────────────────
AI_SYSTEM_PROMPT = (
    "Eres Opti, el asistente inteligente del 'Optimizador de Funciones': un tutor de "
    "matemáticas y optimización numérica amigable, profesional, paciente y motivador. "
    "Recibes el resultado REAL del motor matemático de la app. Reglas: "
    "1) Responde en español, en un solo párrafo de 70 a 130 palabras, sin markdown, "
    "sin tablas ni listas, apto para leerse en voz alta. "
    "2) NO saludes ni te presentes: el saludo ya se dijo. Empieza directamente. "
    "3) NUNCA inventes ni recalcules números: usa solo los valores que se te dan. "
    "4) Si fue un problema, sigue este orden: (a) qué ocurrió, con qué función y qué "
    "resultado se obtuvo, y qué condición matemática o numérica lo provocó (infinito, NaN, "
    "desbordamiento, dominio, división entre cero, convergencia…); (b) una recomendación "
    "concreta: qué parámetro revisar, qué valor probar y por qué funcionaría, compatible "
    "con el dominio de la función; (c) termina con una invitación breve y natural a "
    "seguir preguntándote por el chat o el micrófono. "
    "5) Si fue éxito: qué significa el resultado, por qué el método llegó ahí y cómo "
    "interpretarlo. Un valor negativo o un mínimo local no son errores. "
    "6) No culpes al usuario; transmite que el problema tiene solución."
)


def build_ai_prompt(ev: OptimizationEvent) -> str:
    lines = [f"Tipo de evento detectado por el motor: {ev.title} ({ev.kind})."]
    for k, v in ev.facts():
        lines.append(f"- {k}: {v}")
    if ev.recommendation:
        lines.append(f"- Recomendación del analizador: {ev.recommendation}")
    if ev.technical_detail:
        lines.append(f"- Detalle técnico: {ev.technical_detail[:400]}")
    lines.append("Explica al usuario qué ocurrió, por qué y qué debería hacer, "
                 "y cierra invitándolo a seguir preguntándote.")
    return "\n".join(lines)
