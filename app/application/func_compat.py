"""
func_compat.py — Compatibilidad de funciones para el Optimizador.

Permite que TODOS los métodos (1D y MD) sean compatibles con funciones como:
  • 1D simples:     x**2 - 4*x + 5
  • 2D/nD:          2*(x1-3)**2 + x1*x2**3
  • Funciones PDF:  Rosenbrock, Himmelblau, Griewank, Rastrigin, Ackley, Beale
  • Con trig/exp:   cos(x)*exp(-x**2) + sin(y)/sqrt(x**2+y**2+1)

Características:
  - Detección automática de variables en la expresión
  - Soporte completo para funciones matemáticas (cos, sin, exp, sqrt, pi, e…)
  - Normalización de sintaxis (^ → **, multiplicación implícita)
  - Lambdify seguro con módulos numpy + manejo de DiracDelta/Heaviside
"""

from __future__ import annotations
import re
import numpy as np
from typing import Callable, List, Tuple

try:
    import sympy as sp
    _SYMPY = True
except ImportError:
    sp = None
    _SYMPY = False


# ─── Constantes y funciones que NO son variables ─────────────────────────────
_RESERVED: set = {
    # funciones trigonométricas
    "sin", "cos", "tan", "sec", "csc", "cot",
    "asin", "acos", "atan", "atan2",
    "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    # otras funciones
    "exp", "log", "log2", "log10", "ln",
    "sqrt", "cbrt", "abs", "sign", "heaviside",
    "floor", "ceil", "round", "mod",
    "max", "min", "sum", "prod",
    # notación Sympy con mayúscula inicial (Abs(x-2), Max(x,y)…) — sympify
    # las reconoce igual que sus versiones en minúscula, pero el detector de
    # variables (case-sensitive) las confundía con variables nuevas.
    "Abs", "Max", "Min", "Sign", "Floor", "Ceiling", "Piecewise",
    # constantes
    "pi", "e", "E", "inf", "nan", "true", "false",
    "True", "False", "None",
    # módulos
    "np", "numpy", "sp", "sympy", "math",
    # Python keywords
    "if", "else", "elif", "for", "while", "def", "return",
    "import", "from", "class", "lambda", "and", "or", "not",
    "in", "is", "print", "range", "len", "int", "float",
    "list", "dict", "set", "tuple", "str", "bool",
    # letras griegas comunes usadas como constantes
    "alpha", "beta", "gamma", "delta", "epsilon", "mu", "sigma",
    # parámetros típicos de funciones de prueba (cuando no son variables)
    "A", "N",
}

# Alias de sustitución en la expresión antes de parsear
_ALIASES = {
    "^"      : "**",          # potencia estilo Excel/matemático
    "ln("    : "log(",        # logaritmo natural
    "log("   : "log(",        # mantener
    "arctan(": "atan(",
    "arcsin(": "asin(",
    "arccos(": "acos(",
}


# ─── Normalización de expresiones ─────────────────────────────────────────────
def normalize_expr(expr: str) -> str:
    """
    Normaliza la expresión para que sympy la entienda correctamente.
      • ^ → **
      • ln → log
      • elimina espacios redundantes
    """
    result = expr.strip()
    for old, new in _ALIASES.items():
        result = result.replace(old, new)
    return result


# ─── Detección de variables ────────────────────────────────────────────────────
def detect_variables(expr: str) -> List[str]:
    """
    Detecta automáticamente los nombres de variables en una expresión.

    Retorna lista ordenada:
      1. x, y, z (orden estándar)
      2. x1, x2, y1, y2… (variables numeradas)
      3. Otros identificadores
    """
    # Encontrar todos los identificadores
    raw = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr)

    seen: set = set()
    variables: List[str] = []
    for c in raw:
        if c not in _RESERVED and c not in seen:
            seen.add(c)
            variables.append(c)

    # Ordenar
    _standard = ['x', 'y', 'z', 'w', 't', 'u', 'v']

    def _sort_key(v: str):
        if v in _standard:
            return (0, _standard.index(v), 0)
        m = re.match(r'^([a-zA-Z]+)(\d+)$', v)
        if m:
            return (1, _standard.index(m.group(1)) if m.group(1) in _standard else 99,
                    int(m.group(2)))
        return (2, 0, 0)

    variables.sort(key=_sort_key)
    return variables


# ─── Crear callable via sympy ─────────────────────────────────────────────────
def make_callable(expr: str, var_names: List[str]) -> Callable:
    """
    Crea una función Python callable a partir de una expresión sympy.

    Soporta todas las funciones matemáticas estándar (numpy backend).
    """
    if not _SYMPY:
        raise RuntimeError("SymPy no está instalado. Instala con: pip install sympy")

    norm = normalize_expr(expr)
    syms = sp.symbols(" ".join(var_names)) if var_names else ()
    if not isinstance(syms, (list, tuple)):
        syms = (syms,)

    # Crear diccionario local para sympify
    local_dict = {v: s for v, s in zip(var_names, syms)}

    f_sym = sp.sympify(norm, locals=local_dict)

    # Módulos: numpy + manejo de funciones especiales de sympy
    modules = [
        {
            "DiracDelta"  : lambda *a: 0.0,
            "Heaviside"   : lambda x, *a: np.heaviside(float(x), 0.5),
            "conjugate"   : np.conj,
            "re"          : np.real,
            "im"          : np.imag,
        },
        "numpy",
    ]

    fn = sp.lambdify(list(syms), f_sym, modules=modules)
    return fn


# ─── Parsear función 1D ───────────────────────────────────────────────────────
def parse_function_1d(expr: str) -> Tuple[Callable, str]:
    """
    Parsea una función 1D, detectando automáticamente la variable.

    Returns:
        (fn, var_name) donde fn(v) → float

    Raises:
        ValueError si la función tiene más de una variable.
    """
    norm = normalize_expr(expr)
    vars_ = detect_variables(norm)

    if len(vars_) == 0:
        # Constante
        try:
            val = float(sp.sympify(norm)) if _SYMPY else float(eval(norm))
        except Exception:
            val = 0.0
        return lambda v: val, "x"

    if len(vars_) > 1:
        raise ValueError(
            f"La función tiene {len(vars_)} variables: {', '.join(vars_)}.\n\n"
            f"Los métodos de Búsqueda Local y Fibonacci son métodos "
            f"de búsqueda en UNA sola dimensión y solo pueden usarse con "
            f"funciones de una variable.\n\n"
            f"Para funciones multivariable use:\n"
            f"  • Armijo o Wolfe  (con variables y punto x₀)\n"
            f"  • Métodos MD (Penalización, Barreras, BFGS…)"
        )

    var_name = vars_[0]
    fn_raw = make_callable(norm, [var_name])
    return lambda v: float(fn_raw(v)), var_name


# ─── Parsear función multivariable ─────────────────────────────────────────────
def parse_function_nd(expr: str,
                      var_names: List[str] | None = None) -> Tuple[Callable, List[str]]:
    """
    Parsea una función multivariable.

    Args:
        expr:      expresión en Python/sympy
        var_names: variables a usar (si None, se auto-detectan)

    Returns:
        (fn, var_names) donde fn(*args) → float
    """
    norm = normalize_expr(expr)
    if var_names is None or len(var_names) == 0:
        var_names = detect_variables(norm)
    if len(var_names) == 0:
        var_names = ["x"]

    fn_raw = make_callable(norm, var_names)
    n = len(var_names)

    def fn(*args):
        vals = args[:n] if len(args) >= n else list(args) + [0.0] * (n - len(args))
        return float(fn_raw(*vals))

    return fn, var_names


# ─── Evaluar en un punto de prueba ─────────────────────────────────────────────
def safe_test_point(expr: str, var_names: List[str], test_vals: List[float]) -> float:
    """
    Evalúa la expresión en un punto de prueba de forma segura.
    Lanza ValueError con mensaje descriptivo si falla.
    """
    try:
        norm = normalize_expr(expr)
        fn = make_callable(norm, var_names)
        result = fn(*test_vals)
        # Intentar convertir a float; si falla, hay variables no sustituidas
        fval = float(result)
        return fval
    except (TypeError, AttributeError):
        # Resultado es expresión simbólica — variables de la expresión ≠ var_names
        # Detectar cuáles son las variables reales
        real_vars = detect_variables(normalize_expr(expr))
        raise ValueError(
            f"La expresión contiene las variables {real_vars} "
            f"pero se intentó evaluar con {var_names}.\n"
            f"Verifica que el nombre de variables coincida con la expresión."
        )
    except Exception as ex:
        point_str = ", ".join(f"{v}={x:.4g}" for v, x in zip(var_names, test_vals))
        raise ValueError(
            f"No se puede evaluar la función:\n"
            f"  f({', '.join(var_names)}) = {expr}\n"
            f"en el punto: {point_str}\n\n"
            f"Error: {ex}\n\n"
            f"Verifica la sintaxis. Ejemplos correctos:\n"
            f"  • x**2 - 4*x + 5\n"
            f"  • (1-x)**2 + 100*(y - x**2)**2\n"
            f"  • cos(x)*exp(-x**2) + sin(y)\n"
            f"  • 2*(x1-3)**2 + x1*x2**3"
        )


# ─── Información de una función ────────────────────────────────────────────────
def function_info(expr: str, var_names: List[str] | None = None) -> dict:
    """
    Retorna información sobre la función detectada.
    """
    norm = normalize_expr(expr)
    detected = detect_variables(norm)
    used_vars = var_names if var_names else detected
    n = len(used_vars)

    return {
        "normalized" : norm,
        "variables"  : used_vars,
        "n_vars"     : n,
        "is_1d"      : n <= 1,
        "is_nd"      : n > 1,
    }


# ─── Funciones de prueba del PDF ──────────────────────────────────────────────
# Expresiones listas para copiar/usar directamente en la interfaz
PDF_FUNCTIONS = {
    # Ejercicios del PDF
    "Cuadrática 1D: x²-4x+5"             : ("x**2 - 4*x + 5",           ["x"]),
    "Cuadrática 1D: x²-6x+10"            : ("x**2 - 6*x + 10",          ["x"]),
    "Simple: x²"                           : ("x**2",                      ["x"]),
    "Armijo 2D: 2(x1-3)²+x1·x2³"         : ("2*(x1-3)**2 + x1*x2**3",   ["x1", "x2"]),
    # Funciones de prueba clásicas
    "Rosenbrock"                           : ("(1-x)**2 + 100*(y-x**2)**2", ["x", "y"]),
    "Himmelblau"                           : ("(x**2+y-11)**2+(x+y**2-7)**2", ["x", "y"]),
    "Rastrigin 2D"                         : ("20+x**2-10*cos(2*pi*x)+y**2-10*cos(2*pi*y)", ["x", "y"]),
    "Griewank 2D"                          : ("1+(x**2+y**2)/4000-cos(x)*cos(y/sqrt(2))", ["x", "y"]),
    "Ackley 2D"                            : ("-20*exp(-0.2*sqrt((x**2+y**2)/2))-exp((cos(2*pi*x)+cos(2*pi*y))/2)+20+exp(1)", ["x", "y"]),
    "Beale"                                : ("(1.5-x+x*y)**2+(2.25-x+x*y**2)**2+(2.625-x+x*y**3)**2", ["x", "y"]),
}
