# app/statistics/inferential.py
# Estadística inferencial: intervalo de confianza, prueba t, regresión lineal
# (simple/múltiple por mínimos cuadrados) y simulación de Monte Carlo.
from __future__ import annotations

import ast
import re
from typing import Any, Callable, Dict, List

import numpy as np
from scipy import stats as sps


def confidence_interval(data: List[float], level: float) -> Dict[str, Any]:
    x = np.asarray(data, dtype=float)
    if x.size < 2:
        raise ValueError("El intervalo de confianza requiere al menos 2 datos.")
    if not 0 < level < 1:
        raise ValueError("El nivel de confianza debe estar entre 0 y 1 (ej.: 0.95).")
    n, mean, s = x.size, float(x.mean()), float(x.std(ddof=1))
    se = s / np.sqrt(n)
    t = float(sps.t.ppf((1 + level) / 2, n - 1))
    low, high = mean - t * se, mean + t * se

    def plot(fig):
        ax = fig.add_subplot(111)
        ax.errorbar([mean], [0], xerr=[[mean - low], [high - mean]], fmt="o",
                    color="#ffd700", ecolor="#4a9eff", capsize=8)
        ax.set_yticks([]); ax.set_title(f"IC del {level:.0%} para la media")

    return {
        "low": low, "high": high, "mean": mean, "t": t, "se": se,
        "hist": [{"Concepto": k, "Valor": v} for k, v in [
            ("n", n), ("Media muestral", mean), ("Desviación estándar", s),
            ("Error estándar", se), ("t crítico", t),
            ("Límite inferior", low), ("Límite superior", high)]],
        "resumen": f"IC {level:.0%}: [{low:.4f}, {high:.4f}]",
        "plot": plot,
    }


def t_test(data: List[float], mu0: float, alternative: str, alpha: float) -> Dict[str, Any]:
    x = np.asarray(data, dtype=float)
    if x.size < 2:
        raise ValueError("La prueba t requiere al menos 2 datos.")
    if not 0 < alpha < 1:
        raise ValueError("El nivel de significancia α debe estar entre 0 y 1.")
    keys = {"Bilateral (≠)": "two-sided", "Mayor (>)": "greater", "Menor (<)": "less"}
    res = sps.ttest_1samp(x, mu0, alternative=keys[alternative])
    t, p = float(res.statistic), float(res.pvalue)
    reject = p < alpha
    df = x.size - 1
    decision = "Se rechaza H0" if reject else "No se rechaza H0"

    def plot(fig):
        ax = fig.add_subplot(111)
        lim = max(4.0, abs(t) + 1)
        grid = np.linspace(-lim, lim, 400)
        ax.plot(grid, sps.t.pdf(grid, df), color="#4a9eff", label=f"t con {df} g.l.")
        ax.axvline(t, color="#ffd700", label=f"t observado = {t:.3f}")
        ax.set_title("Distribución bajo H0"); ax.legend()

    return {
        "t": t, "p": p, "reject": reject,
        "hist": [{"Concepto": k, "Valor": v} for k, v in [
            ("H0", f"μ = {mu0:g}"), ("H1", alternative), ("Media muestral", float(x.mean())),
            ("Estadístico t", t), ("Grados de libertad", df), ("Valor p", p),
            ("α", alpha), ("Decisión", decision)]],
        "resumen": f"t = {t:.4f}, p = {p:.4f}: {decision}",
        "plot": plot,
    }


def regression(table: List[List[float]]) -> Dict[str, Any]:
    """OLS con intercepto. La última columna de cada fila es la respuesta y."""
    M = np.asarray(table, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("Cada fila debe tener al menos un valor de x y uno de y.")
    y = M[:, -1]
    X = np.column_stack([np.ones(len(y)), M[:, :-1]])
    n, p = X.shape
    if n <= p:
        raise ValueError(f"Se requieren más de {p} observaciones para estimar {p} coeficientes.")
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("Las variables independientes son colineales o constantes.")
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    fitted = X @ beta
    res = y - fitted
    sse = float(res @ res)
    sst = float(((y - y.mean()) ** 2).sum())
    if sst == 0:
        raise ValueError("La variable y es constante; no hay nada que explicar.")
    r2 = 1 - sse / sst
    adj = 1 - (1 - r2) * (n - 1) / (n - p)
    mse = sse / (n - p)
    se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))
    with np.errstate(divide="ignore"):
        tv = beta / se
    pv = 2 * sps.t.sf(np.abs(tv), n - p)
    f = ((sst - sse) / (p - 1)) / mse if mse > 0 else float("inf")
    pf = float(sps.f.sf(f, p - 1, n - p)) if np.isfinite(f) else 0.0

    names = ["Intercepto"] + [f"x{i}" for i in range(1, p)]
    rows: List[Dict[str, Any]] = [
        {"Coeficiente": nm, "Estimación": float(b), "Error estándar": float(s),
         "t": float(t), "Valor p": float(q)}
        for nm, b, s, t, q in zip(names, beta, se, tv, pv)]
    rows += [{"Coeficiente": k, "Estimación": v} for k, v in
             [("R²", r2), ("R² ajustado", adj), ("F", f), ("Valor p de F", pf)]]

    eq = f"ŷ = {beta[0]:.4f}" + "".join(f" {b:+.4f}·x{i}" for i, b in enumerate(beta[1:], 1))

    def plot(fig):
        ax = fig.add_subplot(111)
        if p == 2:
            order = np.argsort(M[:, 0])
            ax.scatter(M[:, 0], y, color="#4a9eff", label="Datos")
            ax.plot(M[order, 0], fitted[order], color="#ffd700", label="Recta ajustada")
            ax.set_xlabel("x"); ax.set_ylabel("y")
        else:
            ax.scatter(fitted, y, color="#4a9eff", label="Observado vs. ajustado")
            ax.plot([y.min(), y.max()], [y.min(), y.max()], color="#ffd700", label="y = ŷ")
            ax.set_xlabel("ŷ"); ax.set_ylabel("y")
        ax.set_title(f"Regresión lineal (R² = {r2:.4f})"); ax.legend()

    return {"beta": beta.tolist(), "r2": r2, "adj_r2": adj, "f": f, "hist": rows,
            "resumen": f"{eq}, R² = {r2:.4f}", "plot": plot, "fx": eq}


# ── Monte Carlo ──────────────────────────────────────────────────────────────
_SAFE_FUNCS: Dict[str, Callable] = {
    "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos,
    "tan": np.tan, "abs": np.abs, "maximo": np.maximum, "minimo": np.minimum,
    "max": np.maximum, "min": np.minimum,
}
_SAFE_CONST = {"pi": np.pi, "e": np.e}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
)


def _compile_expr(expr: str, names: List[str]):
    """Compila una expresión solo con aritmética, comparaciones y funciones de
    la lista blanca: el texto del usuario nunca llega a un eval sin revisar."""
    try:
        tree = ast.parse(expr.replace("^", "**"), mode="eval")
    except SyntaxError as ex:
        raise ValueError(f"No se pudo interpretar la expresión «{expr}».") from ex
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"La expresión contiene un elemento no permitido: «{expr}».")
        if isinstance(node, ast.Name) and node.id not in names \
                and node.id not in _SAFE_FUNCS and node.id not in _SAFE_CONST:
            raise ValueError(f"Variable o función desconocida: «{node.id}».")
        if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS):
            raise ValueError("Solo se permiten funciones matemáticas básicas.")
    return compile(tree, "<monte_carlo>", "eval")


def _sampler(spec: str, rng: np.random.Generator, n: int) -> np.ndarray:
    m = re.fullmatch(r"\s*([a-záéíóúñ]+)\s*\(([^)]*)\)\s*", spec, re.I)
    if not m:
        raise ValueError(f"Distribución inválida: «{spec}». Ejemplo: normal(0, 1).")
    name = m.group(1).lower()
    try:
        a = [float(v) for v in m.group(2).split(",") if v.strip()]
    except ValueError as ex:
        raise ValueError(f"Parámetros no numéricos en «{spec}».") from ex
    table = {
        "uniforme": (2, lambda: rng.uniform(a[0], a[1], n)),
        "normal": (2, lambda: rng.normal(a[0], a[1], n)),
        "triangular": (3, lambda: rng.triangular(a[0], a[1], a[2], n)),
        "exponencial": (1, lambda: rng.exponential(a[0], n)),
        "lognormal": (2, lambda: rng.lognormal(a[0], a[1], n)),
        "bernoulli": (1, lambda: (rng.random(n) < a[0]).astype(float)),
    }
    if name not in table:
        raise ValueError(f"Distribución no soportada: «{name}». Use: {', '.join(table)}.")
    arity, fn = table[name]
    if len(a) != arity:
        raise ValueError(f"«{name}» requiere {arity} parámetro(s).")
    try:
        return fn()
    except ValueError as ex:
        raise ValueError(f"Parámetros inválidos en «{spec}».") from ex


def monte_carlo(expr: str, variables: str, n: int, seed: int, level: float = 0.95) -> Dict[str, Any]:
    if not 100 <= n <= 5_000_000:
        raise ValueError("El número de simulaciones debe estar entre 100 y 5 000 000.")
    rng = np.random.default_rng(seed)
    env: Dict[str, Any] = {}
    for line in filter(str.strip, variables.splitlines()):
        if "~" not in line:
            raise ValueError(f"Línea inválida: «{line}». Formato: x ~ normal(0, 1).")
        var, dist = (s.strip() for s in line.split("~", 1))
        if not var.isidentifier():
            raise ValueError(f"Nombre de variable inválido: «{var}».")
        env[var] = _sampler(dist, rng, n)
    if not env:
        raise ValueError("Define al menos una variable aleatoria.")
    code = _compile_expr(expr, list(env))
    scope = {**_SAFE_FUNCS, **_SAFE_CONST, **env}
    y = np.asarray(eval(code, {"__builtins__": {}}, scope), dtype=float)
    if y.ndim == 0:
        y = np.full(n, float(y))
    mean, std = float(y.mean()), float(y.std(ddof=1))
    se = std / np.sqrt(n)
    z = float(sps.norm.ppf((1 + level) / 2))
    low, high = mean - z * se, mean + z * se
    p5, p50, p95 = (float(v) for v in np.percentile(y, [5, 50, 95]))
    running = np.cumsum(y) / np.arange(1, n + 1)
    idx = np.unique(np.geomspace(1, n, 300).astype(int)) - 1

    def plot(fig):
        ax1, ax2 = fig.subplots(1, 2)
        ax1.hist(y, bins=50, color="#4a9eff", edgecolor="#0a1228")
        ax1.set_title("Distribución simulada")
        ax2.semilogx(idx + 1, running[idx], color="#ffd700")
        ax2.set_title("Convergencia de la media"); ax2.set_xlabel("Simulaciones")

    m = min(n, 4000)   # submuestra real (las primeras m simulaciones) para la vista 3D
    return {
        "mean": mean, "std": std, "se": se, "low": low, "high": high,
        "sample_vars": {k: np.asarray(v[:m], dtype=float) for k, v in env.items()},
        "sample_y": np.asarray(y[:m], dtype=float),
        "hist": [{"Concepto": k, "Valor": v} for k, v in [
            ("Simulaciones", n), ("Semilla", seed), ("Media estimada", mean),
            ("Desviación estándar", std), ("Error estándar", se),
            (f"IC {level:.0%} inferior", low), (f"IC {level:.0%} superior", high),
            ("Percentil 5", p5), ("Mediana", p50), ("Percentil 95", p95)]],
        "resumen": f"E[{expr}] ≈ {mean:.5f}, IC {level:.0%}: [{low:.5f}, {high:.5f}]",
        "plot": plot, "fx": expr,
    }
