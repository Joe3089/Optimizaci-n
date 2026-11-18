# app/application/problem_catalog.py
# Catálogo declarativo de los métodos que NO parten de una función f(x):
# Estadística, Investigación de Operaciones y los problemas de Optimización
# (combinatoria, punto interior, cuántica). Cada entrada define su formulario,
# un ejemplo precargado y la función que lo resuelve; app.ui.problem_panel
# dibuja el formulario sin conocer ningún método en particular.
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.domain.inventory import loader as inv_loader
from app.operations_research import assignment as asg
from app.operations_research import dynamic_programming as dp
from app.operations_research import linear_programming as lp
from app.operations_research import transportation as tr
from app.optimization import advanced_metaheuristics as amh
from app.optimization import meta_extra as mex
from app.optimization import combinatorial as comb
from app.optimization import interior_point as ip
from app.optimization import quantum as qc
from app.statistics import descriptive as desc
from app.statistics import chebyshev as cheb
from app.statistics import inferential as inf

SECTION_STATS = "Estadística"
SECTION_OR = "Investigación de Operaciones"
SECTION_OPT = "Optimización"
SECTIONS = [SECTION_STATS, SECTION_OR, SECTION_OPT]
# Marca el campo "fx" de una sesión cuyo planteamiento son datos, no una f(x).
DATA_PREFIX = "Datos del problema:"


@dataclass
class Field:
    key: str
    label: str
    kind: str = "line"  # line | text | choice | file
    default: str = ""
    options: Sequence[str] = ()


@dataclass
class ProblemSpec:
    label: str
    section: str
    group: str
    info: str
    fields: List[Field]
    solve: Callable[[Dict[str, str]], Dict[str, Any]]


def style_figure(fig) -> None:
    """Aplica el tema oscuro de la aplicación a todos los ejes de una figura."""
    fg = "#c8d8f0"
    for ax in fig.axes:
        ax.set_facecolor("#0a1228")
        ax.tick_params(colors=fg)
        ax.title.set_color("#ffffff")
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        for spine in ax.spines.values():
            spine.set_color("#1a2e58")
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor("#0a1228")
            for text in leg.get_texts():
                text.set_color(fg)


# ── Lectura de entradas ──────────────────────────────────────────────────────
_SPLIT = re.compile(r"[,;\s]+")
_CONSTRAINT = re.compile(r"(<=|>=|≤|≥|=)")


def floats(text: str, name: str = "los datos") -> List[float]:
    parts = [p for p in _SPLIT.split(text.strip()) if p]
    if not parts:
        raise ValueError(f"Falta ingresar {name}.")
    try:
        return [float(p) for p in parts]
    except ValueError as ex:
        raise ValueError(f"Hay un valor no numérico en {name}. "
                         "Usa punto decimal y separa con comas.") from ex


def integer(text: str, name: str) -> int:
    try:
        value = float(text.strip())
    except ValueError as ex:
        raise ValueError(f"«{name}» debe ser un número entero.") from ex
    if int(value) != value:
        raise ValueError(f"«{name}» debe ser un número entero.")
    return int(value)


def number(text: str, name: str) -> float:
    try:
        return float(text.strip())
    except ValueError as ex:
        raise ValueError(f"«{name}» debe ser numérico.") from ex


def matrix(text: str, name: str = "la matriz") -> List[List[float]]:
    rows = [floats(line, name) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Falta ingresar {name}.")
    if len({len(r) for r in rows}) != 1:
        raise ValueError(f"Todas las filas de {name} deben tener la misma longitud.")
    return rows


def constraints(text: str) -> Tuple[List[List[float]], List[str], List[float]]:
    A, signs, b = [], [], []
    for line in filter(str.strip, text.splitlines()):
        parts = _CONSTRAINT.split(line)
        if len(parts) != 3:
            raise ValueError(f"Restricción inválida: «{line.strip()}». Ejemplo: 1, 2 <= 10")
        A.append(floats(parts[0], "los coeficientes"))
        signs.append({"≤": "<=", "≥": ">="}.get(parts[1], parts[1]))
        b.append(floats(parts[2], "el lado derecho")[0])
    if not A:
        raise ValueError("Ingresa al menos una restricción.")
    if len({len(r) for r in A}) != 1:
        raise ValueError("Todas las restricciones deben tener la misma cantidad de coeficientes.")
    return A, signs, b


def bounds(text: str, n: int):
    chunks = [c for c in re.split(r"[;\n]", text) if c.strip()]
    if len(chunks) != n:
        raise ValueError(f"Indica una cota «inferior, superior» por variable ({n} en total), "
                         "separadas por punto y coma.")
    lb, ub = [], []
    for c in chunks:
        p = [v.strip().lower() for v in c.split(",")]
        if len(p) != 2:
            raise ValueError(f"Cota inválida: «{c.strip()}». Ejemplo: 0, 4")
        lb.append(number(p[0], "cota inferior"))
        ub.append(float("inf") if p[1] in ("inf", "∞") else number(p[1], "cota superior"))
    return lb, ub


def graph(text: str) -> Tuple[int, List[Tuple[int, int, float]]]:
    edges = []
    for line in filter(str.strip, text.splitlines()):
        p = floats(line, "las aristas")
        if len(p) not in (2, 3) or int(p[0]) != p[0] or int(p[1]) != p[1]:
            raise ValueError(f"Arista inválida: «{line.strip()}». Formato: nodo nodo peso")
        edges.append((int(p[0]), int(p[1]), p[2] if len(p) == 3 else 1.0))
    if not edges:
        raise ValueError("Ingresa al menos una arista.")
    return max(max(u, v) for u, v, _ in edges) + 1, edges


def arcs(text: str):
    out = []
    for line in filter(str.strip, text.splitlines()):
        p = line.replace(",", " ").split()
        if len(p) != 3:
            raise ValueError(f"Arco inválido: «{line.strip()}». Formato: A B 4")
        out.append((p[0], p[1], number(p[2], "costo del arco")))
    return out


_SENSE = {"Maximizar": "max", "Minimizar": "min"}


def _lp_fields(sense: str, c: str, rows: str, extra: Sequence[Field] = ()) -> List[Field]:
    return [Field("sentido", "Objetivo:", "choice", sense, tuple(_SENSE)),
            Field("c", "Coeficientes de Z (c1, c2, …):", "line", c),
            Field("restricciones", "Restricciones (coeficientes  signo  LD):", "text", rows),
            *extra]


def _lp_read(v: Dict[str, str]):
    A, s, b = constraints(v["restricciones"])
    return floats(v["c"], "los coeficientes de Z"), A, s, b, _SENSE[v["sentido"]]


def _lp_solver(method: str, bounded: bool = False):
    def run(v: Dict[str, str]):
        c, A, s, b, sense = _lp_read(v)
        lb, ub = bounds(v["cotas"], len(c)) if bounded else (None, None)
        return lp.solve_lp(c, A, s, b, sense, method, lb, ub)
    return run


def _ip(v):
    c, A, s, b, sense = _lp_read(v)
    return ip.interior_point(c, A, s, b, sense)


def _graphical(v):
    c, A, s, b, sense = _lp_read(v)
    return lp.graphical(c, A, s, b, sense)


# ── Textos de apoyo ──────────────────────────────────────────────────────────
_T_COMPLEX = ("P: resolubles en tiempo polinomial (PL, asignación, ruta más corta). "
              "NP: una solución se verifica en tiempo polinomial. "
              "NP-duro: al menos tan difíciles como todo NP (Máximo Corte, viajante, mochila).")
_T_DATA = "12, 15, 14, 10, 18, 20, 13, 16, 15, 17, 22, 14, 15, 19, 11"
_T_GRAPH = "0 1 1\n1 2 1\n2 3 1\n3 0 1\n0 2 1"
_T_TRANSPORT = dict(oferta="15, 25, 10", demanda="5, 15, 15, 15",
                    costos="10 2 20 11\n12 7 9 20\n4 14 16 18")
_F_TRANSPORT = [Field("oferta", "Oferta por origen:", "line", _T_TRANSPORT["oferta"]),
                Field("demanda", "Demanda por destino:", "line", _T_TRANSPORT["demanda"]),
                Field("costos", "Costos (una fila por origen):", "text", _T_TRANSPORT["costos"])]
_F_SEED = Field("semilla", "Semilla aleatoria:", "line", "42")


def _transport(method: str):
    def run(v):
        return tr.solve_transport(floats(v["oferta"], "la oferta"), floats(v["demanda"], "la demanda"),
                                  matrix(v["costos"], "los costos"), method)
    return run


def _qaoa(v):
    n, e = graph(v["aristas"])
    return qc.qaoa(n, e, integer(v["capas"], "Capas"), integer(v["semilla"], "Semilla"))


def _annealing(v):
    n, e = graph(v["aristas"])
    return qc.quantum_annealing(n, e, number(v["tiempo"], "Tiempo"), integer(v["pasos"], "Pasos"))


_SCHED_ALGS = ("Recocido Simulado", "Algoritmo Genético",
               "Iterated Greedy (NEH, solo flow shop)", "Comparar todos")


def _scheduling(v):
    flow = v["tipo"] == "Flow shop"
    alg = {"Algoritmo Genético": "Algoritmo Genético" if flow else "Algoritmo Genético (POX)",
           "Iterated Greedy (NEH, solo flow shop)": "Iterated Greedy (NEH)"}.get(
        v["algoritmo"], v["algoritmo"])
    if not flow and alg == "Iterated Greedy (NEH)":
        raise ValueError("Iterated Greedy (NEH) es solo para flow shop; en job shop usa el "
                         "Algoritmo Genético, el Recocido Simulado o «Comparar todos».")
    return mex.solve_scheduling(v["tipo"], v["datos"], alg, integer(v["iter"], "Iteraciones"),
                                integer(v["pob"], "Población"), integer(v["semilla"], "Semilla"))


def _maxcut(v):
    return comb.max_cut(*graph(v["aristas"]))


SPECS: List[ProblemSpec] = [
    # ── Estadística ──────────────────────────────────────────────────────────
    ProblemSpec("Estadística Descriptiva", SECTION_STATS, "Estadística Descriptiva",
                "Medidas de tendencia central, dispersión y forma, con histograma y caja.",
                [Field("datos", "Datos:", "text", _T_DATA)],
                lambda v: desc.descriptive(floats(v["datos"]))),
    ProblemSpec("Intervalo de Confianza", SECTION_STATS, "Estadística Inferencial",
                "Intervalo para la media con varianza desconocida (distribución t).",
                [Field("datos", "Datos:", "text", _T_DATA),
                 Field("nivel", "Nivel de confianza:", "line", "0.95")],
                lambda v: inf.confidence_interval(floats(v["datos"]), number(v["nivel"], "Nivel"))),
    ProblemSpec("Prueba t de Hipótesis", SECTION_STATS, "Estadística Inferencial",
                "Prueba t para la media de una muestra.",
                [Field("datos", "Datos:", "text", _T_DATA),
                 Field("mu0", "Media bajo H0 (μ0):", "line", "14"),
                 Field("alt", "Hipótesis alterna:", "choice", "Bilateral (≠)",
                       ("Bilateral (≠)", "Mayor (>)", "Menor (<)")),
                 Field("alpha", "Significancia α:", "line", "0.05")],
                lambda v: inf.t_test(floats(v["datos"]), number(v["mu0"], "μ0"), v["alt"],
                                     number(v["alpha"], "α"))),
    ProblemSpec("Regresión Lineal", SECTION_STATS, "Estadística Inferencial",
                "Mínimos cuadrados, simple o múltiple. La última columna es y.",
                [Field("datos", "Datos (x1, …, y por fila):", "text",
                       "1, 2.1\n2, 3.9\n3, 6.2\n4, 7.8\n5, 10.1\n6, 12.2")],
                lambda v: inf.regression(matrix(v["datos"]))),
    ProblemSpec("Simulación de Monte Carlo", SECTION_STATS, "Estadística Inferencial",
                "Estima E[expresión]. Distribuciones: uniforme(a,b), normal(m,s), "
                "triangular(a,c,b), exponencial(media), lognormal(m,s), bernoulli(p).",
                [Field("expr", "Expresión:", "line", "4*(x**2 + y**2 <= 1)"),
                 Field("vars", "Variables aleatorias:", "text", "x ~ uniforme(0, 1)\ny ~ uniforme(0, 1)"),
                 Field("n", "Simulaciones:", "line", "100000"), _F_SEED],
                lambda v: inf.monte_carlo(v["expr"], v["vars"], integer(v["n"], "Simulaciones"),
                                          integer(v["semilla"], "Semilla"))),

    ProblemSpec("Desigualdad de Chebyshev", SECTION_STATS, "Probabilidad",
                "P(|X − μ| < kσ) ≥ 1 − 1/k² para k > 1 y cualquier distribución. Ingresa μ y σ "
                "(o σ², o una muestra) y k, o los límites a, b del intervalo para deducir k.",
                [Field("modo", "Datos disponibles:", "choice", cheb.MODE_SIGMA, cheb.MODES),
                 Field("mu", "Media μ:", "line", "50"),
                 Field("sigma", "Desviación estándar σ:", "line", "10"),
                 Field("var", "Varianza σ² (si eliges ese modo):", "line", ""),
                 Field("datos", "Datos de la muestra (si eliges ese modo):", "text", ""),
                 Field("tipo_var", "Varianza de la muestra:", "choice", cheb.VAR_SAMPLE,
                       cheb.VAR_TYPES),
                 Field("k", "k (desviaciones estándar):", "line", "2"),
                 Field("limites", "O límites del intervalo a, b (opcional):", "line", "")],
                lambda v: cheb.chebyshev(v["modo"], v.get("mu", ""), v.get("sigma", ""),
                                         v.get("var", ""), v.get("datos", ""),
                                         v.get("tipo_var", cheb.VAR_SAMPLE), v.get("k", ""),
                                         v.get("limites", ""))),

    # ── Investigación de Operaciones: programación lineal ────────────────────
    ProblemSpec("PL: Método Gráfico", SECTION_OR, "Programación Lineal",
                "Solo 2 variables: enumera los vértices y dibuja la región factible.",
                _lp_fields("Maximizar", "3, 5", "1, 0 <= 4\n0, 2 <= 12\n3, 2 <= 18"), _graphical),
    ProblemSpec("PL: Simplex", SECTION_OR, "Programación Lineal",
                "Simplex tabular con restricciones ≤ y lado derecho no negativo.",
                _lp_fields("Maximizar", "3, 5", "1, 0 <= 4\n0, 2 <= 12\n3, 2 <= 18"), _lp_solver("simplex")),
    ProblemSpec("PL: Simplex Acotado (cotas en variables)", SECTION_OR, "Programación Lineal",
                "Variables con cotas l ≤ x ≤ u sin agregarlas como filas. Usa inf si no hay cota.",
                _lp_fields("Maximizar", "3, 2", "1, 1 <= 4",
                           [Field("cotas", "Cotas (inf, sup; …):", "line", "0, 3; 0, 2")]),
                _lp_solver("big_m", bounded=True)),
    ProblemSpec("PL: Simplex No Acotado", SECTION_OR, "Programación Lineal",
                "Detecta que Z crece sin límite e informa la dirección de crecimiento.",
                _lp_fields("Maximizar", "2, 1", "1, -1 <= 10\n2, 0 <= 40"), _lp_solver("big_m")),
    ProblemSpec("PL: Degenerado", SECTION_OR, "Programación Lineal",
                "Reporta pivotes con θ = 0 y aplica la regla de Bland para evitar ciclos.",
                _lp_fields("Maximizar", "3, 9", "1, 4 <= 8\n1, 2 <= 4"), _lp_solver("big_m")),
    ProblemSpec("PL: Big M", SECTION_OR, "Programación Lineal",
                "Penaliza las variables artificiales con M simbólica (se muestra como aM+b).",
                _lp_fields("Minimizar", "4, 1", "3, 1 = 3\n4, 3 >= 6\n1, 2 <= 4"), _lp_solver("big_m")),
    ProblemSpec("PL: Dos Fases (variables artificiales)", SECTION_OR, "Programación Lineal",
                "La Fase 1 minimiza las variables artificiales; la Fase 2 optimiza Z.",
                _lp_fields("Minimizar", "0.4, 0.5", "0.3, 0.1 <= 2.7\n0.5, 0.5 = 6\n0.6, 0.4 >= 6"),
                _lp_solver("two_phase")),

    # ── Transporte y asignación ──────────────────────────────────────────────
    ProblemSpec("Transporte: Esquina Noroeste", SECTION_OR, "Transporte y Asignación",
                "Solución inicial factible. Balancea con origen o destino ficticio.",
                _F_TRANSPORT, _transport("northwest")),
    ProblemSpec("Transporte: Método de Vogel", SECTION_OR, "Transporte y Asignación",
                "Solución inicial por mayores penalizaciones.", _F_TRANSPORT, _transport("vogel")),
    ProblemSpec("Transporte: Simplex de Transporte", SECTION_OR, "Transporte y Asignación",
                "Parte de Vogel y mejora con potenciales u, v (MODI) hasta el óptimo.",
                _F_TRANSPORT, _transport("modi")),
    ProblemSpec("Asignación: Método Húngaro", SECTION_OR, "Transporte y Asignación",
                "Asigna agentes a tareas uno a uno. Acepta matrices rectangulares.",
                [Field("sentido", "Objetivo:", "choice", "Minimizar", tuple(_SENSE)),
                 Field("costos", "Costos (fila = agente):", "text", "15 10 9\n9 15 10\n10 12 8")],
                lambda v: asg.hungarian(matrix(v["costos"]), _SENSE[v["sentido"]])),

    # ── Programación dinámica ────────────────────────────────────────────────
    ProblemSpec("PD: Ruta Más Corta por Etapas", SECTION_OR, "Programación Dinámica",
                "Recursión hacia atrás sobre una red acíclica. Cada línea: origen destino costo.",
                [Field("arcos", "Arcos:", "text",
                       "A B 2\nA C 4\nA D 3\nB E 7\nB F 4\nC E 3\nC F 2\nD F 5\nE G 1\nF G 4"),
                 Field("ini", "Nodo inicial:", "line", "A"), Field("fin", "Nodo final:", "line", "G")],
                lambda v: dp.shortest_path(arcs(v["arcos"]), v["ini"].strip(), v["fin"].strip())),
    ProblemSpec("PD: Mochila (asignación de recursos)", SECTION_OR, "Programación Dinámica",
                "Tabla f(etapa, capacidad) con pesos enteros.",
                [Field("valores", "Valores:", "line", "60, 100, 120"),
                 Field("pesos", "Pesos (enteros):", "line", "10, 20, 30"),
                 Field("cap", "Capacidad:", "line", "50")],
                lambda v: dp.knapsack_dp(floats(v["valores"]), floats(v["pesos"]),
                                         integer(v["cap"], "Capacidad"))),
    ProblemSpec("PD: Reemplazo de Equipos", SECTION_OR, "Reemplazo y Mantenimiento",
                "Decide cuándo cambiar el equipo: compra + mantenimiento − salvamento.",
                [Field("t", "Horizonte (años):", "line", "4"),
                 Field("precio", "Precio del equipo nuevo:", "line", "1000"),
                 Field("mant", "Mantenimiento por año de uso:", "line", "60, 80, 120, 200"),
                 Field("salv", "Salvamento según años de uso:", "line", "800, 600, 500, 400")],
                lambda v: dp.equipment_replacement(integer(v["t"], "Horizonte"), number(v["precio"], "Precio"),
                                                   floats(v["mant"]), floats(v["salv"]))),
    ProblemSpec("PD: Mantenimiento de Equipos", SECTION_OR, "Reemplazo y Mantenimiento",
                "Compara mantenimiento preventivo cada n periodos contra solo correctivo.",
                [Field("n", "Número de máquinas:", "line", "50"),
                 Field("p", "P(falla en el periodo k):", "line", "0.05, 0.10, 0.15, 0.30, 0.40"),
                 Field("cp", "Costo preventivo por máquina:", "line", "20"),
                 Field("cf", "Costo por falla:", "line", "100")],
                lambda v: dp.maintenance_policy(integer(v["n"], "Máquinas"), floats(v["p"]),
                                                number(v["cp"], "Costo preventivo"), number(v["cf"], "Costo por falla"))),
    ProblemSpec("Inventario: Cargar Archivo (CSV/Excel)", SECTION_OR, "Inventario",
                "Selecciona un .csv/.xlsx o edita los datos. Obligatorias: producto, demanda, "
                "costo_pedido, costo_mantener. Opcionales: precio, demanda_diaria, desv_diaria, "
                "lead_time, nivel_servicio.",
                [Field("archivo", "Archivo:", "file", ""),
                 Field("datos", "O pega los datos:", "text", inv_loader.SAMPLE)],
                lambda v: inv_loader.load_and_analyze(v["archivo"].strip(), v["datos"])),

    # ── Optimización: PL avanzada ────────────────────────────────────────────
    ProblemSpec("PL: Método de Punto Interior", SECTION_OPT, "Programación Lineal Avanzada",
                "Primal-dual predictor-corrector de Mehrotra.",
                _lp_fields("Maximizar", "3, 5", "1, 0 <= 4\n0, 2 <= 12\n3, 2 <= 18"), _ip),
    ProblemSpec("PL: Problema de Klee-Minty", SECTION_OPT, "Programación Lineal Avanzada",
                "Caso peor del Simplex de Dantzig: 2ⁿ − 1 pivotes. Se compara con punto interior.",
                [Field("n", "Dimensión máxima n (2 a 12):", "line", "8")],
                lambda v: ip.klee_minty(integer(v["n"], "n"))),

    # ── Optimización combinatoria ────────────────────────────────────────────
    ProblemSpec("Complejidad Computacional", SECTION_OPT, "Optimización Combinatoria", _T_COMPLEX,
                [Field("n", "Tamaños n:", "line", "10, 20, 30, 50, 100"),
                 Field("v", "Operaciones por segundo:", "line", "1e9")],
                lambda v: comb.complexity(floats(v["n"]), number(v["v"], "Velocidad"))),
    ProblemSpec("Combinatoria: Mochila 0/1", SECTION_OPT, "Optimización Combinatoria",
                "Ramificación y acotamiento con cota fraccionaria.",
                [Field("valores", "Valores:", "line", "60, 100, 120"),
                 Field("pesos", "Pesos:", "line", "10, 20, 30"), Field("cap", "Capacidad:", "line", "50")],
                lambda v: comb.knapsack_bb(floats(v["valores"]), floats(v["pesos"]), number(v["cap"], "Capacidad"))),
    ProblemSpec("Combinatoria: Viajante (TSP)", SECTION_OPT, "Optimización Combinatoria",
                "Vecino más cercano, 2-opt y exacto (Held-Karp) hasta 13 ciudades.",
                [Field("xy", "Ciudades (x, y por línea):", "text", "0, 0\n0, 2\n2, 2\n2, 0\n1, 3\n3, 1")],
                lambda v: comb.tsp(matrix(v["xy"], "las ciudades"))),
    ProblemSpec("Problema de Máximo Corte", SECTION_OPT, "Optimización Combinatoria",
                "Divide los nodos en dos grupos maximizando el peso entre grupos. NP-duro.",
                [Field("aristas", "Aristas (nodo nodo peso):", "text", _T_GRAPH)], _maxcut),

    # ── Metaheurísticas avanzadas ────────────────────────────────────────────
    ProblemSpec("Metaheurística: Problema General", SECTION_OPT, "Metaheurísticas avanzadas",
                "Cualquier f(x) de n variables, minimizar o maximizar, con variables reales, "
                "enteras o binarias y restricciones ≤, ≥ o =. Variables: una por línea "
                "(x1: -5, 5 · x2: entera 0, 10 · x3: binaria · todas: -10, 10).",
                [Field("sentido", "Objetivo:", "choice", "Minimizar", tuple(_SENSE)),
                 Field("fx", "Función objetivo f(x):", "line",
                       "(x1 - 2)**2 + (x2 - 3)**2 + x3**2"),
                 Field("variables", "Variables y cotas (una por línea):", "text",
                       "x1: -5, 5\nx2: -5, 5\nx3: entera -3, 3"),
                 Field("restricciones", "Restricciones (una por línea, opcional):", "text",
                       "x1 + x2 <= 4"),
                 Field("algoritmo", "Algoritmo:", "choice", "Comparar todos", amh.ALGORITHMS),
                 Field("iter", "Iteraciones:", "line", "200"),
                 Field("pob", "Población / vecinos por iteración:", "line", "30"),
                 Field("runs", "Ejecuciones independientes (semillas distintas):", "line", "1"),
                 Field("pulido", "Pulido local final:", "choice", "Sí", ("Sí", "No")), _F_SEED],
                lambda v: amh.solve_general(v["sentido"], v["fx"], v["variables"],
                                            v.get("restricciones", ""), v["algoritmo"],
                                            integer(v["iter"], "Iteraciones"),
                                            integer(v["pob"], "Población"),
                                            integer(v["semilla"], "Semilla"),
                                            integer(v.get("runs") or "1", "Ejecuciones"),
                                            v.get("pulido", "Sí") == "Sí")),
    ProblemSpec("Metaheurística: Multiobjetivo (NSGA-II)", SECTION_OPT, "Metaheurísticas avanzadas",
                "Frente de Pareto de 2 a 6 objetivos (uno por línea, «min:» o «max:»), con las "
                "mismas variables y restricciones que el problema general.",
                [Field("objetivos", "Objetivos (uno por línea):", "text",
                       "min: x1\nmin: (1 + 9*x2)*(1 - sqrt(x1/(1 + 9*x2)))"),
                 Field("variables", "Variables y cotas (una por línea):", "text", "todas: 0, 1"),
                 Field("restricciones", "Restricciones (una por línea, opcional):", "text", ""),
                 Field("gen", "Generaciones:", "line", "150"),
                 Field("pob", "Población:", "line", "60"), _F_SEED],
                lambda v: mex.solve_multiobjective(v["objetivos"], v["variables"],
                                                   v.get("restricciones", ""),
                                                   integer(v["gen"], "Generaciones"),
                                                   integer(v["pob"], "Población"),
                                                   integer(v["semilla"], "Semilla"))),
    ProblemSpec("Metaheurística: Ruteo de Vehículos (CVRP)", SECTION_OPT, "Metaheurísticas avanzadas",
                "Reparto desde un depósito con vehículos de capacidad limitada. Clientes: «x, y, "
                "demanda» por línea. Vehículos = 0 significa flota sin límite.",
                [Field("deposito", "Depósito (x, y):", "line", "50, 50"),
                 Field("clientes", "Clientes (x, y, demanda por línea):", "text",
                       "\n".join(f"{a:g}, {b:g}, {d}" for (a, b), d in zip(
                           amh.random_cities(12, 7), (4, 7, 3, 6, 5, 8, 2, 6, 4, 7, 3, 5)))),
                 Field("cap", "Capacidad por vehículo:", "line", "20"),
                 Field("veh", "Número máximo de vehículos (0 = sin límite):", "line", "0"),
                 Field("algoritmo", "Algoritmo:", "choice", "Comparar todos", mex.VRP_ALGORITHMS),
                 Field("iter", "Iteraciones:", "line", "60"),
                 Field("pob", "Población / movimientos:", "line", "20"), _F_SEED],
                lambda v: mex.solve_vrp(floats(v["deposito"], "el depósito"),
                                        matrix(v["clientes"], "los clientes"),
                                        number(v["cap"], "Capacidad"),
                                        integer(v.get("veh") or "0", "Vehículos"),
                                        v["algoritmo"], integer(v["iter"], "Iteraciones"),
                                        integer(v["pob"], "Población"),
                                        integer(v["semilla"], "Semilla"))),
    ProblemSpec("Metaheurística: Programación de Tareas", SECTION_OPT, "Metaheurísticas avanzadas",
                "Minimiza el makespan. Flow shop: una fila por trabajo con su tiempo en cada "
                "máquina. Job shop: una fila por trabajo con pares «máquina tiempo» en orden "
                "(máquinas desde 1). Ejemplo: FT06 de Fisher-Thompson (óptimo 55).",
                [Field("tipo", "Tipo de taller:", "choice", "Job shop", ("Job shop", "Flow shop")),
                 Field("datos", "Trabajos (uno por línea):", "text", mex.FT06),
                 Field("algoritmo", "Algoritmo:", "choice", "Comparar todos", _SCHED_ALGS),
                 Field("iter", "Iteraciones:", "line", "200"),
                 Field("pob", "Población / movimientos:", "line", "30"), _F_SEED],
                _scheduling),
    ProblemSpec("Metaheurística: Viajante (TSP grande)", SECTION_OPT, "Metaheurísticas avanzadas",
                "Rutas de hasta 300 ciudades con Colonia de Hormigas, Recocido Simulado (2-opt) "
                "o Algoritmo Genético (cruce OX). Escribe las ciudades (x, y por línea) o indica "
                "cuántas generar al azar.",
                [Field("xy", "Ciudades (x, y por línea):", "text",
                       "\n".join(f"{a:g}, {b:g}" for a, b in amh.random_cities(30, 42))),
                 Field("aleatorias", "O genera N ciudades al azar:", "line", ""),
                 Field("algoritmo", "Algoritmo:", "choice", "Colonia de Hormigas (ACO)",
                       amh.TSP_ALGORITHMS),
                 Field("iter", "Iteraciones:", "line", "100"),
                 Field("agentes", "Hormigas / población / movimientos:", "line", "20"), _F_SEED],
                lambda v: amh.solve_tsp(
                    matrix(v["xy"], "las ciudades") if v.get("xy", "").strip()
                    else amh.random_cities(integer(v.get("aleatorias", ""), "N ciudades"),
                                           integer(v["semilla"], "Semilla")),
                    v["algoritmo"], integer(v["iter"], "Iteraciones"),
                    integer(v["agentes"], "Agentes"), integer(v["semilla"], "Semilla"))),

    # ── Optimización cuántica ────────────────────────────────────────────────
    ProblemSpec("Cuántica: Cambio de Paradigma", SECTION_OPT, "Optimización Cuántica",
                "Compara el modelo clásico con el cuántico y el costo de buscar en N elementos.",
                [Field("q", "Qubits a comparar:", "line", "4, 8, 16, 32, 50")],
                lambda v: qc.paradigm_shift(floats(v["q"]))),
    ProblemSpec("Cuántica: Fenómeno Cuántico", SECTION_OPT, "Optimización Cuántica",
                "Simula superposición, entrelazamiento o interferencia y mide el resultado.",
                [Field("f", "Fenómeno:", "choice", "Superposición",
                       ("Superposición", "Entrelazamiento", "Interferencia")),
                 Field("shots", "Mediciones:", "line", "1000"), _F_SEED],
                lambda v: qc.phenomenon(v["f"], integer(v["shots"], "Mediciones"), integer(v["semilla"], "Semilla"))),
    ProblemSpec("Cuántica: Bit y Qubit", SECTION_OPT, "Optimización Cuántica",
                "|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}·sen(θ/2)|1⟩, con su punto en la esfera de Bloch.",
                [Field("t", "θ (grados):", "line", "60"), Field("p", "φ (grados):", "line", "45"),
                 Field("shots", "Mediciones:", "line", "1000"), _F_SEED],
                lambda v: qc.bit_vs_qubit(number(v["t"], "θ"), number(v["p"], "φ"),
                                          integer(v["shots"], "Mediciones"), integer(v["semilla"], "Semilla"))),
    ProblemSpec("Cuántica: QAOA (Máximo Corte)", SECTION_OPT, "Optimización Cuántica",
                "Algoritmo variacional híbrido. Simulado en NumPy, hasta 12 nodos.",
                [Field("aristas", "Aristas (nodo nodo peso):", "text", _T_GRAPH),
                 Field("capas", "Capas p:", "line", "3"), _F_SEED], _qaoa),
    ProblemSpec("Cuántica: VQE", SECTION_OPT, "Optimización Cuántica",
                "Estima la energía mínima de un Hamiltoniano. Formato: coeficiente PAULI.",
                [Field("h", "Hamiltoniano:", "text", "-1 ZZ\n-0.5 XI\n-0.5 IX"),
                 Field("capas", "Capas:", "line", "2"), _F_SEED],
                lambda v: qc.vqe(v["h"], integer(v["capas"], "Capas"), integer(v["semilla"], "Semilla"))),
    ProblemSpec("Cuántica: Grover", SECTION_OPT, "Optimización Cuántica",
                "Búsqueda no estructurada con ventaja cuadrática.",
                [Field("n", "Qubits:", "line", "3"), Field("m", "Estados marcados:", "line", "101")],
                lambda v: qc.grover(integer(v["n"], "Qubits"), [s for s in _SPLIT.split(v["m"].strip()) if s])),
    ProblemSpec("Cuántica: Recocido Cuántico", SECTION_OPT, "Optimización Cuántica",
                "Evolución adiabática simulada H(s) = −(1−s)·ΣX − s·C sobre Máximo Corte.",
                [Field("aristas", "Aristas (nodo nodo peso):", "text", _T_GRAPH),
                 Field("tiempo", "Tiempo total T:", "line", "50"),
                 Field("pasos", "Pasos:", "line", "1000")], _annealing),
]

SPEC_BY_LABEL: Dict[str, ProblemSpec] = {s.label: s for s in SPECS}


def is_problem_method(label: str) -> bool:
    return label in SPEC_BY_LABEL


def groups_for(section: str) -> List[Tuple[str, List[str]]]:
    """Grupos del catálogo para una sección, conservando el orden de SPECS."""
    out: Dict[str, List[str]] = {}
    for s in SPECS:
        if s.section == section:
            out.setdefault(s.group, []).append(s.label)
    return list(out.items())


def solve(label: str, values: Dict[str, str]) -> Dict[str, Any]:
    return SPEC_BY_LABEL[label].solve(values)


def defaults(label: str) -> Dict[str, str]:
    return {f.key: f.default for f in SPEC_BY_LABEL[label].fields}


def describe_input(label: str, values: Dict[str, str], limit: Optional[int] = 220) -> str:
    """Texto corto de la entrada, para historial y reportes."""
    parts = [DATA_PREFIX.rstrip()]
    for f in SPEC_BY_LABEL[label].fields:
        txt = " / ".join(s.strip() for s in values.get(f.key, "").splitlines() if s.strip())
        if txt:
            parts.append(f"{f.label.rstrip(':')}: {txt}")
    text = parts[0] + " " + "; ".join(parts[1:])
    return text if limit is None or len(text) <= limit else text[:limit - 1] + "…"
