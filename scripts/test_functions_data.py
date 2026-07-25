# -*- coding: utf-8 -*-
"""
test_functions_data.py — Funciones matemáticas usadas para probar los 32
métodos del Optimizador (Tarea 2/3 de la sesión de QA), con qué métodos son
compatibles y con cuáles se probó deliberadamente el bloqueo.

Cada método se prueba con (a) una función simple/básica y (b), cuando el
método soporta 2 o más variables por diseño, una función compleja no
convexa de 2+ variables (benchmarks clásicos: Rosenbrock, Himmelblau). Los
métodos que son estrictamente de 1 variable por diseño (Búsqueda Local,
Fibonacci, Sección Áurea/Dorada, y los 14 multiobjetivo) no reciben la
prueba "compleja de 2+ variables" porque la rechazan por diseño — ese
rechazo ya está verificado en INCOMPATIBLE_TESTS más abajo.

Fuente: scripts/smoke_test.py (_case_for, caso vigente en el repo) más las
pruebas manuales de incompatibilidad hechas durante la Tarea 3, y las
pruebas de funciones complejas 2D (Rosenbrock/Himmelblau) de la Tarea 4,
ejecutadas de verdad contra app.application.optimization_service.run_method
y verificadas con los valores reales devueltos (no simulados).
"""
from __future__ import annotations

# ─── Funciones usadas para ejecución exitosa (agrupadas por caso compartido) ─
TEST_FUNCTIONS = [
    dict(
        funcion="f(x) = x² − 4x + 5",
        parametros="1 variable (x), rango [0, 5], sin restricción g(x)",
        aplica_a=[
            "Búsqueda Local", "Fibonacci", "Sección Áurea", "Sección Dorada",
            "Goldstein", "Newton-Raphson", "Grad. Conjugado", "Armijo", "Wolfe",
            "SA: Recocido Simulado", "PSO: Enjambre", "GA: Algoritmo Genético",
        ],
        no_aplica_a=[
            "Los 6 métodos MD (requieren g(x) — con esta función sin restricción, "
            "la app bloquea la ejecución con mensaje y recomendación)",
            "Los 14 métodos multiobjetivo (parten de otra convención de prueba, "
            "ver función de 1 variable dedicada abajo, aunque técnicamente "
            "también son compatibles con cualquier f(x) de 1 variable)",
        ],
        resultado="Todos los 12 métodos convergen correctamente a x*≈2, f*≈1.",
    ),
    dict(
        funcion="f(x1,x2) = (x1−2)² + (x2−1)²",
        parametros="2 variables (x1, x2), g(x1,x2) = x1+x2−2, x0=[0,0]",
        aplica_a=[
            "MD: Penalización (Newton)", "MD: Barreras (Newton)",
            "MD: Pen. (BFGS)", "MD: Pes. (BFGS)", "MD: Pes. (Nelder-Mead)",
            "MD: Sum. (BFGS)",
        ],
        no_aplica_a=[
            "Búsqueda Local, Fibonacci, Sección Áurea/Dorada (métodos estrictamente "
            "1D — 2 variables no soportadas, bloqueado con mensaje)",
            "Los 14 métodos multiobjetivo (requieren f(x) de 1 sola variable)",
        ],
        resultado=(
            "Los 6 métodos convergen al óptimo analítico conocido "
            "x*=[1.5, 0.5], f*=0.5 (restricción activa x1+x2=2), cada uno por "
            "un camino de iteración distinto según su solver interno."
        ),
    ),
    dict(
        funcion="f(x) = (x−3)²",
        parametros="1 variable (x); la app genera automáticamente F(x)=[f1,f2] "
                   "con f1=f(x), f2=(x−x_centro)², α=0.5",
        aplica_a=[
            "Escalarización (Suma Ponderada)", "MO — Bisección",
            "MO — Sección Dorada", "MO — Frente de Pareto",
            "MO — Análisis Jacobiano", "MO — Lexicográfico",
            "MO — Goal Programming", "MO — ε-Constraint", "MO — ASF (Logro)",
            "MO — Chebyshev", "MO — NBI", "MO — Restricción Normal",
            "MO — Peso Adaptativo", "MO — Punto de Referencia",
        ],
        no_aplica_a=[
            "Los 6 métodos MD (esta función no tiene g(x); están pensados "
            "para f(x1,x2) con restricción)",
        ],
        resultado="Los 14 métodos resuelven correctamente el problema escalarizado.",
    ),
    dict(
        funcion="f(x,y) = (1−x)² + 100(y−x²)²  — Rosenbrock (benchmark clásico, "
                "no convexo, valle curvo estrecho)",
        parametros="2 variables (x, y), sin restricción g(x), x0=[−1, 1], "
                   "rango [−2, 2]. Óptimo conocido: (1,1), f*=0.",
        aplica_a=["Armijo", "Wolfe", "Goldstein", "Newton-Raphson",
                  "Grad. Conjugado", "SA: Recocido Simulado", "PSO: Enjambre",
                  "GA: Algoritmo Genético"],
        no_aplica_a=[
            "Búsqueda Local, Fibonacci, Sección Áurea/Dorada (estrictamente 1D)",
            "Los 6 métodos MD (necesitan g(x); ver función compleja dedicada abajo)",
            "Los 14 métodos multiobjetivo (requieren f(x) de 1 sola variable)",
        ],
        resultado=(
            "Armijo, Goldstein, Newton-Raphson y Grad. Conjugado convergen "
            "exactamente a x*=[1, 1], f*=0.0. Las 3 metaheurísticas se "
            "aproximan (SA f≈0.192, PSO f≈2.3e-7, GA f≈1.0e-4), esperado por "
            "su naturaleza estocástica. Esta prueba detectó y permitió "
            "corregir un bug real: el método Wolfe fallaba con \"función no "
            "compatible\" en cualquier función de 2+ variables al ejecutarse "
            "en una consola Windows real (cp1252) — la causa era que "
            "app/optimization/line_search/nd_symbolic.py imprimía símbolos "
            "Unicode (∇, ᵀ) no representables en esa codificación; el "
            "UnicodeEncodeError se camuflaba como incompatibilidad de "
            "función. Corregido reemplazando los símbolos por texto ASCII "
            "en los prints de diagnóstico (sin efecto funcional). Verificado "
            "tras el fix: Wolfe converge en 8 iteraciones, condición de Wolfe "
            "satisfecha (1.976563 ≤ 4.8)."
        ),
    ),
    dict(
        funcion="f(x1,x2) = (x1²+x2−11)² + (x1+x2²−7)²  — Himmelblau (no convexa, "
                "4 mínimos globales) con restricción g=x1+x2−2",
        parametros="2 variables (x1, x2), x0=[0, 0], rango [−5, 5].",
        aplica_a=["MD: Penalización (Newton)", "MD: Barreras (Newton)",
                  "MD: Pen. (BFGS)", "MD: Pes. (BFGS)", "MD: Pes. (Nelder-Mead)",
                  "MD: Sum. (BFGS)"],
        no_aplica_a=[
            "Búsqueda Local, Fibonacci, Sección Áurea/Dorada (estrictamente 1D)",
            "Los 14 métodos multiobjetivo (requieren f(x) de 1 sola variable)",
        ],
        resultado=(
            "Los 6 métodos ejecutan sin errores. Al ser Himmelblau no "
            "convexa (4 mínimos globales conocidos en el problema libre), "
            "cada método converge a un mínimo local distinto según su punto "
            "de partida y solver interno — Pen. (BFGS) y Pes. (BFGS) llegan "
            "a x≈[3.58,−1.85] con f≈0; Sum. (BFGS) a x≈[2.99, 1.97] con "
            "f≈2.98; Penalización/Barreras (Newton) quedan en otro mínimo "
            "local con gradiente ≈0. Este comportamiento es el esperado en "
            "problemas no convexos (métodos locales, no defecto) y confirma "
            "que ninguno de los 6 solvers se rompe con una f(x) genuinamente "
            "compleja de 2 variables."
        ),
    ),
]

# ─── Funciones usadas deliberadamente para verificar que el sistema BLOQUEA
#     una combinación método↔función incompatible (Tarea 3) ──────────────────
INCOMPATIBLE_TESTS = [
    dict(
        funcion="f(x,y) = x² + y²  (2 variables)",
        metodo="Búsqueda Local",
        motivo="Método estrictamente unidimensional — no acepta 2+ variables.",
        resultado_app=(
            'BLOQUEADO. Mensaje: "La función tiene 2 variables: x, y. Los '
            'métodos de Búsqueda Local y Fibonacci son métodos de búsqueda en '
            'UNA sola dimensión... Para funciones multivariable use: Armijo o '
            'Wolfe, o Métodos MD."'
        ),
    ),
    dict(
        funcion="f(x,y) = x² + y²  (2 variables)",
        metodo="Escalarización (Suma Ponderada)",
        motivo="Los métodos multiobjetivo requieren f(x) de exactamente 1 variable.",
        resultado_app=(
            'BLOQUEADO. Mensaje: "La función ingresada no es compatible con el '
            'método «Escalarización (Suma Ponderada)» (no se pudo aplicar este '
            'tipo de estudio a la expresión dada)."'
        ),
    ),
    dict(
        funcion="f(x) = Abs(x−2)  (no derivable en x=2)",
        metodo="Armijo",
        motivo="Método basado en gradiente/Newton — necesita una derivada "
               "simbólica que Sympy pueda evaluar numéricamente.",
        resultado_app=(
            'BLOQUEADO. Mensaje: "La función ingresada no es compatible con el '
            'método «Armijo» (no se pudo aplicar este tipo de estudio a la '
            'expresión dada, por ejemplo por no ser derivable donde lo '
            'requiere el método)."'
        ),
    ),
    dict(
        funcion="f(x1,x2) = (x1−2)² + (x2−1)²  sin g(x) definido",
        metodo="Los 6 métodos MD (Penalización, Barreras, Pen./Pes./Sum. BFGS, "
               "Pes. Nelder-Mead)",
        motivo="Son técnicas de penalización/barrera/suma sobre una "
               "restricción — sin g(x) el resultado sería engañoso.",
        resultado_app=(
            'BLOQUEADO antes de ejecutar. Mensaje: "El método «...» aplica '
            'penalización/barrera/suma sobre una restricción, así que '
            'necesita que definas g(x)... usa en su lugar un método '
            'multivariable sin restricción: Grad. Conjugado, Newton-Raphson, '
            'Goldstein, Armijo, Wolfe, SA, PSO o GA."'
        ),
    ),
    dict(
        funcion="Cualquier f(x) válida, rango invertido (mínimo > máximo)",
        metodo="Cualquier método (probado con GA: Algoritmo Genético)",
        motivo="Un rango de búsqueda inválido no tiene sentido para ningún método.",
        resultado_app=(
            'BLOQUEADO. Mensaje: "El valor mínimo (5.0) debe ser menor que el '
            'máximo (0.0). Corrige el rango de búsqueda."'
        ),
    ),
    dict(
        funcion="f(x) válida, tolerancia L = −1",
        metodo="Fibonacci",
        motivo="Fibonacci exige una tolerancia positiva para calcular cuántos "
               "términos de la secuencia generar.",
        resultado_app=(
            'BLOQUEADO. Mensaje: "La tolerancia L debe ser un número positivo '
            'mayor que cero."'
        ),
    ),
    dict(
        funcion="Campo f(x) vacío",
        metodo="Cualquier método (probado con Búsqueda Local)",
        motivo="No hay función que optimizar.",
        resultado_app=(
            'BLOQUEADO. Mensaje: "Por favor ingresa la función f(x). '
            'Ejemplo: -(x-3)**2 + 10"'
        ),
    ),
]

# ─── Módulo de Inventario — funciones/modelos usados para probar cada uno
#     de los 4 modelos (app.domain.inventory.models), con datos reales
#     verificados contra su solución analítica en la sesión de QA ───────────
INVENTORY_TESTS = [
    dict(
        funcion="EOQ Clásico: C(q) = (D/q)·S + (q/2)·H",
        parametros="D=1000, S=50, H=2 (demanda anual, costo de pedido, costo de mantenimiento). 1 variable: q.",
        aplica_a=["Búsqueda Local", "Fibonacci", "Sección Áurea", "Sección Dorada",
                  "Goldstein", "Newton-Raphson", "Grad. Conjugado", "Armijo", "Wolfe"],
        no_aplica_a=["Los 6 métodos MD (no hay restricción g(x))",
                     "Los 14 métodos multiobjetivo (una sola función de costo)"],
        resultado="Los métodos convergen a q*≈223.61, Costo*≈447.21 — igual al valor analítico √(2DS/H).",
    ),
    dict(
        funcion="EOQ con Faltantes Planeados: C(q,b) = (D/q)·S + H·(q−b)²/(2q) + B·b²/(2q)",
        parametros="D=1000, S=50, H=2, B=8. 2 variables (q, b); restricción real g(x)=b−q≤0.",
        aplica_a=["Armijo", "Wolfe", "MD: Penalización (Newton)", "MD: Barreras (Newton)",
                  "MD: Pen. (BFGS)", "MD: Pes. (BFGS)", "MD: Pes. (Nelder-Mead)", "MD: Sum. (BFGS)"],
        no_aplica_a=["Búsqueda Local, Fibonacci, Sección Áurea/Dorada (estrictamente 1D)",
                     "Los 14 métodos multiobjetivo"],
        resultado="Armijo y los 6 métodos MD convergen a q*≈250.0, b*≈50.0, Costo*≈400.0 — igual al valor analítico.",
    ),
    dict(
        funcion="EOQ con Descuentos por Cantidad: C_i(q) por tramos de precio",
        parametros="D=1000, S=50, tasa=0.2, tramos=[(0,10),(100,9),(500,8)]. 1 variable: q, función Piecewise.",
        aplica_a=["Búsqueda Local", "SA: Recocido Simulado", "PSO: Enjambre", "GA: Algoritmo Genético"],
        no_aplica_a=["Newton-Raphson, Goldstein, Armijo, Wolfe, Grad. Conjugado (no derivable en los quiebres)",
                     "Fibonacci, Sección Áurea/Dorada (asumen unimodalidad)",
                     "Los 6 métodos MD y los 14 multiobjetivo"],
        resultado="Búsqueda Local converge a q*≈500.0 (tramo de mayor volumen), Costo*≈8500.0 — óptimo global entre los 3 tramos.",
    ),
    dict(
        funcion="Punto de Reorden (ROP) Probabilístico",
        parametros="d̄=20, σ_d=5, lead time=7 días, nivel de servicio=0.95, D=1000, S=50, H=2.",
        aplica_a=["Búsqueda Local", "Fibonacci", "Sección Áurea", "Sección Dorada",
                  "Goldstein", "Newton-Raphson", "Grad. Conjugado", "Armijo", "Wolfe"],
        no_aplica_a=["Los 6 métodos MD", "Los 14 métodos multiobjetivo"],
        resultado="z≈1.645, ROP≈161.76 (calculado analíticamente); q* converge igual que en EOQ Clásico.",
    ),
]

INVENTORY_INCOMPATIBLE_TESTS = [
    dict(
        funcion="EOQ con Faltantes Planeados (2 variables: q, b)",
        metodo="Búsqueda Local",
        motivo="Método estrictamente unidimensional — este modelo tiene 2 variables de decisión.",
        resultado_app="BLOQUEADO por diseño: el combo de algoritmo del panel de Inventario solo "
                      "lista los métodos compatibles con el modelo elegido (Armijo, Wolfe y los 6 "
                      "MD) — Búsqueda Local ni siquiera aparece como opción seleccionable.",
    ),
    dict(
        funcion="EOQ Clásico con D = −10",
        metodo="Cualquiera",
        motivo="Una demanda anual negativa no tiene sentido físico.",
        resultado_app='BLOQUEADO antes de construir la función. Mensaje: "El parámetro '
                      "'Demanda anual (D)' debe ser un número mayor que 0\".",
    ),
    dict(
        funcion="ROP con nivel de servicio = 1.5",
        metodo="Cualquiera",
        motivo="Un nivel de servicio debe ser una probabilidad estrictamente entre 0 y 1.",
        resultado_app='BLOQUEADO. Mensaje: "El nivel de servicio debe estar estrictamente entre '
                      '0 y 1 (p.ej. 0.95 = 95%)".',
    ),
    dict(
        funcion="EOQ con Descuentos, tramos desordenados [(100,9), (0,10)]",
        metodo="Cualquiera",
        motivo="Las cantidades mínimas de los tramos de precio deben ser crecientes.",
        resultado_app='BLOQUEADO. Mensaje: "Las cantidades mínimas de los tramos de precio deben '
                      'ser estrictamente crecientes y sin repetir".',
    ),
]
