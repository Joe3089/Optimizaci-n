# -*- coding: utf-8 -*-
"""
test_functions_data.py — Funciones matemáticas usadas para probar los 32
métodos del Optimizador (Tarea 2/3 de la sesión de QA), con qué métodos son
compatibles y con cuáles se probó deliberadamente el bloqueo.

Fuente: scripts/smoke_test.py (_case_for, caso vigente en el repo) más las
pruebas manuales de incompatibilidad hechas durante la Tarea 3.
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
