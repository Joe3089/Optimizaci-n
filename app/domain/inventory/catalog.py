# app/domain/inventory/catalog.py
# Catálogo documental de las funciones de inventario — misma estructura de
# diccionario que scripts/method_docs_data.py (una sola fuente de verdad),
# para que el reporte maestro de funciones y generate_docs.py puedan listar
# "Funciones para Problemas de Inventario" sin duplicar contenido ni tocar
# su lógica de render.
from __future__ import annotations

INVENTORY_CATEGORY = "Funciones para Problemas de Inventario"

# (clave interna del modelo, etiqueta EXACTA mostrada en el combo de método
# de main_window / usada por el system prompt de la IA para recomendar un
# modelo por nombre) — fuente única compartida por app.ui.inventory_panel
# (arma el combo) y app.infrastructure.ai_assistant (detecta cuándo la IA
# nombró un modelo, para activar el panel automáticamente).
INVENTORY_MODEL_LABELS = [
    ("eoq_clasico",        "Inventario: EOQ Clásico"),
    ("eoq_backorders",     "Inventario: EOQ con Faltantes"),
    ("eoq_descuentos",     "Inventario: EOQ con Descuentos"),
    ("rop_probabilistico", "Inventario: Punto de Reorden (ROP)"),
]
INVENTORY_LABEL_TO_KEY = {lbl: key for key, lbl in INVENTORY_MODEL_LABELS}
INVENTORY_METHOD_LABELS = [lbl for _, lbl in INVENTORY_MODEL_LABELS]

INVENTORY_FUNCTIONS = [
    dict(
        nombre="EOQ Clásico (Cantidad Económica de Pedido)",
        categoria=INVENTORY_CATEGORY,
        formula="C(Q) = (D/Q)·S + (Q/2)·H,  Q* = √(2DS/H)",
        descripcion=("Modelo determinista clásico de Harris-Wilson: balancea "
                     "el costo de ordenar (D/Q·S, decrece con Q) contra el "
                     "costo de mantener inventario (Q/2·H, crece con Q)."),
        requisitos=("Variables: Q > 0 (cantidad de pedido). Parámetros: "
                    "D (demanda anual), S (costo fijo por pedido), H (costo "
                    "de mantenimiento por unidad-año). Sin restricciones "
                    "activas ni faltantes."),
        convergencia=("Función convexa 1D con mínimo único — cualquier "
                      "método de búsqueda 1D o basado en gradiente converge "
                      "al Q* analítico."),
        ventajas="Fórmula cerrada conocida para validar la convergencia numérica; problema convexo simple.",
        desventajas="No modela faltantes, descuentos por volumen ni demanda variable.",
        cuando_usar="Gestión de inventario con demanda y costos constantes y conocidos.",
        cuando_no_usar="Cuando hay descuentos por volumen, faltantes permitidos o demanda incierta (usar los otros 3 modelos).",
        compatibles=("Búsqueda Local, Fibonacci, Sección Áurea, Sección "
                    "Dorada, Goldstein, Newton-Raphson, Grad. Conjugado, "
                    "Armijo, Wolfe, y metaheurísticas (SA/PSO/GA)."),
        incompatibles="Los 6 métodos MD (no hay restricción g(x)) y los 14 métodos multiobjetivo (una sola función de costo).",
        ejemplo="D=1000, S=50, H=2 → Q*≈223.6, Costo*≈447.2",
    ),
    dict(
        nombre="EOQ con Faltantes Planeados (Backorders)",
        categoria=INVENTORY_CATEGORY,
        formula="C(Q,b) = (D/Q)·S + H·(Q−b)²/(2Q) + B·b²/(2Q)",
        descripcion=("Extiende el EOQ clásico permitiendo un déficit "
                     "controlado b, penalizado con un costo de faltante B "
                     "por unidad-año."),
        requisitos=("Variables: Q > 0, 0 ≤ b ≤ Q. Parámetros: D, S, H, B "
                    "(costo de faltante). 2 variables de decisión."),
        convergencia="Función convexa en (Q,b) — óptimo analítico conocido para validar la solución numérica.",
        ventajas="Reduce el costo total frente al EOQ clásico cuando el faltante es tolerable.",
        desventajas="Requiere estimar B (costo de faltante), a menudo más difícil de medir que H.",
        cuando_usar="Cuando el negocio puede tolerar backorders (pedidos pendientes) sin perder la venta.",
        cuando_no_usar="Cuando un faltante implica pérdida de venta o cliente (usar EOQ clásico).",
        compatibles=("Armijo, Wolfe (multivariable) y los 6 métodos MD "
                    "(Penalización/Barreras/BFGS/Nelder-Mead) — todos aceptan "
                    "2 variables."),
        incompatibles="Búsqueda Local, Fibonacci, Sección Áurea/Dorada (estrictamente 1D) y los 14 métodos multiobjetivo.",
        ejemplo="D=1000, S=50, H=2, B=8 → Q*≈250.0, b*≈50.0",
    ),
    dict(
        nombre="EOQ con Descuentos por Cantidad",
        categoria=INVENTORY_CATEGORY,
        formula="C_i(Q) = (D/Q)·S + (Q/2)·(r·p_i) + D·p_i,  por tramo i",
        descripcion=("Función de costo por tramos: el precio unitario p_i "
                     "(y por tanto el costo de mantenimiento r·p_i) cambia "
                     "según el rango de cantidad pedida."),
        requisitos=("Variable: Q > 0. Parámetros: D, S, tasa de "
                    "mantenimiento r (fracción del precio), y ≥2 tramos "
                    "(cantidad_mínima, precio_unitario) ordenados."),
        convergencia=("No derivable en los puntos de quiebre entre tramos — "
                      "requiere métodos que no dependan de gradiente."),
        ventajas="Modela descuentos reales por volumen de compra.",
        desventajas="Función no suave; los métodos basados en gradiente pueden fallar o dar resultados incorrectos cerca de los quiebres.",
        cuando_usar="Cuando el proveedor ofrece precios escalonados por volumen.",
        cuando_no_usar="Cuando el precio unitario es fijo (usar EOQ clásico).",
        compatibles="Búsqueda Local y metaheurísticas (SA: Recocido Simulado, PSO: Enjambre, GA: Algoritmo Genético).",
        incompatibles=("Newton-Raphson, Goldstein, Armijo, Wolfe, Grad. "
                       "Conjugado (requieren derivadas continuas); Fibonacci "
                       "y Sección Áurea/Dorada (asumen unimodalidad, que los "
                       "saltos de precio pueden romper); los 6 métodos MD y "
                       "los 14 multiobjetivo."),
        ejemplo="D=1000, S=50, r=0.2, tramos=[(0,10),(100,9),(500,8)] → compara costo total entre tramos factibles",
    ),
    dict(
        nombre="Revisión Periódica / Punto de Reorden (ROP) Probabilístico",
        categoria=INVENTORY_CATEGORY,
        formula="ROP = μ_L + z·σ_L,  μ_L=d̄·L,  σ_L=σ_d·√L,  Φ(z)=nivel_servicio",
        descripcion=("Calcula el punto de reorden bajo demanda "
                     "probabilística: stock de seguridad = z·σ_L, donde z "
                     "es la inversa de la normal estándar evaluada en el "
                     "nivel de servicio deseado. Q* se sigue optimizando "
                     "con la función de costo del EOQ clásico."),
        requisitos=("Parámetros: d̄ (demanda diaria promedio), σ_d "
                    "(desviación estándar diaria), L (lead time en días), "
                    "nivel de servicio ∈ (0,1), más D, S, H para Q*."),
        convergencia="El ROP se calcula analíticamente (no requiere optimizador); Q* converge igual que en EOQ clásico.",
        ventajas="Incorpora incertidumbre de la demanda con un nivel de servicio objetivo explícito.",
        desventajas="Asume demanda normalmente distribuida durante el lead time.",
        cuando_usar="Cuando la demanda es variable y se requiere un nivel de servicio (probabilidad de no quiebre de stock) explícito.",
        cuando_no_usar="Cuando la demanda es determinista y constante (usar EOQ clásico).",
        compatibles="Mismos métodos que EOQ clásico para optimizar Q (el ROP en sí es analítico, no pasa por el optimizador).",
        incompatibles="Los 6 métodos MD y los 14 multiobjetivo (misma razón que EOQ clásico).",
        ejemplo="d̄=20, σ_d=5, L=7, nivel_servicio=0.95 → z≈1.645, ROP≈171.9",
    ),
]
