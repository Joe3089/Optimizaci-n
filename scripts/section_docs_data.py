# -*- coding: utf-8 -*-
"""
section_docs_data.py — Documentación técnica de los métodos de Estadística,
Investigación de Operaciones y Optimización basados en datos
(app.application.problem_catalog).

El campo «ejemplo» no se escribe a mano: se calcula ejecutando el ejemplo
precargado de cada método, así que el documento siempre muestra lo que la
aplicación devuelve de verdad. Lo usan generate_docs.py y
generate_test_functions_doc.py.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.application.problem_catalog import (  # noqa: E402
    SPEC_BY_LABEL, SPECS, defaults, describe_input, solve,
)

_KEYS = ("formula", "descripcion", "requisitos", "convergencia", "ventajas", "desventajas",
         "cuando_usar", "cuando_no_usar", "compatibles", "incompatibles")

_LP_IN = "Coeficientes numéricos de Z y de cada restricción (≤, ≥ o =)."

_TEXT = {
    "Estadística Descriptiva": (
        "x̄ = Σxᵢ/n,  s² = Σ(xᵢ−x̄)²/(n−1),  RIC = Q3 − Q1,  atípico si x < Q1−1.5·RIC o x > Q3+1.5·RIC",
        "Resume una muestra: tendencia central, dispersión, forma (asimetría y curtosis corregidas) y valores atípicos.",
        "Al menos 2 datos numéricos; asimetría desde 3 y curtosis desde 4 datos.",
        "Cálculo directo, sin iteraciones.",
        "Visión rápida y completa de los datos, con histograma y diagrama de caja.",
        "No permite generalizar a la población: para eso se usa la sección inferencial.",
        "Primer análisis de cualquier conjunto de datos.",
        "Cuando se necesita una conclusión sobre la población con nivel de confianza.",
        "Cualquier lista de números reales.", "Datos categóricos o no numéricos."),
    "Intervalo de Confianza": (
        "x̄ ± t_{(1+nivel)/2, n−1} · s/√n",
        "Rango que contiene la media poblacional con el nivel de confianza indicado, con varianza desconocida.",
        "Al menos 2 datos; nivel entre 0 y 1. Supone normalidad o muestra grande.",
        "Cálculo directo con la distribución t de Student.",
        "Cuantifica la incertidumbre de la media muestral.",
        "Sensible a valores atípicos en muestras pequeñas.",
        "Estimar una media con su margen de error.",
        "Datos muy asimétricos con pocas observaciones.",
        "Muestras numéricas independientes.", "Series con autocorrelación fuerte."),
    "Prueba t de Hipótesis": (
        "t = (x̄ − μ0) / (s/√n),  p según H1 (bilateral, mayor o menor)",
        "Contrasta si la media poblacional difiere de un valor μ0.",
        "Al menos 2 datos; α entre 0 y 1. Supone normalidad aproximada.",
        "Cálculo directo; decide rechazar H0 si p < α.",
        "Decisión objetiva con nivel de significancia explícito.",
        "No rechazar H0 no prueba que sea cierta.",
        "Verificar si un proceso cumple un valor objetivo.",
        "Comparar dos grupos (requiere una prueba de dos muestras).",
        "Una muestra numérica.", "Datos pareados o de dos grupos."),
    "Regresión Lineal": (
        "β̂ = (XᵀX)⁻¹Xᵀy,  R² = 1 − SSE/SST,  t_j = β̂_j/ee(β̂_j)",
        "Ajusta y = β0 + β1x1 + … por mínimos cuadrados, con pruebas t por coeficiente y F global.",
        "Más observaciones que coeficientes; variables independientes no colineales; y no constante.",
        "Solución cerrada de las ecuaciones normales.",
        "Interpretable; admite regresión simple y múltiple.",
        "Solo capta relaciones lineales; sensible a atípicos.",
        "Explicar o predecir una variable a partir de otras.",
        "Relaciones claramente no lineales o respuesta binaria.",
        "Tabla numérica con la respuesta y en la última columna.", "Variables colineales (se rechaza con aviso)."),
    "Simulación de Monte Carlo": (
        "E[g(X)] ≈ (1/N)·Σ g(xᵢ),  error estándar = s/√N",
        "Estima el valor esperado de una expresión de variables aleatorias por muestreo, con intervalo de confianza y gráfica de convergencia.",
        "Expresión con aritmética, comparaciones y funciones básicas; variables con distribución uniforme, normal, triangular, exponencial, lognormal o Bernoulli.",
        "El error decrece como 1/√N.",
        "Resuelve problemas sin solución analítica; la semilla hace reproducible el resultado.",
        "Convergencia lenta: 100 veces más simulaciones para 10 veces menos error.",
        "Riesgo, integrales, proyectos con incertidumbre.",
        "Cuando existe una fórmula cerrada sencilla.",
        "Expresiones numéricas evaluables con NumPy.", "Código arbitrario (se bloquea por seguridad)."),
    "Desigualdad de Chebyshev": (
        "P(|X − μ| < kσ) ≥ 1 − 1/k²  (k > 1);  % mínimo = (1 − 1/k²)·100;  intervalo μ ± kσ",
        "Cota inferior de la proporción de valores a menos de k desviaciones estándar de la media, válida para cualquier distribución.",
        "μ y σ (o σ², o datos de una muestra con varianza muestral o poblacional) y k > 1, o límites a < μ < b.",
        "Cálculo cerrado; con límites asimétricos usa la distancia menor (cota conservadora).",
        "No supone normalidad; sirve con cualquier distribución de media y varianza finitas.",
        "Es una cota mínima, a menudo holgada; no da la probabilidad exacta.",
        "Garantías mínimas de calidad o de dispersión sin conocer la distribución.",
        "Cuando se conoce la distribución (se calcula la probabilidad exacta).",
        "Media, desviación estándar o varianza, o una muestra numérica.", "k ≤ 1, σ ≤ 0 o σ² < 0 (se rechazan con aviso)."),
    "Metaheurística: Problema General": (
        "mín/máx f(x), x ∈ [l, u] ⊂ ℝⁿ (con xⱼ entera o binaria), s.a. gᵢ(x) ≤ 0, hⱼ(x) = 0; comparación por reglas de factibilidad de Deb",
        "Motor metaheurístico para problemas complejos: f no lineal, multimodal o no diferenciable de n variables, mixta entera/binaria y con restricciones. Algoritmos: Evolución Diferencial autoadaptativa (jDE), CMA-ES con reinicios IPOP, PSO con inercia decreciente, Algoritmo Genético (BLX + mutación gaussiana, elitismo), Recocido Simulado (basin hopping en problemas continuos sin restricciones; intercambio de réplicas en los demás) y Búsqueda Tabú por coordenadas; «Comparar todos» los ejecuta con la misma semilla y elige el mejor. Ejecuciones independientes con estadística y pulido local final.",
        "Cotas finitas por variable; f y restricciones evaluables con el parser seguro (sin código).",
        "Estocástica: no garantiza el óptimo global, pero las reglas de Deb priorizan la factibilidad y el mejor punto se conserva siempre (elitismo).",
        "No requiere derivadas ni convexidad; maneja enteros, binarios y restricciones; escala a cientos de variables.",
        "Resultado aproximado y dependiente de la semilla; las igualdades se cumplen con tolerancia 1e-4.",
        "Funciones multimodales (Rastrigin, Ackley), diseño mixto-entero, mochila, asignación con penalizaciones.",
        "Problemas lineales o convexos pequeños, donde un método exacto es más preciso y rápido.",
        "Expresiones matemáticas de n variables y restricciones con ≤, ≥ o =.",
        "Variables sin cotas o expresiones con código (se rechazan)."),
    "Metaheurística: Multiobjetivo (NSGA-II)": (
        "mín/máx (f₁(x), …, fₘ(x)), 2 ≤ m ≤ 6, con cotas y restricciones; ordenamiento no dominado + distancia de hacinamiento",
        "Algoritmo genético NSGA-II (Deb et al., 2002) con cruce SBX, mutación polinomial, elitismo y dominancia restringida. Devuelve el frente de Pareto, su evolución por generación y el hipervolumen (2 objetivos).",
        "Objetivos y restricciones evaluables con el parser seguro; cotas finitas por variable.",
        "Verificado: en ZDT1 (30 variables) el frente obtenido queda a 0.002 del frente analítico.",
        "Encuentra muchas soluciones de compromiso en una sola corrida, sin pesos ni escalarización.",
        "Con más de 3 objetivos el frente se vuelve difícil de visualizar.",
        "Diseño con objetivos en conflicto: costo vs. calidad, riesgo vs. rendimiento.",
        "Cuando existe un único objetivo o se conocen los pesos (usar el problema general).",
        "Objetivos de n variables (reales, enteras o binarias) con restricciones opcionales.",
        "Objetivos con código o sin cotas (se rechazan)."),
    "Metaheurística: Ruteo de Vehículos (CVRP)": (
        "mín Σₖ (costo fijoₖ + costo por distanciaₖ·longitudₖ) s.a. capacidad por tipo de vehículo, ventanas horarias [eᵢ, lᵢ] con servicio sᵢ, horario del depósito y cantidad de vehículos por tipo",
        "Representa la solución como una ruta gigante que el procedimiento Split de Prins divide de forma óptima en viajes factibles; GA memético (cruce OX, 2-opt) y Recocido Simulado (inversión, intercambio, inserción), con 2-opt dentro de cada ruta.",
        "Depósito y clientes con coordenadas; demandas positivas ≤ capacidad.",
        "Verificado contra el óptimo por fuerza bruta en instancias pequeñas.",
        "Escala a cientos de clientes; maneja límite de flota.",
        "No certifica optimalidad en instancias grandes.",
        "Reparto, recolección, logística de última milla.",
        "Velocidades distintas a 1 (el tiempo de viaje se toma igual a la distancia).",
        "Clientes «x, y, demanda», capacidad y número máximo de vehículos.",
        "Demandas mayores que la capacidad o mayores que la flota (se rechazan con aviso)."),
    "Metaheurística: Programación de Tareas": (
        "mín Cmax, Σ Tⱼ, Σ wⱼTⱼ o Lmax (Tⱼ = máx(0, Cⱼ − dⱼ)) en flow shop o job shop, con preparación dependiente de la secuencia",
        "Flow shop: NEH + Iterated Greedy (Ruiz y Stützle), Recocido Simulado (inserción) y GA. Job shop: GA con cruce POX sobre secuencias con repetición y Recocido Simulado; decodificación en programas semiactivos. Muestra el diagrama de Gantt.",
        "Tiempos de proceso no negativos; en job shop, el orden de máquinas de cada trabajo.",
        "Verificado: alcanza el óptimo 55 del benchmark FT06 y el óptimo exacto en flow shop pequeños.",
        "Resuelve instancias clásicas NP-difíciles en segundos.",
        "Para tardanza se requieren las fechas de entrega de todos los trabajos.",
        "Talleres, líneas de producción, planificación de cómputo.",
        "Cuando hay restricciones de recursos adicionales no modeladas.",
        "Flow shop: matriz trabajos × máquinas. Job shop: pares «máquina tiempo» por trabajo.",
        "Tiempos negativos o máquinas numeradas desde 0 (se rechazan con aviso)."),
    "Metaheurística: Viajante (TSP grande)": (
        "mín Σ d(πₖ, πₖ₊₁) sobre las permutaciones π de las ciudades (ruta cerrada)",
        "Resuelve el viajante de hasta 300 ciudades con Colonia de Hormigas (Max-Min, rutas en paralelo), Recocido Simulado con movimientos 2-opt o Algoritmo Genético memético (cruce OX + 2-opt); todos terminan con una mejora local 2-opt.",
        "Coordenadas (x, y) de al menos 4 ciudades; distancia euclidiana.",
        "Heurística: en instancias pequeñas suele alcanzar el óptimo exacto (verificado contra Held-Karp); en grandes da rutas de buena calidad en segundos.",
        "Escala mucho más allá del método exacto (13 ciudades).",
        "No certifica optimalidad en instancias grandes.",
        "Rutas de reparto, recolección y secuenciación.",
        "Cuando hay pocas ciudades y se necesita el óptimo certificado (usar el TSP exacto).",
        "Ciudades «x, y» por línea o N ciudades generadas al azar con semilla.",
        "Distancias asimétricas o con restricciones de ventana horaria."),
    "PL: Método Gráfico": (
        "Óptimo en un vértice de la región factible: evaluar Z en cada intersección factible",
        "Enumera los vértices de la región factible, evalúa Z y dibuja restricciones, región y recta óptima.",
        "Exactamente 2 variables no negativas.",
        "Exacto; enumera todas las intersecciones.",
        "Muy didáctico: muestra la geometría del problema.",
        "Solo sirve para 2 variables.",
        "Enseñar y verificar problemas pequeños.",
        "Problemas de 3 o más variables.",
        _LP_IN, "Más de 2 variables (se rechaza con aviso)."),
    "PL: Simplex": (
        "Entra la columna con menor cⱼ−zⱼ (Dantzig); sale la fila con menor razón LD/aᵢⱼ > 0",
        "Simplex tabular con base inicial de holguras; muestra cada tableau y la fila Cj−Zj.",
        "Restricciones ≤ con lado derecho no negativo.",
        "Finito con la regla de Bland ante pivotes degenerados; en el peor caso exponencial (ver Klee-Minty).",
        "Muestra el razonamiento completo iteración por iteración.",
        "No arranca con restricciones ≥ o =.",
        "Problemas de maximización con recursos limitados.",
        "Restricciones ≥ o = (usar Big M o Dos Fases).",
        _LP_IN, "Restricciones ≥ o = (se rechaza con aviso y alternativa)."),
    "PL: Simplex Acotado (cotas en variables)": (
        "l ≤ x ≤ u: si x_j alcanza u_j se sustituye x_j = u_j − x'_j sin agregar filas",
        "Técnica de variables acotadas: las cotas superiores se manejan en la prueba de razón, no como restricciones.",
        "Cotas inferiores finitas; cotas superiores numéricas o inf.",
        "Igual al Simplex; cada iteración considera tres razones (básica a 0, básica a su cota, entrante a su cota).",
        "Tableau más pequeño que agregando una fila por cota.",
        "Más casos que seguir a mano.",
        "Variables con capacidad máxima (producción, horas, inventario).",
        "Sin cotas superiores (equivale al Simplex normal).",
        _LP_IN + " Cotas «inferior, superior» por variable.", "Variables libres sin cota inferior."),
    "PL: Simplex No Acotado": (
        "No acotado si la columna entrante no tiene ningún aᵢⱼ > 0",
        "Detecta que Z crece sin límite y reporta la dirección en la que crece.",
        "Mismos datos que el Simplex.",
        "Termina en cuanto ninguna fila limita a la variable entrante.",
        "Identifica modelos mal formulados (falta una restricción).",
        "No hay solución óptima que reportar.",
        "Depurar modelos cuyo resultado no tiene sentido.",
        "Problemas con región factible acotada (dan óptimo normal).",
        _LP_IN, "—"),
    "PL: Degenerado": (
        "Pivote degenerado: θ = 0, la base cambia pero Z no",
        "Reporta los pivotes degenerados y las soluciones con variables básicas en cero; aplica la regla de Bland para no ciclar.",
        "Mismos datos que el Simplex.",
        "Con Bland tras cada pivote degenerado el método termina (probado con el ejemplo de ciclado de Beale).",
        "Explica por qué el Simplex puede estancarse.",
        "La regla de Bland puede requerir más iteraciones.",
        "Restricciones redundantes que pasan por un mismo vértice.",
        "—",
        _LP_IN, "—"),
    "PL: Big M": (
        "Min Z + M·Σaᵢ, con M simbólica: Cj−Zj se muestra como aM + b",
        "Añade variables artificiales penalizadas con M y resuelve en una sola fase.",
        "Cualquier combinación de restricciones ≤, ≥ e =.",
        "M se trata de forma simbólica (comparación lexicográfica), sin errores numéricos por un M enorme.",
        "Una sola fase; detecta infactibilidad si una artificial queda positiva.",
        "Tableau con expresiones en M, más difícil de leer.",
        "Problemas con restricciones ≥ o =.",
        "Solo restricciones ≤ (basta el Simplex).",
        _LP_IN, "—"),
    "PL: Dos Fases (variables artificiales)": (
        "Fase 1: min Σaᵢ; si es 0, Fase 2: optimizar Z sin artificiales",
        "Busca primero una base factible y luego optimiza; elimina restricciones redundantes.",
        "Cualquier combinación de restricciones ≤, ≥ e =.",
        "Infactible si el mínimo de la Fase 1 es positivo.",
        "Numéricamente más estable que Big M; separa factibilidad de optimalidad.",
        "Dos corridas del Simplex.",
        "Problemas con restricciones = o ≥.",
        "Solo restricciones ≤.",
        _LP_IN, "—"),
    "Transporte: Esquina Noroeste": (
        "Asignar min(oferta, demanda) empezando en la celda superior izquierda",
        "Solución inicial factible rápida; balancea el problema con un origen o destino ficticio.",
        "Oferta, demanda y costos no negativos; matriz de orígenes × destinos.",
        "m + n − 1 asignaciones; no considera costos.",
        "Muy simple de aplicar.",
        "Suele quedar lejos del óptimo.",
        "Punto de partida para el Simplex de transporte.",
        "Cuando se necesita el costo mínimo.",
        "Oferta, demanda y matriz de costos.", "Cantidades negativas."),
    "Transporte: Método de Vogel": (
        "Penalización = diferencia de los dos costos menores por fila y columna; asignar en la mayor",
        "Solución inicial que considera costos de oportunidad.",
        "Iguales al de esquina noroeste.",
        "Normalmente muy cerca del óptimo.",
        "Buena calidad inicial con poco cálculo.",
        "No garantiza el óptimo.",
        "Solución inicial para MODI.",
        "Cuando se requiere el óptimo exacto.",
        "Oferta, demanda y matriz de costos.", "Cantidades negativas."),
    "Transporte: Simplex de Transporte": (
        "uᵢ + vⱼ = cᵢⱼ en celdas básicas; entra la celda con menor cᵢⱼ − uᵢ − vⱼ < 0 por un ciclo",
        "Método MODI: parte de Vogel, completa una base de m+n−1 celdas y mejora hasta el óptimo.",
        "Iguales al de esquina noroeste.",
        "Óptimo cuando todos los costos reducidos son ≥ 0; maneja degeneración con celdas en cero.",
        "Óptimo exacto, verificado contra programación lineal.",
        "Más pasos que los métodos iniciales.",
        "Distribución de mínimo costo.",
        "—",
        "Oferta, demanda y matriz de costos.", "Cantidades negativas."),
    "Asignación: Método Húngaro": (
        "Reducir filas y columnas; cubrir ceros con el mínimo de líneas; ajustar con el menor valor descubierto",
        "Asigna agentes a tareas uno a uno con costo mínimo o beneficio máximo.",
        "Matriz numérica; si es rectangular se completa con ceros.",
        "Termina cuando el número de líneas iguala el tamaño (teorema de König).",
        "Exacto y en tiempo polinomial.",
        "Solo asignación uno a uno.",
        "Personal–tareas, máquinas–trabajos.",
        "Un agente para varias tareas (usar transporte).",
        "Matriz de costos o beneficios.", "—"),
    "PD: Ruta Más Corta por Etapas": (
        "f(n) = min_{(n,m)} { c(n,m) + f(m) },  f(destino) = 0",
        "Recursión hacia atrás sobre una red por etapas.",
        "Red acíclica con arcos «origen destino costo».",
        "Exacto; cada nodo se evalúa una vez.",
        "Muestra el principio de optimalidad de Bellman.",
        "No admite ciclos (se rechazan con aviso).",
        "Planeación por etapas, rutas.",
        "Redes con ciclos.",
        "Arcos con costos numéricos.", "Redes con ciclos."),
    "PD: Mochila (asignación de recursos)": (
        "f(i, c) = max{ f(i−1, c), vᵢ + f(i−1, c−wᵢ) }",
        "Tabla de programación dinámica para elegir objetos sin exceder la capacidad.",
        "Pesos y capacidad enteros positivos (capacidad ≤ 200 000).",
        "Exacto en O(n·capacidad).",
        "Exacto y rápido con capacidades moderadas.",
        "Pseudo-polinomial: crece con la capacidad.",
        "Presupuestos, carga de vehículos.",
        "Pesos fraccionarios (usar la versión de ramificación y acotamiento).",
        "Valores reales, pesos enteros.", "Pesos no enteros."),
    "PD: Reemplazo de Equipos": (
        "f(t) = min_k { P + Σ_{a<k} m_a − R_k + f(t+k) },  f(T) = 0",
        "Decide cuántos años usar cada equipo antes de reemplazarlo en el horizonte T.",
        "Horizonte entero; mantenimiento y salvamento con la misma longitud (vida máxima).",
        "Exacto por recursión hacia atrás.",
        "Incluye compra, mantenimiento y salvamento.",
        "Supone costos conocidos y sin inflación.",
        "Flotas, maquinaria, equipos de cómputo.",
        "Costos inciertos (usar Monte Carlo).",
        "Costos numéricos por año de uso.", "—"),
    "PD: Mantenimiento de Equipos": (
        "B(n) = N·Σ_{k≤n} p_k + Σ_{k<n} B(n−k)·p_k;  costo(n) = [C_p·N + C_f·B(n)]/n",
        "Compara el mantenimiento preventivo cada n periodos con el solo correctivo, cuyo costo es C_f·N/E[T].",
        "Probabilidades de falla no negativas que sumen 1; costos no negativos.",
        "Exacto por recursión de fallas esperadas.",
        "Decisión cuantitativa entre preventivo y correctivo.",
        "Supone reemplazo inmediato y probabilidades estables.",
        "Grupos de máquinas o componentes similares.",
        "Equipos únicos con fallas sin historial.",
        "Probabilidades por periodo y costos.", "Probabilidades que no suman 1."),
    "Inventario: Cargar Archivo (CSV/Excel)": (
        "Q* = √(2DS/H),  costo = √(2DSH),  ROP = d̄·L + z·σ_d·√L,  ABC por valor anual acumulado (80 % / 95 %)",
        "Lee un inventario completo y calcula por artículo EOQ, pedidos al año, costo anual, stock de seguridad, punto de reorden y clase ABC.",
        "Columnas producto, demanda, costo_pedido y costo_mantener. Opcionales: precio, demanda_diaria, desv_diaria, lead_time y nivel_servicio.",
        "Cálculo directo por artículo.",
        "Analiza cientos de artículos de una vez; acepta encabezados con o sin acentos.",
        "Usa el modelo EOQ clásico para cada artículo.",
        "Revisar la política de compras de todo un almacén.",
        "Un solo artículo con descuentos o faltantes (usar los modelos de inventario individuales).",
        "Archivos .csv, .txt o .xlsx, o datos pegados.", "Valores no numéricos o no positivos (se indica la fila)."),
    "PL: Método de Punto Interior": (
        "Predictor-corrector de Mehrotra: resolver A·D·Aᵀ·Δy = r con D = X·S⁻¹;  σ = (μ_aff/μ)³",
        "Recorre el interior de la región factible siguiendo la trayectoria central hasta el óptimo.",
        "Problema de PL en cualquier forma; la región debe tener interior.",
        "Polinomial; típicamente entre 5 y 30 iteraciones, casi independiente del tamaño.",
        "Muy eficiente en problemas grandes; inmune al caso Klee-Minty.",
        "Da un punto cercano al vértice, no un tableau.",
        "PL de gran tamaño.",
        "Cuando se necesita el análisis por tableau.",
        _LP_IN, "Problemas infactibles o no acotados (se reporta que no convergió)."),
    "PL: Problema de Klee-Minty": (
        "max Σ 2^{n−j}·x_j  s.a.  Σ_{j<i} 2^{i−j+1}·x_j + x_i ≤ 5^i",
        "Caso peor del Simplex de Dantzig: recorre los 2ⁿ vértices del cubo deformado. Se compara con punto interior.",
        "n entre 2 y 12.",
        "Simplex: exactamente 2ⁿ − 1 pivotes; punto interior: pocas iteraciones.",
        "Demuestra la diferencia entre complejidad exponencial y polinomial.",
        "Es un caso construido, no un problema práctico.",
        "Enseñar complejidad de algoritmos de PL.",
        "—", "n entero.", "n fuera de 2 a 12."),
    "Complejidad Computacional": (
        "Tiempo ≈ operaciones / velocidad, para O(log n), O(n), O(n log n), O(n²), O(n³), O(2ⁿ) y O(n!)",
        "Compara el crecimiento de cada clase de complejidad y su tiempo real; resume las clases P, NP y NP-duro.",
        "Tamaños enteros positivos y velocidad en operaciones por segundo.",
        "Cálculo directo en escala logarítmica, sin desbordamiento.",
        "Muestra por qué los problemas NP-duros necesitan heurísticas.",
        "Es una estimación del orden, no una medición.",
        "Justificar la elección de un método exacto o heurístico.",
        "—", "Tamaños enteros.", "—"),
    "Combinatoria: Mochila 0/1": (
        "Cota superior = valor actual + relajación fraccionaria; podar si cota ≤ mejor valor",
        "Ramificación y acotamiento, mejor nodo primero.",
        "Pesos y capacidad positivos (pueden ser reales).",
        "Exacto; explora solo una fracción del árbol de 2ⁿ⁺¹ nodos.",
        "Admite pesos reales; muestra la poda.",
        "Puede crecer exponencialmente en el peor caso.",
        "Selección de proyectos con presupuesto.",
        "Capacidades enteras grandes con muchos objetos (usar PD).",
        "Valores y pesos numéricos.", "Pesos no positivos."),
    "Combinatoria: Viajante (TSP)": (
        "Held-Karp: C(S, k) = min_{j∈S} C(S∖{k}, j) + d(j, k)",
        "Ruta más corta que visita todas las ciudades: vecino más cercano, mejora 2-opt y exacto hasta 13 ciudades.",
        "Coordenadas (x, y); entre 3 y 60 ciudades.",
        "Exacto en O(n²·2ⁿ); las heurísticas son polinomiales.",
        "Compara la calidad de las heurísticas con el óptimo.",
        "Exacto solo para pocas ciudades.",
        "Rutas de reparto y recorridos.",
        "Más de 60 ciudades.",
        "Coordenadas numéricas.", "Matrices de distancia asimétricas."),
    "Problema de Máximo Corte": (
        "max Σ_{(i,j)∈E} w_ij·[lado(i) ≠ lado(j)]",
        "Divide los nodos en dos grupos maximizando el peso de las aristas entre grupos. Compara voraz, búsqueda local y exacto.",
        "Aristas «nodo nodo peso» con nodos numerados desde 0; hasta 22 nodos.",
        "Exacto por enumeración de 2ⁿ particiones (vectorizada).",
        "Problema de referencia de la optimización cuántica (QAOA).",
        "NP-duro: el exacto no escala.",
        "Agrupamiento, diseño de circuitos, redes.",
        "Grafos de más de 22 nodos.",
        "Grafos no dirigidos con pesos.", "Lazos (arista de un nodo a sí mismo)."),
    "Cuántica: Cambio de Paradigma": (
        "Búsqueda: clásico N/2 consultas,  Grover (π/4)·√N",
        "Contrasta el modelo clásico con el cuántico: información, operaciones, medición y ventaja demostrada.",
        "Número de qubits entre 1 y 60.",
        "Cálculo directo.",
        "Muestra con números el alcance real de la ventaja cuántica.",
        "Comparación teórica de consultas, no de hardware.",
        "Introducción a la computación cuántica.",
        "—", "Qubits enteros.", "—"),
    "Cuántica: Fenómeno Cuántico": (
        "Superposición H|0⟩;  entrelazamiento CNOT·(H⊗I)|00⟩;  interferencia H·H|0⟩ = |0⟩",
        "Simula el fenómeno elegido y lo mide muchas veces para comparar frecuencias con probabilidades.",
        "Mediciones entre 1 y 1 000 000.",
        "Exacto (vector de estado) más muestreo multinomial.",
        "Hace visibles los tres fenómenos en que se basan los algoritmos cuánticos.",
        "Simulado en computador clásico.",
        "Enseñanza.", "—", "—", "—"),
    "Cuántica: Bit y Qubit": (
        "|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}·sen(θ/2)|1⟩;  Bloch (sen θ cos φ, sen θ sen φ, cos θ)",
        "Construye un qubit, calcula sus probabilidades y su punto en la esfera de Bloch, y lo compara con un bit clásico.",
        "θ entre 0 y 180 grados.",
        "Exacto más muestreo.",
        "Intuición geométrica del qubit.",
        "Un solo qubit.",
        "Enseñanza.", "—", "—", "—"),
    "Cuántica: QAOA (Máximo Corte)": (
        "|γ,β⟩ = Π_p e^{−iβ_p·ΣX}·e^{−iγ_p·C}|+⟩ⁿ;  maximizar ⟨C⟩ con COBYLA (12 arranques)",
        "Algoritmo variacional híbrido cuántico-clásico aplicado a Máximo Corte.",
        "Grafo de hasta 12 nodos con pesos positivos; capas p entre 1 y 6.",
        "Optimización clásica de 2p parámetros; la razón ⟨C⟩/óptimo crece con p (0.75 en el anillo con p = 1).",
        "Algoritmo cuántico de optimización más estudiado.",
        "Simulación con costo 2ⁿ; parámetros con óptimos locales.",
        "Problemas QUBO o de Ising.",
        "Grafos grandes.", "Grafos con pesos positivos.", "Más de 12 nodos."),
    "Cuántica: VQE": (
        "E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ ≥ E₀;  ansatz de capas Ry–Rz con CNOT en cadena",
        "Estima la energía mínima de un Hamiltoniano escrito como suma de cadenas de Pauli y la compara con la exacta.",
        "Términos «coeficiente PAULI» de igual longitud (hasta 8 qubits).",
        "L-BFGS-B con 6 arranques; el principio variacional garantiza E(θ) ≥ E₀.",
        "Base de la química cuántica y de problemas de Ising.",
        "Puede quedar en mínimos locales si hay pocas capas.",
        "Energía de estado base, problemas de Ising.",
        "Más de 8 qubits.", "Operadores I, X, Y, Z.", "Otros símbolos."),
    "Cuántica: Grover": (
        "θ = arcsen√(M/N);  P(éxito tras k iteraciones) = sen²((2k+1)θ);  k* = ⌊(π/4)·√(N/M)⌋",
        "Búsqueda no estructurada con ventaja cuadrática; compara la probabilidad simulada con la teórica.",
        "Entre 1 y 16 qubits; estados marcados como cadenas de bits.",
        "Exacto (vector de estado).",
        "Ventaja cuadrática demostrada.",
        "Iterar de más reduce la probabilidad de éxito.",
        "Búsqueda en espacios sin estructura.",
        "Marcar todo el espacio.", "Cadenas de n bits.", "Cadenas de otra longitud."),
    "Cuántica: Recocido Cuántico": (
        "H(s) = −(1−s)·ΣX − s·C,  s = t/T;  evolución con división de Trotter simétrica",
        "Simula la evolución adiabática desde el estado base de ΣX hasta el de Máximo Corte.",
        "Grafo de hasta 12 nodos; T > 0; entre 10 y 20 000 pasos.",
        "Con T grande la probabilidad de terminar en el óptimo tiende a 1 (teorema adiabático).",
        "Muestra el principio de los computadores de recocido.",
        "Requiere T grande si la brecha espectral es pequeña.",
        "Problemas QUBO o de Ising.",
        "Grafos grandes.", "Grafos con pesos.", "Más de 12 nodos."),
}

SECTION_CATEGORIES = []
SECTION_METHODS = []
for _spec in SPECS:
    cat = f"{_spec.section} — {_spec.group}"
    if cat not in SECTION_CATEGORIES:
        SECTION_CATEGORIES.append(cat)
    entry = dict(nombre=_spec.label, categoria=cat, **dict(zip(_KEYS, _TEXT[_spec.label])))
    SECTION_METHODS.append(entry)

missing = [s.label for s in SPECS if s.label not in _TEXT]
if missing:
    raise RuntimeError(f"Falta documentar: {missing}")


def with_examples(methods):
    """Completa «ejemplo» ejecutando el ejemplo precargado de cada método."""
    out = []
    for m in methods:
        vals = defaults(m["nombre"])
        res = solve(m["nombre"], vals)
        entry = dict(m)
        entry["ejemplo"] = f"{describe_input(m['nombre'], vals, None)} → {res['resumen']}"
        out.append(entry)
    return out


def section_tests():
    """Casos de prueba para el documento de funciones de prueba."""
    tests = []
    for s in SPECS:
        vals = defaults(s.label)
        tests.append(dict(funcion=s.label, seccion=f"{s.section} — {s.group}",
                          parametros=describe_input(s.label, vals, None),
                          resultado=solve(s.label, vals)["resumen"]))
    return tests


assert all(m["nombre"] in SPEC_BY_LABEL for m in SECTION_METHODS)
