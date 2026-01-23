import numpy as np

def metodo_fibonacci(funcion, a, b, tolerancia=1e-5, max_n=200, return_history=False):
    """
    Búsqueda por Fibonacci (minimización 1D).

    Retorna:
      x_opt, f_opt, iteraciones, tipo, distancia [, history, limit_reached, target]

    - limit_reached: True si NO se pudo alcanzar la tolerancia porque se llegó a max_n
    - target: valor objetivo (L/tolerancia) usado para construir la secuencia de Fibonacci
    """

    a = float(a)
    b = float(b)
    tolerancia = float(tolerancia)

    if b <= a:
        raise ValueError("b debe ser mayor que a")
    if tolerancia <= 0:
        raise ValueError("La tolerancia debe ser mayor que 0")

    L = b - a
    target = L / max(tolerancia, 1e-15)

    # Construir Fibonacci hasta alcanzar target o max_n
    fib = [1, 1]
    limit_reached = False

    # fib[-1] corresponde a F_k; queremos F_k >= target
    while len(fib) < max_n and fib[-1] < target:
        fib.append(fib[-1] + fib[-2])

    if fib[-1] < target:
        # No logramos la secuencia necesaria para esa tolerancia
        limit_reached = True

    N = len(fib) - 1  # índice final

    # Si el intervalo es muy pequeño o N no es suficiente para iterar, devolvemos centro
    if N < 3:
        x_opt = (a + b) / 2.0
        f_opt = funcion(x_opt)
        entorno = np.linspace(x_opt - 1e-3, x_opt + 1e-3, 5)
        vals = [funcion(xx) for xx in entorno]
        tipo = "mínimo" if funcion(x_opt) == min(vals) else "máximo"
        distancia = abs(b - a)

        out = (x_opt, f_opt, 0, tipo, distancia)
        if return_history:
            return out + ([], limit_reached, target)
        return out

    # Inicialización
    k = 1
    x1 = a + (fib[N - 2] / fib[N]) * (b - a)
    x2 = a + (fib[N - 1] / fib[N]) * (b - a)
    f1 = funcion(x1)
    f2 = funcion(x2)

    history = []
    # Iteramos N-2 veces (estándar)
    while k <= (N - 2):
        history.append({
            "iter": k,
            "a": a, "b": b,
            "x1": x1, "x2": x2,
            "f1": f1, "f2": f2,
            "interval": abs(b - a)
        })

        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (fib[N - k - 1] / fib[N - k]) * (b - a)
            f2 = funcion(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (fib[N - k - 2] / fib[N - k]) * (b - a)
            f1 = funcion(x1)

        k += 1

    x_opt = (a + b) / 2.0
    f_opt = funcion(x_opt)

    entorno = np.linspace(x_opt - 1e-3, x_opt + 1e-3, 5)
    vals = [funcion(xx) for xx in entorno]
    tipo = "mínimo" if funcion(x_opt) == min(vals) else "máximo"

    distancia = abs(b - a)

    out = (x_opt, f_opt, k - 1, tipo, distancia)
    if return_history:
        return out + (history, limit_reached, target)
    return out


# Alias de compatibilidad (algunas versiones de la interfaz importan fibonacci_search)
def fibonacci_search(*args, **kwargs):
    return metodo_fibonacci(*args, **kwargs)
