import numpy as np

def busqueda_local(funcion, x0, paso=0.1, max_iter=100, tolerancia=1e-5, return_history=False):
    """
    Búsqueda local simple (1D). Devuelve:
      x_opt, f_opt, iteraciones, tipo, distancia [, history]
    history: lista de dicts con info por iteración
    """
    x = float(x0)
    paso_actual = float(paso)
    iteraciones = 0
    history = []

    for k in range(max_iter):
        fx = funcion(x)
        f_plus = funcion(x + paso_actual)
        f_minus = funcion(x - paso_actual)

        # Elegir mejor movimiento (max/min lo inferimos luego, aquí movemos por mejora absoluta)
        best_x = x
        best_f = fx

        if f_plus < best_f:
            best_f = f_plus
            best_x = x + paso_actual
        if f_minus < best_f:
            best_f = f_minus
            best_x = x - paso_actual

        improved = (best_x != x)

        history.append({
            "iter": k + 1,
            "x": x,
            "f(x)": fx,
            "paso": paso_actual,
            "x_candidate": best_x,
            "f_candidate": best_f,
            "improved": improved
        })

        if improved:
            x = best_x
        else:
            paso_actual *= 0.5

        iteraciones = k + 1
        if paso_actual < tolerancia:
            break

    # Determinar si es mínimo o máximo (entorno)
    entorno = np.linspace(x - 1e-3, x + 1e-3, 5)
    valores = [funcion(xx) for xx in entorno]
    tipo = "mínimo" if funcion(x) == min(valores) else "máximo"

    distancia = paso_actual
    out = (x, funcion(x), iteraciones, tipo, distancia)
    if return_history:
        return out + (history,)
    return out



# Alias de compatibilidad (algunas versiones de la interfaz importan buscar_maximo)
def buscar_maximo(funcion, a, b, paso=0.1, max_iter=200, x0=None):
    """Wrapper: usa busqueda_local() iniciando en el punto medio si no se pasa x0."""
    if x0 is None:
        x0 = (a + b) / 2.0
    return busqueda_local(funcion, x0, paso, max_iter)
