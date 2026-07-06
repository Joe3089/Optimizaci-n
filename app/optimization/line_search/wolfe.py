def _approx_grad(func, x, eps=1e-6):
    return (func(x + eps) - func(x - eps)) / (2 * eps)

def wolfe_search(func, x0, alpha0=1.0, rho=0.5, c1=1e-4, c2=0.9, max_iter=50, return_history=False):
    """
    Wolfe simplificado (Armijo + curvatura aproximada).
    Devuelve:
      x_new, f_new, iteraciones, distancia [, history]
    """
    x0 = float(x0)
    alpha = float(alpha0)
    f0 = func(x0)
    grad0 = _approx_grad(func, x0)
    d = -grad0  # descenso

    history = []
    for k in range(max_iter):
        x_new = x0 + alpha * d
        f_new = func(x_new)
        grad_new = _approx_grad(func, x_new)

        cond_armijo = (f_new <= f0 + c1 * alpha * grad0 * d)
        cond_curvatura = (abs(grad_new) <= c2 * abs(grad0))

        history.append({
            "iter": k + 1,
            "alpha": alpha,
            "x_new": x_new,
            "f_new": f_new,
            "armijo_ok": cond_armijo,
            "curv_ok": cond_curvatura
        })

        if cond_armijo and cond_curvatura:
            distancia = abs(alpha * d)
            out = (x_new, f_new, k + 1, distancia)
            if return_history:
                return out + (history,)
            return out

        alpha *= rho

    distancia = abs(alpha * d)
    out = (x_new, func(x_new), max_iter, distancia)
    if return_history:
        return out + (history,)
    return out

# ---------------------------------------------------------------------
# Compatibilidad con versiones anteriores del proyecto:
# Algunos módulos/interfaz intentan importar `metodo_wolfe`.
# Este wrapper usa `wolfe_search` desde un x0 dentro del rango [a,b].
def metodo_wolfe(func, a, b, alpha0=1.0, rho=0.5, c1=1e-4, c2=0.9, max_iter=50, return_history=True):
    """Wrapper compatible.

    Parámetros:
      func: función f(x)
      a,b: rango (se usa x0 = (a+b)/2)
      return_history:
        - True  -> retorna la tabla de iteraciones (list[dict]) para mostrar en la UI
        - False -> retorna (x_new, f_new, iteraciones, distancia)

    Nota: `wolfe_search` implementa Wolfe simplificado para MINIMIZACIÓN.
    """
    a = float(a)
    b = float(b)
    x0 = (a + b) / 2.0

    out = wolfe_search(
        func,
        x0,
        alpha0=alpha0,
        rho=rho,
        c1=c1,
        c2=c2,
        max_iter=max_iter,
        return_history=True
    )

    # out: (x_new, f_new, iteraciones, distancia, history)
    x_new, f_new, iters, distancia, history = out

    if return_history:
        return history
    return (x_new, f_new, iters, distancia)
