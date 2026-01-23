def _approx_grad(func, x, eps=1e-6):
    return (func(x + eps) - func(x - eps)) / (2 * eps)

def armijo_search(func, x0, alpha0=1.0, rho=0.5, c=1e-4, max_iter=50, return_history=False):
    """
    Backtracking line search (Armijo) (minimización).
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
        cond = (f_new <= f0 + c * alpha * grad0 * d)

        history.append({
            "iter": k + 1,
            "alpha": alpha,
            "x_new": x_new,
            "f_new": f_new,
            "armijo_ok": cond
        })

        if cond:
            distancia = abs(alpha * d)
            out = (x_new, f_new, k + 1, distancia)
            if return_history:
                return out + (history,)
            return out

        alpha *= rho

    x_new = x0 + alpha * d
    distancia = abs(alpha * d)
    out = (x_new, func(x_new), max_iter, distancia)
    if return_history:
        return out + (history,)
    return out
