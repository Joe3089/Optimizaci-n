import sympy
import numpy as np
from scipy.optimize import minimize
from line_search import newton_armijo

def penalty_method_newton(
    func_str: str,
    constraint_str: str,
    var_str: str,
    x0: list,
    penalty_r: float = 100.0
):
    """
    Prepara la función de penalización P(x) = f(x) + R * max(0, g(x))^2
    y ejecuta el método de Newton.
    """
    print(f"--- Iniciando Método de Penalización ---")
    print(f"Objetivo: {func_str}")
    print(f"Restricción (g(x)<=0): {constraint_str}")
    print(f"Penalización R: {penalty_r}")

    # Construcción de la cadena para SymPy
    # Usamos Max(0, g(x))**2 para penalizar solo si g(x) > 0
    # Nota: SymPy maneja 'Max' simbólicamente.
    combined_func_str = f"({func_str}) + {penalty_r} * Max(0, {constraint_str})**2"
    
    # Delegar al "obrero" (Newton)
    return newton_armijo(combined_func_str, var_str, x0)

def barrier_method_newton(
    func_str: str,
    constraint_str: str, 
    var_str: str,
    x0: list,
    mu: float = 10.0     
):
    """
    Implementa el Método de Barrera Logarítmica (P|unto Interior).
    B(x, mu) = f(x) - mu * sum(ln(-g(x)))
    
    IMPORTANTE: El punto inicial x0 DEBE ser factible (cumplir g(x) < 0)
    para empezar, o el logaritmo dará error matemático.
    """
    print(f"--- Iniciando Método de Barrera (Logarítmica) ---")
    print(f"Objetivo: {func_str}")
    print(f"Restricción (g(x)<=0): {constraint_str}")
    print(f"Parámetro de Barrera mu: {mu}")

    # Construcción de la función de Barrera
    # Si la restricción es g(x) <= 0, entonces -g(x) >= 0.
    combined_func_str = f"({func_str}) - {mu} * log(-({constraint_str}))"
    
    # Delegar al "obrero" (Newton funciona bien aquí porque log es suave)
    return newton_armijo(combined_func_str, var_str, x0)

# ==============================================================================
# NUEVOS MOTORES (SCIPY: BFGS, NELDER-MEAD)
# ==============================================================================

def _scipy_solver(func_str, var_str, x0, method):
    """
    Motor genérico que conecta los wrappers con scipy.optimize.minimize.
    """
    print(f"--- Iniciando Motor Scipy ({method}) ---")
    
    # 1. Preparar función numérica
    vars_sym = sympy.symbols(var_str)
    if not isinstance(vars_sym, (list, tuple)):
        vars_sym = (vars_sym,)
        
    f_sym = sympy.sympify(func_str)
    f_num = sympy.lambdify(vars_sym, f_sym, 'numpy')
    
    # Wrapper para que scipy consuma array 1D
    def func_to_minimize(x):
        return f_num(*x)
    
    # 2. Configurar Logging (Callback)
    log_data = []
    iteration = 0
    
    # Registrar punto inicial
    x_curr = np.array(x0)
    log_data.append({
        "k": 0, "x_k": x_curr, "x_k_str": np.array2string(x_curr, precision=3),
        "f_k": func_to_minimize(x_curr), "grad_norm": 0.0, "alpha_k": 0.0, "d_k_str": method
    })

    def callback(xk):
        nonlocal iteration
        iteration += 1
        val = func_to_minimize(xk)
        # Nota: Scipy no entrega gradiente en el callback estándar, ponemos 0.0
        log_data.append({
            "k": iteration, "x_k": xk, "x_k_str": np.array2string(xk, precision=3),
            "f_k": val, "grad_norm": 0.0, "alpha_k": 0.0, "d_k_str": method
        })

    # 3. Ejecutar Optimización
    res = minimize(func_to_minimize, x_curr, method=method, callback=callback)
    
    print(f"Fin Scipy. x*: {res.x}, fun: {res.fun}")
    return res.x, res.fun, log_data

# --- Wrappers con Motor BFGS (Suave y Rápido) ---

def penalty_method_bfgs(func_str, constraint_str, var_str, x0, penalty_r=100.0):
    combined_func_str = f"({func_str}) + {penalty_r} * Max(0, {constraint_str})**2"
    return _scipy_solver(combined_func_str, var_str, x0, 'BFGS')

def weighted_sum_bfgs(f1_str, f2_str, var_str, x0, weight=0.5):
    combined_func_str = f"{weight} * ({f1_str}) + (1 - {weight}) * ({f2_str})"
    return _scipy_solver(combined_func_str, var_str, x0, 'BFGS')

# --- Wrappers con Motor Nelder-Mead (Robusto / Sin Derivadas) ---

def penalty_method_nelder(func_str, constraint_str, var_str, x0, penalty_r=100.0):
    combined_func_str = f"({func_str}) + {penalty_r} * Max(0, {constraint_str})**2"
    return _scipy_solver(combined_func_str, var_str, x0, 'Nelder-Mead')

def weighted_sum_nelder(f1_str, f2_str, var_str, x0, weight=0.5):
    combined_func_str = f"{weight} * ({f1_str}) + (1 - {weight}) * ({f2_str})"
    return _scipy_solver(combined_func_str, var_str, x0, 'Nelder-Mead')

def weighted_sum_newton(
    f1_str: str,
    f2_str: str,
    var_str: str,
    x0: list,
    weight: float = 0.5
):
    """
    Prepara la función de suma ponderada F(x) = w*f1(x) + (1-w)*f2(x)
    y ejecuta el método de Newton.
    """
    print(f"--- Iniciando Suma Ponderada ---")
    print(f"f1: {f1_str}")
    print(f"f2: {f2_str}")
    print(f"Peso w: {weight}")

    # Construcción de la cadena combinada
    combined_func_str = f"{weight} * ({f1_str}) + (1 - {weight}) * ({f2_str})"
    
    # Delegar al "obrero" (Newton)
    return newton_armijo(combined_func_str, var_str, x0)
