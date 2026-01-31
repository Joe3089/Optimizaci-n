import sympy
import numpy as np

def newton_armijo(
    func_str: str,
    var_str: str,
    x0: list,
    mu: float = 0.6,
    alpha_reduce: float = 0.8,
    max_iter: int = 50,
    tolerance: float = 1e-6
):
    """
    Minimiza una función usando el Método de Newton con backtracking 
    de Armijo y diferenciación simbólica.
    
    Devuelve: (x_opt, y_opt, log_data)
    """
    
    print(f"--- Iniciando Optimización Newton (Armijo) ---")
    print(f"Función: {func_str}, Vars: {var_str}, x0: {x0}")

    # --- 1. Configuración Simbólica ---
    vars_sym = sympy.symbols(var_str)
    if not isinstance(vars_sym, (list, tuple)):
        vars_sym = (vars_sym,)
    
    try:
        f_sym = sympy.sympify(func_str)
    except sympy.SympifyError as e:
        print(f"Error al parsear la función: {e}")
        return None, None, []

    grad_sym = [f_sym.diff(var) for var in vars_sym]
    hess_sym = [[f_sym.diff(v1, v2) for v2 in vars_sym] for v1 in vars_sym]

    # --- 2. Conversión a Funciones Numéricas ---
    # Definir módulos para manejar DiracDelta (que surge de derivar Max) y Heaviside
    modules = [{'DiracDelta': lambda x: 0.0, 'Heaviside': lambda x: np.heaviside(x, 1.0)}, 'numpy']
    f_num = sympy.lambdify(vars_sym, f_sym, modules)
    grad_num = sympy.lambdify(vars_sym, grad_sym, modules)
    hess_num = sympy.lambdify(vars_sym, hess_sym, modules)

    # --- 3. Bucle de Optimización ---
    log_data = []
    x_k = np.array(x0, dtype=float)

    for k in range(max_iter):
        
        # Evaluar en el punto actual x_k
        try:
            f_k = f_num(*x_k)
            g_k = np.array(grad_num(*x_k))
            H_k = np.array(hess_num(*x_k))
        except Exception as e:
            print(f"Error al evaluar la función/grad/hess en {x_k}: {e}")
            break

        grad_norm = np.linalg.norm(g_k)

        # --- INICIO DE LA CORRECCIÓN ---
        # Crear el log entry AQUÍ, al inicio del bucle,
        # para que contenga los valores que main.py espera.
        log_entry = {
            "k": k,
            "x_k": x_k,
            "x_k_str": np.array2string(x_k, precision=3),
            "f_k": f_k,
            "grad_norm": grad_norm, # <-- La clave que faltaba
            "alpha_k": 0.0, # Valor default si converge
            "d_k_str": "N/A" # Valor default si converge
        }
        # --- FIN DE LA CORRECCIÓN ---

        # Criterio de parada
        if grad_norm < tolerance:
            print(f"Convergencia alcanzada en la iteración {k}.")
            log_data.append(log_entry) # Añadir el último estado
            break

        # --- 4. Calcular Dirección de Newton ---
        try:
            d_k = np.linalg.solve(H_k, -g_k)
            # Actualizar el log entry con el d_k calculado
            log_entry["d_k_str"] = np.array2string(d_k, precision=3) # <-- La clave que faltaba
        except np.linalg.LinAlgError:
            print("Error: El Hessiano es singular.")
            log_entry["d_k_str"] = "Hessiano Singular"
            log_data.append(log_entry) # Añadir el estado de error
            break

        # --- 5. Búsqueda de Línea (Armijo) ---
        alpha = 1.0
        RHS_base_term = mu * (g_k @ d_k) 

        while True:
            x_new = x_k + alpha * d_k
            f_new = f_num(*x_new)
            
            # --- PROTECCIÓN PARA BARRERA (Logaritmos) ---
            # Si nos salimos de la zona factible, f_new será NaN o Infinito.
            if not np.isfinite(f_new):
                alpha = alpha * alpha_reduce
                continue
            
            RHS = f_k + alpha * RHS_base_term
            
            if f_new > RHS:
                alpha = alpha * alpha_reduce
                if alpha < 1e-9:
                    print("Alpha se hizo demasiado pequeño.")
                    break
            else:
                break # Se cumple la condición
        
        alpha_k = alpha
        
        # Actualizar el log_entry con el alfa final
        log_entry["alpha_k"] = alpha_k
        
        # Añadir el log de la iteración completada
        log_data.append(log_entry)

        # --- 7. Actualizar x^k ---
        x_k = x_k + alpha_k * d_k
        
        if k == max_iter - 1:
            print("Se alcanzó el máximo de iteraciones.")

    y_opt = f_num(*x_k)
    print(f"Optimización finalizada. x* = {x_k}, f(x*) = {y_opt}")
    
    return x_k, y_opt, log_data

def wolfe_line_search(
    func_str: str,
    var_str: str,
    x_k_list: list,
    d_k_list: list,
    u_in: float,
    v_in: float,
    mu: float,
    max_iter: int = 20
):
    """
    Busca un alfa que satisfaga la Condición de Curvatura Fuerte de Wolfe
    usando un método de bisección.
    Devuelve: (float, list): El alfa final encontrado y el log_data.
    """
    
    print(f"--- Iniciando Búsqueda de Wolfe ---")
    print(f"Función: {func_str}, x_k: {x_k_list}, d_k: {d_k_list}")
    print(f"Intervalo: [{u_in}, {v_in}], mu: {mu}\n")
    
    # --- 1. Configuración Simbólica ---
    vars_sym = sympy.symbols(var_str)
    if not isinstance(vars_sym, (list, tuple)):
        vars_sym = (vars_sym,)
    
    try:
        f_sym = sympy.sympify(func_str)
    except sympy.SympifyError as e:
        print(f"Error al parsear la función: {e}")
        return None, []

    grad_sym = [f_sym.diff(var) for var in vars_sym]
    modules = [{'DiracDelta': lambda x: 0.0, 'Heaviside': lambda x: np.heaviside(x, 1.0)}, 'numpy']
    grad_num = sympy.lambdify(vars_sym, grad_sym, modules)

    # --- 2. Valores Iniciales ---
    x_k = np.array(x_k_list, dtype=float)
    d_k = np.array(d_k_list, dtype=float)
    
    g_k = np.array(grad_num(*x_k))
    g_k_dot_d = g_k @ d_k
    
    RHS = mu * abs(g_k_dot_d)
    
    u = u_in
    v = v_in
    alfa = v
    
    log_data = []
    final_alfa = alfa

    print("--- Iniciando Bisección ---")
    print(f"Condición a cumplir: |∇f(α)ᵀd| <= {RHS:.6f}  (calculado de μ * |{g_k_dot_d:.2f}|)")
    print("k | ALFA  | |∇f(α)ᵀd| (LHS) | ∇f(α)ᵀd (Sign) |  u    |  v    | next_α")
    print("-" * 70)

    for k in range(max_iter):
        
        # --- 3. Evaluar en el alfa actual ---
        x_new = x_k + alfa * d_k
        g_new = np.array(grad_num(*x_new))
        g_alpha_dot_d = g_new @ d_k
        LHS = abs(g_alpha_dot_d)
        
        # Guardar estado para el log
        log_entry = {
            "k": k,
            "alfa": alfa,
            "LHS": LHS,
            "RHS": RHS,
            "g_alpha_dot_d": g_alpha_dot_d,
            "u": u,
            "v": v,
            "next_alpha": (u + v) / 2
        }
        
        # En la primera iteración (k=0), guardamos los datos
        # iniciales para que la gráfica los pueda usar.
        if k == 0:
            log_entry['x_k_start'] = x_k_list
            log_entry['d_k_start'] = d_k_list
            
        log_data.append(log_entry)
        
        print(f"{k:<1} | {alfa:5.3f} | {LHS:14.6f} | {g_alpha_dot_d:15.6f} | {u:5.3f} | {v:5.3f} | {log_entry['next_alpha']:6.3f}")

        # --- 4. Comprobar Condición de Parada ---
        if LHS <= RHS:
            print(f"\nCondición de Wolfe Satisfecha: {LHS:.6f} <= {RHS:.6f}")
            final_alfa = alfa
            break
            
        if (v - u) < 1e-7:
            print("\nIntervalo de bisección muy pequeño, deteniendo.")
            final_alfa = alfa
            break

        # --- 5. Actualizar Intervalo (Bisección) ---
        if g_alpha_dot_d > 0:
            v = alfa
        else:
            u = alfa
            
        alfa = (u + v) / 2
        
        if k == max_iter - 1:
            print("\nMáximo de iteraciones alcanzado.")
            final_alfa = alfa

    print(f"Alfa final encontrado: {final_alfa:.6f}")
    
    # Devuelve el alfa final y el log detallado de la bisección
    return final_alfa, log_data


def newton_wolfe_step(
    func_str: str,
    var_str: str,
    x_k_list: list,
    # Estos parámetros son ahora fijos para este modo
    u_in: float = 0.0,
    v_in: float = 1.0,
    mu: float = 0.6
):
    """
    Wrapper dinámico que:
    1. Calcula la dirección de Newton (d_k) en x_k.
    2. Ejecuta la búsqueda de línea de Wolfe para encontrar alfa.
    Devuelve: (x_opt, y_opt, log_data) 
             (Donde x_opt es el alfa, y_opt es 0 (no usado), 
              y log_data es el log de Wolfe)
    """
    
    print(f"--- Iniciando Paso Newton-Wolfe ---")
    print(f"Calculando d_k en x_k = {x_k_list}...")
    
    # --- 1. Configuración Simbólica para calcular d_k ---
    vars_sym = sympy.symbols(var_str)
    if not isinstance(vars_sym, (list, tuple)):
        vars_sym = (vars_sym,)
    
    try:
        f_sym = sympy.sympify(func_str)
    except sympy.SympifyError as e:
        print(f"Error al parsear la función: {e}")
        return None, None, []

    grad_sym = [f_sym.diff(var) for var in vars_sym]
    hess_sym = [[f_sym.diff(v1, v2) for v2 in vars_sym] for v1 in vars_sym]
    
    modules = [{'DiracDelta': lambda x: 0.0, 'Heaviside': lambda x: np.heaviside(x, 1.0)}, 'numpy']
    grad_num = sympy.lambdify(vars_sym, grad_sym, modules)
    hess_num = sympy.lambdify(vars_sym, hess_sym, modules)

    # --- 2. Calcular d_k dinámicamente ---
    x_k = np.array(x_k_list, dtype=float)
    
    try:
        g_k = np.array(grad_num(*x_k))
        H_k = np.array(hess_num(*x_k))
    except Exception as e:
        print(f"Error al evaluar grad/hess en {x_k}: {e}")
        return None, None, []

    try:
        d_k = np.linalg.solve(H_k, -g_k)
        print(f"d_k calculado = {d_k}")
    except np.linalg.LinAlgError:
        print("Error: El Hessiano es singular. No se puede calcular d_k.")
        return None, None, []

    # --- 3. Llamar a la búsqueda de Wolfe ---
    final_alfa, log_data = wolfe_line_search(
        func_str,
        var_str,
        x_k_list,
        d_k.tolist(), # Convertir a lista
        u_in,
        v_in,
        mu
    )
    
    # Devolver una firma compatible con la GUI
    return final_alfa, 0.0, log_data