import numpy as np
import math

def fibonacci_search(f, a, b, L):
    """
    Implementa el Método de Búsqueda de Fibonacci con registro de iteraciones.
    Retorna: (x_optimo, f_min, a_final, b_final, log_data)
    """
    log_data = []
    
    try:
        A = (b - a) / L
        
        # Generar Secuencia de Fibonacci
        fib_seq = [1, 1] 
        n = 2
        while fib_seq[-1] < A:
            fib_seq.append(fib_seq[-1] + fib_seq[-2])
            n += 1

        Fn = fib_seq[-1] 
        
        ak, bk = a, b
        
        # Puntos de prueba iniciales
        # fib_seq[n - 3] es F_{n-2}, fib_seq[n - 2] es F_{n-1}
        lambda_k = ak + (fib_seq[n - 3] / Fn) * (bk - ak)
        mu_k = ak + (fib_seq[n - 2] / Fn) * (bk - ak) 

        theta_lambda = f(lambda_k)
        theta_mu = f(mu_k)

        k = 1
        while k <= n - 2:
            ak_next, bk_next = ak, bk
            L_k = bk - ak

            # Guardar estado de la iteración actual
            log_data.append({
                "k": k,
                "a_k": ak,
                "b_k": bk,
                "L_k": L_k,
                "lambda_k": lambda_k,
                "mu_k": mu_k,
                "f_lambda": theta_lambda,
                "f_mu": theta_mu
            })
            
            # --- Lógica de Reducción ---
            if theta_lambda > theta_mu:
                ak_next = lambda_k
                lambda_k_next = mu_k
                theta_lambda_next = theta_mu
                
                F_n_menos_k_menos_1 = fib_seq[n - k - 2]
                F_n_menos_k = fib_seq[n - k - 1] 
                
                mu_k_next = ak_next + (F_n_menos_k_menos_1 / F_n_menos_k) * (bk_next - ak_next)
                theta_mu_next = f(mu_k_next)

            else: # theta_lambda <= theta_mu
                bk_next = mu_k
                mu_k_next = lambda_k
                theta_mu_next = theta_lambda
                
                F_n_menos_k_menos_2 = fib_seq[n - k - 3] 
                F_n_menos_k = fib_seq[n - k - 1]
                
                lambda_k_next = ak_next + (F_n_menos_k_menos_2 / F_n_menos_k) * (bk_next - ak_next)
                theta_lambda_next = f(lambda_k_next)
                
            # Actualización
            ak = ak_next
            bk = bk_next
            lambda_k = lambda_k_next
            mu_k = mu_k_next
            theta_lambda = theta_lambda_next
            theta_mu = theta_mu_next
            k += 1

        # Agregar la última iteración
        log_data.append({
            "k": k,
            "a_k": ak,
            "b_k": bk,
            "L_k": bk - ak,
            "lambda_k": lambda_k,
            "mu_k": mu_k,
            "f_lambda": theta_lambda,
            "f_mu": theta_mu
        })

        # Resultado final: centro del intervalo final [ak, bk]
        optimo_x = (ak + bk) / 2 
        optimo_y = f(optimo_x)

        return optimo_x, optimo_y, ak, bk, log_data

    except Exception as e:
        raise ValueError(f"Error en el cálculo de Fibonacci: {e}")

def fixed_step_search(f, a, b, N):
    """
    Implementa el método de Búsqueda Exhaustiva (Pasos Fijos).
    N es el número de fragmentos (divisiones).
    Retorna: (x_optimo, f_min, a_final, b_final, log_data)
    """
    log_data = []
    
    try:
        N = int(N) # Asegurar que N es un entero
        if N <= 1:
            raise ValueError("El número de pasos (N) debe ser mayor que 1.")

        # Generación de puntos de prueba
        x_puntos = np.linspace(a, b, N + 1)
        
        # Inicialización del registro
        min_f = float('inf')
        x_minimo = None
        
        # Evaluación de la función en cada punto
        for i, x in enumerate(x_puntos):
            y = f(x)
            
            # Registrar cada evaluación como un 'paso'
            # L_k se utiliza para representar el paso, excepto en el último punto
            log_data.append({
                "k": i + 1,
                "a_k": x,
                "b_k": x_puntos[i + 1] if i < N else x,
                "L_k": (b - a) / N if i < N else 0,
                "lambda_k": x,
                "mu_k": y,
                "f_lambda": y,
                "f_mu": float('nan') 
            })
            
            # Encontrar el mínimo
            if y < min_f:
                min_f = y
                x_minimo = x
                
        # El intervalo final de incertidumbre es la mitad del paso de búsqueda
        paso = (b - a) / N
        final_a = x_minimo - paso / 2
        final_b = x_minimo + paso / 2
        
        return x_minimo, min_f, final_a, final_b, log_data

    except Exception as e:
        raise ValueError(f"Error en el cálculo de Pasos Fijos: {e}")



def local_neighborhood_search(f, a_initial, b_initial, d_radius, n_points):
    """
    Implementa la Búsqueda Local de Vecindario (Pasos Fijos)
    c_initial se calcula como el mejor entre a, b, y c_random.
    d_radius es el radio del vecindario.
    n_points es el número de divisiones del vecindario.
    
    Retorna: (x_optimo, f_min, neighborhood_start, neighborhood_end, log_data)
    """
    log_data = []
    
    try:
        d_radius = float(d_radius)
        n_points = int(n_points)
        if d_radius <= 0 or n_points <= 1:
             raise ValueError("El radio 'd' debe ser positivo y N > 1.")
        
        # 1. EVALUACIÓN INICIAL
        # En la GUI, el punto inicial 'c' y la evaluación ya se dan por implícitas
        # en la función, pero aquí usaremos el centro del intervalo [a, b] como c_best inicial
        
        # 1.1 Encontrar el mejor punto inicial entre a, b, y el centro c
        evaluations = {
            'a': (a_initial, f(a_initial)),
            'b': (b_initial, f(b_initial)),
            'c': ((a_initial + b_initial) / 2, f((a_initial + b_initial) / 2))
        }
        c_best = min(evaluations.values(), key=lambda item: item[1])[0]
        
        # 2. DEFINICIÓN DEL VECINDARIO LOCAL
        neighborhood_start = c_best - d_radius
        neighborhood_end = c_best + d_radius
        
        # Asegurarse de que el vecindario no exceda el rango inicial [a_initial, b_initial]
        neighborhood_start = max(neighborhood_start, a_initial)
        neighborhood_end = min(neighborhood_end, b_initial)

        # 2.1 Calcular el tamaño del paso (delta x)
        total_range = neighborhood_end - neighborhood_start
        step_size = total_range / n_points
        
        # 2.2 Generar los n+1 puntos en el vecindario
        points_to_check = np.linspace(neighborhood_start, neighborhood_end, n_points + 1)

        min_f = f(c_best) # El mínimo inicial es el mejor punto encontrado
        x_minimo = c_best
        
        # 3. EVALUACIÓN Y LOG
        for i, x in enumerate(points_to_check):
            y = f(x)
            
            # Registrar la evaluación
            log_data.append({
                "k": i + 1,
                "a_k": neighborhood_start, # Inicio del vecindario
                "b_k": neighborhood_end,   # Fin del vecindario
                "L_k": step_size,          # Tamaño del paso
                "lambda_k": x,             # X evaluado
                "mu_k": y,                 # F(X) evaluado
                "f_lambda": y,
                "f_mu": float('nan') 
            })
            
            # 4. ACTUALIZAR MÍNIMO GLOBAL (si el nuevo punto es mejor)
            if y < min_f:
                min_f = y
                x_minimo = x
                
        # El intervalo final de incertidumbre es el vecindario completo
        final_a = neighborhood_start
        final_b = neighborhood_end
        
        return x_minimo, min_f, final_a, final_b, log_data
        
    except Exception as e:
        raise ValueError(f"Error en el cálculo de Búsqueda Local: {e}")