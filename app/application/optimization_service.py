# app/application/optimization_service.py
# Dispatcher de métodos de optimización — extraído de
# InterfazOptimizacion._run (app/ui/main_window.py), Fase 4b.
# Función pura: sin dependencia de Qt ni de estado de instancia de la UI.
from __future__ import annotations

import importlib
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import sympy as sp
    _SYMPY = True
except Exception:
    sp = None                                              # type: ignore
    _SYMPY = False

_func_compat = None
def _get_fc():
    global _func_compat
    if _func_compat is None:
        try:
            import app.application.func_compat as _fc
            _func_compat = _fc
        except ImportError:
            pass
    return _func_compat

_multiobj_mod = None
def _get_multiobj():
    global _multiobj_mod
    if _multiobj_mod is None:
        try:
            import app.optimization.multiobjective.schaffer as _mo
            _multiobj_mod = _mo
        except ImportError:
            pass
    return _multiobj_mod

_line_search_nd = None
_local_search   = None
_heuristics_mod = None
_metaheur_mod   = None
_wrappers_mod   = None
_modules_ready  = False

def _try_import(name):
    try:    return importlib.import_module(name)
    except: return None

def _init_algorithm_modules():
    """Carga (una sola vez) los módulos de algoritmos usados por run_method."""
    global _line_search_nd, _local_search, _heuristics_mod, _metaheur_mod
    global _wrappers_mod, _modules_ready
    if _modules_ready:
        return
    _line_search_nd = _try_import("app.optimization.line_search.nd_symbolic")
    _local_search   = _try_import("app.optimization.local_search")
    _heuristics_mod = _try_import("app.optimization.heuristics")
    _metaheur_mod   = _try_import("app.optimization.metaheuristics")
    _wrappers_mod   = _try_import("app.optimization.constrained.penalty_barrier")
    _modules_ready  = True


def run_method(metodo, fx, gx, vars_, x0, lo, hi, tol_L,
               mo_x0: Optional[float] = None,
               mo_alpha: Optional[float] = None):
    """
    Ejecuta el método de optimización solicitado y devuelve (hist, mo_context).

    `mo_context` es None salvo para métodos "MO — ..." (multiobjetivo), en cuyo
    caso es un dict con las claves tabla_ref/hist_iters/alpha/x_c/f1_call/f2_call/
    var1 que la UI necesita para el ploteo posterior (antes se guardaba en
    self._mo_*; ahora se devuelve explícitamente).

    `mo_x0`/`mo_alpha` reemplazan la lectura directa de los widgets
    edt_x0_mo/edt_alpha que hacía la versión original — la UI los lee y los
    pasa como parámetros.
    """
    _init_algorithm_modules()
    newton_armijo_fn = getattr(_line_search_nd, "newton_armijo",            None) if _line_search_nd else None
    newton_wolfe_fn  = getattr(_line_search_nd, "newton_wolfe_step",         None) if _line_search_nd else None
    fibonacci_fn     = getattr(_local_search,   "fibonacci_search",          None) if _local_search   else None
    local_fn         = getattr(_local_search,   "local_neighborhood_search", None) if _local_search   else None
    _heuristics_mod_ = _heuristics_mod
    _metaheur_mod_   = _metaheur_mod
    _wrappers_mod_   = _wrappers_mod

    def coerce(res):
        if res is None: return []
        if isinstance(res,list):
            return res if (res and isinstance(res[0],dict)) \
                   else [{"k":i,"val":r} for i,r in enumerate(res)]
        if isinstance(res,tuple) and len(res)==3: return coerce(res[2])
        if isinstance(res,dict):
            return coerce(res["history"]) if "history" in res else [res]
        try:
            import pandas as pd
            if isinstance(res,pd.DataFrame): return res.to_dict(orient="records")
        except: pass
        return [{"val":res}]

    # ── Helper: normalizar expresión y detectar/resolver variables ────────
    fc = _get_fc()

    def _norm(expr):
        return fc.normalize_expr(expr) if fc else expr

    def _detect_vars(expr):
        """Detecta variables, priorizando las que el usuario especificó."""
        if vars_ and vars_ != ["x1", "x2"] and any(v.strip() for v in vars_):
            return [v for v in vars_ if v.strip()]
        if fc:
            detected = fc.detect_variables(_norm(expr))
            return detected if detected else ["x"]
        return ["x"]

    def _get_x0_for(vnames):
        """Devuelve punto inicial: usa UI si fue especificado, si no usa centro del rango."""
        mid = (lo + hi) / 2
        if x0 and len(x0) == len(vnames):
            return list(x0)
        # Si solo hay una variable, usar el punto medio del rango
        if len(vnames) == 1:
            return [mid]
        # Multivariable sin x0: intentar x0 de UI truncado/ampliado
        if x0:
            base = list(x0)
            while len(base) < len(vnames): base.append(mid)
            return base[:len(vnames)]
        return [mid] * len(vnames)

    # ── Función 1D con detección automática de variable ──────────────────
    def f1d():
        if not _SYMPY: raise RuntimeError("SymPy no disponible.")
        if fc:
            norm = _norm(fx)
            fn_callable, var_name = fc.parse_function_1d(norm)
            return fn_callable
        else:
            # Fallback clásico
            xs = sp.Symbol("x")
            fl = sp.lambdify(xs, sp.sympify(fx, locals={"x": xs}), modules=["numpy"])
            return lambda v: float(fl(v))

    def _f1d_callable(norm_expr: str) -> object:
        """Alias para f1d que acepta expresión ya normalizada."""
        if fc:
            fn_c, _ = fc.parse_function_1d(norm_expr)
            return fn_c
        return f1d()

    # ── 1D ────────────────────────────────────────────────────────────────
    if metodo == "Búsqueda Local":
        if not local_fn: raise RuntimeError(
            "localsearch.py no encontrado.\nVerifica que el archivo esté en la carpeta del proyecto.")
        xo,fm,fa,fb,log = local_fn(f1d(), lo, hi, (hi-lo)*0.1, 20)
        for r in log: r.setdefault("x", r.get("lambda_k",0.))
        return log, None

    if metodo == "Fibonacci":
        if not fibonacci_fn: raise RuntimeError(
            "localsearch.py no encontrado.\nVerifica que el archivo esté en la carpeta del proyecto.")
        xo,fm,fa,fb,log = fibonacci_fn(f1d(), lo, hi, max(tol_L,1e-10))
        for r in log: r.setdefault("x", r.get("lambda_k",0.))
        return log, None

    if metodo == "Armijo":
        if not newton_armijo_fn: raise RuntimeError(
            "line_search.py no encontrado.\nVerifica que el archivo esté en la carpeta del proyecto.")
        norm_fx  = _norm(fx)
        v_names  = _detect_vars(norm_fx)
        var_str  = " ".join(v_names)
        x0_use   = _get_x0_for(v_names)
        xo, yo, log = newton_armijo_fn(
            func_str=norm_fx, var_str=var_str, x0=x0_use,
            max_iter=50, tolerance=1e-6)
        return log or [], None

    if metodo == "Wolfe":
        if not newton_wolfe_fn: raise RuntimeError(
            "line_search.py no encontrado.\nVerifica que el archivo esté en la carpeta del proyecto.")
        norm_fx  = _norm(fx)
        v_names  = _detect_vars(norm_fx)
        x0_use   = _get_x0_for(v_names)
        xo, yo, log = newton_wolfe_fn(
            func_str=norm_fx, var_str=" ".join(v_names),
            x_k_list=x0_use, u_in=lo, v_in=hi, mu=0.6)
        return log or [], None

    # ── BISECCIÓN y SECCIÓN DORADA (autónomas, sin módulos externos) ───────
    if metodo == "Bisección":
        fn1d   = _f1d_callable(_norm(fx))
        h      = max((hi - lo) * 1e-5, 1e-9)
        fprime = lambda xv: (fn1d(xv + h) - fn1d(xv - h)) / (2.0 * h)
        a_b, b_b = float(lo), float(hi)
        tol_b, max_b = 1e-8, 300

        # Buscar subintervalo con cambio de signo en f'
        if fprime(a_b) * fprime(b_b) > 0:
            n_sub  = 400
            xs_sub = np.linspace(a_b, b_b, n_sub)
            found  = False
            for _i in range(n_sub - 1):
                if fprime(xs_sub[_i]) * fprime(xs_sub[_i+1]) <= 0:
                    a_b, b_b = float(xs_sub[_i]), float(xs_sub[_i+1])
                    found = True; break
            if not found:
                ys_sub = np.array([fn1d(xv) for xv in xs_sub])
                xopt   = float(xs_sub[np.argmin(ys_sub)])
                return [{"k": 0, "x_k": round(xopt, 8),
                         "f_k": round(fn1d(xopt), 8),
                         "f'(x_k)": round(fprime(xopt), 8),
                         "b-a": round(b_b - a_b, 8),
                         "nota": "sin cambio signo; retorna mín. en grid"}], None

        hist_b: list = []
        for k in range(max_b):
            xm  = (a_b + b_b) / 2.0
            fpm = fprime(xm)
            largo = b_b - a_b
            hist_b.append({
                "k":        k,
                "a_k":      round(a_b, 8),
                "b_k":      round(b_b, 8),
                "x_k":      round(xm,  8),
                "f_k":      round(fn1d(xm), 8),
                "f'(x_k)":  round(fpm,  8),
                "b-a":      round(largo, 8),
                "converged": largo < tol_b or abs(fpm) < tol_b,
            })
            if largo < tol_b or abs(fpm) < tol_b:
                break
            if fprime(a_b) * fpm < 0:
                b_b = xm
            else:
                a_b = xm
        return hist_b, None

    if metodo == "Sección Dorada":
        fn1d = _f1d_callable(_norm(fx))
        phi  = (math.sqrt(5.0) - 1.0) / 2.0   # ≈ 0.61803…
        a_g, b_g = float(lo), float(hi)
        tol_g, max_g = 1e-8, 300

        c_g  = b_g - phi * (b_g - a_g)
        d_g  = a_g + phi * (b_g - a_g)
        fc_g = fn1d(c_g); fd_g = fn1d(d_g)

        hist_g: list = []
        for k in range(max_g):
            largo = b_g - a_g
            xm    = (a_g + b_g) / 2.0
            hist_g.append({
                "k":      k,
                "a_k":    round(a_g,  8),
                "b_k":    round(b_g,  8),
                "c_k":    round(c_g,  8),
                "f(c_k)": round(fc_g, 8),
                "d_k":    round(d_g,  8),
                "f(d_k)": round(fd_g, 8),
                "x_k":    round(xm,   8),
                "f_k":    round(fn1d(xm), 8),
                "b-a":    round(largo, 8),
                "r_φ":    round(phi,   6),
                "converged": largo < tol_g,
            })
            if largo < tol_g:
                break
            if fc_g < fd_g:
                b_g = d_g; d_g = c_g; fd_g = fc_g
                c_g = b_g - phi * (b_g - a_g); fc_g = fn1d(c_g)
            else:
                a_g = c_g; c_g = d_g; fc_g = fd_g
                d_g = a_g + phi * (b_g - a_g); fd_g = fn1d(d_g)

        xopt_g = (a_g + b_g) / 2.0
        hist_g[-1]["x_k"] = round(xopt_g, 8)
        hist_g[-1]["f_k"] = round(fn1d(xopt_g), 8)
        return hist_g, None

    # ── HEURÍSTICOS ───────────────────────────────────────────────────────
    if metodo == "Sección Áurea":
        if not _heuristics_mod_: raise RuntimeError(
            "heuristics.py no encontrado.")
        fn1d = _f1d_callable(_norm(fx))
        return _heuristics_mod_.seccion_aurea(fn1d, lo, hi, tol=1e-6, max_iter=200), None

    if metodo == "Goldstein":
        if not _heuristics_mod_: raise RuntimeError(
            "heuristics.py no encontrado.")
        norm_fx = _norm(fx)
        v_names = _detect_vars(norm_fx)
        x0_use  = _get_x0_for(v_names)
        return _heuristics_mod_.goldstein_search(
            norm_fx, " ".join(v_names), x0_use,
            mu=0.2, alpha0=1.0, rho=0.5, max_iter=60), None

    if metodo == "Newton-Raphson":
        if not _heuristics_mod_: raise RuntimeError(
            "heuristics.py no encontrado.")
        norm_fx = _norm(fx)
        v_names = _detect_vars(norm_fx)
        x0_use  = _get_x0_for(v_names)
        return _heuristics_mod_.newton_raphson(
            norm_fx, " ".join(v_names), x0_use,
            tol=1e-7, max_iter=60), None

    if metodo == "Grad. Conjugado":
        if not _heuristics_mod_: raise RuntimeError(
            "heuristics.py no encontrado.")
        norm_fx = _norm(fx)
        v_names = _detect_vars(norm_fx)
        x0_use  = _get_x0_for(v_names)
        return _heuristics_mod_.gradient_conjugate(
            norm_fx, " ".join(v_names), x0_use,
            tol=1e-7, max_iter=150), None

    # ── METAHEURÍSTICOS ───────────────────────────────────────────────────
    if metodo == "SA: Recocido Simulado":
        if not _metaheur_mod_: raise RuntimeError(
            "metaheuristics.py no encontrado.")
        norm_fx = _norm(fx)
        v_names = _detect_vars(norm_fx)
        x0_use  = _get_x0_for(v_names)
        return _metaheur_mod_.simulated_annealing(
            norm_fx, " ".join(v_names), x0_use,
            T_init=100.0, T_min=1e-5, cooling=0.95,
            max_iter=600, step_size=(hi - lo) * 0.1), None

    if metodo == "PSO: Enjambre":
        if not _metaheur_mod_: raise RuntimeError(
            "metaheuristics.py no encontrado.")
        norm_fx = _norm(fx)
        v_names = _detect_vars(norm_fx)
        return _metaheur_mod_.pso(
            norm_fx, " ".join(v_names),
            bounds=[lo, hi],
            n_particles=30, max_iter=150,
            w=0.7, c1=1.5, c2=1.5), None

    if metodo == "GA: Algoritmo Genético":
        if not _metaheur_mod_: raise RuntimeError(
            "metaheuristics.py no encontrado.")
        norm_fx = _norm(fx)
        v_names = _detect_vars(norm_fx)
        return _metaheur_mod_.genetic_algorithm(
            norm_fx, " ".join(v_names),
            bounds=[lo, hi],
            pop_size=50, max_gen=150,
            p_cross=0.8, p_mut=0.15, elite=3), None

    # ── MD ────────────────────────────────────────────────────────────────
    if metodo.startswith("MD:"):
        w = _wrappers_mod_
        if not w: raise RuntimeError(
            "wrappers.py no encontrado.\nVerifica que el archivo esté en la carpeta del proyecto.")
        norm_fx = _norm(fx)
        norm_gx = _norm(gx) if gx else ""

        # ── Reconciliar variables ─────────────────────────────────────────
        # wrappers.py usa sympify con los símbolos dados; si la expresión
        # tiene variables distintas a las del campo UI (ej. usuario pone 'x1 x2'
        # pero la función usa 'x'), la evaluación falla con TypeError.
        # Solución: construir la lista efectiva de variables tomando la unión
        # de las detectadas en la expresión + las de la restricción + las del campo UI,
        # dando prioridad a las detectadas en la expresión para que coincidan.
        if fc:
            _fx_vars  = fc.detect_variables(norm_fx)
            _gx_vars  = fc.detect_variables(norm_gx) if norm_gx else []
            _ui_vars  = [v for v in vars_ if v.strip()]

            # Unión en orden: vars de fx primero, luego gx, luego UI
            _merged: list = list(_fx_vars)
            for v in _gx_vars + _ui_vars:
                if v not in _merged:
                    _merged.append(v)

            # Si la lista efectiva está vacía, usar las del UI o fallback
            effective_vars = _merged if _merged else (_ui_vars if _ui_vars else ["x1", "x2"])
        else:
            effective_vars = [v for v in vars_ if v.strip()] or ["x1", "x2"]

        var_str = " ".join(effective_vars)

        # Ajustar x0 al número de variables efectivas
        mid = (lo + hi) / 2.0
        x0_md = list(x0) if x0 else []
        while len(x0_md) < len(effective_vars):
            x0_md.append(mid)
        x0_md = x0_md[:len(effective_vars)]

        fn_map = {
            "MD: Penalización (Newton)": "penalty_method_newton",
            "MD: Barreras (Newton)":     "barrier_method_newton",
            "MD: Pen. (BFGS)":           "penalty_bfgs",
            "MD: Pes. (BFGS)":           "steepest_bfgs",
            "MD: Pes. (Nelder-Mead)":    "nelder_mead",
            "MD: Sum. (BFGS)":           "sum_bfgs",
        }
        fn_name = fn_map.get(metodo)
        fn = getattr(w, fn_name, None) if fn_name else None
        if fn is None:
            if "Penalización" in metodo or "Pen." in metodo:
                fn = getattr(w, "penalty_method_newton", None)
            elif "Barrera" in metodo:
                fn = getattr(w, "barrier_method_newton", None)
            else:
                fn = getattr(w, "penalty_method_newton", None)
        if fn is None:
            raise RuntimeError(f"Método '{metodo}' no implementado en wrappers.py.")

        # Llamada principal con firma correcta
        _,_,hist = fn(norm_fx, norm_gx, var_str, x0_md)
        return hist, None

    # ── MULTIOBJETIVO ─────────────────────────────────────────────────────
    # Flujo: f(x) mono-objetivo del usuario
    #        → Escalarización: F(x)=(f1,f2) donde f1=f(x), f2=(x-x_c)²
    #        → φ(x,α) = α·f1(x) + (1-α)·f2(x)   para α dado
    #        → Resolver φ con Bisección o Sección Dorada
    #        → La tabla de α es un RESULTADO de referencia (no método)
    if _is_multiobj(metodo):
        mo = _get_multiobj()
        if mo is None:
            raise RuntimeError(
                "multiobj_schaffer.py no encontrado.\n"
                "Coloca el archivo junto a interfaz_qt.py.")
        if not fx or fx == "schaffer_interno":
            raise RuntimeError(
                "Ingresa la función f(x) mono-objetivo en el campo superior.\n"
                "El optimizador la convertirá en problema multiobjetivo.")

        import math as _math
        norm_fx_mo = _norm(fx)
        v_names    = _detect_vars(norm_fx_mo)
        var1       = v_names[0] if v_names else "x"

        # Punto inicial (antes leído de self.edt_x0_mo)
        x0_start = float(mo_x0) if mo_x0 is not None else (lo + hi) / 2

        # α (antes leído de self.edt_alpha)
        a_val = float(mo_alpha) if mo_alpha is not None else 0.5

        # Construir f1 y f2 callables
        if not _SYMPY:
            raise RuntimeError("SymPy es necesario para métodos multiobjetivo.")
        sym_x   = sp.Symbol(var1)
        sym_f1  = sp.sympify(norm_fx_mo)
        sym_df1 = sp.diff(sym_f1, sym_x)
        sym_d2f1= sp.diff(sym_df1, sym_x)
        f1_call  = sp.lambdify(sym_x, sym_f1,   modules=["numpy"])
        df1_call = sp.lambdify(sym_x, sym_df1,  modules=["numpy"])
        d2f1_call= sp.lambdify(sym_x, sym_d2f1, modules=["numpy"])

        x_c  = (lo + hi) / 2.0          # centro del rango → f2 centrada
        f2   = lambda xv: (float(xv) - x_c)**2
        df2  = lambda xv: 2.0*(float(xv) - x_c)
        d2f2 = lambda _xv: 2.0

        # φ(x,α) escalarizada y su derivada
        phi   = lambda xv, a: a*float(f1_call(xv)) + (1-a)*float(f2(xv))
        dphi  = lambda xv, a: a*float(df1_call(xv)) + (1-a)*float(df2(xv))
        d2phi = lambda xv, a: a*float(d2f1_call(xv)) + (1-a)*float(d2f2(xv))

        # ── Tabla de referencia α (siempre se calcula, se guarda en hist) ──
        # Resuelve φ con Sección Dorada para α = {0,1/5,1/4,1/3,1/2,1}
        _ALPHAS_REF = [
            (0.0,   "0"),
            (1/5,   "1/5"),
            (1/4,   "1/4"),
            (1/3,   "1/3"),
            (1/2,   "1/2"),
            (1.0,   "1"),
        ]
        tabla_ref = []
        for a_r, a_str in _ALPHAS_REF:
            # Sección Dorada rápida para cada α
            phi_r = lambda xv, a=a_r: a*float(f1_call(xv)) + (1-a)*float(f2(xv))
            p_g = (_math.sqrt(5)-1)/2
            ag_, bg_ = float(lo), float(hi)
            cg_ = bg_-p_g*(bg_-ag_); dg_ = ag_+p_g*(bg_-ag_)
            fcg_=phi_r(cg_); fdg_=phi_r(dg_)
            for _ in range(200):
                if bg_-ag_ < 1e-8: break
                if fcg_<fdg_: bg_=dg_; dg_=cg_; fdg_=fcg_; cg_=bg_-p_g*(bg_-ag_); fcg_=phi_r(cg_)
                else:         ag_=cg_; cg_=dg_; fcg_=fdg_; dg_=ag_+p_g*(bg_-ag_); fdg_=phi_r(dg_)
            xopt_r = (ag_+bg_)/2
            try:
                f1r = float(f1_call(xopt_r)); f2r = float(f2(xopt_r))
                phi_r_val = float(phi_r(xopt_r))
            except Exception:
                f1r = f2r = phi_r_val = float("nan")
            tabla_ref.append({
                "α": a_str,
                "f(x,α)": f"α·f₁+(1-α)·(x-{x_c:.2f})²",
                "x*(α)":  round(xopt_r, 6),
                "f₁(x*)": round(f1r, 6),
                "f₂(x*)": round(f2r, 6),
                "φ(x*,α)": round(phi_r_val, 6),
            })

        # ═══════════════════════════════════════════════════════════════════
        # MO — BISECCIÓN
        # ═══════════════════════════════════════════════════════════════════
        if metodo == "MO — Bisección":
            h = max((hi-lo)*1e-5, 1e-9)
            fp_num = lambda xv: (phi(xv+h,a_val) - phi(xv-h,a_val))/(2*h)
            a_b, b_b = float(lo), float(hi)
            # buscar cambio de signo en φ'
            fa_s, fb_s = fp_num(a_b), fp_num(b_b)
            if fa_s * fb_s > 0:
                xs_ = np.linspace(a_b, b_b, 500)
                for _i in range(len(xs_)-1):
                    if fp_num(xs_[_i])*fp_num(xs_[_i+1]) <= 0:
                        a_b, b_b = float(xs_[_i]), float(xs_[_i+1]); break
            hist_b = []
            for k in range(300):
                xm = (a_b+b_b)/2.0; fpm = fp_num(xm); largo = b_b-a_b
                try:
                    f1v = float(f1_call(xm)); f2v = float(f2(xm))
                    phiv= float(phi(xm, a_val))
                except Exception:
                    f1v = f2v = phiv = float("nan")
                hist_b.append({
                    "k":          k,
                    "a_k":        round(a_b,   8),
                    "b_k":        round(b_b,   8),
                    "x_k":        round(xm,    8),
                    "φ(x_k,α)":  round(phiv,  8),
                    "φ'(x_k,α)": round(fpm,   8),
                    "b−a":        round(largo,  8),
                    "f₁(x_k)":   round(f1v,   6),
                    "f₂(x_k)":   round(f2v,   6),
                    "α":          round(a_val, 4),
                    "converged":  largo < 1e-8 or abs(fpm) < 1e-8,
                })
                if largo < 1e-8 or abs(fpm) < 1e-8: break
                if fp_num(a_b)*fpm < 0: b_b = xm
                else:                   a_b = xm
            # Adjuntar tabla de referencia α al final como filas separadas
            hist_b.append({"k": "──", "a_k": "── TABLA DE REFERENCIA α ──",
                            "b_k":"", "x_k":"", "φ(x_k,α)":"",
                            "φ'(x_k,α)":"","b−a":"","f₁(x_k)":"","f₂(x_k)":"",
                            "α":"","converged":""})
            for row in tabla_ref:
                hist_b.append({
                    "k":          "ref",
                    "a_k":        row["α"],
                    "b_k":        row["f(x,α)"],
                    "x_k":        row["x*(α)"],
                    "φ(x_k,α)":  row["φ(x*,α)"],
                    "φ'(x_k,α)": "",
                    "b−a":        "",
                    "f₁(x_k)":   row["f₁(x*)"],
                    "f₂(x_k)":   row["f₂(x*)"],
                    "α":          row["α"],
                    "converged":  "",
                })
            mo_context = {
                "tabla_ref": tabla_ref,
                "hist_iters": [r for r in hist_b if r.get("k") not in ("──","ref")],
                "alpha": a_val, "x_c": x_c,
                "f1_call": f1_call, "f2_call": f2, "var1": var1,
            }
            return hist_b, mo_context

        # ═══════════════════════════════════════════════════════════════════
        # MO — SECCIÓN DORADA
        # ═══════════════════════════════════════════════════════════════════
        if metodo == "MO — Sección Dorada":
            p_g = (_math.sqrt(5)-1)/2
            ag_, bg_ = float(lo), float(hi)
            cg_ = bg_-p_g*(bg_-ag_); dg_ = ag_+p_g*(bg_-ag_)
            fcg_ = phi(cg_,a_val); fdg_ = phi(dg_,a_val)
            hist_g = []
            for k in range(300):
                largo = bg_-ag_; xm = (ag_+bg_)/2
                try:
                    f1v = float(f1_call(xm)); f2v = float(f2(xm))
                    phiv= float(phi(xm, a_val))
                except Exception:
                    f1v = f2v = phiv = float("nan")
                hist_g.append({
                    "k":         k,
                    "a_k":       round(ag_,  8),
                    "b_k":       round(bg_,  8),
                    "c_k":       round(cg_,  8),
                    "φ(c)":      round(fcg_, 8),
                    "d_k":       round(dg_,  8),
                    "φ(d)":      round(fdg_, 8),
                    "x_k":       round(xm,   8),
                    "φ(x_k,α)": round(phiv, 8),
                    "b−a":       round(largo, 8),
                    "φ":         round(p_g,  6),
                    "f₁(x_k)":  round(f1v,  6),
                    "f₂(x_k)":  round(f2v,  6),
                    "α":         round(a_val, 4),
                    "converged": largo < 1e-8,
                })
                if largo < 1e-8: break
                if fcg_ < fdg_:
                    bg_=dg_; dg_=cg_; fdg_=fcg_
                    cg_=bg_-p_g*(bg_-ag_); fcg_=phi(cg_,a_val)
                else:
                    ag_=cg_; cg_=dg_; fcg_=fdg_
                    dg_=ag_+p_g*(bg_-ag_); fdg_=phi(dg_,a_val)
            xopt_g = (ag_+bg_)/2
            try:
                hist_g[-1]["x_k"]       = round(xopt_g, 8)
                hist_g[-1]["φ(x_k,α)"] = round(phi(xopt_g,a_val), 8)
                hist_g[-1]["f₁(x_k)"]  = round(float(f1_call(xopt_g)), 6)
                hist_g[-1]["f₂(x_k)"]  = round(float(f2(xopt_g)), 6)
            except Exception:
                pass
            # Adjuntar tabla de referencia α
            hist_g.append({"k":"──","a_k":"── TABLA DE REFERENCIA α ──",
                            "b_k":"","c_k":"","φ(c)":"","d_k":"","φ(d)":"",
                            "x_k":"","φ(x_k,α)":"","b−a":"","φ":"",
                            "f₁(x_k)":"","f₂(x_k)":"","α":"","converged":""})
            for row in tabla_ref:
                hist_g.append({
                    "k":         "ref",
                    "a_k":       row["α"],
                    "b_k":       row["f(x,α)"],
                    "c_k": "", "φ(c)":"","d_k":"","φ(d)":"",
                    "x_k":       row["x*(α)"],
                    "φ(x_k,α)": row["φ(x*,α)"],
                    "b−a": "", "φ":"",
                    "f₁(x_k)":  row["f₁(x*)"],
                    "f₂(x_k)":  row["f₂(x*)"],
                    "α":         row["α"],
                    "converged": "",
                })
            mo_context = {
                "tabla_ref": tabla_ref,
                "hist_iters": [r for r in hist_g if r.get("k") not in ("──","ref")],
                "alpha": a_val, "x_c": x_c,
                "f1_call": f1_call, "f2_call": f2, "var1": var1,
            }
            return hist_g, mo_context

        # ═══════════════════════════════════════════════════════════════════
        # MO — FRENTE DE PARETO
        # ═══════════════════════════════════════════════════════════════════
        if metodo == "MO — Frente de Pareto":
            result = mo.pareto_front_generic(
                func_expr=norm_fx_mo, var_name=var1,
                x0=x0_start, lo=lo, hi=hi, n_alpha=80)
            mo_context = {
                "tabla_ref": tabla_ref, "hist_iters": None, "alpha": a_val,
                "x_c": x_c, "f1_call": f1_call, "f2_call": f2, "var1": var1,
            }
            return result["history"], mo_context

        # ═══════════════════════════════════════════════════════════════════
        # MO — ANÁLISIS JACOBIANO
        # ═══════════════════════════════════════════════════════════════════
        if metodo == "MO — Análisis Jacobiano":
            rows_j = mo.jacobian_generic(
                func_expr=norm_fx_mo, var_name=var1,
                x0=x0_start, lo=lo, hi=hi, n_points=15)
            mo_context = {
                "tabla_ref": tabla_ref, "hist_iters": None, "alpha": a_val,
                "x_c": x_c, "f1_call": f1_call, "f2_call": f2, "var1": var1,
            }
            return rows_j, mo_context

    return [], None


# Duplicado deliberado de app.ui.main_window.METHODS_MULTIOBJ: ese registro es
# UI (alimenta el combo de métodos) y esta capa no debe depender de la UI.
# Si METHODS_MULTIOBJ cambia, actualizar también aquí.
_METHODS_MULTIOBJ = frozenset({
    "Escalarización (Suma Ponderada)",
    "MO — Bisección",
    "MO — Sección Dorada",
    "MO — Frente de Pareto",
    "MO — Análisis Jacobiano",
})

def _is_multiobj(metodo: str) -> bool:
    return metodo in _METHODS_MULTIOBJ
