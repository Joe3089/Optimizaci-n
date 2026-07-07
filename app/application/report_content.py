# app/application/report_content.py
# Generación de contenido de reporte (descripciones, clasificación, paso a paso,
# gráficas para exportación) — extraído de app/ui/main_window.py (Fase 4a).
# Funciones puras: no dependen de Qt ni de estado de instancia de la UI.
from __future__ import annotations

import io
import os
import tempfile
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


# ── Evaluación numérica de expresiones (compartido con app.ui.main_window) ───
def build_callable(expr, vnames):
    """Crea callable numpy para una expresión con las variables dadas.
    Normaliza la expresión primero (^→**, ln→log, etc.) y maneja
    funciones especiales de sympy (DiracDelta, Heaviside)."""
    if not _SYMPY: raise RuntimeError("SymPy no disponible.")
    fc = _get_fc()
    norm = fc.normalize_expr(expr) if fc else expr
    # Construir símbolos
    lmap = {v: sp.Symbol(v) for v in vnames}
    f_sym = sp.sympify(norm, locals=lmap)
    modules = [
        {
            "DiracDelta"  : lambda *a: 0.0,
            "Heaviside"   : lambda x, *a: np.heaviside(np.asarray(x, dtype=float), 0.5),
            "conjugate"   : np.conj,
            "re"          : np.real,
            "im"          : np.imag,
        },
        "numpy",
    ]
    return sp.lambdify([lmap[v] for v in vnames], f_sym, modules=modules)

def evaluate_1d(expr, vnames, xs):
    return np.asarray(build_callable(expr, vnames)(xs), dtype=float)

def make_grid(lo, hi, n=400):
    if lo==hi: lo-=1; hi+=1
    if lo>hi:  lo,hi=hi,lo
    return np.linspace(lo, hi, max(50,int(n)))


def load_logo(path: str, max_px: int = 200) -> Optional[io.BytesIO]:
    """Carga el logo desde `path`, lo reduce a `max_px` (mantiene proporción)
    y lo devuelve como buffer PNG en memoria — evita incrustar el asset
    original (puede ser de alta resolución) tal cual en cada reporte
    exportado. Devuelve None si el archivo no existe o no se puede leer."""
    if not path or not os.path.exists(path):
        return None
    try:
        from PIL import Image as PILImage
        img = PILImage.open(path)
        img.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception:
        return None


def generate_plot_png(entry: dict) -> list:
    """Genera PNGs 2D y 3D de alta calidad para exportación, con misma estética que pantalla."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs  = []
    fx    = entry["fx"]
    vrs   = entry["vars"]
    lo    = entry["lo"]; hi = entry["hi"]
    traj  = entry["traj"]
    metod = entry["metodo"]

    BG = "#0a1228"; CC = "#4a9eff"; CT = "#ff6b35"; CM = "#ffd700"; CG = "#1a2e58"

    def _sax2(ax):
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_color("#2a3e6e")
        ax.tick_params(colors="#7aaaea", labelsize=8)
        ax.xaxis.label.set_color("#7aaaea"); ax.yaxis.label.set_color("#7aaaea")
        ax.title.set_color("#d8e8f8"); ax.grid(True, color=CG, alpha=0.4, lw=0.6)

    def _sax3(ax):
        ax.set_facecolor(BG)
        for pn in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pn.fill=False; pn.set_edgecolor("#1a2e58")
        ax.tick_params(colors="#7aaaea", labelsize=7)
        for lbl in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
            lbl.set_color("#7aaaea")

    # ── 2D ────────────────────────────────────────────────────────────────
    try:
        fig2, ax = plt.subplots(figsize=(8,5), facecolor=BG)
        _sax2(ax)

        if len(vrs) == 1:
            xg = make_grid(lo, hi, 500); yg = evaluate_1d(fx, vrs, xg)
            ax.plot(xg, yg, color=CC, lw=2.2, label=f"f(x) = {fx}", zorder=2)
            if traj and traj[0].size > 0:
                tx = traj[0]; ty = evaluate_1d(fx, vrs, tx)
                for i in range(len(tx)-1):
                    al = 0.3 + 0.7*(i/max(len(tx)-1,1))
                    ax.plot(tx[i:i+2], ty[i:i+2], '-', color=CT, lw=2, alpha=al, zorder=4)
                ax.scatter(tx, ty, c=range(len(tx)), cmap='YlOrRd',
                           s=45, zorder=5, edgecolors='none', label="Iteraciones")
                ax.plot(tx[0],  ty[0],  'o', color="#00e5ff", ms=9,
                        label=f"x₀={tx[0]:.4f}", zorder=6)
                ax.plot(tx[-1], ty[-1], '*', color=CM, ms=16,
                        label=f"x*={tx[-1]:.4f}, f*={ty[-1]:.4f}", zorder=7)
                ax.axvline(tx[-1], color=CM, lw=0.8, ls='--', alpha=0.5)
                ax.axhline(ty[-1], color=CM, lw=0.8, ls='--', alpha=0.5)
                yrng = max(yg)-min(yg) if max(yg)!=min(yg) else 1
                ax.annotate(f"x*={tx[-1]:.4f}  f*={ty[-1]:.4f}",
                            xy=(tx[-1], ty[-1]),
                            xytext=(tx[-1]+0.06*(hi-lo), ty[-1]+0.06*yrng),
                            color=CM, fontsize=8,
                            arrowprops=dict(arrowstyle='->', color=CM, lw=1.2))
            ax.set_xlabel(vrs[0]); ax.set_ylabel("f(x)")
        else:
            g = np.linspace(min(lo,hi), max(lo,hi), 130)
            X1, X2 = np.meshgrid(g, g)
            Z = np.asarray(build_callable(fx, vrs[:2])(X1, X2), dtype=float)
            cf = ax.contourf(X1, X2, Z, levels=30, cmap='coolwarm', alpha=0.88)
            ax.contour(X1, X2, Z, levels=12, colors='white', alpha=0.2, lw=0.5)
            fig2.colorbar(cf, ax=ax, fraction=0.035, pad=0.04).ax.tick_params(colors="#7aaaea")
            if traj and len(traj)>=2 and traj[0].size>0:
                px, py = traj[0], traj[1]
                ax.plot(px, py, '-', color=CT, lw=2, alpha=0.8, zorder=4)
                ax.scatter(px, py, c=range(len(px)), cmap='YlOrRd',
                           s=50, zorder=5, edgecolors='none', label="Iteraciones")
                ax.plot(px[0],  py[0],  'o', color="#00e5ff", ms=9, label="x₀", zorder=6)
                ax.plot(px[-1], py[-1], '*', color=CM, ms=16, label="x*", zorder=7)
                rng = hi-lo if hi!=lo else 1
                ax.annotate(f"x*=({px[-1]:.3f},{py[-1]:.3f})",
                            xy=(px[-1], py[-1]),
                            xytext=(px[-1]+0.06*rng, py[-1]+0.06*rng),
                            color=CM, fontsize=8,
                            arrowprops=dict(arrowstyle='->', color=CM, lw=1.2))
            ax.set_xlabel(vrs[0]); ax.set_ylabel(vrs[1])

        ax.set_title(f"2D — {metod}", fontsize=11, pad=8)
        ax.legend(facecolor="#141e36", edgecolor="#2a3e6e",
                  labelcolor="#d8e8f8", fontsize=8,
                  loc="upper left",
                  ncol=1, framealpha=0.88,
                  borderaxespad=0.5)
        fig2.tight_layout()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close(fig2); buf2.seek(0); imgs.append((buf2, "Gráfica 2D"))
    except Exception as e:
        print(f"[plot2d export err] {e}")

    # ── 3D ────────────────────────────────────────────────────────────────
    try:
        for azim in (-55, 30, 110):   # 3 vistas estáticas del 3D
            fig3 = plt.figure(figsize=(7, 5), facecolor=BG)
            ax3  = fig3.add_subplot(111, projection='3d'); _sax3(ax3)

            if len(vrs) == 1:
                xg = make_grid(lo, hi, 130); yg = evaluate_1d(fx, vrs, xg)
                it = np.arange(xg.size, dtype=float)
                ax3.plot(it, xg, yg, color=CC, lw=1.8, label="f(x)", alpha=0.9)
                if traj and traj[0].size > 0:
                    tx = traj[0]; ty = evaluate_1d(fx, vrs, tx)
                    pit = np.arange(tx.size, dtype=float)
                    ax3.plot(pit, tx, ty, 'o-', color=CT, lw=2.5,
                             markersize=6, label="Trayectoria", zorder=5)
                    ax3.plot([pit[0]],  [tx[0]],  [ty[0]],  'o', color="#00e5ff", ms=9, label="x₀")
                    ax3.plot([pit[-1]], [tx[-1]], [ty[-1]], '*', color=CM, ms=14, label="x*")
                ax3.set_xlabel("iter."); ax3.set_ylabel(vrs[0]); ax3.set_zlabel("f(x)")
            else:
                g = np.linspace(min(lo,hi), max(lo,hi), 60)
                X1, X2 = np.meshgrid(g, g)
                f2c = build_callable(fx, vrs[:2])
                Z   = np.asarray(f2c(X1, X2), dtype=float)
                ax3.plot_surface(X1, X2, Z, cmap='coolwarm', alpha=0.80,
                                 linewidth=0, antialiased=True, rcount=60, ccount=60)
                if traj and len(traj)>=2 and traj[0].size>0:
                    px, py = traj[0], traj[1]
                    pz = np.asarray(f2c(px, py), dtype=float)
                    zmin = float(Z.min())
                    ax3.plot(px, py, [zmin]*len(px), '--', color=CT, lw=1.2, alpha=0.35)
                    ax3.plot(px, py, pz, 'o-', color=CT, lw=2.5,
                             markersize=6, label="Trayectoria", zorder=5)
                    ax3.plot([px[0]],  [py[0]],  [pz[0]],  'o', color="#00e5ff", ms=9, label="x₀")
                    ax3.plot([px[-1]], [py[-1]], [pz[-1]], '*', color=CM, ms=14, label="x*")
                ax3.set_xlabel(vrs[0]); ax3.set_ylabel(vrs[1]); ax3.set_zlabel("f(x)")

            lbl_azim = {-55:"Vista Frontal", 30:"Vista Lateral", 110:"Vista Superior"}
            ax3.set_title(f"3D — {metod}  ({lbl_azim.get(azim,'')})",
                          fontsize=10, color="#d8e8f8", pad=10)
            _h3, _l3 = ax3.get_legend_handles_labels()
            if _h3:
                fig3.legend(_h3, _l3,
                           facecolor="#141e36", edgecolor="#2a3e6e",
                           labelcolor="#d8e8f8", fontsize=8,
                           loc="upper left",
                           bbox_to_anchor=(0.02, 0.97),
                           bbox_transform=fig3.transFigure,
                           ncol=1, framealpha=0.88,
                           borderaxespad=0.5)
            ax3.view_init(elev=28, azim=azim)
            fig3.tight_layout()
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
            plt.close(fig3)
            buf3.seek(0)
            imgs.append((buf3, f"Gráfica 3D ({lbl_azim.get(azim,'')})"))
    except Exception as e:
        print(f"[plot3d export err] {e}")

    return imgs


# ══════════════════════════════════════════════════════════════════════════
# MÉTODOS DE DESCRIPCIÓN PARA PDF / EXCEL
# ══════════════════════════════════════════════════════════════════════════

def describe_function(fx: str) -> str:
    """Descripción breve y matemática de la función."""
    fx_l = fx.lower().replace(" ", "")
    if any(k in fx_l for k in ("sin", "cos", "tan")):
        return (f"La función f(x) = {fx} es de tipo trigonométrico. "
                "Este tipo de funciones presenta comportamiento oscilatorio periódico, "
                "con múltiples máximos y mínimos locales distribuidos uniformemente. "
                "Su análisis requiere considerar el intervalo de búsqueda con cuidado "
                "para evitar convergencia a óptimos locales indeseados.")
    if any(k in fx_l for k in ("exp", "log", "ln")):
        return (f"La función f(x) = {fx} es de tipo exponencial o logarítmico. "
                "Estas funciones presentan crecimiento/decrecimiento monótono o "
                "con un único punto crítico, lo que generalmente facilita la convergencia "
                "de los métodos de optimización. Su derivada es continua y bien definida "
                "en el dominio de interés.")
    if "**" in fx_l or "sqrt" in fx_l:
        import re
        degs = [int(m) for m in re.findall(r"\*\*(\d+)", fx_l)]
        grado = max(degs) if degs else 2
        return (f"La función f(x) = {fx} es un polinomio de grado {grado}. "
                f"{'Al ser de grado par, posee un mínimo o máximo global y posiblemente mínimos locales. ' if grado % 2 == 0 else 'Al ser de grado impar, la función no está acotada en ambos extremos del dominio. '}"
                "La derivada es un polinomio de grado inferior, lo que permite calcular "
                "analíticamente el gradiente necesario para los métodos de descenso.")
    if "/" in fx_l:
        return (f"La función f(x) = {fx} es de tipo racional. "
                "Este tipo de funciones puede presentar discontinuidades (polos) en los "
                "puntos donde el denominador se anula. Es fundamental verificar que el "
                "intervalo de búsqueda no contenga dichos puntos para garantizar la "
                "correcta aplicación de los métodos numéricos.")
    return (f"La función f(x) = {fx} es una función polinomial o algebraica. "
            "Presenta un comportamiento suave y diferenciable en todo su dominio, "
            "lo que la hace adecuada para la aplicación de métodos de optimización "
            "basados en gradiente. Su análisis permite identificar claramente "
            "puntos críticos mediante condiciones de primer y segundo orden.")

def describe_applicable_studies(fx: str) -> str:
    """Explica qué tipos de estudio son aplicables a la función y por qué."""
    fx_l = fx.lower().replace(" ", "")
    es_trig = any(k in fx_l for k in ("sin", "cos", "tan"))
    es_exp  = any(k in fx_l for k in ("exp", "log", "ln"))
    es_poly = "**" in fx_l or "sqrt" in fx_l or (not es_trig and not es_exp and "/" not in fx_l)

    partes = []
    partes.append(
        "Búsqueda Unidimensional: Aplicable cuando la función depende de una sola "
        "variable. Los métodos de sección áurea (Fibonacci) reducen iterativamente "
        "el intervalo de búsqueda sin necesitar derivadas, siendo robustos ante "
        "funciones no diferenciables o con ruido.")
    partes.append(
        "Búsqueda de Línea (Armijo / Wolfe): Adecuada para funciones diferenciables. "
        "Determina el tamaño de paso óptimo en la dirección de descenso. "
        "La condición de Armijo garantiza reducción suficiente del valor de la función, "
        "mientras que Wolfe añade la condición de curvatura para paso óptimo.")
    if es_trig or es_poly:
        partes.append(
            "Búsqueda Local: Explora el entorno de la solución actual mediante "
            "perturbaciones del punto corriente. Útil para funciones con múltiples "
            "extremos locales donde métodos de gradiente podrían quedar atrapados.")
    partes.append(
        "Optimización Multidimensional (MD): Cuando la función depende de varias "
        "variables. Los métodos de penalización y barrera transforman el problema "
        "con restricciones en uno sin restricciones. BFGS y Nelder-Mead son "
        "eficientes para funciones no lineales con múltiples variables.")
    return "  ".join(partes)

def justify_method(metodo: str, fx: str) -> str:
    """Explica por qué se seleccionó este método para esta función."""
    fx_l = fx.lower().replace(" ", "")
    justificaciones = {
        "Armijo": (
            "El método de Newton con condición de Armijo fue seleccionado porque la "
            f"función f(x) = {fx} es diferenciable en el dominio de análisis. "
            "Este método calcula la dirección de Newton usando el gradiente y la "
            "segunda derivada, luego aplica la condición de Armijo (condición de "
            "descenso suficiente) para elegir un tamaño de paso α que garantice "
            "una reducción real del valor de la función. La condición exige que "
            "f(x + α·d) ≤ f(x) + c·α·∇f(x)ᵀ·d, con c ∈ (0,1), lo que asegura "
            "convergencia evitando pasos demasiado grandes o pequeños."),
        "Wolfe": (
            "El método de Newton con condiciones de Wolfe fue seleccionado para "
            f"f(x) = {fx} por su mayor robustez comparado con Armijo. "
            "Además de la condición de descenso suficiente (Armijo), impone la "
            "condición de curvatura: |∇f(x+α·d)ᵀ·d| ≤ c₂·|∇f(x)ᵀ·d| con c₂ ∈ (c₁,1). "
            "Esto garantiza que el paso elegido no sea excesivamente corto, "
            "mejorando la velocidad de convergencia en funciones con curvatura variable."),
        "Fibonacci": (
            "La búsqueda de Fibonacci fue seleccionada porque es el método óptimo "
            "para minimizar una función unimodal en un intervalo cerrado sin "
            f"necesitar derivadas de f(x) = {fx}. "
            "Usando los números de Fibonacci para determinar los puntos de evaluación, "
            "reduce el intervalo de incertidumbre de forma óptima con el mínimo "
            "número de evaluaciones funcionales."),
        "Búsqueda Local": (
            "La búsqueda local por vecindad fue seleccionada como método exploratorio "
            f"sobre f(x) = {fx}. "
            "A partir del punto inicial, evalúa candidatos en el entorno inmediato "
            "(x ± paso) y se mueve al mejor vecino encontrado, repitiendo el proceso "
            "iterativamente."),
        "Sección Áurea": (
            f"La Sección Áurea fue seleccionada para f(x) = {fx} por ser el método "
            "de búsqueda unidimensional con convergencia garantizada en funciones "
            "unimodales. Utiliza la proporción τ = (√5−1)/2 ≈ 0.618 para reducir "
            "el intervalo de búsqueda en cada paso, reutilizando una evaluación "
            "de función por iteración. Es más eficiente que la bisección clásica."),
        "Goldstein": (
            f"Las condiciones de Goldstein fueron aplicadas a f(x) = {fx} para "
            "controlar el tamaño de paso con dos restricciones simultáneas: "
            "1) Sufficient decrease (evita pasos demasiado grandes), y "
            "2) Cota inferior (evita pasos demasiado pequeños). "
            "Con μ ∈ (0, 0.5), garantiza progreso real en cada iteración."),
        "Newton-Raphson": (
            f"El método de Newton-Raphson fue elegido para f(x) = {fx} por su "
            "convergencia cuadrática cerca del mínimo. Calcula la dirección de "
            "Newton d = −H⁻¹·∇f usando el gradiente y el Hessiano exactos "
            "(via diferenciación simbólica), lo que proporciona la actualización "
            "x_{k+1} = x_k − H⁻¹·∇f(x_k) con precisión máxima."),
        "Grad. Conjugado": (
            f"El Gradiente Conjugado (Fletcher-Reeves) fue seleccionado para "
            f"f(x) = {fx} por su eficiencia en funciones cuadráticas y no lineales. "
            "Genera direcciones conjugadas d_{k+1} = −∇f_{k+1} + β_k·d_k, donde "
            "β_k = ‖∇f_{k+1}‖²/‖∇f_k‖², logrando convergencia superlineal con "
            "solo información de primer orden."),
        "SA: Recocido Simulado": (
            f"El Recocido Simulado fue elegido para f(x) = {fx} por su capacidad "
            "de escapar mínimos locales. Inspirado en el enfriamiento de metales, "
            "acepta soluciones peores con probabilidad P=exp(−ΔE/T), donde T "
            "decrece gradualmente. Esto permite explorar el espacio de búsqueda "
            "globalmente antes de converger al mínimo."),
        "PSO: Enjambre": (
            f"PSO fue seleccionado para f(x) = {fx} por su eficacia en funciones "
            "multimodales sin derivadas. Un enjambre de partículas actualiza su "
            "velocidad combinando inercia propia, atracción a su mejor posición "
            "personal (c₁) y al mejor global del enjambre (c₂), explorando "
            "el espacio de forma paralela y cooperativa."),
        "GA: Algoritmo Genético": (
            f"El Algoritmo Genético fue aplicado a f(x) = {fx} por su robustez "
            "en espacios de búsqueda complejos y multimodales. Opera sobre una "
            "población de soluciones mediante selección por torneo, cruce aritmético "
            "y mutación gaussiana, con elitismo para preservar los mejores individuos. "
            "No requiere derivadas y puede encontrar el óptimo global."),
    }
    # MD methods
    if "Penalización" in metodo or "Pen." in metodo:
        return (f"El método de penalización fue seleccionado para resolver el problema "
                f"con restricciones sobre f(x) = {fx}. "
                "Transforma el problema restringido en una secuencia de problemas sin "
                "restricciones añadiendo un término de penalización μ·P(x) al objetivo, "
                "donde μ → ∞ fuerza el cumplimiento de las restricciones. "
                "Newton o BFGS resuelven cada subproblema, garantizando convergencia "
                "cuadrática o superlineal respectivamente.")
    if "Barrera" in metodo:
        return (f"El método de barrera (punto interior) fue seleccionado para f(x) = {fx} "
                "con restricciones de desigualdad. "
                "Añade términos logarítmicos −μ·ln(−gᵢ(x)) que tienden a infinito "
                "cuando x se acerca al borde factible, manteniendo la solución en el "
                "interior. A medida que μ → 0, la solución converge al óptimo restringido.")
    if "BFGS" in metodo:
        return (f"BFGS (Broyden–Fletcher–Goldfarb–Shanno) fue seleccionado para f(x) = {fx} "
                "como método cuasi-Newton eficiente. "
                "Aproxima la inversa del Hessiano usando solo gradientes de primer orden, "
                "logrando convergencia superlineal sin el costo computacional de calcular "
                "el Hessiano exacto. Es altamente efectivo para funciones suaves con "
                "múltiples variables.")
    if "Nelder-Mead" in metodo:
        return (f"Nelder-Mead fue seleccionado para f(x) = {fx} porque no requiere "
                "derivadas. Opera con un simplex que se deforma (reflexión, expansión, "
                "contracción) para explorar el espacio de búsqueda. Adecuado cuando "
                "la función no es diferenciable o el gradiente es costoso de evaluar.")
    return justificaciones.get(metodo,
        f"El método {metodo} fue aplicado a f(x) = {fx} como técnica de optimización "
        "numérica para encontrar el mínimo o máximo de la función en el dominio dado.")

def math_to_unicode(text: str) -> str:
    """
    Convierte notación matemática a superíndices/subíndices Unicode.
    Usado para celdas de Excel donde no hay HTML.
    """
    SUP_MAP = str.maketrans(
        "0123456789+-=()",
        "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    )
    SUB_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    def _sup(m):
        exp = m.group(2).strip("{}")
        return m.group(1) + exp.translate(SUP_MAP)

    def _sub(m):
        idx = m.group(2).strip("{}")
        return m.group(1) + idx.translate(SUB_MAP)

    import re as _re
    # x**{k+1}, x**2
    text = _re.sub(r'([a-zA-Zα-ω\)\]])\*\*(\{[^}]+\}|-?\d+)', _sup, text)
    # x^{n}, x^2
    text = _re.sub(r'([a-zA-Zα-ω\)\]])\^(\{[^}]+\}|-?\d+)', _sup, text)
    # x_{k+1}, x_k  (solo dígitos para subscript unicode)
    text = _re.sub(r'([a-zA-Zα-ω])_\{(\d+)\}', _sub, text)
    text = _re.sub(r'([a-zA-Zα-ω])_(\d+)',     _sub, text)
    # Fracciones comunes
    for raw, uni in [("1/2","½"),("1/3","⅓"),("1/4","¼"),
                     ("2/3","⅔"),("3/4","¾"),("1/8","⅛")]:
        text = text.replace(raw, uni)
    return text

def math_to_reportlab(text: str) -> str:
    """
    Convierte notación matemática a etiquetas XML de ReportLab
    (<super>, <sub>) para usar dentro de Paragraph().
    Aplica html.escape primero para proteger < y >.
    """
    import html as _hl
    import re as _re

    s = _hl.escape(text)

    # Superíndices: x**{k+1}, x**2, x^{n}, x^2
    s = _re.sub(r'([a-zA-Zα-ωΑ-Ω∇\)\]])\*\*\{([^}]+)\}',
                lambda m: f"{m.group(1)}<super>{m.group(2)}</super>", s)
    s = _re.sub(r'([a-zA-Zα-ωΑ-Ω∇\)\]])\*\*(-?\d+(?:\.\d+)?)',
                lambda m: f"{m.group(1)}<super>{m.group(2)}</super>", s)
    s = _re.sub(r'([a-zA-Zα-ωΑ-Ω\)\]])\^\{([^}]+)\}',
                lambda m: f"{m.group(1)}<super>{m.group(2)}</super>", s)
    s = _re.sub(r'([a-zA-Zα-ωΑ-Ω\)\]])\^(-?\d+)',
                lambda m: f"{m.group(1)}<super>{m.group(2)}</super>", s)
    # Subíndices: x_{k+1}, x_k
    s = _re.sub(r'([a-zA-Zα-ωΑ-Ω])_\{([^}]+)\}',
                lambda m: f"{m.group(1)}<sub>{m.group(2)}</sub>", s)
    s = _re.sub(r'([a-zA-Zα-ωΑ-Ω])_([a-zA-Z0-9\+\-\*]+)',
                lambda m: f"{m.group(1)}<sub>{m.group(2)}</sub>", s)
    # Fracciones comunes
    for raw, uni in [("1/2","½"),("1/3","⅓"),("1/4","¼"),
                     ("2/3","⅔"),("3/4","¾"),("1/8","⅛")]:
        s = s.replace(raw, uni)
    return s

def render_math_img_file(expr: str,
                          fontsize: float = 11,
                          fg: str = '#1a1a3a',
                          dpi: int = 150) -> 'str | None':
    """
    Renderiza una expresión mathtext a un archivo PNG temporal.
    Retorna la ruta del archivo o None si matplotlib no está disponible.
    Usado para incrustar fórmulas en PDF y Excel.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import tempfile, os
        import re as _re2

        # ── Convertir a mathtext ($...$) ──────────────────────────────
        s = expr.strip()
        UNI = {
            'α':r'\alpha','β':r'\beta','γ':r'\gamma','δ':r'\delta',
            'λ':r'\lambda','μ':r'\mu','σ':r'\sigma','ω':r'\omega',
            '∇':r'\nabla','∂':r'\partial','∞':r'\infty',
            '≤':r'\leq','≥':r'\geq','≠':r'\neq',
            '·':r'\cdot','×':r'\times','±':r'\pm','‖':r'\|',
        }
        for uc, lt in UNI.items():
            s = s.replace(uc, lt)
        # Potencias
        s = _re2.sub(r'([a-zA-Z0-9\)\|])\*\*\{([^}]+)\}', r'\1^{\2}', s)
        s = _re2.sub(r'([a-zA-Z0-9\)\|])\*\*(-?\d+)',      r'\1^{\2}', s)
        # Multiplicación
        s = _re2.sub(r'(?<!\^)(?<!\{)\*(?!\*)(?!\{)', r' \\cdot ', s)
        # exp / sqrt / fracciones
        s = _re2.sub(r'exp\(([^)]+)\)',  r'e^{\1}',        s)
        s = _re2.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}',    s)
        s = _re2.sub(r'\b(\d+)/(\d+)\b', r'\\frac{\1}{\2}', s)

        mathtext = f'${s}$'

        fig = plt.figure(figsize=(0.01, 0.01))
        fig.patch.set_facecolor('#e8f2ff')
        fig.text(0.5, 0.5, mathtext, fontsize=fontsize,
                 color=fg, ha='center', va='center')
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.close()
        fig.savefig(tmp.name, format='png', dpi=dpi,
                    bbox_inches='tight', pad_inches=0.12)
        plt.close(fig)
        return tmp.name
    except Exception:
        return None

def build_step_by_step(entry: dict) -> list:
    """
    Genera lista de strings con los cálculos paso a paso del método,
    derivados directamente del historial de iteraciones.
    Retorna: lista de (título, descripción) por iteración.
    """
    hist   = entry.get("hist", [])
    metodo = entry.get("metodo", "")
    fx     = entry.get("fx", "")
    lo     = entry.get("lo", 0)
    hi     = entry.get("hi", 1)
    x0     = entry.get("x0", [])
    steps  = []

    if not hist:
        return [("Sin datos", "No se registraron iteraciones para este método.")]

    # ── Paso 0: Condiciones iniciales ─────────────────────────────────────
    x0_str = (f"[{', '.join(f'{v:.4g}' for v in x0)}]" if x0
              else "no especificado")
    steps.append((
        "Condiciones iniciales",
        f"Función objetivo: f(x) = {fx}\n"
        f"Intervalo de búsqueda: [{lo:.4g}, {hi:.4g}]\n"
        f"Punto inicial: x₀ = {x0_str}\n"
        f"Método aplicado: {metodo}"
    ))

    # ── Pasos por iteración ───────────────────────────────────────────────
    for i, rec in enumerate(hist):
        titulo = f"Iteración {i + 1}"

        # Extraer campos comunes
        k      = rec.get("k",    rec.get("iter", i))
        x_k    = rec.get("x_k",  rec.get("x",    rec.get("lambda_k", None)))
        f_k    = rec.get("f_k",  rec.get("f(x)", rec.get("f_new",    rec.get("f_lambda", None))))
        alpha  = rec.get("alpha_k", rec.get("alpha", rec.get("paso", None)))
        grad   = rec.get("grad_norm", None)
        d_k    = rec.get("d_k_str",   rec.get("d_k", None))

        def _fmt(v):
            if v is None: return "N/A"
            if isinstance(v, np.ndarray):
                return "[" + ", ".join(f"{x:.5g}" for x in v.flat) + "]"
            try:    return f"{float(v):.6g}"
            except: return str(v)

        lines = []

        # Armijo / Wolfe
        if metodo in ("Armijo", "Wolfe"):
            armijo_ok = rec.get("armijo_ok", None)
            curv_ok   = rec.get("curv_ok",   None)
            x_new     = rec.get("x_new",     x_k)
            f_new     = rec.get("f_new",      f_k)
            lines.append(f"  Punto actual:       x_k = {_fmt(x_k)}")
            lines.append(f"  Norma del gradiente: ‖∇f(x_k)‖ = {_fmt(grad)}")
            if d_k is not None:
                lines.append(f"  Dirección de descenso: d_k = {_fmt(d_k)}")
            lines.append(f"  Tamaño de paso:      α = {_fmt(alpha)}")
            lines.append(f"  Nuevo punto:         x_{{k+1}} = x_k + α·d_k = {_fmt(x_new)}")
            lines.append(f"  Valor de la función: f(x_{{k+1}}) = {_fmt(f_new)}")
            if armijo_ok is not None:
                lines.append(f"  Condición Armijo cumplida: {'✓ Sí' if armijo_ok else '✗ No'}")
            if curv_ok is not None:
                lines.append(f"  Condición curvatura (Wolfe) cumplida: {'✓ Sí' if curv_ok else '✗ No'}")

        # Fibonacci / Búsqueda Local
        elif metodo in ("Fibonacci", "Búsqueda Local"):
            a_i   = rec.get("a", rec.get("a_n", None))
            b_i   = rec.get("b", rec.get("b_n", None))
            x_c   = rec.get("x_candidate", rec.get("x1", rec.get("x2", None)))
            f_c   = rec.get("f_candidate", rec.get("f1", rec.get("f2", None)))
            improved = rec.get("improved", None)
            lines.append(f"  Punto actual:      x = {_fmt(x_k)},  f(x) = {_fmt(f_k)}")
            if a_i is not None and b_i is not None:
                lines.append(f"  Intervalo activo:  [{_fmt(a_i)},  {_fmt(b_i)}]")
                lines.append(f"  Longitud intervalo: Δ = {abs(float(b_i)-float(a_i)):.5g}" if
                             b_i is not None and a_i is not None else "")
            if x_c is not None:
                lines.append(f"  Candidato evaluado: x_c = {_fmt(x_c)},  f(x_c) = {_fmt(f_c)}")
            if alpha is not None:
                lines.append(f"  Paso utilizado:    δ = {_fmt(alpha)}")
            if improved is not None:
                lines.append(f"  Mejora encontrada: {'✓ Sí → actualizar punto' if improved else '✗ No → mantener punto'}")

        # MD methods
        else:
            x_str = rec.get("x_k_str", rec.get("x_str", None))
            f_mu  = rec.get("f_mu",  rec.get("f_lambda", f_k))
            mu    = rec.get("mu",    rec.get("lambda",  None))
            lines.append(f"  Iteración global k = {_fmt(k)}")
            lines.append(f"  Punto x_k = {_fmt(x_k)}")
            if x_str: lines.append(f"  Representación: {x_str}")
            lines.append(f"  f(x_k) = {_fmt(f_k)}")
            if f_mu is not None and f_mu != f_k:
                lines.append(f"  f aumentada/penalizada = {_fmt(f_mu)}")
            if mu is not None:
                lines.append(f"  Parámetro μ/λ = {_fmt(mu)}")
            if grad is not None:
                lines.append(f"  ‖∇f(x_k)‖ = {_fmt(grad)}")
            # Resto de campos no estándar
            for kk, vv in rec.items():
                if kk not in ("k","iter","x_k","x","x_k_str","f_k","f_mu",
                              "f_lambda","mu","lambda","grad_norm","x_str"):
                    lines.append(f"  {kk} = {_fmt(vv)}")

        steps.append((titulo, "\n".join(l for l in lines if l.strip())))

    # ── Paso final: conclusión ────────────────────────────────────────────
    last = hist[-1]
    x_final = last.get("x_k", last.get("x", last.get("lambda_k", None)))
    f_final = last.get("f_k", last.get("f(x)", last.get("f_new",
              last.get("f_lambda", None))))
    def _fv(v):
        if v is None: return "N/D"
        if isinstance(v, np.ndarray):
            return "[" + ", ".join(f"{x:.6g}" for x in v.flat) + "]"
        try:    return f"{float(v):.6g}"
        except: return str(v)

    steps.append((
        "Resultado final",
        f"Tras {len(hist)} iteración(es) el método convergió a:\n"
        f"  x* = {_fv(x_final)}\n"
        f"  f(x*) = {_fv(f_final)}\n"
        f"El valor de x* representa el {'mínimo' if f_final is not None and float(f_final if isinstance(f_final,(int,float)) else 0) < 0 else 'óptimo'} "
        f"encontrado por {metodo} en el dominio [{lo:.4g}, {hi:.4g}]."
    ))
    return steps

def classify_study(metodo: str) -> str:
    """Clasifica el tipo de estudio según el método."""
    md_keywords = ("MD","Penalización","Pen.","Barrera","BFGS","Nelder","Newton MD","Grad. Conj.")
    if any(k in metodo for k in md_keywords):
        return "Optimización Multidimensional (MD)"
    if metodo == "Fibonacci":
        return "Búsqueda Unidimensional — Sección Áurea / Fibonacci"
    if metodo == "Sección Áurea":
        return "Heurístico — Búsqueda por Proporción Áurea"
    if metodo == "Goldstein":
        return "Heurístico — Búsqueda de Línea (Condiciones de Goldstein)"
    if metodo == "Newton-Raphson":
        return "Heurístico — Método de Newton-Raphson (2ᵃ derivada)"
    if metodo == "Grad. Conjugado":
        return "Heurístico — Gradiente Conjugado (Fletcher-Reeves)"
    if metodo in ("Armijo", "Wolfe"):
        return "Búsqueda de Línea — Condiciones de Wolfe/Armijo"
    if "Recocido" in metodo or "SA:" in metodo:
        return "Metaheurístico — Recocido Simulado (Simulated Annealing)"
    if "PSO" in metodo or "Enjambre" in metodo:
        return "Metaheurístico — Optimización por Enjambre de Partículas (PSO)"
    if "GA:" in metodo or "Genético" in metodo:
        return "Metaheurístico — Algoritmo Genético (GA)"
    return "Búsqueda Unidimensional Local"

def classify_function(fx: str) -> str:
    """Clasifica la función por tipo matemático, incluyendo funciones de prueba conocidas."""
    fx_l = fx.lower().replace(" ", "")
    # Funciones de prueba conocidas del PDF
    if "rosenbrock" in fx_l or ("(1-x)**2" in fx_l and "100" in fx_l):
        return "No Convexa (Rosenbrock)"
    if "himmelblau" in fx_l or ("x**2+y-11" in fx_l):
        return "Multimodal (Himmelblau)"
    if "rastrigin" in fx_l or ("10*cos(2*pi*x)" in fx_l):
        return "Multimodal (Rastrigin)"
    if "griewank" in fx_l or ("4000" in fx_l and "cos(x)" in fx_l):
        return "Multimodal (Griewank)"
    if "ackley" in fx_l or ("20*exp" in fx_l and "cos(2*pi" in fx_l):
        return "Multimodal (Ackley)"
    if "beale" in fx_l or ("1.5-x+x*y" in fx_l.replace(" ", "")):
        return "No Convexa (Beale)"
    # Por características matemáticas
    fc = _get_fc()
    if fc:
        vars_ = fc.detect_variables(fx)
        n = len(vars_)
    else:
        n = 1
    dim_str = f" ({n}D)" if n > 1 else ""
    if any(k in fx_l for k in ("sin", "cos", "tan", "pi")):
        return f"Trigonométrica{dim_str}"
    if any(k in fx_l for k in ("exp", "log", "ln")):
        return f"Exponencial/Logarítmica{dim_str}"
    if "**" in fx_l or "sqrt" in fx_l:
        import re
        degs = [int(m) for m in re.findall(r"\*\*(\d+)", fx_l)]
        mx = max(degs) if degs else 2
        return f"Polinomial grado {mx}{dim_str}"
    if "/" in fx_l:
        return f"Racional{dim_str}"
    return f"Polinomial/Algebraica{dim_str}"

def build_conclusion(session: list, fx: str) -> str:
    """Genera conclusión general basada en los métodos ejecutados."""
    if not session: return "No se ejecutaron métodos."
    n = len(session)
    metodos = [s["metodo"] for s in session]
    resultados = []
    for s in session:
        h = s["hist"]
        if h:
            last = h[-1]
            for kx in ("x_k","x","lambda_k"):
                v = last.get(kx)
                if v is None: continue
                # safe extraction regardless of type
                try:
                    if isinstance(v, np.ndarray):
                        xs = float(v.flat[0])
                    elif isinstance(v, (list, tuple)) and len(v) > 0:
                        xs = float(v[0])
                    else:
                        xs = float(v)
                    resultados.append(f"{s['metodo']}: x*≈{xs:.4f}")
                except Exception:
                    pass
                break
    lines = [
        f"Se aplicaron {n} método(s) de optimización a la función f(x) = {fx}.",
        "Métodos utilizados: " + ", ".join(metodos) + ".",
    ]
    if resultados:
        lines.append("Resultados obtenidos — " + ";  ".join(resultados) + ".")
    lines.append(
        "Los métodos de búsqueda de línea (Armijo, Wolfe) garantizan "
        "descenso suficiente en cada iteración. Los métodos de región de "
        "confianza como Fibonacci son robustos ante funciones no convexas. "
        "Los métodos MD con penalización/barrera manejan restricciones de "
        "igualdad o desigualdad."
    )
    return "  ".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# ÁREAS DE APLICACIÓN — para el selector de finalidad del reporte PDF
# ══════════════════════════════════════════════════════════════════════════
AREA_ORDER = ["ingenieria", "ia_ml", "economia", "ciencias", "robotica"]

_AREAS: Dict[str, Dict[str, Any]] = {
    "ingenieria": {
        "titulo": "Ingeniería y Diseño",
        "aplicacion": (
            "Los métodos de optimización son fundamentales en el diseño de "
            "estructuras, circuitos eléctricos, sistemas de control y rutas "
            "logísticas. La capacidad de encontrar mínimos de funciones de costo "
            "o máximos de funciones de rendimiento es clave en cualquier proceso "
            "de diseño óptimo."
        ),
        "metodos_recomendados": [
            "MD: Penalización (Newton)", "MD: Barreras (Newton)",
            "MD: Pen. (BFGS)", "Grad. Conjugado",
        ],
        "ventajas": (
            "Manejo nativo de restricciones de igualdad/desigualdad — las "
            "limitantes físicas reales de un diseño (esfuerzos máximos, "
            "capacidades, geometría) se incorporan directamente al problema."
        ),
        "beneficios": (
            "Reduce el tiempo de iteración manual de diseño y produce resultados "
            "reproducibles y documentados, en vez de ajustes por prueba y error."
        ),
        "innovacion": (
            "Permite explorar automáticamente el espacio de diseño y encontrar "
            "configuraciones no evidentes para el diseñador humano."
        ),
    },
    "ia_ml": {
        "titulo": "Inteligencia Artificial y Machine Learning",
        "aplicacion": (
            "El entrenamiento de redes neuronales es en esencia un problema de "
            "minimización de una función de pérdida en miles de dimensiones. "
            "Los métodos de descenso de gradiente con condiciones de Wolfe/Armijo "
            "son la base de optimizadores como Adam, SGD y L-BFGS usados en "
            "frameworks como TensorFlow y PyTorch."
        ),
        "metodos_recomendados": [
            "Armijo", "Wolfe", "SA: Recocido Simulado",
            "PSO: Enjambre", "GA: Algoritmo Genético",
        ],
        "ventajas": (
            "Las condiciones de Armijo/Wolfe garantizan convergencia estable del "
            "descenso de gradiente; los metaheurísticos (SA/PSO/GA) exploran "
            "globalmente para evitar mínimos locales pobres en problemas no "
            "convexos como la búsqueda de hiperparámetros."
        ),
        "beneficios": (
            "Acelera la búsqueda de hiperparámetros y arquitecturas, reduciendo "
            "el costo computacional total de entrenamiento."
        ),
        "innovacion": (
            "Es la base conceptual de los optimizadores usados en los "
            "frameworks de deep learning modernos."
        ),
    },
    "economia": {
        "titulo": "Economía y Finanzas",
        "aplicacion": (
            "La optimización de portafolios de inversión, la minimización de "
            "riesgo financiero y la maximización de utilidad en modelos "
            "econométricos dependen directamente de métodos numéricos robustos "
            "como los implementados en esta herramienta."
        ),
        "metodos_recomendados": [
            "Escalarización (Suma Ponderada)", "MO — Goal Programming",
            "MO — ε-Constraint",
        ],
        "ventajas": (
            "Modelan explícitamente el trade-off entre objetivos en conflicto "
            "(por ejemplo, riesgo vs. retorno), en vez de forzar una única "
            "métrica combinada arbitraria."
        ),
        "beneficios": (
            "Produce decisiones de inversión trazables y cuantificables en vez "
            "de heurísticas informales."
        ),
        "innovacion": (
            "Permite generar fronteras eficientes completas (frente de Pareto) "
            "en vez de una única solución de compromiso."
        ),
    },
    "ciencias": {
        "titulo": "Ciencias Naturales e Investigación",
        "aplicacion": (
            "En física, química y biología computacional, la minimización de "
            "energía de sistemas moleculares, la calibración de modelos y la "
            "estimación de parámetros de ecuaciones diferenciales son problemas "
            "de optimización donde estos métodos encuentran aplicación directa."
        ),
        "metodos_recomendados": [
            "Newton-Raphson", "Grad. Conjugado", "Sección Áurea",
        ],
        "ventajas": (
            "Convergencia rápida (cuadrática en Newton-Raphson) para las "
            "funciones suaves típicas de modelos físicos y experimentales."
        ),
        "beneficios": (
            "Acelera la calibración de modelos experimentales y reduce el "
            "número de simulaciones necesarias."
        ),
        "innovacion": (
            "Aplicable a la estimación de parámetros en modelos donde la "
            "derivada analítica está disponible simbólicamente."
        ),
    },
    "robotica": {
        "titulo": "Robótica y Control Automático",
        "aplicacion": (
            "La planificación de trayectorias de robots, el control predictivo "
            "de procesos industriales (MPC) y la calibración de controladores "
            "PID óptimos son aplicaciones directas de los métodos de optimización "
            "con restricciones estudiados en este reporte."
        ),
        "metodos_recomendados": [
            "MD: Pen. (BFGS)", "MD: Pes. (BFGS)",
            "MD: Pes. (Nelder-Mead)", "PSO: Enjambre",
        ],
        "ventajas": (
            "Soporta restricciones dinámicas y cinemáticas del robot de forma "
            "nativa, sin necesitar relajaciones artificiales del problema."
        ),
        "beneficios": (
            "Reduce el tiempo de sintonización manual de controladores y "
            "trayectorias."
        ),
        "innovacion": (
            "Permite re-optimizar trayectorias en tiempo real ante cambios del "
            "entorno."
        ),
    },
}


def describe_area_utility(area_key: str) -> Dict[str, Any]:
    """Devuelve el contenido estructurado (aplicación, métodos recomendados,
    ventajas, beneficios, innovación) para el área `area_key` — una de
    AREA_ORDER. Usado por el PDF para mostrar, según la finalidad que elija
    el usuario, una sola área o las cinco ("estudio general")."""
    try:
        return _AREAS[area_key]
    except KeyError:
        raise ValueError(f"Área desconocida: {area_key!r}. Disponibles: {AREA_ORDER}")
