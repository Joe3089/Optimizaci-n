# app/statistics/chebyshev.py
# Desigualdad de Chebyshev: P(|X − μ| < kσ) ≥ 1 − 1/k², válida para CUALQUIER
# distribución con media μ y desviación estándar σ finitas (k > 1).
# Resuelve ejercicios a partir de (μ, σ), (μ, σ²) o de los datos de una
# muestra, con k dado directamente o deducido de los límites de un intervalo.
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MODE_SIGMA = "Media y desviación estándar (μ, σ)"
MODE_VAR = "Media y varianza (μ, σ²)"
MODE_DATA = "Datos de una muestra"
MODES = (MODE_SIGMA, MODE_VAR, MODE_DATA)
VAR_SAMPLE = "Muestral (divide entre n − 1)"
VAR_POP = "Poblacional (divide entre n)"
VAR_TYPES = (VAR_SAMPLE, VAR_POP)

_SPLIT = re.compile(r"[,;\s]+")


def _num(text: str, name: str, required: bool = True) -> Optional[float]:
    t = (text or "").strip()
    if not t:
        if required:
            raise ValueError(f"Falta el dato «{name}». Escríbelo en el campo correspondiente.")
        return None
    try:
        v = float(t.replace(",", "."))
    except ValueError as ex:
        raise ValueError(f"«{name}» debe ser un número (se recibió «{t}»). "
                         "Usa punto decimal, por ejemplo 2.5.") from ex
    if not math.isfinite(v):
        raise ValueError(f"«{name}» debe ser un número finito.")
    return v


def _sample(text: str) -> List[float]:
    parts = [p for p in _SPLIT.split((text or "").strip()) if p]
    if not parts:
        raise ValueError("Elegiste «Datos de una muestra» pero no ingresaste datos. "
                         "Escríbelos separados por comas, por ejemplo: 45, 52, 61, 38.")
    try:
        x = [float(p) for p in parts]
    except ValueError as ex:
        raise ValueError("Hay un valor no numérico en los datos de la muestra. "
                         "Usa punto decimal y separa los valores con comas.") from ex
    if not all(math.isfinite(v) for v in x):
        raise ValueError("Los datos de la muestra deben ser números finitos.")
    return x


def _limits(text: str) -> Optional[Tuple[float, float]]:
    t = (text or "").strip()
    if not t:
        return None
    parts = [p for p in re.split(r"[;,\s]+", t.strip("[]() ")) if p]
    if len(parts) != 2:
        raise ValueError("Los límites del intervalo deben ser dos números «a, b», "
                         "por ejemplo: 30, 70.")
    a = _num(parts[0], "límite inferior a")
    b = _num(parts[1], "límite superior b")
    if a >= b:
        raise ValueError(f"El límite inferior ({a:g}) debe ser menor que el superior ({b:g}).")
    return a, b


def _g(v: float) -> str:
    return f"{v:.6g}"


def chebyshev(mode: str, mu_txt: str = "", sigma_txt: str = "", var_txt: str = "",
              data_txt: str = "", var_type: str = VAR_SAMPLE, k_txt: str = "",
              limits_txt: str = "") -> Dict[str, Any]:
    """Resuelve un ejercicio con la desigualdad de Chebyshev, paso a paso."""
    if mode not in MODES:
        raise ValueError(f"Modo de datos desconocido: «{mode}».")
    steps: List[Tuple[str, str, str]] = []   # (descripción, cálculo, resultado)
    data: Optional[np.ndarray] = None

    # 1-3. Datos, validación y estadísticos faltantes
    if mode == MODE_DATA:
        x = _sample(data_txt)
        pop = var_type == VAR_POP
        if len(x) < (1 if pop else 2):
            raise ValueError("Se necesitan al menos 2 datos para calcular la varianza muestral "
                             "(con 1 dato, n − 1 = 0).")
        data = np.asarray(x, dtype=float)
        n = data.size
        mu = float(data.mean())
        ddof = 0 if pop else 1
        var = float(data.var(ddof=ddof))
        div = "n" if pop else "n − 1"
        steps.append(("Datos de la muestra", f"n = {n}", ", ".join(_g(v) for v in x[:12])
                      + (" …" if n > 12 else "")))
        steps.append(("Media aritmética", f"μ = Σxᵢ / n = {_g(float(data.sum()))} / {n}", _g(mu)))
        steps.append((f"Varianza {'poblacional' if pop else 'muestral'}",
                      f"σ² = Σ(xᵢ − μ)² / ({div}) = {_g(float(((data - mu) ** 2).sum()))} / "
                      f"{n - ddof}", _g(var)))
        sigma = math.sqrt(var)
        steps.append(("Desviación estándar", f"σ = √σ² = √{_g(var)}", _g(sigma)))
    else:
        mu = _num(mu_txt, "media μ")
        if mode == MODE_VAR:
            var = _num(var_txt, "varianza σ²")
            if var < 0:
                raise ValueError(f"La varianza no puede ser negativa (se recibió σ² = {var:g}). "
                                 "Corrige el valor de σ².")
            sigma = math.sqrt(var)
            steps.append(("Datos iniciales", f"μ = {_g(mu)},  σ² = {_g(var)}", "—"))
            steps.append(("Desviación estándar", f"σ = √σ² = √{_g(var)}", _g(sigma)))
        else:
            sigma = _num(sigma_txt, "desviación estándar σ")
            if sigma < 0:
                raise ValueError(f"La desviación estándar no puede ser negativa (se recibió "
                                 f"σ = {sigma:g}). Corrige el valor de σ.")
            var = sigma ** 2
            steps.append(("Datos iniciales", f"μ = {_g(mu)},  σ = {_g(sigma)}", "—"))
    if sigma == 0:
        raise ValueError("La desviación estándar es 0: todos los valores son iguales a la media "
                         "y la desigualdad de Chebyshev no aporta información. "
                         "Revisa σ, σ² o los datos.")

    # 5. k: directo o deducido de los límites a, b
    lims = _limits(limits_txt)
    k_in = _num(k_txt, "k", required=lims is None)
    note = ""
    if lims is not None:
        a, b = lims
        if not (a < mu < b):
            raise ValueError(f"La media μ = {_g(mu)} debe quedar dentro del intervalo "
                             f"({_g(a)}, {_g(b)}) para aplicar la desigualdad.")
        d = min(mu - a, b - mu)
        k_lim = d / sigma
        sym = math.isclose(mu - a, b - mu, rel_tol=1e-9, abs_tol=1e-12)
        if not sym:
            note = (f"El intervalo no es simétrico respecto a μ; se usa la distancia menor "
                    f"({_g(d)}), así que la cota vale para (μ − kσ, μ + kσ) ⊆ ({_g(a)}, {_g(b)}) "
                    "y es conservadora para el intervalo pedido.")
        steps.append(("Valor de k a partir de los límites",
                      f"k = mín(μ − a, b − μ) / σ = {_g(d)} / {_g(sigma)}", _g(k_lim)))
        if k_in is not None and not math.isclose(k_in, k_lim, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"El k ingresado ({k_in:g}) no coincide con el que dan los límites "
                             f"({k_lim:.6g}). Deja vacío uno de los dos campos o corrígelos.")
        k = k_lim
    else:
        k = k_in
        steps.append(("Valor de k", "k (desviaciones estándar respecto a la media)", _g(k)))
    if k <= 1:
        raise ValueError(f"k = {k:.6g} no es válido: la cota 1 − 1/k² solo es informativa para "
                         "k > 1 (con k ≤ 1 vale 0 o menos). Usa un k mayor que 1 o un intervalo "
                         "más amplio.")

    # 6-9. Sustitución, desarrollo, porcentaje e intervalo
    k2 = k * k
    inv = 1.0 / k2
    bound = 1.0 - inv
    pct = 100.0 * bound
    low, high = mu - k * sigma, mu + k * sigma
    steps.append(("Fórmula", "P(|X − μ| < kσ) ≥ 1 − 1/k²", "k > 1"))
    steps.append(("Sustitución", f"P(|X − {_g(mu)}| < {_g(k)}·{_g(sigma)}) ≥ 1 − 1/{_g(k)}²",
                  f"kσ = {_g(k * sigma)}"))
    steps.append(("Desarrollo", f"1 − 1/{_g(k2)} = 1 − {_g(inv)}", _g(bound)))
    steps.append(("Porcentaje mínimo garantizado", f"(1 − 1/k²) × 100 = {_g(bound)} × 100",
                  f"{pct:.4g} %"))
    steps.append(("Límites del intervalo", f"μ ± kσ = {_g(mu)} ± {_g(k * sigma)}",
                  f"[{_g(low)}, {_g(high)}]"))
    steps.append(("Cota complementaria", f"P(|X − μ| ≥ kσ) ≤ 1/k² = {_g(inv)}",
                  f"≤ {100 * inv:.4g} %"))
    observed = None
    if data is not None:
        inside = int(np.sum(np.abs(data - mu) < k * sigma))
        observed = inside / data.size
        steps.append(("Comprobación con la muestra",
                      f"{inside} de {data.size} datos cumplen |xᵢ − μ| < kσ",
                      f"{100 * observed:.4g} % (≥ {pct:.4g} %)"))
    interp = (f"Al menos el {pct:.4g} % de los valores de X está a menos de {_g(k)} desviaciones "
              f"estándar de la media, es decir, entre {_g(low)} y {_g(high)}. Es una cota "
              "mínima válida para cualquier distribución: el porcentaje real puede ser mayor "
              "y no supone que los datos sean normales.")
    if note:
        interp += " " + note
    steps.append(("Interpretación", "", interp))

    hist = [{"Paso": i + 1, "Descripción": d, "Cálculo": c, "Resultado": r}
            for i, (d, c, r) in enumerate(steps)]

    def plot(fig):
        ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
        span = k * sigma
        pad = max(span * 0.6, sigma)
        lo_ax, hi_ax = low - pad, high + pad
        if data is not None:
            lo_ax, hi_ax = min(lo_ax, float(data.min()) - 0.1 * pad), max(hi_ax, float(data.max()) + 0.1 * pad)
        ax1.axvspan(low, high, color="#3cff8a", alpha=0.18,
                    label=f"(μ − kσ, μ + kσ): al menos {pct:.4g} %")
        ax1.axvline(mu, color="#ffd700", lw=2, label=f"μ = {_g(mu)}")
        for v, t in ((low, "μ − kσ"), (high, "μ + kσ")):
            ax1.axvline(v, color="#3cff8a", lw=1.5, ls="--")
            ax1.annotate(f"{t}\n{_g(v)}", xy=(v, 0.92), xycoords=("data", "axes fraction"),
                         ha="center", va="top", color="#c8d8f0", fontsize=8)
        ax1.annotate("", xy=(high, 0.45), xytext=(mu, 0.45), xycoords=("data", "axes fraction"),
                     arrowprops=dict(arrowstyle="<->", color="#ff6b35"))
        ax1.text((mu + high) / 2, 0.48, f"kσ = {_g(span)}", transform=ax1.get_xaxis_transform(),
                 ha="center", color="#ff6b35", fontsize=8)
        if data is not None:
            inside = np.abs(data - mu) < span
            ax1.plot(data[inside], np.full(int(inside.sum()), 0.2), "o", color="#4a9eff",
                     transform=ax1.get_xaxis_transform(), label="Datos dentro")
            if (~inside).any():
                ax1.plot(data[~inside], np.full(int((~inside).sum()), 0.2), "x", color="#ff5050",
                         transform=ax1.get_xaxis_transform(), label="Datos fuera")
        ax1.set_xlim(lo_ax, hi_ax)
        ax1.set_yticks([])
        ax1.set_xlabel("X")
        ax1.set_title("Intervalo μ ± kσ", fontsize=10)
        ax1.legend(fontsize=7, loc="lower left")
        ks = np.linspace(1.0, max(4.0, k + 1.0), 300)
        ax2.plot(ks, 100 * (1 - 1 / ks ** 2), color="#4a9eff", label="Cota 1 − 1/k²")
        ax2.plot([k], [pct], "D", color="#ffd700", label=f"k = {_g(k)} → {pct:.4g} %")
        ax2.set_xlabel("k")
        ax2.set_ylabel("% mínimo")
        ax2.set_ylim(0, 100)
        ax2.set_title("Cota mínima 1 − 1/k² (no exacta)", fontsize=10)
        ax2.legend(fontsize=7, loc="lower right")

    return {
        "mu": mu, "sigma": sigma, "var": var, "k": k, "bound": bound, "percent": pct,
        "low": low, "high": high, "observed": observed,
        "hist": hist,
        "resumen": (f"P(|X − {_g(mu)}| < {_g(k * sigma)}) ≥ {bound:.4g} → al menos {pct:.4g} % "
                    f"en [{_g(low)}, {_g(high)}]"),
        "plot": plot,
    }
