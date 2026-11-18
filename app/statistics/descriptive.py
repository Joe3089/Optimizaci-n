# app/statistics/descriptive.py
# Estadística descriptiva de una muestra numérica.
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np


def descriptive(data: List[float]) -> Dict[str, Any]:
    x = np.asarray(data, dtype=float)
    if x.size < 2:
        raise ValueError("La estadística descriptiva requiere al menos 2 datos.")
    n = x.size
    mean = float(x.mean())
    var = float(x.var(ddof=1))
    std = float(np.sqrt(var))
    q1, median, q3 = (float(v) for v in np.percentile(x, [25, 50, 75]))
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = [float(v) for v in x if v < low or v > high]

    counts = Counter(x.tolist())
    top = max(counts.values())
    modes = sorted(k for k, v in counts.items() if v == top) if top > 1 else []

    # Coeficientes corregidos por sesgo de muestra (mismos que Excel).
    skew = kurt = float("nan")
    if std > 0 and n > 2:
        z = (x - mean) / std
        skew = float(n / ((n - 1) * (n - 2)) * np.sum(z ** 3))
    if std > 0 and n > 3:
        z = (x - mean) / std
        kurt = float(n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(z ** 4)
                     - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))

    stats = [
        ("n (tamaño de muestra)", n),
        ("Media", mean),
        ("Mediana", median),
        ("Moda", ", ".join(f"{m:g}" for m in modes) if modes else "Sin moda"),
        ("Varianza muestral", var),
        ("Desviación estándar", std),
        ("Coeficiente de variación (%)", 100 * std / abs(mean) if mean else float("nan")),
        ("Mínimo", float(x.min())),
        ("Máximo", float(x.max())),
        ("Rango", float(x.max() - x.min())),
        ("Cuartil 1 (Q1)", q1),
        ("Cuartil 3 (Q3)", q3),
        ("Rango intercuartílico", iqr),
        ("Asimetría", skew),
        ("Curtosis (exceso)", kurt),
        ("Valores atípicos (regla 1.5·RIC)", ", ".join(f"{v:g}" for v in outliers) or "Ninguno"),
    ]

    def plot(fig):
        ax1, ax2 = fig.subplots(1, 2)
        ax1.hist(x, bins="auto", color="#4a9eff", edgecolor="#0a1228")
        ax1.axvline(mean, color="#ffd700", label=f"Media = {mean:.4g}")
        ax1.axvline(median, color="#ff6b35", linestyle="--", label=f"Mediana = {median:.4g}")
        ax1.set_title("Histograma"); ax1.legend()
        ax2.boxplot(x, vert=True)
        ax2.set_title("Diagrama de caja")

    return {
        "mean": mean, "median": median, "modes": modes, "var": var, "std": std,
        "q1": q1, "q3": q3, "iqr": iqr, "skew": skew, "kurt": kurt, "outliers": outliers,
        "hist": [{"Estadístico": k, "Valor": v} for k, v in stats],
        "resumen": f"Media = {mean:.4f}, mediana = {median:.4f}, desv. estándar = {std:.4f}",
        "plot": plot,
    }
