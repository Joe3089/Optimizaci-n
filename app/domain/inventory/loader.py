# app/domain/inventory/loader.py
# Carga de un inventario desde CSV/Excel (o texto pegado) y análisis por
# artículo: EOQ, costo anual, punto de reorden y clasificación ABC.
from __future__ import annotations

import csv
import io
import math
import os
import unicodedata
from typing import Any, Dict, List

from app.domain.inventory.models import _z_from_service_level

# Encabezados aceptados, ya normalizados (minúsculas, sin acentos ni espacios).
ALIASES = {
    "producto": ("producto", "articulo", "item", "sku", "nombre", "codigo", "descripcion"),
    "demanda": ("demanda", "demandaanual", "d"),
    "costo_pedido": ("costopedido", "costodepedido", "costoordenar", "costodeordenar", "s"),
    "costo_mantener": ("costomantener", "costodemantener", "costomantenimiento", "h"),
    "precio": ("precio", "preciounitario", "costounitario"),
    "demanda_diaria": ("demandadiaria", "ddiaria"),
    "desv_diaria": ("desvdiaria", "desviaciondiaria", "sigma", "sigmad"),
    "lead_time": ("leadtime", "tiempoentrega", "tiempodeentrega", "l"),
    "nivel_servicio": ("nivelservicio", "niveldeservicio", "servicio"),
}
REQUIRED = ("producto", "demanda", "costo_pedido", "costo_mantener")
SAMPLE = (
    "producto;demanda;costo_pedido;costo_mantener;precio;demanda_diaria;desv_diaria;lead_time;nivel_servicio\n"
    "Tornillo;12000;40;0.5;0.8;33;6;5;0.95\n"
    "Motor;400;120;30;250;1.1;0.4;14;0.98\n"
    "Correa;2500;60;2;12;7;2;7;0.95\n"
    "Filtro;5000;50;1.5;6;14;3;4;0.90\n"
    "Sensor;150;80;18;95;0.4;0.2;21;0.99\n"
)


def _norm(text: Any) -> str:
    raw = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return "".join(ch for ch in raw.lower() if ch.isalnum())


def _number(value: Any, row: int, col: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    txt = str(value or "").strip()
    if "," in txt and "." not in txt:
        txt = txt.replace(",", ".")
    try:
        return float(txt)
    except ValueError as ex:
        raise ValueError(f"Fila {row}: el valor «{value}» de «{col}» no es numérico.") from ex


def read_rows(path: str = "", text: str = "") -> List[List[Any]]:
    if path:
        if not os.path.isfile(path):
            raise ValueError(f"No se encontró el archivo: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext in (".xlsx", ".xlsm"):
            from openpyxl import load_workbook
            wb = load_workbook(path, read_only=True, data_only=True)
            try:
                return [list(r) for r in wb.active.iter_rows(values_only=True)
                        if any(c not in (None, "") for c in r)]
            finally:
                wb.close()
        if ext not in (".csv", ".txt"):
            raise ValueError("Formato no soportado. Usa un archivo .csv, .txt o .xlsx.")
        for enc in ("utf-8-sig", "cp1252"):
            try:
                with open(path, encoding=enc, newline="") as fh:
                    text = fh.read()
                break
            except UnicodeDecodeError:
                continue
    if not text.strip():
        raise ValueError("Selecciona un archivo o pega los datos del inventario.")
    head = text.splitlines()[0]
    delim = max(";\t,", key=head.count)
    return [r for r in csv.reader(io.StringIO(text.strip()), delimiter=delim) if any(c.strip() for c in r)]


def parse_inventory(rows: List[List[Any]]) -> List[Dict[str, Any]]:
    if len(rows) < 2:
        raise ValueError("El inventario debe tener encabezados y al menos un artículo.")
    lookup = {alias: key for key, names in ALIASES.items() for alias in names}
    columns: Dict[str, int] = {}
    for i, name in enumerate(rows[0]):
        key = lookup.get(_norm(name))
        if key and key not in columns:
            columns[key] = i
    missing = [k for k in REQUIRED if k not in columns]
    if missing:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(missing) +
                         ".\nEncabezados esperados: producto, demanda, costo_pedido, costo_mantener.")
    items = []
    for n, row in enumerate(rows[1:], start=2):
        def cell(key):
            i = columns.get(key)
            return row[i] if i is not None and i < len(row) and row[i] not in (None, "") else None

        name = cell("producto")
        if name is None:
            raise ValueError(f"Fila {n}: falta el nombre del producto.")
        item: Dict[str, Any] = {"producto": str(name).strip()}
        for key in ALIASES:
            if key == "producto":
                continue
            value = cell(key)
            if value is not None:
                item[key] = _number(value, n, key)
        for key in REQUIRED[1:]:
            if key not in item:
                raise ValueError(f"Fila {n}: falta «{key}».")
            if item[key] <= 0:
                raise ValueError(f"Fila {n}: «{key}» debe ser mayor que cero.")
        items.append(item)
    return items


def analyze_inventory(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = []
    for it in items:
        D, S, H = it["demanda"], it["costo_pedido"], it["costo_mantener"]
        q = math.sqrt(2 * D * S / H)
        row: Dict[str, Any] = {
            "Producto": it["producto"], "Demanda": D, "Q* (EOQ)": q, "Pedidos/año": D / q,
            "Costo anual": math.sqrt(2 * D * S * H),
        }
        if "precio" in it:
            row["Valor anual"] = D * it["precio"]
        if all(k in it for k in ("demanda_diaria", "desv_diaria", "lead_time", "nivel_servicio")):
            ns = it["nivel_servicio"]
            if not 0 < ns < 1:
                raise ValueError(f"{it['producto']}: el nivel de servicio debe estar entre 0 y 1.")
            ss = _z_from_service_level(ns) * it["desv_diaria"] * math.sqrt(it["lead_time"])
            row["Stock seguridad"] = ss
            row["ROP"] = it["demanda_diaria"] * it["lead_time"] + ss
        out.append(row)

    # ABC por valor anual; sin precios, por costo anual de inventario.
    key = "Valor anual" if all("Valor anual" in r for r in out) else "Costo anual"
    out.sort(key=lambda r: r[key], reverse=True)
    grand = sum(r[key] for r in out)
    acc = 0.0
    for r in out:
        start = acc
        acc += r[key] / grand
        r["% acumulado"] = 100 * acc
        r["Clase ABC"] = "A" if start < 0.8 else ("B" if start < 0.95 else "C")

    total = sum(r["Costo anual"] for r in out)
    counts = {c: sum(1 for r in out if r["Clase ABC"] == c) for c in "ABC"}
    colors = {"A": "#ffd700", "B": "#4a9eff", "C": "#35507f"}

    def plot(fig):
        ax = fig.add_subplot(111)
        names = [r["Producto"] for r in out]
        ax.bar(names, [r[key] for r in out], color=[colors[r["Clase ABC"]] for r in out])
        ax.set_ylabel(key)
        ax2 = ax.twinx()
        ax2.plot(names, [r["% acumulado"] for r in out], "o-", color="#ff6b35")
        ax2.set_ylim(0, 105); ax2.set_ylabel("% acumulado")
        ax2.tick_params(colors="#c8d8f0"); ax2.yaxis.label.set_color("#c8d8f0")
        ax.tick_params(axis="x", rotation=30)
        ax.set_title("Clasificación ABC del inventario")

    return {"hist": out, "total": total, "abc": counts, "plot": plot,
            "resumen": (f"{len(out)} artículos, costo anual total = {total:,.2f}. "
                        f"ABC: {counts['A']} A, {counts['B']} B, {counts['C']} C")}


def load_and_analyze(path: str = "", text: str = "") -> Dict[str, Any]:
    return analyze_inventory(parse_inventory(read_rows(path, text)))
