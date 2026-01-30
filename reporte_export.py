# -*- coding: utf-8 -*-
"""
reporte_export.py (estable)

- exportar_reporte_excel(resultados, output_path=None, app_title="...")
  Acepta múltiples formatos de entrada (dict/list) y genera un .xlsx usando openpyxl
  SIN requerir xlsxwriter.

- exportar_reporte_pdf(...)
  PDF opcional: intenta importar reportlab dentro de la función y da un error claro si no está.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
import os

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, Reference


@dataclass
class ReportItem:
    funcion: str
    metodo: str
    iteraciones: List[Dict[str, Any]]
    resumen: Dict[str, Any] = field(default_factory=dict)
    grafica_path: Optional[str] = None
    grafica_bytes: Optional[bytes] = None


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_sheet_name(name: str) -> str:
    # Excel: max 31 chars, no []:*?/\
    bad = '[]:*?/\\'
    for ch in bad:
        name = name.replace(ch, '-')
    name = name.strip()
    if not name:
        name = "Hoja"
    return name[:31]


def _infer_last_x_fx(rows: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not rows:
        return None, None
    last = rows[-1]
    # x key candidates
    for k in ("x", "x_new", "x*", "x_opt", "x_optimo"):
        if k in last:
            try:
                x = float(last[k])
                break
            except Exception:
                x = None
            break
    else:
        x = None

    for k in ("f(x)", "f_new", "f*", "fx", "f_opt", "f_optimo"):
        if k in last:
            try:
                fx = float(last[k])
                break
            except Exception:
                fx = None
            break
    else:
        fx = None

    return x, fx


def _normalize_resultados(
    resultados: Any,
    funcion_default: str = "f(x)"
) -> List[ReportItem]:
    """
    Normaliza entradas a List[ReportItem].

    Soporta:
    1) dict{funcion: dict{metodo: history}}
    2) dict{metodo: history}  (funcion_default)
    3) list[ReportItem]
    4) list[dict]  -> un método "Resultado"
    """
    if resultados is None:
        return []

    # 3) list[ReportItem]
    if isinstance(resultados, list) and resultados and isinstance(resultados[0], ReportItem):
        return resultados

    items: List[ReportItem] = []

    # 4) list[dict] (history directa)
    if isinstance(resultados, list) and (not resultados or isinstance(resultados[0], dict)):
        items.append(ReportItem(funcion=funcion_default, metodo="Resultado", iteraciones=list(resultados)))
        return items

    # 1) / 2) dict
    if isinstance(resultados, dict):
        # dict{funcion: ...} o dict{metodo: ...}
        # Detectamos si el valor es dict de métodos (history)
        # Si alguna key parece método y el valor es list, asumimos forma 2.
        is_form2 = False
        for k, v in resultados.items():
            if isinstance(v, list):
                is_form2 = True
                break

        if is_form2:
            func_expr = funcion_default
            for metodo, history in resultados.items():
                if isinstance(history, list):
                    items.append(ReportItem(funcion=func_expr, metodo=str(metodo), iteraciones=history))
            return items

        # Forma 1: dict{funcion: dict{metodo: history}}
        for func_expr, per_method in resultados.items():
            if isinstance(per_method, dict):
                for metodo, history in per_method.items():
                    if isinstance(history, list):
                        items.append(ReportItem(funcion=str(func_expr), metodo=str(metodo), iteraciones=history))
        return items

    # Fallback
    raise TypeError("resultados debe ser dict, list[dict] o list[ReportItem]")


def exportar_reporte_excel(
    resultados: Any,
    output_path: Optional[str] = None,
    app_title: str = "Optimizador de Funciones"
) -> str:
    """
    Genera un archivo Excel:
    - Hoja Resumen (no redundante): por método incluye #iters, x_final, f_final
    - Una hoja por método con su tabla completa
    - Incluye un gráfico (openpyxl) f vs iter si hay datos numéricos

    `resultados` puede ser dict o list (ver _normalize_resultados).
    """
    items = _normalize_resultados(resultados)

    if not items:
        raise ValueError("No hay resultados para exportar (ejecute al menos un método).")

    if output_path is None or not str(output_path).strip():
        output_path = os.path.abspath(f"reporte_{_now_stamp()}.xlsx")
    else:
        output_path = os.path.abspath(output_path)

    wb = Workbook()
    ws_sum = wb.active
    ws_sum.title = "Resumen"

    header_fill = PatternFill("solid", fgColor="1F2937")  # gris oscuro
    header_font = Font(color="FFFFFF", bold=True)
    center = Alignment(horizontal="center", vertical="center")

    ws_sum["A1"] = app_title
    ws_sum["A1"].font = Font(bold=True, size=14)
    ws_sum["A2"] = f"Generado: {_now_stamp()}"
    ws_sum["A3"] = "Resumen por método (sin duplicar tablas):"
    ws_sum["A3"].font = Font(bold=True)

    sum_headers = ["Función", "Método", "Iteraciones", "x_final", "f_final"]
    for col, h in enumerate(sum_headers, 1):
        cell = ws_sum.cell(row=5, column=col, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center

    r = 6
    for it in items:
        x_final, f_final = _infer_last_x_fx(it.iteraciones)
        ws_sum.cell(row=r, column=1, value=it.funcion)
        ws_sum.cell(row=r, column=2, value=it.metodo)
        ws_sum.cell(row=r, column=3, value=len(it.iteraciones))
        ws_sum.cell(row=r, column=4, value=x_final)
        ws_sum.cell(row=r, column=5, value=f_final)
        r += 1

    # Auto ancho resumen
    for col in range(1, 6):
        ws_sum.column_dimensions[get_column_letter(col)].width = 22

    # Hojas por método
    for it in items:
        sheet_name = _safe_sheet_name(it.metodo)
        # Evitar duplicados
        base = sheet_name
        idx = 2
        while sheet_name in wb.sheetnames:
            sheet_name = _safe_sheet_name(f"{base}_{idx}")
            idx += 1

        ws = wb.create_sheet(sheet_name)
        ws["A1"] = f"Función: {it.funcion}"
        ws["A2"] = f"Método: {it.metodo}"
        ws["A1"].font = Font(bold=True)
        ws["A2"].font = Font(bold=True)

        rows = it.iteraciones or []
        if not rows:
            ws["A4"] = "Sin datos de iteración."
            continue

        keys = list(rows[0].keys())

        # Header tabla
        start_row = 4
        for c, k in enumerate(keys, 1):
            cell = ws.cell(row=start_row, column=c, value=str(k))
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center

        # Data
        for i, row in enumerate(rows, 1):
            rr = start_row + i
            for c, k in enumerate(keys, 1):
                ws.cell(row=rr, column=c, value=row.get(k))

        # Freeze y ancho
        ws.freeze_panes = ws["A5"]
        for c, k in enumerate(keys, 1):
            ws.column_dimensions[get_column_letter(c)].width = max(12, min(28, len(str(k)) + 4))

        # Chart: f vs iter (si existen columnas numéricas)
        # Buscar columna de iter y de f
        iter_col = None
        f_col = None
        for idx_k, k in enumerate(keys, 1):
            lk = str(k).lower()
            if lk in ("iter", "it", "k", "iteracion", "iteración"):
                iter_col = idx_k
            if lk in ("f(x)", "fx", "f_new", "f", "f_final"):
                f_col = idx_k

        if iter_col is None:
            # si no hay iter, usamos índice de fila
            ws.cell(row=start_row, column=len(keys) + 2, value="iter").fill = header_fill
            ws.cell(row=start_row, column=len(keys) + 2).font = header_font
            for i in range(1, len(rows) + 1):
                ws.cell(row=start_row + i, column=len(keys) + 2, value=i)
            iter_col = len(keys) + 2

        if f_col is not None:
            chart = LineChart()
            chart.title = "f(x) vs iteración"
            chart.y_axis.title = "f(x)"
            chart.x_axis.title = "iteración"

            data_ref = Reference(ws, min_col=f_col, min_row=start_row, max_row=start_row + len(rows))
            cats_ref = Reference(ws, min_col=iter_col, min_row=start_row + 1, max_row=start_row + len(rows))
            chart.add_data(data_ref, titles_from_data=True)
            chart.set_categories(cats_ref)

            ws.add_chart(chart, f"{get_column_letter(max(1, len(keys) + 2))}2")

    wb.save(output_path)
    return output_path


def exportar_reporte_pdf(
    resultados: Any,
    output_path: Optional[str] = None,
    app_title: str = "Optimizador de Funciones"
) -> str:
    """
    Exporta PDF con reportlab (si está disponible).
    Si no está reportlab instalado, lanza un error claro (sin romper imports globales).
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception as e:
        raise ModuleNotFoundError(
            "No se encontró 'reportlab'. Para exportar PDF instala reportlab "
            "(por ejemplo: pip install reportlab) o desactiva exportar PDF."
        ) from e

    items = _normalize_resultados(resultados)
    if not items:
        raise ValueError("No hay resultados para exportar (ejecute al menos un método).")

    if output_path is None or not str(output_path).strip():
        output_path = os.path.abspath(f"reporte_{_now_stamp()}.pdf")
    else:
        output_path = os.path.abspath(output_path)

    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, app_title)
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generado: {_now_stamp()}")
    y -= 30

    for it in items:
        x_final, f_final = _infer_last_x_fx(it.iteraciones)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Método: {it.metodo}")
        y -= 16
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Función: {it.funcion}")
        y -= 14
        c.drawString(50, y, f"Iteraciones: {len(it.iteraciones)}   x_final: {x_final}   f_final: {f_final}")
        y -= 24
        if y < 80:
            c.showPage()
            y = height - 50

    c.save()
    return output_path
