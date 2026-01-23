# -*- coding: utf-8 -*-
"""
reporte_export.py
Genera un reporte en Excel (.xlsx) con:
- Un resumen (función, método, parámetros clave)
- Tabla de iteraciones
- Imagen de la gráfica (si se provee ruta o bytes)

Diseñado para ser compatible con distintas versiones del proyecto:
- Expone: ReportItem, exportar_reporte_excel
- Alias: export_reporte_excel, export_report_excel (compatibilidad)
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as XLImage


def _safe_sheet_name(name: str) -> str:
    # Excel: max 31 chars, no []:*?/\
    name = re.sub(r'[\[\]\:\*\?\/\\]', '-', name).strip()
    return (name[:31] if len(name) > 31 else name) or "Hoja"


def _autosize_columns(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                val = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(val))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max(10, max_len + 2), 45)


def _header_style():
    fill = PatternFill("solid", fgColor="1F2A38")  # azul oscuro
    font = Font(color="FFFFFF", bold=True)
    align = Alignment(horizontal="center", vertical="center")
    border = Border(
        left=Side(style="thin", color="2E3B4E"),
        right=Side(style="thin", color="2E3B4E"),
        top=Side(style="thin", color="2E3B4E"),
        bottom=Side(style="thin", color="2E3B4E"),
    )
    return fill, font, align, border


@dataclass
class ReportItem:
    funcion: str
    metodo: str
    # filas: lista de dicts (recomendado) o lista de listas con headers aparte
    iteraciones: Union[List[Dict[str, Any]], List[List[Any]]]
    headers: Optional[List[str]] = None  # si iteraciones es lista de listas
    resumen: Dict[str, Any] = field(default_factory=dict)
    grafica_path: Optional[str] = None   # ruta a PNG/JPG
    grafica_bytes: Optional[bytes] = None  # alternativo (PNG/JPG)


def exportar_reporte_excel(
    report_items: Sequence[ReportItem],
    output_path: Optional[str] = None,
    app_title: str = "Optimizador de Funciones"
) -> str:
    """
    Exporta un SOLO archivo .xlsx:
    - Agrupa por función.
    - Para cada función, crea/usa un workbook distinto si output_path incluye {func}.
      Si output_path es una ruta fija, mete TODAS las funciones en el mismo archivo
      con pestañas: <Metodo> - <hash/idx>.
    - Para una misma función con varios métodos, crea una pestaña por método.

    Retorna la ruta del archivo generado.
    """
    report_items = list(report_items or [])
    if not report_items:
        raise ValueError("No hay report_items para exportar.")

    # Si no dan ruta: mismo directorio, nombre base por primera función
    if output_path is None:
        safe = re.sub(r'[^a-zA-Z0-9]+', '_', report_items[0].funcion).strip('_')
        output_path = os.path.abspath(f"reporte_{safe or 'funcion'}.xlsx")
    else:
        output_path = os.path.abspath(output_path)

    # Cargar o crear (para permitir agregar pestañas en el mismo archivo)
    if os.path.exists(output_path):
        wb = load_workbook(output_path)
    else:
        wb = Workbook()
        # borrar hoja por defecto
        if wb.sheetnames:
            ws0 = wb[wb.sheetnames[0]]
            wb.remove(ws0)

    # estilos
    fill, font, align, border = _header_style()

    # Agrupar por función
    by_func: Dict[str, List[ReportItem]] = {}
    for it in report_items:
        by_func.setdefault(it.funcion, []).append(it)

    # Si hay múltiples funciones y quieren un solo archivo: se crearán sheets por método+func
    multi_func = len(by_func) > 1

    for func, items in by_func.items():
        # Orden estable por método
        for idx, item in enumerate(items, start=1):
            base_name = item.metodo if not multi_func else f"{item.metodo}_{idx}"
            sheet_name = _safe_sheet_name(base_name)
            # evitar colisión
            if sheet_name in wb.sheetnames:
                k = 2
                while _safe_sheet_name(f"{sheet_name}_{k}") in wb.sheetnames:
                    k += 1
                sheet_name = _safe_sheet_name(f"{sheet_name}_{k}")

            ws = wb.create_sheet(sheet_name)

            # Título
            ws["A1"] = app_title
            ws["A1"].font = Font(bold=True, size=16, color="FFFFFF")
            ws.merge_cells("A1:H1")
            ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
            ws["A1"].fill = PatternFill("solid", fgColor="0B1220")

            # Subtitulo
            ws["A2"] = f"Método: {item.metodo}"
            ws["A3"] = f"Función: {item.funcion}"
            ws.merge_cells("A2:H2"); ws.merge_cells("A3:H3")
            for cell in ("A2","A3"):
                ws[cell].font = Font(color="FFFFFF", bold=True)
                ws[cell].fill = PatternFill("solid", fgColor="0B1220")

            # Resumen (key/value)
            r0 = 5
            ws[f"A{r0}"] = "RESUMEN"
            ws[f"A{r0}"].font = Font(color="FFFFFF", bold=True)
            ws[f"A{r0}"].fill = PatternFill("solid", fgColor="1F2A38")
            ws.merge_cells(f"A{r0}:C{r0}")
            rr = r0 + 1
            for k, v in (item.resumen or {}).items():
                ws[f"A{rr}"] = str(k)
                ws[f"B{rr}"] = "" if v is None else v
                rr += 1

            # Tabla
            table_start = max(rr + 1, 12)
            ws[f"A{table_start}"] = "RESULTADOS / ITERACIONES"
            ws[f"A{table_start}"].font = Font(color="FFFFFF", bold=True)
            ws[f"A{table_start}"].fill = PatternFill("solid", fgColor="1F2A38")
            ws.merge_cells(f"A{table_start}:H{table_start}")

            # headers + rows
            if item.iteraciones and isinstance(item.iteraciones[0], dict):
                headers = list(item.iteraciones[0].keys())
                rows = [[row.get(h) for h in headers] for row in item.iteraciones]  # type: ignore
            else:
                headers = item.headers or []
                rows = item.iteraciones  # type: ignore

            hdr_row = table_start + 1
            for c, h in enumerate(headers, start=1):
                cell = ws.cell(row=hdr_row, column=c, value=h)
                cell.fill, cell.font, cell.alignment, cell.border = fill, font, align, border

            for r, row in enumerate(rows, start=hdr_row + 1):
                for c, val in enumerate(row, start=1):
                    ws.cell(row=r, column=c, value=val)

            _autosize_columns(ws)

            # Insertar gráfica (si existe)
            img_added = False
            img_anchor_row = hdr_row + len(rows) + 3
            if item.grafica_path and os.path.exists(item.grafica_path):
                try:
                    img = XLImage(item.grafica_path)
                    img.anchor = f"A{img_anchor_row}"
                    ws.add_image(img)
                    img_added = True
                except Exception:
                    img_added = False
            elif item.grafica_bytes:
                # escribir temporal
                try:
                    tmp = os.path.join(os.path.dirname(output_path), f"_tmp_graf_{sheet_name}.png")
                    with open(tmp, "wb") as f:
                        f.write(item.grafica_bytes)
                    img = XLImage(tmp)
                    img.anchor = f"A{img_anchor_row}"
                    ws.add_image(img)
                    img_added = True
                except Exception:
                    img_added = False

            if not img_added:
                ws[f"A{img_anchor_row}"] = "Gráfica no disponible"
                ws[f"A{img_anchor_row}"].font = Font(color="888888", italic=True)

    wb.save(output_path)
    return output_path


# --- Aliases de compatibilidad (distintas versiones del proyecto) ---
def export_reporte_excel(*args, **kwargs):
    return exportar_reporte_excel(*args, **kwargs)

def export_report_excel(*args, **kwargs):
    return exportar_reporte_excel(*args, **kwargs)
