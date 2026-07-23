# app/infrastructure/reporting/xlsx_export.py
# Exportación a Excel multi-pestaña — extraído de
# InterfazOptimizacion.export_xlsx (app/ui/main_window.py), Fase 4c.
from __future__ import annotations

import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as XLImg

from app.application import report_content

# Duplicado deliberado de app.ui.main_window.METHODS_MD: ese registro es UI
# (alimenta el combo de métodos) y esta capa no debe depender de la UI.
_METHODS_MD = frozenset({
    "MD: Penalización (Newton)", "MD: Barreras (Newton)",
    "MD: Pen. (BFGS)",           "MD: Pes. (BFGS)",
    "MD: Pes. (Nelder-Mead)",    "MD: Sum. (BFGS)",
})


def write_xlsx(path: str, session: list, *, logo_path: str = None) -> None:
    """
    Exporta a Excel con estructura multi-pestaña:
      1) "RESUMEN" — función insertada, tipo de estudio, métodos aplicados
      2+) Una pestaña por método — resumen de resultados, tabla completa de
          iteraciones + gráficas 2D y 3D

    `logo_path`: si se da y existe, se ancla en la esquina de la pestaña
    RESUMEN. No rompe el export si falta.
    """
    fx_global = session[0]["fx"] if session else ""

    # ── Paleta de colores ─────────────────────────────────────────────
    C = {
        "bg_dark":  "080E22",
        "bg_mid":   "101E3A",
        "bg_head":  "162D5A",
        "bg_head2": "0D1E44",
        "bg_resumen":"0A1830",
        "bg_alt":   "E8F2FF",
        "fg_white": "FFFFFF",
        "fg_blue":  "7AAAEA",
        "fg_cyan":  "5DC8D0",
        "fg_gold":  "FFD700",
        "border":   "2A50A0",
        "accent":   "FF6B35",
    }
    def _font(bold=False, size=11, color=None, italic=False):
        return Font(bold=bold, size=size, italic=italic,
                    name="Segoe UI",
                    color=color or C["fg_white"])
    def _fill(hex_color):
        return PatternFill("solid", fgColor=hex_color)
    def _border(color=None):
        s = Side(style="thin", color=color or C["border"])
        return Border(left=s, right=s, top=s, bottom=s)
    def _center(wrap=False):
        return Alignment(horizontal="center", vertical="center", wrap_text=wrap)
    def _left(wrap=False):
        return Alignment(horizontal="left", vertical="center", wrap_text=wrap)
    def _set_col_width(ws, col_idx, width):
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    def _merge_write(ws, cell_range, value, font=None, fill=None,
                     align=None, height=None, border=None):
        ws.merge_cells(cell_range)
        c = ws[cell_range.split(":")[0]]
        c.value = value
        if font:   c.font      = font
        if fill:   c.fill      = fill
        if align:  c.alignment = align
        if border: c.border    = border
        if height:
            rn = int(''.join(filter(str.isdigit, cell_range.split(":")[0])))
            ws.row_dimensions[rn].height = height

    def _classify_study(metodo: str) -> str:
        if metodo in _METHODS_MD: return "Optimización Multidimensional (MD)"
        if metodo == "Fibonacci": return "Búsqueda Unidimensional — Sección Áurea / Fibonacci"
        if metodo in ("Armijo","Wolfe"): return "Búsqueda de Línea — Condiciones de Wolfe/Armijo"
        return "Búsqueda Unidimensional Local"

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # ══════════════════════════════════════════════════════════════════
    # PESTAÑA 1: RESUMEN
    # ══════════════════════════════════════════════════════════════════
    ws_r = wb.create_sheet("RESUMEN")
    ws_r.sheet_view.showGridLines = False
    for ci, w in [(1,6),(2,26),(3,52),(4,3)]:
        _set_col_width(ws_r, ci, w)

    # Título principal
    _merge_write(ws_r, "B1:C1",
        "Optimizador de Funciones — Reporte de Análisis",
        font=_font(bold=True, size=16),
        fill=_fill(C["bg_dark"]), align=_center(), height=40)
    ws_r.row_dimensions[2].height = 6

    r = 3
    # ── Bloque: Función analizada ─────────────────────────────────────
    _merge_write(ws_r, f"B{r}:C{r}", "FUNCIÓN ANALIZADA",
        font=_font(bold=True, size=12, color=C["fg_cyan"]),
        fill=_fill(C["bg_head2"]), align=_center(), height=24)
    r += 1

    fn_rows = [
        ("Función:",         fx_global),
        ("Variables:",       ", ".join(session[0]["vars"])),
        ("Intervalo / Dom.", f"[{session[0]['lo']:.4g},  {session[0]['hi']:.4g}]"),
        ("Tipo de función:", report_content.classify_function(fx_global)),
    ]
    for lbl, val in fn_rows:
        ws_r.row_dimensions[r].height = 20
        cl = ws_r.cell(row=r, column=2, value=lbl)
        cl.font = _font(bold=True, size=11, color=C["fg_blue"])
        cl.fill = _fill(C["bg_mid"]); cl.alignment = _left()
        cl.border = _border()
        cv = ws_r.cell(row=r, column=3, value=val)
        cv.font = _font(size=11); cv.fill = _fill(C["bg_mid"])
        cv.alignment = _left(); cv.border = _border()
        r += 1

    ws_r.row_dimensions[r].height = 8; r += 1

    # ── Bloque: Tipo de estudio ───────────────────────────────────────
    _merge_write(ws_r, f"B{r}:C{r}", "TIPO DE ESTUDIO APLICADO",
        font=_font(bold=True, size=12, color=C["fg_cyan"]),
        fill=_fill(C["bg_head2"]), align=_center(), height=24)
    r += 1

    all_types = list(dict.fromkeys(_classify_study(s["metodo"]) for s in session))
    for t in all_types:
        ws_r.row_dimensions[r].height = 20
        cl = ws_r.cell(row=r, column=2, value="Tipo:")
        cl.font = _font(bold=True, size=11, color=C["fg_blue"])
        cl.fill = _fill(C["bg_mid"]); cl.alignment = _left(); cl.border = _border()
        cv = ws_r.cell(row=r, column=3, value=t)
        cv.font = _font(size=11); cv.fill = _fill(C["bg_mid"])
        cv.alignment = _left(); cv.border = _border()
        r += 1

    ws_r.row_dimensions[r].height = 8; r += 1

    # ── Bloque: Métodos aplicados ─────────────────────────────────────
    _merge_write(ws_r, f"B{r}:C{r}", "MÉTODOS APLICADOS",
        font=_font(bold=True, size=12, color=C["fg_cyan"]),
        fill=_fill(C["bg_head2"]), align=_center(), height=24)
    r += 1

    for idx_s, s in enumerate(session, 1):
        ws_r.row_dimensions[r].height = 20
        cl = ws_r.cell(row=r, column=2, value=f"  {idx_s}.")
        cl.font = _font(bold=True, size=11, color=C["fg_gold"])
        cl.fill = _fill(C["bg_mid"]); cl.alignment = _center(); cl.border = _border()
        cv = ws_r.cell(row=r, column=3, value=s["metodo"])
        cv.font = _font(size=11); cv.fill = _fill(C["bg_mid"])
        cv.alignment = _left(); cv.border = _border()
        r += 1

    ws_r.row_dimensions[r].height = 8; r += 1

    # ── Bloque: Problemas de Inventario (solo si aplica) ──────────────
    inv_entries = [s for s in session if s.get("inventario_modelo")]
    if inv_entries:
        _merge_write(ws_r, f"B{r}:C{r}", "PROBLEMAS DE INVENTARIO",
            font=_font(bold=True, size=12, color=C["fg_cyan"]),
            fill=_fill(C["bg_head2"]), align=_center(), height=24)
        r += 1
        for s in inv_entries:
            im = s["inventario_modelo"]
            ws_r.row_dimensions[r].height = 20
            cl = ws_r.cell(row=r, column=2, value=s.get("metodo", ""))
            cl.font = _font(bold=True, size=11, color=C["fg_blue"])
            cl.fill = _fill(C["bg_mid"]); cl.alignment = _left(); cl.border = _border()
            extra_txt = "  ·  ".join(f"{k}: {v}" for k, v in im.get("extra_results", {}).items())
            cv = ws_r.cell(row=r, column=3, value=f"{im.get('titulo','')} — {extra_txt}")
            cv.font = _font(size=10); cv.fill = _fill(C["bg_mid"])
            cv.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
            cv.border = _border()
            r += 1
        ws_r.row_dimensions[r].height = 8; r += 1

    # ── Pie ───────────────────────────────────────────────────────────
    _merge_write(ws_r, f"B{r}:C{r}",
        f"Generado automáticamente  ·  {len(session)} método(s) ejecutado(s)",
        font=_font(italic=True, size=9, color=C["fg_blue"]),
        fill=_fill(C["bg_dark"]), align=_center(), height=16)

    # ══════════════════════════════════════════════════════════════════
    # PESTAÑAS POR MÉTODO
    # ══════════════════════════════════════════════════════════════════
    _tab_counts: dict = {}    # contador de nombres de pestaña usados

    def _safe_tab_name(metodo: str) -> str:
        """Genera nombre de pestaña único: 'Armijo', 'Armijo (2)', 'Armijo (3)'..."""
        base = metodo[:28]
        for ch in (chr(92), "/", "[", "]", "*", "?", ":"):
            base = base.replace(ch, "-")
        base = base.strip()
        if base not in _tab_counts:
            _tab_counts[base] = 1
            return base
        else:
            _tab_counts[base] += 1
            return f"{base} ({_tab_counts[base]})"

    try:
        for idx_s, entry in enumerate(session, 1):
            metodo = entry["metodo"]
            hist   = entry["hist"]

            tab_name = _safe_tab_name(metodo)
            ws_m = wb.create_sheet(tab_name)
            ws_m.sheet_view.showGridLines = False

            # Columnas fijas: A=margen, B=etiqueta, C=valor (ancho generoso)
            _set_col_width(ws_m, 1, 3)    # A — margen
            _set_col_width(ws_m, 2, 22)   # B — etiquetas
            _set_col_width(ws_m, 3, 52)   # C — valores

            # Banner
            ws_m.row_dimensions[1].height = 34
            _merge_write(ws_m, "B1:C1",
                f"Optimizador de Funciones  —  {metodo}",
                font=_font(bold=True, size=14),
                fill=_fill(C["bg_dark"]), align=_center(), height=34)

            ws_m.row_dimensions[2].height = 4
            r_m = 3

            # Info básica
            info_rows = [
                ("Método:",          metodo),
                ("Función:",         entry["fx"]),
                ("Tipo de estudio:", _classify_study(metodo)),
                ("Variables:",       ", ".join(entry["vars"])),
                ("Intervalo:",       f"[{entry['lo']:.4g},  {entry['hi']:.4g}]"),
            ]
            for lbl, val in info_rows:
                ws_m.row_dimensions[r_m].height = 20
                cl = ws_m.cell(row=r_m, column=2, value=lbl)
                cl.font = _font(bold=True, size=11, color=C["fg_blue"])
                cl.fill = _fill(C["bg_mid"]); cl.alignment = _left(); cl.border = _border()
                cv = ws_m.cell(row=r_m, column=3, value=val)
                cv.font = _font(size=11); cv.fill = _fill(C["bg_mid"])
                cv.alignment = Alignment(horizontal="left", vertical="center",
                                         wrap_text=True)
                cv.border = _border()
                r_m += 1

            ws_m.row_dimensions[r_m].height = 6; r_m += 1

            # ── Bloque Inventario (solo si este método resolvió un modelo
            # del módulo de Inventario) ───────────────────────────────────
            inv_model = entry.get("inventario_modelo")
            if inv_model:
                _merge_write(ws_m, f"B{r_m}:C{r_m}", "Problemas de Inventario",
                    font=_font(bold=True, size=11, color=C["fg_cyan"]),
                    fill=_fill(C["bg_head2"]), align=_center(), height=22)
                r_m += 1
                inv_rows = [("Modelo:", inv_model.get("titulo", "")),
                            ("Restricciones:", inv_model.get("restricciones", ""))]
                inv_rows += [(f"{k}:", str(v)) for k, v in inv_model.get("extra_results", {}).items()]
                for lbl, val in inv_rows:
                    ws_m.row_dimensions[r_m].height = 18
                    cl = ws_m.cell(row=r_m, column=2, value=lbl)
                    cl.font = _font(bold=True, size=10, color=C["fg_gold"])
                    cl.fill = _fill(C["bg_resumen"]); cl.alignment = _left(); cl.border = _border()
                    cv = ws_m.cell(row=r_m, column=3, value=val)
                    cv.font = _font(size=10); cv.fill = _fill(C["bg_resumen"])
                    cv.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
                    cv.border = _border()
                    r_m += 1
                ws_m.row_dimensions[r_m].height = 6; r_m += 1

            # ── Bloque RESUMEN (resultados óptimos) ───────────────────────
            _merge_write(ws_m, f"B{r_m}:C{r_m}", "Resumen de Resultados",
                font=_font(bold=True, size=11, color=C["fg_cyan"]),
                fill=_fill(C["bg_head2"]), align=_center(), height=22)
            r_m += 1

            if hist:
                last = hist[-1]
                # x_opt
                for kx in ("x_k","x","lambda_k"):
                    if kx in last:
                        v = last[kx]
                        if isinstance(v, np.ndarray):
                            xstr = "[" + ", ".join(f"{x:.6g}" for x in v.flat) + "]"
                        else:
                            xstr = f"{float(v):.8g}"
                        ws_m.row_dimensions[r_m].height = 18
                        cl = ws_m.cell(row=r_m, column=2, value="x_opt")
                        cl.font = _font(bold=True, size=11, color=C["fg_gold"])
                        cl.fill = _fill(C["bg_resumen"]); cl.alignment = _center(); cl.border = _border()
                        cv = ws_m.cell(row=r_m, column=3, value=xstr)
                        cv.font = _font(size=11); cv.fill = _fill(C["bg_resumen"])
                        cv.alignment = _left(); cv.border = _border()
                        r_m += 1
                        break
                # f_opt
                for kf in ("f_k","f_lambda","f_mu","f(x)"):
                    if kf in last and isinstance(last[kf],(int,float,np.number)):
                        ws_m.row_dimensions[r_m].height = 18
                        cl = ws_m.cell(row=r_m, column=2, value="f_opt")
                        cl.font = _font(bold=True, size=11, color=C["fg_gold"])
                        cl.fill = _fill(C["bg_resumen"]); cl.alignment = _center(); cl.border = _border()
                        cv = ws_m.cell(row=r_m, column=3, value=f"{float(last[kf]):.8g}")
                        cv.font = _font(size=11); cv.fill = _fill(C["bg_resumen"])
                        cv.alignment = _left(); cv.border = _border()
                        r_m += 1
                        break
                # iter count
                ws_m.row_dimensions[r_m].height = 18
                cl = ws_m.cell(row=r_m, column=2, value="iter")
                cl.font = _font(bold=True, size=11, color=C["fg_gold"])
                cl.fill = _fill(C["bg_resumen"]); cl.alignment = _center(); cl.border = _border()
                cv = ws_m.cell(row=r_m, column=3, value=len(hist))
                cv.font = _font(size=11); cv.fill = _fill(C["bg_resumen"])
                cv.alignment = _left(); cv.border = _border()
                r_m += 1

            ws_m.row_dimensions[r_m].height = 6; r_m += 1

            if hist:
                h_cols: list = []
                for rec in hist:
                    for k in rec:
                        if k not in h_cols: h_cols.append(k)

                h_rows = []
                for rec in hist:
                    row_vals = []
                    for col in h_cols:
                        v = rec.get(col, "")
                        if isinstance(v, np.ndarray):
                            v = "[" + ", ".join(f"{x:.5g}" for x in v.flat) + "]"
                        elif isinstance(v, float):
                            v = f"{v:.6g}"
                        row_vals.append(v if isinstance(v, (int, bool)) else str(v))
                    h_rows.append(row_vals)

                brd = _border()
                n_data_cols = len(h_cols)

                # Ajustar anchos SOLO de columnas D en adelante (B y C ya tienen
                # ancho fijo del bloque de info — 22 y 52 respectivamente)
                for j, col in enumerate(h_cols, 2):
                    if j <= 3:
                        continue   # B y C ya son 22/52 — no sobreescribir
                    cw = max(len(str(col)) + 2,
                             max((len(str(rv[j-2])) for rv in h_rows), default=0) + 2)
                    _set_col_width(ws_m, j, min(cw, 24))

                # ── Sección header "Tabla de iteraciones" ─────────────────
                last_col_letter = get_column_letter(1 + n_data_cols)
                _merge_write(ws_m, f"B{r_m}:{last_col_letter}{r_m}",
                    "Tabla de iteraciones",
                    font=_font(bold=True, size=11, color=C["fg_cyan"]),
                    fill=_fill(C["bg_head2"]), align=_center(), height=22)
                r_m += 1

                # ── Encabezados de columna ────────────────────────────────
                col_header_row = r_m   # fila que queremos congelar
                ws_m.row_dimensions[r_m].height = 20
                for j, col in enumerate(h_cols, 2):
                    c_cell = ws_m.cell(row=r_m, column=j, value=col)
                    c_cell.font      = _font(bold=True, size=10)
                    c_cell.fill      = _fill(C["bg_head"])
                    c_cell.alignment = _center()
                    c_cell.border    = brd
                r_m += 1

                # ── Filas de datos ────────────────────────────────────────
                for i_r, row_vals in enumerate(h_rows):
                    ws_m.row_dimensions[r_m].height = 15
                    use_alt = i_r % 2 == 1
                    row_fill = _fill(C["bg_alt"]) if use_alt else _fill(C["bg_mid"])
                    row_font_color = "111830" if use_alt else C["fg_white"]
                    for j, val in enumerate(row_vals, 2):
                        c_cell = ws_m.cell(row=r_m, column=j, value=val)
                        c_cell.fill      = row_fill
                        c_cell.font      = _font(size=9, color=row_font_color)
                        c_cell.alignment = _center()
                        c_cell.border    = brd
                    r_m += 1

                # Sin freeze_panes — todo el documento desplaza normalmente
            else:
                _merge_write(ws_m, f"B{r_m}:C{r_m}", "Sin datos de iteración.",
                    font=_font(size=10), fill=_fill(C["bg_mid"]),
                    align=_left(), height=18)
                r_m += 1

            ws_m.row_dimensions[r_m].height = 10; r_m += 1

            # ── Cálculos paso a paso (ANTES de las gráficas) ─────────────
            _merge_write(ws_m, f"B{r_m}:C{r_m}",
                "Cálculos Paso a Paso",
                font=_font(bold=True, size=11, color=C["fg_cyan"]),
                fill=_fill(C["bg_head2"]), align=_center(), height=22)
            r_m += 1

            steps = report_content.build_step_by_step(entry)
            for step_title, step_body in steps:
                ws_m.row_dimensions[r_m].height = 16
                ct = ws_m.cell(row=r_m, column=2, value=report_content.math_to_unicode(step_title))
                ct.font = _font(bold=True, size=10, color=C["fg_gold"])
                ct.fill = _fill(C["bg_head"]); ct.alignment = _left()
                ct.border = _border()
                r_m += 1
                lines = step_body.split("\n")
                for line in lines:
                    if not line.strip():
                        continue
                    ws_m.row_dimensions[r_m].height = 14
                    cb = ws_m.cell(row=r_m, column=2,
                                   value=report_content.math_to_unicode(line.strip()))
                    cb.font = _font(size=9)
                    cb.fill = _fill(C["bg_resumen"])
                    cb.alignment = Alignment(horizontal="left", vertical="center",
                                             wrap_text=True)
                    cb.border = _border()
                    ws_m.merge_cells(f"B{r_m}:C{r_m}")
                    r_m += 1
                ws_m.row_dimensions[r_m].height = 4; r_m += 1

            ws_m.row_dimensions[r_m].height = 10; r_m += 1

            # ── Gráficas 2D y 3D ─────────────────────────────────────────
            _merge_write(ws_m, f"B{r_m}:C{r_m}", "Gráficas",
                font=_font(bold=True, size=11, color=C["fg_cyan"]),
                fill=_fill(C["bg_head2"]), align=_center(), height=22)
            r_m += 1

            imgs = report_content.generate_plot_png(entry)

            IMG_ROW_HEIGHT = 15
            IMG_ROWS       = 23

            for img_buf, lbl in imgs:
                ws_m.row_dimensions[r_m].height = 16
                _merge_write(ws_m, f"B{r_m}:C{r_m}", lbl,
                    font=_font(bold=True, size=10, color=C["fg_blue"]),
                    fill=_fill(C["bg_mid"]), align=_center(), height=16)
                r_m += 1

                img_start_row = r_m
                for i in range(IMG_ROWS):
                    ws_m.row_dimensions[img_start_row + i].height = IMG_ROW_HEIGHT

                try:
                    xl_img = XLImg(img_buf)
                    xl_img.width  = 520
                    xl_img.height = IMG_ROWS * IMG_ROW_HEIGHT
                    ws_m.add_image(xl_img, f"{get_column_letter(2)}{img_start_row}")
                except Exception as ie:
                    ws_m.cell(row=r_m, column=2,
                              value=f"[imagen no disponible: {ie}]")

                r_m += IMG_ROWS + 1

        wb.save(path)
    except Exception:
        raise
