# app/infrastructure/reporting/docx_export.py
# Exportación a Word (.docx) — mismo contenido/estructura profesional que
# pdf_export.py y xlsx_export.py, reutilizando app.application.report_content
# como única fuente de descripciones/gráficas/pasos (nada de lógica de
# negocio duplicada aquí, solo el layout específico de Word).
from __future__ import annotations

import numpy as np
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from app.application import report_content
from app.domain.inventory import INVENTORY_FUNCTIONS, INVENTORY_CATEGORY, INVENTORY_REPORT_FIELDS

# ── Paleta (mismos tonos que pdf_export.py, adaptados a Word) ───────────────
_C_DARK   = RGBColor(0x0A, 0x14, 0x28)
_C_HEAD   = RGBColor(0x1A, 0x3A, 0x6A)
_C_ACCENT = RGBColor(0x1A, 0x50, 0xC0)
_C_GRAY   = RGBColor(0x55, 0x55, 0x55)
_SHADE_HEAD  = "1A3A6A"   # mismo azul que los demás encabezados del informe
_SHADE_LIGHT = "F4F7FB"


def _shade_cell(cell, hex_color: str) -> None:
    """Aplica color de fondo a una celda (python-docx no lo expone directo)."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def _no_split_row(row) -> None:
    """Evita que una fila se divida entre dos páginas (celda cortada a la
    mitad) — equivalente Word del KeepTogether de reportlab en pdf_export.py."""
    trPr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    trPr.append(cant_split)


def _keep_block_together(paragraphs) -> None:
    """Marca una secuencia de párrafos para que Word los mantenga juntos en
    la misma página (keep_with_next encadenado + keep_together)."""
    for i, p in enumerate(paragraphs):
        p.paragraph_format.keep_together = True
        if i < len(paragraphs) - 1:
            p.paragraph_format.keep_with_next = True


def _heading(doc, text, level=1, color=_C_HEAD, size=14):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(10)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(size)
    run.font.color.rgb = color
    return p


def _body(doc, text, size=10, italic=False, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.italic = italic
    if color:
        run.font.color.rgb = color
    return p


def _label_value_table(doc, rows, col1_w=4.0, col2_w=12.0):
    tbl = doc.add_table(rows=0, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    for lbl, val in rows:
        r = tbl.add_row()
        _no_split_row(r)
        r.cells[0].width = Cm(col1_w); r.cells[1].width = Cm(col2_w)
        rp = r.cells[0].paragraphs[0]; rr = rp.add_run(str(lbl))
        rr.bold = True; rr.font.size = Pt(9); rr.font.color.rgb = _C_ACCENT
        _shade_cell(r.cells[0], _SHADE_LIGHT)
        vp = r.cells[1].paragraphs[0]; vr = vp.add_run(str(val))
        vr.font.size = Pt(9)
    return tbl


def _banner(doc, title, subtitle=""):
    """Banner de portada: título en celda con fondo azul oscuro (Word no
    tiene 'banner' nativo — una tabla de 1x1 con sombreado cumple la misma
    función visual que en pdf_export.py/xlsx_export.py)."""
    tbl = doc.add_table(rows=2 if subtitle else 1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _no_split_row(tbl.rows[0])
    cell = tbl.rows[0].cells[0]
    _shade_cell(cell, _SHADE_HEAD)
    p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title); run.bold = True; run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    if subtitle:
        _no_split_row(tbl.rows[1])
        cell2 = tbl.rows[1].cells[0]
        _shade_cell(cell2, _SHADE_HEAD)
        p2 = cell2.paragraphs[0]; p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run2 = p2.add_run(subtitle); run2.font.size = Pt(10)
        run2.font.color.rgb = RGBColor(0xDD, 0xEC, 0xFF)
    return tbl


def _iteration_table(doc, hist: list):
    if not hist:
        _body(doc, "Sin datos de iteración.")
        return
    h_cols: list = []
    for rec in hist:
        for k in rec:
            if k not in h_cols:
                h_cols.append(k)

    tbl = doc.add_table(rows=1, cols=len(h_cols))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.style = "Light Grid Accent 1"
    hdr = tbl.rows[0]
    _no_split_row(hdr)
    for j, col in enumerate(h_cols):
        c = hdr.cells[j]
        _shade_cell(c, _SHADE_HEAD)
        rp = c.paragraphs[0]; rp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = rp.add_run(str(col)); rr.bold = True; rr.font.size = Pt(8)
        rr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for rec in hist:
        row = tbl.add_row()
        _no_split_row(row)
        for j, col in enumerate(h_cols):
            v = rec.get(col, "")
            if isinstance(v, np.ndarray):
                v = "[" + ", ".join(f"{x:.4g}" for x in v.flat) + "]"
            elif isinstance(v, float):
                v = f"{v:.5g}"
            cp = row.cells[j].paragraphs[0]; cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cr = cp.add_run(str(v)); cr.font.size = Pt(7.5)
    return tbl


def write_docx(path: str, session: list, *,
               area_key: str | list[str] = "general", logo_path: str = None) -> None:
    """
    Exporta a Word (.docx) profesional con la MISMA estructura que el PDF:
      - Portada con logo + banner + resumen de función/estudios/métodos
      - Sección "Problemas de Inventario" (si algún método resolvió uno)
      - Una sección por método: info, tabla de iteraciones (sin cortes
        entre páginas — w:cantSplit en cada fila), cálculos paso a paso,
        problema de inventario asociado (si aplica) y gráficas 2D/3D
      - Sección de utilidad por área de aplicación (misma lógica que
        report_content.AREA_ORDER / describe_area_utility usada en el PDF)

    `area_key`: str o list[str], igual que en pdf_export.write_pdf.
    """
    area_keys_arg = [area_key] if isinstance(area_key, str) else (list(area_key) or ["general"])
    fx_global = session[0]["fx"] if session else ""

    doc = Document()
    section = doc.sections[0]
    section.left_margin = section.right_margin = Cm(2.0)
    section.top_margin = section.bottom_margin = Cm(2.0)

    # ── Portada ──────────────────────────────────────────────────────────
    logo_buf = report_content.load_logo(logo_path)
    if logo_buf is not None:
        try:
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(logo_buf, width=Cm(2.2))
        except Exception:
            pass

    _banner(doc, "Optimizador de Funciones",
           f"Reporte de Análisis · {len(session)} método(s) ejecutado(s)")
    doc.add_paragraph()

    _heading(doc, "1. Función Analizada")
    _label_value_table(doc, [
        ("Expresión:", fx_global),
        ("Tipo:", report_content.classify_function(fx_global)),
        ("Variables:", ", ".join(session[0]["vars"]) if session else ""),
        ("Dominio:", f"[{session[0]['lo']:.4g}, {session[0]['hi']:.4g}]" if session else ""),
    ])
    _body(doc, report_content.describe_function(fx_global))

    _heading(doc, "2. Tipos de Estudio Aplicables a la Función")
    _body(doc, report_content.describe_applicable_studies(fx_global))

    _heading(doc, "3. Métodos Aplicados")
    for i, s in enumerate(session, 1):
        justif = report_content.justify_method(s["metodo"], fx_global)
        _heading(doc, f"{i}. {s['metodo']}", size=11, color=_C_ACCENT)
        _body(doc, justif)

    # ── 3b. Problemas de Inventario (solo si aplica) ────────────────────
    inv_entries = [s for s in session if s.get("inventario_modelo")]
    if inv_entries:
        _heading(doc, "Problemas de Inventario")
        for s in inv_entries:
            im = s["inventario_modelo"]
            hdr = _heading(doc, f"▸ {im.get('titulo','')} (método: {s['metodo']})",
                          size=11, color=_C_ACCENT)
            extra_txt = "  ·  ".join(f"{k}: {v}" for k, v in im.get("extra_results", {}).items())
            body1 = _body(doc, im.get("restricciones", ""))
            body2 = _body(doc, f"Resultados: {extra_txt}")
            _keep_block_together([hdr, body1, body2])

    doc.add_page_break()

    # ── Una sección por método ──────────────────────────────────────────
    for idx_s, entry in enumerate(session, 1):
        metodo = entry["metodo"]
        hist = entry["hist"]
        if idx_s > 1:
            doc.add_page_break()

        tipo_est = report_content.classify_study(metodo)
        _banner(doc, f"Método {idx_s}: {metodo}", f"f(x) = {entry['fx']}")
        _label_value_table(doc, [
            ("Método:", metodo),
            ("Función:", entry["fx"]),
            ("Tipo de estudio:", tipo_est),
            ("Variables:", ", ".join(entry["vars"])),
            ("Intervalo:", f"[{entry['lo']:.4g}, {entry['hi']:.4g}]"),
        ])

        inv_m = entry.get("inventario_modelo")
        if inv_m:
            _heading(doc, "Problema de Inventario", size=11)
            inv_extra = "  ·  ".join(f"{k}: {v}" for k, v in inv_m.get("extra_results", {}).items())
            _label_value_table(doc, [
                ("Modelo:", inv_m.get("titulo", "")),
                ("Restricciones:", inv_m.get("restricciones", "")),
                ("Resultados:", inv_extra),
            ])

        _heading(doc, "Tabla de Iteraciones", size=11)
        _iteration_table(doc, hist)

        _heading(doc, "Cálculos Paso a Paso", size=11)
        for step_title, step_body in report_content.build_step_by_step(entry):
            t = _heading(doc, report_content.math_to_unicode(step_title), size=10, color=_C_DARK)
            bodies = [t]
            for line in step_body.split("\n"):
                if line.strip():
                    bodies.append(_body(doc, report_content.math_to_unicode(line.strip()), size=9))
            _keep_block_together(bodies)

        graf_heading = _heading(doc, "Gráficas", size=11)
        try:
            first_img = True
            for img_buf, lbl in report_content.generate_plot_png(entry):
                lbl_p = _body(doc, lbl, size=9, color=_C_GRAY)
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(img_buf, width=Cm(15))
                # Etiqueta + imagen siempre juntas; la primera además pegada
                # al título "Gráficas" para que no quede huérfano.
                block = [graf_heading, lbl_p, p] if first_img else [lbl_p, p]
                _keep_block_together(block)
                first_img = False
        except Exception:
            pass

    # ── Utilidad por área de aplicación (misma lógica que el PDF) ───────
    doc.add_page_break()
    if "general" in area_keys_arg:
        _heading(doc, "Utilidad en Diversas Áreas a Futuro")
        area_keys_to_show = report_content.AREA_ORDER
    elif len(area_keys_arg) == 1:
        _heading(doc, "Utilidad en el Área de Aplicación Seleccionada")
        area_keys_to_show = area_keys_arg
    else:
        _heading(doc, "Utilidad en las Áreas de Aplicación Seleccionadas")
        area_keys_to_show = [k for k in report_content.AREA_ORDER if k in area_keys_arg]

    for key in area_keys_to_show:
        info = report_content.describe_area_utility(key)
        h = _heading(doc, f"▸ {info['titulo']}", size=12, color=_C_ACCENT)
        b = _body(doc, info["aplicacion"])
        _keep_block_together([h, b])
        _label_value_table(doc, [
            ("Métodos recomendados:", ", ".join(info["metodos_recomendados"])),
            ("Ventajas:", info["ventajas"]),
            ("Beneficios:", info["beneficios"]),
            ("Innovación:", info["innovacion"]),
        ])

    # ── Funciones Matemáticas para Problemas de Inventario (apéndice de
    # referencia, siempre presente) ─────────────────────────────────────
    _heading(doc, INVENTORY_CATEGORY)
    for fdict in INVENTORY_FUNCTIONS:
        h = _heading(doc, f"▸ {fdict['nombre']}", size=12, color=_C_ACCENT)
        _keep_block_together([h])
        _label_value_table(doc, [(label, str(fdict[key])) for key, label in INVENTORY_REPORT_FIELDS])

    _body(doc, f"Reporte generado automáticamente por el Optimizador de Funciones "
              f"· {len(session)} método(s) analizado(s) · f(x) = {fx_global}",
         size=8, italic=True, color=_C_GRAY)

    doc.save(path)
