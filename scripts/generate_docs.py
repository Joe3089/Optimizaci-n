# -*- coding: utf-8 -*-
"""
generate_docs.py — Genera la documentación técnica del Optimizador
(Tarea 4): Documentación_Técnica_Optimizador.docx y .pdf en docs/.

Usa method_docs_data.py como única fuente de contenido — no duplica texto
entre el renderer de Word y el de PDF.

Uso:
    python scripts/generate_docs.py
"""
from __future__ import annotations
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.method_docs_data import METHODS, COMPAT_MATRIX, CATEGORIES, AI_IMPLEMENTATION  # noqa: E402
from app.domain.inventory.catalog import INVENTORY_FUNCTIONS, INVENTORY_CATEGORY  # noqa: E402

# Funciones de inventario (Fase 6 del módulo de Inventario): mismo esquema
# de diccionario que METHODS, así que se agregan sin tocar la lógica de
# render de generate_docx/generate_pdf — solo aparece una categoría más.
METHODS    = METHODS + INVENTORY_FUNCTIONS
CATEGORIES = CATEGORIES + [INVENTORY_CATEGORY]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
os.makedirs(OUT_DIR, exist_ok=True)
LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "Menu", "fondo_optimizacion.png")

FIELDS_ORDER = [
    ("formula", "Fórmula"),
    ("descripcion", "Descripción"),
    ("requisitos", "Requisitos matemáticos"),
    ("convergencia", "Convergencia"),
    ("ventajas", "Ventajas"),
    ("desventajas", "Desventajas / limitaciones"),
    ("cuando_usar", "Cuándo usarlo"),
    ("cuando_no_usar", "Cuándo NO usarlo"),
    ("compatibles", "Funciones compatibles"),
    ("incompatibles", "Funciones incompatibles"),
    ("ejemplo", "Ejemplo"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  DOCX
# ═══════════════════════════════════════════════════════════════════════════
def generate_docx(path: str) -> None:
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    # Estilo base
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    # Portada
    if os.path.exists(LOGO_PATH):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(LOGO_PATH, width=Cm(3.5))
    title = doc.add_heading("Optimizador de Funciones", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_heading("Documentación Técnica de Métodos de Optimización", level=1)
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(f"Generado automáticamente — {date.today().isoformat()}")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    # Índice manual (por categoría)
    doc.add_heading("Contenido", level=1)
    doc.add_paragraph("1. Introducción")
    doc.add_paragraph("2. Matriz de compatibilidad función ↔ método")
    for i, cat in enumerate(CATEGORIES, start=3):
        doc.add_paragraph(f"{i}. {cat}")
    doc.add_page_break()

    # Introducción
    doc.add_heading("1. Introducción", level=1)
    doc.add_paragraph(
        "Este documento describe, de forma completa y verificada contra la "
        "implementación real del Optimizador (no valores genéricos de libro "
        "de texto), los 32 métodos de optimización disponibles en la "
        "aplicación: 8 unidimensionales/de gradiente, 3 metaheurísticos, "
        "6 restringidos (penalización/barrera/suma), y 14 multiobjetivo "
        "(4 técnicas clásicas del proyecto + 10 del motor de escalarización, "
        "Fase 6-8)."
    )
    doc.add_paragraph(
        "Para cada método se incluye: fórmula, descripción, requisitos "
        "matemáticos, convergencia, ventajas, desventajas, cuándo usarlo y "
        "cuándo no, funciones compatibles e incompatibles, y un ejemplo "
        "verificado. Cuando un usuario selecciona una combinación "
        "método-función incompatible, la aplicación bloquea la ejecución y "
        "muestra una explicación en español con métodos alternativos "
        "recomendados — ver la matriz de compatibilidad (sección 2)."
    )

    # Matriz de compatibilidad
    doc.add_heading("2. Matriz de compatibilidad función ↔ método", level=1)
    table = doc.add_table(rows=1, cols=4)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = (
        "Tipo de función", "Métodos compatibles", "Métodos incompatibles", "Nota")
    for row in COMPAT_MATRIX:
        cells = table.add_row().cells
        cells[0].text = row["tipo_funcion"]
        cells[1].text = row["compatibles"]
        cells[2].text = row["incompatibles"]
        cells[3].text = row["nota"]
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                para.style.font.size = Pt(9)
    doc.add_page_break()

    # Métodos por categoría
    section_num = 3
    for cat in CATEGORIES:
        doc.add_heading(f"{section_num}. {cat}", level=1)
        methods_in_cat = [m for m in METHODS if m["categoria"] == cat]
        for m in methods_in_cat:
            doc.add_heading(m["nombre"], level=2)
            for key, label in FIELDS_ORDER:
                p = doc.add_paragraph()
                run = p.add_run(f"{label}: ")
                run.bold = True
                run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
                p.add_run(m[key])
        section_num += 1
        doc.add_page_break()

    # Implementación del Agente de IA
    doc.add_heading(f"{section_num}. Implementación del Agente de IA", level=1)
    doc.add_paragraph(
        "Esta sección documenta cómo está implementado y funcionando el "
        "asistente conversacional dentro de la app (no es una descripción "
        "genérica) — ver app/infrastructure/ai_assistant.py.")
    for subtitulo, texto in AI_IMPLEMENTATION:
        doc.add_heading(subtitulo, level=2)
        doc.add_paragraph(texto)

    doc.save(path)


# ═══════════════════════════════════════════════════════════════════════════
#  PDF (reportlab — mismo estilo visual que los reportes que ya genera la app)
# ═══════════════════════════════════════════════════════════════════════════
def generate_pdf(path: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, PageBreak, Image as RLImg)
    from xml.sax.saxutils import escape as _xesc
    from scripts.pdf_cover import build_cover_elements

    styles = getSampleStyleSheet()
    h_title = ParagraphStyle("h_title", parent=styles["Title"], fontSize=22,
                              textColor=colors.HexColor("#1F4E79"))
    h1 = ParagraphStyle("h1c", parent=styles["Heading1"], fontSize=15,
                         textColor=colors.HexColor("#1F4E79"), spaceBefore=14, spaceAfter=8)
    h2 = ParagraphStyle("h2c", parent=styles["Heading2"], fontSize=12.5,
                         textColor=colors.HexColor("#2E75B6"), spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("bodyc", parent=styles["BodyText"], fontSize=9.3, leading=12.5)
    label = ParagraphStyle("labelc", parent=body, textColor=colors.HexColor("#1F4E79"))

    M = 1.8 * cm
    doc = SimpleDocTemplate(path, pagesize=A4,
                             leftMargin=M, rightMargin=M,
                             topMargin=M, bottomMargin=M)
    elems = []

    elems += build_cover_elements(
        LOGO_PATH, "Optimizador de Funciones",
        "Documentación Técnica de Métodos de Optimización",
        intro_text="", date_str=date.today().isoformat(),
        page_size=A4, top_margin=M, bottom_margin=M)
    elems.append(PageBreak())

    elems.append(Paragraph("1. Introducción", h1))
    elems.append(Paragraph(
        "Este documento describe, de forma completa y verificada contra la "
        "implementación real del Optimizador, los 32 métodos de optimización "
        "disponibles: 8 unidimensionales/de gradiente, 3 metaheurísticos, "
        "6 restringidos (penalización/barrera/suma), y 14 multiobjetivo "
        "(4 técnicas clásicas del proyecto + 10 del motor de escalarización).", body))
    elems.append(Spacer(1, 0.3*cm))
    elems.append(Paragraph(
        "Cuando un usuario selecciona una combinación método-función "
        "incompatible, la aplicación bloquea la ejecución y muestra una "
        "explicación en español con métodos alternativos recomendados — "
        "ver la matriz de compatibilidad (sección 2).", body))
    elems.append(PageBreak())

    elems.append(Paragraph("2. Matriz de compatibilidad función ↔ método", h1))
    data = [["Tipo de función", "Compatibles", "Incompatibles", "Nota"]]
    for row in COMPAT_MATRIX:
        data.append([
            Paragraph(_xesc(row["tipo_funcion"]), body),
            Paragraph(_xesc(row["compatibles"]), body),
            Paragraph(_xesc(row["incompatibles"]), body),
            Paragraph(_xesc(row["nota"]), body),
        ])
    t = Table(data, colWidths=[3.6*cm, 4.2*cm, 4.2*cm, 4.5*cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B0B0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
    ]))
    elems.append(t)
    elems.append(PageBreak())

    section_num = 3
    for cat in CATEGORIES:
        elems.append(Paragraph(f"{section_num}. {cat}", h1))
        methods_in_cat = [m for m in METHODS if m["categoria"] == cat]
        for m in methods_in_cat:
            elems.append(Paragraph(m["nombre"], h2))
            for key, flabel in FIELDS_ORDER:
                elems.append(Paragraph(
                    f"<b><font color='#1F4E79'>{_xesc(flabel)}:</font></b> {_xesc(m[key])}", body))
            elems.append(Spacer(1, 0.25*cm))
        section_num += 1
        elems.append(PageBreak())

    # Implementación del Agente de IA
    elems.append(Paragraph(f"{section_num}. Implementación del Agente de IA", h1))
    elems.append(Paragraph(
        "Esta sección documenta cómo está implementado y funcionando el "
        "asistente conversacional dentro de la app (no es una descripción "
        "genérica) — ver app/infrastructure/ai_assistant.py.", body))
    elems.append(Spacer(1, 0.2*cm))
    for subtitulo, texto in AI_IMPLEMENTATION:
        elems.append(Paragraph(_xesc(subtitulo), h2))
        elems.append(Paragraph(_xesc(texto), body))
        elems.append(Spacer(1, 0.2*cm))

    doc.build(elems)


def main() -> None:
    docx_path = os.path.join(OUT_DIR, "Documentacion_Tecnica_Optimizador.docx")
    pdf_path = os.path.join(OUT_DIR, "Documentacion_Tecnica_Optimizador.pdf")
    generate_docx(docx_path)
    print(f"OK docx: {docx_path} ({os.path.getsize(docx_path)} bytes)")
    generate_pdf(pdf_path)
    print(f"OK pdf:  {pdf_path} ({os.path.getsize(pdf_path)} bytes)")


if __name__ == "__main__":
    main()
