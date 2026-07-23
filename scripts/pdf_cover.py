# -*- coding: utf-8 -*-
"""
pdf_cover.py — Portada centrada y reutilizable para los PDF generados por
generate_docs.py y generate_test_functions_doc.py (logo grande, bloque
completo centrado verticalmente en la página, no pegado arriba).
"""
from __future__ import annotations
import os


def build_cover_elements(logo_path: str, title: str, subtitle: str,
                          intro_text: str, date_str: str,
                          page_size, top_margin: float, bottom_margin: float) -> list:
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Spacer, Image as RLImg
    from reportlab.lib.enums import TA_CENTER

    h_title = ParagraphStyle("cover_title", fontSize=22, leading=26,
                              alignment=TA_CENTER, fontName="Helvetica-Bold",
                              textColor=colors.HexColor("#1F4E79"))
    h_sub = ParagraphStyle("cover_sub", fontSize=13, leading=17,
                            alignment=TA_CENTER, textColor=colors.HexColor("#2E75B6"))
    h_date = ParagraphStyle("cover_date", fontSize=9.3, leading=12,
                             alignment=TA_CENTER, textColor=colors.HexColor("#606060"))
    h_intro = ParagraphStyle("cover_intro", fontSize=9.5, leading=13.5,
                              alignment=TA_CENTER, textColor=colors.HexColor("#303030"))

    LOGO_SIZE = 6.5 * cm
    usable_h  = page_size[1] - top_margin - bottom_margin

    block: list = []
    content_h = 0.0
    if logo_path and os.path.exists(logo_path):
        block.append(RLImg(logo_path, width=LOGO_SIZE, height=LOGO_SIZE, hAlign="CENTER"))
        content_h += LOGO_SIZE + 0.6 * cm
    block.append(Spacer(1, 0.5 * cm)); content_h += 0.5 * cm
    block.append(Paragraph(title, h_title)); content_h += 1.0 * cm
    block.append(Paragraph(subtitle, h_sub)); content_h += 0.8 * cm
    block.append(Spacer(1, 0.3 * cm)); content_h += 0.3 * cm
    block.append(Paragraph(f"Generado automáticamente — {date_str}", h_date)); content_h += 0.5 * cm
    if intro_text:
        block.append(Spacer(1, 0.6 * cm)); content_h += 0.6 * cm
        block.append(Paragraph(intro_text, h_intro)); content_h += 2.4 * cm  # ~varias líneas

    # Spacer superior que empuja el bloque hacia el centro vertical de la
    # página (estimación de altura suficiente para un bloque corto de
    # estructura fija como esta portada; nunca negativo).
    top_spacer_h = max((usable_h - content_h) / 2.0, 0.5 * cm)
    return [Spacer(1, top_spacer_h)] + block
