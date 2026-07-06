# app/infrastructure/reporting/pdf_export.py
# Exportación a PDF profesional — extraído de
# InterfazOptimizacion.export_pdf (app/ui/main_window.py), Fase 4c.
from __future__ import annotations

import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, Image as RLImg,
                                HRFlowable, PageBreak, KeepTogether,
                                FrameBreak)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus.flowables import HRFlowable

from app.application import report_content


def write_pdf(path: str, session: list) -> None:
    """
    PDF profesional en A4 vertical:
      Pág 1: RESUMEN — descripción función, tipo, estudios aplicables,
                       métodos aplicados y justificación de cada uno.
      Pág N: Por cada método — info detallada, tabla de iteraciones,
             cálculos paso a paso y gráficas 2D / 3D.
    Todo ajustado a los márgenes del documento.
    """
    fx_global = session[0]["fx"] if session else ""

    # A4 vertical con márgenes estándar
    ML = MR = 2.0 * cm
    MT = MB = 2.0 * cm
    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=ML, rightMargin=MR,
                            topMargin=MT, bottomMargin=MB)
    PW = A4[0] - ML - MR   # ancho útil ≈ 17.0 cm

    # ── Paleta ─────────────────────────────────────────────────────────
    cDark  = colors.HexColor("#0A1428")
    cMid   = colors.HexColor("#101E38")
    cHead  = colors.HexColor("#1A3060")
    cHead2 = colors.HexColor("#0D1E44")
    cBlue  = colors.HexColor("#2A50A0")
    cAqua  = colors.HexColor("#5DC8D0")
    cWhite = colors.white
    cAlt   = colors.HexColor("#E8F2FF")
    cGold  = colors.HexColor("#FFD700")
    cGray  = colors.HexColor("#B0C4D8")
    cDkTxt = colors.HexColor("#111830")

    # ── Estilos de párrafo ─────────────────────────────────────────────
    def _PS(name, **kw):
        defaults = dict(fontName="Helvetica", fontSize=10,
                        textColor=cWhite, leading=14)
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)

    sBanner  = _PS("Ban", fontSize=16, fontName="Helvetica-Bold",
                   alignment=TA_CENTER, leading=20, textColor=cWhite)
    sSub     = _PS("Sub", fontSize=10, textColor=cAqua,
                   alignment=TA_CENTER, leading=13)
    sSecHdr  = _PS("SHd", fontSize=12, fontName="Helvetica-Bold",
                   textColor=cAqua, leading=15, spaceBefore=6, spaceAfter=3)
    sBody    = _PS("Bod", fontSize=9, textColor=cGray,
                   alignment=TA_JUSTIFY, leading=13, spaceAfter=3)
    sBodyDk  = _PS("BDk", fontSize=9, textColor=cDkTxt,
                   alignment=TA_JUSTIFY, leading=13)
    sLabel   = _PS("Lbl", fontSize=9, fontName="Helvetica-Bold",
                   textColor=cAqua, leading=12)
    sVal     = _PS("Val", fontSize=9, textColor=cWhite, leading=12)
    sStep    = _PS("Stp", fontSize=9, fontName="Helvetica-Bold",
                   textColor=cGold, leading=12)
    sStepBd  = _PS("StB", fontSize=8, textColor=cGray,
                   leading=11, leftIndent=10)
    sSmall   = _PS("Sml", fontSize=7, textColor=cGray,
                   alignment=TA_CENTER, leading=10)
    sItalic  = _PS("Itl", fontSize=9, fontName="Helvetica-Oblique",
                   textColor=cGray, leading=12, alignment=TA_JUSTIFY)

    def _HR(thick=0.8, color=None):
        return HRFlowable(width="100%", thickness=thick,
                          color=color or cBlue,
                          spaceBefore=4, spaceAfter=4)

    def _banner(title, subtitle=""):
        rows = [[Paragraph(title, sBanner)]]
        if subtitle:
            rows.append([Paragraph(subtitle, sSub)])
        t = Table(rows, colWidths=[PW])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), cDark),
            ("TOPPADDING",    (0,0), (-1,-1), 12),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING",   (0,0), (-1,-1), 14),
            ("RIGHTPADDING",  (0,0), (-1,-1), 14),
        ]))
        return t

    def _section_box(content_rows, col_w=None):
        """Tabla de dos columnas etiqueta/valor con fondo cMid."""
        cw = col_w or [PW * 0.30, PW * 0.70]
        t = Table(content_rows, colWidths=cw)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), cMid),
            ("GRID",          (0,0), (-1,-1), 0.3, cBlue),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("RIGHTPADDING",  (0,0), (-1,-1), 8),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ]))
        return t

    def _full_box(text_para):
        """Caja de ancho completo con fondo cMid."""
        t = Table([[text_para]], colWidths=[PW])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), cMid),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ]))
        return t

    def _step_box(title, body_lines):
        """Caja de paso con super/subíndices via ReportLab (sin imágenes)."""
        rows = [[Paragraph(report_content.math_to_reportlab(title), sStep)]]
        for line in body_lines:
            if line.strip():
                rows.append([Paragraph(
                    report_content.math_to_reportlab(line.strip()), sStepBd)])
        t = Table(rows, colWidths=[PW])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,0),  cHead2),
            ("BACKGROUND",    (0,1), (-1,-1), cMid),
            ("GRID",          (0,0), (-1,-1), 0.3, cBlue),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("RIGHTPADDING",  (0,0), (-1,-1), 8),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ]))
        return t

    # ── Estilos BLANCOS para Resumen y Conclusión ─────────────────────
    cBlack   = colors.HexColor("#111111")
    cDkBlue  = colors.HexColor("#1A3A6A")
    cLightBg = colors.HexColor("#F4F7FB")
    cBorder  = colors.HexColor("#B0C4D8")
    cAccent  = colors.HexColor("#1A3A6A")

    sWTitle  = ParagraphStyle("WTi", fontName="Helvetica-Bold", fontSize=18,
                   textColor=colors.white, alignment=TA_CENTER, leading=22)
    sWSub    = ParagraphStyle("WSu", fontName="Helvetica", fontSize=10,
                   textColor=colors.HexColor("#DDECFF"), alignment=TA_CENTER, leading=13)
    sWSecHdr = ParagraphStyle("WSH", fontName="Helvetica-Bold", fontSize=12,
                   textColor=cDkBlue, leading=15, spaceBefore=6, spaceAfter=3)
    sWLabel  = ParagraphStyle("WLb", fontName="Helvetica-Bold", fontSize=9,
                   textColor=cDkBlue, leading=12)
    sWVal    = ParagraphStyle("WVa", fontName="Helvetica", fontSize=9,
                   textColor=cBlack, leading=12)
    sWBody   = ParagraphStyle("WBo", fontName="Helvetica", fontSize=9,
                   textColor=cBlack, alignment=TA_JUSTIFY, leading=13, spaceAfter=3)
    sWStep   = ParagraphStyle("WSt", fontName="Helvetica-Bold", fontSize=9,
                   textColor=colors.white, leading=12)
    sWStepBd = ParagraphStyle("WSb", fontName="Helvetica", fontSize=9,
                   textColor=cBlack, leading=12, leftIndent=10,
                   alignment=TA_JUSTIFY)
    sWSmall  = ParagraphStyle("WSm", fontName="Helvetica", fontSize=7,
                   textColor=colors.HexColor("#666666"), alignment=TA_CENTER, leading=10)
    sWConc   = ParagraphStyle("WCo", fontName="Helvetica", fontSize=9,
                   textColor=cBlack, alignment=TA_JUSTIFY, leading=14, spaceAfter=4)
    sWConcH  = ParagraphStyle("WCH", fontName="Helvetica-Bold", fontSize=10,
                   textColor=cDkBlue, leading=14, spaceBefore=8, spaceAfter=3)

    def _HR_w(thick=0.8):
        return HRFlowable(width="100%", thickness=thick,
                          color=cBorder, spaceBefore=4, spaceAfter=4)

    def _banner_w(title, subtitle=""):
        """Banner con fondo azul oscuro para la página de resumen."""
        rows = [[Paragraph(title, sWTitle)]]
        if subtitle:
            rows.append([Paragraph(subtitle, sWSub)])
        t = Table(rows, colWidths=[PW])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), cDkBlue),
            ("TOPPADDING",    (0,0), (-1,-1), 14),
            ("BOTTOMPADDING", (0,0), (-1,-1), 12),
            ("LEFTPADDING",   (0,0), (-1,-1), 14),
            ("RIGHTPADDING",  (0,0), (-1,-1), 14),
        ]))
        return t

    def _section_box_w(content_rows, col_w=None):
        """Tabla etiqueta/valor con fondo blanco."""
        cw = col_w or [PW * 0.30, PW * 0.70]
        t = Table(content_rows, colWidths=cw)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), colors.white),
            ("GRID",          (0,0), (-1,-1), 0.5, cBorder),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("RIGHTPADDING",  (0,0), (-1,-1), 8),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("BACKGROUND",    (0,0), (0,-1),  cLightBg),
        ]))
        return t

    def _full_box_w(text_para):
        """Caja de ancho completo con fondo blanco."""
        t = Table([[text_para]], colWidths=[PW])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), cLightBg),
            ("BOX",           (0,0), (-1,-1), 0.5, cBorder),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ]))
        return t

    def _method_box_w(number, name, body_text):
        """Caja de método con encabezado azul y cuerpo blanco."""
        rows = [
            [Paragraph(f"{number}. {name}", sWStep)],
            [Paragraph(body_text, sWStepBd)],
        ]
        t = Table(rows, colWidths=[PW])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,0),  cDkBlue),
            ("BACKGROUND",    (0,1), (-1,-1), colors.white),
            ("BOX",           (0,0), (-1,-1), 0.5, cBorder),
            ("LINEBELOW",     (0,0), (-1,0),  0.5, cBorder),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ]))
        return t

    elems = []

    # ══════════════════════════════════════════════════════════════════
    # PÁGINA 1-2: RESUMEN GENERAL  (fondo blanco, letras negras)
    # ══════════════════════════════════════════════════════════════════
    elems.append(_banner_w(
        "Optimizador de Funciones",
        f"Reporte de Análisis  ·  {len(session)} método(s) ejecutado(s)"))
    elems.append(Spacer(1, 0.3*cm))

    # ── 1. Función analizada ───────────────────────────────────────────
    elems.append(_HR_w(1.5))
    elems.append(Paragraph("1. Función Analizada", sWSecHdr))
    tipo_fn = report_content.classify_function(fx_global)
    elems.append(_section_box_w([
        [Paragraph("Expresión:", sWLabel),  Paragraph(fx_global, sWVal)],
        [Paragraph("Tipo:", sWLabel),        Paragraph(tipo_fn, sWVal)],
        [Paragraph("Variables:", sWLabel),   Paragraph(", ".join(session[0]["vars"]), sWVal)],
        [Paragraph("Dominio:", sWLabel),     Paragraph(f"[{session[0]['lo']:.4g},  {session[0]['hi']:.4g}]", sWVal)],
    ]))
    elems.append(Spacer(1, 0.2*cm))
    desc_fn = report_content.describe_function(fx_global)
    elems.append(_full_box_w(Paragraph(desc_fn, sWBody)))
    elems.append(Spacer(1, 0.25*cm))

    # ── 2. Tipos de estudio aplicables ────────────────────────────────
    elems.append(_HR_w(1.5))
    elems.append(Paragraph("2. Tipos de Estudio Aplicables a la Función", sWSecHdr))
    desc_estudios = report_content.describe_applicable_studies(fx_global)
    elems.append(_full_box_w(Paragraph(desc_estudios, sWBody)))
    elems.append(Spacer(1, 0.25*cm))

    # ── 3. Métodos aplicados y justificación ──────────────────────────
    elems.append(_HR_w(1.5))
    elems.append(Paragraph("3. Métodos Aplicados", sWSecHdr))
    for i, s in enumerate(session, 1):
        justif = report_content.justify_method(s["metodo"], fx_global)
        elems.append(Spacer(1, 0.15*cm))
        # KeepTogether: el encabezado azul y el cuerpo blanco
        # nunca se separan entre páginas
        elems.append(KeepTogether([_method_box_w(i, s["metodo"], justif)]))

    elems.append(Spacer(1, 0.15*cm))
    elems.append(_HR_w(0.5))
    elems.append(Paragraph("Generado automáticamente · Optimizador de Funciones", sWSmall))

    # ══════════════════════════════════════════════════════════════════
    # PÁGINAS POR MÉTODO
    # ══════════════════════════════════════════════════════════════════
    try:
        for idx_s, entry in enumerate(session, 1):
            metodo = entry["metodo"]
            hist   = entry["hist"]
            elems.append(PageBreak())

            # Banner del método + Info: se mantienen juntos en la misma página
            tipo_est = report_content.classify_study(metodo)
            elems.append(KeepTogether([
                _banner(
                    f"Método {idx_s}: {metodo}",
                    f"f(x) = {entry['fx']}"),
                Spacer(1, 0.2*cm),
                _HR(1.0, cBlue),
                Paragraph("Información del Método", sSecHdr),
                _section_box([
                    [Paragraph("Método:", sLabel),          Paragraph(metodo, sVal)],
                    [Paragraph("Función:", sLabel),         Paragraph(entry["fx"], sVal)],
                    [Paragraph("Tipo de estudio:", sLabel), Paragraph(tipo_est, sVal)],
                    [Paragraph("Variables:", sLabel),       Paragraph(", ".join(entry["vars"]), sVal)],
                    [Paragraph("Intervalo:", sLabel),       Paragraph(f"[{entry['lo']:.4g},  {entry['hi']:.4g}]", sVal)],
                ]),
                Spacer(1, 0.2*cm),
            ]))

            # ── Tabla de iteraciones ───────────────────────────────────────
            if hist:
                h_cols: list = []
                for rec in hist:
                    for k in rec:
                        if k not in h_cols: h_cols.append(k)

                h_rows_raw = []
                for rec in hist:
                    row_v = []
                    for col in h_cols:
                        v = rec.get(col, "")
                        if isinstance(v, np.ndarray):
                            v = "[" + ", ".join(f"{x:.4g}" for x in v.flat) + "]"
                        elif isinstance(v, float):
                            v = f"{v:.5g}"
                        row_v.append(str(v))
                    h_rows_raw.append(row_v)

                nc = len(h_cols)

                # Calcular ancho proporcional al contenido de cada columna
                col_max_len = []
                for j, col in enumerate(h_cols):
                    max_len = len(col)
                    for rv in h_rows_raw:
                        max_len = max(max_len, len(rv[j]))
                    col_max_len.append(max_len)
                total_chars = sum(col_max_len) or 1
                col_widths = [PW * (cl / total_chars) for cl in col_max_len]
                # Mínimo 1.0 cm por columna
                from reportlab.lib.units import cm as _cm
                col_widths = [max(w, 1.0*_cm) for w in col_widths]
                # Si la suma excede PW, escalar hacia abajo
                total_w = sum(col_widths)
                if total_w > PW:
                    scale = PW / total_w
                    col_widths = [w * scale for w in col_widths]

                hdr_st = _PS(f"th{idx_s}", fontSize=7, fontName="Helvetica-Bold",
                             textColor=cWhite, alignment=TA_CENTER)
                cel_st = _PS(f"td{idx_s}", fontSize=7, textColor=cWhite,
                             alignment=TA_CENTER, leading=10)
                cel_alt= _PS(f"ta{idx_s}", fontSize=7, textColor=cWhite,
                             alignment=TA_CENTER, leading=10)

                tbl_data = [[Paragraph(c, hdr_st) for c in h_cols]]
                for i_r, row_v in enumerate(h_rows_raw):
                    st = cel_alt if i_r % 2 == 1 else cel_st
                    tbl_data.append([Paragraph(v, st) for v in row_v])

                # Dos tonos oscuros para filas alternas — texto blanco siempre legible
                cRowA = colors.HexColor("#101E38")   # azul oscuro (igual que cMid)
                cRowB = colors.HexColor("#162848")   # azul medio-oscuro

                # Estilo base reutilizable para la tabla de iteraciones
                def _make_tbl_style(bg0=cRowA, bg1=cRowB):
                    return TableStyle([
                        ("BACKGROUND",    (0,0), (-1,0),  cHead),
                        ("ROWBACKGROUNDS",(0,1), (-1,-1), [bg0, bg1]),
                        ("GRID",          (0,0), (-1,-1), 0.4, cBlue),
                        ("TOPPADDING",    (0,0), (-1,-1), 3),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                        ("LEFTPADDING",   (0,0), (-1,-1), 2),
                        ("RIGHTPADDING",  (0,0), (-1,-1), 2),
                        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                    ])

                # ── Ancla: título de sección + encabezado + primeras filas
                # (se mantienen juntos para que el encabezado nunca quede
                #  huérfano al final de una página)
                N_ANCHOR = min(4, len(tbl_data))   # encabezado + hasta 3 filas de datos
                anchor_tbl = Table(tbl_data[:N_ANCHOR], colWidths=col_widths)
                anchor_tbl.setStyle(_make_tbl_style())
                elems.append(KeepTogether([
                    _HR(1.0, cBlue),
                    Paragraph("Tabla de Iteraciones", sSecHdr),
                    anchor_tbl,
                ]))

                # ── Filas restantes: encabezado repetido al inicio de cada
                # página nueva, con colores alternados en continuidad
                if len(tbl_data) > N_ANCHOR:
                    n_anchor_data = N_ANCHOR - 1        # filas de datos ya mostradas
                    bg0 = cRowB if (n_anchor_data % 2 == 1) else cRowA
                    bg1 = cRowA if bg0 == cRowB else cRowB
                    remaining_data = [tbl_data[0]] + tbl_data[N_ANCHOR:]
                    remaining_tbl = Table(
                        remaining_data, colWidths=col_widths, repeatRows=1)
                    remaining_tbl.setStyle(_make_tbl_style(bg0, bg1))
                    elems.append(remaining_tbl)
            else:
                elems.append(_HR(1.0, cBlue))
                elems.append(Paragraph("Tabla de Iteraciones", sSecHdr))
                elems.append(Paragraph("Sin datos de iteración.", sBody))

            # ── Cálculos paso a paso ───────────────────────────────────────
            elems.append(Spacer(1, 0.4*cm))
            steps = report_content.build_step_by_step(entry)
            if steps:
                # Anclar el título de la sección con el primer paso para que
                # el encabezado nunca quede solo al final de una página
                first_title, first_body = steps[0]
                first_lines = [l for l in first_body.split("\n") if l.strip()]
                elems.append(KeepTogether([
                    _HR(1.0, cBlue),
                    Paragraph("Cálculos Paso a Paso", sSecHdr),
                    Spacer(1, 0.15*cm),
                    _step_box(first_title, first_lines),
                ]))
                for step_title, step_body in steps[1:]:
                    body_lines = [l for l in step_body.split("\n") if l.strip()]
                    elems.append(Spacer(1, 0.12*cm))
                    # Cada caja se mantiene unida: el título nunca queda
                    # solo al fondo de una página separado de su contenido
                    elems.append(KeepTogether([_step_box(step_title, body_lines)]))
                elems.append(Spacer(1, 0.12*cm))
            else:
                elems.append(_HR(1.0, cBlue))
                elems.append(Paragraph("Cálculos Paso a Paso", sSecHdr))

            # ── Gráficas (siempre en página nueva) ────────────────────────
            elems.append(PageBreak())
            imgs = report_content.generate_plot_png(entry)

            if imgs:
                img_w = (PW - 0.4*cm) / 2
                img_h = img_w * 0.55

                def _img_cell(img_buf, lbl):
                    try:
                        return [
                            Paragraph(lbl, _PS(f"il{lbl}",
                                fontSize=8, textColor=cAqua,
                                alignment=TA_CENTER,
                                fontName="Helvetica-Bold")),
                            RLImg(img_buf, width=img_w, height=img_h)
                        ]
                    except:
                        return [Paragraph(lbl, sSmall),
                                Paragraph("[imagen no disponible]", sSmall)]

                pairs = [imgs[i:i+2] for i in range(0, len(imgs), 2)]
                img_tables = []
                for pair in pairs:
                    row_cells = [_img_cell(img_buf, lbl) for img_buf, lbl in pair]
                    while len(row_cells) < 2:
                        row_cells.append([Spacer(1,1), Spacer(1,1)])
                    rt = Table([row_cells], colWidths=[PW/2, PW/2])
                    rt.setStyle(TableStyle([
                        ("VALIGN",        (0,0), (-1,-1), "TOP"),
                        ("LEFTPADDING",   (0,0), (-1,-1), 2),
                        ("RIGHTPADDING",  (0,0), (-1,-1), 2),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
                    ]))
                    img_tables.append(rt)
                    img_tables.append(Spacer(1, 0.2*cm))

                # Banner de gráficas + primera fila juntos (no se parte)
                graf_banner = _banner(f"Gráficas — {metodo}", f"f(x) = {entry['fx']}")
                first_block = [graf_banner, Spacer(1, 0.25*cm)]
                if img_tables:
                    first_block.append(img_tables[0])
                elems.append(KeepTogether(first_block))
                # Resto de filas de imágenes fluyen libremente
                for tbl in img_tables[1:]:
                    elems.append(tbl)


        # ══════════════════════════════════════════════════════════════════
        # PÁGINA FINAL: CONCLUSIÓN GENERALIZADA  (fondo blanco, letra negra)
        # ══════════════════════════════════════════════════════════════════
        elems.append(PageBreak())
        elems.append(_banner_w(
            "Conclusión General",
            f"Análisis de f(x) = {fx_global}  ·  {len(session)} método(s)"))
        elems.append(Spacer(1, 0.35*cm))

        # ── Qué se hizo ────────────────────────────────────────────────────
        elems.append(_HR_w(1.5))
        elems.append(Paragraph("¿Qué se realizó?", sWSecHdr))
        metodos_lista = ", ".join(s["metodo"] for s in session)
        n_iter_total  = sum(len(s.get("hist") or []) for s in session)
        qué_texto = (
            f"Se llevó a cabo el análisis y optimización numérica de la función "
            f"f(x) = {fx_global}, clasificada como función de tipo "
            f"{report_content.classify_function(fx_global)}. "
            f"Para ello se aplicaron {len(session)} método(s) de optimización: "
            f"{metodos_lista}. "
            f"En total se ejecutaron {n_iter_total} iteración(es) computacionales, "
            f"registrando en cada paso los valores del punto actual, el valor de la "
            f"función, el gradiente, el tamaño de paso y los criterios de convergencia. "
            f"Los resultados se presentan en tablas de iteraciones completas y se "
            f"visualizan mediante gráficas 2D y 3D del comportamiento de la función "
            f"y la trayectoria de convergencia de cada método."
        )
        elems.append(_full_box_w(Paragraph(qué_texto, sWConc)))
        elems.append(Spacer(1, 0.25*cm))

        # ── Ventajas y beneficios ──────────────────────────────────────────
        elems.append(_HR_w(1.5))
        elems.append(Paragraph("Ventajas y Beneficios", sWSecHdr))

        ventajas = [
            ("Diversidad metodológica",
             "La aplicación de múltiples métodos sobre la misma función permite "
             "comparar su eficiencia, velocidad de convergencia y robustez, "
             "eligiendo el más adecuado según las características del problema."),
            ("Trazabilidad completa",
             "El registro iteración por iteración de todos los parámetros "
             "(x_k, f(x_k), ‖∇f‖, α, dirección de descenso) garantiza total "
             "transparencia del proceso numérico y facilita la detección de "
             "anomalías o comportamientos inesperados."),
            ("Visualización clara",
             "Las gráficas 2D y 3D permiten interpretar visualmente el "
             "comportamiento de la función y la trayectoria de convergencia, "
             "facilitando la comprensión sin necesidad de interpretar solo datos numéricos."),
            ("Exportación profesional",
             "La generación automática de reportes en Excel y PDF con todos los "
             "cálculos paso a paso reduce el tiempo de documentación y garantiza "
             "reproducibilidad y presentación formal de los resultados."),
            ("Criterios de convergencia garantizados",
             "Los métodos Armijo y Wolfe incluyen condiciones matemáticas "
             "verificadas en cada iteración, asegurando que el algoritmo avance "
             "hacia el óptimo sin pasos excesivamente grandes o pequeños."),
        ]
        for titulo_v, texto_v in ventajas:
            elems.append(Spacer(1, 0.1*cm))
            v_rows = [
                [Paragraph(f"✦  {titulo_v}", sWLabel)],
                [Paragraph(texto_v, sWConc)],
            ]
            vt = Table(v_rows, colWidths=[PW])
            vt.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (0,0),  cLightBg),
                ("BACKGROUND",    (0,1), (-1,-1), colors.white),
                ("BOX",           (0,0), (-1,-1), 0.4, cBorder),
                ("LINEBELOW",     (0,0), (-1,0),  0.4, cBorder),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("LEFTPADDING",   (0,0), (-1,-1), 10),
                ("RIGHTPADDING",  (0,0), (-1,-1), 10),
            ]))
            elems.append(vt)

        elems.append(Spacer(1, 0.25*cm))

        # ── Utilidad futura ────────────────────────────────────────────────
        elems.append(_HR_w(1.5))
        elems.append(Paragraph("Utilidad en Diversas Áreas a Futuro", sWSecHdr))

        areas = [
            ("Ingeniería y Diseño",
             "Los métodos de optimización son fundamentales en el diseño de "
             "estructuras, circuitos eléctricos, sistemas de control y rutas "
             "logísticas. La capacidad de encontrar mínimos de funciones de costo "
             "o máximos de funciones de rendimiento es clave en cualquier proceso "
             "de diseño óptimo."),
            ("Inteligencia Artificial y Machine Learning",
             "El entrenamiento de redes neuronales es en esencia un problema de "
             "minimización de una función de pérdida en miles de dimensiones. "
             "Los métodos de descenso de gradiente con condiciones de Wolfe/Armijo "
             "son la base de optimizadores como Adam, SGD y L-BFGS usados en "
             "frameworks como TensorFlow y PyTorch."),
            ("Economía y Finanzas",
             "La optimización de portafolios de inversión, la minimización de "
             "riesgo financiero y la maximización de utilidad en modelos "
             "econométricos dependen directamente de métodos numéricos robustos "
             "como los implementados en esta herramienta."),
            ("Ciencias Naturales e Investigación",
             "En física, química y biología computacional, la minimización de "
             "energía de sistemas moleculares, la calibración de modelos y la "
             "estimación de parámetros de ecuaciones diferenciales son problemas "
             "de optimización donde estos métodos encuentran aplicación directa."),
            ("Robótica y Control Automático",
             "La planificación de trayectorias de robots, el control predictivo "
             "de procesos industriales (MPC) y la calibración de controladores "
             "PID óptimos son aplicaciones directas de los métodos de optimización "
             "con restricciones estudiados en este reporte."),
        ]
        for area, desc_area in areas:
            elems.append(Spacer(1, 0.1*cm))
            elems.append(Paragraph(f"▸  {area}", sWConcH))
            elems.append(_full_box_w(Paragraph(desc_area, sWConc)))

        elems.append(Spacer(1, 0.3*cm))
        elems.append(_HR_w(1.0))
        elems.append(Paragraph(
            f"Reporte generado automáticamente por el Optimizador de Funciones  "
            f"·  {len(session)} método(s) analizado(s)  ·  f(x) = {fx_global}",
            sWSmall))

        doc.build(elems)
    except Exception:
        raise
