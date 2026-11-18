# app/infrastructure/data_import.py
# Lectura de los datos que el usuario vuelca en un método: CSV/TXT, Excel,
# Word (.docx) y PDF. Solo se extrae TEXTO (celdas o líneas); nada del
# archivo se ejecuta: los valores pasan a los campos de la interfaz y los
# valida el parser matemático seguro de siempre al calcular.
from __future__ import annotations

import csv
import os
import re
import unicodedata
from typing import Dict, List

FILE_FILTER = ("Datos (*.csv *.txt *.xlsx *.xlsm *.docx *.pdf);;"
               "CSV o texto (*.csv *.txt);;Excel (*.xlsx *.xlsm);;Word (*.docx);;PDF (*.pdf)")
SUPPORTED = (".csv", ".txt", ".xlsx", ".xlsm", ".docx", ".pdf")

_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def is_number(text: str) -> bool:
    return bool(_NUM.match(text.strip().replace(",", ".")))


def _split_lines(lines: List[str]) -> List[List[str]]:
    """Separa líneas de texto en celdas (coma, punto y coma, tabulador o espacios)."""
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return []
    sample = "\n".join(lines[:8])
    if sample.count(";") > sample.count(","):
        rows = [[c.strip() for c in r] for r in csv.reader(lines, delimiter=";")]
        # coma decimal (1,5;2,7) cuando el separador es ;
        return [[c.replace(",", ".") if is_number(c) else c for c in r] for r in rows]
    if "\t" in sample:
        return [[c.strip() for c in r] for r in csv.reader(lines, delimiter="\t")]
    if "," in sample:
        return [[c.strip() for c in r] for r in csv.reader(lines, delimiter=",")]
    return [ln.split() for ln in lines]


def read_text_lines(path: str) -> List[str]:
    """Texto del archivo línea por línea (para «clave: valor» o una expresión)."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Formato no admitido ({ext or 'sin extensión'}). "
                         "Usa CSV, TXT, Excel (.xlsx), Word (.docx) o PDF.")
    if not os.path.isfile(path):
        raise ValueError(f"No se encontró el archivo «{path}».")
    try:
        if ext in (".xlsx", ".xlsm"):
            return [", ".join(r) for r in _excel_rows(path)]
        if ext == ".docx":
            import docx
            d = docx.Document(path)
            out = [p.text for p in d.paragraphs if p.text.strip()]
            for t in d.tables:
                for row in t.rows:
                    cells = [c.text.strip() for c in row.cells]
                    out.append(", ".join(c for c in cells if c))
            return out
        if ext == ".pdf":
            import fitz  # PyMuPDF
            with fitz.open(path) as doc:
                text = "\n".join(page.get_text() for page in doc)
            return [ln for ln in text.splitlines() if ln.strip()]
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            return [ln for ln in fh.read().splitlines() if ln.strip()]
    except ValueError:
        raise
    except Exception as ex:
        raise ValueError(f"No se pudo leer «{os.path.basename(path)}»: {ex}") from ex


def _excel_rows(path: str) -> List[List[str]]:
    from openpyxl import load_workbook
    ws = load_workbook(path, read_only=True, data_only=True).active
    rows = []
    for r in ws.iter_rows(values_only=True):
        cells = ["" if c is None else str(c).strip() for c in r]
        while cells and not cells[-1]:
            cells.pop()
        if any(cells):
            rows.append(cells)
    return rows


def read_rows(path: str) -> List[List[str]]:
    """Filas de celdas del archivo, sin interpretar encabezados."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        try:
            return _excel_rows(path)
        except Exception as ex:
            raise ValueError(f"No se pudo leer el Excel: {ex}") from ex
    if ext == ".docx":
        try:
            import docx
            d = docx.Document(path)
        except Exception as ex:
            raise ValueError(f"No se pudo leer el Word: {ex}") from ex
        if d.tables:      # una tabla de Word ya trae las celdas separadas
            rows = []
            for t in d.tables:
                for row in t.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        rows.append(cells)
            return rows
    return [r for r in _split_lines(read_text_lines(path)) if any(c for c in r)]


def read_table(path: str) -> List[List[str]]:
    """Como read_rows, pero descarta un encabezado de texto si el resto es numérico."""
    rows = read_rows(path)
    if len(rows) > 1 and not all(is_number(c) for c in rows[0] if c) and \
            all(is_number(c) for r in rows[1:] for c in r if c):
        rows = rows[1:]
    return rows


def norm_key(text: str) -> str:
    """«Punto inicial (x₀)» → «puntoinicialx0»: sin acentos, minúsculas, solo letras y dígitos."""
    t = unicodedata.normalize("NFKD", text.replace("₀", "0").replace("α", "alfa")
                              .replace("σ", "sigma").replace("μ", "mu"))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]", "", t)


def read_pairs(path: str) -> Dict[str, str]:
    """Pares «clave → valor» de un archivo: líneas «clave: valor» / «clave = valor»
    o tablas de dos o más columnas (la primera celda es la clave)."""
    pairs: Dict[str, str] = {}
    for ln in read_text_lines(path):
        m = re.match(r"^\s*([^:=]{1,60}?)\s*[:=]\s*(.+?)\s*$", ln)
        if m and not is_number(m.group(1)):
            pairs.setdefault(norm_key(m.group(1)), m.group(2))
    if not pairs:
        for r in read_rows(path):
            if len(r) >= 2 and r[0] and not is_number(r[0]):
                pairs.setdefault(norm_key(r[0]), ", ".join(c for c in r[1:] if c))
    return pairs


# ── Archivos con varios ejercicios ───────────────────────────────────────────
import tempfile  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Callable, Optional, Tuple  # noqa: E402

# «Ejercicio 1», «Problema 2:», «Ej. 3)», «Caso 4 –», «Pregunta N° 5», «## Actividad 6»
EXERCISE_HEAD = re.compile(
    r"^\s*(?:#+\s*)?(?:ejercicio|problema|ej\.?|caso|actividad|pregunta|exercise)"
    r"\s*(?:n\s*[º°o.]*\s*)?(\d+)\b", re.IGNORECASE)


@dataclass
class Exercise:
    title: str
    lines: List[str] = field(default_factory=list)

    def preview(self, limit: int = 180) -> str:
        text = " · ".join(ln.strip() for ln in self.lines if ln.strip())
        return text if len(text) <= limit else text[:limit - 1] + "…"


def _docx_lines_in_order(path: str) -> List[str]:
    """Párrafos y filas de tabla en el orden del documento (celdas con tabulador)."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    d = docx.Document(path)
    out: List[str] = []
    for el in d.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            t = Paragraph(el, d).text
            if t.strip():
                out.append(t)
        elif tag == "tbl":
            for row in Table(el, d).rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    out.append("\t".join(cells))
    return out


def _split_by_headers(lines: List[str], default_title: str) -> List[Exercise]:
    exs: List[Exercise] = []
    for ln in lines:
        if EXERCISE_HEAD.match(ln):
            exs.append(Exercise(ln.strip()[:90]))
        elif exs:
            exs[-1].lines.append(ln)
    exs = [e for e in exs if any(l.strip() for l in e.lines)]
    return exs if len(exs) >= 2 else [Exercise(default_title, lines)]


def split_exercises(path: str) -> List[Exercise]:
    """Ejercicios del archivo: separados por encabezados («Ejercicio 1», «Problema 2»…)
    y, en Excel, una hoja por ejercicio. Devuelve uno solo si no hay varios."""
    name = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        found: List[Exercise] = []
        for ws in wb.worksheets:
            lines = []
            for r in ws.iter_rows(values_only=True):
                cells = ["" if c is None else str(c).strip() for c in r]
                while cells and not cells[-1]:
                    cells.pop()
                if any(cells):
                    lines.append("\t".join(cells))
            if not lines:
                continue
            parts = _split_by_headers(lines, f"Hoja «{ws.title}»")
            if len(parts) >= 2:
                found += [Exercise(f"{ws.title} · {p.title}", p.lines) for p in parts]
            else:
                found.append(parts[0])
        return found if found else [Exercise(name, [])]
    lines = _docx_lines_in_order(path) if ext == ".docx" else read_text_lines(path)
    return _split_by_headers(lines, name)


def exercise_file(ex: Exercise) -> str:
    """Escribe un ejercicio en un CSV temporal para leerlo con los lectores de siempre."""
    folder = os.path.join(tempfile.gettempdir(), "optimizador_ejercicios")
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="ejercicio_", dir=folder)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(ex.lines) + "\n")
    return tmp


def resolve_exercise(path: str, exercise: Optional[int] = None,
                     chooser: Optional[Callable[[List[Exercise], str], Optional[int]]] = None
                     ) -> Optional[Tuple[str, str]]:
    """Si el archivo trae varios ejercicios, elige uno (índice dado o `chooser`, que
    muestra el cuadro de selección). Devuelve (ruta a leer, nombre a mostrar) o None
    si el usuario canceló."""
    name = os.path.basename(path)
    try:
        exs = split_exercises(path)
    except ValueError:
        raise
    except Exception:
        exs = []
    if len(exs) < 2:
        return path, name
    idx = exercise if exercise is not None else (chooser(exs, name) if chooser else None)
    if idx is None:
        return None
    if not 0 <= idx < len(exs):
        raise ValueError(f"El archivo tiene {len(exs)} ejercicios; no existe el n.º {idx + 1}.")
    return exercise_file(exs[idx]), f"{name} — {exs[idx].title}"
