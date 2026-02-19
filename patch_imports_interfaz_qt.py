import re
from pathlib import Path

PATH = Path("interfaz_qt.py")

if not PATH.exists():
    raise SystemExit("No encuentro interfaz_qt.py en esta carpeta.")

text = PATH.read_text(encoding="utf-8", errors="ignore")

# Reemplaza imports viejos o rotos por los puentes
replacements = [
    (r"from\s+canvas_2d_FIXED\s+import\s+Function2DCanvas", "from canvas_2d_estilo_fixed import Function2DCanvas"),
    (r"from\s+canvas_2d_dual\s+import\s+Function2DCanvas", "from canvas_2d_estilo_fixed import Function2DCanvas"),
    (r"from\s+canvas_2d\s+import\s+Function2DCanvas", "from canvas_2d_estilo_fixed import Function2DCanvas"),

    (r"from\s+rotacion_3d_FIXED\s+import\s+Rotating3DCanvas", "from rotacion_3d_superficie import Rotating3DCanvas"),
    (r"from\s+rotacion_3d_dual\s+import\s+Rotating3DCanvas", "from rotacion_3d_superficie import Rotating3DCanvas"),
    (r"from\s+rotacion_3d\s+import\s+Rotating3DCanvas", "from rotacion_3d_superficie import Rotating3DCanvas"),
]

new_text = text
for pattern, repl in replacements:
    new_text = re.sub(pattern, repl, new_text)

# Si NO existían esos imports, los inyecta sin tocar layout/estilos
if "from canvas_2d_estilo_fixed import Function2DCanvas" not in new_text:
    # intenta insertarlo luego de los imports iniciales
    m = re.search(r"(^import .*?$|^from .*? import .*?$)", new_text, flags=re.M)
    if m:
        insert_at = m.end()
        new_text = new_text[:insert_at] + "\nfrom canvas_2d_estilo_fixed import Function2DCanvas\n" + new_text[insert_at:]
    else:
        new_text = "from canvas_2d_estilo_fixed import Function2DCanvas\n" + new_text

if "from rotacion_3d_superficie import Rotating3DCanvas" not in new_text:
    # insert después del import anterior si existe
    idx = new_text.find("from canvas_2d_estilo_fixed import Function2DCanvas")
    if idx != -1:
        line_end = new_text.find("\n", idx)
        new_text = new_text[:line_end+1] + "from rotacion_3d_superficie import Rotating3DCanvas\n" + new_text[line_end+1:]
    else:
        new_text = "from rotacion_3d_superficie import Rotating3DCanvas\n" + new_text

if new_text != text:
    PATH.write_text(new_text, encoding="utf-8")
    print("OK: interfaz_qt.py parchado (solo imports).")
else:
    print("No hubo cambios (ya estaba correcto).")
