# -*- coding: utf-8 -*-
"""
Pruebas de las mejoras del dashboard (sin red ni audio: IA y voz simuladas):

  1. Recuadro de resultado: una sola instancia, dentro del chat de Opti.
  2. Cambio de método: el dashboard se vacía de inmediato, se recalcula con el
     nuevo método y un cálculo obsoleto nunca sobrescribe al más reciente.
  3. Tabla, 2D y 3D de TODOS los métodos, construidas con los mismos datos del
     motor (se comparan celdas, coordenadas dibujadas y punto óptimo).
  4. Desigualdad de Chebyshev (μ=50, σ=10, k=2 → ≥ 75 % en [30, 70]; k=3;
     muestra; entradas inválidas).

    venv\\Scripts\\python -m pytest tests/test_dashboard_sync.py -v
    venv\\Scripts\\python tests/test_dashboard_sync.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PyQt5 import QtCore, QtWidgets  # noqa: E402

import app.ui.main_window as _mw  # noqa: E402
from app.application import result_analysis as ra  # noqa: E402
from app.application.problem_catalog import SPECS  # noqa: E402
from app.infrastructure.auto_intervention import MockAIExplainer, MockSpeaker  # noqa: E402
from app.ui import dashboard_views as dv  # noqa: E402
from app.application.problem_catalog import is_problem_method  # noqa: E402
from app.ui.main_window import (  # noqa: E402
    METHODS_1D, METHODS_HEU, METHODS_MD, METHODS_META, METHODS_MULTIOBJ)
from app.ui.result_box import ResultBox  # noqa: E402

FUNCTION_METHODS = METHODS_1D + METHODS_HEU + METHODS_MD + METHODS_META + METHODS_MULTIOBJ
RESULTS: list = []          # (prueba, ok, detalle)
_APP = None
_WIN = None


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return bool(ok)


def _window():
    global _APP, _WIN
    if _WIN is None:
        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        _mw._err = lambda parent, title, msg, detail="": None   # modales: ver smoke_test
        QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: None)
        QtWidgets.QMessageBox.critical = staticmethod(lambda *a, **k: None)
        _WIN = _mw.InterfazOptimizacion(resource_path=lambda p: p)
        _WIN.intervention.set_backends(explainer=MockAIExplainer(),
                                       speaker=MockSpeaker(_WIN.intervention))
    return _WIN


def _wait(cond, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _APP.processEvents(QtCore.QEventLoop.AllEvents, 50)
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def _case(m):
    if m in METHODS_MD:
        return dict(fx="(x1-2)**2 + (x2-1)**2", gx="x1+x2-2", vars_="x1 x2", x0="0.0, 0.0")
    if m in METHODS_MULTIOBJ:
        return dict(fx="(x-3)**2", gx="", vars_="", x0="1.0")
    return dict(fx="x**2 - 4*x + 5", gx="", vars_="", x0="1.0")


def _set_inputs(win, fx, gx="", vars_="", x0="1.0", lo=0.0, hi=5.0, tol="0.1"):
    win.edt_fx.setText(fx); win.edt_gx.setText(gx); win.edt_vars.setText(vars_)
    win.edt_x0.setText(x0); win.smin.setValue(lo); win.smax.setValue(hi)
    win.edt_tol.setText(tol)


def _table(win):
    return [[(win.table.item(r, c).text() if win.table.item(r, c) else "")
             for c in range(win.table.columnCount())] for r in range(win.table.rowCount())]


def _switch(win, m, tag):
    """Selecciona m como lo haría el usuario y comprueba la invalidación inmediata."""
    prev = win._shown_method
    win.select_method(m)
    if prev and prev != m:
        check(f"[cambio] {tag}: al elegir «{m}» desaparece «{prev}» de inmediato",
              win.table.rowCount() == 0 and win._shown_method == ""
              and not (win.canvas2d and win.canvas2d.isVisibleTo(win))
              and not win.result_box.isVisibleTo(win.result_box.parentWidget()),
              f"filas={win.table.rowCount()} mostrado={win._shown_method!r}")
    if is_problem_method(m) and not win.prob_panel.has_user_data():
        # Sin datos del usuario no se calcula ningún ejemplo precargado
        check(f"[datos] {m}: sin datos del usuario no se calcula nada",
              win.table.rowCount() == 0 and "ingresa tus datos" in win.lbl_dash.text()
              and not win._auto_timer.isActive(), win.lbl_dash.text())
        win.prob_panel.load_example()
        win.ejecutar()
    ok = _wait(lambda: win._shown_method == m or win.lbl_dash.text().startswith(("✗", "⚠")))
    return ok and win._shown_method == m


# ── 1. Recuadro de resultado dentro del chat ─────────────────────────────────
def _test_result_box_location(win):
    boxes = win.findChildren(ResultBox)
    check("[C1] existe una sola instancia del recuadro de resultado", len(boxes) == 1,
          f"{len(boxes)} instancias")
    box = win.result_box

    def inside(w, ancestor):
        while w is not None:
            if w is ancestor:
                return True
            w = w.parentWidget()
        return False

    check("[C1] el recuadro está dentro del panel de Opti (área de mensajes)",
          inside(box, win.ai_panel) and inside(box, win.ai_panel._chat_split))
    check("[C1] el recuadro ya no está en el dashboard", not inside(box, win.right))
    _set_inputs(win, "x**2 - 4*x + 5")
    win.select_method("Fibonacci")
    n_log = len(win.intervention.log)
    win.ejecutar()
    _wait(lambda: win.intervention.current_explanation != "", 10)
    e = win.intervention.log[-1] if len(win.intervention.log) > n_log else {}
    check("[IA] resultado correcto: Opti no interviene (ni IA, ni voz, ni chat automático)",
          e.get("ai_status", "").startswith("sin intervención") and not e.get("speech_requested")
          and not e.get("t_chat_open") and win.ai_panel.isHidden(), str(e.get("ai_status")))
    check("[IA] resultado correcto: el recuadro se actualiza con el resumen del motor",
          box.last_event is e.get("event") and "Resumen del motor" in box.lbl_ai.text())
    _set_inputs(win, "1 / (x - 2)", lo=1.0, hi=3.0, tol="0.1")
    win.ejecutar()
    _wait(lambda: win.intervention.current_explanation != "" and not win.ai_panel.isHidden(), 10)
    e = win.intervention.log[-1]
    check("[IA] función incorrecta: Opti interviene y abre el chat por su cuenta",
          e.get("chat") and bool(e.get("t_chat_open")) and e["event"].kind != ra.SUCCESS)
    check("[C1] tras el error, el recuadro y el chat quedan visibles",
          box.isVisibleTo(win) and win.ai_panel.isVisibleTo(win))
    chat = win.ai_panel.chat_view.toPlainText()
    expl = win.intervention.current_explanation
    check("[C1] sin redundancia: la explicación aparece en el chat y la del recuadro queda oculta",
          box.lbl_ai.isHidden() and not box.btn_text.isChecked()
          and bool(expl) and expl[:60] in chat, expl[:80])
    box.btn_text.setChecked(True)
    shown = not box.lbl_ai.isHidden()
    box.btn_text.setChecked(False)
    check("[C1] el botón 📄 conserva la función de mostrar la explicación", shown)
    check("[C1] el recuadro conserva sus datos (título, hechos, botones)",
          bool(box.lbl_title.text()) and bool(box.lbl_facts.text())
          and box.btn_voice.isEnabled())


# ── 2/3. Cambio de método + consistencia motor ↔ tabla ↔ gráficas ────────────
def _consistency(win, m):
    hist = win._hist
    keys = win._table_keys
    cells = _table(win)
    ok_tab = (len(cells) == len(hist) and all(
        cells[i][j] == dv.cell_text(hist[i].get(k, "")) for i in range(len(hist))
        for j, k in enumerate(keys)))
    check(f"[tabla] {m}: celdas = datos del motor ({len(hist)} filas, {len(keys)} columnas)",
          ok_tab and len(hist) > 0)
    if m in METHODS_MULTIOBJ:
        check(f"[gráficas] {m}: 2D y 3D dibujadas",
              win.canvas2d.isVisibleTo(win) and win.canvas3d.isVisibleTo(win))
        return
    pd = win._plot_data
    pts = [p for p in (dv.row_point(r) for r in hist) if p is not None]
    nv = len(pd.get("vars", []))
    ok_traj = bool(pts) and nv in (1, 2) and np.allclose(
        pd["traj"][0], [p[0] for p in pts]) and (nv == 1 or np.allclose(
            pd["traj"][1], [p[1] for p in pts]))
    check(f"[2D] {m}: cada punto dibujado es una iteración real ({len(pts)} puntos)", ok_traj)
    ax = win.canvas2d.fig.axes[0]
    marker = [ln for ln in ax.get_lines() if ln.get_marker() == "D"]
    last = pts[-1]
    ok_opt = bool(marker) and np.isclose(marker[-1].get_xdata()[0], last[0])
    if nv == 1:
        ok_opt = ok_opt and np.isclose(marker[-1].get_ydata()[0], pd["opt"][1])
    else:
        ok_opt = ok_opt and np.isclose(marker[-1].get_ydata()[0], last[1])
    check(f"[2D] {m}: el punto óptimo marcado = última iteración del motor", ok_opt,
          f"marcador={[(l.get_xdata()[0], l.get_ydata()[0]) for l in marker]} motor={last}")
    # la celda de la tabla con el punto final coincide con lo dibujado
    pkey = next((k for k in dv.POINT_KEYS if k in keys), None)
    if pkey:
        cell_pt = dv.parse_point(cells[-1][keys.index(pkey)])
        check(f"[tabla↔2D] {m}: el punto de la última fila coincide con el dibujado",
              cell_pt is not None and np.allclose(cell_pt[:nv], last[:nv], rtol=1e-4, atol=1e-6))
    fk = hist[-1].get("f_k", hist[-1].get("f(x)", hist[-1].get("f_opt")))
    if (isinstance(fk, (int, float, np.number)) and m not in METHODS_MD
            and "f_best" not in hist[-1]):
        check(f"[2D] {m}: f(x*) dibujado = f del motor",
              abs(pd["f"][-1] - float(fk)) <= 1e-6 * max(1.0, abs(float(fk))),
              f"{pd['f'][-1]} vs {fk}")
    ax3 = win.canvas3d.fig.axes[0] if win.canvas3d.fig.axes else None
    ok3 = win.canvas3d.isVisibleTo(win) and ax3 is not None
    if ok3 and nv == 1:
        ok3 = ax3.get_xlabel() == "Iteración k"
        path = [ln for ln in ax3.get_lines() if ln.get_label() == "xₖ por iteración"]
        ok3 = ok3 and bool(path) and np.allclose(path[0].get_data_3d()[1], pd["traj"][0])
    elif ok3:
        path = [ln for ln in ax3.get_lines() if ln.get_label() == "Trayectoria"]
        ok3 = bool(path) and np.allclose(path[0].get_data_3d()[0], pd["traj"][0]) \
            and np.allclose(path[0].get_data_3d()[1], pd["traj"][1])
    check(f"[3D] {m}: vista 3D con los mismos puntos reales "
          f"({'iteración·x·f' if nv == 1 else 'superficie f(x1,x2)'})", ok3)


def _test_method_switching(win):
    # Recorrido pedido: Fibonacci → Armijo → Wolfe → Búsqueda Local → multidimensional
    _set_inputs(win, "x**2 - 4*x + 5")
    win.select_method("Fibonacci"); win.ejecutar()
    for m in ("Armijo", "Wolfe", "Búsqueda Local"):
        ok = _switch(win, m, "recorrido")
        check(f"[cambio] recorrido → {m}: se recalcula solo y muestra «{m}»", ok,
              win.lbl_dash.text())
        check(f"[cambio] {m}: sesión y tabla pertenecen al método nuevo",
              win._session[-1]["metodo"] == m and win._metodo == m)
    _set_inputs(win, "(x1-1)**2 + (x2-2)**2", x0="0, 0", vars_="x1 x2")
    ok = _switch(win, "Grad. Conjugado", "recorrido")
    check("[cambio] recorrido → Grad. Conjugado (multidimensional)", ok, win.lbl_dash.text())
    if ok:
        _consistency(win, "Grad. Conjugado")
        tr = win._plot_data.get("traj", [])
        check("[cambio] Grad. Conjugado: llega al óptimo real (1, 2)",
              len(tr) == 2 and np.allclose([tr[0][-1], tr[1][-1]], [1, 2], atol=1e-4))

    # Todos los métodos de función, uno tras otro, por cambio de método
    for m in FUNCTION_METHODS:
        c = _case(m)
        _set_inputs(win, c["fx"], c["gx"], c["vars_"], c["x0"])
        if win._shown_method == m:
            win.select_method("Fibonacci" if m != "Fibonacci" else "Armijo")
            _wait(lambda: win._shown_method != m, 10)
        ok = _switch(win, m, "todos")
        check(f"[cambio] → {m}: recalculado y mostrado", ok, win.lbl_dash.text())
        if ok:
            _consistency(win, m)

    # Métodos del catálogo (Estadística, IO, Optimización)
    for spec in SPECS:
        ok = _switch(win, spec.label, "catálogo")
        check(f"[cambio] → {spec.label}: recalculado y mostrado", ok, win.lbl_dash.text())
        if not ok:
            continue
        hist = win._hist
        cells = _table(win)
        check(f"[tabla] {spec.label}: celdas = datos del motor",
              len(cells) == len(hist) > 0 and all(
                  cells[i][j] == dv.cell_text(hist[i].get(k, ""))
                  for i in range(len(hist)) for j, k in enumerate(win._table_keys)))
        check(f"[2D] {spec.label}: gráfica 2D del resultado", win.canvas2d.isVisibleTo(win))
        has3d = win._plot_data.get("has_3d")
        check(f"[3D] {spec.label}: " + ("vista 3D dibujada" if has3d else
              "sin 3D natural y se informa el motivo"),
              (has3d and win.canvas3d.isVisibleTo(win)) or
              (not has3d and win.lbl_3d.text().startswith("Sin gráfica 3D")
               and len(win.lbl_3d.text()) > 40), win.lbl_3d.text()[:120])


def _test_stale_results(win):
    """Un cálculo lento del método anterior no debe pisar al más reciente."""
    _set_inputs(win, "x**2 - 4*x + 5")
    win.select_method("Búsqueda Local"); win.ejecutar()
    orig = _mw._run_method
    calls = []

    def slow(metodo, *a, **k):
        calls.append(metodo)
        if metodo == "Fibonacci":
            time.sleep(1.2)          # el usuario cambia de método mientras tanto
        return orig(metodo, *a, **k)

    _mw._run_method = slow
    try:
        n_sess = len(win._session)
        win.select_method("Fibonacci")
        win._auto_timer.stop()
        # Mientras Fibonacci calcula (1.2 s en su hilo), el usuario elige Armijo
        QtCore.QTimer.singleShot(300, lambda: win.select_method("Armijo"))
        win._auto_run()              # vuelve cuando el hilo de Fibonacci termina
        _wait(lambda: win._shown_method == "Armijo", 15)
    finally:
        _mw._run_method = orig
    check("[obsoletos] se calcularon ambos métodos", "Fibonacci" in calls and "Armijo" in calls,
          str(calls))
    check("[obsoletos] el dashboard muestra el método más reciente (Armijo)",
          win._shown_method == "Armijo" and win._metodo == "Armijo")
    check("[obsoletos] el resultado de Fibonacci se descartó (ni tabla ni sesión)",
          [s["metodo"] for s in win._session[n_sess:]] == ["Armijo"],
          str([s["metodo"] for s in win._session[n_sess:]]))
    _consistency(win, "Armijo")


def _test_failure_clears(win):
    """Caso de la captura: Monte Carlo y luego Fibonacci con 1/(x-2) en [1, 3]."""
    ok = _switch(win, "Simulación de Monte Carlo", "fallo")
    check("[fallo] Monte Carlo calculado", ok)
    _set_inputs(win, "1 / (x - 2)", lo=1.0, hi=3.0, tol="2")
    win.select_method("Fibonacci")
    _wait(lambda: win._shown_method == "Fibonacci" or
          win.lbl_dash.text().startswith(("✗", "⚠", "Método")) and not win._auto_timer.isActive()
          and win.intervention.current_event is not None
          and win.intervention.current_event.metodo == "Fibonacci", 15)
    text = " ".join(" ".join(r) for r in _table(win))
    check("[fallo] no quedan datos de Monte Carlo en la tabla ni en las gráficas",
          "Error estándar" not in text and win._shown_method != "Simulación de Monte Carlo"
          and win._plot_data.get("kind") != "problem", win.lbl_dash.text())
    ev = win.intervention.current_event
    check("[fallo] el error se presenta para Fibonacci (recuadro + Opti)",
          ev is not None and ev.metodo == "Fibonacci" and ev.kind != ra.SUCCESS,
          getattr(ev, "kind", None))
    _set_inputs(win, "x**2 - 4*x + 5")


def _test_no_duplicates(win):
    _set_inputs(win, "x**2 - 4*x + 5")
    win.select_method("Sección Dorada"); win.ejecutar()
    n1, rows1, s1 = len(win._hist), win.table.rowCount(), len(win._session)
    win.ejecutar()
    check("[duplicados] recalcular no duplica filas ni datos",
          win.table.rowCount() == rows1 == n1 and len(win._session) == s1 + 1)


# ── 4. Chebyshev ─────────────────────────────────────────────────────────────
def _test_chebyshev(win):
    from app.statistics import chebyshev as cb
    ok = _switch(win, "Desigualdad de Chebyshev", "chebyshev")
    check("[Chebyshev] disponible en la sección Estadística",
          ok and win.cmb_sec.currentText() == "Estadística")

    def run(**vals):
        base = dict(modo=cb.MODE_SIGMA, mu="50", sigma="10", var="", datos="",
                    tipo_var=cb.VAR_SAMPLE, k="2", limites="")
        base.update(vals)
        win.prob_panel.set_values(base)
        win.ejecutar()
        return " | ".join(" ".join(r) for r in _table(win))

    t = run()
    r = win._session[-1]["resumen"]
    check("[Chebyshev] vista 3D de la cota según el intervalo (a, b)",
          win._plot_data.get("has_3d") and win.canvas3d.isVisibleTo(win))
    check("[Chebyshev] μ=50, σ=10, k=2 → P(|X − 50| < 20) ≥ 0.75", "P(|X − 50| < 20) ≥ 0.75" in r, r)
    check("[Chebyshev] porcentaje mínimo 75 % e intervalo [30, 70] en la tabla",
          "75 %" in t and "[30, 70]" in t)
    check("[Chebyshev] tabla paso a paso (fórmula, sustitución, desarrollo, interpretación)",
          all(s in t for s in ("Fórmula", "Sustitución", "Desarrollo", "Interpretación")))
    check("[Chebyshev] gráfica 2D del intervalo alrededor de μ", win.canvas2d.isVisibleTo(win))
    t = run(k="3")
    check("[Chebyshev] k=3 → ≥ 88.89 % en [20, 80]", "88.89 %" in t and "[20, 80]" in t)
    t = run(modo=cb.MODE_VAR, var="100", sigma="")
    check("[Chebyshev] con varianza σ²=100 da el mismo resultado", "75 %" in t and "[30, 70]" in t)
    t = run(k="", limites="30, 70")
    check("[Chebyshev] k deducido de los límites 30, 70 → k = 2", "75 %" in t and "[30, 70]" in t)
    t = run(modo=cb.MODE_DATA, datos="12, 15, 14, 10, 18, 20, 13, 16, 15, 17", mu="", sigma="")
    check("[Chebyshev] muestra: media 15 y varianza muestral 78/9 calculadas",
          "150 / 10" in t and "78 / 9" in t and "8.66667" in t)
    for name, vals, frag in (
            ("σ negativa", dict(sigma="-10"), "no puede ser negativa"),
            ("varianza negativa", dict(modo=cb.MODE_VAR, var="-4"), "varianza no puede"),
            ("k ≤ 1", dict(k="1"), "k > 1"),
            ("dato faltante", dict(mu=""), "Falta el dato"),
            ("texto no numérico", dict(sigma="diez"), "debe ser un número"),
            ("muestra insuficiente", dict(modo=cb.MODE_DATA, datos="5"), "al menos 2 datos")):
        run(**vals)
        ev = win.intervention.current_event
        check(f"[Chebyshev] entrada inválida ({name}) se rechaza con mensaje claro",
              win.table.rowCount() == 0 and win.lbl_dash.text().startswith("✗")
              and ev is not None and ev.kind == ra.PARAMETER_ERROR
              and frag in (ev.error_message + win.lbl_dash.text()),
              win.lbl_dash.text()[:140])


def _test_file_loading(win):
    """El usuario vuelca SUS datos desde un archivo y obtiene los resultados."""
    import tempfile
    from app.statistics import chebyshev as cb
    tmp = tempfile.mkdtemp(prefix="opt_datos_")
    data = [12, 15, 14, 10, 18, 20, 13, 16, 15, 17]
    csv1 = os.path.join(tmp, "muestra.csv")
    with open(csv1, "w", encoding="utf-8") as fh:
        fh.write("valor\n" + "\n".join(str(v) for v in data) + "\n")
    csv2 = os.path.join(tmp, "regresion.csv")
    with open(csv2, "w", encoding="utf-8") as fh:
        fh.write("x;y\n1;2,1\n2;3,9\n3;6,2\n4;7,8\n5;10,1\n6;12,2\n")
    xlsx = os.path.join(tmp, "muestra.xlsx")
    from openpyxl import Workbook
    wb = Workbook(); ws = wb.active; ws.append(["dato"])
    for v in data:
        ws.append([v])
    wb.save(xlsx)

    _switch(win, "Estadística Descriptiva", "archivo")
    for path in (csv1, xlsx):
        win.prob_panel.clear_values()
        win.prob_panel.load_file(path)
        win.ejecutar()
        mean = win._session[-1]["hist"][1]["Valor"] if win._session else None
        check(f"[archivo] Descriptiva con {os.path.basename(path)}: encabezado omitido y "
              "media calculada con los datos del archivo (15)",
              win._shown_method == "Estadística Descriptiva" and mean == 15.0, str(mean))
    _switch(win, "Regresión Lineal", "archivo")
    win.prob_panel.clear_values()
    win.prob_panel.load_file(csv2)
    win.ejecutar()
    r = win._session[-1]["resumen"]
    check("[archivo] Regresión con CSV «;» y coma decimal → ŷ = −0.02 + 2.02·x",
          "2.0200" in r and win._plot_data.get("has_3d"), r)
    _switch(win, "Desigualdad de Chebyshev", "archivo")
    win.prob_panel.clear_values()
    win.prob_panel.set_values({"k": "2"})
    win.prob_panel.load_file(csv1)
    win.ejecutar()
    check("[archivo] Chebyshev: al cargar la muestra se activa el modo «Datos de una muestra»",
          win.prob_panel.values()["modo"] == cb.MODE_DATA
          and "78 / 9" in " ".join(" ".join(x) for x in _table(win)))


def _make_files(tmp):
    import docx
    import fitz
    from openpyxl import Workbook
    txt = "f(x): x**2 - 4*x + 5\nx0: 1\nrango: 0, 5\ntolerancia: 0.1"
    out = {}
    d = docx.Document()
    for ln in txt.splitlines():
        d.add_paragraph(ln)
    out["docx"] = os.path.join(tmp, "funcion.docx"); d.save(out["docx"])
    p = fitz.open(); p.new_page().insert_text((72, 72), txt)
    out["pdf"] = os.path.join(tmp, "funcion.pdf"); p.save(out["pdf"])
    wb = Workbook(); ws = wb.active
    for r in (("función", "x**2 - 4*x + 5"), ("x0", 1), ("mínimo", 0), ("máximo", 5)):
        ws.append(r)
    out["xlsx"] = os.path.join(tmp, "funcion.xlsx"); wb.save(out["xlsx"])
    out["csv"] = os.path.join(tmp, "funcion.csv")
    with open(out["csv"], "w", encoding="utf-8") as fh:
        fh.write("campo,valor\nf(x),x**2 - 4*x + 5\nx0,1\nmínimo,0\nmáximo,5\n")
    d = docx.Document(); t = d.add_table(rows=0, cols=1)
    for v in ["dato", 12, 15, 14, 10, 18, 20, 13, 16, 15, 17]:
        t.add_row().cells[0].text = str(v)
    out["tabla_docx"] = os.path.join(tmp, "muestra.docx"); d.save(out["tabla_docx"])
    p = fitz.open(); p.new_page().insert_text(
        (72, 72), "Valores: 60, 100, 120\nPesos: 10, 20, 30\nCapacidad: 50")
    out["mochila_pdf"] = os.path.join(tmp, "mochila.pdf"); p.save(out["mochila_pdf"])
    d = docx.Document()
    for ln in ("Demanda anual (D): 1000", "Costo de pedido (S): 50",
               "Costo de mantenimiento (H): 2"):
        d.add_paragraph(ln)
    out["inv_docx"] = os.path.join(tmp, "inventario.docx"); d.save(out["inv_docx"])
    return out


def _test_user_data_everywhere(win):
    """Cargar archivo (CSV/Excel/Word/PDF), ejemplo, vaciar y exportar en los
    métodos con f(x), en inventario y en el catálogo."""
    import tempfile
    from app.ui.main_window import INVENTORY_METHOD_LABELS
    files = _make_files(tempfile.mkdtemp(prefix="opt_user_"))

    win.select_method("Armijo")
    from PyQt5.QtWidgets import QPushButton
    bar_texts = [b.text() for b in win.row_fdata.findChildren(QPushButton)]
    check("[f(x)] la barra tiene solo «Cargar archivo» y «Usar ejemplo» (sin Vaciar ni Exportar)",
          win.row_fdata.isVisibleTo(win.row_fdata.parentWidget())
          and bar_texts == ["📂 Cargar archivo", "📋 Usar ejemplo"], str(bar_texts))
    for kind in ("csv", "xlsx", "docx", "pdf"):
        win.f_clear()
        got = win.f_load_file(files[kind])
        win.ejecutar()
        check(f"[f(x)] función cargada desde {kind.upper()} y calculada (x* = 2)",
              win.edt_fx.text() == "x**2 - 4*x + 5" and win._shown_method == "Armijo"
              and np.isclose(win._plot_data["opt"][0], 2, atol=1e-4), f"{got} {win.lbl_dash.text()}")
    win.limpiar()
    check("[f(x)] el botón azul «Limpiar» borra la función y el dashboard",
          win.edt_fx.text() == "" and win.table.rowCount() == 0)
    win.select_method("MD: Pen. (BFGS)")
    win.f_load_example()
    win.ejecutar()
    check("[f(x)] «Usar ejemplo» de un método restringido calcula x* ≈ (1.5, 0.5)",
          win._shown_method == "MD: Pen. (BFGS)" and np.allclose(
              [win._plot_data["opt"][0], win._plot_data["opt"][1]], [1.5, 0.5], atol=0.01))

    inv = next(lbl for lbl in INVENTORY_METHOD_LABELS if "Clásico" in lbl)
    win.select_method(inv)
    ip = win.inv_panel
    from PyQt5.QtWidgets import QPushButton
    inv_texts = {b.text() for b in ip.findChildren(QPushButton)}
    check("[inventario] Cargar archivo y Usar ejemplo, sin Vaciar ni Exportar; arranca vacío",
          {"📂 Cargar archivo", "📋 Usar ejemplo"} <= inv_texts
          and not any("Vaciar" in t or "Exportar" in t for t in inv_texts)
          and not ip.has_user_data(), str(inv_texts))
    ip._on_generar()
    check("[inventario] sin datos no se genera la función (indica qué falta)",
          "Falta ingresar" in ip.lbl_error.text() and ip.current_model() is None,
          ip.lbl_error.text())
    ip.load_file(files["inv_docx"])
    ip._on_generar()
    _wait(lambda: win._shown_method == inv, 15)
    q = float(win._plot_data.get("opt", (0,))[0])
    check("[inventario] datos cargados desde Word → Q* = √(2·1000·50/2) ≈ 223.61",
          win._shown_method == inv and abs(q - 223.607) < 0.01, f"{q} {win.lbl_dash.text()}")
    win.limpiar()
    check("[inventario] el botón azul «Limpiar» deja los campos sin datos",
          not ip.has_user_data())
    win.select_method(inv)

    _switch(win, "Estadística Descriptiva", "archivo")
    win.prob_panel.clear_values()
    win.prob_panel.load_file(files["tabla_docx"])
    win.ejecutar()
    check("[catálogo] tabla de Word cargada en Descriptiva (media 15)",
          win._session[-1]["hist"][1]["Valor"] == 15.0)
    win.select_method("Combinatoria: Mochila 0/1")
    win.prob_panel.clear_values()
    filled = win.prob_panel.load_file(files["mochila_pdf"])
    win.ejecutar()
    check("[catálogo] PDF «campo: valor» llena valores, pesos y capacidad (óptimo 220)",
          sorted(filled) == ["cap", "pesos", "valores"] and "220" in win._session[-1]["resumen"],
          f"{filled} {win._session[-1]['resumen']}")


def _test_multi_exercise(win):
    """Archivo con varios ejercicios: cuadro de selección con Aceptar/Cancelar."""
    import tempfile
    import docx
    import fitz
    from openpyxl import Workbook
    from app.infrastructure import data_import as di
    from app.ui.main_window import INVENTORY_METHOD_LABELS
    tmp = tempfile.mkdtemp(prefix="opt_ejercicios_")
    d = docx.Document()
    for ln in ("Guía de ejercicios de optimización",
               "Ejercicio 1: Minimizar una parábola", "f(x): x**2 - 4*x + 5", "rango: 0, 5",
               "Ejercicio 2: Desplazada", "f(x): (x - 3)**2 + 1", "rango: 0, 6",
               "Ejercicio 3: Cuártica", "f(x): (x - 1)**4", "rango: -2, 4"):
        d.add_paragraph(ln)
    p_docx = os.path.join(tmp, "guia.docx"); d.save(p_docx)
    p = fitz.open(); p.new_page().insert_text((72, 72), "Problema 1\n12, 15, 14, 10, 18\n"
                                              "Problema 2\n2, 4, 4, 4, 5, 5, 7, 9")
    p_pdf = os.path.join(tmp, "muestras.pdf"); p.save(p_pdf)
    wb = Workbook(); ws = wb.active; ws.title = "Tienda A"
    for r in (("Demanda anual (D)", 1000), ("Costo de pedido (S)", 50), ("Costo de mantenimiento (H)", 2)):
        ws.append(r)
    ws = wb.create_sheet("Tienda B")
    for r in (("Demanda anual (D)", 4000), ("Costo de pedido (S)", 25), ("Costo de mantenimiento (H)", 5)):
        ws.append(r)
    p_xlsx = os.path.join(tmp, "tiendas.xlsx"); wb.save(p_xlsx)

    exs = di.split_exercises(p_docx)
    check("[ejercicios] Word con 3 ejercicios detectados (el texto previo no cuenta)",
          len(exs) == 3 and exs[1].title.startswith("Ejercicio 2"), [e.title for e in exs])
    check("[ejercicios] PDF con 2 problemas y Excel con 2 hojas",
          len(di.split_exercises(p_pdf)) == 2 and len(di.split_exercises(p_xlsx)) == 2)

    seen = {}

    def click(choice):
        def act():
            dlg = QtWidgets.QApplication.activeModalWidget()
            seen["dlg"] = dlg
            if dlg is None:
                return
            seen["opciones"] = [r.text() for r in dlg.radios]
            seen["previa"] = dlg.findChildren(QtWidgets.QLabel, "exPreview")[1].text()
            if choice is None:
                dlg.btn_cancel.click()
            else:
                dlg.radios[choice].setChecked(True)
                dlg.btn_ok.click()
        QtCore.QTimer.singleShot(250, act)

    win.select_method("Armijo")
    win.f_clear()
    click(1)
    win.f_load_file(p_docx)
    check("[ejercicios] se abre el cuadro con las 3 opciones y su vista previa",
          seen.get("opciones") == ["1. Ejercicio 1: Minimizar una parábola",
                                   "2. Ejercicio 2: Desplazada", "3. Ejercicio 3: Cuártica"]
          and "(x - 3)**2 + 1" in seen.get("previa", ""), str(seen.get("opciones")))
    check("[ejercicios] «Aceptar» carga el ejercicio elegido (2)",
          win.edt_fx.text() == "(x - 3)**2 + 1" and win.smax.value() == 6.0
          and "Ejercicio 2" in win.lbl_f_file.text(), win.edt_fx.text())
    win.ejecutar()
    check("[ejercicios] el ejercicio elegido se calcula (x* = 3)",
          win._shown_method == "Armijo" and np.isclose(win._plot_data["opt"][0], 3, atol=1e-4))
    click(None)
    r = win.f_load_file(p_docx)
    check("[ejercicios] «Cancelar» no cambia los datos", r == [] and
          win.edt_fx.text() == "(x - 3)**2 + 1")

    win.select_method("Estadística Descriptiva")
    win.prob_panel.clear_values()
    click(1)
    win.prob_panel.load_file(p_pdf)
    win.ejecutar()
    check("[ejercicios] catálogo: PDF, «Problema 2» elegido → media 5",
          win._session[-1]["hist"][1]["Valor"] == 5.0, str(win._session[-1]["hist"][1]))

    inv = next(lbl for lbl in INVENTORY_METHOD_LABELS if "Clásico" in lbl)
    win.select_method(inv)
    ip = win.inv_panel
    ip.clear_values()
    click(1)
    ip.load_file(p_xlsx)
    ip._on_generar()
    _wait(lambda: win._shown_method == inv, 15)
    check("[ejercicios] inventario: hoja «Tienda B» elegida → Q* = √(2·4000·25/5) = 200",
          np.isclose(win._plot_data.get("opt", (0,))[0], 200, atol=0.01)
          and "Tienda B" in ip.lbl_file.text(), ip.lbl_file.text())
    q = ip.load_file(p_xlsx, exercise=0)
    check("[ejercicios] un solo ejercicio o índice dado: sin cuadro", bool(q) and
          ip.spin_D.value() == 1000.0)


def _test_rotation_and_legends(win):
    """Todas las vistas 3D giran (sola y con el ratón) y todas las leyendas se
    arrastran y se contraen/expanden con doble clic."""
    from matplotlib.backend_bases import MouseEvent
    cases = [("Búsqueda Local", None), ("Grad. Conjugado", None), ("MO — Bisección", None),
             ("Regresión Lineal", "cat"), ("PL: Simplex", "cat"),
             ("Desigualdad de Chebyshev", "cat"), ("Simulación de Monte Carlo", "cat")]
    for m, kind in cases:
        if kind is None:
            win.select_method(m)
            win.f_load_example()
            win.ejecutar()
        else:
            win.select_method(m)
            win.prob_panel.load_example()
            win.ejecutar()
        _wait(lambda: win._shown_method == m, 10)
        ax3 = next((a for a in win.canvas3d.fig.axes if getattr(a, "name", "") == "3d"), None)
        az0 = ax3.azim if ax3 is not None else None
        _wait(lambda: False, 0.4)
        check(f"[3D gira] {m}: la vista 3D gira sola",
              ax3 is not None and win._rot_timer.isActive() and ax3.azim != az0,
              f"azim {az0} → {getattr(ax3, 'azim', None)}")
        legs = []
        for c in (win.canvas2d, win.canvas3d):
            legs += [(c, l) for l in _dv_legends(c.fig)]
        if not legs:          # p. ej. Monte Carlo: sus gráficas no tienen leyenda
            continue
        check(f"[leyendas] {m}: todas las leyendas se pueden arrastrar ({len(legs)})",
              all(l.get_draggable() for _, l in legs))
        c, leg = legs[0]
        c.draw()
        bb = leg.get_window_extent(c.get_renderer())
        x, y = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
        n0 = len(leg.get_texts())
        ev = MouseEvent("button_press_event", c, x, y, button=1, dblclick=True)
        c.callbacks.process("button_press_event", ev)
        new = [l for l in _dv_legends(c.fig) if getattr(l, "_opti_entries", None)]
        collapsed = bool(new) and len(new[0].get_texts()) == 0 and \
            new[0].get_title().get_text().startswith("▸")
        c.draw()
        bb = new[0].get_window_extent(c.get_renderer()) if new else bb
        ev = MouseEvent("button_press_event", c, (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2,
                        button=1, dblclick=True)
        c.callbacks.process("button_press_event", ev)
        back = [l for l in _dv_legends(c.fig) if len(l.get_texts()) == n0
                and not getattr(l, "_opti_entries", None)]
        check(f"[leyendas] {m}: doble clic la contrae y otro doble clic la expande",
              collapsed and bool(back) and back[0].get_draggable())
    win._rot_paused = False


def _dv_legends(fig):
    from app.ui.dashboard_views import _legends
    return _legends(fig)


def _test_layout(win):
    from PyQt5.QtWidgets import QSplitter
    root = win.left.parentWidget()
    sps = [w for w in win.findChildren(QSplitter) if w is not win.ai_panel._chat_split]
    check("[pantalla] ningún divisor permite aplastar un panel hasta cortarlo",
          isinstance(root, QSplitter) and all(not sp.childrenCollapsible() for sp in sps)
          and not win.ai_panel._chat_split.childrenCollapsible())
    check("[pantalla] tabla, 2D y 3D con altura mínima legible",
          win.frame_2d.minimumHeight() >= 150 and win.frame_3d.minimumHeight() >= 150)
    check("[pantalla] panel de Opti y panel de control con desplazamiento (no se recortan)",
          win.ai_panel._scroll.widgetResizable() and win._left_scroll.widgetResizable())


def run_all():
    RESULTS.clear()
    win = _window()
    for fn in (_test_result_box_location, _test_method_switching, _test_stale_results,
               _test_failure_clears, _test_no_duplicates, _test_chebyshev,
               _test_file_loading, _test_user_data_everywhere, _test_multi_exercise,
               _test_rotation_and_legends, _test_layout):
        try:
            fn(win)
        except Exception as ex:  # una excepción también es un fallo visible
            import traceback
            check(f"{fn.__name__}: excepción", False, f"{ex}\n{traceback.format_exc()}")
    return RESULTS


def test_dashboard_sync():
    res = run_all()
    failed = [f"{n} — {d}" for n, ok, d in res if not ok]
    assert res, "no se ejecutó ninguna comprobación"
    assert not failed, "Comprobaciones FALLIDAS:\n" + "\n".join(failed)


if __name__ == "__main__":
    res = run_all()
    for n, ok, d in res:
        print(f"[{'OK' if ok else 'X '}] {n}" + ("" if ok else f"   ← {d}"))
    bad = sum(1 for _, ok, _ in res if not ok)
    print(f"\nTOTAL: {len(res)}   OK: {len(res) - bad}   FALLIDAS: {bad}")
    sys.exit(1 if bad else 0)
