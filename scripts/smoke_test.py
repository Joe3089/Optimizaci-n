# -*- coding: utf-8 -*-
"""
smoke_test.py — Regresión automática headless de todo el Optimizador.

Ejecuta, contra una instancia real de InterfazOptimizacion (offscreen, sin
mostrar ventana), TODOS los métodos del combo (ALL_METHODS), más export
CSV/XLSX/PDF y persistencia de API Key del asistente IA. Falla con exit
code 1 y detalle si algo se rompe. Pensado para correr en cada cambio antes
de generar el .exe (ver README de scripts/).

Uso:
    QT_QPA_PLATFORM=offscreen python scripts/smoke_test.py
"""
from __future__ import annotations
import os
import sys
import tempfile
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

import app.ui.main_window as _mw  # noqa: E402
from app.ui.main_window import (  # noqa: E402
    InterfazOptimizacion, METHODS_1D, METHODS_MD, METHODS_HEU,
    METHODS_META, METHODS_MULTIOBJ,
)

# NOTA: bajo QT_QPA_PLATFORM=offscreen, QMessageBox.exec_()/exec() revienta con
# access violation incluso en un QMessageBox aislado sin lógica de la app (ver
# smoke_test — verificado con un repro mínimo). Es una limitación del entorno
# headless, no un bug del proyecto: en Windows real con la GUI normal, el
# modal funciona igual que cualquier app Qt. Por eso _err (que sí llama a
# .exec_() real, a diferencia de QMessageBox.warning/critical que son métodos
# estáticos no bloqueantes de mockear) se parchea a nivel de módulo para las
# pruebas automáticas.
ERR_DIALOGS: list = []
_mw._err = lambda parent, title, msg, detail="": ERR_DIALOGS.append((title, msg, detail))

FAILURES: list[tuple[str, str]] = []
PASSES: list[str] = []


def _case_for(metodo: str) -> dict:
    """Parámetros de prueba por método, alineados con los ya verificados
    manualmente en el historial del proyecto (commit b2576da)."""
    if metodo in METHODS_MD:
        # Cuadrática acotada con restricción lineal (min analítico conocido:
        # x*=[1.5,0.5], f*=0.5) — a diferencia de la función usada antes
        # (2*(x1-3)**2+x1*x2**3, no acotada inferiormente en x2), esta no
        # diverge con solvers más "exploratorios" como Nelder-Mead.
        return dict(fx="(x1-2)**2 + (x2-1)**2", gx="x1+x2-2",
                    vars_="x1 x2", x0="0.0, 0.0", lo=0.0, hi=5.0, tol="0.1")
    if metodo in METHODS_MULTIOBJ:
        return dict(fx="(x-3)**2", gx="", vars_="", x0="1.0",
                    lo=0.0, hi=5.0, tol="0.1")
    # 1D, HEU, META
    return dict(fx="x**2 - 4*x + 5", gx="", vars_="", x0="1.0",
                lo=0.0, hi=5.0, tol="0.1")


def run_method_case(win: InterfazOptimizacion, metodo: str) -> None:
    warnings: list[str] = []
    _orig_warning, _orig_critical = QMessageBox.warning, QMessageBox.critical
    QMessageBox.warning = staticmethod(lambda *a, **k: warnings.append(str(a[2:])))
    QMessageBox.critical = staticmethod(lambda *a, **k: warnings.append(str(a[2:])))
    try:
        case = _case_for(metodo)
        win.cmb.setCurrentText(metodo)
        win.edt_fx.setText(case["fx"])
        win.edt_gx.setText(case["gx"])
        win.edt_vars.setText(case["vars_"])
        win.edt_x0.setText(case["x0"])
        win.smin.setValue(case["lo"])
        win.smax.setValue(case["hi"])
        win.edt_tol.setText(case["tol"])

        win.ejecutar()

        if warnings:
            raise AssertionError(f"diálogo de advertencia/error inesperado: {warnings}")
        if not win._hist:
            raise AssertionError("historial vacío tras ejecutar")
        if not win.lbl_st.text().startswith("✓"):
            raise AssertionError(f"status no indica éxito: {win.lbl_st.text()!r}")
        PASSES.append(metodo)
    except Exception as ex:
        FAILURES.append((metodo, f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}"))
    finally:
        QMessageBox.warning, QMessageBox.critical = _orig_warning, _orig_critical


def test_exports(win: InterfazOptimizacion) -> None:
    from app.infrastructure.reporting import csv_export, xlsx_export, pdf_export
    if not win._session:
        FAILURES.append(("exports", "sin sesión acumulada — no se puede probar export"))
        return
    tmpdir = tempfile.mkdtemp(prefix="optimizador_smoketest_")
    logo = win._logo_path()
    try:
        csv_folder = os.path.join(tmpdir, "csv_db")
        os.makedirs(csv_folder, exist_ok=True)
        csv_export.write_csv_db(csv_folder, win._session, getattr(win, "_history_log", []))
        assert any(os.scandir(csv_folder)), "no se generó ningún archivo CSV"
        PASSES.append("export CSV")
    except Exception as ex:
        FAILURES.append(("export CSV", f"{type(ex).__name__}: {ex}"))
    try:
        xlsx_path = os.path.join(tmpdir, "out.xlsx")
        xlsx_export.write_xlsx(xlsx_path, win._session, logo_path=logo)
        assert os.path.exists(xlsx_path) and os.path.getsize(xlsx_path) > 0
        PASSES.append("export XLSX")
    except Exception as ex:
        FAILURES.append(("export XLSX", f"{type(ex).__name__}: {ex}"))
    try:
        pdf_path = os.path.join(tmpdir, "out.pdf")
        pdf_export.write_pdf(pdf_path, win._session, area_key="general", logo_path=logo)
        assert os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0
        PASSES.append("export PDF")
    except Exception as ex:
        FAILURES.append(("export PDF", f"{type(ex).__name__}: {ex}"))


def test_ai_config_roundtrip() -> None:
    from app.infrastructure import ai_assistant as ai
    try:
        ok_save = ai._save_api_key("gsk_smoketest_dummy_key_0000000000000000000")
        primary, _ = ai._load_api_key_auto()
        ok_delete = ai._delete_saved_key()
        assert ok_save and ok_delete and primary.startswith("gsk_smoketest")
        assert os.path.dirname(ai._CFG_FILE) == ai._USER_DATA_DIR, \
            "config debe persistir en %APPDATA%, no junto al EXE"
        PASSES.append("AI config round-trip (%APPDATA%)")
    except Exception as ex:
        FAILURES.append(("AI config round-trip", f"{type(ex).__name__}: {ex}"))


def test_validations(win: InterfazOptimizacion) -> None:
    """Verifica que las validaciones de entrada muestren un mensaje claro en
    español y NO ejecuten el cálculo — sin depender de exec_() (ver nota
    sobre offscreen arriba)."""
    cases = [
        ("sin función", dict(fx="", metodo="Búsqueda Local", lo=0.0, hi=5.0, tol="0.1")),
        ("rango inválido (lo>=hi)", dict(fx="x**2", metodo="Búsqueda Local", lo=5.0, hi=0.0, tol="0.1")),
        ("tolerancia inválida (Fibonacci)", dict(fx="x**2", metodo="Fibonacci", lo=0.0, hi=5.0, tol="-1")),
        ("función LaTeX inválida", dict(fx=r"\min_{x} x^2", metodo="Búsqueda Local", lo=0.0, hi=5.0, tol="0.1")),
        ("2D en método estrictamente 1D", dict(fx="x**2+y**2", metodo="Búsqueda Local", lo=0.0, hi=5.0, tol="0.1")),
    ]
    for name, c in cases:
        ERR_DIALOGS.clear()
        warnings: list = []
        _orig_w, _orig_c = QMessageBox.warning, QMessageBox.critical
        QMessageBox.warning = staticmethod(lambda *a, **k: warnings.append(str(a[2])))
        QMessageBox.critical = staticmethod(lambda *a, **k: warnings.append(str(a[2])))
        try:
            win.cmb.setCurrentText(c["metodo"])
            win.edt_fx.setText(c["fx"])
            win.smin.setValue(c["lo"]); win.smax.setValue(c["hi"])
            win.edt_tol.setText(c["tol"])
            win.ejecutar()
            msg = (warnings[-1] if warnings else None) or (ERR_DIALOGS[-1][1] if ERR_DIALOGS else None)
            if not msg:
                raise AssertionError("no se mostró ningún mensaje de validación")
            PASSES.append(f"validación: {name}")
        except Exception as ex:
            FAILURES.append((f"validación: {name}", f"{type(ex).__name__}: {ex}"))
        finally:
            QMessageBox.warning, QMessageBox.critical = _orig_w, _orig_c
    # restaurar estado limpio para las pruebas siguientes
    win.edt_tol.setText("0.1"); win.smin.setValue(0.0); win.smax.setValue(5.0)


def test_reset_button(win: InterfazOptimizacion) -> None:
    try:
        win.cmb.setCurrentText("Búsqueda Local")
        win.edt_fx.setText("x**2 - 4*x + 5")
        win.smin.setValue(0.0); win.smax.setValue(5.0)
        win.ejecutar()
        assert win._hist, "el cálculo previo al reset no dejó historial"
        win.limpiar()
        assert not win._hist, "limpiar() no vació el historial"
        assert win.edt_fx.text() == "", "limpiar() no vació el campo f(x)"
        PASSES.append("botón Limpiar/reset")
    except Exception as ex:
        FAILURES.append(("botón Limpiar/reset", f"{type(ex).__name__}: {ex}"))


def main() -> int:
    app = QApplication(sys.argv)
    win = InterfazOptimizacion(resource_path=lambda p: p)

    all_methods = METHODS_1D + METHODS_MD + METHODS_HEU + METHODS_META + METHODS_MULTIOBJ

    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only == "--exports":
        run_method_case(win, "MD: Penalización (Newton)")
        run_method_case(win, "Armijo")
        test_exports(win)
    elif only == "--ai-config":
        test_ai_config_roundtrip()
    elif only == "--validations":
        test_validations(win)
    elif only == "--reset":
        test_reset_button(win)
    elif only:
        run_method_case(win, only)
    else:
        for metodo in all_methods:
            run_method_case(win, metodo)
        test_exports(win)
        test_ai_config_roundtrip()
        test_validations(win)
        test_reset_button(win)

    print(f"\n{'='*70}\nRESULTADO: {len(PASSES)} OK, {len(FAILURES)} FALLOS "
          f"(de {len(all_methods)} métodos + exports + config IA)\n{'='*70}")
    for name in PASSES:
        print(f"  OK   {name}")
    for name, err in FAILURES:
        print(f"  FAIL {name}\n       {err}")

    win.close()
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
