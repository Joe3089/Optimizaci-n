# -*- coding: utf-8 -*-
"""
Prueba del circuito completo de intervención automática:
función → motor de optimización → detector de eventos → IA → explicación
→ TTS → confirmación de reproducción → recuadro de la interfaz.

    # Automatizada, sin API ni audio (repetible):
    venv\\Scripts\\python -m pytest tests/test_ai_automatic_intervention.py -v

    # Integración real (IA configurada en la app + voz SAPI5; se escucha audio):
    venv\\Scripts\\python tests/test_ai_automatic_intervention.py --real

La prueba real NUNCA se ejecuta desde pytest ni al iniciar la aplicación.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

if "--real" not in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.infrastructure.intervention_selftest import (  # noqa: E402
    run_ai_intervention_suite, scenario_results)


def test_ai_automatic_intervention(real=False):
    results = run_ai_intervention_suite(real=real, verbose=True)
    rows = scenario_results(results)
    failed = [f"{r['id']} {r['prueba']}: " +
              ", ".join(k for k, v in r["checks"].items() if not v)
              for r in rows if r["estado"] != "APROBADO"]
    assert rows, "no se ejecutó ningún escenario"
    assert not failed, "Pruebas FALLIDAS:\n" + "\n".join(failed)


if __name__ == "__main__":
    real = "--real" in sys.argv
    try:
        test_ai_automatic_intervention(real=real)
    except AssertionError as ex:
        print(f"\n{ex}")
        sys.exit(1)
    sys.exit(0)
