# -*- coding: utf-8 -*-
"""
Proxy wrappers loader
Permite importar wrappers desde multi-dimensionals
"""

import os
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_PATH = os.path.join(BASE_DIR, "multi-dimensionals", "wrappers.py")

if not os.path.exists(REAL_PATH):
    raise FileNotFoundError(f"No se encontró wrappers real en: {REAL_PATH}")

spec = importlib.util.spec_from_file_location("md_wrappers", REAL_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# exportar todo
for name in dir(module):
    if not name.startswith("_"):
        globals()[name] = getattr(module, name)

__all__ = [n for n in globals() if not n.startswith("_")]
