# OptimizadorFunciones.spec  —  VERSIÓN CORREGIDA
# ─────────────────────────────────────────────────────────────────────────────
# Genera el EXE con:  python -m PyInstaller OptimizadorFunciones.spec --noconfirm
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
from PyInstaller.utils.hooks import collect_data_files

# ── Certificados SSL de certifi (necesario para Groq / OpenRouter en el EXE) ──
try:
    _certifi_datas = collect_data_files('certifi')
except Exception:
    _certifi_datas = []

# ── Recursos del proyecto ─────────────────────────────────────────────────────
_project_datas_raw = [
    # Recursos gráficos
    ('Menu/fondo_optimizacion.png',   'Menu'),
    ('Menu/Fondo/Imgen de fondo.jpg', 'Menu/Fondo'),
    ('Menu/Fondo/Imgen de fondo.png', 'Menu/Fondo'),
    ('Menu/WindowsIcon-min.ico',      'Menu'),
    # Módulos Python propios — copiados como .py al root del bundle
    # para que __import__() dinámico los encuentre en _MEIPASS
    ('localsearch.py',              '.'),
    ('line_search.py',              '.'),
    ('wrappers.py',                 '.'),
    ('heuristics.py',               '.'),
    ('metaheuristics.py',           '.'),
    ('func_compat.py',              '.'),
    ('canvas_2d.py',                '.'),
    ('canvas_2d_estilo_fixed.py',   '.'),
    ('canvas_2d_dual.py',           '.'),
    ('rotacion_3d.py',              '.'),
    ('rotacion_3d_superficie.py',   '.'),
    ('ai_assistant.py',             '.'),
    ('interfaz_qt.py',              '.'),
    ('splash_screen.py',            '.'),
]
_project_datas = [(s, d) for s, d in _project_datas_raw if os.path.exists(s)]

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=_certifi_datas + _project_datas,
    hiddenimports=[
        # Módulos propios (importados dinámicamente con __import__ / lazy)
        'ai_assistant',
        'interfaz_qt',
        'line_search',
        'localsearch',
        'wrappers',
        'heuristics',
        'metaheuristics',
        'func_compat',
        'canvas_2d_estilo_fixed',
        'canvas_2d',
        'rotacion_3d_superficie',
        'rotacion_3d',
        # SSL / red
        'certifi',
        'ssl',
        'urllib.request',
        'urllib.error',
        'json',
        # Numérico
        'numpy',
        'numpy.core',
        'scipy',
        'scipy.optimize',
        'sympy',
        # Matplotlib
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_qt5agg',
        # PyQt6
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.sip',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='OptimizadorFunciones',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Menu\\WindowsIcon-min.ico',
)
