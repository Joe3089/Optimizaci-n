# Auto-generated bridge module.
# Do not edit the original file: "line_search (1).py"
import importlib.util
import pathlib
import sys

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_TARGET = _THIS_DIR / "line_search (1).py"

# Ensure sibling modules can be imported if needed
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

spec = importlib.util.spec_from_file_location("line_search_impl", _TARGET)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec from {_TARGET}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore[attr-defined]

newton_armijo = mod.newton_armijo
