# Auto-generated bridge module.
# Do not edit the original file: "localsearch (1).py"
import importlib.util
import pathlib
import sys

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_TARGET = _THIS_DIR / "localsearch (1).py"

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

spec = importlib.util.spec_from_file_location("localsearch_impl", _TARGET)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec from {_TARGET}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore[attr-defined]

# Re-export everything that doesn't start with underscore
__all__ = [name for name in dir(mod) if not name.startswith("_")]
globals().update({name: getattr(mod, name) for name in __all__})
