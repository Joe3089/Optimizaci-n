import importlib.util
import pathlib

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_TARGET = _THIS_DIR / "line_search (1).py"

spec = importlib.util.spec_from_file_location("line_search_impl", _TARGET)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

newton_armijo = mod.newton_armijo
wolfe_line_search = mod.wolfe_line_search
