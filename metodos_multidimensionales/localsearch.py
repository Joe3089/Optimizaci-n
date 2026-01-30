import importlib.util
import pathlib

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_TARGET = _THIS_DIR / "localsearch (1).py"

spec = importlib.util.spec_from_file_location("localsearch_impl", _TARGET)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

fibonacci_search = mod.fibonacci_search
fixed_step_search = mod.fixed_step_search
local_neighborhood_search = mod.local_neighborhood_search

