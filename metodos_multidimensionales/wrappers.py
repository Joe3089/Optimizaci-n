import importlib.util
import pathlib

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_TARGET = _THIS_DIR / "wrappers (1).py"

spec = importlib.util.spec_from_file_location("wrappers_impl", _TARGET)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

penalty_method_newton = mod.penalty_method_newton
barrier_method_newton = mod.barrier_method_newton

penalty_method_bfgs = mod.penalty_method_bfgs
weighted_sum_bfgs = mod.weighted_sum_bfgs

penalty_method_nelder = mod.penalty_method_nelder
weighted_sum_nelder = mod.weighted_sum_nelder
