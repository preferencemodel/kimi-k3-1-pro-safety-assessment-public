"""`import config` from the repository root finds this folder first; it gives the loader in scripts/lib/config.py."""
import importlib.util as _u
from pathlib import Path as _P

_spec = _u.spec_from_file_location("_eval_config", _P(__file__).resolve().parents[1] / "scripts" / "lib" / "config.py")
_mod = _u.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
